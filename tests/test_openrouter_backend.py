"""The OpenAI-compatible backend records only answers the provider finished."""
import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


def _call(monkeypatch, finish_reason, content='{"predicted_branches": []}'):
    from scripts import run_openrouter_cell as cell
    monkeypatch.setattr(cell.api_backend, "get_api_key", lambda *a, **k: "k")
    monkeypatch.setattr(cell, "_stream_chat", lambda *a, **k: {
        "provider": "Host", "usage": {"completion_tokens": 7},
        "choices": [{"message": {"content": content}, "finish_reason": finish_reason}]})
    return cell._openai_compat_backend("scene", [], "m", {})


@pytest.mark.parametrize("finish_reason", [None, "error"])
def test_a_stream_the_provider_did_not_finish_is_never_recorded(monkeypatch, finish_reason):
    from analysis.backends.api import TransientApiError
    with pytest.raises(TransientApiError, match="Host"):
        _call(monkeypatch, finish_reason, content='{"predicted_branches": [{"branch_id"')


@pytest.mark.parametrize("finish_reason", ["stop", "length"])
def test_a_finished_answer_is_recorded_even_at_the_token_cap(monkeypatch, finish_reason):
    result = _call(monkeypatch, finish_reason)
    assert result.metadata["finish_reason"] == finish_reason


def test_a_self_hosted_server_gets_the_thinking_switch_and_no_provider_routing(monkeypatch):
    """vLLM: base URL from the environment, no key unless one is configured,
    thinking switched by the chat template, usage requested in the stream."""
    from scripts import run_openrouter_cell as cell
    seen = {}

    def fake_stream(url, payload, headers, timeout, **limits):
        seen.update(url=url, payload=payload, headers=headers)
        return {"model": "Qwen/Qwen3.5-9B", "usage": {"completion_tokens": 9},
                "choices": [{"message": {"content": "{}", "reasoning_content": "thought"},
                             "finish_reason": "stop"}]}

    monkeypatch.setattr(cell, "_stream_chat", fake_stream)
    monkeypatch.setenv("CORIVA_VLLM_BASE_URL", "http://localhost:8000/v1/")
    result = cell._openai_compat_backend("scene", [], "Qwen/Qwen3.5-9B", {
        "api_base_url_env": "CORIVA_VLLM_BASE_URL", "api_key_env": None,
        "chat_template_kwargs": {"enable_thinking": True}})
    assert seen["url"] == "http://localhost:8000/v1/chat/completions"
    assert "Authorization" not in seen["headers"]
    assert seen["payload"]["chat_template_kwargs"] == {"enable_thinking": True}
    assert seen["payload"]["stream_options"] == {"include_usage": True}
    assert "provider" not in seen["payload"] and "reasoning" not in seen["payload"]
    assert result.thinking_trace == "thought"
    assert result.metadata["chat_template_kwargs"] == {"enable_thinking": True}


def test_a_self_hosted_server_without_its_address_is_refused(monkeypatch):
    from analysis.backends.api import ApiError
    from scripts import run_openrouter_cell as cell
    monkeypatch.delenv("CORIVA_VLLM_BASE_URL", raising=False)
    with pytest.raises(ApiError, match="CORIVA_VLLM_BASE_URL"):
        cell._openai_compat_backend("scene", [], "m", {"api_base_url_env": "CORIVA_VLLM_BASE_URL",
                                                       "api_key_env": None})


def test_the_stream_reader_keeps_vllm_reasoning_content(monkeypatch):
    import io, json as _json
    from scripts import run_openrouter_cell as cell
    events = [{"choices": [{"delta": {"reasoning_content": "think "}}]},
              {"choices": [{"delta": {"reasoning_content": "more"}}]},
              {"choices": [{"delta": {"content": "{}"}, "finish_reason": "stop"}]},
              {"usage": {"completion_tokens": 3}, "choices": []}]
    body = "".join(f"data: {_json.dumps(e)}\n\n" for e in events) + "data: [DONE]\n\n"
    monkeypatch.setattr(cell.urllib.request if hasattr(cell, "urllib") else __import__("urllib.request").request,
                        "urlopen", lambda req, timeout: io.BytesIO(body.encode()))
    out = cell._stream_chat("http://x/v1/chat/completions", {"model": "m"}, {}, 5)
    assert out["choices"][0]["message"]["reasoning"] == "think more"
    assert out["choices"][0]["message"]["content"] == "{}"
    assert out["usage"] == {"completion_tokens": 3}


def test_a_declared_sampling_setting_reaches_the_server_and_greedy_sends_none(monkeypatch):
    from scripts import run_openrouter_cell as cell
    sent = []

    def fake_stream(url, payload, headers, timeout, **limits):
        sent.append(payload)
        return {"usage": {"completion_tokens": 3},
                "choices": [{"message": {"content": "{}"}, "finish_reason": "stop"}]}

    monkeypatch.setattr(cell, "_stream_chat", fake_stream)
    monkeypatch.setenv("CORIVA_VLLM_BASE_URL", "http://localhost:8000/v1")
    server = {"api_base_url_env": "CORIVA_VLLM_BASE_URL", "api_key_env": None}
    sampling = {"temperature": 0.6, "top_p": 0.95, "top_k": 20, "min_p": 0.0,
                "presence_penalty": 0.0, "repetition_penalty": 1.0}
    result = cell._openai_compat_backend("scene", [], "m", dict(
        server, sampling=sampling, temperature=0.6, seed=0))
    cell._openai_compat_backend("scene", [], "m", dict(server, temperature=0.0, seed=None))
    sampled, greedy = sent
    assert sampled["temperature"] == 0.6 and sampled["seed"] == 0
    assert {k: sampled[k] for k in sampling if k != "temperature"} == {
        k: v for k, v in sampling.items() if k != "temperature"}
    assert greedy["temperature"] == 0.0
    assert not {"seed", "top_p", "top_k", "min_p", "presence_penalty",
                "repetition_penalty"} & set(greedy)
    assert result.metadata["sampling"] == sampling


def test_a_request_the_provider_holds_without_generating_is_abandoned(monkeypatch):
    """OpenRouter keeps a queued request alive with comments, which resets the
    per-read timeout; with no token after the first-token limit the call is a
    transient failure (retried on resume, never recorded). A stream that is
    generating is never cut by that limit."""
    import io
    import pytest
    from analysis.backends.api import TransientApiError
    from scripts import run_openrouter_cell as cell
    clock = {"t": 0.0}
    monkeypatch.setattr(cell.time, "monotonic", lambda: clock["t"])

    class Stream(io.BytesIO):
        def __init__(self, lines, step):
            super().__init__(b"".join(lines)); self.step = step
        def __iter__(self):
            for line in self.getvalue().splitlines(keepends=True):
                clock["t"] += self.step
                yield line

    def opener(lines, step):
        return lambda req, timeout: Stream(lines, step)
    queued = [b": OPENROUTER PROCESSING\n"] * 50
    monkeypatch.setattr("urllib.request.urlopen", opener(queued, 30.0))
    with pytest.raises(TransientApiError, match="no token after"):
        cell._stream_chat("https://openrouter.ai/api/v1/chat/completions", {}, {}, 300,
                          first_token_s=600, max_s=3600)
    token = b'data: {"choices": [{"delta": {"content": "{}"}, "finish_reason": null}]}\n'
    done = b'data: {"choices": [{"delta": {}, "finish_reason": "stop"}]}\n'
    monkeypatch.setattr("urllib.request.urlopen", opener([token] * 40 + [done, b"data: [DONE]\n"], 30.0))
    out = cell._stream_chat("https://openrouter.ai/api/v1/chat/completions", {}, {}, 300,
                            first_token_s=600, max_s=3600)
    assert out["choices"][0]["finish_reason"] == "stop"            # 20 min of tokens: kept
    clock["t"] = 0.0
    monkeypatch.setattr("urllib.request.urlopen", opener([token] * 200, 30.0))
    with pytest.raises(TransientApiError, match="still open"):
        cell._stream_chat("https://openrouter.ai/api/v1/chat/completions", {}, {}, 300,
                          first_token_s=600, max_s=3600)
