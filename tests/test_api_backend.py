"""
API backends (`analysis/backends/api.py`): Gemini + OpenAI-compatible
chat over stdlib urllib (no new deps), with retries/backoff on transient
errors, per-call token usage logging, and the runner's opts contract
(seed / temperature). Transport is a module function tests monkeypatch —
no network here; the real-call smoke belongs to Step 4.
"""

import base64
import json
import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from analysis.backends import api  # noqa: E402


@pytest.fixture()
def frames(tmp_path):
    paths = []
    for i in range(2):
        p = tmp_path / f"f{i}.jpg"
        p.write_bytes(b"\xff\xd8\xff" + bytes([i]) * 8)  # fake jpeg bytes
        paths.append(str(p))
    return paths


@pytest.fixture(autouse=True)
def _reset_usage():
    api.USAGE.clear()
    yield


def _gemini_ok(text="hello", prompt_tokens=100, out_tokens=50,
               model_version=None):
    value = {
        "candidates": [{"content": {"parts": [{"text": text}]}}],
        "usageMetadata": {"promptTokenCount": prompt_tokens,
                          "candidatesTokenCount": out_tokens},
    }
    if model_version is not None:
        value["modelVersion"] = model_version
    return value


# ── .env loading ─────────────────────────────────────────────────────────

def test_load_env_parses_and_does_not_override(tmp_path, monkeypatch):
    env_file = tmp_path / ".env"
    env_file.write_text("# comment\nGEMINI_API_KEY=abc123\nOPENAI_API_KEY=def456\n")
    monkeypatch.setenv("OPENAI_API_KEY", "from-environ")
    vals = api.load_env(str(env_file))
    assert vals["GEMINI_API_KEY"] == "abc123"
    assert api.get_api_key("OPENAI_API_KEY", str(env_file)) == "from-environ"
    assert api.get_api_key("GEMINI_API_KEY", str(env_file)) == "abc123"


# ── Gemini payload + response ────────────────────────────────────────────

def test_gemini_payload_and_text_extraction(frames, monkeypatch):
    captured = {}

    def fake_post(url, payload, headers, timeout):
        captured["url"], captured["payload"] = url, payload
        return _gemini_ok("the answer")

    monkeypatch.setattr(api, "_post_json", fake_post)
    text = api.gemini_generate("describe", frames, "gemini-2.5-pro",
                               seed=7, temperature=0.2, api_key="k")
    assert text == "the answer"
    assert "gemini-2.5-pro" in captured["url"]
    parts = captured["payload"]["contents"][0]["parts"]
    assert parts[0]["text"] == "describe"
    imgs = [p for p in parts if "inline_data" in p]
    assert len(imgs) == 2
    assert imgs[0]["inline_data"]["mime_type"] == "image/jpeg"
    base64.b64decode(imgs[0]["inline_data"]["data"])  # valid b64
    gen = captured["payload"]["generationConfig"]
    assert gen["temperature"] == 0.2 and gen["seed"] == 7


def test_gemini_blind_omits_images_and_greedy_omits_temperature(monkeypatch):
    captured = {}

    def fake_post(url, payload, headers, timeout):
        captured["payload"] = payload
        return _gemini_ok()

    monkeypatch.setattr(api, "_post_json", fake_post)
    api.gemini_generate("p", [], "gemini-2.5-pro", seed=None,
                        temperature=None, api_key="k")
    parts = captured["payload"]["contents"][0]["parts"]
    assert len(parts) == 1 and "text" in parts[0]
    assert "temperature" not in captured["payload"].get("generationConfig", {})


def test_usage_is_recorded_per_call(monkeypatch):
    monkeypatch.setattr(api, "_post_json",
                        lambda *a, **k: _gemini_ok(prompt_tokens=111, out_tokens=22))
    api.gemini_generate("p", [], "gemini-2.5-pro", api_key="k")
    assert api.USAGE[-1]["input_tokens"] == 111
    assert api.USAGE[-1]["output_tokens"] == 22
    assert api.USAGE[-1]["model"] == "gemini-2.5-pro"


def test_gemini_panel_result_enforces_cap_version_and_separates_thoughts(
        frames, monkeypatch):
    captured = {}

    def fake_post(url, payload, headers, timeout):
        captured["url"], captured["payload"] = url, payload
        return {
            "candidates": [{
                "content": {"parts": [
                    {"text": "private thought", "thought": True},
                    {"text": '{"predicted_branches": []}'},
                ]},
                "finishReason": "STOP",
            }],
            "modelVersion": "gemini-2.5-pro-20260801",
            "responseId": "response-1",
            "usageMetadata": {
                "promptTokenCount": 111,
                "candidatesTokenCount": 22,
                "thoughtsTokenCount": 7,
            },
        }

    monkeypatch.setattr(api, "_post_json", fake_post)
    result = api.gemini_generate_result(
        "prompt", frames, "gemini-2.5-pro", temperature=0.0,
        max_output_tokens=32768, api_endpoint=api.GEMINI_URL,
        api_version="v1beta", require_model_version=True, api_key="k")
    assert captured["payload"]["generationConfig"]["maxOutputTokens"] == 32768
    assert result.thinking_trace == "private thought"
    assert result.final_text == '{"predicted_branches": []}'
    assert result.raw_text == 'private thought{"predicted_branches": []}'
    assert result.metadata["model_version"] == "gemini-2.5-pro-20260801"
    assert result.metadata["api_version"] == "v1beta"
    assert result.metadata["thinking_tokens"] == 7
    assert api.USAGE[-1]["model_version"] == "gemini-2.5-pro-20260801"


def test_gemini_panel_result_fails_if_provider_omits_model_version(monkeypatch):
    monkeypatch.setattr(api, "_post_json", lambda *args, **kwargs: _gemini_ok())
    with pytest.raises(api.ApiError, match="omitted required modelVersion"):
        api.gemini_generate_result(
            "prompt", [], "gemini-2.5-pro", max_output_tokens=32768,
            require_model_version=True, api_key="k")


# ── Retries ──────────────────────────────────────────────────────────────

def test_transient_errors_retry_then_succeed(monkeypatch):
    calls = {"n": 0}

    def flaky(url, payload, headers, timeout):
        calls["n"] += 1
        if calls["n"] < 3:
            raise api.TransientApiError("429 rate limited")
        return _gemini_ok("ok")

    monkeypatch.setattr(api, "_post_json", flaky)
    monkeypatch.setattr(api, "_sleep", lambda s: None)
    assert api.gemini_generate("p", [], "m", api_key="k") == "ok"
    assert calls["n"] == 3


def test_non_transient_error_raises_immediately(monkeypatch):
    calls = {"n": 0}

    def bad_request(url, payload, headers, timeout):
        calls["n"] += 1
        raise api.ApiError("400 invalid request")

    monkeypatch.setattr(api, "_post_json", bad_request)
    monkeypatch.setattr(api, "_sleep", lambda s: None)
    with pytest.raises(api.ApiError):
        api.gemini_generate("p", [], "m", api_key="k")
    assert calls["n"] == 1


# ── OpenAI-compatible backend ────────────────────────────────────────────

def test_openai_payload_and_extraction(frames, monkeypatch):
    captured = {}

    def fake_post(url, payload, headers, timeout):
        captured["payload"], captured["headers"] = payload, headers
        return {"choices": [{"message": {"content": "gpt says"}}],
                "usage": {"prompt_tokens": 10, "completion_tokens": 5}}

    monkeypatch.setattr(api, "_post_json", fake_post)
    text = api.openai_generate("look", frames, "gpt-5.2", seed=3,
                               temperature=0.1, api_key="sk-x")
    assert text == "gpt says"
    assert captured["headers"]["Authorization"] == "Bearer sk-x"
    msg = captured["payload"]["messages"][0]
    assert msg["content"][0]["text"] == "look"
    assert sum(1 for c in msg["content"] if c["type"] == "image_url") == 2
    assert captured["payload"]["seed"] == 3
    assert api.USAGE[-1]["input_tokens"] == 10


# ── Registration in the runner ───────────────────────────────────────────

def test_runner_registers_gemini_and_gpt_backends():
    from analysis.run_model import BACKENDS
    assert "gemini" in BACKENDS and "gpt" in BACKENDS


@pytest.mark.skip(reason="needs pilot fixture data that is not part of this release")
def test_runner_gemini_end_to_end_writes_predictions(tmp_path, monkeypatch):
    """run_model --backend gemini over one real pilot GT, transport mocked:
    prediction files appear and parse cleanly."""
    monkeypatch.setattr(api, "_post_json", lambda *a, **k: _gemini_ok(json.dumps({
        "predicted_branches": [{"branch_id": "B1", "is_factual": True,
                                "chain": [{"timestamp": 0.5, "event_type": "safe_stop",
                                           "terminal": True, "action_type": "ego_no_action",
                                           "actors_update": [{"id": "ego",
                                                              "entity_type": "ego_vehicle",
                                                              "role_type": "ego_agent",
                                                              "x": 0.0, "y": 0.0}]}]}],
        "predicted_counterfactuals": [],
    })))
    monkeypatch.setenv("GEMINI_API_KEY", "test-key")
    from analysis.run_model import run_model
    stats = run_model("mini_dataset", str(tmp_path), backend="gemini",
                      model_id="gemini-2.5-pro", seed=0)
    assert stats["scenarios"] > 0 and not stats["failed"]
    assert (tmp_path / "scenario_1" / "l3_pre.json").exists()
    # raw model text is archived per scenario (closed models drift,
    # compliance analysis needs the unparsed output)
    assert (tmp_path / "scenario_1" / "raw_response.txt").exists()
    assert (tmp_path / "scenario_1" / "final_response.txt").exists()
    assert (tmp_path / "scenario_1" / "generation_metadata.json").exists()


# ── the prompt must state the full output contract ───────────────────────
# A prompt that says "ontology-grounded" without giving the ontology yields
# clean JSON that fails the schema gates (dict-valued action_type,
# off-vocabulary labels, missing branch_id / entity_type / role_type), while
# the dissertation's recall argument REQUIRES the ontology to be model input.

def test_prompt_contains_full_output_contract():
    import json as _json
    import yaml as _yaml
    from simulation.build_pilot import build_prompt
    ontology = _yaml.safe_load(open("ctms/ontology.yaml"))
    schema = _json.load(open("ctms/schema_branch.json"))
    root = {"event_type": "observed_state", "depth": 0, "timestamp": 0.0,
            "actors": [{"id": "ego", "entity_type": "ego_vehicle",
                        "role_type": "ego_agent", "x": 0.0, "y": 0.0,
                        "vx": 5.0, "vy": 0.0, "speed": 5.0}]}
    prompt = build_prompt(root, "test_scenario")
    # every enum label the schema will enforce must be stated verbatim
    for ev in schema["$defs"]["node"]["properties"]["event_type"]["enum"]:
        if ev != "observed_state":
            assert ev in prompt, f"event_type {ev} missing from prompt"
    # action labels come from the SCHEMA enum (the enforced contract), not
    # ontology.yaml's action FAMILIES (brake/steer/passive/...)
    for ac in schema["$defs"]["node"]["properties"]["action_type"]["enum"]:
        assert ac in prompt, f"action_type {ac} missing from prompt"
    assert "passive" not in prompt.split("Allowed `action_type`")[1].split("Terminal")[0]
    # required structural fields must be named
    for field in ("branch_id", "is_factual", "entity_type", "role_type",
                  "intervention_type", "target_actor_id", "trigger_depth"):
        assert field in prompt, f"required field {field} missing from prompt"
    # action_type must be demanded as a plain string (the smoke's top failure)
    assert "string" in prompt
    assert "## Coordinate convention" in prompt
    assert "+x is forward, +y is right, and -y is left" in prompt
    assert "axes do not rotate" in " ".join(prompt.lower().split())


# ── run-matrix manifest expansion ────────────────────────────────────────

@pytest.mark.skip(reason="needs the legacy run matrix, which is not part of this release")
def test_run_matrix_expands_models_conditions_seeds(tmp_path):
    import yaml as _yaml
    from analysis.run_matrix import expand_matrix
    manifest = {
        "gt_dir": "output_nuscenes",
        "out_root": "results",
        "seeds": [0, 1, 2],
        "conditions": ["video", "history_json", "video_history"],
        "models": [
            {"name": "gemini-2.5-pro", "backend": "gemini", "model_id": "gemini-2.5-pro"},
            {"name": "qwen3-vl-8b", "backend": "local",
             "model_id": "Qwen/Qwen3-VL-8B-Instruct"},
        ],
    }
    cells = expand_matrix(manifest)
    assert len(cells) == 2 * 3 * 3
    c = cells[0]
    assert set(c) >= {"backend", "model_id", "gt_dir", "out_dir", "seed", "condition"}
    # out_dir encodes model/condition/seed so runs never collide and resume works
    dirs = {c["out_dir"] for c in cells}
    assert len(dirs) == len(cells)
    assert any("history_json" in d for d in dirs) and any("seed1" in d for d in dirs)


@pytest.mark.skip(reason="needs the legacy run matrix, which is not part of this release")
def test_run_matrix_execute_calls_runner_per_cell(tmp_path):
    from analysis.run_matrix import expand_matrix, execute_matrix
    manifest = {"gt_dir": "g", "out_root": str(tmp_path), "seeds": [0],
                "conditions": ["video"],
                "models": [{"name": "m1", "backend": "mock", "model_id": ""}]}
    calls = []

    def fake_runner(**kwargs):
        calls.append(kwargs)
        return {"scenarios": 1, "failed": [], "skipped": 0}

    results = execute_matrix(expand_matrix(manifest), runner=fake_runner)
    assert len(calls) == 1
    assert calls[0]["skip_existing"] is True
    assert results[0]["scenarios"] == 1


@pytest.mark.skip(reason="needs the legacy run matrix, which is not part of this release")
@pytest.mark.parametrize("path", [
    "analysis/run_matrix.yaml",
    "analysis/run_matrix_pilot30.yaml",
])
def test_protocol_manifests_use_one_greedy_run(path):
    from analysis.run_matrix import expand_matrix, load_run_manifest

    manifest = load_run_manifest(path)
    assert manifest["temperature"] == 0.0
    assert manifest["seeds"] == [0]
    assert manifest["conditions"] == [
        "video", "history_json", "video_history"]
    cells = expand_matrix(manifest)
    assert len(manifest["models"]) == 5
    assert len(cells) == 15
    assert all(cell["temperature"] == 0.0 and cell["seed"] == 0
               for cell in cells)
