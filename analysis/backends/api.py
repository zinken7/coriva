"""API backends for the CORIVA runner (TASK.md P3.2): Gemini + any
OpenAI-compatible chat endpoint, over stdlib urllib — no new dependencies.

Design notes
------------
- Keys come from the environment first, then a `.env` file (KEY=VALUE lines;
  see `.env.example`). Keys are never logged.
- `_post_json` is the single transport function; unit tests monkeypatch it,
  the P3.4 smoke run exercises the real network.
- Transient failures (429 / 5xx / network) retry with exponential backoff up
  to MAX_RETRIES; client errors (4xx) raise immediately — a malformed request
  will not fix itself.
- Every successful call appends {model, input_tokens, output_tokens} to
  `USAGE` (and to the JSONL file named by $CORIVA_API_USAGE_LOG, if set) —
  this is what the P3.6 cost estimate reads.
"""

from __future__ import annotations

import base64
import datetime as dt
import json
import os
import time
import urllib.error
import urllib.request
from typing import Any, Dict, List, Optional

from analysis.backends.result import GenerationResult

DEFAULT_GEMINI_MODEL = "gemini-2.5-pro"
DEFAULT_OPENAI_MODEL = "gpt-5.2"
GEMINI_URL = ("https://generativelanguage.googleapis.com/v1beta/models/"
              "{model}:generateContent")
OPENAI_URL = "https://api.openai.com/v1/chat/completions"
MAX_RETRIES = 5
BASE_DELAY_S = 2.0
TIMEOUT_S = 300

USAGE: List[Dict[str, Any]] = []


class ApiError(RuntimeError):
    """Non-retryable API failure (4xx other than 429, malformed response)."""


class TransientApiError(ApiError):
    """Retryable failure: rate limit, server error, network hiccup."""


def load_env(path: str = ".env") -> Dict[str, str]:
    """Parse a KEY=VALUE .env file (comments/blank lines ignored)."""
    vals: Dict[str, str] = {}
    if not os.path.exists(path):
        return vals
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, value = line.partition("=")
            vals[key.strip()] = value.strip().strip('"').strip("'")
    return vals


def get_api_key(name: str, env_path: str = ".env") -> Optional[str]:
    """Environment wins over .env; returns None if absent in both."""
    return os.environ.get(name) or load_env(env_path).get(name)


def _sleep(seconds: float) -> None:  # separated so tests can neutralize it
    time.sleep(seconds)


def _post_json(url: str, payload: Dict[str, Any], headers: Dict[str, str],
               timeout: float = TIMEOUT_S) -> Dict[str, Any]:
    """Single transport function. Raises TransientApiError on 429/5xx/network,
    ApiError on other HTTP errors."""
    body = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(url, data=body, method="POST",
                                 headers={"Content-Type": "application/json", **headers})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        detail = ""
        try:
            detail = e.read().decode("utf-8", "replace")[:500]
        except Exception:
            pass
        if e.code == 429 or e.code >= 500:
            raise TransientApiError(f"HTTP {e.code}: {detail}") from e
        raise ApiError(f"HTTP {e.code}: {detail}") from e
    except (urllib.error.URLError, TimeoutError, ConnectionError) as e:
        raise TransientApiError(f"network error: {e}") from e


def _with_retries(fn):
    last: Optional[Exception] = None
    for attempt in range(MAX_RETRIES):
        try:
            return fn()
        except TransientApiError as e:
            last = e
            _sleep(BASE_DELAY_S * (2 ** attempt))
    raise TransientApiError(f"gave up after {MAX_RETRIES} retries: {last}")


def _record_usage(model: str, input_tokens: int, output_tokens: int,
                  **extra: Any) -> None:
    entry = {"model": model, "input_tokens": int(input_tokens),
             "output_tokens": int(output_tokens)}
    entry.update({key: value for key, value in extra.items()
                  if value is not None})
    USAGE.append(entry)
    log_path = os.environ.get("CORIVA_API_USAGE_LOG")
    if log_path:
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry) + "\n")


def _mime(path: str) -> str:
    return "image/png" if path.lower().endswith(".png") else "image/jpeg"


def _b64(path: str) -> str:
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode("ascii")


# ── Gemini ───────────────────────────────────────────────────────────────

def gemini_generate_result(
        prompt: str, frame_paths: List[str], model_id: str,
        seed: Optional[int] = None, temperature: Optional[float] = None,
        max_output_tokens: Optional[int] = None,
        api_endpoint: str = GEMINI_URL,
        api_version: Optional[str] = None,
        require_model_version: bool = False,
        api_key: Optional[str] = None,
        system_prompt: Optional[str] = None) -> GenerationResult:
    """Generate through the pinned Gemini endpoint and retain provider metadata.
    `system_prompt`, when given, is sent as the request's `systemInstruction`."""
    model = model_id or DEFAULT_GEMINI_MODEL
    key = api_key or get_api_key("GEMINI_API_KEY")
    if not key:
        raise ApiError("GEMINI_API_KEY not set (environment or .env)")

    parts: List[Dict[str, Any]] = [{"text": prompt}]
    for p in frame_paths:
        parts.append({"inline_data": {"mime_type": _mime(p), "data": _b64(p)}})
    gen_cfg: Dict[str, Any] = {}
    if temperature is not None:
        gen_cfg["temperature"] = float(temperature)
    if seed is not None:
        gen_cfg["seed"] = int(seed)
    if max_output_tokens is not None:
        if not isinstance(max_output_tokens, int) or max_output_tokens <= 0:
            raise ValueError("max_output_tokens must be a positive integer")
        gen_cfg["maxOutputTokens"] = max_output_tokens
    payload: Dict[str, Any] = {"contents": [{"parts": parts}]}
    if system_prompt is not None:
        payload["systemInstruction"] = {"parts": [{"text": system_prompt}]}
    if gen_cfg:
        payload["generationConfig"] = gen_cfg

    url = api_endpoint.format(model=model)

    def call():
        return _post_json(url, payload, {"x-goog-api-key": key}, TIMEOUT_S)

    resp = _with_retries(call)
    try:
        cand = resp["candidates"][0]
        content = cand.get("content") or {}
        # A thinking model can spend the whole output cap on thoughts it does not
        # return; the candidate then has no parts. That is the model's (empty)
        # answer at our cap, recorded like any capped answer, not a malformed
        # response. Any other candidate without parts still raises.
        if "parts" not in content and cand.get("finishReason") == "MAX_TOKENS":
            response_parts = []
        else:
            response_parts = content["parts"]
        raw_text = "".join(part.get("text", "") for part in response_parts)
        thinking_trace = "".join(
            part.get("text", "") for part in response_parts
            if part.get("thought") is True)
        final_text = "".join(
            part.get("text", "") for part in response_parts
            if part.get("thought") is not True)
    except (KeyError, IndexError, TypeError) as e:
        raise ApiError(f"unexpected Gemini response shape: {str(resp)[:300]}") from e
    model_version = resp.get("modelVersion")
    if require_model_version and not model_version:
        raise ApiError("Gemini response omitted required modelVersion")
    access_date = dt.datetime.now(dt.timezone.utc).date().isoformat()
    meta = resp.get("usageMetadata", {})
    _record_usage(model, meta.get("promptTokenCount", 0),
                  meta.get("candidatesTokenCount", 0),
                  thinking_tokens=meta.get("thoughtsTokenCount"),
                  model_version=model_version,
                  access_date=access_date,
                  api_version=api_version,
                  max_output_tokens=max_output_tokens,
                  response_id=resp.get("responseId"))
    return GenerationResult(
        final_text=final_text,
        raw_text=raw_text,
        thinking_trace=thinking_trace or None,
        metadata={
            "adapter": "gemini_api",
            "api_endpoint": api_endpoint,
            "api_version": api_version,
            "model_version": model_version,
            "access_date": access_date,
            "response_id": resp.get("responseId"),
            "finish_reason": cand.get("finishReason"),
            "input_tokens": int(meta.get("promptTokenCount", 0)),
            "output_tokens": int(meta.get("candidatesTokenCount", 0)),
            "thinking_tokens": int(meta.get("thoughtsTokenCount", 0)),
            "max_output_tokens": max_output_tokens,
            "ordered_image_count": len(frame_paths),
        })


def gemini_generate(prompt: str, frame_paths: List[str], model_id: str,
                    seed: Optional[int] = None, temperature: Optional[float] = None,
                    max_output_tokens: Optional[int] = None,
                    api_endpoint: str = GEMINI_URL,
                    api_version: Optional[str] = None,
                    require_model_version: bool = False,
                    api_key: Optional[str] = None) -> str:
    """Compatibility wrapper returning scorer-visible text only."""
    return gemini_generate_result(
        prompt, frame_paths, model_id, seed=seed, temperature=temperature,
        max_output_tokens=max_output_tokens, api_endpoint=api_endpoint,
        api_version=api_version,
        require_model_version=require_model_version,
        api_key=api_key).final_text


# ── OpenAI-compatible chat ───────────────────────────────────────────────

def openai_generate(prompt: str, frame_paths: List[str], model_id: str,
                    seed: Optional[int] = None, temperature: Optional[float] = None,
                    api_key: Optional[str] = None, base_url: Optional[str] = None) -> str:
    model = model_id or DEFAULT_OPENAI_MODEL
    key = api_key or get_api_key("OPENAI_API_KEY")
    if not key:
        raise ApiError("OPENAI_API_KEY not set (environment or .env)")

    content: List[Dict[str, Any]] = [{"type": "text", "text": prompt}]
    for p in frame_paths:
        content.append({"type": "image_url",
                        "image_url": {"url": f"data:{_mime(p)};base64,{_b64(p)}"}})
    payload: Dict[str, Any] = {"model": model,
                               "messages": [{"role": "user", "content": content}]}
    if temperature is not None:
        payload["temperature"] = float(temperature)
    if seed is not None:
        payload["seed"] = int(seed)

    url = (base_url or os.environ.get("OPENAI_BASE_URL") or OPENAI_URL)

    def call():
        return _post_json(url, payload, {"Authorization": f"Bearer {key}"}, TIMEOUT_S)

    resp = _with_retries(call)
    try:
        text = resp["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError) as e:
        raise ApiError(f"unexpected OpenAI response shape: {str(resp)[:300]}") from e
    usage = resp.get("usage", {})
    _record_usage(model, usage.get("prompt_tokens", 0),
                  usage.get("completion_tokens", 0))
    return text if isinstance(text, str) else json.dumps(text)
