"""Run one CORIVA cell against an OpenAI-compatible endpoint (OpenRouter by default).

Why this exists
---------------
Prompt iteration on rented GPUs is slow and expensive: one qwen3.5-9b cell of
40 scenarios costs about 8.5 GPU-hours, against roughly 14 cents here. This
script is for finding prompt DEFECTS fast; it is not campaign evidence.

What it is not
--------------
OpenRouter serves a model slug, not a pinned checkpoint. Providers behind one
slug differ in quantisation and serving stack, so a score here is not
comparable with a locally pinned bf16 run and must never be pooled with one.
`--provider` pins a single provider and records it, which makes a run
reproducible against ITSELF, not against the panel's local revision.

Every run_manifest therefore carries `execution_purpose: smoke_admission`
(set by run_model's provisional path) and `revision: null`. Confirm any prompt
change on the real local stack before it reaches the campaign.

Endpoints
---------
Defaults to OpenRouter. Any OpenAI-compatible chat-completions API works via
--base-url and --api-key-env, e.g. NVIDIA's hosted catalogue:
  --base-url https://integrate.api.nvidia.com/v1 --api-key-env NVIDIA_API_KEY
The OpenRouter-only `provider` routing block is sent only to OpenRouter.
A configuration may instead name an environment variable holding the address
(`api_base_url_env`, e.g. CORIVA_VLLM_BASE_URL for our own vLLM server) and
`api_key_env: null` when the server takes no key; `chat_template_kwargs`
switches Qwen3.5's thinking on vLLM, which ignores OpenRouter's `reasoning`.

Usage
-----
  export OPENROUTER_API_KEY=...
  python scripts/run_openrouter_cell.py \
      --model-id qwen/qwen3-vl-8b-instruct \
      --dataset-root data/output_nuscenes_protocol10 \
      --split nuscenes --condition video --scenarios 40 \
      --out-dir /tmp/or_check/qwenvl_nuscenes_video
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
import time
from typing import Any, Dict, List

# Work from wherever this file sits: walk up to the directory holding analysis/.
_here = Path(__file__).resolve()
_root = next((p for p in _here.parents if (p / "analysis").is_dir()), _here.parent)
sys.path.insert(0, str(_root))

from analysis.backends import api as api_backend
from analysis.backends.result import GenerationResult, split_thinking_trace
from analysis.evaluation_campaign import (
    CONDITIONS, dataset_inventory, sha, score_cell, write_campaign_identity, write_json)
from analysis.evaluation_report import collect_cell, export_tables, mean_stat
from analysis.model_panel import MAX_OUTPUT_TOKENS
from analysis.run_model import backend, run_model

OPENROUTER_BASE = "https://openrouter.ai/api/v1"
# A request that OpenRouter keeps alive with comments but the provider never
# starts (seen on Parasail, 2026-09-29: requests open for hours, no token)
# is infrastructure, not an answer: it is abandoned and retried on resume.
# Mistral answers took at most 19.3 min (median 11), so neither limit binds a
# request that is generating.
OPENROUTER_FIRST_TOKEN_S = 600
OPENROUTER_MAX_S = 3600
ENV_PATH = str(_root / ".env")


def _is_openrouter(base_url: str) -> bool:
    return "openrouter.ai" in base_url


def _stream_chat(url: str, payload: Dict[str, Any], headers: Dict[str, str],
                 timeout: float, first_token_s: float = None, max_s: float = None) -> Dict[str, Any]:
    """POST with `stream: true` and fold the SSE events into one response dict.

    A long non-streaming request leaves the connection silent for minutes and
    intermediaries cut it ("Network connection lost" after 5-12k tokens on two
    providers). Streaming keeps bytes flowing. The returned dict has the same
    shape as a non-streaming completion, so callers need not change. `timeout`
    bounds each read, not the whole generation; keep-alive comments reset it,
    so `first_token_s` (no content or reasoning yet) and `max_s` (whole stream)
    bound a request the provider holds without generating.
    """
    import urllib.error
    import urllib.request
    body = json.dumps({**payload, "stream": True}).encode("utf-8")
    req = urllib.request.Request(url, data=body, method="POST",
                                 headers={"Content-Type": "application/json", **headers})
    try:
        resp = urllib.request.urlopen(req, timeout=timeout)
    except urllib.error.HTTPError as e:
        detail = e.read().decode("utf-8", "replace")[:300]
        if e.code == 429 or e.code >= 500:
            raise api_backend.TransientApiError(f"HTTP {e.code}: {detail}") from e
        raise api_backend.ApiError(f"HTTP {e.code}: {detail}") from e
    except (urllib.error.URLError, TimeoutError, OSError) as e:
        raise api_backend.TransientApiError(f"network: {e}") from e

    content, reasoning = [], []
    out: Dict[str, Any] = {"choices": [{"message": {}, "finish_reason": None}]}
    choice = out["choices"][0]
    started = time.monotonic()
    try:
        with resp:
            for raw_line in resp:
                elapsed = time.monotonic() - started
                if first_token_s is not None and not (content or reasoning) and elapsed > first_token_s:
                    raise api_backend.TransientApiError(
                        f"no token after {elapsed:.0f} s: the provider holds the request without generating")
                if max_s is not None and elapsed > max_s:
                    raise api_backend.TransientApiError(f"stream still open after {elapsed:.0f} s")
                line = raw_line.decode("utf-8", "replace").strip()
                if not line or line.startswith(":"):      # keep-alive comments
                    continue
                if not line.startswith("data:"):
                    continue
                data = line[5:].strip()
                if data == "[DONE]":
                    break
                chunk = json.loads(data)
                for key in ("id", "provider", "model"):
                    if chunk.get(key):
                        out[key] = chunk[key]
                if chunk.get("usage"):
                    out["usage"] = chunk["usage"]
                if chunk.get("error"):                     # error sent mid-stream
                    choice["finish_reason"] = "error"
                    choice["error"] = chunk["error"]
                    break
                for c in chunk.get("choices") or []:
                    delta = c.get("delta") or {}
                    if delta.get("content"):
                        content.append(delta["content"])
                    thought = delta.get("reasoning") or delta.get("reasoning_content")
                    if thought:                            # OpenRouter / vLLM
                        reasoning.append(thought)
                    if c.get("finish_reason"):
                        choice["finish_reason"] = c["finish_reason"]
                    if c.get("error"):
                        choice["error"] = c["error"]
    except (TimeoutError, OSError) as e:
        raise api_backend.TransientApiError(f"stream interrupted: {e}") from e
    choice["message"]["content"] = "".join(content)
    if reasoning:
        choice["message"]["reasoning"] = "".join(reasoning)
    return out


@backend("openai_compat", system_prompt=True)
def _openai_compat_backend(prompt: str, frame_paths: List[str], model_id: str,
                           opts: Dict[str, Any]) -> GenerationResult:
    """OpenAI-compatible call; on OpenRouter the provider is pinned and recorded.
    The fixed instructions (`opts["system_prompt"]`) go in a system message."""
    address_env = opts.get("api_base_url_env")
    base_url = opts.get("api_base_url") or (os.environ.get(address_env) if address_env else None)
    if address_env and not base_url:
        raise api_backend.ApiError(f"{address_env} not set: export the server address "
                                   f"(e.g. http://localhost:8000/v1)")
    base_url = (base_url or OPENROUTER_BASE).rstrip("/")
    headers: Dict[str, str] = {}
    key_env = opts.get("api_key_env", "OPENROUTER_API_KEY")
    if key_env:                            # null: a local server that takes no key
        key = api_backend.get_api_key(key_env, ENV_PATH)
        if not key:
            raise api_backend.ApiError(f"{key_env} not set (environment or .env)")
        headers["Authorization"] = f"Bearer {key}"

    content: List[Dict[str, Any]] = [{"type": "text", "text": prompt}]
    for p in frame_paths:
        content.append({"type": "image_url",
                        "image_url": {"url": f"data:{api_backend._mime(p)};"
                                             f"base64,{api_backend._b64(p)}"}})
    messages: List[Dict[str, Any]] = [{"role": "user", "content": content}]
    if opts.get("system_prompt") is not None:
        messages.insert(0, {"role": "system", "content": opts["system_prompt"]})
    payload: Dict[str, Any] = {
        "model": model_id,
        "messages": messages,
        "temperature": float(opts.get("temperature") or 0.0),
        "max_tokens": int(opts.get("max_output_tokens") or MAX_OUTPUT_TOKENS),
    }
    if opts.get("seed") is not None:
        payload["seed"] = int(opts["seed"])
    # A configuration that cannot decode greedily declares its model card's
    # sampling setting; the temperature itself arrives through opts above.
    for key, value in (opts.get("sampling") or {}).items():
        if key != "temperature":
            payload[key] = value
    # Reasoning control. {"effort": "none"} truly disables thinking; OpenRouter's
    # {"exclude": true} only hides it while the model still thinks and is billed.
    # Reasoning tokens count toward max_tokens, so thinking + answer share the cap.
    if opts.get("reasoning") is not None:
        payload["reasoning"] = opts["reasoning"]
    if opts.get("chat_template_kwargs") is not None:
        payload["chat_template_kwargs"] = opts["chat_template_kwargs"]
    if not _is_openrouter(base_url):
        payload["stream_options"] = {"include_usage": True}   # vLLM omits usage otherwise
    if _is_openrouter(base_url):
        # Pin the serving stack so a run is reproducible against itself. Without
        # this OpenRouter may silently route to a provider at another quantisation.
        routing: Dict[str, Any] = {"allow_fallbacks": False}
        if opts.get("provider"):
            routing["only"] = [opts["provider"]]
        if opts.get("quantizations"):
            routing["quantizations"] = list(opts["quantizations"])
        payload["provider"] = routing

    limits = (dict(first_token_s=OPENROUTER_FIRST_TOKEN_S, max_s=OPENROUTER_MAX_S)
              if _is_openrouter(base_url) else {})          # a vLLM pod queues by design
    resp = api_backend._with_retries(lambda: _stream_chat(
        f"{base_url}/chat/completions", payload, headers, api_backend.TIMEOUT_S, **limits))
    try:
        choice = resp["choices"][0]
        message = choice["message"]
        text = message.get("content") or ""
    except (KeyError, IndexError, TypeError) as exc:
        raise api_backend.ApiError(
            f"unexpected response shape from {base_url}: {str(resp)[:300]}") from exc
    usage = resp.get("usage", {}) or {}
    api_backend._record_usage(model_id, usage.get("prompt_tokens", 0),
                              usage.get("completion_tokens", 0))
    # A provider failure mid-generation is infrastructure, not a model answer:
    # never record it as an output. Raise so the call is retried, and a resume
    # does not skip the scenario as finished. A stream that closes without any
    # finish reason was cut (SiliconFlow did so after ~10 min on 2026-09-23).
    # `length` (our max_tokens cap) IS a model outcome and stays recorded.
    finish = choice.get("finish_reason")
    if finish in (None, "error"):
        err = choice.get("error") or resp.get("error") or {}
        how = "with an error" if finish == "error" else "without a finish reason (stream cut)"
        raise api_backend.TransientApiError(
            f"provider {resp.get('provider')} ended generation {how} after "
            f"{usage.get('completion_tokens')} tokens: {str(err)[:200]}")
    raw = text if isinstance(text, str) else json.dumps(text)
    trace, final = split_thinking_trace(raw)
    # A separate reasoning channel is the thinking trace; it is stored, not scored.
    channel = message.get("reasoning") or message.get("reasoning_content")
    if channel:
        trace = channel if not trace else f"{channel}\n{trace}"
    details = (usage.get("completion_tokens_details") or {})
    return GenerationResult(
        final_text=final, raw_text=raw, thinking_trace=trace,
        metadata={
            # What actually served the request — the only provenance available.
            "api_base_url": base_url,
            "served_provider": resp.get("provider"),
            "served_model": resp.get("model"),
            "openrouter_generation_id": resp.get("id"),
            "finish_reason": choice.get("finish_reason"),
            "reasoning_tokens": details.get("reasoning_tokens"),
            "reasoning_control": opts.get("reasoning"),
            "chat_template_kwargs": opts.get("chat_template_kwargs"),
            "sampling": opts.get("sampling"),
            "input_tokens": usage.get("prompt_tokens"),
            "output_tokens": usage.get("completion_tokens"),
            "requested_provider": opts.get("provider"),
            "requested_quantizations": opts.get("quantizations"),
        })


def model_config(args) -> Dict[str, Any]:
    """A resolved-panel-shaped mapping for an alias with no pinned revision."""
    return {
        "name": args.name or args.model_id.replace("/", "_"),
        "model_id": args.model_id,
        "backend": "openai_compat",
        "adapter": "openai_compatible_api",
        "revision": None,            # honest: a hosted alias pins no checkpoint
        "inference_mode": "instruct",
        "runtime_ready": False,      # never admitted; diagnostic only
        "adapter_implemented": True,
        "max_output_tokens": args.max_output_tokens,
        "thinking_trace_storage": "separate_artifact",
        "scorer_input": "final_json_only",
        "provider": args.provider,
        "quantizations": args.quantizations,
        "api_base_url": args.base_url,
        "api_key_env": args.api_key_env,
        "reasoning": getattr(args, "reasoning", None),
    }


def run(args) -> Path:
    source = args.dataset_root.resolve()
    chosen = sorted(p for p in source.glob("scenario_*") if p.is_dir())
    if args.scenarios is not None:
        if args.scenarios < 1:
            raise ValueError("--scenarios must be a positive integer")
        chosen = chosen[:args.scenarios]
        if len(chosen) != args.scenarios:
            raise ValueError(f"{source} holds fewer than {args.scenarios} scenarios")
    else:
        chosen = [source / args.scenario]
        if not chosen[0].is_dir():
            raise FileNotFoundError(chosen[0])

    out = args.out_dir.resolve()
    out.mkdir(parents=True, exist_ok=False)
    subset = out / "inputs" / args.split
    subset.mkdir(parents=True)
    for scenario in chosen:
        (subset / scenario.name).symlink_to(scenario, target_is_directory=True)
    rows = dataset_inventory(subset, len(chosen), args.split)

    cfg = model_config(args)
    campaign = {"campaign_dir": str(out), "splits": {args.split: {"path": str(subset)}}}
    frozen = {
        "purpose": "hosted-API prompt diagnostic, not campaign evidence",
        "provenance_warning": (
            "API alias with no pinned revision; not comparable with a local "
            "bf16 run and never pooled with one"),
        "model": cfg,
        "datasets": {args.split: rows},
    }
    write_json(out / "frozen.json", frozen)

    cell = dict(split=args.split, backend="openai_compat", condition=args.condition,
                configuration=cfg["name"],
                out_dir=str(out / "cells" / args.split / f"{cfg['name']}_{args.condition}"))
    write_campaign_identity(campaign, cell)
    print(f"RUN {args.model_id} {args.split}/{len(chosen)} scenario(s)/{args.condition}"
          f"; endpoint={args.base_url}; provider={args.provider or 'auto'}", flush=True)
    stats = run_model(str(subset), cell["out_dir"], backend="openai_compat",
                      condition=args.condition, model_id=args.model_id,
                      model_config=cfg, temperature=0.0, seed=0,
                      provisional_smoke=True)
    write_json(Path(cell["out_dir"]) / "run_stats.json", stats)
    score_cell(campaign, cell, frozen)
    data = collect_cell(cell, rows, split_path=str(subset),
                        frozen_sha256=sha(out / "frozen.json"))
    write_json(Path(cell["out_dir"]) / "collected.json", data)
    records = [dict(split=args.split, configuration=cfg["name"], condition=args.condition,
                    comparison="scores", metric=metric, **mean_stat(list(values.values())))
               for metric, values in data["metrics"].items()]
    export_tables(out, records)
    spend = sum(u.get("input_tokens", 0) + u.get("output_tokens", 0)
                for u in api_backend.USAGE)
    print(f"DONE: {out} ({len(api_backend.USAGE)} calls, {spend} tokens). "
          f"Diagnostic only — confirm on the local stack before the campaign.",
          flush=True)
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model-id", required=True, help="OpenRouter slug, e.g. qwen/qwen3-vl-8b-instruct")
    p.add_argument("--dataset-root", type=Path, required=True)
    p.add_argument("--out-dir", type=Path, required=True, help="New directory; never overwrites")
    p.add_argument("--split", choices=("carla", "nuscenes"), required=True)
    p.add_argument("--condition", choices=CONDITIONS, default="video")
    p.add_argument("--scenario", default="scenario_1")
    p.add_argument("--scenarios", type=int, help="First N scenarios instead of one")
    p.add_argument("--name", help="Cell name; defaults to the slug with / replaced")
    p.add_argument("--provider", help="Pin one OpenRouter provider (recommended)")
    p.add_argument("--quantizations", nargs="*", help="Pin quantisation, e.g. bf16 fp16")
    p.add_argument("--max-output-tokens", type=int, default=MAX_OUTPUT_TOKENS)
    p.add_argument("--base-url", default=OPENROUTER_BASE,
                   help="OpenAI-compatible API base, without /chat/completions")
    p.add_argument("--api-key-env", default="OPENROUTER_API_KEY",
                   help="Name of the environment/.env variable holding the key")
    run(p.parse_args())


if __name__ == "__main__":
    main()
