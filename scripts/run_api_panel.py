"""Run the four-configuration CORIVA panel and its decoding control.

  qwen3.5-9b-direct          our vLLM server, pinned bf16 revision, thinking off, greedy
  qwen3.5-9b-thinking        the same checkpoint, thinking on, the model card's sampling
  qwen3.5-9b-direct-sampled  decoding control: thinking off, sampled as thinking
  mistral-small-3.2          OpenRouter, Parasail bf16, greedy
  gemini-2.5-pro             Gemini Batch API (asynchronous, half the interactive price)

Each configuration runs 2 splits x 3 conditions = 6 cells over the same
scenarios and the same prompt. Every manifest carries
`execution_purpose: smoke_admission`; the OpenRouter and Gemini aliases pin no
revision. Never pool these runs with the local-GPU campaign.

The Qwen configurations call the OpenAI-compatible server whose address is in
CORIVA_VLLM_BASE_URL (e.g. http://localhost:8000/v1). An output root records
the server's vLLM version and served model, and refuses to resume on another.

Subcommands
  run      vLLM and OpenRouter configurations; concurrent inside a cell; resumable
  gemini   submit | status | fetch | retry for every Gemini cell; a cell can be
           submitted in parts, and retry resends only requests without an answer
  summary  one table over every finished cell
  probe    recognition probe: the same 20 nuScenes scenes for every model
  report   the dissertation tables for the whole panel (after every cell is scored)

Pilot first, then the full run:
  python scripts/run_api_panel.py run --config all --scenarios 4
  python scripts/run_api_panel.py gemini submit --scenarios 4 --confirm
  python scripts/run_api_panel.py summary

  python scripts/run_api_panel.py run --config mistral-small-3.2 --workers 16
  python scripts/run_api_panel.py gemini submit --confirm
  python scripts/run_api_panel.py gemini status
  python scripts/run_api_panel.py gemini fetch

Re-running the same command resumes: finished scenarios are skipped, failed
calls are retried. A cell refuses to resume under a different prompt version.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import glob
import json
import os
from pathlib import Path
import sys
import threading
import time
from types import SimpleNamespace
from typing import Any, Dict, List, Optional

_here = Path(__file__).resolve()
_root = next((p for p in _here.parents if (p / "analysis").is_dir()), _here.parent)
sys.path.insert(0, str(_root))

from analysis.backends import api as api_backend
from analysis.evaluation_campaign import (
    BASELINES, CONDITIONS, dataset_inventory, score_cell, sha, write_campaign_identity, write_json)
from analysis.evaluation_report import (
    cell_rows, collect_cell, comparison_rows, export_tables, mean_stat, write_report)
from analysis.evaluation_probe import run_panel_probe
from analysis.model_panel import MAX_OUTPUT_TOKENS, load_model_panel
from analysis.run_model import run_model
from analysis.smoke_admission import repository_state
from scripts import run_gemini_batch as gemini_batch
from scripts import run_openrouter_cell  # noqa: F401  registers the openai_compat backend
from simulation.build_pilot import PROMPT_VARIANTS, PROMPT_VERSION, prompt_version, validate_prompt_variant

OPENROUTER = "https://openrouter.ai/api/v1"
VLLM_ADDRESS_ENV = "CORIVA_VLLM_BASE_URL"
# A served context must hold the longest prompt (~20k tokens with eight images)
# plus the whole output cap.
VLLM_MIN_MODEL_LEN = MAX_OUTPUT_TOKENS + 32768


def _pinned_revision(checkpoint: str) -> str:
    panel = load_model_panel(str(_root / "analysis" / "model_panel.yaml"))
    return next(c["revision"] for c in panel["checkpoints"] if c["key"] == checkpoint)


# Qwen3.5-9B runs on our own vLLM server (2026-09-23): the hosted providers cut
# its streams (Parasail and DeepInfra after 5-6.5 min, SiliconFlow after 7-10
# min), so the longest answers never finished. Mistral stays on Parasail, which
# never cut an answer; it lists 32,768 as its output maximum but accepted a
# 65,536 request, and no Mistral answer exceeded 27k tokens.
#
# Every configuration decodes greedily except Qwen3.5 thinking, which the model
# card says not to decode greedily. In the pod smoke (2026-09-23) greedy
# decoding repeated its thinking until the cap in 4 of 4 answers, and the card's
# precise-task setting (temperature 0.6, no presence penalty) still did in 5 of
# 8 (archived under results/_archive/2026-09-23_qwen_thinking_*_smoke). It takes
# the card's general thinking-mode setting, whose presence penalty is the
# card's measure against endless repetition. Seed 0.
# qwen3.5-9b-direct-sampled is a decoding control, not another model: thinking
# off, sampled exactly as the thinking configuration (the card gives the same
# setting for non-thinking reasoning tasks). Direct greedy vs direct sampled
# isolates the decoding; direct sampled vs thinking isolates the thinking.
QWEN_THINKING = dict(temperature=1.0, top_p=0.95, top_k=20, min_p=0.0,
                     presence_penalty=1.5, repetition_penalty=1.0)
# Cosmos-Reason2-8B model card, "Other Properties Related to Input": "Append
# `Answer the question in the following format: <think>\nyour reasoning\n</think>
# \n\n<answer>\nyour answer\n</answer>.` in the system prompt to encourage long
# chain-of-thought reasoning response." Appended after a blank line.
PAIR_REASONING_SUFFIX = ("\n\nAnswer the question in the following format: <think>\nyour reasoning\n"
                         "</think>\n\n<answer>\nyour answer\n</answer>.")
# The shipped generation_config.json of both checkpoints (identical): NVIDIA's
# reference inference samples with it. Greedy decoding of long reasoning
# repeated until the cap in the Qwen3.5 smoke (4 of 4), so the reasoning
# configurations sample, seed 0; the direct pair stays greedy.
PAIR_SAMPLING = dict(temperature=0.7, top_p=0.8, top_k=20, repetition_penalty=1.0)
PANEL: Dict[str, Dict[str, Any]] = {
    "qwen3.5-9b-direct": dict(
        model_id="Qwen/Qwen3.5-9B", serving="vllm", revision=_pinned_revision("qwen3.5-9b"),
        chat_template_kwargs={"enable_thinking": False}),
    "qwen3.5-9b-thinking": dict(
        model_id="Qwen/Qwen3.5-9B", serving="vllm", revision=_pinned_revision("qwen3.5-9b"),
        chat_template_kwargs={"enable_thinking": True}, sampling=QWEN_THINKING),
    "qwen3.5-9b-direct-sampled": dict(
        model_id="Qwen/Qwen3.5-9B", serving="vllm", revision=_pinned_revision("qwen3.5-9b"),
        chat_template_kwargs={"enable_thinking": False}, sampling=QWEN_THINKING),
    "mistral-small-3.2": dict(
        model_id="mistralai/mistral-small-3.2-24b-instruct", serving="openrouter",
        provider="Parasail", quantizations=["bf16"], reasoning=None),
    # Added 2026-09-24 for the arm A / arm B contrast: a matched pair at equal
    # size. Cosmos-Reason2-8B is NVIDIA's physical-reasoning post-training of
    # Qwen3-VL-8B-Instruct (model card: "post-trained based on
    # Qwen3-VL-8B-Instruct"; gated, NVIDIA Open Model License; its long
    # chain-of-thought is requested through a system prompt, which CORIVA's
    # fixed system prompt does not do, so both run direct and greedy). Neither
    # chat template has a thinking switch, so no chat_template_kwargs.
    "qwen3-vl-8b": dict(
        model_id="Qwen/Qwen3-VL-8B-Instruct", serving="vllm",
        revision=_pinned_revision("qwen3-vl-8b-instruct")),
    "cosmos-reason2-8b": dict(
        model_id="nvidia/Cosmos-Reason2-8B", serving="vllm",
        revision=_pinned_revision("cosmos-reason2-8b")),
    # The pair in the reasoning mode the Cosmos card documents. Both append the
    # card's sentence to CORIVA's system prompt; Qwen3-VL gets it too, so a
    # difference is the post-training, not the request to reason. Same servers as
    # the direct pair (no reasoning parser): the runner keeps the <think> block as
    # the trace and scores the text after it. Smoke only: Cosmos looped to the
    # cap in 2 of 8 arm-A answers and Qwen3-VL ignored the request, so neither
    # runs in full and neither enters the report.
    "qwen3-vl-8b-reasoning": dict(
        model_id="Qwen/Qwen3-VL-8B-Instruct", serving="vllm",
        revision=_pinned_revision("qwen3-vl-8b-instruct"),
        system_prompt_suffix=PAIR_REASONING_SUFFIX, sampling=PAIR_SAMPLING),
    "cosmos-reason2-8b-reasoning": dict(
        model_id="nvidia/Cosmos-Reason2-8B", serving="vllm",
        revision=_pinned_revision("cosmos-reason2-8b"),
        system_prompt_suffix=PAIR_REASONING_SUFFIX, sampling=PAIR_SAMPLING),
}
GEMINI = "gemini-2.5-pro"
SPLITS = {"carla": "data/output_carla400_protocol10",
          "nuscenes": "data/output_nuscenes_protocol10"}
# Batch price of one AI Studio answer to prompt 23d (2026-09-23): 17k input and
# 28.7k output tokens, thinking included.
GEMINI_USD_PER_SCENARIO = 0.154

_print_lock = threading.Lock()


def say(msg: str) -> None:
    with _print_lock:
        print(msg, flush=True)


def model_config(name: str, spec: Dict[str, Any]) -> Dict[str, Any]:
    """A resolved-panel-shaped mapping. The server address is not part of it,
    so a root survives a new pod; the vLLM version is checked in prepare()."""
    cfg = {"name": name, "model_id": spec["model_id"], "backend": "openai_compat",
           "adapter": "openai_compatible_api", "serving": spec["serving"],
           "runtime_ready": False, "adapter_implemented": True,
           "max_output_tokens": MAX_OUTPUT_TOKENS,
           "thinking_trace_storage": "separate_artifact", "scorer_input": "final_json_only"}
    if spec["serving"] == "vllm":
        kwargs = spec.get("chat_template_kwargs")     # absent: the template has no thinking switch
        thinking = bool(kwargs and kwargs.get("enable_thinking"))
        cfg.update(revision=spec["revision"], inference_mode="thinking" if thinking else "direct",
                   enable_thinking=thinking, chat_template_kwargs=kwargs,
                   api_base_url_env=VLLM_ADDRESS_ENV, api_key_env=None)
        if spec.get("system_prompt_suffix"):   # reasoning requested in the prompt text
            cfg.update(inference_mode="prompted_reasoning",
                       system_prompt_suffix=spec["system_prompt_suffix"])
    else:
        effort = (spec.get("reasoning") or {}).get("effort")
        cfg.update(revision=None,                  # a hosted alias pins no checkpoint
                   inference_mode="thinking" if effort not in (None, "none") else "direct",
                   provider=spec["provider"], quantizations=spec["quantizations"],
                   api_base_url=OPENROUTER, api_key_env="OPENROUTER_API_KEY",
                   reasoning=spec.get("reasoning"))
    if spec.get("sampling"):              # absent means greedy, so greedy roots still resume
        cfg["sampling"] = dict(spec["sampling"])
    return cfg


def vllm_server_info(base_url: str) -> Dict[str, Any]:
    """What a vLLM server is serving: its version, model ids and context."""
    import urllib.request
    base = base_url.rstrip("/")
    def get(url):
        with urllib.request.urlopen(url, timeout=30) as r:
            return json.load(r)
    models = get(f"{base}/models").get("data") or []
    version = get(f"{base[:-3] if base.endswith('/v1') else base}/version").get("version")
    return {"vllm_version": version, "served_models": sorted(m.get("id") for m in models),
            "max_model_len": min((m.get("max_model_len") or 0) for m in models) if models else 0}


def choose(split: str, n: Optional[int]) -> List[Path]:
    """All scenarios of a split, or the first n in the campaign's own order."""
    dirs = sorted(p for p in (_root / SPLITS[split]).resolve().glob("scenario_*") if p.is_dir())
    if n is not None:
        if n < 1 or n > len(dirs):
            raise ValueError(f"--scenarios must be 1..{len(dirs)} for {split}")
        dirs = dirs[:n]
    return dirs


def link(subset: Path, chosen: List[Path]) -> None:
    subset.mkdir(parents=True, exist_ok=True)
    for s in chosen:
        target = subset / s.name
        if not target.exists():
            target.symlink_to(s, target_is_directory=True)


# ── run: OpenRouter configurations ───────────────────────────────────────

def prepare(root: Path, name: str, cfg: Dict[str, Any], n: Optional[int],
            prompt_variant: str = "full") -> Dict[str, Any]:
    """Freeze one configuration's inputs; refuse to resume under another prompt
    (version or variant: arm A `full` and arm B `recipe_withheld` never share
    a root)."""
    prompt_variant = validate_prompt_variant(prompt_variant)
    chosen = {split: choose(split, n) for split in SPLITS}
    campaign = {"campaign_dir": str(root), "splits": {}}
    rows = {}
    for split, dirs in chosen.items():
        subset = root / "inputs" / split
        link(subset, dirs)
        campaign["splits"][split] = {"path": str(subset)}
        rows[split] = dataset_inventory(subset, len(dirs), split)
    frozen = {"purpose": "hosted-API panel run, not the local-GPU campaign",
              "prompt_version": prompt_version(prompt_variant), "prompt_variant": prompt_variant,
              "model": cfg,
              "repository": repository_state(str(_root)), "datasets": rows}
    if cfg.get("serving") == "vllm":
        address = os.environ.get(cfg["api_base_url_env"])
        if not address:
            raise RuntimeError(f"export {cfg['api_base_url_env']} (the vLLM server address)")
        server = vllm_server_info(address)
        if cfg["model_id"] not in server["served_models"]:
            raise RuntimeError(f"the vLLM server serves {server['served_models']}, not {cfg['model_id']}")
        if server["max_model_len"] < VLLM_MIN_MODEL_LEN:
            raise RuntimeError(f"start vLLM with --max-model-len >= {VLLM_MIN_MODEL_LEN} "
                               f"(it serves {server['max_model_len']})")
        frozen["serving"] = server
    fpath = root / "frozen.json"
    if fpath.exists():
        old = json.loads(fpath.read_text())
        if old.get("serving") != frozen.get("serving"):
            raise RuntimeError(f"{root} was run on vLLM {old.get('serving')}, now "
                               f"{frozen.get('serving')}; never mix serving stacks in a root")
        if old.get("prompt_version") != prompt_version(prompt_variant):
            raise RuntimeError(
                f"{root} was run with prompt {old.get('prompt_version')}, now "
                f"{prompt_version(prompt_variant)}. Use a new --out-root; never mix prompt "
                f"versions or variants in a cell.")
        if old.get("model") != cfg:
            raise RuntimeError(f"{root} was run with a different model configuration")
        def identity(datasets):
            return {s: [(r["name"], r.get("fingerprint")) for r in v] for s, v in datasets.items()}
        if identity(old["datasets"]) != identity(rows):
            raise RuntimeError(f"{root} was run on a different scenario set, or on scenario inputs whose "
                               f"files have changed since; use a new --out-root")
        frozen = old
    else:
        root.mkdir(parents=True, exist_ok=True)
        write_json(fpath, frozen)
    return dict(campaign=campaign, frozen=frozen, chosen=chosen, prompt_variant=prompt_variant)


def run_cell(root: Path, name: str, cfg: Dict[str, Any], ctx: Dict[str, Any],
             split: str, condition: str, workers: int) -> Optional[Dict[str, Any]]:
    prompt_variant = ctx.get("prompt_variant", "full")
    cell = dict(split=split, backend="openai_compat", condition=condition,
                configuration=name,
                out_dir=str(root / "cells" / split / f"{name}_{condition}"))
    if cfg.get("sampling"):               # check_cell accepts only the declared decoding
        cell["sampling"] = cfg["sampling"]
    Path(cell["out_dir"]).mkdir(parents=True, exist_ok=True)
    write_campaign_identity(ctx["campaign"], cell)
    subset = ctx["campaign"]["splits"][split]["path"]
    names = [p.name for p in ctx["chosen"][split]]
    tag = f"[{name} {split}/{condition}]"
    done = sum((Path(cell["out_dir"]) / n / "run_manifest.json").exists() for n in names)
    say(f"{tag} {len(names)} scenario(s), {done} already done, {workers} worker(s)")

    temperature = (cfg.get("sampling") or {}).get("temperature", 0.0)

    def one(scen: str):
        try:
            run_model(subset, cell["out_dir"], backend="openai_compat", condition=condition,
                      model_id=cfg["model_id"], model_config=cfg, temperature=temperature, seed=0,
                      skip_existing=True, scenario_names=[scen], provisional_smoke=True,
                      prompt_variant=prompt_variant)
            return scen, None
        except Exception as exc:          # transient or client error: retried on resume
            return scen, f"{type(exc).__name__}: {str(exc)[:160]}"

    failed = {}
    started = time.monotonic()
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(one, s) for s in names]
        for k, fut in enumerate(as_completed(futures), 1):
            scen, err = fut.result()
            if err:
                failed[scen] = err
                say(f"{tag} FAILED {scen}: {err}")
            if k % 10 == 0 or k == len(names):
                say(f"{tag} {k}/{len(names)} ({len(failed)} failed, "
                    f"{time.monotonic() - started:.0f}s)")
    if failed:
        write_json(Path(cell["out_dir"]) / "failed_calls.json", failed)
        say(f"{tag} {len(failed)} call(s) failed; re-run the same command to retry them")
        return None
    score_cell(ctx["campaign"], cell, ctx["frozen"])
    data = collect_cell(cell, ctx["frozen"]["datasets"][split], split_path=subset,
                        frozen_sha256=sha(root / "frozen.json"))
    write_json(Path(cell["out_dir"]) / "collected.json", data)
    return {"cell": cell, "data": data}


def cmd_run(args) -> None:
    names = list(PANEL) if args.config == "all" else [args.config]
    conditions = args.conditions or list(CONDITIONS)
    splits = args.splits or list(SPLITS)
    for name in names:
        cfg = model_config(name, PANEL[name])
        root = args.out_root.resolve() / name
        ctx = prepare(root, name, cfg, args.scenarios, getattr(args, "prompt", "full"))
        records = []
        for split in splits:
            for condition in conditions:
                res = run_cell(root, name, cfg, ctx, split, condition, args.workers)
                if res is None:
                    continue
                records += [dict(split=split, configuration=name, condition=condition,
                                 comparison="scores", metric=m, **mean_stat(list(v.values())))
                            for m, v in res["data"]["metrics"].items()]
        if records:
            export_tables(root, records)
        say(f"[{name}] finished; tables in {root}")


# ── gemini: batch submit / status / fetch ────────────────────────────────

def gemini_dirs(args):
    base = args.out_root.resolve() / GEMINI
    for split in (args.splits or list(SPLITS)):
        for condition in (args.conditions or list(CONDITIONS)):
            yield split, condition, base / f"{split}_{condition}"


def cmd_gemini(args) -> None:
    cells = list(gemini_dirs(args))
    if args.action == "submit":
        # A cell can be submitted in parts (a probe, then the rest); only the
        # scenarios it has not submitted yet are sent and priced.
        plans = []
        for split, condition, out in cells:
            state_file = out / gemini_batch.STATE_FILE
            done = set(json.loads(state_file.read_text()).get("mapping") or {}) \
                if state_file.exists() else set()
            plans.append((split, condition, out,
                          sum(p.name not in done for p in choose(split, args.scenarios))))
        n_total = sum(n for *_, n in plans)
        est = n_total * GEMINI_USD_PER_SCENARIO
        say(f"Gemini: {len(cells)} cell(s), {n_total} new request(s), estimated ${est:.2f} "
            f"(batch rate, thinking included, from the measured probe)")
        if not args.confirm:
            say("Nothing submitted. Add --confirm to spend this.")
            return
        for split, condition, out, n_new in plans:
            if not n_new:
                say(f"  {split}/{condition}: already submitted, skipped")
                continue
            say(f"  submitting {split}/{condition}: {n_new} new request(s)")
            gemini_batch.submit(SimpleNamespace(
                dataset_root=_root / SPLITS[split], out_dir=out, split=split,
                condition=condition, scenario="scenario_1",
                scenarios=len(choose(split, args.scenarios)), offset=0,
                display_name=f"coriva-{split}-{condition}", dry_run=False,
                prompt_variant=getattr(args, "prompt", "full")))
    elif args.action == "status":
        for split, condition, out in cells:
            if not (out / gemini_batch.STATE_FILE).exists():
                say(f"{split}/{condition}: not submitted"); continue
            say(f"{split}/{condition}:")
            gemini_batch.status(SimpleNamespace(out_dir=out))
    elif args.action == "fetch":
        for split, condition, out in cells:
            if not (out / gemini_batch.STATE_FILE).exists():
                say(f"{split}/{condition}: not submitted"); continue
            state = json.loads((out / gemini_batch.STATE_FILE).read_text())
            written = {p.parent.name for p in out.glob("cells/*/*/scenario_*/run_manifest.json")}
            if state.get("mapping") and set(state["mapping"]) <= written:
                say(f"{split}/{condition}: already fetched ({len(written)} scenario(s))"); continue
            st = gemini_batch.status(SimpleNamespace(out_dir=out))
            if st == "RUNNING":
                say(f"{split}/{condition}: running, fetch later"); continue
            try:
                gemini_batch.fetch(SimpleNamespace(out_dir=out))
            except RuntimeError as exc:           # answers missing: retry resends them
                say(f"{split}/{condition}: {exc}")
    elif args.action == "retry":
        # Only requests without an answer are resent: an error line, a line a
        # finished job left out, or a failed, cancelled or expired job.
        plans = []
        for split, condition, out in cells:
            if not (out / gemini_batch.STATE_FILE).exists():
                continue
            state = json.loads((out / gemini_batch.STATE_FILE).read_text())
            answers, failures, pending = gemini_batch._collect(out, state)
            if pending:
                say(f"{split}/{condition}: {len(pending)} job(s) still running, retry later")
                continue
            todo = [n for n in state["mapping"] if n not in answers]
            if todo:
                plans.append((split, condition, out, len(todo)))
        n_total = sum(n for *_, n in plans)
        say(f"Gemini: {n_total} request(s) without an answer in {len(plans)} cell(s), "
            f"estimated ${n_total * GEMINI_USD_PER_SCENARIO:.2f}")
        if not args.confirm:
            say("Nothing resent. Add --confirm to spend this.")
            return
        for split, condition, out, n in plans:
            say(f"  retrying {split}/{condition}: {n} request(s)")
            gemini_batch.retry(SimpleNamespace(out_dir=out))
    elif args.action == "verify":
        # The saved requests of every submitted cell must share one system
        # instruction and be what the current code builds; a state written
        # before the fingerprint check gets its fingerprint from those files.
        for split, condition, out in cells:
            if not (out / gemini_batch.STATE_FILE).exists():
                say(f"{split}/{condition}: not submitted"); continue
            gemini_batch.verify(SimpleNamespace(out_dir=out, rebuild=True))
    elif args.action == "usage":
        # Tokens and batch cost of the answers so far, nothing scored: read after
        # a probe to price the rest of the cell before submitting it.
        for split, condition, out in cells:
            if not (out / gemini_batch.STATE_FILE).exists():
                say(f"{split}/{condition}: not submitted"); continue
            state = json.loads((out / gemini_batch.STATE_FILE).read_text())
            answers, failures, pending = gemini_batch._collect(out, state)
            rows = [((r.get("usageMetadata") or {}), ((r.get("candidates") or [{}])[0]).get("finishReason"))
                    for r in answers.values()]
            if not rows:
                say(f"{split}/{condition}: no answer yet, {len(pending)} job(s) running"); continue
            tok_in = [u.get("promptTokenCount", 0) for u, _ in rows]
            tok_out = [u.get("candidatesTokenCount", 0) + u.get("thoughtsTokenCount", 0) for u, _ in rows]
            usd = (sum(tok_in) * 0.625 + sum(tok_out) * 5.0) / 1e6    # batch rate, as in fetch
            per = usd / len(rows)
            rest = len(choose(split, None)) - len(state["mapping"])
            finish = {}
            for _, f in rows:
                finish[f] = finish.get(f, 0) + 1
            say(f"{split}/{condition}: {len(rows)} answer(s), {len(failures)} failed, "
                f"{len(pending)} job(s) running; per request {sum(tok_in) / len(rows):.0f} in, "
                f"{sum(tok_out) / len(rows):.0f} out (max {max(tok_out)} of {MAX_OUTPUT_TOKENS}); "
                f"finish {finish}; ${usd:.2f} so far (${per:.3f}/request); "
                f"{rest} not yet submitted, about ${rest * per:.2f}")


# ── summary ──────────────────────────────────────────────────────────────

AXES = ("monotonicity", "responsiveness", "locality", "action_direction", "speed_response")


def cmd_summary(args) -> None:
    rows = []
    for rep in sorted(glob.glob(str(args.out_root.resolve() / "**" / "report.json"), recursive=True)):
        cell = Path(rep).parent
        r = json.load(open(rep))
        s = r["summary"]
        co = json.load(open(cell / "coherence.json"))["axes"] if (cell / "coherence.json").exists() else {}
        scen = sorted(cell.glob("scenario_*"))
        nbr, cut, think = [], 0, 0
        gt_n = r["per_sample"][0]["layer3"]["diagnostics"]["gt_count"] if r["per_sample"] else 0
        for d in scen:
            l3 = json.load(open(d / "l3_pre.json")) if (d / "l3_pre.json").exists() else {}
            nbr.append(len(l3.get("predicted_branches") or []))
            m = (json.load(open(d / "run_manifest.json")).get("generation_metadata") or {}) \
                if (d / "run_manifest.json").exists() else {}
            cut += m.get("finish_reason") in ("length", "MAX_TOKENS")
            think += m.get("reasoning_tokens") or 0
        full = sum(1 for b in nbr if b >= 2)
        rows.append(dict(
            cell=cell.name, split=cell.parent.name, n=len(scen),
            multi=f"{full}/{len(scen)}",
            rq1=f"{sum(co.get(a, {}).get('mean') is not None for a in AXES)}/5",
            pair=f"{s['L4_pair_valid_rate']:.2f}",
            ctms=f"{s['CTMS_mean']:.3f}", cut=cut))
    if not rows:
        say(f"No scored cell under {args.out_root}"); return
    say(f"{'cell':42s} {'split':9s} {'n':>4s} {'>=2 branch':>10s} {'RQ1 axes':>9s} "
        f"{'L4 pair':>8s} {'CTMS':>6s} {'truncated':>9s}")
    for x in rows:
        say(f"{x['cell']:42s} {x['split']:9s} {x['n']:>4d} {x['multi']:>10s} {x['rq1']:>9s} "
            f"{x['pair']:>8s} {x['ctms']:>6s} {x['cut']:>9d}")


# ── recognition probe ────────────────────────────────────────────────────

PROBE_LIMIT, PROBE_SEED = 20, 0          # evaluation/config.yaml: probe.limit, statistics.seed


def probe_configs() -> Dict[str, Dict[str, Any]]:
    """The models of the panel. The decoding control is not a model and is
    not probed. The probe's answer is a short JSON object, so the Qwen3.5
    configurations are called through OpenRouter; Qwen3-VL-8B and
    Cosmos-Reason2-8B are probed on the pod's vLLM server like their cells."""
    qwen = dict(model_id="qwen/qwen3.5-9b", serving="openrouter", provider="Parasail",
                quantizations=["bf16"])
    return {
        "qwen3.5-9b-direct": model_config("qwen3.5-9b-direct",
                                          dict(qwen, reasoning={"effort": "none"})),
        "qwen3.5-9b-thinking": model_config("qwen3.5-9b-thinking",
                                            dict(qwen, reasoning={"effort": "medium"},
                                                 sampling=QWEN_THINKING)),
        "mistral-small-3.2": model_config("mistral-small-3.2", PANEL["mistral-small-3.2"]),
        "qwen3-vl-8b": model_config("qwen3-vl-8b", PANEL["qwen3-vl-8b"]),
        "cosmos-reason2-8b": model_config("cosmos-reason2-8b", PANEL["cosmos-reason2-8b"]),
        GEMINI: gemini_batch._panel_config(),
    }


def probe_generator(model: Dict[str, Any], backend=None):
    """run_panel_probe calls every model at temperature 0; a configuration that
    declares sampling (Qwen thinking) is probed with it, seed 0."""
    from analysis.run_model import BACKENDS
    call = backend or BACKENDS[model["backend"]]
    temperature = (model.get("sampling") or {}).get("temperature", 0.0)

    def generate(prompt, frames, model_id, opts):
        return call(prompt, frames, model_id,
                    dict(opts, temperature=temperature, seed=0 if temperature else None))
    return generate


def cmd_probe(args) -> None:
    """The same 20 nuScenes scenes for every model; reports under probes/<model>/."""
    out_root = args.out_root.resolve()
    frozen = json.loads((out_root / "qwen3.5-9b-direct" / "frozen.json").read_text())
    cfg = dict(campaign_dir=str(out_root), statistics=dict(seed=PROBE_SEED),
               probe=dict(limit=PROBE_LIMIT),
               splits=dict(nuscenes=dict(path=str((_root / SPLITS["nuscenes"]).resolve()))))
    configs = probe_configs()
    for name in (list(configs) if args.config == "all" else [args.config]):
        report = run_panel_probe(cfg, name, frozen, generator=probe_generator(configs[name]),
                                 resolved_model=configs[name])
        say(f"[probe {name}] {report['summary']}")


# ── report ────────────────────────────────────────────────────────────────

# Every model of the panel is ranked and compared with the floors; a cell the
# root lacks (arm B runs video only; a model may not have run) is left out and
# listed. The decoding control enters only its two paired comparisons: sampled
# minus greedy with thinking off, and thinking minus no thinking under the same
# sampling. The matched pair enters as Cosmos-Reason2 minus Qwen3-VL, direct
# (posttraining_effect). The pair's reasoning configurations stopped at the
# smoke test and are left out.
DECODING_CONTROL = "qwen3.5-9b-direct-sampled"
SMOKE_ONLY = ("qwen3-vl-8b-reasoning", "cosmos-reason2-8b-reasoning")
REPORT_MODELS = tuple(name for name in PANEL if name not in (DECODING_CONTROL, *SMOKE_ONLY)) + (GEMINI,)
MODEL_PAIRS = ((DECODING_CONTROL, "qwen3.5-9b-direct", "decoding_effect"),
               ("qwen3.5-9b-thinking", DECODING_CONTROL, "thinking_effect"),
               ("cosmos-reason2-8b", "qwen3-vl-8b", "posttraining_effect"))
REPORT_BOOT, REPORT_SEED = 10000, 0      # evaluation/config.yaml: statistics


def panel_cells(out_root: Path, baselines_root: Optional[Path] = None):
    """(key, cell, root) for every model cell, the decoding control and the six
    baselines; `root` holds the frozen.json the cell was scored under. The
    baselines read no prompt, so an arm may take them from another root
    (`baselines_root`, default `<out_root>/baselines`)."""
    for name in REPORT_MODELS + (DECODING_CONTROL,):
        for split in SPLITS:
            for condition in CONDITIONS:
                if name == GEMINI:
                    root = out_root / GEMINI / f"{split}_{condition}"
                    cell = dict(split=split, backend="gemini", condition=condition, configuration=name,
                                out_dir=str(root / "cells" / split / f"{name}_{condition}"))
                else:
                    root = out_root / name
                    cell = dict(split=split, backend="openai_compat", condition=condition,
                                configuration=name, out_dir=str(root / "cells" / split / f"{name}_{condition}"))
                    if PANEL[name].get("sampling"):
                        cell["sampling"] = dict(PANEL[name]["sampling"])
                yield (split, name, condition), cell, root
    for split in SPLITS:
        for baseline in BASELINES:
            root = (baselines_root or out_root / "baselines").resolve()
            yield (split, baseline, "video"), dict(split=split, backend=baseline, condition="video",
                                                   configuration=None,
                                                   out_dir=str(root / "cells" / split / baseline)), root


def cmd_report(args) -> None:
    """The dissertation tables for the API panel under <out-root>/report."""
    out_root = args.out_root.resolve()
    baselines_root = (getattr(args, "baselines_root", None) or out_root / "baselines").resolve()
    cells, summaries, records, frozen_sha, missing, variants = {}, {}, [], {}, [], set()
    for key, cell, root in panel_cells(out_root, baselines_root):
        split, name, _ = key
        if not (root / "frozen.json").exists() or not any(Path(cell["out_dir"]).glob("scenario_*/run_manifest.json")):
            missing.append(key); continue
        frozen = json.loads((root / "frozen.json").read_text())
        variants.add(frozen.get("prompt_variant", "full") if name not in BASELINES else None)
        label = str(root.relative_to(out_root)) if root.is_relative_to(out_root) else str(root)
        frozen_sha[label] = sha(root / "frozen.json")
        cells[key] = collect_cell(cell, frozen["datasets"][split],
                                  split_path=str((_root / SPLITS[split]).resolve()),
                                  frozen_sha256=sha(root / "frozen.json"))
        if name != DECODING_CONTROL:
            rows, summaries["/".join(key)] = cell_rows(key, cells[key], REPORT_BOOT, REPORT_SEED)
            records += rows
    variants.discard(None)
    if len(variants) > 1:
        raise ValueError(f"{out_root} mixes prompt variants {sorted(variants)}; one report per arm")
    variant = variants.pop() if variants else "full"
    for key in missing:
        say(f"[report] no cell for {key}; left out")
    records += comparison_rows(cells, list(REPORT_MODELS), list(SPLITS), REPORT_BOOT, REPORT_SEED,
                               MODEL_PAIRS)
    probes = {p.parent.name: json.loads(p.read_text()) for p in (out_root / "probes").glob("*/report.json")}
    (out_root / "report").mkdir(parents=True, exist_ok=True)
    write_report(out_root / "report", frozen_sha, records, summaries, probes, REPORT_BOOT, REPORT_SEED,
                 prompt_variant=variant, cells_missing=missing, baselines_root=str(baselines_root))


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out-root", type=Path, default=_root / "results" / "api_panel")
    p.add_argument("--prompt", choices=PROMPT_VARIANTS, default="full",
                   help="Prompt variant: full (arm A, prints the motion profiles) or "
                        "recipe_withheld (arm B). A variant needs its own --out-root.")
    sub = p.add_subparsers(dest="command", required=True)

    r = sub.add_parser("run", help="OpenRouter configurations")
    r.add_argument("--config", choices=list(PANEL) + ["all"], required=True)
    r.add_argument("--scenarios", type=int, help="First N scenarios of each split (pilot)")
    r.add_argument("--splits", nargs="*", choices=list(SPLITS))
    r.add_argument("--conditions", nargs="*", choices=list(CONDITIONS))
    r.add_argument("--workers", type=int, default=8, help="Concurrent calls inside a cell")

    g = sub.add_parser("gemini", help="Gemini batch cells")
    g.add_argument("action", choices=("submit", "status", "fetch", "retry", "verify", "usage"))
    g.add_argument("--scenarios", type=int)
    g.add_argument("--splits", nargs="*", choices=list(SPLITS))
    g.add_argument("--conditions", nargs="*", choices=list(CONDITIONS))
    g.add_argument("--confirm", action="store_true", help="Actually spend money on submit")

    sub.add_parser("summary", help="Table over every finished cell")
    pr = sub.add_parser("probe", help="Recognition probe: 20 nuScenes scenes per model")
    pr.add_argument("--config", default="all", choices=[*probe_configs(), "all"])
    rp = sub.add_parser("report", help="Dissertation tables for the whole panel")
    rp.add_argument("--baselines-root", type=Path,
                    help="Root holding baselines/cells (default <out-root>/baselines); the controls read "
                         "no prompt, so an arm B root may point at the arm A baselines")
    args = p.parse_args()
    {"run": cmd_run, "gemini": cmd_gemini, "summary": cmd_summary,
     "probe": cmd_probe, "report": cmd_report}[args.command](args)


if __name__ == "__main__":
    main()
