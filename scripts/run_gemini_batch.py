"""Run a CORIVA Gemini cell through the Batch API at half the interactive price.

Why two phases
--------------
Batch jobs are asynchronous with a 24-hour target turnaround, so submitting and
collecting cannot be one call. `submit` builds the request file and starts the
job; `fetch` collects it and writes the normal cell artifacts.

How it avoids a second implementation
-------------------------------------
Neither phase re-implements prompt building, parsing or scoring. Both run the
real `run_model` gemini path with `analysis.backends.api._post_json` stubbed:

  submit  the stub RECORDS the payload the live API would have received, so
          every batch line is byte-identical to an interactive request
  fetch   the stub REPLAYS the batch response for that prompt

Requests are keyed by the sha256 of the prompt text. If the prompt changes
between the two phases the lookup fails loudly instead of silently pairing a
new prompt with an old answer.

Cost note
---------
Batch is 50% of interactive pricing. Probe with one or two scenarios first and
read `usage.json` before committing a full split: image tokenisation differs
from the local tokeniser, so the local `input_tokens` is not a usable estimate.

Usage
-----
  export GEMINI_API_KEY=...
  python scripts/run_gemini_batch.py submit --scenarios 2 \
      --dataset-root data/output_nuscenes_protocol10 \
      --split nuscenes --condition video --out-dir /tmp/gb/probe
  python scripts/run_gemini_batch.py status --out-dir /tmp/gb/probe
  python scripts/run_gemini_batch.py fetch  --out-dir /tmp/gb/probe

A cell can be submitted in parts: a later submit sends only the scenarios the
cell has not submitted yet and appends its jobs. If a request errors, a job
fails or expires, `retry` resends only the scenarios without an answer; fetch
takes each scenario's answer from whichever job gave it. An answer is never
resent, whatever it says.
  python scripts/run_gemini_batch.py retry  --out-dir /tmp/gb/probe
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import sys
import time
import urllib.request
from typing import Any, Dict, List, Optional, Tuple

_here = Path(__file__).resolve()
_root = next((p for p in _here.parents if (p / "analysis").is_dir()), _here.parent)
sys.path.insert(0, str(_root))

from analysis.backends import api as api_backend
from analysis.evaluation_campaign import (
    CONDITIONS, dataset_inventory, sha, score_cell, write_campaign_identity, write_json)
from analysis.evaluation_report import collect_cell, export_tables, mean_stat
from analysis.model_panel import load_model_panel, resolve_configurations
from analysis.run_model import run_model
from simulation.build_pilot import PROMPT_VARIANTS, prompt_version

PANEL = _root / "analysis/model_panel.yaml"
BASE = "https://generativelanguage.googleapis.com"
UPLOAD_URL = f"{BASE}/upload/v1beta/files"
BATCH_URL = BASE + "/v1beta/models/{model}:batchGenerateContent"
STATE_FILE = "batch_state.json"
UPLOAD_FLOOR_BYTES_PER_S = 50_000
# Google caps one input file at 2GB, but a file that size takes over an hour
# on a slow uplink and one broken connection loses all of it (a 1.8GB CARLA
# upload broke after 35 minutes, 2026-09-24). Each job file is recorded as its
# job is created, so smaller files cost only themselves when an upload breaks:
# ~40 CARLA image requests (~9MB each) per file.
CHUNK_LIMIT = 400_000_000
# A well-formed response that lets the staging run finish; its text is never
# scored, the staging directory is discarded.
STAGING_RESPONSE = {"candidates": [{"content": {"parts": [{"text": "{}"}]},
                                    "finishReason": "STOP"}],
                    "modelVersion": "staging-capture", "usageMetadata": {}}


def _prompt_sha(payload: Dict[str, Any]) -> str:
    """Key a request by its user prompt text. The system instruction and the
    generation config are the same for every request of a cell and are frozen
    separately (`_request_fingerprint`), so the key stays stable across the
    cell's jobs while a drift in either is still refused."""
    for part in payload.get("contents", [{}])[0].get("parts", []):
        if "text" in part:
            return hashlib.sha256(part["text"].encode("utf-8")).hexdigest()
    raise ValueError("payload carries no text part")


def _request_fingerprint(payload: Dict[str, Any]) -> Dict[str, Any]:
    """What a cell shares across requests and must not change between submit
    and fetch: the system instruction (worked example and rules) and the
    generation config."""
    parts = (payload.get("systemInstruction") or {}).get("parts") or []
    system = "".join(p.get("text", "") for p in parts if isinstance(p, dict))
    return {"system_sha256": hashlib.sha256(system.encode("utf-8")).hexdigest(),
            "generation_config": payload.get("generationConfig")}


def _check_fingerprint(state: Dict[str, Any], payload: Dict[str, Any], stage: str) -> None:
    """Refuse a request whose system instruction or generation config differs
    from what the cell was submitted with. A state written before this check
    carries no fingerprint and is accepted as it is."""
    expected = state.get("request_fingerprint")
    if not expected:
        return
    actual = _request_fingerprint(payload)
    if actual != expected:
        raise RuntimeError(
            f"the system prompt or generation config changed between submit and {stage}: "
            f"{expected} != {actual}. Re-submit in a new --out-dir rather than pairing "
            f"an old answer with a new prompt.")


def _refuse_fingerprint_drift(out: Path, previous: Optional[Dict[str, Any]],
                              cfg: Dict[str, Any], stage: str) -> None:
    """Refuse to send a request whose system instruction or generation config
    differs from the cell's recorded fingerprint (submit and retry alike)."""
    if previous is None or not previous.get("request_fingerprint") or not cfg.get("request_fingerprint"):
        return
    if previous["request_fingerprint"] != cfg["request_fingerprint"]:
        raise RuntimeError(f"{out}: the system prompt or generation config changed since the cell was "
                           f"submitted ({stage}); use a new --out-dir")


def _key(name: str, headers: Dict[str, str]) -> Dict[str, str]:
    return headers


def _http(url: str, data: Optional[bytes], headers: Dict[str, str],
          method: str = "POST", timeout: Optional[float] = None) -> bytes:
    req = urllib.request.Request(url, data=data, headers=headers, method=method)
    with urllib.request.urlopen(req, timeout=timeout or api_backend.TIMEOUT_S) as resp:
        return resp.read()


def _gemini_key() -> str:
    key = api_backend.get_api_key("GEMINI_API_KEY")
    if not key:
        raise api_backend.ApiError("GEMINI_API_KEY not set (environment or .env)")
    return key


def _panel_config() -> Dict[str, Any]:
    panel = load_model_panel(str(PANEL))
    cfg = resolve_configurations(panel, ["gemini-2.5-pro"])[0]
    # Deliberately NOT tagged with model_panel_version: that field marks an
    # admitted campaign run and would then require the smoke-admission fields.
    # This path is a diagnostic, so the honest metadata is "no panel version".
    return cfg


def _select(source: Path, args) -> List[Path]:
    dirs = sorted(p for p in source.glob("scenario_*") if p.is_dir())
    offset = getattr(args, "offset", 0) or 0
    if args.scenarios is not None:
        if args.scenarios < 1:
            raise ValueError("--scenarios must be a positive integer")
        window = dirs[offset:offset + args.scenarios]
        if len(window) != args.scenarios:
            raise ValueError(
                f"{source} holds {len(dirs)} scenarios; "
                f"offset {offset} + {args.scenarios} runs past the end")
        return window
    chosen = source / args.scenario
    if not chosen.is_dir():
        raise FileNotFoundError(chosen)
    return [chosen]


def _link_inputs(out: Path, split: str, chosen: List[Path], name: str = "inputs") -> Path:
    subset = out / name / split
    subset.mkdir(parents=True, exist_ok=True)
    for scenario in chosen:
        link = subset / scenario.name
        if not link.exists():
            link.symlink_to(scenario, target_is_directory=True)
    return subset


def _run_with_stub(stub, subset: Path, out_dir: str, condition: str,
                   cfg: Dict[str, Any], prompt_variant: str = "full") -> Dict[str, Any]:
    """Run the real gemini path with the transport replaced."""
    original_post, original_key = api_backend._post_json, api_backend.get_api_key
    api_backend._post_json = stub
    # The transport is stubbed, so no credential is used inside this call. Keep
    # a real key when one is configured; supply a placeholder when it is not,
    # so capture and replay work without credentials.
    api_backend.get_api_key = lambda name, *a, **k: original_key(name, *a, **k) or "stubbed"
    try:
        return run_model(str(subset), out_dir, backend="gemini",
                         condition=condition, model_id=cfg["model_id"],
                         model_config=cfg, temperature=0.0, seed=0,
                         provisional_smoke=True, prompt_variant=prompt_variant)
    finally:
        api_backend._post_json, api_backend.get_api_key = original_post, original_key


# ── submit ────────────────────────────────────────────────────────────────

def _build_lines(out: Path, source: Path, split: str, condition: str, chosen: List[Path],
                 submitted: Dict[str, str], resend: frozenset = frozenset(),
                 prompt_variant: str = "full"):
    """Batch lines for the chosen scenarios, built through the real prompt path.

    A scenario in `submitted` must still produce the request that was sent (its
    prompt digest); it is left out unless it is in `resend`."""
    # Requests are built from exactly the chosen scenarios; the cell's own
    # inputs/ is written by fetch alone, from the scenarios the state records.
    if (out / "staging_inputs").exists():
        shutil.rmtree(out / "staging_inputs")
    subset = _link_inputs(out, split, chosen, name="staging_inputs")
    dataset_inventory(subset, len(chosen), split)
    cfg = _panel_config()
    captured: Dict[str, Dict[str, Any]] = {}

    fingerprints: List[Dict[str, Any]] = []

    def capture(url, payload, headers, timeout):
        captured[_prompt_sha(payload)] = payload
        fingerprints.append(_request_fingerprint(payload))
        return STAGING_RESPONSE

    staging = out / "staging"
    if staging.exists():
        shutil.rmtree(staging)
    print(f"Building {len(chosen)} request(s) through the real prompt path...", flush=True)
    _run_with_stub(capture, subset, str(staging), condition, cfg, prompt_variant)
    lines: List[str] = []
    mapping: Dict[str, str] = {}
    for scenario in chosen:
        prompt_file = staging / scenario.name / "prompt.txt"
        digest = hashlib.sha256(prompt_file.read_text(encoding="utf-8").encode("utf-8")).hexdigest()
        payload = captured.get(digest)
        if payload is None:
            raise RuntimeError(f"no captured request for {scenario.name}")
        if scenario.name in submitted:
            if submitted[scenario.name] != digest:
                raise RuntimeError(
                    f"the prompt of {scenario.name} changed since it was submitted; "
                    f"its answer would not match. Use a new --out-dir.")
            if scenario.name not in resend:
                continue
        mapping[scenario.name] = digest
        lines.append(json.dumps({"key": scenario.name, "request": payload}))
    shutil.rmtree(staging)
    shutil.rmtree(out / "staging_inputs")
    if any(f != fingerprints[0] for f in fingerprints):
        raise RuntimeError("the captured requests do not share one system instruction and generation config")
    cfg["request_fingerprint"] = fingerprints[0] if fingerprints else None
    return cfg, lines, mapping


def _send_lines(out: Path, source: Path, split: str, condition: str, cfg: Dict[str, Any],
                lines: List[str], mapping: Dict[str, str], previous: Optional[Dict[str, Any]],
                display_name: Optional[str], dry_run: bool = False, retry: bool = False,
                prompt_variant: str = "full") -> None:
    """Write the lines as job files after the cell's existing ones, upload them
    and append one job per file to the cell's state."""
    first_chunk = len(_jobs(previous)) if previous is not None else 0
    chunks: List[List[str]] = [[]]
    size = 0
    for line in lines:
        n = len(line.encode("utf-8")) + 1
        if chunks[-1] and size + n > CHUNK_LIMIT:
            chunks.append([]); size = 0
        chunks[-1].append(line); size += n
    files = []
    for k, chunk in enumerate(chunks, start=first_chunk):
        f = out / f"batch_input_{k}.jsonl"
        f.write_text("\n".join(chunk) + "\n", encoding="utf-8")
        files.append(f)
    total = sum(f.stat().st_size for f in files)
    print(f"  {len(lines)} request(s), {total/1e6:.1f} MB in {len(files)} job file(s)", flush=True)

    order = {name: i for i, name in enumerate(sorted(p.name for p in source.glob("scenario_*")))}

    def cell_state(names: Dict[str, str]) -> Dict[str, Any]:
        return dict(mapping=names, split=split, condition=condition, prompt_variant=prompt_variant,
                    request_fingerprint=cfg.get("request_fingerprint"),
                    dataset_root=str(source), scenarios=sorted(names, key=order.__getitem__))

    if dry_run:
        if previous is None:
            write_json(out / STATE_FILE, dict(cell_state(mapping), dry_run=True,
                                              files=[f.name for f in files]))
        print(f"DRY RUN: wrote {len(files)} file(s) in {out}; nothing submitted.", flush=True)
        return

    key = _gemini_key()
    now = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    jobs = list(_jobs(previous)) if previous is not None else []
    sent = dict((previous or {}).get("mapping") or {})
    for f in files:
        k = int(f.stem.rsplit("_", 1)[1])
        file_name = _upload(key, f.read_bytes())
        name = _create_job(key, cfg["model_id"], f"{display_name or out.name}-{k}", file_name)
        keys = [json.loads(l)["key"] for l in f.read_text(encoding="utf-8").splitlines() if l.strip()]
        jobs.append(dict(batch_name=name, input_file=file_name, chunk=k, keys=keys,
                         submitted_at=now, **({"retry": True} if retry else {})))
        sent.update({n: mapping[n] for n in keys})
        print(f"  job {k}: {name} ({len(keys)} request(s), uploaded as {file_name})", flush=True)
        # Saved after every job and only with the scenarios that job carries, so
        # an interruption neither loses a paid request nor marks an unsent one.
        write_json(out / STATE_FILE, dict(cell_state(sent), jobs=jobs, model_id=cfg["model_id"],
                   submitted_at=(previous or {}).get("submitted_at", now)))
    print(f"SUBMITTED {len(files)} job(s)\n  poll: .venv/bin/python scripts/run_gemini_batch.py "
          f"status --out-dir {out}", flush=True)


def submit(args) -> None:
    """Submit the chosen scenarios that this cell has not submitted yet.

    A cell can be built in parts (a one-scenario probe, then the rest): each
    submit appends jobs to the cell's state, and fetch replays every job at
    once. A scenario already submitted is never sent again, and its prompt must
    still produce the request that was sent."""
    out = args.out_dir.resolve()
    variant = getattr(args, "prompt_variant", "full") or "full"
    previous = json.loads((out / STATE_FILE).read_text()) if (out / STATE_FILE).exists() else None
    if previous is not None:
        if previous.get("dry_run"):
            raise FileExistsError(f"{out/STATE_FILE} holds a dry run; use a new --out-dir")
        if (previous["split"], previous["condition"]) != (args.split, args.condition):
            raise ValueError(f"{out} holds {previous['split']}/{previous['condition']}, "
                             f"not {args.split}/{args.condition}")
        if previous.get("prompt_variant", "full") != variant:
            raise ValueError(f"{out} holds prompt variant {previous.get('prompt_variant', 'full')}, "
                             f"not {variant}; use a new --out-dir")
    out.mkdir(parents=True, exist_ok=True)
    source = args.dataset_root.resolve()
    chosen = _select(source, args)
    cfg, lines, mapping = _build_lines(out, source, args.split, args.condition, chosen,
                                       dict((previous or {}).get("mapping") or {}),
                                       prompt_variant=variant)
    _refuse_fingerprint_drift(out, previous, cfg, "submit")
    if not lines:
        print(f"Nothing new: all {len(chosen)} scenario(s) are already submitted.", flush=True)
        return
    _send_lines(out, source, args.split, args.condition, cfg, lines, mapping, previous,
                args.display_name, dry_run=args.dry_run, prompt_variant=variant)


def retry(args) -> None:
    """Resend only the submitted scenarios that have no answer: an error line,
    a line missing from a finished job, or a failed, cancelled or expired job.
    An answer (whatever it says, a capped one included) is never resent."""
    out = args.out_dir.resolve()
    state = json.loads((out / STATE_FILE).read_text())
    if state.get("dry_run"):
        raise ValueError("this directory holds a dry run; submit for real first")
    answers, failures, pending = _collect(out, state)
    if pending:
        raise RuntimeError(f"{len(pending)} job(s) not finished ({pending[0]}); "
                           f"retry only resends once every job has ended")
    todo = [n for n in state["scenarios"] if n not in answers]
    if not todo:
        print("Nothing to retry: every submitted scenario has an answer.", flush=True)
        return
    write_json(out / "batch_errors.json", {n: failures.get(n, "no answer") for n in todo})
    print(f"Retrying {len(todo)} request(s) without an answer", flush=True)
    source = Path(state["dataset_root"])
    variant = state.get("prompt_variant", "full")
    cfg, lines, mapping = _build_lines(out, source, state["split"], state["condition"],
                                       [source / n for n in todo], state["mapping"],
                                       resend=frozenset(todo), prompt_variant=variant)
    _refuse_fingerprint_drift(out, state, cfg, "retry")
    _send_lines(out, source, state["split"], state["condition"], cfg, lines, mapping, state,
                f"coriva-{state['split']}-{state['condition']}-retry", retry=True,
                prompt_variant=variant)


def verify(args) -> None:
    """Prove that a cell's saved requests are the ones the current code builds,
    and give a state written before the fingerprint check its fingerprint
    from the saved request files (never from the network).

    Every `batch_input_*.jsonl` line must share one system instruction and
    generation config, and carry the prompt digest the state recorded for its
    scenario. With `rebuild` (the default) the first submitted scenario is
    built again through the real prompt path and must reproduce its prompt
    digest and that fingerprint, which is the batch counterpart of the pod's
    per-scenario `system_prompt_sha256` check."""
    out = args.out_dir.resolve()
    state = json.loads((out / STATE_FILE).read_text())
    # Every saved line is checked on its own before it is keyed by scenario:
    # a scenario sent twice (a retry) must carry the same prompt digest and
    # the same fingerprint both times.
    lines: List[Tuple[str, str, Dict[str, Any]]] = []
    for f in sorted(out.glob("batch_input_*.jsonl")):
        for line in f.read_text(encoding="utf-8").splitlines():
            if line.strip():
                rec = json.loads(line)
                lines.append((rec["key"], _prompt_sha(rec["request"]), _request_fingerprint(rec["request"])))
    if not lines:
        raise RuntimeError(f"{out}: no batch_input_*.jsonl to verify")
    fingerprints = {json.dumps(fp, sort_keys=True) for _, _, fp in lines}
    if len(fingerprints) != 1:
        raise RuntimeError(f"{out}: the saved requests do not share one system instruction and "
                           f"generation config ({len(fingerprints)} found)")
    fingerprint = lines[0][2]
    saved = {key for key, _, _ in lines}
    missing = [n for n in state["mapping"] if n not in saved]
    wrong = sorted({key for key, digest, _ in lines if state["mapping"].get(key) != digest})
    if missing or wrong:
        raise RuntimeError(f"{out}: saved requests missing for {missing[:3]}, prompt digest differs "
                           f"for {wrong[:3]}")
    if state.get("request_fingerprint"):
        if state["request_fingerprint"] != fingerprint:
            raise RuntimeError(f"{out}: the state's fingerprint differs from the saved requests")
        print(f"{out}: fingerprint recorded and equal to the saved requests", flush=True)
    else:
        state["request_fingerprint"] = fingerprint
        write_json(out / STATE_FILE, state)
        print(f"{out}: recorded the fingerprint of the {len(saved)} saved request(s)", flush=True)
    if getattr(args, "rebuild", True):
        source = Path(state["dataset_root"])
        first = state["scenarios"][0]
        variant = state.get("prompt_variant", "full")
        cfg, _, _ = _build_lines(out, source, state["split"], state["condition"], [source / first],
                                 dict(state["mapping"]), prompt_variant=variant)
        if cfg.get("request_fingerprint") != fingerprint:
            raise RuntimeError(f"{out}: the current code builds another system instruction or "
                               f"generation config than the one submitted; the answers belong to "
                               f"the submitted text, not to this code")
        print(f"{out}: the current code rebuilds {first} exactly (prompt digest, system "
              f"instruction, generation config)", flush=True)


def _upload(key: str, raw: bytes) -> str:
    """Resumable upload to the File API; returns the file name."""
    start = urllib.request.Request(
        UPLOAD_URL, data=b"", method="POST",
        headers={"x-goog-api-key": key,
                 "X-Goog-Upload-Protocol": "resumable",
                 "X-Goog-Upload-Command": "start",
                 "X-Goog-Upload-Header-Content-Length": str(len(raw)),
                 "X-Goog-Upload-Header-Content-Type": "application/jsonl",
                 "Content-Type": "application/json"})
    with urllib.request.urlopen(start, timeout=api_backend.TIMEOUT_S) as resp:
        upload_url = resp.headers.get("X-Goog-Upload-URL")
    if not upload_url:
        raise api_backend.ApiError("File API did not return an upload URL")
    # One sendall carries the whole file, and its timeout covers the whole
    # transfer: allow a slow, shared uplink rather than the per-call default.
    body = _http(upload_url, raw,
                 {"Content-Length": str(len(raw)), "X-Goog-Upload-Offset": "0",
                  "X-Goog-Upload-Command": "upload, finalize"},
                 timeout=max(api_backend.TIMEOUT_S, len(raw) / UPLOAD_FLOOR_BYTES_PER_S))
    return json.loads(body)["file"]["name"]


def _create_job(key: str, model: str, display_name: str, file_name: str) -> str:
    job = json.loads(_http(
        BATCH_URL.format(model=model),
        json.dumps({"batch": {"display_name": display_name,
                              "input_config": {"file_name": file_name}}}).encode(),
        {"x-goog-api-key": key, "Content-Type": "application/json"}))
    return job["name"]


# ── status ────────────────────────────────────────────────────────────────

def _jobs(state: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Job list; a state written before chunking held one `batch_name`."""
    if state.get("jobs"):
        return state["jobs"]
    return [dict(batch_name=state["batch_name"], input_file=state.get("input_file"), chunk=0)]


def _job(name: str) -> Dict[str, Any]:
    return json.loads(_http(f"{BASE}/v1beta/{name}", None,
                            {"x-goog-api-key": _gemini_key()}, method="GET"))


def _job_state(job: Dict[str, Any]) -> str:
    """The live API answers BATCH_STATE_*, the docs say JOB_STATE_*.

    Compare on the suffix so either spelling works and a future rename does
    not silently reject a finished job.
    """
    return ((job.get("metadata") or {}).get("state")
            or job.get("state") or "UNKNOWN")


ENDED = ("_SUCCEEDED", "_FAILED", "_CANCELLED", "_EXPIRED")


def status(args) -> str:
    """Print every job. RUNNING while any job has not ended, SUCCEEDED when all
    succeeded, else the first failed state (retry resends what has no answer)."""
    state = json.loads((args.out_dir.resolve() / STATE_FILE).read_text())
    if state.get("dry_run"):
        print("DRY RUN state: nothing was submitted."); return "DRY_RUN"
    states = []
    for j in _jobs(state):
        js = _job_state(_job(j["batch_name"]))
        states.append(js)
        print(f"{j['batch_name']}: {js}", flush=True)
    if not all(x.endswith(ENDED) for x in states):
        return "RUNNING"
    if all(x.endswith("_SUCCEEDED") for x in states):
        return "SUCCEEDED"
    return next(x for x in states if not x.endswith("_SUCCEEDED"))


def _collect(out: Path, state: Dict[str, Any]):
    """Answers, failures and unfinished jobs over every job of a cell.

    A scenario's answer is the first successful line for it in job order (a
    retry only resends scenarios without one). A failure is an error line, a
    line missing from a finished job, or a failed, cancelled or expired job;
    it counts only while the scenario has no answer."""
    answers: Dict[str, Any] = {}
    failures: Dict[str, Any] = {}
    pending: List[str] = []
    for j in _jobs(state):
        job = _job(j["batch_name"])
        js = _job_state(job)
        keys = j.get("keys") or list(state["mapping"])
        if js.endswith("_SUCCEEDED"):
            dest = ((job.get("response") or {}).get("responsesFile")
                    or ((job.get("response") or {}).get("dest") or {}).get("fileName"))
            if not dest:
                raise api_backend.ApiError(f"no responses file in job: {str(job)[:300]}")
            body = _http(f"{BASE}/download/v1beta/{dest}:download?alt=media", None,
                         {"x-goog-api-key": _gemini_key()}, method="GET")
            (out / f"batch_output_{j.get('chunk', 0)}.jsonl").write_bytes(body)
            seen = set()
            for line in body.decode("utf-8").splitlines():
                if not line.strip():
                    continue
                item = json.loads(line)
                name = item.get("key")
                seen.add(name)
                if item.get("error"):
                    failures.setdefault(name, item["error"])
                else:
                    answers.setdefault(name, item.get("response") or item)
            for name in keys:
                if name not in seen:
                    failures.setdefault(name, f"missing from the output of {j['batch_name']}")
        elif js.endswith(ENDED):
            for name in keys:
                failures.setdefault(name, f"{j['batch_name']} {js}")
        else:
            pending.append(j["batch_name"])
    return answers, {k: v for k, v in failures.items() if k not in answers}, pending


# ── fetch ─────────────────────────────────────────────────────────────────

def fetch(args) -> Path:
    out = args.out_dir.resolve()
    state = json.loads((out / STATE_FILE).read_text())
    if state.get("dry_run"):
        raise ValueError("this directory holds a dry run; submit for real first")
    answers, failures, pending = _collect(out, state)
    if pending:
        raise RuntimeError(f"{pending[0]} has not finished yet ({len(pending)} job(s) running)")
    missing = [n for n in state["scenarios"] if n not in answers]
    if missing:
        write_json(out / "batch_errors.json", {n: failures.get(n, "no answer") for n in missing})
        raise RuntimeError(f"{len(missing)} request(s) have no answer; see batch_errors.json "
                           f"and run retry")
    by_prompt = {state["mapping"][name]: resp for name, resp in answers.items()
                 if name in state["mapping"]}

    def replay(url, payload, headers, timeout):
        _check_fingerprint(state, payload, "fetch")
        digest = _prompt_sha(payload)
        if digest not in by_prompt:
            raise RuntimeError(
                "prompt changed between submit and fetch: no batch response for "
                "this request. Re-submit rather than pairing a new prompt with "
                "an old answer.")
        return by_prompt[digest]

    source = Path(state["dataset_root"])
    subset = _link_inputs(out, state["split"], [source / n for n in state["scenarios"]])
    rows = dataset_inventory(subset, len(state["scenarios"]), state["split"])
    cfg = _panel_config()
    variant = state.get("prompt_variant", "full")
    campaign = {"campaign_dir": str(out), "splits": {state["split"]: {"path": str(subset)}}}
    frozen = {"purpose": "Gemini batch cell, not campaign admission",
              "batch_names": [j["batch_name"] for j in _jobs(state)], "model": cfg,
              "prompt_version": prompt_version(variant), "prompt_variant": variant,
              "datasets": {state["split"]: rows}}
    write_json(out / "frozen.json", frozen)
    cell = dict(split=state["split"], backend="gemini", condition=state["condition"],
                configuration=cfg["name"],
                out_dir=str(out / "cells" / state["split"] /
                            f"{cfg['name']}_{state['condition']}"))
    # The cell is regenerated in full from the batch answers, so an identity
    # left by an earlier fetch of a part of it (the probe) is replaced, not
    # compared: the answers are matched by prompt digest, never relabelled.
    identity = Path(cell["out_dir"]) / "campaign_identity.json"
    if identity.exists():
        identity.unlink()
    write_campaign_identity(campaign, cell)
    api_backend.USAGE.clear()
    stats = _run_with_stub(replay, subset, cell["out_dir"], state["condition"], cfg, variant)
    write_json(Path(cell["out_dir"]) / "run_stats.json", stats)
    write_json(out / "usage.json", dict(calls=len(api_backend.USAGE),
                                        records=api_backend.USAGE))
    score_cell(campaign, cell, frozen)
    data = collect_cell(cell, rows, split_path=str(subset),
                        frozen_sha256=sha(out / "frozen.json"))
    write_json(Path(cell["out_dir"]) / "collected.json", data)
    records = [dict(split=state["split"], configuration=cfg["name"],
                    condition=state["condition"], comparison="scores", metric=metric,
                    **mean_stat(list(values.values())))
               for metric, values in data["metrics"].items()]
    export_tables(out, records)
    tok_in = sum(u.get("input_tokens", 0) for u in api_backend.USAGE)
    tok_out = sum(u.get("output_tokens", 0) for u in api_backend.USAGE)
    # Gemini reports thinking OUTSIDE candidatesTokenCount
    # (totalTokenCount = prompt + candidates + thoughts) and bills it at the
    # output rate. Leaving it out understated the cost by about a third.
    tok_think = sum(u.get("thinking_tokens") or 0 for u in api_backend.USAGE)
    billed_out = tok_out + tok_think
    n = max(len(api_backend.USAGE), 1)
    print(f"DONE: {out}\n  {n} response(s); {tok_in} input + {tok_out} response"
          f" + {tok_think} thinking tokens\n"
          f"  per scenario: {tok_in/n:.0f} in / {billed_out/n:.0f} billed out"
          f" ({tok_think/n:.0f} of it thinking)\n"
          f"  batch rate gemini-2.5-pro: $0.625/M in, $5.00/M out -> "
          f"${(tok_in*0.625 + billed_out*5.0)/1e6:.4f} for this run", flush=True)
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="command", required=True)
    s = sub.add_parser("submit")
    s.add_argument("--dataset-root", type=Path, required=True)
    s.add_argument("--out-dir", type=Path, required=True)
    s.add_argument("--split", choices=("carla", "nuscenes"), required=True)
    s.add_argument("--condition", choices=CONDITIONS, default="video")
    s.add_argument("--scenario", default="scenario_1")
    s.add_argument("--scenarios", type=int, help="N scenarios instead of one")
    s.add_argument("--offset", type=int, default=0,
                   help="Skip the first N scenarios; with --scenarios this "
                        "submits one chunk of a split that exceeds 2GB")
    s.add_argument("--display-name")
    s.add_argument("--prompt", dest="prompt_variant", choices=PROMPT_VARIANTS, default="full",
                   help="Prompt variant: full (arm A) or recipe_withheld (arm B); one per cell")
    s.add_argument("--dry-run", action="store_true",
                   help="Build batch_input.jsonl and stop; costs nothing")
    for name in ("status", "fetch", "retry", "verify"):
        q = sub.add_parser(name)
        q.add_argument("--out-dir", type=Path, required=True)
        if name == "verify":
            q.add_argument("--no-rebuild", dest="rebuild", action="store_false",
                           help="Check the saved requests only; do not rebuild one through the current code")
    args = p.parse_args()
    {"submit": submit, "status": status, "fetch": fetch, "retry": retry,
     "verify": verify}[args.command](args)


if __name__ == "__main__":
    main()
