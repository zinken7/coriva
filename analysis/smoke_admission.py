"""Step 4 runtime smoke evidence and fail-closed model-panel admission.

The smoke run uses one protocol-1.0 scenario, which may come from a benchmark
split. Each inference configuration must execute all three input
conditions. The gate checks runtime identity, ordered inputs, generation
metadata, response artifacts and parser completion; prediction correctness and
schema compliance are recorded but deliberately do not decide admission.
"""

from __future__ import annotations

import argparse
import copy
import datetime as dt
import glob
import hashlib
import json
import os
import re
import subprocess
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple

from analysis.model_panel import (MAX_OUTPUT_TOKENS, TRANSFORMERS_RUNTIME_REVISION,
                                  load_model_panel, resolve_configurations,
                                  validate_model_panel)
from simulation.protocol import INPUT_CONDITIONS


SMOKE_ADMISSION_VERSION = "coriva-smoke-admission-1.0"
SMOKE_EVIDENCE_VERSION = "coriva-smoke-evidence-1.0"
_SHA40 = re.compile(r"^[0-9a-f]{40}$")
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_REQUIRED_INPUT_ASSETS = (
    "ground_truth.json", "input_manifest.json", "observed_state.json",
    "observed_history.json", "query.json", "ontology.json",
)
# Every other file the prompt builder reads (`run_model.build_scenario_prompt`
# reads the CARLA scene configuration for its declared intervention
# magnitudes). Hashed whenever present so that a byte that changes the prompt
# also changes the frozen input identity. Kept after the required assets and
# before the frames.
_OPTIONAL_PROMPT_SOURCE_ASSETS = ("scenario_config.json",)


def _utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat()


def _sha256_file(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_json(path: str) -> Dict[str, Any]:
    with open(path, encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"expected a JSON object: {path}")
    return value


def _write_json(path: str, value: Dict[str, Any]) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    temporary = path + ".tmp"
    with open(temporary, "w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, sort_keys=True)
        handle.write("\n")
    os.replace(temporary, path)


def repository_state(repo_dir: Optional[str] = None) -> Dict[str, Any]:
    """Return the exact git commit and refuse hidden code changes at runtime."""
    cwd = repo_dir or os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    def git(*args: str) -> str:
        completed = subprocess.run(
            ["git", *args], cwd=cwd, check=True, text=True,
            stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        return completed.stdout.strip()

    repo_root = git("rev-parse", "--show-toplevel")
    relative_coding = os.path.relpath(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))), repo_root)
    required_runtime_files = [
        os.path.join(relative_coding, path).replace(os.sep, "/")
        for path in (
            "analysis/smoke_admission.py", "analysis/model_panel.py",
            "analysis/model_panel.yaml", "analysis/run_model.py",
            "analysis/run_matrix.py", "analysis/backends/local_hf.py",
            "analysis/backends/api.py", "analysis/backends/result.py",
            "simulation/build_pilot.py", "simulation/protocol.py",
            "ctms/ctms.py",
        )
    ]
    commit = git("rev-parse", "HEAD")
    dirty_lines = [line for line in
                   git("status", "--porcelain", "--untracked-files=no").splitlines()
                   if line.strip()]
    # Required paths are relative to the repository root, even when Git is
    # invoked from Coding/ (the default runtime working directory).
    tracked = set(git("ls-files", "--full-name").splitlines())
    missing_tracked = [path for path in required_runtime_files if path not in tracked]
    return {
        "code_commit": commit,
        "worktree_clean": not dirty_lines and not missing_tracked,
        "tracked_dirty_entry_count": len(dirty_lines),
        "tracked_dirty_entries_preview": dirty_lines[:20],
        "untracked_required_runtime_files": missing_tracked,
    }


def input_file_fingerprint(records: Sequence[Dict[str, str]]) -> str:
    """Validate and hash one ordered, self-contained smoke input inventory."""
    if not isinstance(records, (list, tuple)):
        raise ValueError("smoke input_files must be an ordered list")
    paths = [record.get("path") if isinstance(record, dict) else None
             for record in records]
    if tuple(paths[:len(_REQUIRED_INPUT_ASSETS)]) != _REQUIRED_INPUT_ASSETS:
        raise ValueError("smoke input_files has missing or reordered JSON assets")
    rest = paths[len(_REQUIRED_INPUT_ASSETS):]
    optional = [p for p in rest if p in _OPTIONAL_PROMPT_SOURCE_ASSETS]
    if optional != [p for p in _OPTIONAL_PROMPT_SOURCE_ASSETS if p in optional] or \
            rest[:len(optional)] != optional:
        raise ValueError("smoke input_files has reordered prompt-source assets")
    frame_paths = rest[len(optional):]
    if len(frame_paths) != 8 or frame_paths != sorted(frame_paths) or not all(
            isinstance(path, str) and path.startswith("frames/") and
            path.lower().endswith((".jpg", ".png"))
            for path in frame_paths):
        raise ValueError("smoke input_files requires eight ordered frame assets")
    if len(paths) != len(set(paths)):
        raise ValueError("smoke input_files contains duplicate paths")

    aggregate = hashlib.sha256()
    for record, path in zip(records, paths):
        file_hash = record.get("sha256")
        if not isinstance(path, str) or not path or os.path.isabs(path) or \
                ".." in path.split("/"):
            raise ValueError(f"invalid smoke input path: {path!r}")
        if not isinstance(file_hash, str) or not _SHA256.fullmatch(file_hash):
            raise ValueError(f"invalid smoke input hash for {path!r}")
        aggregate.update(path.encode("utf-8"))
        aggregate.update(b"\0")
        aggregate.update(file_hash.encode("ascii"))
        aggregate.update(b"\n")
    return aggregate.hexdigest()


def scenario_input_fingerprint(scenario_dir: str) -> Tuple[str, List[Dict[str, str]]]:
    """Hash the complete smoke scenario inputs and private prompt source."""
    paths: List[str] = []
    for name in _REQUIRED_INPUT_ASSETS:
        path = os.path.join(scenario_dir, name)
        if not os.path.isfile(path):
            raise ValueError(f"smoke scenario is missing {name}: {scenario_dir}")
        paths.append(path)
    for name in _OPTIONAL_PROMPT_SOURCE_ASSETS:
        path = os.path.join(scenario_dir, name)
        if os.path.isfile(path):
            paths.append(path)
    frames = sorted(glob.glob(os.path.join(scenario_dir, "frames", "*.jpg")) +
                    glob.glob(os.path.join(scenario_dir, "frames", "*.png")))
    if len(frames) != 8:
        raise ValueError(
            f"smoke scenario requires exactly eight model-facing frames, found {len(frames)}")
    paths.extend(frames)
    records = []
    for path in paths:
        relative = os.path.relpath(path, scenario_dir).replace(os.sep, "/")
        file_hash = _sha256_file(path)
        records.append({"path": relative, "sha256": file_hash})
    return input_file_fingerprint(records), records


def _resolved_config(panel: Dict[str, Any], name: str) -> Dict[str, Any]:
    config = resolve_configurations(panel, [name])[0]
    config.update({
        "model_panel_version": panel["panel_version"],
        "protocol_version": panel["protocol_version"],
        "scorer_version": panel["scorer_version"],
    })
    return config


def _cell_identity(panel: Dict[str, Any], config: Dict[str, Any],
                   condition: str, scenario_name: str,
                   input_fingerprint: str,
                   repository: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "evidence_version": SMOKE_EVIDENCE_VERSION,
        "panel_version": panel["panel_version"],
        "protocol_version": panel["protocol_version"],
        "scorer_version": panel["scorer_version"],
        "transformers_revision": panel["runtime_environment"][
            "transformers_revision"],
        "weight_dtype": panel["runtime_environment"]["weight_dtype"],
        "configuration_name": config["name"],
        "checkpoint": config["checkpoint"],
        "model_id": config["model_id"],
        "model_revision": config.get("revision"),
        "backend": config["backend"],
        "adapter": config["adapter"],
        "inference_mode": config["inference_mode"],
        "enable_thinking": config.get("enable_thinking"),
        "condition": condition,
        "max_output_tokens": config["max_output_tokens"],
        "scenario_name": scenario_name,
        "input_fingerprint": input_fingerprint,
        "code_commit": repository["code_commit"],
        "worktree_clean": repository["worktree_clean"],
    }


def _expected_frame_count(condition: str) -> int:
    return 8 if condition in {"video", "video_history"} else 0


def inspect_cell_output(panel: Dict[str, Any], config: Dict[str, Any],
                        condition: str, scenario_name: str, out_dir: str,
                        input_fingerprint: str,
                        repository: Dict[str, Any],
                        parse_failures: Sequence[Dict[str, Any]],
                        input_files: Sequence[Dict[str, str]]) -> Dict[str, Any]:
    """Turn one real inference result into self-contained mechanical evidence."""
    identity = _cell_identity(
        panel, config, condition, scenario_name, input_fingerprint, repository)
    scenario_out = os.path.join(out_dir, scenario_name)
    artifact_names = (
        "raw_response.txt", "final_response.txt", "generation_metadata.json",
        "run_manifest.json", "l3_pre.json", "l4_pre.json",
    )
    artifact_paths = {name: os.path.join(scenario_out, name)
                      for name in artifact_names}
    artifacts_exist = all(os.path.isfile(path)
                          for path in artifact_paths.values())
    manifest: Dict[str, Any] = {}
    metadata: Dict[str, Any] = {}
    raw_text = ""
    final_text = ""
    if artifacts_exist:
        manifest = _read_json(artifact_paths["run_manifest.json"])
        metadata = _read_json(artifact_paths["generation_metadata.json"])
        with open(artifact_paths["raw_response.txt"], encoding="utf-8") as handle:
            raw_text = handle.read()
        with open(artifact_paths["final_response.txt"], encoding="utf-8") as handle:
            final_text = handle.read()

    expected_frames = _expected_frame_count(condition)
    frame_inputs = [item for item in input_files
                    if str(item.get("path", "")).startswith("frames/")]
    expected_frame_files = ([os.path.basename(item["path"])
                             for item in frame_inputs]
                            if expected_frames else [])
    expected_frame_hashes = ([item["sha256"] for item in frame_inputs]
                             if expected_frames else [])
    raw_hash = hashlib.sha256(raw_text.encode("utf-8")).hexdigest()
    final_hash = hashlib.sha256(final_text.encode("utf-8")).hexdigest()
    output_tokens = metadata.get("output_tokens")
    latency = metadata.get("latency_seconds")
    checks: Dict[str, bool] = {
        "worktree_clean": repository.get("worktree_clean") is True,
        "code_commit_is_sha": bool(_SHA40.fullmatch(
            str(repository.get("code_commit", "")))),
        "required_artifacts_exist": artifacts_exist,
        "protocol_version_matches": manifest.get("protocol_version") ==
        panel["protocol_version"],
        "scorer_version_matches": manifest.get("scorer_version") ==
        panel["scorer_version"],
        "model_panel_version_matches": manifest.get("model_panel_version") ==
        panel["panel_version"],
        "primary_track_matches": manifest.get("scorer_input_track") == "primary",
        "execution_purpose_is_smoke": manifest.get("execution_purpose") ==
        "smoke_admission",
        "configuration_matches": manifest.get("configuration_name") == config["name"],
        "model_id_matches": manifest.get("model_id") == config["model_id"],
        "model_revision_matches": manifest.get("model_revision") == config.get("revision"),
        "backend_matches": manifest.get("backend") == config["backend"],
        "adapter_matches": manifest.get("adapter") == config["adapter"],
        "inference_mode_matches": manifest.get("inference_mode") ==
        config["inference_mode"],
        "thinking_switch_matches": manifest.get("enable_thinking") ==
        config.get("enable_thinking"),
        "condition_matches": manifest.get("condition") == condition,
        "cap_matches": manifest.get("max_output_tokens") ==
        config["max_output_tokens"] == MAX_OUTPUT_TOKENS,
        "greedy_decoding": manifest.get("decoding") == "greedy" and
        manifest.get("temperature") == 0.0 and
        manifest.get("generation_seed") is None and manifest.get("seed") == 0,
        "frame_count_matches": manifest.get("frame_count") == expected_frames,
        "ordered_frame_files_match_input": manifest.get("frame_files") ==
        expected_frame_files,
        "ordered_frame_hashes_match_input": manifest.get("frame_sha256") ==
        expected_frame_hashes,
        "backend_image_count_matches": metadata.get("ordered_image_count") ==
        expected_frames,
        "raw_response_nonempty": bool(raw_text),
        "final_response_nonempty": bool(final_text),
        "raw_response_hash_matches": manifest.get("raw_response_sha256") == raw_hash,
        "final_response_hash_matches": manifest.get("final_response_sha256") == final_hash,
        "prompt_hash_recorded": bool(_SHA256.fullmatch(
            str(manifest.get("prompt_sha256", "")))),
        "generation_metadata_matches_manifest": manifest.get(
            "generation_metadata") == metadata,
        "thinking_not_in_final_prefix": not final_text.lstrip().startswith("<think>"),
        "output_tokens_recorded": isinstance(output_tokens, int) and output_tokens > 0,
        "latency_recorded": isinstance(latency, (int, float)) and latency > 0,
        "metadata_adapter_matches": metadata.get("adapter") == config["adapter"],
        "metadata_cap_matches": metadata.get("max_output_tokens") == MAX_OUTPUT_TOKENS,
        "parser_completed": os.path.isfile(artifact_paths["l3_pre.json"]) and
        os.path.isfile(artifact_paths["l4_pre.json"]),
    }

    if config["backend"] == "local":
        peak_memory = metadata.get("peak_cuda_memory_bytes")
        checks.update({
            "model_commit_matches": metadata.get("model_commit_hash") ==
            config.get("revision"),
            "requested_revision_matches": metadata.get("requested_revision") ==
            config.get("revision"),
            "transformers_commit_matches": metadata.get(
                "transformers_commit_hash") == TRANSFORMERS_RUNTIME_REVISION,
            "cuda_memory_recorded": isinstance(peak_memory, int) and peak_memory > 0,
            "cuda_device_recorded": bool(metadata.get("cuda_device_name")),
            "torch_version_recorded": bool(metadata.get("torch_version")),
            "transformers_version_recorded": bool(
                metadata.get("transformers_version")),
            "bf16_dtype_confirmed": "bfloat16" in str(
                metadata.get("model_dtype", "")).lower(),
        })
    elif config["backend"] == "gemini":
        checks.update({
            "provider_model_version_recorded": bool(metadata.get("model_version")),
            "api_version_matches": metadata.get("api_version") ==
            config.get("api_version"),
            "api_endpoint_matches": metadata.get("api_endpoint") ==
            config.get("api_endpoint"),
            "access_date_recorded": bool(metadata.get("access_date")),
            "response_id_recorded": bool(metadata.get("response_id")),
        })
    else:
        checks["supported_panel_backend"] = False

    observations = {
        "scenario_output_dir": os.path.abspath(scenario_out),
        "prompt_sha256": manifest.get("prompt_sha256"),
        "frame_files": manifest.get("frame_files", []),
        "frame_sha256": manifest.get("frame_sha256", []),
        "raw_response_sha256": raw_hash,
        "final_response_sha256": final_hash,
        "final_response_nonempty": bool(final_text),
        "thinking_trace_present": manifest.get("thinking_trace_present"),
        "parse_failure_count": len(parse_failures),
        "parse_failures": list(parse_failures),
        "generation_metadata": metadata,
    }
    pair_path = os.path.join(out_dir, 'l4_pair.json')
    if os.path.isfile(pair_path):
        pair_report = _read_json(pair_path)
        observations['l4_pair'] = {
            'l4_pair_version': pair_report.get('l4_pair_version'),
            'coverage': pair_report.get('coverage'), 'axes': pair_report.get('axes'),
            'per_scenario': pair_report.get('per_scenario'),
            'sha256': _sha256_file(pair_path),
            'role': 'protocol-development observation; not a score-based admission gate',
        }
    failures = sorted(name for name, passed in checks.items() if not passed)
    return {
        **identity,
        "status": "passed" if not failures else "failed",
        "completed_at": _utc_now(),
        "checks": checks,
        "failed_checks": failures,
        "observations": observations,
    }


def _failed_cell_evidence(panel: Dict[str, Any], config: Dict[str, Any],
                          condition: str, scenario_name: str,
                          input_fingerprint: str,
                          repository: Dict[str, Any], exc: Exception) -> Dict[str, Any]:
    identity = _cell_identity(
        panel, config, condition, scenario_name, input_fingerprint, repository)
    return {
        **identity,
        "status": "failed",
        "completed_at": _utc_now(),
        "checks": {"runtime_completed": False},
        "failed_checks": ["runtime_completed"],
        "observations": {
            "error_type": type(exc).__name__,
            "error": str(exc),
        },
    }


def run_configuration_smoke(
        panel_path: str, gt_dir: str, out_root: str, scenario_name: str,
        configuration_name: str, *, repo_dir: Optional[str] = None,
        repository: Optional[Dict[str, Any]] = None,
        runner: Optional[Callable[..., Dict[str, Any]]] = None
        ) -> List[Dict[str, Any]]:
    """Execute one configuration over all three conditions and write evidence."""
    panel = load_model_panel(panel_path)
    config = _resolved_config(panel, configuration_name)
    repository = dict(repository or repository_state(repo_dir))
    if repository.get("worktree_clean") is not True:
        preview = repository.get("tracked_dirty_entries_preview", [])
        untracked = repository.get("untracked_required_runtime_files", [])
        raise RuntimeError(
            "smoke admission requires committed runtime files and no tracked "
            f"changes; dirty={preview}, untracked_runtime={untracked}")
    if not _SHA40.fullmatch(str(repository.get("code_commit", ""))):
        raise RuntimeError("smoke admission requires an exact 40-character git commit")
    scenario_dir = os.path.join(gt_dir, scenario_name)
    if not os.path.isdir(scenario_dir):
        raise ValueError(f"smoke scenario does not exist: {scenario_dir}")
    input_fingerprint, input_files = scenario_input_fingerprint(scenario_dir)
    from simulation.l4_control import validate_pair_reference
    validate_pair_reference(_read_json(os.path.join(scenario_dir, 'ground_truth.json')),
                            _read_json(os.path.join(scenario_dir, 'input_manifest.json')))
    if runner is None:
        from analysis.run_model import run_model as runner

    evidence_records = []
    for condition in INPUT_CONDITIONS:
        cell_dir = os.path.join(
            out_root, f"{configuration_name}_{condition}_seed0")
        evidence_path = os.path.join(cell_dir, "smoke_evidence.json")
        if os.path.isfile(evidence_path):
            existing = _read_json(evidence_path)
            expected = _cell_identity(
                panel, config, condition, scenario_name,
                input_fingerprint, repository)
            conflicts = {key: (existing.get(key), value)
                         for key, value in expected.items()
                         if existing.get(key) != value}
            if conflicts:
                raise ValueError(
                    f"existing smoke evidence conflicts with this run: {conflicts}")
            if existing.get("status") == "passed" and all(
                    existing.get("checks", {}).values()):
                evidence_records.append(existing)
                continue
        try:
            stats = runner(
                gt_dir=gt_dir, out_dir=cell_dir, backend=config["backend"],
                model_id=config["model_id"], num_frames=8, seed=0,
                condition=condition, track="primary", temperature=0.0,
                skip_existing=False, model_config=config,
                scenario_names=[scenario_name], provisional_smoke=True)
            from analysis.intervention_pair import run_l4_pair
            _write_json(os.path.join(cell_dir, 'l4_pair.json'), run_l4_pair(gt_dir, cell_dir))
            evidence = inspect_cell_output(
                panel, config, condition, scenario_name, cell_dir,
                input_fingerprint, repository, stats.get("failed", []),
                input_files)
            evidence["input_files"] = input_files
        except Exception as exc:  # preserve auditable failure evidence
            evidence = _failed_cell_evidence(
                panel, config, condition, scenario_name,
                input_fingerprint, repository, exc)
            evidence["input_files"] = input_files
        _write_json(evidence_path, evidence)
        evidence_records.append(evidence)
    return evidence_records


def _expected_cells(panel: Dict[str, Any]) -> Dict[Tuple[str, str], Dict[str, Any]]:
    configs = resolve_configurations(
        panel, [item["name"] for item in panel["configurations"]])
    return {(config["name"], condition): config
            for config in configs for condition in INPUT_CONDITIONS}


def build_admission(panel: Dict[str, Any], evidence_records: Iterable[Dict[str, Any]],
                    *, skip_gemini_smoke: bool = False
                    ) -> Dict[str, Any]:
    """Aggregate the panel's independent cell records into one admission artifact."""
    validate_model_panel(panel)
    records = list(evidence_records)
    expected = _expected_cells(panel)
    if skip_gemini_smoke:
        expected = {key: config for key, config in expected.items()
                    if config['backend'] != 'gemini'}
    by_cell: Dict[Tuple[str, str], Dict[str, Any]] = {}
    failures: List[str] = []
    for record in records:
        key = (record.get("configuration_name"), record.get("condition"))
        if key in by_cell:
            failures.append(f"duplicate smoke evidence for {key}")
        else:
            by_cell[key] = record
    missing = sorted(set(expected) - set(by_cell))
    unexpected = sorted(set(by_cell) - set(expected))
    if missing:
        failures.append(f"missing smoke cells: {missing}")
    if unexpected:
        failures.append(f"unexpected smoke cells: {unexpected}")

    for key, config in expected.items():
        record = by_cell.get(key)
        if record is None:
            continue
        identity = {
            "evidence_version": SMOKE_EVIDENCE_VERSION,
            "panel_version": panel["panel_version"],
            "protocol_version": panel["protocol_version"],
            "scorer_version": panel["scorer_version"],
            "transformers_revision": panel["runtime_environment"][
                "transformers_revision"],
            "weight_dtype": panel["runtime_environment"]["weight_dtype"],
            "configuration_name": config["name"],
            "checkpoint": config["checkpoint"],
            "model_id": config["model_id"],
            "model_revision": config.get("revision"),
            "backend": config["backend"],
            "adapter": config["adapter"],
            "inference_mode": config["inference_mode"],
            "enable_thinking": config.get("enable_thinking"),
            "condition": key[1],
            "max_output_tokens": config["max_output_tokens"],
        }
        mismatches = {field: (record.get(field), value)
                      for field, value in identity.items()
                      if record.get(field) != value}
        if mismatches:
            failures.append(f"identity mismatch for {key}: {mismatches}")
        if record.get("status") != "passed":
            failures.append(f"smoke cell failed: {key}")
        checks = record.get("checks")
        if not isinstance(checks, dict) or not checks or not all(
                value is True for value in checks.values()):
            failures.append(f"smoke checks failed: {key}")
        try:
            embedded_fingerprint = input_file_fingerprint(
                record.get("input_files"))
        except (TypeError, ValueError) as exc:
            failures.append(f"invalid embedded input files for {key}: {exc}")
        else:
            if embedded_fingerprint != record.get("input_fingerprint"):
                failures.append(f"embedded input fingerprint mismatch for {key}")

    code_commits = {record.get("code_commit") for record in records}
    scenario_names = {record.get("scenario_name") for record in records}
    input_fingerprints = {record.get("input_fingerprint") for record in records}
    for label, values in (
            ("code commits", code_commits),
            ("smoke scenarios", scenario_names),
            ("input fingerprints", input_fingerprints)):
        if len(values) != 1 or None in values:
            failures.append(f"mixed or missing {label}: {sorted(str(v) for v in values)}")
    if code_commits and not _SHA40.fullmatch(str(next(iter(code_commits)))):
        failures.append("smoke code commit is not an immutable SHA")
    if any(record.get("worktree_clean") is not True for record in records):
        failures.append("one or more smoke cells used a dirty worktree")

    gemini_versions = {
        ((record.get("observations") or {}).get("generation_metadata") or {}).get(
            "model_version")
        for record in records if record.get("backend") == "gemini"
    }
    if not skip_gemini_smoke and (len(gemini_versions) != 1 or None in gemini_versions):
        failures.append(
            f"mixed or missing Gemini provider modelVersion: {sorted(str(v) for v in gemini_versions)}")

    completed_values = [record.get("completed_at") for record in records
                        if isinstance(record.get("completed_at"), str)]
    admission = {
        "admission_version": SMOKE_ADMISSION_VERSION,
        "status": "passed" if not failures else "failed",
        "generated_at": _utc_now(),
        "smoke_completed_at": max(completed_values) if completed_values else None,
        "panel_version": panel["panel_version"],
        "protocol_version": panel["protocol_version"],
        "scorer_version": panel["scorer_version"],
        "transformers_revision": panel["runtime_environment"][
            "transformers_revision"],
        "weight_dtype": panel["runtime_environment"]["weight_dtype"],
        "code_commit": next(iter(code_commits)) if len(code_commits) == 1 else None,
        "worktree_clean": all(record.get("worktree_clean") is True
                              for record in records),
        "scenario_name": next(iter(scenario_names)) if len(scenario_names) == 1 else None,
        "input_fingerprint": (next(iter(input_fingerprints))
                              if len(input_fingerprints) == 1 else None),
        "gemini_model_version": (next(iter(gemini_versions))
                                 if len(gemini_versions) == 1 else None),
        "expected_cell_count": len(expected),
        "observed_cell_count": len(records),
        "conditions": list(INPUT_CONDITIONS),
        "failures": failures,
        "cells": sorted(records, key=lambda item: (
            str(item.get("configuration_name")), str(item.get("condition")))),
    }
    if skip_gemini_smoke:
        # This is an explicit execution policy, not fabricated smoke evidence.
        admission['gemini_smoke'] = {'status': 'not_run', 'reason': 'user_requested_skip'}
    return admission


def validate_admission(panel: Dict[str, Any], admission: Dict[str, Any]
                       ) -> Dict[str, Any]:
    """Rebuild the gate from embedded evidence; never trust a pass flag alone."""
    validate_model_panel(panel)
    if not isinstance(admission, dict):
        raise ValueError("smoke admission must be a JSON object")
    if admission.get("admission_version") != SMOKE_ADMISSION_VERSION:
        raise ValueError("smoke admission version mismatch")
    if admission.get("panel_version") != panel["panel_version"] or \
            admission.get("protocol_version") != panel["protocol_version"] or \
            admission.get("scorer_version") != panel["scorer_version"]:
        raise ValueError("smoke admission panel/protocol/scorer identity mismatch")
    if admission.get("transformers_revision") != TRANSFORMERS_RUNTIME_REVISION:
        raise ValueError("smoke admission Transformers revision mismatch")
    if admission.get("weight_dtype") != "bfloat16":
        raise ValueError("smoke admission dtype mismatch")
    gemini_smoke = admission.get('gemini_smoke')
    if gemini_smoke is not None and gemini_smoke != {
            'status': 'not_run', 'reason': 'user_requested_skip'}:
        raise ValueError('invalid Gemini smoke exception')
    rebuilt = build_admission(panel, admission.get("cells", []),
                              skip_gemini_smoke=gemini_smoke is not None)
    if rebuilt["status"] != "passed":
        raise ValueError(
            "smoke admission evidence does not pass: " +
            "; ".join(rebuilt["failures"]))
    for field in (
            "code_commit", "smoke_completed_at", "scenario_name", "input_fingerprint",
            "gemini_model_version", "expected_cell_count", "observed_cell_count",
            "worktree_clean", "conditions"):
        if admission.get(field) != rebuilt.get(field):
            raise ValueError(f"smoke admission summary mismatch: {field}")
    if admission.get("status") != "passed" or admission.get("failures") != []:
        raise ValueError("smoke admission is not marked as a clean pass")
    return admission


def admit_panel(panel: Dict[str, Any], admission_path: str,
                *, repo_dir: Optional[str] = None,
                repository: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Return a panel copy whose configurations are enabled by valid evidence."""
    admission = validate_admission(panel, _read_json(admission_path))
    current_repository = dict(repository or repository_state(repo_dir))
    if current_repository.get("worktree_clean") is not True:
        raise RuntimeError(
            "full inference requires committed runtime files and no tracked changes")
    if current_repository.get("code_commit") != admission["code_commit"]:
        raise RuntimeError(
            "full inference code commit differs from the smoke admission: "
            f"{current_repository.get('code_commit')} != {admission['code_commit']}")
    admitted = copy.deepcopy(panel)
    admission_hash = _sha256_file(admission_path)
    checkpoint_backends = {item['key']: item['backend'] for item in admitted['checkpoints']}
    for config in admitted["configurations"]:
        config["runtime_ready"] = True
        config["runtime_blocker"] = None
        config["smoke_admission_version"] = SMOKE_ADMISSION_VERSION
        config["smoke_admission_sha256"] = admission_hash
        config["smoke_code_commit"] = admission["code_commit"]
        config["smoke_completed_at"] = admission["smoke_completed_at"]
        config['smoke_status'] = ('not_run' if admission.get('gemini_smoke') and
                                  checkpoint_backends[config['checkpoint']] == 'gemini' else 'passed')
    validate_model_panel(admitted)
    return admitted


def _select_configuration(panel: Dict[str, Any], name: Optional[str],
                          backend: Optional[str], index: Optional[int]) -> str:
    configs = resolve_configurations(
        panel, [item["name"] for item in panel["configurations"]])
    if backend:
        configs = [item for item in configs if item["backend"] == backend]
    if name:
        matches = [item for item in configs if item["name"] == name]
        if len(matches) != 1:
            raise ValueError(
                f"configuration {name!r} is not available for backend {backend!r}")
        return matches[0]["name"]
    if index is None:
        raise ValueError("provide --configuration or --configuration-index")
    if index < 0 or index >= len(configs):
        raise ValueError(
            f"configuration index {index} outside 0..{len(configs) - 1}")
    return configs[index]["name"]


def _load_evidence(root: str) -> List[Dict[str, Any]]:
    paths = sorted(glob.glob(os.path.join(root, "**", "smoke_evidence.json"),
                             recursive=True))
    return [_read_json(path) for path in paths]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--panel", default="analysis/model_panel.yaml")
    subparsers = parser.add_subparsers(dest="command", required=True)

    list_parser = subparsers.add_parser("list", help="list smoke configurations")
    list_parser.add_argument("--backend", choices=["local", "gemini"], default=None)

    run_parser = subparsers.add_parser("run", help="run one configuration smoke")
    run_parser.add_argument("--gt-dir", required=True)
    run_parser.add_argument("--out-root", required=True)
    run_parser.add_argument("--scenario", required=True,
                            help="smoke scenario basename; benchmark scenarios are allowed")
    run_parser.add_argument("--backend", choices=["local", "gemini"], default=None)
    run_parser.add_argument("--configuration", default=None)
    run_parser.add_argument("--configuration-index", type=int, default=None)
    run_parser.add_argument("--repo-dir", default=None)

    finalize_parser = subparsers.add_parser(
        "finalize", help="combine all panel cell evidence files")
    finalize_parser.add_argument("--evidence-root", required=True)
    finalize_parser.add_argument("--out", required=True)

    validate_parser = subparsers.add_parser(
        "validate", help="validate a completed admission artifact")
    validate_parser.add_argument("--admission", required=True)

    args = parser.parse_args()
    panel = load_model_panel(args.panel)
    if args.command == "list":
        configs = resolve_configurations(
            panel, [item["name"] for item in panel["configurations"]])
        if args.backend:
            configs = [item for item in configs if item["backend"] == args.backend]
        for index, config in enumerate(configs):
            print(f"{index} {config['name']} {config['backend']} {config['model_id']}")
        return 0
    if args.command == "run":
        name = _select_configuration(
            panel, args.configuration, args.backend, args.configuration_index)
        records = run_configuration_smoke(
            args.panel, args.gt_dir, args.out_root, args.scenario, name,
            repo_dir=args.repo_dir)
        passed = sum(record["status"] == "passed" for record in records)
        print(f"{name}: {passed}/{len(records)} smoke cells passed -> {args.out_root}")
        return 0 if passed == len(records) else 2
    if args.command == "finalize":
        admission = build_admission(panel, _load_evidence(args.evidence_root))
        _write_json(args.out, admission)
        print(f"{admission['status']}: {admission['observed_cell_count']}/"
              f"{admission['expected_cell_count']} cells -> {args.out}")
        if admission["status"] != "passed":
            for failure in admission["failures"]:
                print(f"  - {failure}")
            return 2
        validate_admission(panel, admission)
        return 0
    if args.command == "validate":
        admission = validate_admission(panel, _read_json(args.admission))
        print(f"{admission['admission_version']}: {admission['observed_cell_count']}/{admission['expected_cell_count']} cells admitted, "
              f"commit={admission['code_commit']}")
        return 0
    raise AssertionError(args.command)


if __name__ == "__main__":
    raise SystemExit(main())
