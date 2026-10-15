"""Load and fail-closed validate the versioned CORIVA model panel.

The YAML file is the single source of truth for checkpoint identity, inference
configuration and the common input/generation contract. Adapter behaviour is
implemented separately; configurations remain runtime-blocked until the real
three-condition smoke tests are complete.
"""

from __future__ import annotations

import argparse
import copy
import os
import re
from typing import Any, Dict, Iterable, List

from ctms.ctms import SCORER_VERSION
from simulation.protocol import (HISTORY_TIMESTAMPS, INPUT_CONDITIONS,
                                 OBSERVED_FRAME_COUNT, PROTOCOL_VERSION)

MODEL_PANEL_VERSION = "coriva-model-panel-1.2"
EXPECTED_CHECKPOINT_COUNT = 5
EXPECTED_CONFIGURATION_COUNT = 5
# Thinking tokens share the cap. At 32,768 Gemini 2.5 Pro stopped at the cap on
# 5 of 24 pilot answers and Qwen3.5 thought 12k tokens before answering, so
# panel 1.2 raised it (2026-09-23) to let every answer end on its own.
MAX_OUTPUT_TOKENS = 65536
TRANSFORMERS_RUNTIME_REVISION = "116224175191c9e77411a71849262595cb80f936"
EXPECTED_CHECKPOINT_KEYS = {
    "qwen3-vl-8b-instruct", "cosmos-reason2-8b", "qwen3.5-9b",
    "molmo2-8b", "gemini-2.5-pro",
}
EXPECTED_CONFIGURATION_NAMES = {
    "qwen3-vl-8b-instruct", "cosmos-reason2-8b",
    "qwen3.5-9b-direct", "molmo2-8b",
    "gemini-2.5-pro",
}
EXPECTED_CHECKPOINT_IDENTITIES = {
    "qwen3-vl-8b-instruct": (
        "Qwen/Qwen3-VL-8B-Instruct",
        "0c351dd01ed87e9c1b53cbc748cba10e6187ff3b", "Apache-2.0"),
    "cosmos-reason2-8b": (
        "nvidia/Cosmos-Reason2-8B",
        "a9fae2cf89dc64db96b12860417f0eb403013bb9",
        "NVIDIA-Open-Model-License"),
    "qwen3.5-9b": (
        "Qwen/Qwen3.5-9B",
        "c202236235762e1c871ad0ccb60c8ee5ba337b9a", "Apache-2.0"),
    "molmo2-8b": (
        "allenai/Molmo2-8B",
        "e28fa28597e5ec5e0cca2201dd8ab33d48bc4a1b", "Apache-2.0"),
    "gemini-2.5-pro": ("gemini-2.5-pro", None, "proprietary-service"),
}
EXPECTED_CHECKPOINT_RUNTIMES = {
    "qwen3-vl-8b-instruct": ("huggingface", "local", "qwen3_vl"),
    "cosmos-reason2-8b": ("huggingface", "local", "cosmos_reason2"),
    "qwen3.5-9b": ("huggingface", "local", "qwen35"),
    "molmo2-8b": ("huggingface", "local", "molmo2"),
    "gemini-2.5-pro": ("google_api", "gemini", "gemini_api"),
}
_SHA40 = re.compile(r"^[0-9a-f]{40}$")
_SHA256 = re.compile(r"^[0-9a-f]{64}$")


def _unique(records: Iterable[Dict[str, Any]], field: str,
            label: str) -> Dict[str, Dict[str, Any]]:
    index: Dict[str, Dict[str, Any]] = {}
    for record in records:
        value = record.get(field)
        if not isinstance(value, str) or not value:
            raise ValueError(f"every {label} requires a non-empty {field}")
        if value in index:
            raise ValueError(f"duplicate {label} {field}: {value}")
        index[value] = record
    return index


def validate_model_panel(panel: Dict[str, Any]) -> Dict[str, Any]:
    """Validate the released panel without accepting silent defaults."""
    if not isinstance(panel, dict):
        raise ValueError("model panel must be a mapping")
    if panel.get("panel_version") != MODEL_PANEL_VERSION:
        raise ValueError("model panel version mismatch")
    if panel.get("protocol_version") != PROTOCOL_VERSION:
        raise ValueError("model panel protocol version mismatch")
    if panel.get("scorer_version") != SCORER_VERSION:
        raise ValueError("model panel scorer version mismatch")
    if not isinstance(panel.get("metadata_verified_on"), str):
        raise ValueError("metadata_verified_on must be an explicit ISO date string")
    runtime = panel.get("runtime_environment")
    if not isinstance(runtime, dict) or \
            runtime.get("transformers_repository") != \
            "https://github.com/huggingface/transformers.git" or \
            runtime.get("transformers_revision") != TRANSFORMERS_RUNTIME_REVISION or \
            runtime.get("weight_dtype") != "bfloat16":
        raise ValueError("Transformers runtime repository/revision/dtype mismatch")

    inputs = panel.get("input_contract")
    if not isinstance(inputs, dict):
        raise ValueError("input_contract must be a mapping")
    if inputs.get("observed_frame_count") != OBSERVED_FRAME_COUNT:
        raise ValueError("model panel must use exactly eight observed frames")
    timestamps = tuple(round(float(v), 3) for v in
                       inputs.get("observed_frame_timestamps_sec", []))
    if timestamps != HISTORY_TIMESTAMPS:
        raise ValueError("model panel frame timestamps must match the dataset protocol")
    if tuple(inputs.get("conditions", [])) != INPUT_CONDITIONS:
        raise ValueError("model panel conditions must match the dataset protocol in order")
    if inputs.get("ordered_image_inputs") is not True:
        raise ValueError("the eight visual inputs must be ordered images")
    if inputs.get("mp4_resampling_allowed") is not False:
        raise ValueError("MP4 resampling must be disabled")

    generation = panel.get("generation_contract")
    if not isinstance(generation, dict):
        raise ValueError("generation_contract must be a mapping")
    if generation.get("decoding") != "greedy" or generation.get("temperature") != 0.0:
        raise ValueError("the main protocol requires greedy decoding at temperature 0")
    if generation.get("run_ids") != [0]:
        raise ValueError("the main protocol requires exactly one stored run with ID 0")
    if generation.get("max_output_tokens") != MAX_OUTPUT_TOKENS:
        raise ValueError(f"max_output_tokens must be {MAX_OUTPUT_TOKENS}")
    if generation.get("thinking_trace_storage") != "separate_artifact":
        raise ValueError("thinking traces must be stored separately")
    if generation.get("scorer_input") != "final_json_only":
        raise ValueError("the scorer may read only the final JSON response")

    reporting = panel.get("reporting_contract")
    if not isinstance(reporting, dict) or \
            reporting.get("dataset_splits_reported_separately") != ["carla", "nuscenes"]:
        raise ValueError("CARLA and nuScenes must be reported separately")

    checkpoints = panel.get("checkpoints")
    configurations = panel.get("configurations")
    if not isinstance(checkpoints, list) or len(checkpoints) != EXPECTED_CHECKPOINT_COUNT:
        raise ValueError(f"model panel must declare exactly {EXPECTED_CHECKPOINT_COUNT} checkpoints")
    if not isinstance(configurations, list) or \
            len(configurations) != EXPECTED_CONFIGURATION_COUNT:
        raise ValueError(
            f"model panel must declare exactly {EXPECTED_CONFIGURATION_COUNT} configurations")

    checkpoint_by_key = _unique(checkpoints, "key", "checkpoint")
    _unique(checkpoints, "model_id", "checkpoint")
    if set(checkpoint_by_key) != EXPECTED_CHECKPOINT_KEYS:
        raise ValueError("checkpoint keys differ from the versioned panel")
    for checkpoint in checkpoints:
        for field in ("source", "license", "backend", "adapter"):
            if not isinstance(checkpoint.get(field), str) or not checkpoint[field]:
                raise ValueError(f"checkpoint {checkpoint['key']} requires {field}")
        source = checkpoint["source"]
        revision = checkpoint.get("revision")
        expected_id, expected_revision, expected_license = \
            EXPECTED_CHECKPOINT_IDENTITIES[checkpoint["key"]]
        if (checkpoint["model_id"], revision, checkpoint["license"]) != \
                (expected_id, expected_revision, expected_license):
            raise ValueError(
                f"checkpoint identity differs from {MODEL_PANEL_VERSION}: {checkpoint['key']}")
        if (source, checkpoint["backend"], checkpoint["adapter"]) != \
                EXPECTED_CHECKPOINT_RUNTIMES[checkpoint["key"]]:
            raise ValueError(
                f"checkpoint runtime route differs from {MODEL_PANEL_VERSION}: "
                f"{checkpoint['key']}")
        if source == "huggingface":
            if not isinstance(revision, str) or not _SHA40.fullmatch(revision):
                raise ValueError(
                    f"open checkpoint {checkpoint['key']} requires an immutable 40-char SHA")
        elif source == "google_api":
            if revision is not None:
                raise ValueError("Gemini must not pretend that its public alias has a weight SHA")
            if checkpoint.get("api_version") != "v1beta":
                raise ValueError("Gemini API version must be pinned")
            if checkpoint.get("api_endpoint") != \
                    "https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent":
                raise ValueError("Gemini API endpoint must be pinned")
            if checkpoint.get("model_version_capture") != "required" or \
                    checkpoint.get("access_date_capture") != "required":
                raise ValueError("Gemini requires runtime modelVersion and access-date capture")
        else:
            raise ValueError(f"unsupported checkpoint source: {source}")

    config_by_name = _unique(configurations, "name", "configuration")
    if set(config_by_name) != EXPECTED_CONFIGURATION_NAMES:
        raise ValueError("configuration names differ from the versioned panel")
    for config in configurations:
        checkpoint_key = config.get("checkpoint")
        if checkpoint_key not in checkpoint_by_key:
            raise ValueError(
                f"configuration {config['name']} references unknown checkpoint {checkpoint_key}")
        if not isinstance(config.get("inference_mode"), str) or not config["inference_mode"]:
            raise ValueError(f"configuration {config['name']} requires inference_mode")
        if not isinstance(config.get("role"), str) or not config["role"]:
            raise ValueError(f"configuration {config['name']} requires role")
        if config.get("adapter_implemented") is not True:
            raise ValueError(f"configuration {config['name']} adapter is not implemented")
        if not isinstance(config.get("runtime_ready"), bool):
            raise ValueError(f"configuration {config['name']} requires boolean runtime_ready")
        if not config["runtime_ready"] and not config.get("runtime_blocker"):
            raise ValueError(f"unready configuration {config['name']} requires runtime_blocker")
        if config["runtime_ready"]:
            if config.get("runtime_blocker") is not None:
                raise ValueError(
                    f"ready configuration {config['name']} cannot retain a runtime blocker")
            if config.get("smoke_admission_version") != \
                    "coriva-smoke-admission-1.0" or not _SHA256.fullmatch(
                        str(config.get("smoke_admission_sha256", ""))) or \
                    not _SHA40.fullmatch(str(config.get("smoke_code_commit", ""))) or \
                    not isinstance(config.get("smoke_completed_at"), str):
                raise ValueError(
                    f"ready configuration {config['name']} requires validated smoke metadata")
        if "max_output_tokens" in config:
            raise ValueError("per-configuration output-cap overrides are forbidden")

    ready_count = sum(config["runtime_ready"] for config in configurations)
    if ready_count not in (0, EXPECTED_CONFIGURATION_COUNT):
        raise ValueError("the Step 4 gate must admit all configurations atomically")

    direct = config_by_name.get("qwen3.5-9b-direct")
    if direct.get("checkpoint") != "qwen3.5-9b":
        raise ValueError("Qwen3.5 direct checkpoint must be qwen3.5-9b")
    if direct.get("inference_mode") != "direct" or direct.get("enable_thinking") is not False:
        raise ValueError("Qwen3.5 direct must explicitly disable thinking")
    if checkpoint_by_key["cosmos-reason2-8b"].get("base_checkpoint") != \
            "qwen3-vl-8b-instruct":
        raise ValueError("Cosmos must declare the matched Qwen3-VL base checkpoint")
    return panel


def load_model_panel(path: str) -> Dict[str, Any]:
    """Read YAML and return a validated panel."""
    import yaml

    with open(path, encoding="utf-8") as handle:
        panel = yaml.safe_load(handle)
    return validate_model_panel(panel)


def resolve_configurations(panel: Dict[str, Any], names: List[str]) -> List[Dict[str, Any]]:
    """Resolve configuration names to runner records with pinned metadata."""
    validate_model_panel(panel)
    if not isinstance(names, list) or not names:
        raise ValueError("matrix configurations must be a non-empty list")
    if len(names) != len(set(names)):
        raise ValueError("matrix configuration names must be unique")

    checkpoints = {item["key"]: item for item in panel["checkpoints"]}
    configurations = {item["name"]: item for item in panel["configurations"]}
    generation = panel["generation_contract"]
    resolved: List[Dict[str, Any]] = []
    for name in names:
        if name not in configurations:
            raise ValueError(f"unknown model-panel configuration: {name}")
        config = copy.deepcopy(configurations[name])
        checkpoint = checkpoints[config["checkpoint"]]
        resolved.append({
            "name": config["name"],
            "checkpoint": checkpoint["key"],
            "model_id": checkpoint["model_id"],
            "revision": checkpoint.get("revision"),
            "backend": checkpoint["backend"],
            "adapter": checkpoint["adapter"],
            "inference_mode": config["inference_mode"],
            "enable_thinking": config.get("enable_thinking"),
            "role": config["role"],
            "adapter_implemented": config["adapter_implemented"],
            "runtime_ready": config["runtime_ready"],
            "runtime_blocker": config.get("runtime_blocker"),
            "max_output_tokens": generation["max_output_tokens"],
            "model_version_capture": checkpoint.get("model_version_capture"),
            "access_date_capture": checkpoint.get("access_date_capture"),
            "api_version": checkpoint.get("api_version"),
            "api_endpoint": checkpoint.get("api_endpoint"),
            "thinking_trace_storage": generation["thinking_trace_storage"],
            "scorer_input": generation["scorer_input"],
            "smoke_admission_version": config.get("smoke_admission_version"),
            "smoke_admission_sha256": config.get("smoke_admission_sha256"),
            "smoke_code_commit": config.get("smoke_code_commit"),
            "smoke_completed_at": config.get("smoke_completed_at"),
            "smoke_status": config.get("smoke_status"),
        })
    return resolved


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--manifest", default=os.path.join("analysis", "model_panel.yaml"))
    parser.add_argument("--smoke-admission", default=None,
                        help="validated Step 4 admission artifact")
    args = parser.parse_args()
    panel = load_model_panel(args.manifest)
    if args.smoke_admission:
        from analysis.smoke_admission import admit_panel
        panel = admit_panel(panel, args.smoke_admission)
    ready = sum(bool(item["runtime_ready"]) for item in panel["configurations"])
    print(f"{panel['panel_version']}: {len(panel['checkpoints'])} checkpoints, "
          f"{len(panel['configurations'])} configurations, {ready} runtime-ready")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
