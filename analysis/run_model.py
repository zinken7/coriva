"""CORIVA model runner — drive a VLM over a directory of scenarios and write
per-scenario predictions (l3_pre.json / l4_pre.json) that scripts/assemble_dataset.py
consumes, then run_benchmark.py scores.

Backends:
  - mock   : constant minimal valid prediction (plumbing / smoke tests).
  - oracle : echoes the GT branches/counterfactuals (test-only upper bound —
             the full loop should reach CTMS ~1.0 through this).
  - local  : pinned Qwen3-VL, Cosmos, Qwen3.5 and Molmo2 GPU adapters.

Protocol video runs pass the exactly eight manifest-declared frames from
<scenario_dir>/frames/*.jpg|png to the backend; otherwise the backend runs
text-only. Datasets without a manifest use deterministic subsampling.
The mock/oracle backends ignore frames, so no dataset is needed to test the loop.

CLI:
    python -m analysis.run_model --gt-dir mini_dataset --out-dir results/qwen3vl \\
        --configuration qwen3-vl-8b-instruct --num-frames 8
"""

import argparse
import glob
import hashlib
import json
import os
import sys
import time
from typing import Any, Callable, Dict, List, Optional, Set, Union

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from analysis.model_panel import MAX_OUTPUT_TOKENS
from simulation.build_pilot import (build_prompt_parts, join_prompt_parts, prompt_version,
                                    validate_prompt_variant)
from simulation.protocol import (INPUT_CONDITIONS, PRIMARY_TRACK, PROTOCOL_VERSION,
                                 validate_condition, validate_protocol_manifest)
from analysis.frames import select_frames
from analysis.backends.result import (GenerationResult,
                                      normalize_generation_result)
from analysis.parse_prediction import PARSER_VERSION, parse_model_output
from ctms.ctms import SCORER_VERSION

# name -> callable(prompt, frame_paths, model_id, opts) -> result/raw text.
# Panel opts also carry pinned adapter/revision/mode metadata. A run identifier
# is not forwarded as a sampling seed under greedy decoding.
BACKENDS: Dict[
    str,
    Callable[[str, List[str], str, Dict[str, Any]], Union[str, GenerationResult]],
] = {}
# Backends that send the fixed instructions as a separate system message
# (`opts["system_prompt"]`, prompt = the scene). Every other backend gets
# the two parts joined into one prompt.
SYSTEM_PROMPT_BACKENDS: Set[str] = set()


def backend(name: str, system_prompt: bool = False):
    def deco(fn):
        BACKENDS[name] = fn
        if system_prompt:
            SYSTEM_PROMPT_BACKENDS.add(name)
        return fn
    return deco


@backend("mock")
def _mock_backend(prompt: str, frame_paths: List[str], model_id: str,
                  opts: Dict[str, Any]) -> str:
    """Constant, structurally-valid prediction — for wiring/smoke tests only."""
    return json.dumps({
        "predicted_branches": [{
            "branch_id": "B1", "is_factual": True,
            "intervention": {"intervention_type": "do_ego_no_action",
                             "target_actor_id": "ego", "trigger_depth": 1},
            "chain": [{"timestamp": 0.5, "event_type": "safe_stop", "terminal": True,
                       "action_type": "ego_no_action",
                       "actors_update": [{"id": "ego", "entity_type": "ego_vehicle",
                                          "role_type": "ego_agent", "x": 0.0, "y": 0.0}]}],
        }],
        "predicted_counterfactuals": [],
    })


@backend("local")
def _local_backend(prompt: str, frame_paths: List[str], model_id: str,
                   opts: Dict[str, Any]) -> GenerationResult:
    from analysis.backends.local_hf import generate_result, DEFAULT_MODEL
    return generate_result(
        prompt, frame_paths, model_id or DEFAULT_MODEL,
        adapter=opts.get("adapter", "qwen3_vl"),
        revision=opts.get("revision"),
        inference_mode=opts.get("inference_mode"),
        enable_thinking=opts.get("enable_thinking"),
        max_new_tokens=opts.get("max_output_tokens", MAX_OUTPUT_TOKENS),
        cache_implementation=opts.get("cache_implementation"),
        seed=opts.get("seed"), temperature=opts.get("temperature"))


@backend("gemini", system_prompt=True)
def _gemini_backend(prompt: str, frame_paths: List[str], model_id: str,
                    opts: Dict[str, Any]) -> GenerationResult:
    from analysis.backends.api import gemini_generate_result
    return gemini_generate_result(
        prompt, frame_paths, model_id, system_prompt=opts.get("system_prompt"),
        seed=opts.get("seed"), temperature=opts.get("temperature"),
        max_output_tokens=opts.get("max_output_tokens"),
        api_endpoint=opts.get("api_endpoint") or
        "https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent",
        api_version=opts.get("api_version"),
        require_model_version=opts.get("model_version_capture") == "required")


@backend("gpt")
def _gpt_backend(prompt: str, frame_paths: List[str], model_id: str,
                 opts: Dict[str, Any]) -> str:
    from analysis.backends.api import openai_generate
    return openai_generate(prompt, frame_paths, model_id,
                           seed=opts.get("seed"), temperature=opts.get("temperature"))


def _oracle_text(gt: Dict[str, Any]) -> str:
    """Echo GT into the predicted format (l4 uses predicted_counterfactuals with
    the {intervention, control_chain, treated_chain} shape — matching a
    perfect model, not raw GT)."""
    query_set = gt.get("l4_queries") or gt.get("counterfactuals", [])
    l4 = []
    for cf in query_set:
        unit = {"intervention": cf["counterfactual_chain"]["intervention"],
                "treated_chain": cf["counterfactual_chain"]["chain"]}
        # Pair protocol (docs/l4_pair_protocol.md): a migrated reference also
        # carries the control chain; the oracle echoes both.
        if isinstance(cf.get("control_chain"), dict):
            unit["control_chain"] = cf["control_chain"].get("chain", [])
        l4.append(unit)
    return json.dumps({"predicted_branches": gt.get("branches", []),
                       "predicted_counterfactuals": l4})


# GT-derived reference backends: name -> callable(gt) -> raw text. The oracle
# and the dissertation's diagnostic baselines live here because they read the
# scenario ground truth (by design), not the prompt/frames.
from analysis.baselines import (action_profile_kinematic_text,  # noqa: E402
                                action_profile_published_text,
                                action_profile_typical_text,
                                copy_factual_text,
                                kinematic_extrapolation_text,
                                random_valid_text)

# Uniform contract: callable(gt, seed, context) -> raw text. Deterministic
# backends ignore the seed. `context` carries the per-scene public parameter
# records the prompt prints (`hazard_profile`, `intervention_profiles`; None
# on a dataset without a released query) so that a control can be
# given exactly the information the model receives; every backend other than
# `action_profile_published` ignores it.
GT_BACKENDS: Dict[str, Callable[[Dict[str, Any], int, Dict[str, Any]], str]] = {
    "oracle": lambda gt, seed, context: _oracle_text(gt),
    "copy_factual": lambda gt, seed, context: copy_factual_text(gt),
    "kinematic_extrapolation": lambda gt, seed, context: kinematic_extrapolation_text(gt),
    "action_profile_kinematic": lambda gt, seed, context: action_profile_kinematic_text(gt),
    "action_profile_typical": lambda gt, seed, context: action_profile_typical_text(gt),
    "action_profile_published": lambda gt, seed, context: action_profile_published_text(
        gt, hazard_profile=context.get("hazard_profile"),
        intervention_profiles=context.get("intervention_profiles")),
    "random_valid": lambda gt, seed, context: random_valid_text(gt, seed),
}


def _resolve_frames(scenario_dir: str, num_frames: int,
                    exact_count: int = None) -> List[str]:
    """Resolve pre-extracted frames from <scenario_dir>/frames/.

    Protocol manifests enforce an exact count and preserve chronological order;
    datasets without a manifest use deterministic subsampling.
    """
    frame_dir = os.path.join(scenario_dir, "frames")
    if not os.path.isdir(frame_dir):
        if exact_count is not None:
            raise ValueError(
                f"protocol frame contract violated in {scenario_dir}: "
                f"frame directory is missing, expected exactly {exact_count}")
        return []
    paths = sorted(glob.glob(os.path.join(frame_dir, "*.jpg")) +
                   glob.glob(os.path.join(frame_dir, "*.png")))
    if exact_count is not None and len(paths) != int(exact_count):
        raise ValueError(
            f"protocol frame contract violated in {scenario_dir}: "
            f"found {len(paths)}, expected exactly {exact_count}")
    if exact_count is not None:
        return paths
    return select_frames(paths, num_frames)


def _validate_model_config(model_config: Optional[Dict[str, Any]],
                           backend_name: str, model_id: str,
                           temperature: float,
                           allow_unadmitted_smoke: bool = False
                           ) -> Optional[Dict[str, Any]]:
    """Validate resolved panel metadata at the runner boundary."""
    if model_config is None:
        return None
    if not isinstance(model_config, dict):
        raise ValueError("model_config must be a resolved panel mapping")
    required = {
        "name", "model_id", "backend", "adapter", "revision",
        "inference_mode", "runtime_ready", "max_output_tokens",
        "thinking_trace_storage", "scorer_input",
    }
    missing = sorted(required - set(model_config))
    if missing:
        raise ValueError(f"model_config missing fields: {missing}")
    if model_config.get("cache_implementation") is not None:
        if (not allow_unadmitted_smoke or backend_name != "local"
                or model_config["cache_implementation"] != "offloaded"):
            raise ValueError("cache offloading is an explicit local diagnostic option only")
    if model_config["runtime_ready"] is not True and not allow_unadmitted_smoke:
        raise RuntimeError(
            f"configuration {model_config['name']} is not runtime-admitted: "
            f"{model_config.get('runtime_blocker', 'unknown blocker')}")
    if allow_unadmitted_smoke and model_config.get("adapter_implemented") is not True:
        raise RuntimeError(
            f"configuration {model_config['name']} has no implemented adapter")
    if model_config["runtime_ready"] is True and not allow_unadmitted_smoke:
        admission_fields = {
            "smoke_admission_version", "smoke_admission_sha256",
            "smoke_code_commit", "smoke_completed_at",
        }
        missing_admission = sorted(
            field for field in admission_fields if not model_config.get(field))
        if missing_admission:
            raise ValueError(
                "runtime-ready model_config lacks smoke admission metadata: "
                f"{missing_admission}")
    if backend_name != model_config["backend"]:
        raise ValueError(
            f"backend {backend_name!r} differs from configuration backend "
            f"{model_config['backend']!r}")
    if model_id != model_config["model_id"]:
        raise ValueError(
            f"model_id {model_id!r} differs from pinned configuration model_id "
            f"{model_config['model_id']!r}")
    # Greedy, unless the configuration declares the sampling its model card
    # requires (a model that cannot decode greedily).
    declared = float((model_config.get("sampling") or {}).get("temperature", 0.0))
    if temperature != declared:
        raise ValueError(f"panel-backed benchmark runs require temperature {declared:g}")
    if model_config["max_output_tokens"] != MAX_OUTPUT_TOKENS:
        raise ValueError(
            f"panel-backed benchmark runs require a {MAX_OUTPUT_TOKENS:,}-token cap")
    if model_config["thinking_trace_storage"] != "separate_artifact" or \
            model_config["scorer_input"] != "final_json_only":
        raise ValueError("model_config violates thinking/final-output separation")
    return dict(model_config)


def _existing_output_complete(out_dir: str,
                              model_config: Optional[Dict[str, Any]],
                              condition: str,
                              prompt_variant: str = "full",
                              prompt_sha256: Optional[str] = None,
                              system_prompt_sha256: Optional[str] = None) -> bool:
    """Return true only for a complete, configuration-compatible scenario.
    With `prompt_sha256` / `system_prompt_sha256` given (the hashes of the
    prompt this run would send), an existing answer to another prompt text is
    refused rather than resumed: a prompt edit between two passes of a cell
    must never leave two prompts in one cell unnoticed."""
    basic = ("l3_pre.json", "l4_pre.json")
    if not all(os.path.isfile(os.path.join(out_dir, name)) for name in basic):
        return False
    if model_config is None:
        return True
    required = ("raw_response.txt", "final_response.txt",
                "generation_metadata.json", "run_manifest.json")
    if not all(os.path.isfile(os.path.join(out_dir, name)) for name in required):
        return False
    with open(os.path.join(out_dir, "run_manifest.json"), encoding="utf-8") as handle:
        existing = json.load(handle)
    # Manifests written before 2026-09-24 carry no variant: they are `full`.
    existing.setdefault("prompt_variant", "full")
    expected = {
        "prompt_variant": prompt_variant,
        "configuration_name": model_config["name"],
        "model_id": model_config["model_id"],
        "model_revision": model_config.get("revision"),
        "adapter": model_config["adapter"],
        "inference_mode": model_config["inference_mode"],
        "max_output_tokens": model_config["max_output_tokens"],
        "cache_implementation": model_config.get("cache_implementation"),
        "condition": condition,
    }
    if model_config.get("runtime_ready") is True:
        expected.update({
            "smoke_admission_version": model_config.get(
                "smoke_admission_version"),
            "smoke_admission_sha256": model_config.get(
                "smoke_admission_sha256"),
            "smoke_code_commit": model_config.get("smoke_code_commit"),
        })
        if model_config.get('smoke_status') is not None:
            expected['smoke_status'] = model_config['smoke_status']
    mismatches = {key: (existing.get(key), value)
                  for key, value in expected.items()
                  if existing.get(key) != value}
    if mismatches:
        raise ValueError(
            f"existing result metadata conflicts with requested configuration: {mismatches}")
    if prompt_sha256 is not None or system_prompt_sha256 is not None:
        drift = {key: (existing.get(key), value)
                 for key, value in (("prompt_sha256", prompt_sha256),
                                    ("system_prompt_sha256", system_prompt_sha256))
                 if existing.get(key) != value}
        if drift:
            raise ValueError(
                f"existing result was generated under another prompt text {drift}; "
                f"a cell holds one prompt: use a new out-dir")
    return True


def run_model(gt_dir: str, out_dir: str, backend: str = "mock",
              model_id: str = "", num_frames: int = 8,
              seed: int = 0,
              condition: str = "video", track: str = PRIMARY_TRACK,
              temperature: float = 0.0,
              skip_existing: bool = False,
              model_config: Optional[Dict[str, Any]] = None,
              scenario_names: Optional[List[str]] = None,
              provisional_smoke: bool = False,
              prompt_variant: str = "full") -> Dict[str, Any]:
    """Run `backend` over every scenario_*/ground_truth.json under gt_dir and write
    l3_pre.json / l4_pre.json under out_dir/scenario_N/. Returns run stats.
    `prompt_variant` selects the prompt (`build_pilot.PROMPT_VARIANTS`) and is
    recorded in every manifest; a resume under another variant is refused."""
    validate_condition(condition)
    prompt_variant = validate_prompt_variant(prompt_variant)
    if backend not in BACKENDS and backend not in GT_BACKENDS:
        raise KeyError(f"unknown backend: {backend!r} "
                       f"(have: {sorted(BACKENDS)} + {sorted(GT_BACKENDS)})")
    resolved_config = _validate_model_config(
        model_config, backend, model_id, temperature,
        allow_unadmitted_smoke=provisional_smoke)

    scenario_dirs = sorted(glob.glob(os.path.join(gt_dir, "scenario_*")))
    if scenario_names is not None:
        requested = set(scenario_names)
        if not requested or len(requested) != len(scenario_names):
            raise ValueError("scenario_names must contain unique scenario basenames")
        indexed = {os.path.basename(path): path for path in scenario_dirs
                   if os.path.isdir(path)}
        missing_scenarios = sorted(requested - set(indexed))
        if missing_scenarios:
            raise ValueError(
                f"requested scenarios are missing from {gt_dir}: {missing_scenarios}")
        scenario_dirs = [indexed[name] for name in scenario_names]
    n, skipped, failed = 0, 0, []
    for scen_dir in scenario_dirs:
        if not os.path.isdir(scen_dir):
            continue
        od_existing = os.path.join(out_dir, os.path.basename(scen_dir))
        if skip_existing and resolved_config is None and _existing_output_complete(
                od_existing, resolved_config, condition, prompt_variant):
            skipped += 1
            continue
        gt = json.load(open(os.path.join(scen_dir, "ground_truth.json")))
        scenario_id = gt.get("scenario_id", os.path.basename(scen_dir))
        # The intervention list is the scenario's rung-3 QUERY set (metadata,
        # not answers): the model must be asked about exactly the GT
        # counterfactuals or L4's exact-type matching can never succeed.
        query_set = gt.get("l4_queries") or gt.get("counterfactuals", [])
        interventions = [cf.get("intervention_type") for cf in query_set
                         if cf.get("intervention_type")]
        manifest_path = os.path.join(scen_dir, "input_manifest.json")
        manifest = json.load(open(manifest_path)) if os.path.isfile(manifest_path) else {}
        if manifest:
            validate_protocol_manifest(manifest)
            if manifest.get("track") != track:
                raise ValueError(
                    f"runner track={track!r} does not match scenario manifest "
                    f"track={manifest.get('track')!r}: {scen_dir}")
            if condition not in manifest.get("available_conditions", INPUT_CONDITIONS):
                raise ValueError(
                    f"condition={condition!r} is not released by scenario manifest: {scen_dir}")
        observed_history = None
        released_ontology = None
        released_query = None
        state_path = os.path.join(scen_dir, "observed_state.json")
        history_path = os.path.join(scen_dir, "observed_history.json")
        if manifest and manifest.get("protocol_version") == PROTOCOL_VERSION:
            from simulation.observed_history import load_observed_artifacts
            standalone_root, observed_history = load_observed_artifacts(scen_dir)
            if standalone_root != gt.get("observed_state"):
                raise ValueError(f"standalone observed_state differs from ground truth: {scen_dir}")
            with open(os.path.join(scen_dir, manifest.get("ontology_asset", "ontology.json")),
                      encoding="utf-8") as f:
                released_ontology = json.load(f)
            with open(os.path.join(scen_dir, manifest.get("query_asset", "query.json")),
                      encoding="utf-8") as f:
                released_query = json.load(f)
        elif condition in {"history_json", "video_history"}:
            if not os.path.isfile(state_path) or not os.path.isfile(history_path):
                raise ValueError(f"condition={condition} requires observed input artifacts: {scen_dir}")
            from simulation.observed_history import load_observed_artifacts
            _, observed_history = load_observed_artifacts(scen_dir)
        l3_actions = [
            (branch.get("intervention") or {}).get("intervention_type")
            for branch in gt.get("branches", [])
            if (branch.get("intervention") or {}).get("target_actor_id", "ego") == "ego"
        ]
        l3_actions = [action for action in l3_actions if action]
        if released_query is not None:
            if released_query.get("l3_ego_actions") != l3_actions:
                raise ValueError(f"query L3 actions differ from ground truth metadata: {scen_dir}")
            if gt.get("l4_queries"):
                from analysis.evaluation_campaign import l4_query_mismatches
                problems = l4_query_mismatches(gt, released_query, manifest)
                if problems:
                    raise ValueError(f"query L4 differs from ground truth reference: {scen_dir}: {problems}")
            interventions = [item.get("intervention_type")
                             for item in released_query.get("l4_interventions", [])
                             if item.get("intervention_type")]
        intervention_profiles = None
        hazard_profile = None
        if released_query is not None:
            # The intervention parameters THIS scene's reference applies:
            # nuScenes carries them in the released query; CARLA executes the
            # scene configuration (private, read for its declared parameters
            # only, never its future states).
            from simulation.build_pilot import public_intervention_profiles
            scenario_config = {}
            config_path = os.path.join(scen_dir, "scenario_config.json")
            if os.path.isfile(config_path):
                with open(config_path, encoding="utf-8") as f:
                    scenario_config = json.load(f) or {}
            env_behavior = scenario_config.get("env_behavior")
            intervention_profiles = public_intervention_profiles(
                gt["observed_state"], released_query, scenario_config, manifest.get('split') or manifest.get('dataset'))
            from simulation.build_pilot import declared_hazard_profile
            hazard_profile = declared_hazard_profile(gt, manifest.get("critical_actor_id"), env_behavior)
        # Factual-branch contract: the released query names it; a query
        # without it falls back to the reference's declared factual label (a
        # declared parameter, never a future state).
        factual_contract = None
        if released_query is not None:
            factual_contract = {k: released_query.get(k) for k in ("factual_action", "factual_reference")}
            if not factual_contract.get("factual_action"):
                from simulation.observed_history import factual_branch_contract
                factual_contract = factual_branch_contract(gt)
        system_prompt, prompt = build_prompt_parts(
            gt["observed_state"], scenario_id,
            interventions=interventions or None,
            critical_actor_id=manifest.get("critical_actor_id"),
            condition=condition, track=track,
            observed_history=observed_history,
            l3_actions=l3_actions or None,
            ontology=released_ontology, query=released_query,
            intervention_profiles=intervention_profiles,
            split=(gt.get("protocol") or {}).get("split") or (manifest or {}).get("split"),
            hazard_profile=hazard_profile,
            factual_action=(factual_contract or {}).get("factual_action"),
            factual_reference=(factual_contract or {}).get("factual_reference"),
            variant=prompt_variant)
        # A configuration may append its model card's own instruction to the
        # fixed system prompt (the Cosmos-Reason2 reasoning request). It is prompt
        # text: it enters the hashes below and the saved system prompt.
        suffix = (resolved_config or {}).get("system_prompt_suffix")
        if suffix:
            system_prompt = system_prompt + suffix
        if backend not in SYSTEM_PROMPT_BACKENDS:
            prompt, system_prompt = join_prompt_parts(system_prompt, prompt), None
        prompt_sha256 = hashlib.sha256(prompt.encode("utf-8")).hexdigest()
        system_prompt_sha256 = (hashlib.sha256(system_prompt.encode("utf-8")).hexdigest()
                                if system_prompt is not None else None)
        # A panel scenario is resumed only under the prompt text it was answered
        # with; the prompt is built first so the hashes can be compared.
        if skip_existing and resolved_config is not None and _existing_output_complete(
                od_existing, resolved_config, condition, prompt_variant,
                prompt_sha256, system_prompt_sha256):
            skipped += 1
            continue
        frames = (_resolve_frames(
            scen_dir, num_frames,
            exact_count=manifest.get("observed_frame_count") if manifest else None)
                  if condition in {"video", "video_history"} else [])
        frame_sha256 = []
        for frame_path in frames:
            with open(frame_path, "rb") as frame_handle:
                frame_sha256.append(
                    hashlib.sha256(frame_handle.read()).hexdigest())

        if backend in GT_BACKENDS:
            # The same published records that went into the prompt above.
            published = {"hazard_profile": hazard_profile,
                         "intervention_profiles": intervention_profiles}
            generation = normalize_generation_result(
                GT_BACKENDS[backend](gt, seed, published))
            sampling_seed = seed
        else:
            sampling_seed = seed if temperature is not None and temperature > 0 else None
            opts: Dict[str, Any] = {"seed": sampling_seed,
                                    "temperature": temperature}
            if resolved_config is not None:
                opts.update(resolved_config)
            if system_prompt is not None:
                opts["system_prompt"] = system_prompt
            started = time.monotonic()
            generation = normalize_generation_result(
                BACKENDS[backend](prompt, frames, model_id, opts))
            generation.metadata.setdefault(
                "latency_seconds", time.monotonic() - started)

        # The parser/scorer sees only the final response.  Raw reasoning is an
        # audit artifact and cannot rescue or contaminate the scored JSON.
        l3, l4, errors = parse_model_output(generation.final_text, scenario_id)
        name = os.path.basename(scen_dir)
        od = os.path.join(out_dir, name)
        os.makedirs(od, exist_ok=True)
        if backend in BACKENDS:  # real model output: archive the raw text
            with open(os.path.join(od, "prompt.txt"), "w", encoding="utf-8") as f:
                f.write(prompt)
            if system_prompt is not None:
                with open(os.path.join(od, "system_prompt.txt"), "w", encoding="utf-8") as f:
                    f.write(system_prompt)
            with open(os.path.join(od, "raw_response.txt"), "w", encoding="utf-8") as f:
                f.write(generation.raw_text)
            with open(os.path.join(od, "final_response.txt"), "w", encoding="utf-8") as f:
                f.write(generation.final_text)
            if generation.thinking_trace is not None:
                with open(os.path.join(od, "thinking_trace.txt"), "w",
                          encoding="utf-8") as f:
                    f.write(generation.thinking_trace or "")
            with open(os.path.join(od, "generation_metadata.json"), "w",
                      encoding="utf-8") as f:
                json.dump(generation.metadata, f, indent=2)
        with open(os.path.join(od, "l3_pre.json"), "w") as f:
            json.dump(l3, f, indent=2)
        with open(os.path.join(od, "l4_pre.json"), "w") as f:
            json.dump(l4, f, indent=2)
        if errors:
            failed.append({"scenario_id": scenario_id, "errors": errors})
        with open(os.path.join(od, "run_manifest.json"), "w", encoding="utf-8") as f:
            json.dump({
                "protocol_version": manifest.get("protocol_version", "legacy"),
                "scorer_input_track": track,
                "condition": condition,
                "model_id": model_id,
                "backend": backend,
                "configuration_name": (resolved_config or {}).get("name"),
                "model_panel_version": (resolved_config or {}).get("model_panel_version"),
                "model_revision": (resolved_config or {}).get("revision"),
                "adapter": (resolved_config or {}).get("adapter"),
                "inference_mode": (resolved_config or {}).get("inference_mode"),
                "enable_thinking": (resolved_config or {}).get("enable_thinking"),
                "max_output_tokens": (resolved_config or {}).get("max_output_tokens"),
                "cache_implementation": (resolved_config or {}).get("cache_implementation"),
                "smoke_admission_version": (resolved_config or {}).get(
                    "smoke_admission_version"),
                "smoke_admission_sha256": (resolved_config or {}).get(
                    "smoke_admission_sha256"),
                "smoke_code_commit": (resolved_config or {}).get(
                    "smoke_code_commit"),
                "smoke_completed_at": (resolved_config or {}).get(
                    "smoke_completed_at"),
                "smoke_status": (resolved_config or {}).get("smoke_status"),
                "seed": seed,
                "generation_seed": sampling_seed,
                "temperature": temperature,
                "decoding": ("sampling" if temperature is not None and temperature > 0
                             else "greedy"),
                "prompt_sha256": prompt_sha256,
                "system_prompt_sha256": system_prompt_sha256,
                "parser_version": PARSER_VERSION,
                "scorer_version": SCORER_VERSION,
                "prompt_variant": prompt_variant,
                "prompt_version": prompt_version(prompt_variant),
                "frame_count": len(frames),
                "frame_files": [os.path.basename(path) for path in frames],
                "frame_sha256": frame_sha256,
                "raw_response_sha256": hashlib.sha256(
                    generation.raw_text.encode("utf-8")).hexdigest(),
                "final_response_sha256": hashlib.sha256(
                    generation.final_text.encode("utf-8")).hexdigest(),
                "thinking_trace_present": generation.thinking_trace is not None,
                "generation_metadata": generation.metadata,
                "primary_measurement": manifest.get("primary_measurement"),
                "reference_evaluation_role": manifest.get("reference_evaluation_role"),
                "reference_semantics": manifest.get("reference_semantics"),
                "execution_purpose": (
                    "smoke_admission" if provisional_smoke else "benchmark"),
            }, f, indent=2)
        n += 1

    return {"scenarios": n, "skipped": skipped, "backend": backend,
            "failed": failed, "out_dir": out_dir,
            "condition": condition, "track": track,
            "configuration_name": (resolved_config or {}).get("name")}


def main():
    p = argparse.ArgumentParser(description="Run a VLM over CORIVA scenarios -> predictions")
    p.add_argument("--gt-dir", required=True, help="dir of scenario_*/ground_truth.json")
    p.add_argument("--out-dir", required=True, help="results root (out-dir/scenario_N/l3_pre.json)")
    p.add_argument("--backend", default=None,
                   help="legacy direct backend; panel configurations resolve this automatically")
    p.add_argument("--model", default="", help="model id or local path (backend-specific)")
    p.add_argument("--configuration", default=None,
                   help="name from the versioned model panel")
    p.add_argument("--model-panel", default="analysis/model_panel.yaml")
    p.add_argument("--smoke-admission", default=None,
                   help="validated Step 4 admission artifact")
    p.add_argument("--num-frames", type=int, default=8)
    p.add_argument("--seed", type=int, default=0,
                   help="run id; also the generation seed only when temperature > 0")
    p.add_argument("--condition", choices=INPUT_CONDITIONS, default="video")
    p.add_argument("--track", default=PRIMARY_TRACK,
                   choices=["primary", "challenge_6s"])
    p.add_argument("--temperature", type=float, default=0.0,
                   help="generation temperature for real backends (0 = greedy)")
    p.add_argument("--skip-existing", action="store_true",
                   help="skip scenarios whose prediction files already exist (resume)")
    args = p.parse_args()
    resolved_config = None
    backend_name = args.backend or "mock"
    model_id = args.model
    if args.configuration:
        from analysis.model_panel import load_model_panel, resolve_configurations
        panel = load_model_panel(args.model_panel)
        if args.smoke_admission:
            from analysis.smoke_admission import admit_panel
            panel = admit_panel(panel, args.smoke_admission)
        resolved_config = resolve_configurations(panel, [args.configuration])[0]
        resolved_config.update({
            "model_panel_version": panel["panel_version"],
            "protocol_version": panel["protocol_version"],
            "scorer_version": panel["scorer_version"],
        })
        if args.backend is not None and args.backend != resolved_config["backend"]:
            p.error("--backend conflicts with --configuration")
        if args.model and args.model != resolved_config["model_id"]:
            p.error("--model conflicts with the pinned --configuration model_id")
        backend_name = resolved_config["backend"]
        model_id = resolved_config["model_id"]
    r = run_model(args.gt_dir, args.out_dir, backend_name, model_id,
                  args.num_frames, seed=args.seed,
                  condition=args.condition, track=args.track,
                  temperature=args.temperature, skip_existing=args.skip_existing,
                  model_config=resolved_config)
    print(f"scenarios={r['scenarios']} backend={r['backend']} "
          f"failed={len(r['failed'])} -> {r['out_dir']}")
    if r["failed"]:
        print("  parse failures:", json.dumps(r["failed"][:5], indent=2))


if __name__ == "__main__":
    main()
