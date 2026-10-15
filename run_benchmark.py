"""
run_benchmark.py — Official CORIVA benchmark runner.

Loads GT/pred datasets, calls ctms.py core scorer, aggregates results,
and writes a structured output report.

This file is the benchmark protocol runner. It does NOT contain any
scoring logic — all scoring is delegated to ctms.py.

Usage:
    python3 run_benchmark.py \\
        --gt data/gt_dataset.json \\
        --pred data/pred_dataset.json \\
        --output results/benchmark_report.json

    python3 run_benchmark.py \\
        --gt data/gt_dataset.json \\
        --pred data/pred_dataset.json \\
        --task layer3 \\
        --output results/l3_report.json
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import sys
import traceback
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Tuple

import yaml

# Core scorer — the only scoring dependency. Import as the ctms.ctms
# submodule (repo root on sys.path) rather than inserting ctms/ itself,
# which breaks whenever the ctms package is already imported.
sys.path.insert(0, str(Path(__file__).resolve().parent))
from ctms.ctms import (
    evaluate_ctms,
    load_ctms_environment,
    score_layer3,
    score_layer4,
    SCORER_VERSION,
    apply_reference_gate_policy,
)
from simulation.protocol import (PROTOCOL_VERSION, PRIMARY_TRACK, PRIMARY_MEASUREMENT,
                                 REFERENCE_EVALUATION_ROLE, reference_semantics,
                                 track_steps)


# ── I/O helpers ──────────────────────────────────────────────────────


def load_yaml(path: str) -> Dict[str, Any]:
    """Load a YAML file. Raises SystemExit on error."""
    try:
        with open(path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f)
    except FileNotFoundError:
        print(f"ERROR: YAML file not found: {path}", file=sys.stderr)
        sys.exit(1)
    except yaml.YAMLError as e:
        print(f"ERROR: Malformed YAML in {path}: {e}", file=sys.stderr)
        sys.exit(1)


def load_json(path: str) -> Dict[str, Any]:
    """Load a JSON file. Raises SystemExit on error."""
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except FileNotFoundError:
        print(f"ERROR: JSON file not found: {path}", file=sys.stderr)
        sys.exit(1)
    except json.JSONDecodeError as e:
        print(f"ERROR: Malformed JSON in {path}: {e}", file=sys.stderr)
        sys.exit(1)


def load_artifacts(args: argparse.Namespace) -> Dict[str, Any]:
    """Load all scoring artifacts via ctms.load_ctms_environment()."""
    try:
        return load_ctms_environment(
            ontology_path=args.ontology,
            config_path=args.config,
            branch_schema_path=args.branch_schema,
            counterfactual_schema_path=args.counterfactual_schema,
        )
    except Exception as e:
        print(f"ERROR: Failed to load artifacts: {e}", file=sys.stderr)
        sys.exit(1)


# ── Dataset alignment ────────────────────────────────────────────────


def index_prediction_samples(
    pred_dataset: Dict[str, Any],
) -> Dict[str, Dict[str, Dict[str, Any]]]:
    """Build {scenario_id: {sample_id: prediction}} lookup from pred dataset."""
    index: Dict[str, Dict[str, Dict[str, Any]]] = {}
    for scenario in pred_dataset.get("scenarios", []):
        sid = scenario.get("scenario_id", "")
        index[sid] = {}
        for sample in scenario.get("samples", []):
            sample_id = sample.get("sample_id", "")
            index[sid][sample_id] = sample.get("prediction", {})
    return index


def adapt_pred_to_scorer(prediction: Dict[str, Any]) -> Dict[str, Any]:
    """Map prediction keys to what ctms.evaluate_ctms() expects.

    LLM output uses 'predicted_branches' but ctms.py reads 'branches'.
    This adapter normalizes without duplicating scoring logic.
    """
    adapted = dict(prediction)
    # predicted_branches → branches
    if "predicted_branches" in adapted and "branches" not in adapted:
        adapted["branches"] = adapted.pop("predicted_branches")
    return adapted


def iter_aligned_samples(
    gt_dataset: Dict[str, Any],
    pred_index: Dict[str, Dict[str, Dict[str, Any]]],
) -> Iterator[Tuple[Dict[str, Any], Dict[str, Any], str, str, List[str]]]:
    """Yield aligned (gt_sample, pred_sample, scenario_id, sample_id, warnings).

    For each GT sample, find matching prediction. Yield empty pred if missing.
    Also track unmatched prediction samples.
    """
    matched_pred_keys: set = set()

    for gt_scenario in gt_dataset.get("scenarios", []):
        sid = gt_scenario.get("scenario_id", "")
        pred_samples = pred_index.get(sid, {})

        for gt_sample in gt_scenario.get("samples", []):
            sample_id = gt_sample.get("sample_id", "")
            gt_data = gt_sample.get("ground_truth", {})
            warnings: List[str] = []

            if sample_id in pred_samples:
                pred_data = pred_samples[sample_id]
                matched_pred_keys.add((sid, sample_id))
            else:
                pred_data = {}
                warnings.append(f"Missing prediction for {sid}/{sample_id}")

            yield gt_data, pred_data, sid, sample_id, warnings

    # Find unmatched prediction samples (in pred but not in GT)
    # This is reported via diagnostics, not yielded as samples


def find_unmatched_predictions(
    gt_dataset: Dict[str, Any],
    pred_index: Dict[str, Dict[str, Dict[str, Any]]],
) -> List[Dict[str, str]]:
    """Find prediction samples that have no corresponding GT sample."""
    gt_keys: set = set()
    for gt_scenario in gt_dataset.get("scenarios", []):
        sid = gt_scenario.get("scenario_id", "")
        for gt_sample in gt_scenario.get("samples", []):
            gt_keys.add((sid, gt_sample.get("sample_id", "")))

    unmatched = []
    for sid, samples in pred_index.items():
        for sample_id in samples:
            if (sid, sample_id) not in gt_keys:
                unmatched.append({"scenario_id": sid, "sample_id": sample_id})
    return unmatched


# ── Per-sample scoring ───────────────────────────────────────────────


def score_sample(
    sample_gt: Dict[str, Any],
    sample_pred: Dict[str, Any],
    env: Dict[str, Any],
    task: str = "full",
    map_data: Optional[Dict[str, Any]] = None,
    d_max: Optional[int] = None,
) -> Dict[str, Any]:
    """Score one sample using ctms.py core functions.

    Delegates entirely to ctms.py — no scoring logic here. ``d_max`` is the
    explicit horizon override used by prefix scoring; otherwise the horizon
    follows the sample's declared track.
    """
    ontology = env["ontology"]
    # Protocol-dependent gates and the track horizon follow the sample's own
    # declaration, so direct callers (validation studies, prefix scoring)
    # agree with the CLI.
    config = apply_reference_gate_policy(env["config"], sample_gt, horizon=d_max)
    branch_schema = env["branch_schema"]
    counterfactual_schema = env["counterfactual_schema"]
    # CARLA protocol samples embed scenario-local waypoint/drivable evidence.
    # An explicitly supplied HD map remains authoritative; otherwise use the
    # sample geometry instead of reconstructing a straight corridor at T0.
    if map_data is None:
        map_data = sample_gt.get("scene_geometry")

    # Adapt prediction keys
    adapted_pred = adapt_pred_to_scorer(sample_pred)

    if task == "full":
        return evaluate_ctms(
            sample_gt=sample_gt,
            sample_pred=adapted_pred,
            ontology=ontology,
            config=config,
            branch_schema=branch_schema,
            counterfactual_schema=counterfactual_schema,
            map_data=map_data,
        )
    elif task == "layer3":
        result = score_layer3(
            gt_branches=sample_gt.get("branches", []),
            pred_branches=adapted_pred.get("branches", []),
            ontology=ontology,
            config=config,
            branch_schema=branch_schema,
            map_data=map_data,
            root=sample_gt.get("observed_state"),
        )
        return {
            "Causal_Foresight_F1": result.get("F1_cf", 0.0),
            "layer3": result,
        }
    elif task == "layer4":
        # For layer4-only, use evaluate_ctms but extract layer4 results
        full_result = evaluate_ctms(
            sample_gt=sample_gt,
            sample_pred=adapted_pred,
            ontology=ontology,
            config=config,
            branch_schema=branch_schema,
            counterfactual_schema=counterfactual_schema,
            map_data=map_data,
        )
        return {
            "Counterfactual_Alignment_Score": full_result.get(
                "Counterfactual_Alignment_Score", 0.0
            ),
            "layer4": full_result.get("layer4", {}),
        }
    else:
        raise ValueError(f"Unknown task: {task}")


def _future_prefix(chain: Any, depth: int, ontology: Optional[Dict[str, Any]]) -> Any:
    """The first `depth` FUTURE nodes: leading observed_state nodes (the echoed
    root, which the scorer strips before scoring) are skipped first, so a
    prediction scores the same prefix whether or not it echoes the root
    (otherwise the root would occupy one of the `depth` slots and the 3 s
    prefix of an echoing answer would lose its last future node)."""
    if not isinstance(chain, list):
        return chain
    ev_syn = ((ontology or {}).get("synonyms") or {}).get("event_type", {})
    start = 0
    while start < len(chain) and isinstance(chain[start], dict) and \
            ev_syn.get(chain[start].get("event_type"), chain[start].get("event_type")) == "observed_state":
        start += 1
    return chain[start:start + depth]


def _truncate_for_horizon(value: Dict[str, Any], depth: int, is_gt: bool,
                          ontology: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Return a score-only prefix without mutating released GT/prediction data."""
    out = copy.deepcopy(value)
    branch_keys = ("branches",) if is_gt else ("predicted_branches", "branches")
    for branch_key in branch_keys:
        for branch in out.get(branch_key, []) or []:
            if isinstance(branch, dict):
                branch["chain"] = _future_prefix(branch.get("chain") or [], depth, ontology)
    for key in (("counterfactuals", "l4_queries") if is_gt else
                ("predicted_counterfactuals",)):
        for item in out.get(key, []) or []:
            if not isinstance(item, dict):
                continue
            if is_gt:
                # treated reference and, on a pair-protocol dataset, the control
                for wrapper_key in ("counterfactual_chain", "control_chain"):
                    chain = item.get(wrapper_key)
                    if isinstance(chain, dict):
                        chain["chain"] = _future_prefix(chain.get("chain") or [], depth, ontology)
            else:
                # treated (`treated_chain`, alias `chain`) and control chains of the
                # predicted pair; a non-list value is left for the gates to reject
                for chain_key in ("chain", "treated_chain", "control_chain"):
                    if chain_key in item and isinstance(item[chain_key], list):
                        item[chain_key] = _future_prefix(item[chain_key], depth, ontology)
    return out


def prefix_scores(sample_gt: Dict[str, Any], sample_pred: Dict[str, Any],
                  env: Dict[str, Any], map_data: Optional[Dict[str, Any]] = None
                  ) -> Dict[str, Dict[str, float]]:
    """Compute actual CTMS prefix scores at 1/2/3 seconds."""
    out: Dict[str, Dict[str, float]] = {}
    horizons = [(1, 2), (2, 4), (3, 6)]
    if (sample_gt.get("protocol") or {}).get("track") == "challenge_6s":
        horizons.append((6, 12))
    for seconds, depth in horizons:
        prefix_gt = _truncate_for_horizon(sample_gt, depth, True, env.get("ontology"))
        prefix_pred = _truncate_for_horizon(sample_pred, depth, False, env.get("ontology"))
        result = score_sample(prefix_gt, prefix_pred, env, task="full",
                              map_data=map_data, d_max=depth)
        out[f"{seconds}s"] = {
            "Causal_Foresight_F1": result.get("Causal_Foresight_F1", 0.0),
            "Counterfactual_Alignment_Score": result.get(
                "Counterfactual_Alignment_Score", 0.0),
            "CTMS": result.get("CTMS", 0.0),
        }
    return out


# ── Aggregation ──────────────────────────────────────────────────────


def _reduce(values: List[float], method: str) -> float:
    """Apply a reduction method to a list of values."""
    if not values:
        return 0.0
    if method == "mean":
        return sum(values) / len(values)
    elif method == "min":
        return min(values)
    elif method == "max":
        return max(values)
    else:
        return sum(values) / len(values)  # default to mean


def _aggregate_compliance(sample_results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Compliance axis: pool gate outcomes across all samples.

    Separates "the model can produce valid structured output" (reported
    here) from "the model reasons correctly" (the score axes), so a low
    leaderboard number can be attributed to one or the other.
    """
    br_total = br_valid = 0
    br_fail = {"schema": 0, "semantic": 0, "scene": 0, "kinematic": 0, "metadata": 0}
    br_first = dict.fromkeys(br_fail, 0)
    k_max_violations = 0
    cf_total = cf_valid = 0
    cf_fail = {"schema": 0, "semantic": 0, "scene": 0, "kinematic": 0, "metadata": 0}
    cf_first = dict.fromkeys(cf_fail, 0)
    # Option B pair accounting (docs/l4_pair_protocol.md §3): the pair is the
    # unit; each chain's own gate outcome is pooled separately.
    pair = {"pair_reference": 0, "legacy_single_chain": 0, "treated_present": 0,
            "control_present": 0, "pair_complete": 0, "treated_valid": 0,
            "control_valid": 0, "pair_valid": 0}
    chain_fail = {"treated": dict.fromkeys(cf_fail, 0), "control": dict.fromkeys(cf_fail, 0)}

    for r in sample_results:
        l3 = r.get("layer3") or {}
        diag = l3.get("diagnostics") or {}
        br_total += int(diag.get("pred_unique_count", 0))
        br_valid += int(diag.get("pred_valid_count", 0))
        for gate, count in (diag.get("gate_failure_counts") or {}).items():
            if gate in br_fail:
                br_fail[gate] += int(count)
        for gate, count in (diag.get("first_failure_counts") or {}).items():
            if gate in br_first:
                br_first[gate] += int(count)
        if float(diag.get("k_max_penalty", 1.0)) < 1.0:
            k_max_violations += 1

        l4 = r.get("layer4") or {}
        for cf in l4.get("results") or []:
            cf_total += 1
            if cf.get("valid"):
                cf_valid += 1
            gates = cf.get("gates") or {}
            for gate in cf_fail:
                if not gates.get(gate, gates.get("intervention", False) if gate == "metadata" else False):
                    cf_fail[gate] += 1
            for gate in ("schema", "semantic", "scene", "kinematic", "metadata"):
                passed = gates.get(gate, gates.get("intervention", False) if gate == "metadata" else False)
                if not passed:
                    cf_first[gate] += 1
                    break
            info = cf.get("pair") or {}
            if info.get("control_reference_present"):
                pair["pair_reference"] += 1
                for key in ("treated_present", "control_present", "pair_complete",
                            "treated_valid", "control_valid", "pair_valid"):
                    if info.get(key):
                        pair[key] += 1
                for side in ("treated", "control"):
                    side_gates = ((cf.get(side) or {}).get("gates")) or {}
                    for gate in cf_fail:
                        if not side_gates.get(gate, side_gates.get("intervention", False)
                                              if gate == "metadata" else False):
                            chain_fail[side][gate] += 1
            else:
                pair["legacy_single_chain"] += 1

    return {
        "branches": {
            "total_unique": br_total,
            "valid": br_valid,
            "gate_pass_rate": (br_valid / br_total) if br_total else 0.0,
            "gate_failure_counts": br_fail,
            "first_failure_counts": br_first,
            "k_max_violations": k_max_violations,
        },
        "counterfactuals": {
            "total": cf_total,
            "valid": cf_valid,
            "gate_pass_rate": (cf_valid / cf_total) if cf_total else 0.0,
            "gate_failure_counts": cf_fail,
            "first_failure_counts": cf_first,
            "pair": pair,
            "chain_gate_failure_counts": chain_fail,
        },
    }


def aggregate_sample_results(
    sample_results: List[Dict[str, Any]],
    aggregation_cfg: Dict[str, Any],
    task: str = "full",
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """Aggregate per-sample results into per-scenario and summary.

    Reads reduction policy from aggregation_cfg (from ctms_config.yaml).
    Returns (per_scenario_list, summary_dict).
    """
    scenario_reduction = aggregation_cfg.get("scenario_reduction", "mean")
    dataset_reduction = aggregation_cfg.get("dataset_reduction", "mean")

    # Group by scenario_id
    groups: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for r in sample_results:
        groups[r["scenario_id"]].append(r)

    # Per-scenario aggregation
    per_scenario = []
    for sid, results in sorted(groups.items()):
        scenario_entry: Dict[str, Any] = {
            "scenario_id": sid,
            "num_samples": len(results),
        }

        if task in ("full", "layer3"):
            f1_vals = [r["scores"].get("Causal_Foresight_F1", 0.0) for r in results]
            scenario_entry["Causal_Foresight_F1_mean"] = _reduce(
                f1_vals, scenario_reduction
            )

        if task in ("full", "layer4"):
            l4_vals = [
                r["scores"].get("Counterfactual_Alignment_Score", 0.0) for r in results
            ]
            scenario_entry["Counterfactual_Alignment_Score_mean"] = _reduce(
                l4_vals, scenario_reduction
            )

        if task == "full":
            ctms_vals = [r["scores"].get("CTMS", 0.0) for r in results]
            scenario_entry["CTMS_mean"] = _reduce(ctms_vals, scenario_reduction)

        per_scenario.append(scenario_entry)

    # Dataset-level summary
    summary: Dict[str, Any] = {
        "num_scenarios": len(per_scenario),
        "num_samples": len(sample_results),
    }

    if task in ("full", "layer3"):
        summary["Causal_Foresight_F1_mean"] = _reduce(
            [s["Causal_Foresight_F1_mean"] for s in per_scenario], dataset_reduction
        )

    if task in ("full", "layer4"):
        summary["Counterfactual_Alignment_Score_mean"] = _reduce(
            [s["Counterfactual_Alignment_Score_mean"] for s in per_scenario],
            dataset_reduction,
        )

    if task == "full":
        summary["CTMS_mean"] = _reduce(
            [s["CTMS_mean"] for s in per_scenario], dataset_reduction
        )

    # ── Compliance vs reasoning separation ───────────────────────────
    if task in ("full", "layer3"):
        f1_valid_vals = [
            r["layer3"].get("F1_cf_valid_only", 0.0)
            for r in sample_results if isinstance(r.get("layer3"), dict)
        ]
        if f1_valid_vals:
            summary["F1_cf_valid_only_mean"] = _reduce(f1_valid_vals, dataset_reduction)
    if task in ("full", "layer4"):
        interv_valid_vals = []
        for r in sample_results:
            l4 = r.get("layer4") or {}
            valid_scores = [cf.get("Score_interv", 0.0)
                            for cf in (l4.get("results") or []) if cf.get("valid")]
            if l4.get("results"):
                interv_valid_vals.append(
                    sum(valid_scores) / len(valid_scores) if valid_scores else 0.0)
        if interv_valid_vals:
            summary["Score_interv_valid_only_mean"] = _reduce(interv_valid_vals, dataset_reduction)
        # Control-chain reference agreement, reported separately from CAS and
        # never folded into CTMS (docs/l4_pair_protocol.md §5).
        control_vals, control_valid_vals, pair_complete, pair_valid, pair_n = [], [], 0, 0, 0
        for r in sample_results:
            l4 = r.get("layer4") or {}
            for cf in l4.get("results") or []:
                info = cf.get("pair") or {}
                if not info.get("control_reference_present"):
                    continue
                pair_n += 1
                pair_complete += int(bool(info.get("pair_complete")))
                pair_valid += int(bool(info.get("pair_valid")))
                control_vals.append(float(cf.get("control_similarity") or 0.0))
                if info.get("control_valid"):
                    control_valid_vals.append(float(cf.get("control_similarity") or 0.0))
        if pair_n:
            summary["L4_control_similarity_mean"] = _reduce(control_vals, dataset_reduction)
            summary["L4_control_similarity_valid_only_mean"] = (
                _reduce(control_valid_vals, dataset_reduction) if control_valid_vals else None)
            summary["L4_pair_complete_rate"] = pair_complete / pair_n
            summary["L4_pair_valid_rate"] = pair_valid / pair_n
            summary["L4_pair_queries"] = pair_n
    summary["compliance"] = _aggregate_compliance(sample_results)

    # L4 is never merged with ego-action L3. Publish named-actor results by
    # target class and intervention family when those queries are present.
    l4_groups: Dict[str, Dict[str, List[float]]] = {
        "target_scope": defaultdict(list), "target_class": defaultdict(list),
        "intervention_type": defaultdict(list)}
    for sample in sample_results:
        for item in (sample.get("layer4") or {}).get("results") or []:
            value = float(item.get("Score_interv", 0.0))
            target_scope = ("ego" if item.get("target_actor_id") == "ego"
                            or item.get("target_class") == "ego_vehicle"
                            else "non_ego")
            l4_groups["target_scope"][target_scope].append(value)
            l4_groups["target_class"][str(item.get("target_class", "unknown"))].append(value)
            l4_groups["intervention_type"][str(item.get("intervention_type", "unknown"))].append(value)
    summary["l4_breakdown"] = {
        group: {name: {"mean": _reduce(vals, dataset_reduction), "n": len(vals)}
                for name, vals in sorted(groups.items())}
        for group, groups in l4_groups.items()
    }

    # Actual prefix rescoring, not the node similarity at a single depth.
    horizon_rows = {}
    horizon_names = ["1s", "2s", "3s"]
    if any("6s" in (r.get("horizon_scores") or {}) for r in sample_results):
        horizon_names.append("6s")
    for horizon in horizon_names:
        for axis in ("Causal_Foresight_F1", "Counterfactual_Alignment_Score", "CTMS"):
            vals = [float((r.get("horizon_scores") or {}).get(horizon, {}).get(axis, 0.0))
                    for r in sample_results if r.get("horizon_scores")]
            horizon_rows.setdefault(horizon, {})[axis] = _reduce(vals, dataset_reduction) if vals else 0.0
    if horizon_rows:
        summary["per_horizon"] = horizon_rows

    # Keep explicit task/split labels in every report so CARLA, nuScenes, L3
    # and L4 tables cannot be accidentally pooled by downstream scripts.
    summary["reported_horizons_seconds"] = [int(name[:-1]) for name in horizon_names]

    return per_scenario, summary


# ── Report builder ───────────────────────────────────────────────────


def build_output_report(
    sample_results: List[Dict[str, Any]],
    per_scenario: List[Dict[str, Any]],
    summary: Dict[str, Any],
    metadata: Dict[str, Any],
    diagnostics: Dict[str, Any],
) -> Dict[str, Any]:
    """Assemble the final benchmark output JSON."""
    return {
        "benchmark_name": metadata.get("benchmark_name", "CORIVA"),
        "task": metadata.get("task", "full"),
        "config_version": metadata.get("config_version", "unknown"),
        "protocol_version": metadata.get("protocol_version", PROTOCOL_VERSION),
        "scorer_version": metadata.get("scorer_version", SCORER_VERSION),
        "l4_pair_protocol": metadata.get("l4_pair_protocol"),
        "track": metadata.get("track", PRIMARY_TRACK),
        "source_split": metadata.get("source_split", "unknown"),
        "condition": metadata.get("condition", "unknown"),
        "measurement_role": metadata.get(
            "reference_evaluation_role", REFERENCE_EVALUATION_ROLE),
        "primary_measurement": metadata.get("primary_measurement", PRIMARY_MEASUREMENT),
        "reference_semantics": metadata.get(
            "reference_semantics", reference_semantics(metadata.get("source_split", "unknown"))),
        "aggregation": metadata.get("aggregation", {}),
        "summary": summary,
        "per_scenario": per_scenario,
        "per_sample": sample_results,
        "diagnostics": diagnostics,
    }


# ── Main ─────────────────────────────────────────────────────────────


def main() -> int:
    parser = argparse.ArgumentParser(
        description="CORIVA Benchmark Runner — official evaluation script"
    )
    parser.add_argument("--gt", required=True, help="GT dataset JSON path")
    parser.add_argument("--pred", required=True, help="Prediction dataset JSON path")
    parser.add_argument("--allow-rescore", action="store_true",
                        help="Re-score saved 1.2 predictions with the compatible current scorer")
    parser.add_argument(
        "--config",
        default="ctms/ctms_config.yaml",
        help="ctms_config.yaml path",
    )
    parser.add_argument(
        "--ontology", default="ctms/ontology.yaml", help="ontology.yaml path"
    )
    parser.add_argument(
        "--branch-schema",
        default="ctms/schema_branch.json",
        help="schema_branch.json path",
    )
    parser.add_argument(
        "--counterfactual-schema",
        default="ctms/schema_counterfactual.json",
        help="schema_counterfactual.json path",
    )
    parser.add_argument("--output", required=True, help="Output report JSON path")
    parser.add_argument(
        "--task",
        choices=["layer3", "layer4", "full"],
        default="full",
        help="Scoring task (default: full)",
    )
    parser.add_argument(
        "--map-data", default=None,
        help=("Optional scene geometry JSON: drivable_polygons and/or "
              "static_boundary_segments in the same coordinates as actor states"),
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Reject samples with any gate failure",
    )
    parser.add_argument(
        "--fail-fast",
        action="store_true",
        help="Abort on first sample error",
    )
    args = parser.parse_args()

    # ── Load artifacts ───────────────────────────────────────────────
    print(f"Loading artifacts...")
    env = load_artifacts(args)
    config = env["config"]
    aggregation_cfg = config.get("aggregation", {})

    # ── Load datasets ────────────────────────────────────────────────
    print(f"Loading GT: {args.gt}")
    gt_dataset = load_json(args.gt)

    track = gt_dataset.get("track") or PRIMARY_TRACK
    if not isinstance(track, str):
        raise ValueError(
            "GT dataset mixes protocol tracks; score each track in a separate run")
    config.setdefault("bounds", {})["D_max"] = track_steps(track)

    print(f"Loading predictions: {args.pred}")
    pred_dataset = load_json(args.pred)
    pred_protocol = pred_dataset.get("protocol_version")
    gt_protocol = gt_dataset.get("protocol_version")
    # Dataset-level gate policy (metadata gate, protocol timeline grid);
    # score_sample re-derives it from each sample's own protocol block.
    config = env["config"] = apply_reference_gate_policy(config, gt_dataset)
    if pred_protocol not in (None, "legacy", gt_protocol):
        print("ERROR: prediction and GT protocol versions differ", file=sys.stderr)
        return 1
    pred_track = pred_dataset.get("track")
    if pred_track not in (None, track):
        print("ERROR: prediction and GT tracks differ", file=sys.stderr)
        return 1
    pred_scorer = pred_dataset.get("scorer_version")
    from ctms.ctms import RESCORABLE_SCORER_VERSIONS
    if (pred_scorer not in (None, "unknown", SCORER_VERSION)
            and not (args.allow_rescore and pred_scorer in RESCORABLE_SCORER_VERSIONS)):
        print("ERROR: prediction campaign was generated for another scorer version",
              file=sys.stderr)
        return 1

    map_data = load_json(args.map_data) if args.map_data else None

    # ── Index and align ──────────────────────────────────────────────
    pred_index = index_prediction_samples(pred_dataset)
    unmatched_preds = find_unmatched_predictions(gt_dataset, pred_index)

    # ── Score each sample ────────────────────────────────────────────
    sample_results: List[Dict[str, Any]] = []
    errors: List[Dict[str, Any]] = []
    missing_samples: List[Dict[str, str]] = []

    total = sum(
        len(s.get("samples", []))
        for s in gt_dataset.get("scenarios", [])
    )
    print(f"\nScoring {total} samples (task={args.task})...")

    for gt_data, pred_data, sid, sample_id, warnings in iter_aligned_samples(
        gt_dataset, pred_index
    ):
        if warnings:
            missing_samples.append({"scenario_id": sid, "sample_id": sample_id})

        try:
            result = score_sample(
                sample_gt=gt_data,
                sample_pred=pred_data,
                env=env,
                task=args.task,
                map_data=map_data,
            )
            horizon_scores = prefix_scores(gt_data, pred_data, env, map_data)

            # Extract top-level scores
            scores: Dict[str, float] = {}
            if args.task in ("full", "layer3"):
                scores["Causal_Foresight_F1"] = result.get(
                    "Causal_Foresight_F1", 0.0
                )
            if args.task in ("full", "layer4"):
                scores["Counterfactual_Alignment_Score"] = result.get(
                    "Counterfactual_Alignment_Score", 0.0
                )
            if args.task == "full":
                scores["CTMS"] = result.get("CTMS", 0.0)

            entry: Dict[str, Any] = {
                "scenario_id": sid,
                "sample_id": sample_id,
                "scores": scores,
                "horizon_scores": horizon_scores,
            }
            if "layer3" in result:
                entry["layer3"] = result["layer3"]
            if "layer4" in result:
                entry["layer4"] = result["layer4"]

            sample_results.append(entry)

            # Progress indicator
            status = "✓" if scores.get("CTMS", scores.get("Causal_Foresight_F1", 0)) > 0 else "✗"
            score_str = ", ".join(f"{k}={v:.4f}" for k, v in scores.items())
            print(f"  {status} {sid}/{sample_id}: {score_str}")

        except Exception as e:
            err_entry = {
                "scenario_id": sid,
                "sample_id": sample_id,
                "error": str(e),
                "traceback": traceback.format_exc(),
            }
            errors.append(err_entry)
            print(f"  ✗ {sid}/{sample_id}: ERROR — {e}", file=sys.stderr)

            if args.fail_fast:
                print("FAIL-FAST: Aborting on first error.", file=sys.stderr)
                return 1

            # Record zero-score entry so aggregation still includes this sample
            sample_results.append({
                "scenario_id": sid,
                "sample_id": sample_id,
                "scores": {"Causal_Foresight_F1": 0.0, "Counterfactual_Alignment_Score": 0.0, "CTMS": 0.0},
                "error": str(e),
            })

    # ── Aggregate ────────────────────────────────────────────────────
    per_scenario, summary = aggregate_sample_results(
        sample_results, aggregation_cfg, args.task
    )

    # ── Build report ─────────────────────────────────────────────────
    metadata = {
        "benchmark_name": config.get("benchmark_name", "CORIVA"),
        "task": args.task,
        "config_version": config.get("version", "unknown"),
        "aggregation": aggregation_cfg,
        "protocol_version": gt_dataset.get("protocol_version", PROTOCOL_VERSION),
        "scorer_version": SCORER_VERSION,
        "l4_pair_protocol": gt_dataset.get("l4_pair_protocol"),
        "track": track,
        "source_split": gt_dataset.get("source_split", "unknown"),
        "condition": pred_dataset.get("condition", "unknown"),
        "primary_measurement": gt_dataset.get("primary_measurement", PRIMARY_MEASUREMENT),
        "reference_evaluation_role": gt_dataset.get(
            "reference_evaluation_role", REFERENCE_EVALUATION_ROLE),
        "reference_semantics": gt_dataset.get(
            "reference_semantics", reference_semantics(gt_dataset.get("source_split", "unknown"))),
    }
    diagnostics = {
        "errors": errors,
        "unmatched_prediction_samples": unmatched_preds,
        "missing_prediction_samples": missing_samples,
    }

    report = build_output_report(
        sample_results, per_scenario, summary, metadata, diagnostics
    )

    # ── Write output ─────────────────────────────────────────────────
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print(f"\n{'─' * 60}")
    print(f"Report: {out_path}")
    print(f"Task:   {args.task}")
    for k, v in summary.items():
        if isinstance(v, float):
            print(f"  {k}: {v:.4f}")
        else:
            print(f"  {k}: {v}")

    if unmatched_preds:
        print(f"⚠ {len(unmatched_preds)} unmatched prediction sample(s)")
    if errors:
        # A sample that raised inside the scorer is an internal
        # error, not a benchmark outcome. The report is still written (the
        # zero entries carry `error`), but the run does not succeed, so a
        # campaign `score_cell` (check=True) stops instead of tabulating them.
        print(f"\n✗ {len(errors)} sample(s) raised inside the scorer; see diagnostics.errors",
              file=sys.stderr)
        return 2

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

# ── Example usage ────────────────────────────────────────────────────
# Full evaluation:
#   python3 run_benchmark.py --gt data/gt_dataset.json --pred data/pred_dataset.json --output results/report.json
#
# Layer 3 only:
#   python3 run_benchmark.py --gt data/gt_dataset.json --pred data/pred_dataset.json --task layer3 --output results/l3_report.json
#
# With fail-fast:
#   python3 run_benchmark.py --gt data/gt_dataset.json --pred data/pred_dataset.json --fail-fast --output results/report.json
