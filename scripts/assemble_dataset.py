"""
assemble_dataset.py — Bundle per-scenario GT/pred files into dataset-level JSON.

Creates the dataset format that run_benchmark.py expects:
  - gt_dataset.json: all ground truth scenarios + samples
  - pred_dataset.json: all prediction scenarios + samples (merged L3 + L4)

Usage:
    python3 scripts/assemble_dataset.py \\
        --gt-dir output \\
        --pred-dir results/gemini \\
        --out-gt data/gt_dataset.json \\
        --out-pred data/pred_dataset.json
"""

from __future__ import annotations

import json
import argparse
import os
import glob
import sys
from typing import Optional

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from simulation.protocol import (PRIMARY_MEASUREMENT, REFERENCE_EVALUATION_ROLE,
                                 reference_semantics)


def _int_from_digits(s: str) -> Optional[int]:
    """Parse leading digit run as int (e.g. 'scenario_001' → 1). None if no digits."""
    digits = "".join(c for c in s if c.isdigit())
    return int(digits) if digits else None


def discover_scenarios(base_dir: str, pattern: str = "scenario_*") -> list:
    """Find all scenario directories matching pattern."""
    dirs = sorted(glob.glob(os.path.join(base_dir, pattern)))
    return [d for d in dirs if os.path.isdir(d)]


def load_json_safe(path: str) -> Optional[dict]:
    """Load JSON file, return None if not found."""
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)


def assemble_gt_dataset(gt_dir: str, dataset_name: str = "CORIVA-dev") -> dict:
    """Bundle per-scenario ground_truth.json files into one dataset."""
    scenarios = []

    for scenario_dir in discover_scenarios(gt_dir):
        gt_path = os.path.join(scenario_dir, "ground_truth.json")
        gt = load_json_safe(gt_path)
        if gt is None:
            print(f"  [WARN] No ground_truth.json in {scenario_dir}, skipping")
            continue

        scenario_id = gt.get("scenario_id", os.path.basename(scenario_dir))

        # Wrap GT as a single sample per scenario
        sample = {
            "sample_id": f"{scenario_id}_sample_001",
            "ground_truth": gt,
        }

        scenarios.append({
            "scenario_id": scenario_id,
            # Directory basename: the only pairing key shared with the
            # prediction root. CARLA ids (lead_car_brake_0000 for scenario_1)
            # do not carry the directory number, so digit matching mispairs.
            "scenario_dir": os.path.basename(scenario_dir),
            "samples": [sample],
        })

        print(f"  ✓ GT: {scenario_id} ({len(gt.get('branches', []))} branches, "
              f"{len(gt.get('l4_queries', gt.get('counterfactuals', [])))} L4 queries)")

    protocols = [s["samples"][0]["ground_truth"].get("protocol", {})
                 for s in scenarios if s.get("samples")]
    versions = {p.get("protocol_version") for p in protocols if p.get("protocol_version")}
    if len(versions) > 1:
        raise ValueError(f"mixed protocol versions are not scoreable: {sorted(versions)}")
    protocol_version = next(iter(versions), None)
    pair_protocols = {p.get("l4_pair_protocol", "single_treated_legacy") for p in protocols}
    if len(pair_protocols) > 1:
        raise ValueError(f"mixed L4 pair protocols are not scoreable: {sorted(pair_protocols)}")
    tracks = sorted({p.get("track") for p in protocols if p.get("track")})
    if len(tracks) > 1:
        raise ValueError(f"mixed tracks are not scoreable: {tracks}")
    splits = sorted({p.get("split") or p.get("dataset") for p in protocols
                     if p.get("split") or p.get("dataset")})
    if len(splits) > 1:
        raise ValueError(f"mixed source splits are not scoreable: {splits}")
    source_split = splits[0] if splits else "unknown"
    primary_measurements = {p.get("primary_measurement") for p in protocols
                            if p.get("primary_measurement")}
    reference_roles = {p.get("reference_evaluation_role") for p in protocols
                       if p.get("reference_evaluation_role")}
    reference_meanings = {p.get("reference_semantics") for p in protocols
                          if p.get("reference_semantics")}
    for name, values in (("primary measurements", primary_measurements),
                         ("reference roles", reference_roles),
                         ("reference semantics", reference_meanings)):
        if len(values) > 1:
            raise ValueError(f"mixed {name} are not scoreable: {sorted(values)}")
    return {
        "dataset_name": dataset_name,
        "protocol_version": protocol_version,
        "l4_pair_protocol": next(iter(pair_protocols), None),
        "track": tracks[0] if len(tracks) == 1 else None,
        "source_split": source_split,
        "primary_measurement": next(iter(primary_measurements), PRIMARY_MEASUREMENT),
        "reference_evaluation_role": next(
            iter(reference_roles), REFERENCE_EVALUATION_ROLE),
        "reference_semantics": next(
            iter(reference_meanings), reference_semantics(source_split)),
        "scenarios": scenarios,
    }



def merged_prediction(l3, l4):
    """One scorer-facing prediction from the parsed L3 and L4 files. The
    scorer reads `branches`; a unit container that is not an array (a
    number, a string, null) is an empty container, so the sample scores as
    having no units of that task instead of stopping the assembly."""
    prediction = {}
    if isinstance(l3, dict):
        branches = l3.get("predicted_branches", l3.get("branches"))
        prediction["branches"] = branches if isinstance(branches, list) else []
    if isinstance(l4, dict):
        cfs = l4.get("predicted_counterfactuals")
        prediction["predicted_counterfactuals"] = cfs if isinstance(cfs, list) else []
    return prediction

def assemble_pred_dataset(pred_dir: str, gt_dataset: dict,
                          dataset_name: str = "CORIVA-dev") -> dict:
    """Bundle per-scenario L3+L4 prediction files into one dataset.

    Matches predictions to GT scenarios. L3 and L4 files are merged into
    one prediction object per sample.
    """
    scenarios = []
    run_manifests = []

    for scenario_index, gt_scenario in enumerate(gt_dataset["scenarios"]):
        scenario_id = gt_scenario["scenario_id"]

        # Find matching prediction directory: by directory name when the GT
        # dataset records it, otherwise the digit/index heuristics.
        pred_scenario_dir = find_pred_dir(
            pred_dir, scenario_id, scenario_index,
            scenario_dir=gt_scenario.get("scenario_dir"))
        if pred_scenario_dir is None:
            print(f"  [WARN] No prediction dir for {scenario_id}")
            scenarios.append({
                "scenario_id": scenario_id,
                "samples": [],
            })
            continue

        # Load L3 and L4 prediction files
        l3 = load_json_safe(os.path.join(pred_scenario_dir, "l3_pre.json"))
        l4 = load_json_safe(os.path.join(pred_scenario_dir, "l4_pre.json"))
        run_manifest = load_json_safe(os.path.join(pred_scenario_dir, "run_manifest.json"))
        if run_manifest:
            run_manifests.append(run_manifest)

        prediction = merged_prediction(l3, l4)

        sample_id = gt_scenario["samples"][0]["sample_id"] if gt_scenario["samples"] else f"{scenario_id}_sample_001"

        samples = [{
            "sample_id": sample_id,
            "prediction": prediction,
        }]

        scenarios.append({
            "scenario_id": scenario_id,
            "samples": samples,
        })

        n_branches = len(prediction.get("branches") or [])
        n_cfs = len(prediction.get("predicted_counterfactuals") or [])
        print(f"  ✓ Pred: {scenario_id} ({n_branches} branches, {n_cfs} CFs)")

    protocol_versions = {m.get("protocol_version") for m in run_manifests}
    conditions = {m.get("condition") for m in run_manifests}
    tracks = {m.get("scorer_input_track") for m in run_manifests}
    scorer_versions = {m.get("scorer_version") for m in run_manifests}
    parser_versions = {m.get("parser_version") for m in run_manifests}
    if (len(protocol_versions) > 1 or len(conditions) > 1 or len(tracks) > 1
            or len(scorer_versions) > 1 or len(parser_versions) > 1):
        raise ValueError("prediction directory mixes protocol, condition, track, parser, or scorer")
    panel_fields = (
        "model_panel_version", "configuration_name", "model_id",
        "model_revision", "adapter", "inference_mode", "max_output_tokens",
        "smoke_admission_version", "smoke_admission_sha256",
        "smoke_code_commit",
    )
    panel_values = {
        field: {manifest.get(field) for manifest in run_manifests}
        for field in panel_fields
    }
    has_panel = [manifest.get("model_panel_version") is not None
                 for manifest in run_manifests]
    if has_panel and any(has_panel) and not all(has_panel):
        raise ValueError("prediction directory mixes legacy and model-panel runs")
    if any(has_panel):
        mixed = {field: values for field, values in panel_values.items()
                 if len(values) != 1 or
                 (field != "model_revision" and None in values)}
        if mixed:
            raise ValueError(
                f"prediction directory mixes model configuration metadata: {mixed}")
        adapters = panel_values["adapter"]
        revisions = panel_values["model_revision"]
        if adapters == {"gemini_api"} and revisions != {None}:
            raise ValueError("Gemini alias must not declare an immutable model revision")
        if adapters != {"gemini_api"} and revisions == {None}:
            raise ValueError("open model prediction directory is missing model revision")
    provider_version_values = [
        (manifest.get("generation_metadata") or {}).get("model_version")
        for manifest in run_manifests
    ]
    provider_versions = {value for value in provider_version_values if value}
    adapters = panel_values.get("adapter", set())
    if any(has_panel) and adapters == {"gemini_api"} and \
            (not provider_version_values or not all(provider_version_values)):
        raise ValueError("Gemini prediction directory is missing provider model versions")
    if len(provider_versions) > 1:
        raise ValueError(
            f"prediction directory mixes provider model versions: {provider_versions}")
    expected_protocol = gt_dataset.get("protocol_version")
    if (run_manifests and expected_protocol is not None
            and protocol_versions != {expected_protocol}):
        raise ValueError("prediction protocol version does not match GT dataset")
    return {
        "dataset_name": dataset_name,
        "protocol_version": next(iter(protocol_versions), "legacy"),
        "condition": next(iter(conditions), "unknown"),
        "track": next(iter(tracks), gt_dataset.get("track")),
        "scorer_version": next(iter(scorer_versions), "unknown"),
        "parser_version": next(iter(parser_versions), "unknown"),
        "model_panel_version": next(
            iter(panel_values.get("model_panel_version", set())), None),
        "configuration_name": next(
            iter(panel_values.get("configuration_name", set())), None),
        "model_id": next(iter(panel_values.get("model_id", set())), None),
        "model_revision": next(
            iter(panel_values.get("model_revision", set())), None),
        "adapter": next(iter(panel_values.get("adapter", set())), None),
        "inference_mode": next(
            iter(panel_values.get("inference_mode", set())), None),
        "max_output_tokens": next(
            iter(panel_values.get("max_output_tokens", set())), None),
        "smoke_admission_version": next(
            iter(panel_values.get("smoke_admission_version", set())), None),
        "smoke_admission_sha256": next(
            iter(panel_values.get("smoke_admission_sha256", set())), None),
        "smoke_code_commit": next(
            iter(panel_values.get("smoke_code_commit", set())), None),
        "provider_model_version": next(iter(provider_versions), None),
        "primary_measurement": gt_dataset.get("primary_measurement", PRIMARY_MEASUREMENT),
        "reference_evaluation_role": gt_dataset.get(
            "reference_evaluation_role", REFERENCE_EVALUATION_ROLE),
        "reference_semantics": gt_dataset.get(
            "reference_semantics", reference_semantics(gt_dataset.get("source_split", "unknown"))),
        "scenarios": scenarios,
    }


def find_pred_dir(pred_base: str, scenario_id: str, scenario_index: int = 0,
                  scenario_dir: Optional[str] = None) -> str | None:
    """Find prediction directory for a scenario.

    When ``scenario_dir`` (the GT scenario's directory basename) is given, the
    prediction directory is ``pred_base/scenario_dir`` and nothing else: the
    runner writes predictions under the same basename as the scenario it read,
    so this is exact. Digit heuristics mispair CARLA scenarios, whose ids
    (``lead_car_brake_0000`` under ``scenario_1``) count from zero. Returns
    None when that directory has no prediction files.

    Order without ``scenario_dir`` (a gt.json that does not record it):
    1. Directory basename digits equal ``scenario_id`` digits (e.g. ``*_2`` ↔ ``...2...``).
    2. Exact basename match to ``scenario_id``.
    3. If ``scenario_id`` has no digits, align by ``scenario_index`` with sorted
       ``scenario_*`` dirs that contain ``l3_pre.json`` or ``l4_pre.json`` (same order
       as :func:`assemble_gt_dataset` uses for ``output/scenario_*``).

    Note: ``'001'.lstrip('0')`` is ``'1'``, not ``''`` — do not compare stripped strings.
    """
    if scenario_dir is not None:
        exact = os.path.join(pred_base, scenario_dir)
        if os.path.isdir(exact) and (os.path.exists(os.path.join(exact, "l3_pre.json")) or
                                     os.path.exists(os.path.join(exact, "l4_pre.json"))):
            return exact
        return None
    candidates = []
    for d in discover_scenarios(pred_base):
        if (os.path.exists(os.path.join(d, "l3_pre.json")) or
                os.path.exists(os.path.join(d, "l4_pre.json"))):
            candidates.append(d)
    candidates.sort()

    num_gt = _int_from_digits(scenario_id)

    if num_gt is not None:
        for d in candidates:
            if _int_from_digits(os.path.basename(d)) == num_gt:
                return d

    for d in candidates:
        if os.path.basename(d) == scenario_id:
            return d

    if num_gt is None and scenario_index < len(candidates):
        return candidates[scenario_index]

    return None


def main():
    parser = argparse.ArgumentParser(
        description='Assemble per-scenario files into dataset-level JSON')
    parser.add_argument('--gt-dir', required=True,
                        help='Directory containing scenario_*/ground_truth.json')
    parser.add_argument('--pred-dir', required=True,
                        help='Directory containing scenario_*/l3_pre.json + l4_pre.json')
    parser.add_argument('--out-gt', required=True,
                        help='Output GT dataset JSON path')
    parser.add_argument('--out-pred', required=True,
                        help='Output prediction dataset JSON path')
    parser.add_argument('--dataset-name', default='CORIVA-dev',
                        help='Dataset name for metadata')
    args = parser.parse_args()

    print("Assembling GT dataset...")
    gt_dataset = assemble_gt_dataset(args.gt_dir, args.dataset_name)

    print(f"\nAssembling prediction dataset from {args.pred_dir}...")
    pred_dataset = assemble_pred_dataset(args.pred_dir, gt_dataset, args.dataset_name)

    os.makedirs(os.path.dirname(args.out_gt) or '.', exist_ok=True)
    with open(args.out_gt, 'w') as f:
        json.dump(gt_dataset, f, indent=2)
    print(f"\n✓ GT dataset: {args.out_gt} ({len(gt_dataset['scenarios'])} scenarios)")

    os.makedirs(os.path.dirname(args.out_pred) or '.', exist_ok=True)
    with open(args.out_pred, 'w') as f:
        json.dump(pred_dataset, f, indent=2)
    print(f"✓ Pred dataset: {args.out_pred} ({len(pred_dataset['scenarios'])} scenarios)")


if __name__ == '__main__':
    main()
