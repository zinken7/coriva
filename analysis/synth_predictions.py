"""
Generate synthetic model-fleet predictions from twin ground truth.

Turns each scenario's GT into l3_pre.json / l4_pre.json prediction files for a
fleet of fidelity policies (perfect / mild / moderate / heavy), in the layout
the eval pipeline expects: results/<model>/<scenario_dir>/{l3_pre,l4_pre}.json.
Used to validate the evaluation pipeline end-to-end and to demo a leaderboard
without a VLM.
"""

import argparse
import copy
import glob
import json
import os
import random
from typing import Any, Dict, List, Tuple

from analysis.rq3a import MODEL_POLICIES, jitter_branches


def _prediction_from_gt(gt: Dict[str, Any], model: str, rng: random.Random) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Build (l3_pre, l4_pre) for one scenario under a model policy."""
    branches = MODEL_POLICIES[model](gt["branches"], rng)
    l3 = {"scenario_id": gt.get("scenario_id"), "predicted_branches": branches}

    cfs: List[Dict[str, Any]] = []
    for cf in gt.get("l4_queries") or gt.get("counterfactuals", []):
        chain = copy.deepcopy(cf["counterfactual_chain"]["chain"])
        if model != "perfect":
            chain = jitter_branches([{"chain": chain}], 1.0, rng)[0]["chain"]
        unit = {"intervention": cf["counterfactual_chain"]["intervention"], "treated_chain": chain}
        if isinstance(cf.get("control_chain"), dict):
            control = copy.deepcopy(cf["control_chain"].get("chain", []))
            if model != "perfect":
                control = jitter_branches([{"chain": control}], 1.0, rng)[0]["chain"]
            unit["control_chain"] = control
        cfs.append(unit)
    l4 = {"scenario_id": gt.get("scenario_id"), "predicted_counterfactuals": cfs}
    return l3, l4


def write_fleet_predictions(gt_dir: str, results_root: str, models: List[str], seed: int = 0) -> Dict[str, int]:
    """For every scenario_* GT under gt_dir, write predictions for each model."""
    counts: Dict[str, int] = {}
    for scen_dir in sorted(glob.glob(os.path.join(gt_dir, "scenario_*"))):
        if not os.path.isdir(scen_dir):
            continue
        gt = json.load(open(os.path.join(scen_dir, "ground_truth.json")))
        name = os.path.basename(scen_dir)
        for model in models:
            rng = random.Random(f"{model}-{name}-{seed}")
            l3, l4 = _prediction_from_gt(gt, model, rng)
            out_dir = os.path.join(results_root, model, name)
            os.makedirs(out_dir, exist_ok=True)
            with open(os.path.join(out_dir, "l3_pre.json"), "w") as f:
                json.dump(l3, f, indent=2)
            with open(os.path.join(out_dir, "l4_pre.json"), "w") as f:
                json.dump(l4, f, indent=2)
            counts[model] = counts.get(model, 0) + 1
    return counts


def main():
    p = argparse.ArgumentParser(description="Write synthetic fleet predictions from twin GT")
    p.add_argument("--gt-dir", required=True, help="dir of scenario_*/ground_truth.json")
    p.add_argument("--results-root", default="results", help="output root for results/<model>/...")
    p.add_argument("--models", nargs="+", default=list(MODEL_POLICIES.keys()))
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()
    counts = write_fleet_predictions(args.gt_dir, args.results_root, args.models, args.seed)
    print("wrote predictions:", counts)


if __name__ == "__main__":
    main()
