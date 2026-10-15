"""
Integration — assemble_dataset + run_benchmark must run end-to-end on
twin ground truth and discriminate a synthetic model fleet (perfect > heavy).
Self-contained (synthetic GTs; no nuScenes). Exercises the real CLI scripts.
"""

import json
import os
import subprocess
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


def test_phase_b_discriminates_fleet(tmp_path):
    from analysis.rq3a import make_synthetic_gts
    from analysis.synth_predictions import write_fleet_predictions

    gt_dir = tmp_path / "gt"
    for i, gt in enumerate(make_synthetic_gts(5), start=1):
        # Mirror the real builder layout: dir scenario_N <-> scenario_id digits N.
        gt["scenario_id"] = f"nuscenes_{i:04d}"
        d = gt_dir / f"scenario_{i}"
        d.mkdir(parents=True)
        with open(d / "ground_truth.json", "w") as f:
            json.dump(gt, f)

    results_root = tmp_path / "results"
    write_fleet_predictions(str(gt_dir), str(results_root), models=["perfect", "heavy"], seed=0)

    ctms = {}
    for model in ("perfect", "heavy"):
        gt_ds = tmp_path / f"{model}_gt.json"
        pred_ds = tmp_path / f"{model}_pred.json"
        report = tmp_path / f"{model}_report.json"
        subprocess.run([sys.executable, "scripts/assemble_dataset.py",
                        "--gt-dir", str(gt_dir), "--pred-dir", str(results_root / model),
                        "--out-gt", str(gt_ds), "--out-pred", str(pred_ds)],
                       cwd=ROOT, check=True, capture_output=True)
        subprocess.run([sys.executable, "run_benchmark.py",
                        "--gt", str(gt_ds), "--pred", str(pred_ds), "--output", str(report)],
                       cwd=ROOT, check=True, capture_output=True)
        rep = json.load(open(report))
        ctms[model] = rep["summary"]["CTMS_mean"]

    assert ctms["perfect"] > 0, ctms
    assert ctms["perfect"] > ctms["heavy"], ctms
