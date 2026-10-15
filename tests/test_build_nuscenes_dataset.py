"""
Batch driver: nuScenes scenes -> QC-accepted CORIVA ground truths.

Runs on whatever split is present (mini for dev, trainval for the real 200).
Skips if v1.0-mini is absent.
"""

import json
import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

DATAROOT = "data/nuscenes"
MINI = os.path.join(DATAROOT, "v1.0-mini")
pytestmark = pytest.mark.skipif(
    not os.path.isdir(MINI), reason="nuScenes v1.0-mini not downloaded")


def test_build_dataset_writes_qc_passing_gts(tmp_path):
    from simulation.build_nuscenes_dataset import build_dataset
    from simulation.qc import qc_scenario_gt
    from ctms.ctms import load_ctms_environment

    env = load_ctms_environment("ctms/ontology.yaml", "ctms/ctms_config.yaml",
                                "ctms/schema_branch.json", "ctms/schema_counterfactual.json")
    result = build_dataset(DATAROOT, "v1.0-mini", limit=3, out_dir=str(tmp_path), env=env)

    assert result["accepted"] >= 1, result
    paths = result["paths"]
    assert len(paths) == result["accepted"]
    for p in paths:
        gt = json.load(open(p))
        assert "observed_state" in gt and gt["branches"]
        ok, reasons = qc_scenario_gt(gt, env["ontology"], env["config"], env["branch_schema"])
        assert ok, (p, reasons)
