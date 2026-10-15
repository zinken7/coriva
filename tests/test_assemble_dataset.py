"""Prediction/GT pairing in scripts.assemble_dataset.

CARLA scenario ids count from zero (``lead_car_brake_0000`` lives in
``scenario_1``), so pairing by the digits of the id would assign predictions
to the neighbouring scene and fail every gate. The assembler pairs by the
scenario directory basename recorded in gt.json.
"""
import json

import pytest

from scripts.assemble_dataset import (assemble_gt_dataset, assemble_pred_dataset,
                                      find_pred_dir)


def _write_split(tmp_path, ids):
    gt_dir = tmp_path / "gt"
    pred_dir = tmp_path / "pred"
    from analysis.rq3a import make_synthetic_gts
    for dirname, (scenario_id, gt) in zip(ids, zip(ids.values(), make_synthetic_gts(len(ids)))):
        gt["scenario_id"] = scenario_id
        d = gt_dir / dirname
        d.mkdir(parents=True)
        (d / "ground_truth.json").write_text(json.dumps(gt))
        p = pred_dir / dirname
        p.mkdir(parents=True)
        # A prediction that names its own directory, so mispairing is visible.
        (p / "l3_pre.json").write_text(json.dumps({"predicted_branches": [{"from": dirname}]}))
        (p / "l4_pre.json").write_text(json.dumps({"predicted_counterfactuals": []}))
    return gt_dir, pred_dir


CARLA_IDS = {"scenario_1": "lead_car_brake_0000", "scenario_2": "lead_large_vehicle_brake_0001",
             "scenario_3": "motorcycle_cut_in_right_0002"}
NUSCENES_IDS = {"scenario_104": "nuscenes_0104", "scenario_119": "nuscenes_0119"}


@pytest.mark.parametrize("ids", [CARLA_IDS, NUSCENES_IDS])
def test_gt_dataset_records_scenario_dir_and_predictions_pair_by_it(tmp_path, ids):
    gt_dir, pred_dir = _write_split(tmp_path, ids)
    gt = assemble_gt_dataset(str(gt_dir))
    assert [s["scenario_dir"] for s in gt["scenarios"]] == sorted(ids)
    pred = assemble_pred_dataset(str(pred_dir), gt)
    for gt_scenario, pred_scenario in zip(gt["scenarios"], pred["scenarios"]):
        assert pred_scenario["scenario_id"] == gt_scenario["scenario_id"]
        branches = pred_scenario["samples"][0]["prediction"]["branches"]
        assert branches == [{"from": gt_scenario["scenario_dir"]}]


def test_missing_prediction_dir_is_reported_not_borrowed(tmp_path):
    gt_dir, pred_dir = _write_split(tmp_path, CARLA_IDS)
    import shutil
    shutil.rmtree(pred_dir / "scenario_2")
    gt = assemble_gt_dataset(str(gt_dir))
    pred = assemble_pred_dataset(str(pred_dir), gt)
    by_id = {s["scenario_id"]: s for s in pred["scenarios"]}
    assert by_id["lead_large_vehicle_brake_0001"]["samples"] == []
    assert by_id["lead_car_brake_0000"]["samples"][0]["prediction"]["branches"] == [{"from": "scenario_1"}]


def test_find_pred_dir_exact_when_scenario_dir_given(tmp_path):
    _, pred_dir = _write_split(tmp_path, CARLA_IDS)
    assert find_pred_dir(str(pred_dir), "lead_car_brake_0000", 0, scenario_dir="scenario_1") \
        == str(pred_dir / "scenario_1")
    assert find_pred_dir(str(pred_dir), "lead_car_brake_0000", 0, scenario_dir="scenario_9") is None


def test_find_pred_dir_legacy_digits_without_scenario_dir(tmp_path):
    _, pred_dir = _write_split(tmp_path, NUSCENES_IDS)
    assert find_pred_dir(str(pred_dir), "nuscenes_0104") == str(pred_dir / "scenario_104")
    # The digit heuristic mispairs CARLA; it applies only to gt.json files
    # without scenario_dir.
    _, carla_pred = _write_split(tmp_path / "carla", CARLA_IDS)
    assert find_pred_dir(str(carla_pred), "lead_car_brake_0000") is None
