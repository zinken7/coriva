"""Final input-condition contract: video, history_json, and video_history."""

import copy
import json
import os
import sys

import pytest
import yaml

ROOT = os.path.join(os.path.dirname(__file__), "..")
sys.path.insert(0, ROOT)

from analysis.rq3a import make_synthetic_gts
from simulation.protocol import HISTORY_TIMESTAMPS, protocol_manifest


def _protocol_gt_dir(tmp_path):
    gt_dir = tmp_path / "gt"
    gt = make_synthetic_gts(1)[0]
    gt["scenario_id"] = "nuscenes_0001"
    root = gt["observed_state"]
    root["timestamp"] = 0.0
    for actor in root["actors"]:
        actor.setdefault("length", 4.5 if actor["id"] == "ego" else 4.0)
        actor.setdefault("width", 2.0)
    d = gt_dir / "scenario_1"
    (d / "frames").mkdir(parents=True)
    for index in range(8):
        (d / "frames" / f"{index:03d}.jpg").write_bytes(b"\xff\xd8fake")
    records = []
    for timestamp in HISTORY_TIMESTAMPS:
        actors = []
        for actor in root["actors"]:
            item = copy.deepcopy(actor)
            item["visibility"] = actor["id"] != "ego"
            actors.append(item)
        records.append({"timestamp_sec": timestamp, "actors": actors})
    history = {"schema_version": 1, "coordinate_frame": "t0_ego", "records": records}
    (d / "ground_truth.json").write_text(json.dumps(gt))
    (d / "observed_state.json").write_text(json.dumps(root))
    (d / "observed_history.json").write_text(json.dumps(history))
    actions = [(branch.get("intervention") or {}).get("intervention_type")
               for branch in gt["branches"]
               if (branch.get("intervention") or {}).get("target_actor_id", "ego") == "ego"]
    query = {"l3_ego_actions": [action for action in actions if action],
             "l4_interventions": [], "l4_ego_policy": "do_ego_no_action"}
    (d / "query.json").write_text(json.dumps(query))
    with open(os.path.join(ROOT, "ctms", "ontology.yaml")) as f:
        ontology = yaml.safe_load(f)
    (d / "ontology.json").write_text(json.dumps(ontology))
    (d / "input_manifest.json").write_text(json.dumps(protocol_manifest(split="v1.0-mini")))
    return gt_dir


def test_three_conditions_supply_exact_modalities(tmp_path):
    from analysis.run_model import run_model, BACKENDS
    seen = []

    def spy(prompt, frame_paths, model_id, opts):
        seen.append((prompt, list(frame_paths)))
        return json.dumps({"predicted_branches": [], "predicted_counterfactuals": []})

    BACKENDS["spy"] = spy
    try:
        gt_dir = _protocol_gt_dir(tmp_path)
        run_model(str(gt_dir), str(tmp_path / "video"), backend="spy", condition="video")
        run_model(str(gt_dir), str(tmp_path / "history"), backend="spy",
                  condition="history_json")
        run_model(str(gt_dir), str(tmp_path / "both"), backend="spy",
                  condition="video_history")
    finally:
        del BACKENDS["spy"]

    assert len(seen[0][1]) == 8 and "## observed_history" not in seen[0][0]
    assert seen[1][1] == [] and "## observed_history" in seen[1][0]
    assert len(seen[2][1]) == 8 and "## observed_history" in seen[2][0]


def test_protocol_manifest_rejects_legacy_root_only(tmp_path):
    from analysis.run_model import run_model

    with pytest.raises(ValueError, match="not released"):
        run_model(str(_protocol_gt_dir(tmp_path)), str(tmp_path / "legacy"),
                  backend="mock", condition="root_only")
