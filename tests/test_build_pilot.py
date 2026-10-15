"""
Pilot packager — per scenario, emit {ground_truth.json, observed.mp4, prompt.md}
from real nuScenes so a VLM can be run with zero server. Skips without mini.
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


def test_build_prompt_contains_actor_ids_and_tasks():
    from simulation.build_pilot import build_prompt
    root = {"actors": [
        {"id": "ego", "entity_type": "ego_vehicle", "role_type": "ego_agent", "x": 0, "y": 0, "vx": 10, "vy": 0, "speed": 10},
        {"id": "npc_1", "entity_type": "vehicle", "role_type": "lead_vehicle", "x": 20, "y": 0, "vx": 8, "vy": 0, "speed": 8},
    ]}
    text = build_prompt(root, "nuscenes_0001")
    assert "npc_1" in text and "ego" in text
    assert "branches" in text and "counterfactual" in text.lower()
    assert "observed_state" in text or "observed state" in text.lower()


def test_build_pilot_packages_scenarios(tmp_path):
    from simulation.build_pilot import build_pilot
    from simulation.qc import qc_scenario_gt
    from ctms.ctms import load_ctms_environment
    env = load_ctms_environment("ctms/ontology.yaml", "ctms/ctms_config.yaml",
                                "ctms/schema_branch.json", "ctms/schema_counterfactual.json")
    result = build_pilot(DATAROOT, "v1.0-mini", limit=2, out_dir=str(tmp_path), env=env)
    assert result["accepted"] >= 1

    for scen in result["scenarios"]:
        d = scen["dir"]
        assert os.path.exists(os.path.join(d, "ground_truth.json"))
        assert os.path.exists(os.path.join(d, "prompt.md"))
        gt = json.load(open(os.path.join(d, "ground_truth.json")))
        ok, reasons = qc_scenario_gt(gt, env["ontology"], env["config"], env["branch_schema"])
        assert ok, reasons
