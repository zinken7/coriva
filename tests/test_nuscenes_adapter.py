"""
nuScenes ingestion adapter (parses JSON tables directly; no devkit).

Skips automatically if the v1.0-mini dataset is not present under data/nuscenes.
Verifies that a real nuScenes sample is turned into a CORIVA observed_state root
that, fed through the kinematic twin, yields a QC-accepted ground truth.
"""

import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

DATAROOT = "data/nuscenes"
MINI = os.path.join(DATAROOT, "v1.0-mini")
pytestmark = pytest.mark.skipif(
    not os.path.isdir(MINI), reason="nuScenes v1.0-mini not downloaded")

ALLOWED_ENTITIES = {"ego_vehicle", "vehicle", "truck", "bus", "motorcycle", "bicycle",
                    "pedestrian", "barrier", "traffic_cone"}


def _first_usable_sample(tables):
    """A sample that has a successor (for velocity) and at least one annotation."""
    anns_by_sample = {}
    for a in tables["sample_annotation"].values():
        anns_by_sample.setdefault(a["sample_token"], 0)
        anns_by_sample[a["sample_token"]] += 1
    for s in tables["sample"].values():
        if s["next"] and anns_by_sample.get(s["token"], 0) > 0:
            return s["token"]
    raise AssertionError("no usable sample found")


def test_extract_observed_state_from_real_scene():
    from simulation.nuscenes_adapter import load_tables, extract_observed_state
    tables = load_tables(DATAROOT, "v1.0-mini")
    sample_token = _first_usable_sample(tables)
    root = extract_observed_state(tables, sample_token, max_actors=3, radius=60.0)

    assert root["event_type"] == "observed_state" and root["depth"] == 0
    ego = next(a for a in root["actors"] if a["role_type"] == "ego_agent")
    assert ego["id"] == "ego" and ego["entity_type"] == "ego_vehicle"
    assert ego["x"] == pytest.approx(0.0) and ego["y"] == pytest.approx(0.0)
    assert ego["vx"] >= 0.0 and abs(ego["vy"]) < 1.0  # ego placed facing +x
    assert len(root["actors"]) >= 2
    for a in root["actors"]:
        assert a["entity_type"] in ALLOWED_ENTITIES, a
        for f in ("id", "entity_type", "role_type", "x", "y", "vx", "vy", "speed"):
            assert f in a


def test_twin_gt_from_real_root_passes_qc():
    from simulation.nuscenes_adapter import load_tables, extract_observed_state
    from simulation.kinematic_twin import build_twin_gt
    from simulation.qc import qc_scenario_gt
    from ctms.ctms import load_ctms_environment

    tables = load_tables(DATAROOT, "v1.0-mini")
    root = extract_observed_state(tables, _first_usable_sample(tables), max_actors=3, radius=60.0)
    npc_ids = [a["id"] for a in root["actors"] if a["role_type"] != "ego_agent"]

    spec = {
        "scenario_id": "nuscenes_twin_0001",
        "scenario_type": "rear_end_collision_risk",
        "n_steps": 6, "dt": 0.5, "sample_interval": 0.5,
        "env_accels": {npc_ids[0]: -1.0} if npc_ids else {},
        "branches": [
            {"branch_id": "Factual_AEB", "is_factual": True, "intervention_type": "do_ego_emergency_brake"},
            {"branch_id": "CF_NoAction", "is_factual": False, "intervention_type": "do_ego_no_action"},
            {"branch_id": "CF_Swerve", "is_factual": False, "intervention_type": "do_ego_swerve_left"},
        ],
    }
    gt = build_twin_gt(root, spec)
    env = load_ctms_environment("ctms/ontology.yaml", "ctms/ctms_config.yaml",
                                "ctms/schema_branch.json", "ctms/schema_counterfactual.json")
    ok, reasons = qc_scenario_gt(gt, env["ontology"], env["config"], env["branch_schema"])
    assert ok, reasons
