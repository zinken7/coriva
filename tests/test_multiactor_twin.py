"""
Multi-actor environment behavior in the twin.

Non-ego actors get a per-actor behavior (longitudinal accel + lateral velocity)
so a realistic cast moves correctly: a lead vehicle brakes, a pedestrian crosses
laterally, a parked vehicle stays put — all in one rollout.
"""

import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


def _multi_actor_root_actors():
    return [
        {"id": "ego", "entity_type": "ego_vehicle", "role_type": "ego_agent",
         "x": 0.0, "y": 0.0, "vx": 18.0, "vy": 0.0, "speed": 18.0},
        {"id": "lead", "entity_type": "vehicle", "role_type": "lead_vehicle",
         "x": 40.0, "y": 0.0, "vx": 18.0, "vy": 0.0, "speed": 18.0},
        {"id": "ped", "entity_type": "pedestrian", "role_type": "crossing_actor",
         "x": 25.0, "y": 6.0, "vx": 0.0, "vy": 0.0, "speed": 0.0},
        {"id": "parked", "entity_type": "vehicle", "role_type": "adjacent_vehicle",
         "x": 15.0, "y": -4.0, "vx": 0.0, "vy": 0.0, "speed": 0.0},
    ]


def test_env_behavior_drives_each_actor():
    from simulation.kinematic_twin import roll_branch
    behavior = {
        "lead": {"accel": -4.0, "lat": 0.0},     # lead brakes
        "ped": {"accel": 0.0, "lat": -2.0},      # pedestrian crosses (toward the lane)
        "parked": {"accel": 0.0, "lat": 0.0},    # parked stays
    }
    states = roll_branch(_multi_actor_root_actors(), "do_ego_no_action",
                         n_steps=4, dt=0.5, env_behavior=behavior)
    last = states[-1]
    assert last["lead"]["vx"] < 18.0                 # lead decelerated
    assert last["ped"]["y"] < 6.0 and last["ped"]["vy"] == pytest.approx(-2.0)  # ped moved laterally
    assert last["parked"]["x"] == pytest.approx(15.0)  # parked did not move
    assert last["parked"]["speed"] == pytest.approx(0.0)


def test_env_accels_still_supported():
    """Backward compat: the old {id: accel} form keeps working."""
    from simulation.kinematic_twin import roll_branch
    states = roll_branch(_multi_actor_root_actors(), "do_ego_no_action",
                         n_steps=3, dt=0.5, env_accels={"lead": -3.0})
    assert states[-1]["lead"]["vx"] < 18.0


def test_multiactor_twin_gt_passes_qc():
    from simulation.kinematic_twin import build_twin_gt
    from simulation.qc import qc_scenario_gt
    from ctms.ctms import load_ctms_environment
    env = load_ctms_environment("ctms/ontology.yaml", "ctms/ctms_config.yaml",
                                "ctms/schema_branch.json", "ctms/schema_counterfactual.json")
    root = {"depth": 0, "timestamp": 0.0, "event_type": "observed_state",
            "actors": _multi_actor_root_actors()}
    spec = {
        "scenario_id": "multi_0001", "scenario_type": "mixed_urban",
        "n_steps": 6, "dt": 0.5, "sample_interval": 0.5,
        "env_behavior": {"lead": {"accel": -4.0, "lat": 0.0}, "ped": {"accel": 0.0, "lat": -1.5}},
        "branches": [
            {"branch_id": "Factual_AEB", "is_factual": True, "intervention_type": "do_ego_emergency_brake"},
            {"branch_id": "CF_NoAction", "is_factual": False, "intervention_type": "do_ego_no_action"},
            {"branch_id": "CF_Swerve", "is_factual": False, "intervention_type": "do_ego_swerve_left"},
        ],
    }
    gt = build_twin_gt(root, spec)
    ok, reasons = qc_scenario_gt(gt, env["ontology"], env["config"], env["branch_schema"])
    assert ok, reasons
    # All four actors are present at every node (closed-world preserved).
    for b in gt["branches"]:
        ids = {a["id"] for a in b["chain"][0]["actors_update"]}
        assert ids == {"ego", "lead", "ped", "parked"}, ids


# ── C#5.4 — static street furniture + front/rear collisions ───────────
def _rich_root_actors():
    return [
        {"id": "ego", "entity_type": "ego_vehicle", "role_type": "ego_agent",
         "x": 0.0, "y": 0.0, "vx": 16.0, "vy": 0.0, "speed": 16.0},
        {"id": "lead", "entity_type": "vehicle", "role_type": "lead_vehicle",
         "x": 38.0, "y": 0.0, "vx": 16.0, "vy": 0.0, "speed": 16.0},
        {"id": "park_l", "entity_type": "parked_vehicle", "role_type": "adjacent_vehicle",
         "x": 25.0, "y": 5.2, "vx": 0.0, "vy": 0.0, "speed": 0.0},
        {"id": "park_r", "entity_type": "parked_vehicle", "role_type": "adjacent_vehicle",
         "x": 18.0, "y": -5.2, "vx": 0.0, "vy": 0.0, "speed": 0.0},
        {"id": "wall_l", "entity_type": "guardrail", "role_type": "road_boundary",
         "x": 30.0, "y": 7.5, "vx": 0.0, "vy": 0.0, "speed": 0.0},
        {"id": "sign_1", "entity_type": "traffic_sign", "role_type": "static_barrier",
         "x": 45.0, "y": 6.0, "vx": 0.0, "vy": 0.0, "speed": 0.0},
    ]


def test_static_furniture_stays_put_and_passes_qc():
    from simulation.kinematic_twin import build_twin_gt
    from simulation.qc import qc_scenario_gt
    from ctms.ctms import load_ctms_environment
    env = load_ctms_environment("ctms/ontology.yaml", "ctms/ctms_config.yaml",
                                "ctms/schema_branch.json", "ctms/schema_counterfactual.json")
    root = {"depth": 0, "timestamp": 0.0, "event_type": "observed_state", "actors": _rich_root_actors()}
    spec = {"scenario_id": "rich_0001", "scenario_type": "urban", "n_steps": 6, "dt": 0.5,
            "sample_interval": 0.5, "env_behavior": {"lead": {"accel": -4.0, "lat": 0.0}},
            "branches": [
                {"branch_id": "Factual_AEB", "is_factual": True, "intervention_type": "do_ego_emergency_brake"},
                {"branch_id": "CF_NoAction", "is_factual": False, "intervention_type": "do_ego_no_action"},
                {"branch_id": "CF_Swerve", "is_factual": False, "intervention_type": "do_ego_swerve_left"}]}
    gt = build_twin_gt(root, spec)
    ok, reasons = qc_scenario_gt(gt, env["ontology"], env["config"], env["branch_schema"])
    assert ok, reasons
    # Static furniture must not move across the chain.
    for b in gt["branches"]:
        last = {a["id"]: a for a in b["chain"][-1]["actors_update"]}
        for sid, sx, sy in [("park_l", 25.0, 5.2), ("wall_l", 30.0, 7.5), ("sign_1", 45.0, 6.0)]:
            assert last[sid]["x"] == pytest.approx(sx) and last[sid]["y"] == pytest.approx(sy)


def test_rear_vehicle_collides_from_behind():
    from simulation.kinematic_twin import build_twin_gt
    root = {"depth": 0, "timestamp": 0.0, "event_type": "observed_state", "actors": [
        {"id": "ego", "entity_type": "ego_vehicle", "role_type": "ego_agent",
         "x": 0.0, "y": 0.0, "vx": 15.0, "vy": 0.0, "speed": 15.0},
        {"id": "tailgater", "entity_type": "vehicle", "role_type": "rear_vehicle",
         "x": -8.0, "y": 0.0, "vx": 15.0, "vy": 0.0, "speed": 15.0}]}
    spec = {"scenario_id": "rear_0001", "scenario_type": "rear_hit", "n_steps": 6, "dt": 0.5,
            "sample_interval": 0.5, "env_behavior": {"tailgater": {"accel": 4.0, "lat": 0.0}},
            "branches": [
                {"branch_id": "Factual_NoAction", "is_factual": True, "intervention_type": "do_ego_no_action"}]}
    gt = build_twin_gt(root, spec)
    chain = gt["branches"][0]["chain"]
    last = chain[-1]
    assert last["event_type"] == "collision", [n["event_type"] for n in chain]
    roles = {a["id"]: a["role_type"] for a in last["actors_update"]}
    assert roles["tailgater"] == "impacted_actor"  # rear vehicle is the collision partner
