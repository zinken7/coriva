"""
Opt-in kinematic bicycle model for the twin (curved swerves).
The roll_branch default is point-mass; bicycle is selected via model="bicycle".
"""

import math
import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


def _ego(vx=20.0, vy=0.0, x=0.0, y=0.0):
    return {"id": "ego", "entity_type": "ego_vehicle", "role_type": "ego_agent",
            "x": x, "y": y, "vx": vx, "vy": vy, "speed": math.hypot(vx, vy)}


def test_bicycle_straight_matches_point_mass_shape():
    from simulation.kinematic_twin import step_actor_bicycle
    nxt = step_actor_bicycle(_ego(vx=20.0), accel=0.0, steer=0.0, dt=0.5)
    assert nxt["x"] == pytest.approx(10.0)
    assert nxt["y"] == pytest.approx(0.0, abs=1e-6)
    assert nxt["vx"] == pytest.approx(20.0, abs=1e-2)
    assert nxt["vy"] == pytest.approx(0.0, abs=1e-6)
    assert set(nxt) == {"id", "entity_type", "role_type", "x", "y", "vx", "vy", "speed"}


def test_bicycle_steer_turns_heading():
    from simulation.kinematic_twin import step_actor_bicycle
    nxt = step_actor_bicycle(_ego(vx=20.0), accel=0.0, steer=0.15, dt=0.5)
    assert nxt["vy"] < -0.5          # positive steer turns left (-y in CORIVA)
    assert nxt["vx"] < 20.0          # speed redirected, x-component drops
    assert nxt["speed"] == pytest.approx(20.0, abs=1e-2)  # speed ~conserved (no accel)


def test_bicycle_swerve_changes_one_lane_then_holds():
    """kinematic-twin-5: the action adds a constant lateral velocity in the
    fixed frame, capped at one lane width, on top of the road-following
    bicycle path (no steering angle for the action any more)."""
    from simulation.kinematic_twin import roll_branch
    from simulation.protocol import EGO_SWERVE_LANE_WIDTH_M, EGO_SWERVE_SPEED_MPS
    states = roll_branch([_ego(vx=20.0)], "do_ego_swerve_left", n_steps=6, dt=0.5, model="bicycle")
    ys = [s["ego"]["y"] for s in states]
    assert ys[:3] == pytest.approx([-1.25, -2.5, -3.5])
    assert ys[3:] == pytest.approx([-EGO_SWERVE_LANE_WIDTH_M] * 3)
    vys = [s["ego"]["vy"] for s in states]
    assert vys[:2] == pytest.approx([-EGO_SWERVE_SPEED_MPS] * 2) and vys[2:] == pytest.approx([0.0] * 4, abs=1e-6)


def test_bicycle_swerve_keeps_the_road_curvature_from_base_steer():
    """With a curved road (base_steer) the swerve offset is measured from the
    road-following path: swerve minus neutral is exactly the lane change."""
    from simulation.kinematic_twin import roll_branch
    neutral = roll_branch([_ego(vx=15.0)], "do_ego_no_action", n_steps=6, dt=0.5, model="bicycle", base_steer=0.03)
    swerve = roll_branch([_ego(vx=15.0)], "do_ego_swerve_right", n_steps=6, dt=0.5, model="bicycle", base_steer=0.03)
    offsets = [s["ego"]["y"] - n["ego"]["y"] for s, n in zip(swerve, neutral)]
    assert offsets == pytest.approx([1.25, 2.5, 3.5, 3.5, 3.5, 3.5])
    assert neutral[-1]["ego"]["y"] != pytest.approx(0.0)  # the road itself curves


def test_bicycle_swerve_names_follow_published_lateral_axis():
    from simulation.kinematic_twin import roll_branch
    left = roll_branch([_ego(vx=20.0)], "do_ego_swerve_left", n_steps=3,
                       dt=0.5, model="bicycle")
    right = roll_branch([_ego(vx=20.0)], "do_ego_swerve_right", n_steps=3,
                        dt=0.5, model="bicycle")
    assert left[-1]["ego"]["y"] < 0.0
    assert right[-1]["ego"]["y"] > 0.0


def test_bicycle_twin_gt_passes_qc():
    from simulation.kinematic_twin import build_twin_gt
    from simulation.qc import qc_scenario_gt
    from ctms.ctms import load_ctms_environment
    env = load_ctms_environment("ctms/ontology.yaml", "ctms/ctms_config.yaml",
                                "ctms/schema_branch.json", "ctms/schema_counterfactual.json")
    root = {"depth": 0, "timestamp": 0.0, "event_type": "observed_state",
            "actors": [_ego(vx=18.0), {"id": "npc_1", "entity_type": "vehicle",
                       "role_type": "lead_vehicle", "x": 55.0, "y": 0.0, "vx": 18.0, "vy": 0.0, "speed": 18.0}]}
    spec = {"scenario_id": "bike_0001", "scenario_type": "rear_end_collision_risk",
            "n_steps": 6, "dt": 0.5, "sample_interval": 0.5, "env_accels": {"npc_1": -2.0}, "model": "bicycle",
            "branches": [{"branch_id": "Factual_AEB", "is_factual": True, "intervention_type": "do_ego_emergency_brake"},
                         {"branch_id": "CF_Swerve", "is_factual": False, "intervention_type": "do_ego_swerve_left"},
                         {"branch_id": "CF_NoAction", "is_factual": False, "intervention_type": "do_ego_no_action"}]}
    gt = build_twin_gt(root, spec)
    ok, reasons = qc_scenario_gt(gt, env["ontology"], env["config"], env["branch_schema"])
    assert ok, reasons
