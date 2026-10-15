"""
Kinematic Digital Twin engine (pure-Python, deterministic).

Rolls a root observed state forward under an ego intervention to produce a
branch's future actor states. Axis-aligned point-mass model: longitudinal
acceleration along x, lateral velocity along y (for swerves), speed clamped
>= 0. Testable without nuScenes or CARLA.
"""

import math
import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


def _ego(vx=20.0, x=0.0, y=0.0):
    return {"id": "ego", "entity_type": "ego_vehicle", "role_type": "ego_agent",
            "x": x, "y": y, "vx": vx, "vy": 0.0, "speed": vx}


def _npc(vx=20.0, x=40.0, y=0.0):
    return {"id": "npc_1", "entity_type": "vehicle", "role_type": "lead_vehicle",
            "x": x, "y": y, "vx": vx, "vy": 0.0, "speed": vx}


# ── step_actor ────────────────────────────────────────────────────────
def test_step_actor_brake_reduces_speed():
    from simulation.kinematic_twin import step_actor
    nxt = step_actor(_ego(vx=20.0), accel=-8.0, lat_v=0.0, dt=0.5)
    assert nxt["vx"] == pytest.approx(16.0)
    assert nxt["x"] > 0.0
    assert nxt["id"] == "ego" and nxt["role_type"] == "ego_agent"


def test_step_actor_speed_clamped_nonnegative():
    from simulation.kinematic_twin import step_actor
    nxt = step_actor(_ego(vx=2.0), accel=-8.0, lat_v=0.0, dt=0.5)
    assert nxt["vx"] == 0.0  # cannot go negative from braking


def test_step_actor_constant_velocity():
    from simulation.kinematic_twin import step_actor
    nxt = step_actor(_ego(vx=20.0, x=0.0), accel=0.0, lat_v=0.0, dt=0.5)
    assert nxt["vx"] == pytest.approx(20.0)
    assert nxt["x"] == pytest.approx(10.0)


def test_step_actor_swerve_moves_laterally():
    from simulation.kinematic_twin import step_actor
    nxt = step_actor(_ego(vx=20.0), accel=0.0, lat_v=-2.5, dt=0.5)
    assert nxt["y"] == pytest.approx(-1.25)
    assert nxt["speed"] == pytest.approx(math.sqrt(20.0 ** 2 + 2.5 ** 2), abs=1e-2)


# ── roll_branch ───────────────────────────────────────────────────────
def test_roll_branch_emergency_brake_stops_ego():
    from simulation.kinematic_twin import roll_branch
    states = roll_branch([_ego(vx=20.0), _npc()], "do_ego_emergency_brake", n_steps=6, dt=0.5)
    assert len(states) == 6
    assert states[-1]["ego"]["vx"] == 0.0  # 20 / 8 = 2.5s < 3.0s horizon


def test_roll_branch_no_action_constant():
    from simulation.kinematic_twin import roll_branch
    states = roll_branch([_ego(vx=20.0)], "do_ego_no_action", n_steps=4, dt=0.5)
    assert states[-1]["ego"]["vx"] == pytest.approx(20.0)
    assert states[-1]["ego"]["x"] == pytest.approx(40.0)  # 20 * (4*0.5)


def test_moving_ego_at_horizon_is_not_safe_stop():
    from simulation.kinematic_twin import build_twin_branch

    root = {"depth": 0, "timestamp": 0.0, "event_type": "observed_state",
            "actors": [_ego(vx=10.0)]}
    branch = build_twin_branch(
        "moving", True, "do_ego_no_action", root,
        n_steps=6, dt=0.5, sample_interval=0.5)
    assert len(branch["chain"]) == 6
    assert branch["chain"][-1]["event_type"] == "maintain_course"
    assert branch["chain"][-1].get("terminal") is not True


def test_safe_stop_requires_ego_to_actually_stop():
    from simulation.kinematic_twin import build_twin_branch

    root = {"depth": 0, "timestamp": 0.0, "event_type": "observed_state",
            "actors": [_ego(vx=2.0)]}
    branch = build_twin_branch(
        "stopping", True, "do_ego_emergency_brake", root,
        n_steps=6, dt=0.5, sample_interval=0.5)
    assert branch["chain"][-1]["event_type"] == "safe_stop"
    assert branch["chain"][-1]["terminal"] is True
    ego = branch["chain"][-1]["actors_update"][0]
    assert ego["speed"] < 0.5


def test_roll_branch_applies_env_accel_to_npc():
    from simulation.kinematic_twin import roll_branch
    states = roll_branch([_ego(vx=20.0), _npc(vx=20.0)], "do_ego_no_action",
                         n_steps=4, dt=0.5, env_accels={"npc_1": -8.0})
    assert states[-1]["npc_1"]["vx"] < 20.0  # npc braked by the environment event


def test_ego_profiles_are_ontology_valid():
    import yaml
    from simulation.kinematic_twin import EGO_PROFILES
    onto = yaml.safe_load(open(os.path.join("ctms", "ontology.yaml")))
    allowed = set()
    for g in onto["intervention_type"].values():
        allowed.update(g)
    assert set(EGO_PROFILES).issubset(allowed)


# ── build_twin_gt → must pass the QC gate ────────────────────────────────
def _twin_spec():
    return {
        "scenario_id": "twin_demo_0001",
        "scenario_type": "rear_end_collision_risk",
        "n_steps": 8, "dt": 0.5, "sample_interval": 0.5,
        "env_accels": {"npc_1": -2.0},
        "branches": [
            {"branch_id": "Factual_AEB", "is_factual": True, "intervention_type": "do_ego_emergency_brake"},
            {"branch_id": "CF_NoAction", "is_factual": False, "intervention_type": "do_ego_no_action"},
            {"branch_id": "CF_Swerve", "is_factual": False, "intervention_type": "do_ego_swerve_left"},
        ],
    }


def _twin_root():
    return {
        "depth": 0, "timestamp": 3.5, "event_type": "observed_state",
        "actors": [_ego(vx=20.0, x=0.0), _npc(vx=20.0, x=60.0)],
    }


def test_build_twin_gt_structure():
    from simulation.kinematic_twin import build_twin_gt
    gt = build_twin_gt(_twin_root(), _twin_spec())
    assert "observed_state" in gt
    assert len(gt["branches"]) == 3
    assert sum(b["is_factual"] for b in gt["branches"]) == 1
    assert len(gt["counterfactuals"]) == 2
    # A branch may legitimately remain non-terminal at the finite horizon.
    swerve = next(b for b in gt["branches"] if b["branch_id"] == "CF_Swerve")
    assert len(swerve["chain"]) == _twin_spec()["n_steps"]
    assert swerve["chain"][-1].get("terminal") is not True


def test_carla_builder_does_not_turn_moving_horizon_into_safe_stop():
    from simulation.build_ground_truth import build_chain, SCENARIOS

    config = dict(SCENARIOS[0])
    config["d_max"] = 2
    steps = []
    for index, timestamp in enumerate((3.5, 3.95, 4.4)):
        steps.append({
            "time_sec": timestamp,
            "collision": False,
            "ego": {"x": index * 5.0, "y": 0.0, "vx": 10.0, "vy": 0.0,
                    "speed_kmh": 36.0},
            "npc": {"x": 50.0 + index * 5.0, "y": 0.0, "vx": 10.0, "vy": 0.0,
                    "speed_kmh": 36.0},
        })
    meta = config["branches"]["Counterfactual_Distracted"]
    chain = build_chain(steps, config, meta)
    assert chain[-1]["event_type"] == "maintain_course"
    assert chain[-1].get("terminal") is not True


def test_twin_gt_passes_qc_gate():
    from simulation.kinematic_twin import build_twin_gt
    from simulation.qc import qc_scenario_gt
    from ctms.ctms import load_ctms_environment
    env = load_ctms_environment("ctms/ontology.yaml", "ctms/ctms_config.yaml",
                                "ctms/schema_branch.json", "ctms/schema_counterfactual.json")
    gt = build_twin_gt(_twin_root(), _twin_spec())
    ok, reasons = qc_scenario_gt(gt, env["ontology"], env["config"], env["branch_schema"])
    assert ok, reasons


# ── Sign-preserving longitudinal motion, single collision partner ────────

def test_oncoming_actor_keeps_its_negative_velocity_under_constant_motion():
    from simulation.kinematic_twin import step_actor
    nxt = step_actor({"id": "o", "entity_type": "vehicle", "role_type": "adjacent_vehicle",
                      "x": 30.0, "y": -3.0, "vx": -10.0, "vy": 0.0, "speed": 10.0}, accel=0.0, lat_v=0.0, dt=0.5)
    assert nxt["vx"] == pytest.approx(-10.0)
    assert nxt["x"] == pytest.approx(25.0)
    assert nxt["speed"] == pytest.approx(10.0)


def test_braking_an_oncoming_actor_reduces_its_speed_without_flipping_sign():
    from simulation.kinematic_twin import step_actor
    nxt = step_actor({"id": "o", "x": 30.0, "y": -3.0, "vx": -10.0, "vy": 0.0, "speed": 10.0},
                     accel=-8.0, lat_v=0.0, dt=0.5)
    assert nxt["vx"] == pytest.approx(-6.0)
    assert 25.0 < nxt["x"] < 30.0
    stopped = step_actor({"id": "o", "x": 30.0, "y": -3.0, "vx": -2.0, "vy": 0.0, "speed": 2.0},
                         accel=-8.0, lat_v=0.0, dt=0.5)
    assert stopped["vx"] == 0.0 and stopped["x"] == pytest.approx(30.0 - 0.25)


def test_collision_marks_only_the_collision_partner_as_impacted():
    from simulation.kinematic_twin import build_twin_branch
    root = {"depth": 0, "timestamp": 0.0, "event_type": "observed_state", "actors": [
        {"id": "ego", "entity_type": "ego_vehicle", "role_type": "ego_agent", "x": 0.0, "y": 0.0,
         "vx": 10.0, "vy": 0.0, "speed": 10.0, "length": 4.0, "width": 1.8},
        {"id": "npc_1", "entity_type": "vehicle", "role_type": "lead_vehicle", "x": 12.0, "y": 0.0,
         "vx": 0.0, "vy": 0.0, "speed": 0.0, "length": 4.5, "width": 1.9},
        {"id": "npc_2", "entity_type": "vehicle", "role_type": "adjacent_vehicle", "x": 50.0, "y": -5.0,
         "vx": 8.0, "vy": 0.0, "speed": 8.0, "length": 4.5, "width": 1.9}]}
    branch = build_twin_branch("b", False, "do_ego_no_action", root, 6, 0.5, 0.5)
    last = branch["chain"][-1]
    assert last["event_type"] == "collision"
    roles = {a["id"]: a["role_type"] for a in last["actors_update"]}
    assert roles["npc_1"] == "impacted_actor"
    assert roles["npc_2"] == "adjacent_vehicle"


# ── safe_stop must not hide a later in-horizon collision ─────────────────
def _ped(x, y, vy):
    return {"id": "ped_1", "entity_type": "pedestrian", "role_type": "crossing_actor",
            "x": x, "y": y, "vx": 0.0, "vy": vy, "speed": abs(vy),
            "length": 0.6, "width": 0.6}


def test_stopped_ego_hit_later_in_horizon_is_a_collision_not_safe_stop():
    from simulation.kinematic_twin import build_twin_branch

    # Ego (4.5 x 2.0) brakes from 8 m/s at -8 m/s^2: stopped by depth 2.
    # A pedestrian 4.2 m ahead walks across the lane at -1 m/s and reaches
    # the stopped ego's footprint at depth 4..6.
    ego = dict(_ego(vx=8.0), length=4.5, width=2.0)
    root = {"depth": 0, "timestamp": 0.0, "event_type": "observed_state",
            "actors": [ego, _ped(4.2, 3.0, -1.0)]}
    branch = build_twin_branch(
        "stop_then_hit", False, "do_ego_emergency_brake", root,
        n_steps=6, dt=0.5, sample_interval=0.5,
        env_behavior={"ped_1": {"accel": 0.0, "lat": -1.0}})
    events = [node["event_type"] for node in branch["chain"]]
    assert "safe_stop" not in events, events
    last = branch["chain"][-1]
    assert last["event_type"] == "collision" and last["terminal"] is True
    assert last["depth"] > 2
    impacted = [a["id"] for a in last["actors_update"] if a["role_type"] == "impacted_actor"]
    assert impacted == ["ped_1"]
    assert all("terminal" not in node for node in branch["chain"][:-1])


def test_stopped_ego_with_no_later_collision_keeps_safe_stop_at_stop_depth():
    from simulation.kinematic_twin import build_twin_branch

    ego = dict(_ego(vx=8.0), length=4.5, width=2.0)
    root = {"depth": 0, "timestamp": 0.0, "event_type": "observed_state",
            "actors": [ego, _ped(4.2, 30.0, -1.0)]}
    branch = build_twin_branch(
        "stop_only", False, "do_ego_emergency_brake", root,
        n_steps=6, dt=0.5, sample_interval=0.5,
        env_behavior={"ped_1": {"accel": 0.0, "lat": -1.0}})
    last = branch["chain"][-1]
    assert last["event_type"] == "safe_stop" and last["depth"] == 2


def test_twin_version_is_bumped_for_terminal_semantics_change():
    from simulation.kinematic_twin import TWIN_VERSION
    assert TWIN_VERSION == "kinematic-twin-5"
