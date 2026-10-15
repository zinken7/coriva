"""Anchor the FACTUAL branch to the real observed nuScenes
future, keep the twin only for (unobservable) counterfactual branches.

Rationale: Pearl rung-3 outcomes cannot be observed, so a transparent kinematic
model is the honest way to define them. But the FACTUAL branch *is* observed —
so it must be built from the recorded future trajectory, not a point-mass
rollout. `factual_fidelity_error` is the supporting (b) sanity check: how far the
twin drifts from reality on the one branch where reality exists.
"""

import math
import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

DATAROOT = "data/nuscenes"
MINI = os.path.join(DATAROOT, "v1.0-mini")
_has_mini = os.path.isdir(MINI)


# ── pure: infer the factual ego action from observed longitudinal speed ──
def test_infer_factual_action_comfortable_brake():
    from simulation.nuscenes_adapter import infer_factual_action
    # 20 -> 12 over (5-1)*0.5 = 2.0 s => -4 m/s^2
    assert infer_factual_action([20.0, 18.0, 16.0, 14.0, 12.0], dt=0.5) == "do_ego_comfortable_brake"


def test_infer_factual_action_emergency_brake():
    from simulation.nuscenes_adapter import infer_factual_action
    assert infer_factual_action([20.0, 10.0], dt=0.5) == "do_ego_emergency_brake"  # -20


def test_infer_factual_action_no_action_when_steady():
    from simulation.nuscenes_adapter import infer_factual_action
    assert infer_factual_action([20.0, 20.0, 20.0], dt=0.5) == "do_ego_no_action"


def test_infer_factual_action_accelerate():
    from simulation.nuscenes_adapter import infer_factual_action
    assert infer_factual_action([10.0, 13.0, 16.0], dt=0.5) == "do_ego_accelerate_pass"  # +6


# ── pure: build_twin_branch honors override_states (uses reality, not twin) ──
def _root2():
    ego = {"id": "ego", "entity_type": "ego_vehicle", "role_type": "ego_agent",
           "x": 0.0, "y": 0.0, "vx": 20.0, "vy": 0.0, "speed": 20.0}
    npc = {"id": "npc_1", "entity_type": "vehicle", "role_type": "lead_vehicle",
           "x": 40.0, "y": 0.0, "vx": 20.0, "vy": 0.0, "speed": 20.0}
    return {"depth": 0, "timestamp": 0.0, "event_type": "observed_state", "actors": [ego, npc]}


def test_build_twin_branch_uses_override_states():
    from simulation.kinematic_twin import build_twin_branch
    # Observed reality: ego crawls to x=1,2 (NOT the 10,20 a 20 m/s point-mass would give).
    observed = [
        {"ego": {"id": "ego", "entity_type": "ego_vehicle", "role_type": "ego_agent",
                 "x": 1.0, "y": 0.0, "vx": 2.0, "vy": 0.0, "speed": 2.0},
         "npc_1": {"id": "npc_1", "entity_type": "vehicle", "role_type": "lead_vehicle",
                   "x": 41.0, "y": 0.0, "vx": 2.0, "vy": 0.0, "speed": 2.0}},
        {"ego": {"id": "ego", "entity_type": "ego_vehicle", "role_type": "ego_agent",
                 "x": 2.0, "y": 0.0, "vx": 0.0, "vy": 0.0, "speed": 0.0},
         "npc_1": {"id": "npc_1", "entity_type": "vehicle", "role_type": "lead_vehicle",
                   "x": 42.0, "y": 0.0, "vx": 0.0, "vy": 0.0, "speed": 0.0}},
    ]
    branch = build_twin_branch("Factual", True, "do_ego_comfortable_brake", _root2(),
                               n_steps=2, dt=0.5, sample_interval=0.5, override_states=observed)
    # First chain node's ego x must be the OBSERVED 1.0, not a twin 10.0.
    ego0 = next(a for a in branch["chain"][0]["actors_update"] if a["id"] == "ego")
    assert ego0["x"] == pytest.approx(1.0)


def test_fidelity_error_zero_when_twin_matches_observed():
    from simulation.kinematic_twin import roll_branch
    from simulation.nuscenes_adapter import factual_fidelity_error
    root = _root2()
    # Observed == what the twin itself produces under no_action -> error must be ~0.
    twin_states = roll_branch(root["actors"], "do_ego_no_action", n_steps=3, dt=0.5)
    err = factual_fidelity_error(root, twin_states, "do_ego_no_action", dt=0.5)
    assert err == pytest.approx(0.0, abs=1e-6)


def test_root_and_future_npc_velocities_share_past_only_t0_frame():
    from simulation.nuscenes_adapter import extract_observed_trajectory

    half = math.sqrt(0.5)
    yaw_90 = [half, 0.0, 0.0, half]
    samples = {
        "s-prev": {"token": "s-prev", "prev": "", "next": "s0", "timestamp": 0},
        "s0": {"token": "s0", "prev": "s-prev", "next": "s1", "timestamp": 500_000},
        "s1": {"token": "s1", "prev": "s0", "next": "", "timestamp": 1_000_000},
    }
    poses = {
        "p-prev": {"translation": [0.0, -1.0, 0.0], "rotation": yaw_90},
        "p0": {"translation": [0.0, 0.0, 0.0], "rotation": yaw_90},
        "p1": {"translation": [0.0, 1.0, 0.0], "rotation": yaw_90},
    }
    annotations = {
        "a-prev": {"token": "a-prev", "sample_token": "s-prev", "instance_token": "inst",
                   "prev": "", "next": "a0", "translation": [9.0, 0.0, 0.0],
                   "size": [2.0, 4.0, 1.5], "rotation": [1.0, 0.0, 0.0, 0.0]},
        "a0": {"token": "a0", "sample_token": "s0", "instance_token": "inst",
               "prev": "a-prev", "next": "a1", "translation": [10.0, 0.0, 0.0],
               "size": [2.0, 4.0, 1.5], "rotation": [1.0, 0.0, 0.0, 0.0]},
        "a1": {"token": "a1", "sample_token": "s1", "instance_token": "inst",
               "prev": "a0", "next": "", "translation": [11.5, 0.0, 0.0],
               "size": [2.0, 4.0, 1.5], "rotation": [1.0, 0.0, 0.0, 0.0]},
    }
    tables = {
        "sample": samples,
        "ego_pose": poses,
        "sample_data": {
            "sd-prev": {"sample_token": "s-prev", "ego_pose_token": "p-prev", "is_key_frame": True},
            "sd0": {"sample_token": "s0", "ego_pose_token": "p0", "is_key_frame": True},
            "sd1": {"sample_token": "s1", "ego_pose_token": "p1", "is_key_frame": True},
        },
        "sample_annotation": annotations,
        "instance": {"inst": {"category_token": "cat"}},
        "category": {"cat": {"name": "vehicle.car"}},
    }

    trajectory = extract_observed_trajectory(tables, "s0", n_steps=1, max_actors=1)
    root_npc = next(a for a in trajectory["root"]["actors"] if a["id"] != "ego")
    future_npc = trajectory["factual_states"][0][root_npc["id"]]

    # Global +x becomes T0-frame +y (right) at yaw=+90 degrees. Root uses a-prev→a0
    # (2 m/s), not the leaked a0→a1 future (3 m/s); future uses a0→a1.
    assert root_npc["x"] == pytest.approx(0.0, abs=1e-6)
    assert root_npc["y"] == pytest.approx(10.0)
    assert root_npc["vx"] == pytest.approx(0.0, abs=1e-6)
    assert root_npc["vy"] == pytest.approx(2.0)
    assert future_npc["vx"] == pytest.approx(0.0, abs=1e-6)
    assert future_npc["vy"] == pytest.approx(3.0)


# ── real nuScenes mini: the factual branch is anchored to reality ──
@pytest.mark.skipif(not _has_mini, reason="nuScenes v1.0-mini not downloaded")
def test_factual_branch_anchored_to_observed_future():
    from simulation.nuscenes_adapter import load_tables, extract_observed_trajectory
    from simulation.kinematic_twin import build_twin_gt, build_twin_branch
    from simulation.qc import qc_scenario_gt
    from ctms.ctms import load_ctms_environment

    tables = load_tables(DATAROOT, "v1.0-mini")
    # Second keyframe of the first scene with enough forward keyframes: the
    # root velocity is derived from the previous keyframe, so the T0 -> first
    # future transition is a real one (the scorer checks it against the root).
    sample_token = None
    for scene in tables["scene"].values():
        tok = tables["sample"][scene["first_sample_token"]]["next"]
        if tok and tables["sample"][tok]["next"]:
            sample_token = tok
            break
    assert sample_token

    traj = extract_observed_trajectory(tables, sample_token, n_steps=6, max_actors=3, radius=60.0)
    root, factual_states, action = traj["root"], traj["factual_states"], traj["factual_action"]
    assert len(factual_states) >= 1
    assert action.startswith("do_ego_")

    spec = {
        "scenario_id": "nuscenes_anchor_0001", "scenario_type": "generic_following",
        "n_steps": 6, "dt": 0.5, "sample_interval": 0.5,
        "factual_states": factual_states, "factual_action": action,
        "branches": [
            {"branch_id": "Factual_Observed", "is_factual": True, "intervention_type": action},
            {"branch_id": "CF_Brake", "is_factual": False, "intervention_type": "do_ego_emergency_brake"},
            {"branch_id": "CF_Swerve", "is_factual": False, "intervention_type": "do_ego_swerve_left"},
        ],
    }
    gt = build_twin_gt(root, spec)

    # The factual branch's first node must equal the OBSERVED first future state.
    fb = next(b for b in gt["branches"] if b["is_factual"])
    ego_node = next(a for a in fb["chain"][0]["actors_update"] if a["id"] == "ego")
    obs_ego = factual_states[0]["ego"]
    assert ego_node["x"] == pytest.approx(obs_ego["x"]) and ego_node["y"] == pytest.approx(obs_ego["y"])

    # A counterfactual branch is still a twin rollout (emergency brake => ego decelerates).
    cf = next(b for b in gt["branches"] if b["branch_id"] == "CF_Brake")
    assert cf["chain"][0]["actors_update"]  # non-empty

    env = load_ctms_environment("ctms/ontology.yaml", "ctms/ctms_config.yaml",
                                "ctms/schema_branch.json", "ctms/schema_counterfactual.json")
    ok, reasons = qc_scenario_gt(gt, env["ontology"], env["config"], env["branch_schema"])
    assert ok, reasons
