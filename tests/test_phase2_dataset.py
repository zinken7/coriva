"""
Twin model and fidelity gate of the nuScenes builder.

The ego twin is a bicycle model with a per-scenario root steer estimated from
the ego's PAST yaw rate, so curved scenes are followed. The steer estimate
uses only sample.prev (observed history), never the future the gate
validates against.

The builder records each scenario's factual_fidelity_error in the GT,
writes a fidelity_report.json distribution, and enforces the epsilon_max
admission gate the protocol defines (dissertation §3.5.2).
"""

import json
import math
import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

NUSCENES_ROOT = os.path.join("data", "nuscenes")
HAS_NUSCENES = os.path.isdir(os.path.join(NUSCENES_ROOT, "v1.0-mini"))


# ── root-steer estimation from past yaw rate ─────────────────────────────

def _mini_tables(yaw_prev, yaw_now, dt_us=500_000):
    """Minimal tables fixture: two keyframes with known ego yaws."""
    return {
        "sample": {
            "prev_tok": {"token": "prev_tok", "prev": "", "next": "root_tok",
                         "timestamp": 1_000_000, "scene_token": "sc"},
            "root_tok": {"token": "root_tok", "prev": "prev_tok", "next": "",
                         "timestamp": 1_000_000 + dt_us, "scene_token": "sc"},
        },
        "sample_data": {
            "sd_prev": {"sample_token": "prev_tok", "ego_pose_token": "ep_prev",
                        "is_key_frame": True, "fileformat": "jpg"},
            "sd_root": {"sample_token": "root_tok", "ego_pose_token": "ep_root",
                        "is_key_frame": True, "fileformat": "jpg"},
        },
        "ego_pose": {
            "ep_prev": {"token": "ep_prev", "translation": [0.0, 0.0, 0.0],
                        "rotation": [math.cos(yaw_prev / 2), 0, 0, math.sin(yaw_prev / 2)]},
            "ep_root": {"token": "ep_root", "translation": [5.0, 0.0, 0.0],
                        "rotation": [math.cos(yaw_now / 2), 0, 0, math.sin(yaw_now / 2)]},
        },
    }


def test_p25_estimate_root_steer_from_past_yaw_rate():
    from simulation.nuscenes_adapter import estimate_root_steer
    # yaw increases 0.1 rad over 0.5 s -> yaw_rate 0.2 rad/s at 10 m/s
    tables = _mini_tables(0.0, 0.1)
    steer = estimate_root_steer(tables, "root_tok", speed=10.0, wheelbase=2.7)
    assert steer == pytest.approx(math.atan(2.7 * 0.2 / 10.0), abs=1e-6)


def test_p25_straight_history_gives_zero_steer():
    from simulation.nuscenes_adapter import estimate_root_steer
    tables = _mini_tables(0.3, 0.3)
    assert estimate_root_steer(tables, "root_tok", speed=10.0) == pytest.approx(0.0)


def test_p25_no_prev_sample_gives_zero_steer():
    from simulation.nuscenes_adapter import estimate_root_steer
    tables = _mini_tables(0.0, 0.1)
    tables["sample"]["root_tok"]["prev"] = ""
    assert estimate_root_steer(tables, "root_tok", speed=10.0) == 0.0


def test_p25_near_zero_speed_gives_zero_steer():
    from simulation.nuscenes_adapter import estimate_root_steer
    tables = _mini_tables(0.0, 0.5)
    assert estimate_root_steer(tables, "root_tok", speed=0.1) == 0.0


# ── base_steer threads through the twin ──────────────────────────────────

_EGO = {"id": "ego", "entity_type": "ego_vehicle", "role_type": "ego_agent",
        "x": 0.0, "y": 0.0, "vx": 10.0, "vy": 0.0, "speed": 10.0}


def test_p25_roll_branch_base_steer_curves_the_ego():
    from simulation.kinematic_twin import roll_branch
    straight = roll_branch([dict(_EGO)], "do_ego_no_action", 6, 0.5, model="bicycle")
    curved = roll_branch([dict(_EGO)], "do_ego_no_action", 6, 0.5,
                         model="bicycle", base_steer=0.1)
    assert straight[-1]["ego"]["y"] == pytest.approx(0.0, abs=1e-6)
    assert abs(curved[-1]["ego"]["y"]) > 1.0


def test_p25_point_mass_ignores_base_steer():
    from simulation.kinematic_twin import roll_branch
    a = roll_branch([dict(_EGO)], "do_ego_no_action", 4, 0.5)
    b = roll_branch([dict(_EGO)], "do_ego_no_action", 4, 0.5, base_steer=0.3)
    assert a == b


def test_p25_build_twin_gt_threads_base_steer():
    from simulation.kinematic_twin import build_twin_gt
    root = {"event_type": "observed_state", "depth": 0, "timestamp": 0.0,
            "actors": [dict(_EGO)]}
    spec = {
        "scenario_id": "steer_test", "n_steps": 6, "dt": 0.5,
        "sample_interval": 1, "model": "bicycle", "base_steer": 0.1,
        "branches": [
            {"branch_id": "F", "is_factual": True, "intervention_type": "do_ego_no_action"},
            {"branch_id": "C", "is_factual": False, "intervention_type": "do_ego_emergency_brake"},
        ],
    }
    gt = build_twin_gt(root, spec)
    cf_chain = gt["counterfactuals"][0]["counterfactual_chain"]["chain"]
    ys = [a["y"] for node in cf_chain for a in node["actors_update"] if a["id"] == "ego"]
    assert any(abs(y) > 0.05 for y in ys), "counterfactual must inherit root curvature"


def test_p25_fidelity_error_accepts_model_and_steer():
    from simulation.nuscenes_adapter import factual_fidelity_error
    from simulation.kinematic_twin import roll_branch
    root = {"actors": [dict(_EGO)]}
    observed = roll_branch([dict(_EGO)], "do_ego_no_action", 6, 0.5,
                           model="bicycle", base_steer=0.1)
    e_curved = factual_fidelity_error(root, observed, "do_ego_no_action", 0.5,
                                      model="bicycle", base_steer=0.1)
    e_straight = factual_fidelity_error(root, observed, "do_ego_no_action", 0.5)
    assert e_curved == pytest.approx(0.0, abs=1e-6)
    assert e_straight > 0.5


# ── builder records epsilon, writes report, enforces the gate ────────────

@pytest.mark.skipif(not HAS_NUSCENES, reason="nuScenes v1.0-mini not present")
def test_p26_builder_records_fidelity_and_writes_report(tmp_path):
    from simulation.build_nuscenes_dataset import build_dataset
    out = build_dataset(NUSCENES_ROOT, "v1.0-mini", limit=3, out_dir=str(tmp_path),
                        epsilon_max=None)
    assert out["accepted"] == 3
    for p in out["paths"]:
        gt = json.load(open(p))
        assert "factual_fidelity_error" in gt
        assert "twin_model" in gt
    report = json.load(open(tmp_path / "fidelity_report.json"))
    assert len(report["per_scenario"]) >= 3
    assert "median" in report["summary"] and "admitted" in report["summary"]


@pytest.mark.skipif(not HAS_NUSCENES, reason="nuScenes v1.0-mini not present")
def test_p26_fidelity_gate_rejects_above_epsilon_max(tmp_path):
    from simulation.build_nuscenes_dataset import build_dataset
    out = build_dataset(NUSCENES_ROOT, "v1.0-mini", limit=5, out_dir=str(tmp_path),
                        epsilon_max=0.001)  # absurdly strict: nothing real passes
    assert out["accepted"] < 5
    assert any("fidelity_gate" in str(r.get("reasons")) for r in out["rejected_reasons"])
