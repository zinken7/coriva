"""Option B step 1: the L4 control reference audit.

nuScenes: a ground truth assembled through the real builder helpers
(`_prepare_candidate_spec`, `_append_reviewed_l4`, `build_twin_gt`,
`evaluate_nuscenes_effect`) must be reconstructed exactly from what the file
records (root, twin_model, interaction_probe.base_steer, resolved profile,
per-step effect). CARLA: the retained `_l4_control_steps` yields a control
chain; `--write` adds it without touching the treated chain.
"""
import copy
import json
import os
import shutil
import subprocess
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from scripts.audit_l4_control_references import (  # noqa: E402
    CONTROL_INTERVENTION_TYPE, audit_carla_scene, audit_nuscenes_scene, write_control)

CARLA400 = os.path.join(os.path.dirname(__file__), "..", "output_carla400_protocol10")


def _ego(vx=10.0):
    return {"id": "ego", "entity_type": "ego_vehicle", "role_type": "ego_agent",
            "x": 0.0, "y": 0.0, "vx": vx, "vy": 0.0, "speed": vx, "length": 4.0, "width": 1.8}


def _lead(x=15.0, vx=10.0):
    return {"id": "npc_1", "entity_type": "vehicle", "role_type": "lead_vehicle",
            "x": x, "y": 0.0, "vx": vx, "vy": 0.0, "speed": vx, "length": 4.5, "width": 2.0}


def _synthetic_nuscenes_gt(base_steer=0.02, twin_model="bicycle", legacy=True):
    from simulation.build_nuscenes_dataset import _append_reviewed_l4, _prepare_candidate_spec
    from simulation.kinematic_twin import TWIN_VERSION, build_twin_gt, roll_branch
    from simulation.nuscenes_admission import (
        canonical_environment_query, evaluate_nuscenes_effect, resolve_nuscenes_intervention)
    from simulation.protocol import protocol_manifest
    root = {"depth": 0, "timestamp": 0.0, "event_type": "observed_state",
            "actors": [_ego(), _lead()]}
    # A recorded future the factual branch is anchored to (not the twin).
    factual = roll_branch(root["actors"], "do_ego_no_action", 6, 0.5, model=twin_model,
                          base_steer=base_steer)
    spec = _prepare_candidate_spec("nuscenes_0001", root, factual, "do_ego_no_action", 6, 0.5,
                                   twin_model, base_steer, t0_visible_ids=["ego", "npc_1"],
                                   critical_actor_id="npc_1")
    applicability = resolve_nuscenes_intervention(_lead(), "npc_sudden_brake")
    assert applicability["passed"]
    query = canonical_environment_query("npc_1", applicability)
    _append_reviewed_l4(spec, root, query)
    gt = build_twin_gt(root, spec)
    gt["scenario_template"] = spec["scenario_template"]  # as the builder does
    gt["twin_model"] = twin_model
    gt["nuscenes_admission"] = {
        "role_state_applicability": applicability,
        "paired_effect_gate": evaluate_nuscenes_effect(
            root, query, model=twin_model, base_steer=base_steer),
    }
    gt["protocol"] = protocol_manifest(track="primary", split="v1.0-trainval")
    gt["protocol"]["twin_version"] = TWIN_VERSION
    gt["protocol"]["l4_ego_policy"] = "do_ego_no_action"
    if legacy:
        # The released nuScenes-200 predates the pair protocol: no control.
        gt["l4_queries"][0].pop("control_chain", None)
    return gt


def _write_scene(tmp_path, gt):
    d = tmp_path / "scenario_1"
    d.mkdir()
    (d / "ground_truth.json").write_text(json.dumps(gt), encoding="utf-8")
    return str(d)


def test_nuscenes_control_is_reconstructed_exactly_from_the_released_file(tmp_path):
    gt = _synthetic_nuscenes_gt()
    rec = audit_nuscenes_scene(_write_scene(tmp_path, gt))
    assert rec["status"] == "reconstructed_exact", rec["reasons"]
    assert rec["treated_reproduced"] and rec["per_step_effect_reproduced"]
    # the control target keeps its T0 speed; the treated one brakes
    assert rec["control_target_drift"]["max_abs_speed_change_mps"] == pytest.approx(0.0, abs=1e-6)
    assert rec["pair_effect"]["max_target_speed_effect_mps"] >= 1.0
    # the released treated chain is untouched by the audit
    assert json.loads((tmp_path / "scenario_1" / "ground_truth.json").read_text()) == json.loads(json.dumps(gt))


def test_native_pair_build_is_reported_as_already_exported_and_reproduced(tmp_path):
    gt = _synthetic_nuscenes_gt(legacy=False)
    assert "control_chain" in gt["l4_queries"][0]
    rec = audit_nuscenes_scene(_write_scene(tmp_path, gt))
    assert rec["status"] == "reconstructed_exact", rec["reasons"]
    assert rec["control_already_exported"] is True and rec["control_reproduced"] is True
    tampered = copy.deepcopy(gt)
    tampered["l4_queries"][0]["control_chain"]["chain"][0]["actors_update"][0]["x"] += 0.5
    (tmp_path / "t").mkdir()
    rec = audit_nuscenes_scene(_write_scene(tmp_path / "t", tampered))
    assert rec["status"] == "needs_review" and rec["control_reproduced"] is False


def test_nuscenes_audit_flags_a_file_that_cannot_be_reproduced(tmp_path):
    gt = _synthetic_nuscenes_gt()
    tampered = copy.deepcopy(gt)
    tampered["scenario_template"]["interaction_probe"]["base_steer"] = 0.05
    rec = audit_nuscenes_scene(_write_scene(tmp_path, tampered))
    assert rec["status"] == "needs_review"
    assert any("treated chain" in r for r in rec["reasons"])
    missing = copy.deepcopy(gt)
    del missing["nuscenes_admission"]["role_state_applicability"]["resolved_motion_profile"]
    (tmp_path / "b").mkdir()
    rec = audit_nuscenes_scene(_write_scene(tmp_path / "b", missing))
    assert rec["status"] == "not_recoverable"


def test_nuscenes_write_adds_control_next_to_the_unchanged_treated_chain(tmp_path):
    gt = _synthetic_nuscenes_gt()
    d = _write_scene(tmp_path, gt)
    rec = audit_nuscenes_scene(d)
    write_control(d, rec, "nuscenes")
    out = json.load(open(os.path.join(d, "ground_truth.json")))
    q = out["l4_queries"][0]
    assert q["counterfactual_chain"] == json.loads(json.dumps(gt["l4_queries"][0]["counterfactual_chain"]))
    assert q["control_chain"]["intervention"]["intervention_type"] == CONTROL_INTERVENTION_TYPE
    assert q["control_chain"]["intervention"]["target_actor_id"] == "npc_1"
    assert q["control_chain"]["intervention"]["paired_with"] == "npc_sudden_brake"
    assert q["control_chain"]["semantics"]["source"] == "deterministic_canonical_twin"
    assert len(q["control_chain"]["chain"]) == 6
    assert out["protocol"]["l4_pair_protocol"] == "matched_control_treated"
    # a second audit sees the export and still reproduces everything
    again = audit_nuscenes_scene(d)
    assert again["status"] == "reconstructed_exact"


@pytest.mark.skipif(not os.path.isdir(os.path.join(CARLA400, "scenario_58")),
                    reason="local CARLA-400 split not present")
def test_carla_control_is_stored_and_write_keeps_the_treated_chain(tmp_path):
    src = os.path.join(CARLA400, "scenario_58")
    d = str(tmp_path / "scenario_58")
    shutil.copytree(src, d, ignore=shutil.ignore_patterns("frames", "raw_frames"))
    before = json.load(open(os.path.join(d, "ground_truth.json")))
    rec = audit_carla_scene(d)
    assert rec["status"] == "stored_exact", rec["reasons"]
    assert rec["treated_reproduced"] and rec["l3_branches_reproduced"]
    write_control(d, rec, "carla")
    after = json.load(open(os.path.join(d, "ground_truth.json")))
    assert after["l4_queries"][0]["counterfactual_chain"] == before["l4_queries"][0]["counterfactual_chain"]
    assert after["branches"] == before["branches"]
    assert after["l4_queries"][0]["control_chain"]["semantics"]["source"] == "carla_paired_control_rollout"
    assert audit_carla_scene(d)["status"] == "stored_exact"


def test_cli_refuses_to_write_a_split_that_is_not_exact(tmp_path):
    gt = _synthetic_nuscenes_gt()
    gt["scenario_template"]["interaction_probe"]["base_steer"] = 0.05
    _write_scene(tmp_path, gt)
    proc = subprocess.run([sys.executable, os.path.join(os.path.dirname(__file__), "..", "scripts",
                                                        "audit_l4_control_references.py"),
                           "--split", "nuscenes", "--root", str(tmp_path), "--write"],
                          capture_output=True, text=True)
    assert proc.returncode == 2, proc.stdout + proc.stderr
    out = json.load(open(tmp_path / "scenario_1" / "ground_truth.json"))
    assert "control_chain" not in out["l4_queries"][0]
