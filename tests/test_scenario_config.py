"""
Unified scenario config is the single source of truth.

The YAML configs under scenarios/ must deserialize to exactly the scenario
definitions in build_ground_truth.SCENARIOS.
"""

import pytest
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


def test_loaded_configs_match_hardcoded_scenarios():
    from simulation.build_ground_truth import SCENARIOS
    from simulation.scenario_config import load_scenario_configs

    loaded = load_scenario_configs("scenarios")
    assert loaded == SCENARIOS, "YAML configs must reproduce SCENARIOS exactly"


def test_configs_ordered_by_scenario_index():
    from simulation.scenario_config import load_scenario_configs

    loaded = load_scenario_configs("scenarios")
    indices = [c["scenario_index"] for c in loaded]
    assert indices == sorted(indices), "configs must be ordered by scenario_index"


def test_resolve_scenarios_uses_yaml_when_present():
    from simulation.build_ground_truth import resolve_scenarios, SCENARIOS
    assert resolve_scenarios("scenarios") == SCENARIOS


def test_resolve_scenarios_falls_back_when_dir_missing():
    from simulation.build_ground_truth import resolve_scenarios, SCENARIOS
    assert resolve_scenarios("nonexistent_dir_xyz") == SCENARIOS


def test_scenario_actions_flip_only_the_lateral_action_for_a_left_side_hazard():
    from simulation.protocol import (SCENARIO_ACTIONS, critical_lateral_offset,
                                     scenario_actions)
    right = scenario_actions("pedestrian_collision_risk", 6.0)
    left = scenario_actions("pedestrian_collision_risk", -6.0)
    assert right == tuple(SCENARIO_ACTIONS["pedestrian_collision_risk"])
    assert left == ("do_ego_comfortable_brake", "do_ego_panic_brake", "do_ego_swerve_right")
    # only the lateral entry differs
    assert [a for a, b in zip(right, left) if a != b] == ["do_ego_swerve_left"]
    # an actor near the ego centreline, an unknown side, and a cut-in set are unchanged
    assert scenario_actions("rear_end_collision_risk", 0.0) == tuple(SCENARIO_ACTIONS["rear_end_collision_risk"])
    assert scenario_actions("cyclist_collision_risk", None) == tuple(SCENARIO_ACTIONS["cyclist_collision_risk"])
    assert scenario_actions("cut_in_collision_risk", -3.5) == tuple(SCENARIO_ACTIONS["cut_in_collision_risk"])
    assert scenario_actions("no_such_type", -5.0) == ()
    root = {"actors": [{"id": "ego", "y": 0.0}, {"id": "npc_1", "y": -4.2}]}
    assert critical_lateral_offset(root, "npc_1") == -4.2
    assert critical_lateral_offset(root, "missing") is None


def test_branch_contract_expects_the_side_aware_action_set():
    from analysis.branch_audit import check_branch_contract
    gt = {"scenario_id": "s", "scenario_type": "pedestrian_collision_risk",
          "observed_state": {"actors": [{"id": "ego", "y": 0.0}, {"id": "npc_1", "y": -5.0}]},
          "branches": [
              {"is_factual": True,
               "intervention": {"intervention_type": "do_ego_comfortable_brake"}},
              {"is_factual": False, "intervention": {"intervention_type": "do_ego_panic_brake"}},
              {"is_factual": False, "intervention": {"intervention_type": "do_ego_swerve_right"}}]}
    query = {"l3_ego_actions": ["do_ego_comfortable_brake", "do_ego_panic_brake", "do_ego_swerve_right"],
             "l4_interventions": [{"target_actor_id": "npc_1"}]}
    result = check_branch_contract(gt, query, "carla")
    assert result["expected_actions"] == sorted(query["l3_ego_actions"])
    assert result["checks"]["references_match_protocol"] is True
    assert result["checks"]["query_matches_protocol"] is True


def test_curved_route_conflict_between_declared_side_and_published_root():
    """The lateral action is declared from the template's route offset before
    the rollout; on a curve the realised T0 offset can land on the other side.
    A published scene whose lateral escape steers toward its own critical
    actor must be rejected."""
    from simulation.protocol import lateral_action_conflict
    left_hazard = {"actors": [{"id": "ego", "y": 0.0}, {"id": "npc_1", "y": -5.6}]}
    right_hazard = {"actors": [{"id": "ego", "y": 0.0}, {"id": "npc_1", "y": 4.3}]}
    ahead = {"actors": [{"id": "ego", "y": 0.0}, {"id": "npc_1", "y": -0.7}]}
    assert lateral_action_conflict(left_hazard, "npc_1", ["do_ego_swerve_right"]) is None
    assert lateral_action_conflict(right_hazard, "npc_1", ["do_ego_swerve_left"]) is None
    assert lateral_action_conflict(ahead, "npc_1", ["do_ego_swerve_left"]) is None
    for root, action in ((left_hazard, "do_ego_swerve_left"),
                         (right_hazard, "do_ego_swerve_right"),
                         (ahead, "do_ego_swerve_right")):
        reason = lateral_action_conflict(root, "npc_1", ["do_ego_panic_brake", action])
        assert reason and action in reason and "steers toward" in reason
    # nothing to check without a lateral action or without the actor
    assert lateral_action_conflict(left_hazard, "npc_1", ["do_ego_no_action"]) is None
    assert lateral_action_conflict(left_hazard, "missing", ["do_ego_swerve_left"]) is None


def test_nuscenes_lateral_side_follows_the_l3_hazard_not_the_l4_target():
    """The L3 template's hazard and the actor reviewed for the L4 query can be
    different actors; the lateral escape must follow the L3 hazard, and the
    audit contract must read the same field."""
    from analysis.branch_audit import check_branch_contract
    from simulation.protocol import critical_lateral_offset, scenario_actions
    root = {"actors": [{"id": "ego", "y": 0.0},
                       {"id": "npc_ped", "y": -4.5},    # L3 hazard, on the left
                       {"id": "npc_lead", "y": 0.2}]}   # reviewed for L4, ahead
    spec_template = {"selected_actor_id": "npc_ped"}
    lateral_actor = spec_template.get("selected_actor_id") or "npc_lead"
    assert scenario_actions("pedestrian_collision_risk",
                            critical_lateral_offset(root, lateral_actor))[2] == "do_ego_swerve_right"
    gt = {"scenario_id": "s", "scenario_type": "pedestrian_collision_risk",
          "observed_state": root, "scenario_template": spec_template,
          "branches": [
              {"is_factual": True, "intervention": {"intervention_type": "do_ego_comfortable_brake"}},
              {"is_factual": False, "intervention": {"intervention_type": "do_ego_panic_brake"}},
              {"is_factual": False, "intervention": {"intervention_type": "do_ego_swerve_right"}}]}
    query = {"l3_ego_actions": ["do_ego_comfortable_brake", "do_ego_panic_brake", "do_ego_swerve_right"],
             "l4_interventions": [{"target_actor_id": "npc_lead"}]}
    result = check_branch_contract(gt, query, "nuscenes")
    assert result["checks"]["references_match_protocol"] is True


def test_generate_configs_runs_as_a_script_and_as_a_module():
    """steps_full.md C4 calls the file directly; the simulation import must not
    break that entry point."""
    import subprocess, sys, pathlib
    root = pathlib.Path(__file__).resolve().parents[1]
    for cmd in ([sys.executable, str(root / 'scenarios/generate_configs.py'), '--help'],
                [sys.executable, '-m', 'scenarios.generate_configs', '--help']):
        out = subprocess.run(cmd, cwd=root, capture_output=True, text=True)
        assert out.returncode == 0, out.stderr
        assert 'usage: generate_configs.py' in out.stdout


def _ns_root(ped_y=-4.5):
    """A root whose L3 hazard (pedestrian entering from the ego's left) is a
    different actor from the one reviewed for the L4 query (the lead ahead)."""
    return {"depth": 0, "timestamp": 0.0, "event_type": "observed_state", "actors": [
        {"id": "ego", "entity_type": "ego_vehicle", "role_type": "ego_agent",
         "x": 0.0, "y": 0.0, "vx": 8.0, "vy": 0.0, "speed": 8.0, "length": 4.0, "width": 1.8},
        {"id": "lead", "entity_type": "vehicle", "role_type": "lead_vehicle",
         "x": 22.0, "y": 0.3, "vx": 7.0, "vy": 0.0, "speed": 7.0, "length": 4.5, "width": 1.9},
        {"id": "ped", "entity_type": "pedestrian", "role_type": "crossing_actor",
         "x": 11.0, "y": ped_y, "vx": 0.0, "vy": 1.4, "speed": 1.4, "length": 0.6, "width": 0.6}]}


@pytest.mark.parametrize("twin_model", ["point_mass", "bicycle"])
def test_builder_takes_the_lateral_side_from_the_l3_hazard_not_the_l4_actor(twin_model):
    """Real builder path: `_prepare_candidate_spec` must not recompute the
    action set from `critical_actor_id`, the actor reviewed for the L4 query,
    which would overwrite the side the L3 template has already chosen from
    its own hazard. Here the hazard is a pedestrian on the ego's left
    while the reviewed actor is the lead straight ahead, so the two choices
    differ: left hazard wants swerve_right, the lead would give swerve_left."""
    from simulation.build_nuscenes_dataset import _prepare_candidate_spec
    from simulation.protocol import critical_lateral_offset, lateral_escape_action
    root = _ns_root()
    spec = _prepare_candidate_spec(
        "s1", root, None, "do_ego_no_action", 6, 0.5, twin_model, 0.0,
        t0_visible_ids=["ego", "lead", "ped"], critical_actor_id="lead")
    assert spec["scenario_type"] == "pedestrian_collision_risk"
    assert (spec.get("scenario_template") or {}).get("selected_actor_id") == "ped"
    actions = [b["intervention_type"] for b in spec["branches"]]
    assert "do_ego_swerve_right" in actions and "do_ego_swerve_left" not in actions
    # the two actors really do disagree, so the assertion above has teeth
    assert lateral_escape_action(critical_lateral_offset(root, "ped")) == "do_ego_swerve_right"
    assert lateral_escape_action(critical_lateral_offset(root, "lead")) == "do_ego_swerve_left"


def test_builder_and_audit_contract_agree_on_the_lateral_side():
    """The published scene and the audit must read the same field, otherwise a
    correctly built scene fails its own contract check."""
    from analysis.branch_audit import check_branch_contract
    from simulation.build_nuscenes_dataset import _prepare_candidate_spec
    root = _ns_root()
    spec = _prepare_candidate_spec(
        "s1", root, None, "do_ego_no_action", 6, 0.5, "bicycle", 0.0,
        t0_visible_ids=["ego", "lead", "ped"], critical_actor_id="lead")
    actions = [b["intervention_type"] for b in spec["branches"]]
    gt = {"scenario_id": "s1", "scenario_type": spec["scenario_type"],
          "observed_state": root, "scenario_template": spec["scenario_template"],
          "branches": [{"is_factual": b["is_factual"],
                        "intervention": {"intervention_type": b["intervention_type"],
                                         "target_actor_id": "ego", "trigger_depth": 1}}
                       for b in spec["branches"]]}
    query = {"l3_ego_actions": actions, "l4_interventions": [{"target_actor_id": "lead"}]}
    result = check_branch_contract(gt, query, "nuscenes")
    assert result["status"] == "pass", result
    assert result["checks"]["references_match_protocol"] is True
