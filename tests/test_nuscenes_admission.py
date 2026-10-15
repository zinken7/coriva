"""Role/state and paired canonical-twin admission for reviewed nuScenes."""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


def _ego(vx=10.0):
    return {
        "id": "ego", "entity_type": "ego_vehicle", "role_type": "ego_agent",
        "x": 0.0, "y": 0.0, "vx": vx, "vy": 0.0, "speed": vx,
        "length": 4.0, "width": 1.8,
    }


def _actor(entity="vehicle", role="lead_vehicle", x=15.0, y=0.0,
           vx=10.0, vy=0.0, length=4.5, width=2.0):
    return {
        "id": "npc_1", "entity_type": entity, "role_type": role,
        "x": x, "y": y, "vx": vx, "vy": vy,
        "speed": (vx * vx + vy * vy) ** 0.5,
        "length": length, "width": width,
    }


def test_role_state_selects_brake_only_for_moving_lead_vehicle():
    from simulation.nuscenes_admission import resolve_nuscenes_intervention

    moving = resolve_nuscenes_intervention(_actor(), "npc_sudden_brake")
    assert moving["passed"] is True
    assert moving["selected_intervention"] == "npc_sudden_brake"
    assert moving["uses_recorded_future"] is False

    static = resolve_nuscenes_intervention(_actor(vx=0.0), "npc_sudden_brake")
    assert static["passed"] is False
    assert "lead_actor_not_moving_forward_fast_enough" in static["reasons"]


def test_adjacent_cut_in_direction_is_derived_toward_ego_lane():
    from simulation.nuscenes_admission import resolve_nuscenes_intervention

    right = resolve_nuscenes_intervention(
        _actor(role="adjacent_vehicle", y=4.0), "npc_cut_in_left")
    left = resolve_nuscenes_intervention(
        _actor(role="adjacent_vehicle", y=-4.0), "npc_cut_in_right")
    assert right["passed"] is True
    assert right["resolved_motion_profile"]["lat"] < 0.0
    assert left["passed"] is True
    assert left["resolved_motion_profile"]["lat"] > 0.0

    wrong = resolve_nuscenes_intervention(
        _actor(role="adjacent_vehicle", y=4.0), "npc_sudden_brake")
    assert wrong["passed"] is False
    assert any("intervention_not_admissible_for_role_state" in reason
               for reason in wrong["reasons"])


def test_road_entry_direction_and_state_are_checked():
    from simulation.nuscenes_admission import resolve_nuscenes_intervention

    right = resolve_nuscenes_intervention(_actor(
        entity="pedestrian", role="crossing_actor", x=10.0, y=4.0,
        vx=0.0, vy=0.0, length=0.6, width=0.6))
    left = resolve_nuscenes_intervention(_actor(
        entity="pedestrian", role="crossing_actor", x=10.0, y=-4.0,
        vx=0.0, vy=0.0, length=0.6, width=0.6))
    assert right["passed"] is True and right["resolved_motion_profile"]["lat"] < 0.0
    assert left["passed"] is True and left["resolved_motion_profile"]["lat"] > 0.0

    already_in_lane = resolve_nuscenes_intervention(_actor(
        entity="pedestrian", role="crossing_actor", x=10.0, y=1.0,
        vx=0.0, vy=0.0, length=0.6, width=0.6))
    assert "road_entry_actor_already_in_ego_lane_corridor" in already_in_lane["reasons"]

    moving_away = resolve_nuscenes_intervention(_actor(
        entity="pedestrian", role="crossing_actor", x=10.0, y=4.0,
        vx=0.0, vy=1.0, length=0.6, width=0.6))
    assert "road_entry_actor_moving_away_from_ego_lane" in moving_away["reasons"]


def test_paired_effect_gate_accepts_material_relevant_brake():
    from simulation.nuscenes_admission import (
        canonical_environment_query, evaluate_nuscenes_effect,
        resolve_nuscenes_intervention)

    target = _actor(x=15.0, vx=10.0)
    root = {"actors": [_ego(vx=10.0), target]}
    applicability = resolve_nuscenes_intervention(target, "npc_sudden_brake")
    effect = evaluate_nuscenes_effect(
        root, canonical_environment_query("npc_1", applicability), model="point_mass")

    assert effect["passed"] is True
    assert effect["max_target_position_effect_m"] >= 1.0
    assert effect["max_target_speed_effect_mps"] >= 1.0
    assert effect["min_treated_ego_target_edge_gap_m"] <= 5.0
    assert effect["reference_semantics"] == "canonical_twin_applicability_not_physical_truth"


def test_paired_effect_gate_rejects_noop_and_irrelevant_interventions():
    from simulation.nuscenes_admission import evaluate_nuscenes_effect

    no_op_root = {"actors": [_ego(vx=0.0), _actor(x=5.0, vx=0.0)]}
    query = {
        "intervention_type": "npc_sudden_brake", "target_actor_id": "npc_1",
        "trigger_depth": 1, "resolved_accel_mps2": -8.0,
        "resolved_lateral_velocity_mps": 0.0,
    }
    no_op = evaluate_nuscenes_effect(no_op_root, query, model="point_mass")
    assert no_op["passed"] is False
    assert "canonical_intervention_has_no_material_target_effect" in no_op["reasons"]

    far_root = {"actors": [_ego(vx=10.0), _actor(x=100.0, vx=10.0)]}
    far = evaluate_nuscenes_effect(far_root, query, model="point_mass")
    assert far["passed"] is False
    assert "critical_actor_does_not_interact_with_ego_within_3s" in far["reasons"]


def test_generated_l4_chain_uses_resolved_direction_but_public_query_stays_schema_clean():
    from simulation.kinematic_twin import build_twin_gt
    from simulation.nuscenes_admission import (
        canonical_environment_query, observed_velocity_baseline,
        resolve_nuscenes_intervention)

    target = _actor(
        entity="pedestrian", role="crossing_actor", x=10.0, y=-4.0,
        vx=0.0, vy=0.0, length=0.6, width=0.6)
    root = {"depth": 0, "timestamp": 0.0, "event_type": "observed_state",
            "actors": [_ego(vx=5.0), target]}
    applicability = resolve_nuscenes_intervention(target)
    private_query = canonical_environment_query("npc_1", applicability)
    gt = build_twin_gt(root, {
        "scenario_id": "directional_l4",
        "n_steps": 2, "dt": 0.5, "sample_interval": 0.5,
        "branches": [{
            "branch_id": "L4_Reviewed", "is_factual": False,
            "intervention_type": applicability["selected_intervention"],
            "target_actor_id": "npc_1", "trigger_depth": 1,
            "environment_intervention": private_query,
            "env_behavior": observed_velocity_baseline(root),
        }],
    })

    chain = gt["l4_queries"][0]["counterfactual_chain"]
    intervention = chain["intervention"]
    assert intervention == {
        "intervention_type": "pedestrian_enters_road",
        "target_actor_id": "npc_1",
        "trigger_depth": 1,
    }
    target_step = next(a for a in chain["chain"][0]["actors_update"]
                       if a["id"] == "npc_1")
    assert target_step["y"] > -4.0
