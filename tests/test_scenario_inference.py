"""nuScenes L3 scenario-template selection (policy ``interaction-v3``).

Category alone never drives the L3 template: a pedestrian or cyclist that is
behind the ego, invisible at T0 or unable to reach the ego corridor is not a
hazard and is not given synthetic crossing motion, so actors retained for the
eight-frame history cannot hijack the T0 scene type. Candidates are actors
visible at T0 that pass the L4 role/state rule, ranked by the interaction
half of the L4 paired-effect gate: under its published intervention profile
and the fixed no-action ego policy the actor must come within the interaction
edge gap of the ego inside the horizon; urgency is the first half-second step
at which that happens, so a lead pulling away at speed or an in-corridor
pedestrian 100 m ahead is not relevant. The reviewed critical actor breaks
ties. Non-template actors keep their observed velocity.
"""

import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from simulation.nuscenes_adapter import (SCENARIO_TEMPLATE_POLICY,  # noqa: E402
                                         infer_scenario_spec, t0_visible_actor_ids)


def _ego(speed=15.0):
    return {"id": "ego", "entity_type": "ego_vehicle", "role_type": "ego_agent",
            "x": 0, "y": 0, "vx": speed, "vy": 0, "speed": speed, "length": 4.0, "width": 1.8}


def _root(*actors, speed=15.0):
    return {"depth": 0, "timestamp": 0.0, "event_type": "observed_state",
            "actors": [_ego(speed), *actors]}


def _lead(x=25.0, vx=12.0, aid="lead"):
    return {"id": aid, "entity_type": "vehicle", "role_type": "lead_vehicle",
            "x": x, "y": 0.3, "vx": vx, "vy": 0.0, "speed": vx, "length": 4.5, "width": 1.9}


def _ped(x=12.0, y=4.0, vx=0.0, vy=0.0, aid="ped", entity="pedestrian"):
    return {"id": aid, "entity_type": entity, "role_type": "crossing_actor",
            "x": x, "y": y, "vx": vx, "vy": vy, "speed": (vx * vx + vy * vy) ** 0.5,
            "length": 0.6, "width": 0.6}


def test_policy_is_versioned_and_recorded():
    spec = infer_scenario_spec("s", _root(_lead()))
    assert SCENARIO_TEMPLATE_POLICY == "interaction-v3"
    assert spec["scenario_template"]["policy"] == SCENARIO_TEMPLATE_POLICY
    assert spec["scenario_template"]["selected_actor_id"] == "lead"


def test_lead_vehicle_ahead_yields_rear_end_template():
    spec = infer_scenario_spec("s1", _root(_lead()))
    assert spec["scenario_type"] == "rear_end_collision_risk"
    assert [b for b in spec["branches"] if b["is_factual"]][0]["intervention_type"] == "do_ego_emergency_brake"
    assert spec["env_behavior"]["lead"]["accel"] < 0


def test_pedestrian_ahead_that_can_reach_the_corridor_yields_pedestrian_template():
    spec = infer_scenario_spec("s2", _root(_ped(x=12, y=4.0, vy=-1.0)))
    assert spec["scenario_type"] == "pedestrian_collision_risk"
    assert spec["env_behavior"]["ped"]["lat"] < 0  # crosses toward the lane (-y)


def test_pedestrian_behind_ego_does_not_drive_the_template():
    spec = infer_scenario_spec("s3", _root(_lead(x=16.7), _ped(x=-10.4, y=3.0)))
    assert spec["scenario_type"] == "rear_end_collision_risk"
    # and it is not given synthetic crossing motion
    assert spec["env_behavior"]["ped"] == {"accel": 0.0, "lat": 0.0}
    reasons = {c["id"]: c["reasons"] for c in spec["scenario_template"]["candidates"]}
    assert "road_entry_actor_not_ahead_of_ego" in reasons["ped"]


def test_pedestrian_too_far_to_reach_within_horizon_is_not_relevant():
    # 12 - 0.3 - 2.0 = 9.7 m of edge distance > 1.6 m/s x 3 s
    spec = infer_scenario_spec("s4", _root(_lead(), _ped(x=15, y=12.0)))
    assert spec["scenario_type"] == "rear_end_collision_risk"
    assert spec["env_behavior"]["ped"]["lat"] == 0.0


def test_pedestrian_moving_away_from_the_lane_is_not_relevant():
    spec = infer_scenario_spec("s5", _root(_lead(), _ped(x=15, y=4.0, vy=+1.2)))
    assert spec["scenario_type"] == "rear_end_collision_risk"
    assert spec["env_behavior"]["ped"]["lat"] == pytest.approx(1.2)  # observed velocity kept


def test_actor_not_visible_at_t0_cannot_drive_the_template():
    root = _root(_lead(), _ped(x=12, y=4.0, vy=-1.0))
    spec = infer_scenario_spec("s6", root, t0_visible_ids={"ego", "lead"})
    assert spec["scenario_type"] == "rear_end_collision_risk"
    reasons = {c["id"]: c["reasons"] for c in spec["scenario_template"]["candidates"]}
    assert "not_visible_at_t0" in reasons["ped"]


def test_most_urgent_relevant_hazard_wins_regardless_of_category():
    # pedestrian entering the lane is within the interaction gap after one step;
    # a lead 40 m ahead braking under its L4 profile only gets there at the end
    spec = infer_scenario_spec("s7", _root(_lead(x=40), _ped(x=12, y=3.5)))
    assert spec["scenario_type"] == "pedestrian_collision_risk"
    steps = {c["id"]: c["urgency_steps"] for c in spec["scenario_template"]["candidates"]}
    assert steps["ped"] < steps["lead"] <= 6
    # lead 8 m ahead is inside the gap after one step and beats that pedestrian
    spec = infer_scenario_spec("s8", _root(_lead(x=8), _ped(x=12, y=3.5)))
    assert spec["scenario_type"] == "rear_end_collision_risk"
    assert spec["env_behavior"]["ped"] == {"accel": 0.0, "lat": 0.0}


def test_reviewed_critical_actor_breaks_ties_at_equal_step_count():
    # both hazards are inside the interaction gap after the first 0.5 s step
    root = _root(_lead(x=8), _ped(x=6, y=2.6))
    steps = {c["id"]: c["urgency_steps"] for c in infer_scenario_spec("t0", root)["scenario_template"]["candidates"]}
    assert steps["lead"] == steps["ped"] == 1
    # without a reviewed critical actor the nearer hazard wins the tie (ped at 6.5 m)
    assert infer_scenario_spec("t1", root)["scenario_type"] == "pedestrian_collision_risk"
    spec = infer_scenario_spec("t2", root, critical_actor_id="lead")
    assert spec["scenario_type"] == "rear_end_collision_risk"
    assert spec["scenario_template"]["critical_actor_id"] == "lead"


def test_in_corridor_pedestrian_beyond_the_horizon_is_not_relevant():
    # A pedestrian already in the corridor but 100 m ahead cannot meet the ego
    # within 3 s at 15 m/s; it is not a hazard for the L3 template at all,
    # and cannot outrank a lead vehicle 8 m ahead.
    root = _root(_lead(x=8), _ped(x=100, y=0.5))
    spec = infer_scenario_spec("h1", root, critical_actor_id="lead")
    assert spec["scenario_type"] == "rear_end_collision_risk"
    cands = {c["id"]: c for c in spec["scenario_template"]["candidates"]}
    assert cands["ped"]["in_corridor"] is True
    assert cands["ped"]["relevant"] is False
    assert "does_not_interact_with_ego_within_horizon" in cands["ped"]["reasons"]
    assert cands["ped"]["urgency_steps"] is None
    assert cands["ped"]["probe_min_edge_gap_m"] > cands["ped"]["interaction_gap_threshold_m"]
    # alone in the root it still does not select the pedestrian template
    alone = infer_scenario_spec("h2", _root(_ped(x=100, y=0.5)))
    assert alone["scenario_type"] == "generic_following"
    assert alone["scenario_template"]["selected_actor_id"] is None


def test_receding_lead_is_not_relevant_even_when_it_is_the_nearest_actor():
    # Lead 8 m ahead pulling away at 25 m/s from an ego at 10 m/s: the L4
    # interaction test rejects it (the gap only grows, even under its braking
    # profile), while the stationary pedestrian beside the lane passes;
    # ego headway alone does not rank the lead first.
    root = _root(_lead(x=8, vx=25.0), _ped(x=12, y=3.2), speed=10.0)
    spec = infer_scenario_spec("r1", root)
    assert spec["scenario_type"] == "pedestrian_collision_risk"
    assert spec["scenario_template"]["selected_actor_id"] == "ped"
    cands = {c["id"]: c for c in spec["scenario_template"]["candidates"]}
    assert cands["lead"]["relevant"] is False
    assert "does_not_interact_with_ego_within_horizon" in cands["lead"]["reasons"]
    assert cands["lead"]["probe_min_edge_gap_m"] > 5.0
    assert cands["ped"]["relevant"] is True and cands["ped"]["probe_min_edge_gap_m"] <= 5.0
    # the lead keeps its observed motion in the L3 alternatives; only the
    # selected pedestrian receives the template's entry profile
    assert spec["env_behavior"]["lead"] == {"accel": 0.0, "lat": 0.0}
    assert spec["env_behavior"]["ped"]["lat"] < 0


def test_urgency_is_the_first_step_of_the_l4_interaction_test():
    # Relevance and urgency come from the same probe as the L4 paired-effect
    # gate: the actor under its published profile, the ego under no-action.
    spec = infer_scenario_spec("p1", _root(_lead(x=8), _ped(x=12, y=4.0, vy=-1.0)))
    template = spec["scenario_template"]
    probe = template["interaction_probe"]
    assert probe["ego_policy"] == "do_ego_no_action"
    assert probe["max_interaction_edge_gap_m"] == 5.0
    assert probe["twin_model"] == "point_mass" and probe["base_steer"] == 0.0
    for c in template["candidates"]:
        assert c["interaction_gap_threshold_m"] == 5.0
        if c["relevant"]:
            assert 1 <= c["urgency_steps"] <= spec["n_steps"]
            assert c["urgency_seconds"] == pytest.approx(c["urgency_steps"] * spec["dt"])
            assert c["probe_interaction_step"] == c["urgency_steps"]
            assert c["probe_min_edge_gap_m"] <= 5.0
            assert c["probe_intervention_type"] is not None
    # the reviewed critical actor's own gate profile is the one probed: a
    # pedestrian on the right (y > 0) enters towards -y
    ped = next(c for c in template["candidates"] if c["id"] == "ped")
    assert ped["probe_intervention_type"] == "pedestrian_enters_road"
    assert ped["probe_lateral_velocity_mps"] == pytest.approx(-1.6)


def test_builder_twin_model_and_steer_are_threaded_into_the_probe():
    spec = infer_scenario_spec("m1", _root(_lead(x=8)), model="bicycle", base_steer=0.01)
    probe = spec["scenario_template"]["interaction_probe"]
    assert probe["twin_model"] == "bicycle" and probe["base_steer"] == pytest.approx(0.01)
    assert spec["scenario_type"] == "rear_end_collision_risk"


def test_stopped_lead_and_pedestrian_already_in_corridor_are_still_hazards():
    spec = infer_scenario_spec("u1", _root(_lead(x=20, vx=0.0)))
    assert spec["scenario_type"] == "rear_end_collision_risk"
    assert spec["env_behavior"]["lead"] == {"accel": 0.0, "lat": 0.0}  # a stopped car stays stopped
    spec = infer_scenario_spec("u2", _root(_ped(x=14, y=0.5, vx=0.0, vy=-0.8)))
    assert spec["scenario_type"] == "pedestrian_collision_risk"
    chosen = next(c for c in spec["scenario_template"]["candidates"] if c["id"] == "ped")
    assert chosen["in_corridor"] is True
    assert spec["scenario_template"]["urgency_steps"] == 1  # gap 4.2 m after the first 0.5 s step
    assert spec["env_behavior"]["ped"]["lat"] == pytest.approx(-0.8)  # observed motion, no synthetic entry


def test_cyclist_uses_cyclist_template_and_adjacent_uses_cut_in():
    spec = infer_scenario_spec("c1", _root(_ped(x=10, y=3.5, entity="bicycle")))
    assert spec["scenario_type"] == "cyclist_collision_risk"
    adj = {"id": "adj", "entity_type": "vehicle", "role_type": "adjacent_vehicle",
           "x": 12, "y": 3.6, "vx": 14, "vy": 0, "speed": 14, "length": 4.5, "width": 1.9}
    spec = infer_scenario_spec("c2", _root(adj))
    assert spec["scenario_type"] == "cut_in_collision_risk"
    assert spec["env_behavior"]["adj"] == {"accel": 0.0, "lat": 0.0}


def test_no_relevant_actor_falls_back_to_generic_or_obstacle():
    spec = infer_scenario_spec("g1", _root(_ped(x=-5, y=3)))
    assert spec["scenario_type"] == "generic_following"
    assert spec["scenario_template"]["selected_actor_id"] is None
    cone = {"id": "cone", "entity_type": "traffic_cone", "role_type": "obstacle_actor",
            "x": 15, "y": 0.2, "vx": 0, "vy": 0, "speed": 0, "length": 0.4, "width": 0.4}
    assert infer_scenario_spec("g2", _root(cone))["scenario_type"] == "obstacle_appearance_risk"


def test_non_template_actors_keep_observed_velocity():
    other = {"id": "adj", "entity_type": "vehicle", "role_type": "adjacent_vehicle",
             "x": 20, "y": 3.6, "vx": 12, "vy": 0.4, "speed": 12.0, "length": 4.5, "width": 1.9}
    spec = infer_scenario_spec("v1", _root(_lead(x=10), other))
    assert spec["env_behavior"]["adj"] == {"accel": 0.0, "lat": pytest.approx(0.4)}


def test_t0_visible_actor_ids_reads_the_universe_at_the_root_token():
    root = _root({**_lead(), "source_instance_token": "inst_lead"},
                 {**_ped(), "source_instance_token": "inst_ped"})
    visibility = {"inst_lead": {"t-1": True, "t0": True}, "inst_ped": {"t-1": True, "t0": False}}
    assert t0_visible_actor_ids(root, visibility, "t0") == {"ego", "lead"}
