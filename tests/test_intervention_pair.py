"""Reference-free L4 pair axes (docs/l4_pair_protocol.md §4): property tests
including curves, crossing, early terminal and the no-reference guarantee.
Chains come from the kinematic twin, never from a dataset."""
import copy
import json
import math
import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from analysis.intervention_pair import (  # noqa: E402
    L4_PAIR_AXES, L4_PAIR_VERSION, direction_consistency, ego_policy_consistency, future_nodes,
    pair_axes, road_normal_at, run_l4_pair, scenario_pair_record, target_responsiveness)
from simulation.kinematic_twin import build_twin_branch  # noqa: E402
from simulation.nuscenes_admission import observed_velocity_baseline  # noqa: E402


def _actor(aid, entity, role, x, y, vx=0.0, vy=0.0, length=4.5, width=2.0):
    return dict(id=aid, entity_type=entity, role_type=role, x=x, y=y, vx=vx, vy=vy,
                speed=math.hypot(vx, vy), length=length, width=width)


def _root(target="lead", ego_speed=10.0):
    ego = _actor("ego", "ego_vehicle", "ego_agent", 0.0, 0.0, ego_speed)
    if target == "lead":
        tgt = _actor("npc_1", "vehicle", "lead_vehicle", 25.0, 0.0, ego_speed)
    elif target == "cutin":  # on the +y (right) side, ahead
        tgt = _actor("npc_1", "vehicle", "adjacent_vehicle", 8.0, 3.5, ego_speed)
    else:  # pedestrian on the right kerb, stationary
        tgt = _actor("npc_1", "pedestrian", "crossing_actor", 20.0, 3.0, 0.0, 0.0, 0.8, 0.6)
    cone = _actor("cone_1", "traffic_cone", "obstacle_actor", 40.0, -6.0, 0.0, 0.0, 0.4, 0.4)
    return {"depth": 0, "timestamp": 0.0, "event_type": "observed_state", "actors": [ego, tgt, cone]}


def _query(itype):
    return {"intervention_type": itype, "target_actor_id": "npc_1", "trigger_depth": 1}


def _pair(root, itype, profile, model="point_mass", base_steer=0.0, n_steps=6):
    """Control/treated chains as the twin produces them (the oracle's answer)."""
    baseline = observed_velocity_baseline(root)
    env_iv = {**_query(itype), "resolved_accel_mps2": profile["accel"],
              "resolved_lateral_velocity_mps": profile["lat"]}
    kw = dict(model=model, base_steer=base_steer, env_behavior=baseline, first_interval="label_first")
    control = build_twin_branch("C", False, "do_ego_no_action", root, n_steps, .5, .5,
                                intervention_meta={"intervention_type": "target_baseline",
                                                   "target_actor_id": "npc_1", "trigger_depth": 1}, **kw)
    treated = build_twin_branch("T", False, itype, root, n_steps, .5, .5, intervention_meta=_query(itype),
                                environment_intervention=env_iv, **kw)
    return control["chain"], treated["chain"]


BRAKE = {"accel": -6.0, "lat": 0.0}
CUT_IN_LEFT = {"accel": 0.0, "lat": -1.8}     # from +y towards the ego lane
PED_ENTRY = {"accel": 0.0, "lat": -2.0}


# ── the twin's own pair satisfies every axis ─────────────────────────────

@pytest.mark.parametrize("target,itype,profile", [
    ("lead", "npc_sudden_brake", BRAKE),
    ("cutin", "npc_cut_in_left", CUT_IN_LEFT),
    ("ped", "pedestrian_enters_road", PED_ENTRY),
])
def test_twin_pair_is_responsive_directed_and_policy_consistent(target, itype, profile):
    root = _root(target)
    c, t = _pair(root, itype, profile)
    axes = pair_axes(c, t, root, _query(itype), profile)
    assert [axes[a] for a in L4_PAIR_AXES] == [1.0, 1.0, 1.0], axes["details"]


def test_identical_chains_are_unresponsive_and_direction_zero():
    root = _root("lead")
    c, _ = _pair(root, "npc_sudden_brake", BRAKE)
    axes = pair_axes(c, copy.deepcopy(c), root, _query("npc_sudden_brake"), BRAKE)
    assert axes["target_responsiveness"] == 0.0
    assert axes["direction_consistency"] == 0.0
    assert axes["ego_policy_consistency"] == 1.0


def test_wrong_direction_is_zero_but_responsive():
    root = _root("cutin")
    c, t = _pair(root, "npc_cut_in_left", CUT_IN_LEFT)
    away = _pair(root, "npc_cut_in_left", {"accel": 0.0, "lat": +1.8})[1]  # moves away from the lane
    axes = pair_axes(c, away, root, _query("npc_cut_in_left"), CUT_IN_LEFT)
    assert axes["target_responsiveness"] == 1.0
    assert axes["direction_consistency"] == 0.0
    accel = _pair(root, "npc_sudden_brake", {"accel": +6.0, "lat": 0.0})[1]
    axes = pair_axes(c, accel, _root("cutin"), _query("npc_sudden_brake"), BRAKE)
    assert axes["direction_consistency"] == 0.0


# ── curves: the rule is frame-invariant ──────────────────────────────────

def _arc_state(aid, entity, role, s, radius, speed, lateral=0.0, length=4.5, width=2.0):
    """State of an actor at arc length `s` on a left-turning circle of radius
    `radius` that starts at the origin heading +x; `lateral` displaces it
    along the local +y-side normal (negative = towards -y = the inside of a
    right turn... here the turn is towards -y, so -lateral is the inside)."""
    theta = s / radius                      # heading rotates towards -y
    x = radius * math.sin(theta)
    y = -radius * (1.0 - math.cos(theta))
    tx, ty = math.cos(theta), -math.sin(theta)          # tangent
    nx, ny = -ty, tx                                     # +y-side normal (-vy, vx)/|v|
    return _actor(aid, entity, role, x + nx * lateral, y + ny * lateral, tx * speed, ty * speed, length, width)


def _arc_pair(lateral_rate, radius=20.0, speed=10.0, gap=2.0, n=6):
    """Hand-built control/treated chains on a 3 s arc that turns ~86 deg:
    the ego and the target keep the road in both chains; the treated target
    is additionally displaced along the LOCAL normal by lateral_rate * t."""
    root = {"depth": 0, "timestamp": 0.0, "event_type": "observed_state", "actors": [
        _arc_state("ego", "ego_vehicle", "ego_agent", 0.0, radius, speed),
        _arc_state("npc_1", "vehicle", "adjacent_vehicle", gap, radius, speed, lateral=3.5)]}
    control, treated = [], []
    for k in range(1, n + 1):
        t = 0.5 * k
        ego = _arc_state("ego", "ego_vehicle", "ego_agent", speed * t, radius, speed)
        tc = _arc_state("npc_1", "vehicle", "adjacent_vehicle", gap + speed * t, radius, speed, lateral=3.5)
        tt = _arc_state("npc_1", "vehicle", "adjacent_vehicle", gap + speed * t, radius, speed,
                        lateral=3.5 + lateral_rate * t)
        control.append({"depth": k, "timestamp": t, "event_type": "maintain_course", "action_type": "ego_no_action",
                        "actors_update": [ego, tc]})
        treated.append({"depth": k, "timestamp": t, "event_type": "maintain_course", "action_type": "ego_no_action",
                        "actors_update": [copy.deepcopy(ego), tt]})
    return root, control, treated


def test_lateral_direction_holds_on_a_sharply_curved_road():
    """After 3 s the road has turned ~86 degrees: the T0 y axis is almost the
    road's tangent, so a naive y reading says nothing about lane changes.
    The CARLA proxy along the control target's normal reads the cut-in."""
    root, c, t = _arc_pair(lateral_rate=-1.8)  # towards -y: towards the ego lane from +y
    nodes = future_nodes(c)
    nx, ny = road_normal_at(nodes, 6, root)
    assert abs(ny) < 0.2, f"the road normal must have rotated: {(nx, ny)}"
    profile = {**CUT_IN_LEFT, 'direction_frame': 'control_target_normal', 'lateral_realization': 'steering'}
    axes = pair_axes(c, t, root, _query("npc_cut_in_left"), profile)
    assert axes["direction_consistency"] == 1.0, axes["details"]["direction_consistency"]
    assert axes["details"]["direction_consistency"]["lateral_effect_m"] == pytest.approx(-5.4, abs=0.1)
    assert axes["target_responsiveness"] == 1.0
    assert axes["ego_policy_consistency"] == 1.0  # curvature is not a policy change
    # the wrong way round on the same curve is read as wrong
    root, c, t = _arc_pair(lateral_rate=+1.8)
    assert pair_axes(c, t, root, _query("npc_cut_in_left"), profile)["direction_consistency"] == 0.0


def test_naive_t0_axis_reading_fails_on_the_curve_where_the_normal_rule_passes():
    root, c, t = _arc_pair(lateral_rate=-1.8)
    tc = next(a for a in c[-1]["actors_update"] if a["id"] == "npc_1")
    tt = next(a for a in t[-1]["actors_update"] if a["id"] == "npc_1")
    # in T0 coordinates the 5.4 m lane manoeuvre shows up mostly along x
    assert abs(tt["y"] - tc["y"]) < 1.0
    assert abs(tt["x"] - tc["x"]) > 5.0
    value, details = direction_consistency(future_nodes(c), future_nodes(t), "npc_1", "npc_cut_in_left",
                                           {**CUT_IN_LEFT, 'direction_frame': 'control_target_normal'}, root)
    assert value == 1.0 and details["family"] == "lateral"


def test_twin_bicycle_curve_keeps_ego_policy_consistent():
    """The twin's bicycle ego on a curve: constant speed and one path in
    both chains, so E = 1 although the T0-frame vy is far from zero."""
    root = _root("lead")
    c, t = _pair(root, "npc_sudden_brake", BRAKE, model="bicycle", base_steer=0.1)
    axes = pair_axes(c, t, root, _query("npc_sudden_brake"), BRAKE)
    assert axes["ego_policy_consistency"] == 1.0 and axes["direction_consistency"] == 1.0


# ── early terminal ───────────────────────────────────────────────────────

def test_collision_at_depth_one_leaves_lateral_direction_undefined_but_braking_defined():
    root = _root("cutin")
    c, t = _pair(root, "npc_cut_in_left", CUT_IN_LEFT)
    t1 = copy.deepcopy(t)[:1]
    t1[0]["event_type"] = "collision"; t1[0]["terminal"] = True; t1[0].pop("action_type", None)
    axes = pair_axes(c, t1, root, _query("npc_cut_in_left"), CUT_IN_LEFT)
    assert axes["direction_consistency"] is None
    assert axes["details"]["direction_consistency"]["reason"] == "insufficient_common_support_for_published_magnitude"
    assert axes["target_responsiveness"] in (0.0, 1.0)  # defined: one common depth
    # ego policy: no non-terminal depth in the treated chain -> undefined
    assert axes["ego_policy_consistency"] is None
    root2 = _root("lead")
    c2, t2 = _pair(root2, "npc_sudden_brake", BRAKE)
    t2 = copy.deepcopy(t2)[:1]
    t2[0]["event_type"] = "collision"; t2[0]["terminal"] = True; t2[0].pop("action_type", None)
    axes = pair_axes(c2, t2, root2, _query("npc_sudden_brake"), BRAKE)
    assert axes["direction_consistency"] == 1.0  # 6 m/s^2 * 0.5 s = 3 m/s >= 1 m/s


def test_common_support_stops_at_the_first_terminal_of_either_chain():
    root = _root("lead")
    c, t = _pair(root, "npc_sudden_brake", BRAKE)
    t3 = copy.deepcopy(t)[:3]
    t3[-1]["event_type"] = "collision"; t3[-1]["terminal"] = True; t3[-1].pop("action_type", None)
    value, details = target_responsiveness(future_nodes(c), future_nodes(t3), "npc_1")
    assert details["common_depths"] == 3 and value == 1.0
    # after the collision the ego state may differ: not compared
    t3[-1]["actors_update"][0]["x"] += 40.0
    value, details = ego_policy_consistency(future_nodes(c), future_nodes(t3), root)
    assert value == 1.0 and details["declarations_treated"] == 2


# ── ego policy ───────────────────────────────────────────────────────────

def test_speed_change_is_a_kinematic_diagnostic_not_a_declared_policy_change():
    root = _root("lead")
    c, t = _pair(root, "npc_sudden_brake", BRAKE)
    t = copy.deepcopy(t)
    for k, node in enumerate(t, start=1):
        ego = next(a for a in node["actors_update"] if a["id"] == "ego")
        ego["vx"] = max(0.0, 10.0 - 4.0 * k); ego["speed"] = ego["vx"]
    axes = pair_axes(c, t, root, _query("npc_sudden_brake"), BRAKE)
    assert axes["ego_policy_consistency"] == 1.0
    assert axes["details"]["ego_kinematics"]["speed_constancy_treated"] == 0
    assert axes["details"]["ego_kinematics"]["speed_constancy_control"] == 1


def test_ego_that_swerves_at_constant_speed_breaks_pair_path_agreement():
    root = _root("lead")
    c, t = _pair(root, "npc_sudden_brake", BRAKE)
    t = copy.deepcopy(t)
    for k, node in enumerate(t, start=1):
        ego = next(a for a in node["actors_update"] if a["id"] == "ego")
        ego["y"] -= 0.7 * k  # 3.5 m lane change, speed unchanged
    axes = pair_axes(c, t, root, _query("npc_sudden_brake"), BRAKE)
    assert axes["ego_policy_consistency"] == 1.0
    assert axes["details"]["ego_kinematics"]["pair_path_agreement"] == 0


# ── malformed input never raises, missing is never 0 ─────────────────────

@pytest.mark.parametrize("bad", [None, "x", 3, {}, [None], [{"depth": "a"}], [{"actors_update": 5}], [[]]])
def test_malformed_chains_give_undefined_not_zero(bad):
    root = _root("lead")
    c, _ = _pair(root, "npc_sudden_brake", BRAKE)
    axes = pair_axes(c, bad, root, _query("npc_sudden_brake"), BRAKE)
    assert all(axes[a] is None for a in L4_PAIR_AXES)


def test_non_numeric_target_position_drops_that_depth_only():
    root = _root("lead")
    c, t = _pair(root, "npc_sudden_brake", BRAKE)
    t = copy.deepcopy(t)
    before = target_responsiveness(future_nodes(c), future_nodes(t), "npc_1")[1]["common_depths"]
    next(a for a in t[2]["actors_update"] if a["id"] == "npc_1")["x"] = "12.0"
    value, details = target_responsiveness(future_nodes(c), future_nodes(t), "npc_1")
    assert details["common_depths"] == before - 1 and value == 1.0


def test_echoed_observed_state_node_is_dropped_like_the_scorer_does():
    root = _root("lead")
    c, t = _pair(root, "npc_sudden_brake", BRAKE)
    echo = {"depth": 0, "timestamp": 0.0, "event_type": "observed_state", "actors": root["actors"]}
    axes = pair_axes([echo] + c, [echo] + t, root, _query("npc_sudden_brake"), BRAKE)
    assert [axes[a] for a in L4_PAIR_AXES] == [1.0, 1.0, 1.0]


# ── runner: reference-free, gated, versioned ─────────────────────────────

def _scene(tmp_path, root, itype, item, with_gt="garbage"):
    d = tmp_path / "split" / "scenario_1"; d.mkdir(parents=True)
    (d / "observed_state.json").write_text(json.dumps(root))
    (d / "query.json").write_text(json.dumps({"l3_ego_actions": ["do_ego_no_action"],
                                              "l4_interventions": [_query(itype)],
                                              "l4_ego_policy": "do_ego_no_action"}))
    (d / "input_manifest.json").write_text(json.dumps({"protocol_version": "coriva-protocol-1.0",
                                                       "track": "primary", "future_steps": 6}))
    if with_gt == "garbage":
        (d / "ground_truth.json").write_text("this is not json {")
    p = tmp_path / "cell" / "scenario_1"; p.mkdir(parents=True)
    (p / "l4_pre.json").write_text(json.dumps({"scenario_id": "scenario_1", "predicted_counterfactuals": [item]}))
    return str(tmp_path / "split"), str(tmp_path / "cell")


def test_runner_never_opens_ground_truth_and_scores_the_twin_pair(tmp_path):
    root = _root("ped")
    c, t = _pair(root, "pedestrian_enters_road", PED_ENTRY)
    split, cell = _scene(tmp_path, root, "pedestrian_enters_road",
                         {"intervention": _query("pedestrian_enters_road"), "control_chain": c, "treated_chain": t})
    report = run_l4_pair(split, cell)
    assert report["l4_pair_version"] == L4_PAIR_VERSION and report["uses_reference_answers"] is False
    row = report["per_scenario"][0]
    assert row["defined"] is True and row["pair"]["pair_valid"] is True
    assert [row[a] for a in L4_PAIR_AXES] == [1.0, 1.0, 1.0], row["details"]
    assert report["coverage"]["pair_valid"] == 1
    assert report["axes"]["direction_consistency"] == {"mean": 1.0, "n": 1, 'n_pairs': 1, 'undefined_reasons': {}}


def test_runner_reports_pair_incomplete_and_invalid_as_undefined_with_coverage(tmp_path):
    root = _root("lead")
    c, t = _pair(root, "npc_sudden_brake", BRAKE)
    split, cell = _scene(tmp_path, root, "npc_sudden_brake",
                         {"intervention": _query("npc_sudden_brake"), "treated_chain": t})
    row = run_l4_pair(split, cell)["per_scenario"][0]
    assert row["defined"] is False and row["reason"] == "pair_incomplete"
    assert row["pair"]["treated_valid"] is True and row["pair"]["control_present"] is False
    assert all(row[a] is None for a in L4_PAIR_AXES)
    bad = copy.deepcopy(c)
    bad[0]["actors_update"].append(dict(id="ghost", entity_type="vehicle", role_type="lead_vehicle",
                                        x=1., y=1., vx=0., vy=0., speed=0.))
    split, cell = _scene(tmp_path / "b", root, "npc_sudden_brake",
                         {"intervention": _query("npc_sudden_brake"), "control_chain": bad, "treated_chain": t})
    row = run_l4_pair(split, cell)["per_scenario"][0]
    assert row["reason"] == "control_chain_invalid" and row["pair"]["pair_complete"] is True


def test_runner_matches_scorer_pair_validity(tmp_path):
    """The runner's gates are the scorer's gates: the same answer is
    pair-valid in both or in neither."""
    from ctms.ctms import evaluate_ctms, load_ctms_environment
    root = _root("cutin")
    c, t = _pair(root, "npc_cut_in_left", CUT_IN_LEFT)
    t = copy.deepcopy(t)
    t[1]["timestamp"] = 1.3  # off the protocol grid -> kinematic gate
    split, cell = _scene(tmp_path, root, "npc_cut_in_left",
                         {"intervention": _query("npc_cut_in_left"), "control_chain": c, "treated_chain": t})
    row = run_l4_pair(split, cell)["per_scenario"][0]
    env = load_ctms_environment("ctms/ontology.yaml", "ctms/ctms_config.yaml",
                                "ctms/schema_branch.json", "ctms/schema_counterfactual.json")
    gt = {"observed_state": root, "branches": [], "protocol": {"protocol_version": "coriva-protocol-1.0",
                                                               "track": "primary", "future_steps": 6},
          "l4_queries": [{"intervention_type": "npc_cut_in_left",
                          "counterfactual_chain": {"intervention": _query("npc_cut_in_left"), "chain": t},
                          "control_chain": {"intervention": {"intervention_type": "target_baseline",
                                                             "target_actor_id": "npc_1", "trigger_depth": 1},
                                            "chain": c}}]}
    from ctms.ctms import apply_reference_gate_policy
    cfg = apply_reference_gate_policy(env["config"], gt)
    scored = evaluate_ctms(gt, {"branches": [], "predicted_counterfactuals": [
        {"intervention": _query("npc_cut_in_left"), "control_chain": c, "treated_chain": t}]},
        env["ontology"], cfg, env["branch_schema"], env["counterfactual_schema"])
    unit = scored["layer4"]["results"][0]
    assert row["pair"]["pair_valid"] is False and unit["pair"]["pair_valid"] is False
    assert row["pair"]["control_valid"] == unit["pair"]["control_valid"]
    assert row["pair"]["treated_valid"] == unit["pair"]["treated_valid"]
