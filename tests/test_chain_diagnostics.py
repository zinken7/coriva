"""Label-free chain diagnostics: the pieces CTMS folds into one number."""
import copy
import json
import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from analysis.chain_diagnostics import METRICS, scenario_diagnostics


def _gt():
    from analysis.rq3a import make_synthetic_gts
    gt = make_synthetic_gts(1)[0]
    gt["scenario_id"] = "nuscenes_0001"
    return gt


def _oracle(gt):
    """Reference copied into the prediction shape (what the oracle control emits)."""
    l3 = {"predicted_branches": [
        {"branch_id": b.get("branch_id", "B"), "is_factual": b.get("is_factual", False),
         "intervention": copy.deepcopy(b["intervention"]), "chain": copy.deepcopy(b["chain"])}
        for b in gt["branches"]]}
    l4 = {"predicted_counterfactuals": [
        {"intervention": copy.deepcopy(q["counterfactual_chain"]["intervention"]),
         "chain": copy.deepcopy(q["counterfactual_chain"]["chain"])}
        for q in (gt.get("l4_queries") or gt.get("counterfactuals") or [])]}
    return l3, l4


def test_reference_copy_is_the_identity():
    gt = _gt(); l3, l4 = _oracle(gt)
    d = scenario_diagnostics(gt, l3, l4)
    assert set(d) == set(METRICS)
    assert d["l3_branch_completeness"] == 1.0 and d["l3_chain_length_ratio"] == 1.0 and d["l3_one_node_share"] == 0.0
    assert d["l3_ego_error_d1"] == 0.0 and d["l3_ego_error_end"] == 0.0
    assert d["l4_chain_length_ratio"] == 1.0 and d["l4_one_node"] == 0.0
    assert d["l4_target_error_d1"] == 0.0 and d["l4_target_error_end"] == 0.0 and d["l4_ego_error_end"] == 0.0
    assert d["snapshot_completeness"] == 1.0
    assert d["l4_target_present"] == 1.0


def test_one_node_answers_and_missing_branches_are_separated_from_geometry():
    gt = _gt(); l3, l4 = _oracle(gt)
    # keep only the first branch and cut every chain to its first node: the geometry at
    # depth 1 stays exact while the structure metrics collapse
    l3["predicted_branches"] = l3["predicted_branches"][:1]
    for b in l3["predicted_branches"]:
        b["chain"] = b["chain"][:1]
    for c in l4["predicted_counterfactuals"]:
        c["chain"] = c["chain"][:1]
    d = scenario_diagnostics(gt, l3, l4)
    assert d["l3_branch_completeness"] == pytest.approx(1 / len(gt["branches"]))
    assert d["l3_one_node_share"] == 1.0 and d["l4_one_node"] == 1.0
    assert d["l3_chain_length_ratio"] < 1.0 and d["l4_chain_length_ratio"] < 1.0
    assert d["l3_ego_error_d1"] == 0.0 and d["l3_ego_error_d3"] is None


def _named_actor_gt():
    """A protocol-style scene: one named-actor L4 query under the no-action policy."""
    def node(depth, ego_x, ego_v, npc_x, npc_v, park=True):
        actors = [{"id": "ego", "entity_type": "ego_vehicle", "role_type": "ego_agent", "x": ego_x, "y": 0.0, "vx": ego_v, "vy": 0.0, "speed": ego_v},
                  {"id": "npc_1", "entity_type": "vehicle", "role_type": "lead_vehicle", "x": npc_x, "y": 0.0, "vx": npc_v, "vy": 0.0, "speed": npc_v}]
        if park:
            actors.append({"id": "park_1", "entity_type": "parked_vehicle", "role_type": "adjacent_vehicle", "x": 15.0, "y": 4.0, "vx": 0.0, "vy": 0.0, "speed": 0.0})
        return {"depth": depth, "timestamp": 0.5 * depth, "event_type": "maintain_course", "action_type": "ego_no_action", "actors_update": actors}
    root = node(0, 0.0, 10.0, 20.0, 10.0); root["event_type"] = "observed_state"; root["actors"] = root.pop("actors_update")
    chain = [node(1, 5.0, 10.0, 24.0, 6.0), node(2, 10.0, 10.0, 26.0, 2.0)]
    return {"scenario_id": "s", "observed_state": root,
            "branches": [{"branch_id": "B1", "is_factual": True, "intervention": {"intervention_type": "do_ego_no_action", "target_actor_id": "ego", "trigger_depth": 1}, "chain": copy.deepcopy(chain)}],
            "l4_queries": [{"intervention_type": "npc_sudden_brake", "counterfactual_chain": {"intervention": {"intervention_type": "npc_sudden_brake", "target_actor_id": "npc_1", "trigger_depth": 1}, "chain": copy.deepcopy(chain)}}]}


def test_omitted_target_and_reacting_ego_are_reported():
    gt = _named_actor_gt(); l3, l4 = _oracle(gt)
    assert scenario_diagnostics(gt, l3, l4)["l4_target_brake_observed"] == 1.0     # the reference brakes the lead
    for node in l4["predicted_counterfactuals"][0]["chain"]:
        node["actors_update"] = [a for a in node["actors_update"] if a["id"] != "npc_1"]
        for a in node["actors_update"]:
            if a["id"] == "ego":                       # the model brakes although the policy is no action
                a["vx"], a["speed"] = 0.0, 0.0
            if a["id"] == "park_1":                    # and shifts the parked car
                a["x"] = 12.0
    d = scenario_diagnostics(gt, l3, l4)
    assert d["l4_target_present"] == 0.0
    assert d["l4_target_brake_observed"] is None       # nothing to read a braking from
    assert d["snapshot_completeness"] < 1.0
    assert d["l4_ego_speed_change"] == pytest.approx(10.0)
    assert d["l4_ego_error_end"] == pytest.approx(0.0)  # positions were left as the reference's
    assert d["static_actor_displacement"] == pytest.approx(3.0)


def test_target_motion_readings():
    from analysis.chain_diagnostics import target_brake_observed, target_lateral_sign_agreement
    brake_root, brake_last = {"speed": 10.0}, {"speed": 6.0}
    assert target_brake_observed(brake_root, brake_last) == 1
    assert target_brake_observed(brake_root, {"speed": 9.8}) == 0
    assert target_brake_observed(brake_root, {"speed": "n/a"}) is None
    # Lateral motion is read against the REFERENCE, not the T0 axis: an actor
    # that crosses the lane, or follows a curve, is "correct" when it moved
    # the way the reference moved.
    root = {"y": -4.0}
    assert target_lateral_sign_agreement(root, {"y": 8.0}, {"y": 8.0}) == 1     # crossed to the far side, like the reference
    assert target_lateral_sign_agreement(root, {"y": -10.0}, {"y": -10.0}) == 1  # curve carries both away from the T0 axis
    assert target_lateral_sign_agreement(root, {"y": -6.0}, {"y": 0.0}) == 0
    assert target_lateral_sign_agreement(root, {"y": -4.1}, {"y": -3.9}) == 1   # both within the dead zone
    assert target_lateral_sign_agreement(root, {"y": None}, {"y": 0.0}) is None


def test_static_actors_and_missing_predictions():
    gt = _gt(); l3, l4 = _oracle(gt)
    d0 = scenario_diagnostics(gt, l3, l4)
    assert d0["static_actor_displacement"] in (None, 0.0)
    empty = scenario_diagnostics(gt, {"predicted_branches": []}, {"predicted_counterfactuals": []})
    assert empty["l3_branch_completeness"] == 0.0 and empty["l4_target_present"] is None and empty["snapshot_completeness"] is None
