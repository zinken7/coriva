"""Option B (docs/l4_pair_protocol.md §3, §5): the L4 query is scored as a
matched control/treated pair. The pair is the unit; each chain keeps its
own gate result; the control agreement is reported separately from CAS.
Synthetic twin scene, no dataset access."""
import copy
import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from ctms.ctms import (  # noqa: E402
    CONTROL_INTERVENTION_TYPE, L4_PAIR_PROTOCOL, evaluate_ctms, layer4_chain_gates,
    load_ctms_environment, score_layer4_pair)
from simulation.kinematic_twin import build_twin_branch, build_twin_gt  # noqa: E402
from simulation.nuscenes_admission import observed_velocity_baseline  # noqa: E402


@pytest.fixture(scope="module")
def env():
    e = load_ctms_environment("ctms/ontology.yaml", "ctms/ctms_config.yaml",
                              "ctms/schema_branch.json", "ctms/schema_counterfactual.json")
    e["config"]["layer3"]["use_metadata_gate"] = True
    e["config"]["timing"]["require_protocol_grid"] = True
    return e


def _root():
    return {"depth": 0, "timestamp": 0.0, "event_type": "observed_state", "actors": [
        dict(id="ego", entity_type="ego_vehicle", role_type="ego_agent", x=0., y=0., vx=8., vy=0., speed=8., length=4.5, width=2.),
        dict(id="npc_1", entity_type="vehicle", role_type="lead_vehicle", x=30., y=0., vx=8., vy=0., speed=8., length=4.5, width=2.),
        dict(id="npc_2", entity_type="vehicle", role_type="adjacent_vehicle", x=10., y=3.5, vx=8., vy=0., speed=8., length=4.5, width=2.),
    ]}


QUERY = dict(intervention_type="npc_sudden_brake", target_actor_id="npc_1", trigger_depth=1)


def _pair_gt():
    root = _root()
    spec = dict(scenario_id="pair_synthetic", n_steps=6, dt=.5, sample_interval=.5, branches=[
        dict(branch_id="Factual_no_action", is_factual=True, intervention_type="do_ego_no_action"),
        dict(branch_id="CF_brake", is_factual=False, intervention_type="do_ego_emergency_brake")])
    g = build_twin_gt(root, spec)
    baseline = observed_velocity_baseline(root)
    treated = build_twin_branch("L4_Reviewed", False, "npc_sudden_brake", root, 6, .5, .5,
                                env_behavior=baseline, intervention_meta=dict(QUERY),
                                environment_intervention=dict(QUERY))
    control = build_twin_branch("L4_Control", False, "do_ego_no_action", root, 6, .5, .5,
                                env_behavior=baseline,
                                intervention_meta={"intervention_type": CONTROL_INTERVENTION_TYPE,
                                                   "target_actor_id": "npc_1", "trigger_depth": 1})
    g["l4_queries"] = [{
        "intervention_type": "npc_sudden_brake",
        "counterfactual_chain": {"intervention": dict(QUERY), "chain": treated["chain"]},
        "control_chain": {"intervention": {**control["intervention"], "paired_with": "npc_sudden_brake"},
                          "chain": control["chain"],
                          "semantics": {"source": "test"}},
    }]
    g["protocol"] = {"protocol_version": "coriva-protocol-1.0", "track": "primary",
                     "future_steps": 6, "l4_pair_protocol": L4_PAIR_PROTOCOL}
    return g


def _pred(g, treated=True, control=True, treated_key="treated_chain"):
    q = g["l4_queries"][0]
    item = {"intervention": dict(QUERY)}
    if treated:
        item[treated_key] = copy.deepcopy(q["counterfactual_chain"]["chain"])
    if control:
        item["control_chain"] = copy.deepcopy(q["control_chain"]["chain"])
    return {"branches": copy.deepcopy(g["branches"]), "predicted_counterfactuals": [item]}


def _l4(env, g, p):
    r = evaluate_ctms(g, p, env["ontology"], env["config"], env["branch_schema"], env["counterfactual_schema"])
    return r, r["layer4"]["results"][0]


def test_reference_pair_scores_one_on_both_chains(env):
    g = _pair_gt()
    r, u = _l4(env, g, _pred(g))
    assert u["pair"] == {"protocol": L4_PAIR_PROTOCOL, "control_reference_present": True,
                         "treated_present": True, "control_present": True,
                         "treated_valid": True, "control_valid": True,
                         "pair_complete": True, "pair_valid": True}
    assert u["valid"] is True and u["Score_interv"] == pytest.approx(1.0)
    assert u["control_similarity"] == pytest.approx(1.0)
    assert u["treated"]["Score_interv"] == pytest.approx(1.0)
    assert u["control"]["Score_interv"] == pytest.approx(1.0)
    assert r["Counterfactual_Alignment_Score"] == pytest.approx(1.0)
    assert u["diagnostics"]["reason"] is None


def test_the_pre_pair_chain_key_is_the_treated_alias(env):
    g = _pair_gt()
    _, u = _l4(env, g, _pred(g, treated_key="chain"))
    assert u["pair"]["pair_valid"] is True and u["Score_interv"] == pytest.approx(1.0)


def test_missing_control_is_complete_false_and_pairwise_score_zero(env):
    g = _pair_gt()
    _, u = _l4(env, g, _pred(g, control=False))
    assert u["pair"]["treated_valid"] is True
    assert u["pair"]["control_present"] is False
    assert u["pair"]["pair_complete"] is False and u["pair"]["pair_valid"] is False
    assert u["valid"] is False and u["Score_interv"] == 0.0
    assert u["diagnostics"]["reason"] == "missing_control_chain"
    # the treated chain's own agreement is still available, separately
    assert u["treated"]["Score_interv"] == pytest.approx(1.0)
    assert u["control_similarity"] == 0.0


def test_invalid_treated_keeps_control_agreement_reported_separately(env):
    g = _pair_gt()
    p = _pred(g)
    # hallucinated actor in the treated chain -> semantic gate fails
    p["predicted_counterfactuals"][0]["treated_chain"][0]["actors_update"].append(
        dict(id="ghost", entity_type="vehicle", role_type="lead_vehicle", x=1., y=1., vx=0., vy=0., speed=0.))
    _, u = _l4(env, g, p)
    assert u["pair"]["pair_complete"] is True and u["pair"]["pair_valid"] is False
    assert u["pair"]["treated_valid"] is False and u["pair"]["control_valid"] is True
    assert u["valid"] is False and u["Score_interv"] == 0.0
    assert u["control_similarity"] == pytest.approx(1.0)
    assert u["diagnostics"]["reason"] == "treated_chain_invalid"
    assert u["gates"]["semantic"] is False and u["control"]["gates"]["semantic"] is True


def test_invalid_control_zeroes_the_pair_but_not_the_treated_own_score(env):
    g = _pair_gt()
    p = _pred(g)
    p["predicted_counterfactuals"][0]["control_chain"][0]["actors_update"][0]["x"] = 500.0  # teleport
    _, u = _l4(env, g, p)
    assert u["pair"]["control_valid"] is False and u["pair"]["treated_valid"] is True
    assert u["valid"] is False and u["Score_interv"] == 0.0
    assert u["treated"]["Score_interv"] == pytest.approx(1.0)
    assert u["control_similarity"] == 0.0
    assert u["diagnostics"]["reason"] == "control_chain_invalid"


def test_control_chain_target_must_be_present(env):
    g = _pair_gt()
    p = _pred(g)
    for node in p["predicted_counterfactuals"][0]["control_chain"]:
        node["actors_update"] = [a for a in node["actors_update"] if a["id"] != "npc_1"]
    _, u = _l4(env, g, p)
    assert u["pair"]["control_valid"] is False
    assert any("target_actor_id" in m for m in u["control"]["diagnostics"]["interv_msgs"])


def test_duplicate_pairs_for_one_intervention_invalidate_the_query(env):
    g = _pair_gt()
    p = _pred(g)
    p["predicted_counterfactuals"].append(copy.deepcopy(p["predicted_counterfactuals"][0]))
    _, u = _l4(env, g, p)
    assert u["valid"] is False and u["diagnostics"]["reason"] == "multiple_chains_for_intervention"
    assert u["pair"]["pair_valid"] is False and u["control_similarity"] == 0.0


def test_legacy_reference_without_control_scores_the_treated_chain_alone(env):
    g = _pair_gt()
    del g["l4_queries"][0]["control_chain"]
    _, u = _l4(env, g, _pred(g, control=False))
    assert u["pair"]["protocol"] == "single_treated_legacy"
    assert u["pair"]["control_reference_present"] is False
    assert u["valid"] is True and u["Score_interv"] == pytest.approx(1.0)
    assert u["control"] is None and u["control_similarity"] is None


@pytest.mark.parametrize("bad", [None, "x", 3, {}, [None], [{"depth": "a"}], [{"actors_update": 5}], [[]]])
def test_malformed_control_never_raises_and_fails_its_own_gate(env, bad):
    g = _pair_gt()
    p = _pred(g)
    p["predicted_counterfactuals"][0]["control_chain"] = bad
    _, u = _l4(env, g, p)
    assert u["pair"]["control_present"] is True
    assert u["pair"]["control_valid"] is False and u["valid"] is False
    assert u["treated"]["valid"] is True


@pytest.mark.parametrize("bad", [None, "x", 3, {}, [], [None]])
def test_malformed_pair_item_never_raises(env, bad):
    g = _pair_gt()
    p = _pred(g)
    p["predicted_counterfactuals"] = [bad]
    r, u = _l4(env, g, p)
    assert u["valid"] is False and u["Score_interv"] == 0.0
    assert r["CTMS"] == 0.0


def test_gate_function_is_reference_free_and_matches_the_scorer(env):
    g = _pair_gt()
    p = _pred(g)
    item = p["predicted_counterfactuals"][0]
    wrapper = {"counterfactual_chain": {"intervention": item["intervention"], "chain": item["treated_chain"]}}
    gated = layer4_chain_gates(dict(QUERY), wrapper, env["ontology"], env["config"],
                               env["counterfactual_schema"], root=g["observed_state"])
    assert gated["valid"] is True and gated["pred_canon"] is not None
    _, u = _l4(env, g, p)
    assert gated["gates"] == u["treated"]["gates"]


def test_score_layer4_pair_reports_missing_pair(env):
    g = _pair_gt()
    u = score_layer4_pair(g["l4_queries"][0], {"intervention": dict(QUERY)}, env["ontology"],
                          env["config"], env["counterfactual_schema"], root=g["observed_state"])
    assert u["diagnostics"]["reason"] == "missing_pair"
    assert u["pair"]["pair_complete"] is False
