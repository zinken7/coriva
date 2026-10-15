"""Scoring properties of ctms-scorer-1.3:

1. P_role compares the actor IDs holding each required role, so naming the
   wrong collision partner does not earn full categorical credit;
2. the kinematic gate rejects chains that leave the protocol grid
   T0 + k*dt when `timing.require_protocol_grid` is on (protocol references);
3. the kinematic gate is seeded with the supplied root, so the T0 -> first
   node transition is checked instead of being exempt.

The fixture is a synthetic twin scene (no dataset access).
"""
import copy
import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from ctms.ctms import (  # noqa: E402
    SCORER_VERSION, evaluate_ctms, load_ctms_environment, check_kinematic_validity,
    canonicalize_branch, _pre_normalize_branch,
)
from simulation.kinematic_twin import build_twin_gt  # noqa: E402


@pytest.fixture(scope="module")
def env():
    e = load_ctms_environment("ctms/ontology.yaml", "ctms/ctms_config.yaml",
                              "ctms/schema_branch.json", "ctms/schema_counterfactual.json")
    # What run_benchmark sets for a coriva-protocol-1.0 reference.
    e["config"]["layer3"]["use_metadata_gate"] = True
    e["config"]["timing"]["require_protocol_grid"] = True
    return e


def _root():
    return {"depth": 0, "timestamp": 0.0, "event_type": "observed_state", "actors": [
        dict(id="ego", entity_type="ego_vehicle", role_type="ego_agent", x=0., y=0., vx=5., vy=0., speed=5., length=4.5, width=2.),
        dict(id="npc_1", entity_type="vehicle", role_type="lead_vehicle", x=12., y=0., vx=0., vy=0., speed=0., length=4.5, width=2.),
        dict(id="npc_2", entity_type="vehicle", role_type="adjacent_vehicle", x=30., y=3.5, vx=0., vy=0., speed=0., length=4.5, width=2.),
    ]}


def _sample():
    spec = dict(scenario_id="scorer10_synthetic", n_steps=6, dt=.5, sample_interval=.5, branches=[
        dict(branch_id="brake", is_factual=True, intervention_type="do_ego_emergency_brake"),
        dict(branch_id="coast", is_factual=False, intervention_type="do_ego_no_action")])
    g = build_twin_gt(_root(), spec)
    l4 = copy.deepcopy(g["branches"][1])
    l4["intervention"] = dict(intervention_type="npc_sudden_brake", target_actor_id="npc_1", trigger_depth=1)
    g["l4_queries"] = [{"intervention_type": "npc_sudden_brake",
                        "counterfactual_chain": {k: l4[k] for k in ("intervention", "chain")}}]
    p = dict(branches=copy.deepcopy(g["branches"]),
             predicted_counterfactuals=[{k: copy.deepcopy(l4[k]) for k in ("intervention", "chain")}])
    return g, p


def _score(env, g, p):
    r = evaluate_ctms(g, p, env["ontology"], env["config"], env["branch_schema"], env["counterfactual_schema"])
    return dict(F1=r["layer3"]["F1_cf"], CAS=r["layer4"]["Score_interv"], CTMS=r["CTMS"],
                gates=r["layer3"]["diagnostics"]["gate_failure_counts"],
                l4_valid=[x["valid"] for x in r["layer4"]["results"]],
                l4_kin=[x["gates"]["kinematic"] for x in r["layer4"]["results"]])


def _impacted(chain_node):
    return sorted(a["id"] for a in chain_node["actors_update"] if a["role_type"] == "impacted_actor")


def test_version():
    assert SCORER_VERSION == "ctms-scorer-1.3"


@pytest.mark.parametrize("start,end,valid", [
    (11.8, 6.8, True),   # exactly -10; binary subtraction falls below the bound
    (6.8, 11.8, True),   # exactly +10; the same rounding issue at the upper bound
    (11.8, 6.799999, False),
    (6.8, 11.800001, False),
    (11.8, 6.3, False),  # -11 was accepted under the old -12 lower bound
    (11.8, 5.8, False),  # the old -12 endpoint is now outside the prompt bound
])
@pytest.mark.parametrize("transition", ["root", "later"])
def test_prompt_acceleration_bounds(env, start, end, valid, transition):
    root = {"timestamp": 0.0, "actors": [
        dict(id="ego", x=0.0, y=0.0, vx=start, vy=0.0, speed=start)]}
    chain = []
    x = 0.0
    if transition == "later":
        x = start * 0.5
        chain.append({"depth": 1, "timestamp": 0.5, "actors_update": [
            dict(id="ego", x=x, y=0.0, vx=start, vy=0.0, speed=start)]})
    depth = len(chain) + 1
    x += (start + end) * 0.25
    chain.append({"depth": depth, "timestamp": depth * 0.5, "actors_update": [
        dict(id="ego", x=x, y=0.0, vx=end, vy=0.0, speed=end)]})
    ok, messages = check_kinematic_validity({"chain": chain}, env["config"], root)
    assert ok is valid, messages
    if not valid:
        assert any("accel out of bounds" in message for message in messages)


def test_reference_copy_still_scores_one(env):
    g, p = _sample()
    assert _impacted(g["branches"][1]["chain"][-1]) == ["npc_1"], "fixture must end in a collision with npc_1"
    out = _score(env, g, p)
    assert out["CTMS"] == pytest.approx(1.0) and out["F1"] == pytest.approx(1.0) and out["CAS"] == pytest.approx(1.0)
    assert all(v == 0 for v in out["gates"].values()) and out["l4_valid"] == [True]


def test_wrong_collision_partner_loses_credit(env):
    """Same IDs, positions and speeds, only the impacted role moved from
    npc_1 to npc_2: the role score drops."""
    g, p = _sample()
    for b in [p["branches"][1], p["predicted_counterfactuals"][0]]:
        for a in b["chain"][-1]["actors_update"]:
            if a["id"] == "npc_1":
                a["role_type"] = "lead_vehicle"
            if a["id"] == "npc_2":
                a["role_type"] = "impacted_actor"
    out = _score(env, g, p)
    assert all(v == 0 for v in out["gates"].values()) and out["l4_valid"] == [True]  # not a gate matter
    assert out["F1"] < 1.0 - 1e-6 and out["CAS"] < 1.0 - 1e-6 and out["CTMS"] < 1.0 - 1e-6


def test_time_shifted_chains_fail_the_kinematic_gate(env):
    """Every timestamp + 100 s fails the kinematic gate."""
    g, p = _sample()
    for b in p["branches"] + p["predicted_counterfactuals"]:
        for n in b["chain"]:
            n["timestamp"] += 100
    out = _score(env, g, p)
    assert out["gates"]["kinematic"] == len(p["branches"])
    assert out["l4_valid"] == [False] and out["l4_kin"] == [False]
    assert out["CTMS"] == pytest.approx(0.0)


def test_off_grid_sampling_fails_the_kinematic_gate(env):
    """Nodes at 0.45 s steps are off the protocol grid."""
    g, p = _sample()
    for b in p["branches"] + p["predicted_counterfactuals"]:
        for k, n in enumerate(b["chain"]):
            n["timestamp"] = round(0.45 * (k + 1), 2)
    out = _score(env, g, p)
    assert out["gates"]["kinematic"] == len(p["branches"]) and out["l4_valid"] == [False]


def test_grid_check_is_scoped_to_protocol_references(env):
    """Variable-dt references (no protocol version) are scored with the grid
    check off, as run_benchmark and dataset QC configure it."""
    legacy = copy.deepcopy(env)
    legacy["config"]["timing"]["require_protocol_grid"] = False
    g, p = _sample()
    for b in p["branches"] + p["predicted_counterfactuals"]:
        for k, n in enumerate(b["chain"]):
            n["timestamp"] = round(0.45 * (k + 1), 2)  # off-grid cadence, within max_dt of the root
    out = _score(legacy, g, p)
    assert out["gates"]["kinematic"] == 0 and out["l4_valid"] == [True]
    # A 100 s jump from the root is still caught by the existing max_dt check,
    # because the root now seeds the chain timeline.
    for b in p["branches"] + p["predicted_counterfactuals"]:
        for n in b["chain"]:
            n["timestamp"] += 100
    out = _score(legacy, g, p)
    assert out["gates"]["kinematic"] == len(p["branches"]) and out["l4_valid"] == [False]


def test_teleport_at_first_step_fails_the_kinematic_gate(env):
    """npc_2 is 100 m away from its root position at the first node and
    stays there: the root-to-first transition is gated."""
    g, p = _sample()
    for b in p["branches"] + p["predicted_counterfactuals"]:
        for n in b["chain"]:
            for a in n["actors_update"]:
                if a["id"] == "npc_2":
                    a["x"] += 100
    out = _score(env, g, p)
    assert out["gates"]["kinematic"] == len(p["branches"]) and out["l4_valid"] == [False]
    assert out["CTMS"] == pytest.approx(0.0)
    # The same chain passes when the gate is not seeded with the root: the
    # check is the root seeding, not a bound change.
    canon = canonicalize_branch(_pre_normalize_branch(p["branches"][0], env["ontology"]), env["ontology"], env["config"])
    assert check_kinematic_validity(canon, env["config"])[0]
    ok, msgs = check_kinematic_validity(canon, env["config"], root=g["observed_state"])
    assert not ok and any("node[0] actor=npc_2" in m for m in msgs)


def test_root_is_seeded_but_never_scored(env):
    """A prediction that echoes the root node scores exactly like one that
    does not (the root is input, never scored, under root seeding)."""
    g, p = _sample()
    with_root = copy.deepcopy(p)
    for b in with_root["branches"] + with_root["predicted_counterfactuals"]:
        b["chain"].insert(0, copy.deepcopy(g["observed_state"]))
    assert _score(env, g, with_root) == _score(env, g, p)


# ── Gate policy follows the reference's own protocol declaration ─────────

def _protocol_block(**extra):
    from simulation.protocol import protocol_manifest
    return protocol_manifest(split="nuscenes", **extra)


def _legacy_cadence(p):
    for b in p["branches"] + p["predicted_counterfactuals"]:
        for k, n in enumerate(b["chain"]):
            n["timestamp"] = round(0.45 * (k + 1), 2)
    return p


def test_gated_protocol_constant_matches_simulation_protocol():
    from ctms.ctms import GATED_PROTOCOL_VERSION
    from simulation.protocol import PROTOCOL_VERSION
    assert GATED_PROTOCOL_VERSION == PROTOCOL_VERSION


def test_reference_protocol_version_reads_builder_block_and_dataset_top_level():
    from ctms.ctms import reference_protocol_version, apply_reference_gate_policy
    assert reference_protocol_version({"protocol": _protocol_block()}) == "coriva-protocol-1.0"
    assert reference_protocol_version({"protocol_version": "coriva-protocol-1.0"}) == "coriva-protocol-1.0"
    assert reference_protocol_version({"protocol": {"protocol_version": "legacy"}}) == "legacy"
    assert reference_protocol_version({"branches": []}) is None
    base = {"layer3": {"use_metadata_gate": False}, "timing": {"require_protocol_grid": False}}
    on = apply_reference_gate_policy(base, {"protocol": _protocol_block()})
    assert on["layer3"]["use_metadata_gate"] is True and on["timing"]["require_protocol_grid"] is True
    off = apply_reference_gate_policy(on, {"protocol": {"protocol_version": "legacy"}})
    assert off["layer3"]["use_metadata_gate"] is False and off["timing"]["require_protocol_grid"] is False
    untouched = apply_reference_gate_policy(on, {"branches": []})
    assert untouched["timing"]["require_protocol_grid"] is True
    assert base["timing"]["require_protocol_grid"] is False  # never mutated


def test_qc_rejects_off_grid_protocol_reference_from_builder_block(env):
    """QC reads gt["protocol"]["protocol_version"], where both builders write
    it: a protocol reference at 0.45 s steps is rejected, the same reference
    on the grid is accepted, and a reference without a protocol version is
    exempt."""
    from simulation.qc import qc_scenario_gt
    shipped = load_ctms_environment("ctms/ontology.yaml", "ctms/ctms_config.yaml",
                                    "ctms/schema_branch.json", "ctms/schema_counterfactual.json")
    g, _ = _sample()
    g["protocol"] = _protocol_block()
    ok, reasons = qc_scenario_gt(g, shipped["ontology"], shipped["config"], shipped["branch_schema"])
    assert ok, reasons
    off = copy.deepcopy(g)
    _legacy_cadence({"branches": off["branches"],
                     "predicted_counterfactuals": [q["counterfactual_chain"] for q in off["l4_queries"]]})
    ok, reasons = qc_scenario_gt(off, shipped["ontology"], shipped["config"], shipped["branch_schema"])
    assert not ok
    assert any("off the protocol grid" in m for r in reasons for m in r.get("kinematic_msgs", []) + r.get("messages", []))
    legacy = copy.deepcopy(off)
    legacy["protocol"] = {"protocol_version": "legacy", "track": "primary"}
    ok, reasons = qc_scenario_gt(legacy, shipped["ontology"], shipped["config"], shipped["branch_schema"])
    assert ok, reasons


def test_score_sample_applies_the_sample_protocol_policy_without_cli():
    """analysis.evaluation_validation calls run_benchmark.score_sample with the
    shipped config: a protocol sample must still be gated on the grid."""
    from run_benchmark import score_sample
    shipped = load_ctms_environment("ctms/ontology.yaml", "ctms/ctms_config.yaml",
                                    "ctms/schema_branch.json", "ctms/schema_counterfactual.json")
    assert shipped["config"]["timing"]["require_protocol_grid"] is False
    g, p = _sample()
    g["protocol"] = _protocol_block()
    assert score_sample(g, copy.deepcopy(p), shipped)["CTMS"] == pytest.approx(1.0)
    r = score_sample(g, _legacy_cadence(copy.deepcopy(p)), shipped)
    assert r["CTMS"] == pytest.approx(0.0)
    assert r["layer3"]["diagnostics"]["gate_failure_counts"]["kinematic"] == len(p["branches"])
    # A sample without a protocol block keeps the shipped, ungated behaviour.
    g_legacy, p_legacy = _sample()
    assert score_sample(g_legacy, _legacy_cadence(p_legacy), shipped)["CTMS"] == pytest.approx(1.0)
    assert shipped["config"]["timing"]["require_protocol_grid"] is False  # env not mutated


def test_nodes_beyond_the_track_horizon_fail_the_gate(env):
    g, p = _sample()
    for b in p["branches"] + p["predicted_counterfactuals"]:
        extra = copy.deepcopy(b["chain"][-1])
        extra["timestamp"] = 3.5
        extra["depth"] = 7
        b["chain"].append(extra)
    out = _score(env, g, p)
    assert out["gates"]["kinematic"] == len(p["branches"]) and out["l4_valid"] == [False]


def test_non_finite_numbers_fail_the_gate(env):
    # `speed` is not listed: canonicalisation re-derives it from vx/vy, so a
    # NaN speed never reaches the scorer.
    for field, where in (("timestamp", "node"), ("x", "actor"), ("vx", "actor")):
        g, p = _sample()
        b = p["branches"][1]  # the coast branch runs several nodes
        assert len(b["chain"]) >= 2
        if where == "node":
            b["chain"][1]["timestamp"] = float("nan")
        else:
            b["chain"][1]["actors_update"][0][field] = float("nan")
        out = _score(env, g, p)
        assert out["gates"]["kinematic"] >= 1, field
        canon = canonicalize_branch(_pre_normalize_branch(b, env["ontology"]), env["ontology"], env["config"])
        ok, msgs = check_kinematic_validity(canon, env["config"], root=g["observed_state"])
        assert not ok and any("not finite" in m for m in msgs), (field, msgs)


# ── Horizon follows the declared track; prefix scoring keeps its override ──

def _challenge_sample():
    spec = dict(scenario_id="scorer10_challenge", n_steps=12, dt=.5, sample_interval=.5, branches=[
        dict(branch_id="hold", is_factual=True, intervention_type="do_ego_no_action"),
        dict(branch_id="coast", is_factual=False, intervention_type="do_ego_decelerate_coast")])
    root = _root()
    root["actors"][1]["x"] = 200.0  # no collision inside 6 s
    root["actors"][2]["x"] = 200.0
    g = build_twin_gt(root, spec)
    g["protocol"] = _protocol_block(track="challenge_6s")
    l4 = copy.deepcopy(g["branches"][0])  # the 12-node hold branch
    l4["intervention"] = dict(intervention_type="npc_sudden_brake", target_actor_id="npc_1", trigger_depth=1)
    g["l4_queries"] = [{"intervention_type": "npc_sudden_brake",
                        "counterfactual_chain": {k: l4[k] for k in ("intervention", "chain")}}]
    p = dict(branches=copy.deepcopy(g["branches"]),
             predicted_counterfactuals=[{k: copy.deepcopy(l4[k]) for k in ("intervention", "chain")}])
    return g, p


def test_reference_future_steps_from_block_or_track():
    from ctms.ctms import reference_future_steps, apply_reference_gate_policy
    assert reference_future_steps({"protocol": _protocol_block(track="challenge_6s")}) == 12
    assert reference_future_steps({"protocol": {"protocol_version": "legacy", "track": "primary"}}) == 6
    assert reference_future_steps({"branches": []}) is None
    base = {"bounds": {"D_max": 6}}
    assert apply_reference_gate_policy(base, {"protocol": _protocol_block(track="challenge_6s")})["bounds"]["D_max"] == 12
    assert apply_reference_gate_policy(base, {"protocol": _protocol_block(track="challenge_6s")}, horizon=4)["bounds"]["D_max"] == 4
    assert apply_reference_gate_policy(base, {"branches": []})["bounds"]["D_max"] == 6


def test_qc_and_score_sample_use_the_track_horizon():
    """A challenge_6s reference with 12 nodes must not be cut at node 7 by the
    shipped D_max = 6, in QC or in score_sample; a primary reference with a
    7th node still is."""
    from simulation.qc import qc_scenario_gt
    from run_benchmark import score_sample, prefix_scores
    shipped = load_ctms_environment("ctms/ontology.yaml", "ctms/ctms_config.yaml",
                                    "ctms/schema_branch.json", "ctms/schema_counterfactual.json")
    assert shipped["config"]["bounds"]["D_max"] == 6
    g, p = _challenge_sample()
    # the hold branch runs all 12 nodes; the coast branch ends in an earlier safe_stop
    assert max(len(b["chain"]) for b in g["branches"]) == 12, [len(b["chain"]) for b in g["branches"]]
    assert len(g["l4_queries"][0]["counterfactual_chain"]["chain"]) == 12
    ok, reasons = qc_scenario_gt(g, shipped["ontology"], shipped["config"], shipped["branch_schema"])
    assert ok, reasons
    r = score_sample(g, copy.deepcopy(p), shipped)
    assert r["layer3"]["diagnostics"]["gate_failure_counts"]["kinematic"] == 0
    assert all(x["valid"] for x in r["layer4"]["results"])
    # prefix scoring still truncates and scores at its own depth
    horizons = prefix_scores(g, copy.deepcopy(p), shipped)
    assert set(horizons) == {"1s", "2s", "3s", "6s"}
    assert horizons["6s"]["CTMS"] > 0 and horizons["1s"]["CTMS"] > 0
    assert shipped["config"]["bounds"]["D_max"] == 6  # env never mutated
    # primary reference declaring 6 steps: a 7th node is beyond the horizon
    root = _root()
    root["actors"][1]["x"] = 200.0
    root["actors"][2]["x"] = 200.0
    g6 = build_twin_gt(root, dict(scenario_id="scorer10_primary_hold", n_steps=6, dt=.5, sample_interval=.5,
                                  branches=[dict(branch_id="hold", is_factual=True, intervention_type="do_ego_no_action")]))
    g6["protocol"] = _protocol_block()
    hold = g6["branches"][0]
    assert len(hold["chain"]) == 6 and not hold["chain"][-1].get("terminal")
    extra = copy.deepcopy(hold["chain"][-1]); extra["timestamp"] = 3.5; extra["depth"] = 7
    hold["chain"].append(extra)
    ok, reasons = qc_scenario_gt(g6, shipped["ontology"], shipped["config"], shipped["branch_schema"])
    assert not ok and any("beyond the 6-node horizon" in m for r_ in reasons for m in r_.get("kinematic_msgs", []))
