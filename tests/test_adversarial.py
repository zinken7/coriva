"""
Adversarial predictions test suite.

Every test here plays an exploit a motivated submitter (or a degenerate
model) could try, and pins what the metric does about it. RQ3a's synthetic
perturbations stay inside the gates by construction; these deliberately
do not.

Covered: factual-only submission (the copy-factual floor, subtracted by the
skill score), input echo, duplicate spam, hallucinated actors,
off-vocabulary labels, shifted chains (positional-alignment brittleness,
documented), Hungarian tie order-invariance, and multiple chains for one
intervention (audit L2 — spec says exactly one).
"""

import copy
import json
import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


@pytest.fixture(scope="module")
def env():
    import yaml
    with open(os.path.join("ctms", "ontology.yaml")) as f:
        ontology = yaml.safe_load(f)
    with open(os.path.join("ctms", "ctms_config.yaml")) as f:
        config = yaml.safe_load(f)
    with open(os.path.join("ctms", "schema_branch.json")) as f:
        branch_schema = json.load(f)
    with open(os.path.join("ctms", "schema_counterfactual.json")) as f:
        cf_schema = json.load(f)
    return {"ontology": ontology, "config": config,
            "branch_schema": branch_schema, "cf_schema": cf_schema}


@pytest.fixture(scope="module")
def gt():
    with open(os.path.join("output", "scenario_1", "ground_truth.json")) as f:
        return json.load(f)


def _score_l3(gt, preds, env):
    from ctms.ctms import score_layer3
    return score_layer3(
        gt_branches=gt["branches"], pred_branches=preds,
        ontology=env["ontology"], config=env["config"],
        branch_schema=env["branch_schema"], root=gt.get("observed_state"),
    )


def _factual(gt):
    return copy.deepcopy(next(b for b in gt["branches"] if b.get("is_factual")))


# ── Exploit 1: submit only the factual branch ───────────────────────────
# The trivial continuation is in the GT pool, so it buys one perfect match:
# P = 1, R = 1/n_gt, F1 = 2/(n_gt+1). This structural floor is exactly what
# the copy_factual reference baseline measures and the skill score subtracts.

def test_factual_only_floor_is_two_over_n_plus_one(gt, env):
    res = _score_l3(gt, [_factual(gt)], env)
    n_gt = len(gt["branches"])
    assert res["F1_cf"] == pytest.approx(2 / (n_gt + 1), abs=1e-6)


# ── Exploit 2: echo the input ───────────────────────────────────────────

def test_input_echo_scores_zero(gt, env):
    """A branch that only repeats the given observed_state root canonicalizes
    to an empty chain: F1 contribution is exactly 0."""
    root = copy.deepcopy(gt["observed_state"])
    echo_branch = {"branch_id": "echo", "chain": [copy.deepcopy(root) for _ in range(5)]}
    res = _score_l3(gt, [echo_branch], env)
    assert res["F1_cf"] == 0.0


# ── Exploit 3: duplicate spam ───────────────────────────────────────────

def test_duplicate_spam_buys_nothing(gt, env):
    single = _score_l3(gt, [_factual(gt)], env)
    spam = _score_l3(gt, [_factual(gt) for _ in range(20)], env)
    assert spam["F1_cf"] == pytest.approx(single["F1_cf"])
    assert spam["diagnostics"]["pred_duplicates_collapsed"] == 19
    assert spam["diagnostics"].get("k_max_penalty", 1.0) == pytest.approx(1.0)


# ── Exploit 4: hallucinated actor ───────────────────────────────────────

def _echo_l3(gt, env):
    """The exact-copy score of the fixture. The
    `output/` fixture keeps a branch that reaches the horizon without a
    terminal event, and two non-terminal branches earn S_term = 0, so the
    ceiling is below 1.0 (dissertation §5.6); the adversarial properties are
    relative to that ceiling, not to 1.0."""
    return _score_l3(gt, [copy.deepcopy(b) for b in gt["branches"]], env)


def test_hallucinated_actor_zeroes_that_branch_only(gt, env):
    ceiling = _echo_l3(gt, env)
    good = [copy.deepcopy(b) for b in gt["branches"]]
    ghost = copy.deepcopy(gt["branches"][1])
    ghost["branch_id"] = "ghost"
    ghost["chain"][0]["actors_update"].append(
        {"id": "ghost_car", "entity_type": "vehicle", "role_type": "lead_vehicle",
         "x": 0.0, "y": 0.0, "vx": 0.0, "vy": 0.0, "speed": 0.0})
    res = _score_l3(gt, good + [ghost], env)
    assert res["diagnostics"]["gate_failure_counts"]["semantic"] == 1
    # the three good branches still earn the echo's recall; only precision pays
    assert res["Recall_cf"] == pytest.approx(ceiling["Recall_cf"], abs=1e-9)
    assert res["F1_cf_valid_only"] == pytest.approx(ceiling["F1_cf"], abs=1e-9)
    assert res["F1_cf"] < ceiling["F1_cf"]


# ── Exploit 5: off-vocabulary labels ────────────────────────────────────

def test_off_vocabulary_event_zeroes_that_branch_only(gt, env):
    ceiling = _echo_l3(gt, env)
    good = [copy.deepcopy(b) for b in gt["branches"]]
    bad = copy.deepcopy(gt["branches"][1])
    bad["branch_id"] = "offvocab"
    bad["chain"][0]["event_type"] = "hyperspace_jump"
    res = _score_l3(gt, good + [bad], env)
    assert res["diagnostics"]["gate_failure_counts"]["schema"] == 1
    assert res["Recall_cf"] == pytest.approx(ceiling["Recall_cf"], abs=1e-9)
    assert res["F1_cf"] < ceiling["F1_cf"]


# ── Known limitation: positional alignment is shift-sensitive ───────────
# A chain missing its FIRST node shifts every comparison. Where consecutive
# nodes share an event label, Sim_cat keeps a 0.5 floor per node (the event
# gate passes), so the chain term degrades but does not vanish; where labels
# change, the hard event gate plus prefix-min zero the tail. Pinned here as
# documentation: a shifted factual chain scores strictly below the intact
# one, and well below full credit. Future work: edit-distance alignment.

def test_one_node_shifted_chain_scores_strictly_below_intact(gt, env):
    intact = _score_l3(gt, [_factual(gt)], env)
    shifted_branch = _factual(gt)
    shifted_branch["chain"] = shifted_branch["chain"][1:]
    shifted = _score_l3(gt, [shifted_branch], env)
    assert shifted["F1_cf"] < intact["F1_cf"]
    assert shifted["matches"][0]["S_pair"] < 0.75  # pinned: ~0.50 today


# ── Hungarian ties: order cannot change the score ───────────────────────

def test_tie_breaking_is_order_invariant(gt, env):
    a = _factual(gt)
    a["branch_id"] = "tie_a"
    b = _factual(gt)
    b["branch_id"] = "tie_b"
    for node in b["chain"]:
        for actor in node.get("actors_update", []) or []:
            if actor.get("y") is not None:
                actor["y"] = float(actor["y"]) + 1.0  # equidistant-ish twin
    r1 = _score_l3(gt, [a, b], env)
    r2 = _score_l3(gt, [b, a], env)
    assert r1["F1_cf"] == pytest.approx(r2["F1_cf"], abs=1e-12)


# ── Audit L2: multiple chains for one intervention → that CF scores 0 ───
# Outline §V.6: "the model must predict exactly one counterfactual chain.
# If the output contains zero chains or more than one chain, the prediction
# is invalid." The old lookup silently kept the LAST duplicate instead.

def _echo_pred(gt):
    # Mirror the field the scorer scores: protocol-aware references declare
    # the named-actor queries in l4_queries; others keep counterfactuals.
    queries = gt.get("l4_queries") or gt["counterfactuals"]
    return {
        "branches": copy.deepcopy(gt["branches"]),
        "predicted_counterfactuals": [
            {"intervention": copy.deepcopy(cf["counterfactual_chain"]["intervention"]),
             "chain": copy.deepcopy(cf["counterfactual_chain"]["chain"])}
            for cf in queries
        ],
    }


def test_duplicate_intervention_chains_invalidate_that_intervention(gt, env):
    from ctms.ctms import evaluate_ctms
    pred = _echo_pred(gt)
    # duplicate the first counterfactual: two chains for one intervention_type
    pred["predicted_counterfactuals"].append(
        copy.deepcopy(pred["predicted_counterfactuals"][0]))
    res = evaluate_ctms(gt, pred, env["ontology"], env["config"],
                        env["branch_schema"], env["cf_schema"])
    dup_type = pred["predicted_counterfactuals"][0]["intervention"]["intervention_type"]
    per_cf = {r["intervention_type"]: r for r in res["layer4"]["results"]}
    assert per_cf[dup_type]["Score_interv"] == 0.0
    assert per_cf[dup_type]["valid"] is False
    # the other intervention is untouched
    others = [r for t, r in per_cf.items() if t != dup_type]
    assert all(r["Score_interv"] == pytest.approx(1.0, abs=1e-9) for r in others)


def test_single_chain_per_intervention_still_scores_full(gt, env):
    from ctms.ctms import evaluate_ctms
    res = evaluate_ctms(gt, _echo_pred(gt), env["ontology"], env["config"],
                        env["branch_schema"], env["cf_schema"])
    assert res["Counterfactual_Alignment_Score"] == pytest.approx(1.0, abs=1e-9)
