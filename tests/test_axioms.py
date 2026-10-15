"""
Axiomatic characterisation of CTMS and the coherence family,
encoded as executable property tests: an axiom you cannot test is a claim; an
axiom that passes here is verified against the released scorer.

The axioms fix the FORM of each metric; the numeric constants (gamma, tau,
lambda/mu/nu, K_max) are the residual free scalars, shown ranking-inert by a
parameter-perturbation study.
"""

import copy
import json
import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


@pytest.fixture(scope="module")
def env():
    from ctms.ctms import load_ctms_environment
    return load_ctms_environment("ctms/ontology.yaml", "ctms/ctms_config.yaml",
                                 "ctms/schema_branch.json", "ctms/schema_counterfactual.json")


@pytest.fixture(scope="module")
def gt():
    with open(os.path.join("output", "scenario_1", "ground_truth.json")) as f:
        return json.load(f)


def _all_gts():
    """Every GT scenario available, so invariance/ceiling axioms are checked as
    PROPERTIES over many scenarios, not on a single fixture (audit #6)."""
    import glob
    out = []
    for pat in ("output/scenario_*/ground_truth.json", "pilot30/scenario_*/ground_truth.json"):
        for p in sorted(glob.glob(pat)):
            try:
                out.append(json.load(open(p)))
            except Exception:
                pass
    return out


_GTS = _all_gts()
# id list for parametrisation labels
_GT_IDS = [g.get("scenario_id", str(i)) for i, g in enumerate(_GTS)]


def _echo_branches(gt):
    return copy.deepcopy(gt["branches"])


# ══ CTMS axioms ═══════════════════════════════════════════════════════════

# AX1 — Order invariance: the score does not depend on the order in which
#       predicted branches are listed (forced by Hungarian matching + sort).
#       Checked as a PROPERTY over every available scenario (audit #6).
@pytest.mark.parametrize("scen", _GTS, ids=_GT_IDS)
def test_ax1_order_invariance(scen, env):
    from ctms.ctms import score_layer3
    preds = _echo_branches(scen)
    a = score_layer3(scen["branches"], preds, env["ontology"], env["config"],
                     env["branch_schema"], root=scen["observed_state"])["F1_cf"]
    b = score_layer3(scen["branches"], list(reversed(preds)), env["ontology"], env["config"],
                     env["branch_schema"], root=scen["observed_state"])["F1_cf"]
    assert a == pytest.approx(b)


# AX2 — Input non-reward: reproducing the given observed_state root earns
#       nothing (forced by root stripping in canonicalisation).
def test_ax2_input_non_reward(gt, env):
    from ctms.ctms import canonicalize_branch, score_chain
    gt_canon = canonicalize_branch(copy.deepcopy(gt["branches"][0]), env["ontology"], env["config"])
    echo = canonicalize_branch({"chain": [copy.deepcopy(gt["observed_state"])]},
                               env["ontology"], env["config"])
    assert echo["chain"] == []
    assert score_chain(gt_canon, echo, env["ontology"], env["config"]) == 0.0


# AX3 — Duplicate non-reward: duplicating a predicted branch does not raise the
#       score (forced by canonical-string dedup). Property over all scenarios.
@pytest.mark.parametrize("scen", _GTS, ids=_GT_IDS)
def test_ax3_duplicate_non_reward(scen, env):
    from ctms.ctms import score_layer3
    preds = _echo_branches(scen)
    single = score_layer3(scen["branches"], preds, env["ontology"], env["config"],
                          env["branch_schema"], root=scen["observed_state"])["F1_cf"]
    dup = score_layer3(scen["branches"], preds + copy.deepcopy(preds), env["ontology"],
                       env["config"], env["branch_schema"], root=scen["observed_state"])["F1_cf"]
    assert dup == pytest.approx(single)


# AX4 — Perfect-prediction ceiling: an exact (canonical) echo of the ground
#       truth scores 1 (forced by both-sided canonicalisation).
#       Property over all scenarios.
# pilot30 fixtures (built without the root-seeded kinematic gate in QC) whose
# own reference violates the T0 -> first-node bounds that the scorer checks:
# NPC accelerations above 10 m/s^2 between the root and the first node, and
# in 0028 a 20 m jump by a "stationary" actor. Any other fixture that the
# gate rejects is a regression.
_LEGACY_ROOT_STEP_VIOLATIONS = {
    "nuscenes_0017", "nuscenes_0018", "nuscenes_0022", "nuscenes_0023",
    "nuscenes_0024", "nuscenes_0028", "nuscenes_0029", "nuscenes_0030",
}


@pytest.mark.parametrize("scen", _GTS, ids=_GT_IDS)
def test_ax4_perfect_prediction_ceiling(scen, env):
    from ctms.ctms import (score_layer3, check_kinematic_validity, canonicalize_branch,
                           _pre_normalize_branch)
    sid = scen.get("scenario_id")
    rejected = []
    for b in scen["branches"]:
        canon = canonicalize_branch(_pre_normalize_branch(b, env["ontology"]), env["ontology"], env["config"])
        ok, msgs = check_kinematic_validity(canon, env["config"], root=scen["observed_state"])
        if not ok:
            rejected.append((b.get("branch_id"), msgs[0]))
    if sid in _LEGACY_ROOT_STEP_VIOLATIONS:
        assert rejected, f"{sid} is listed as a legacy violation but now passes; drop it from the list"
        pytest.skip(f"known legacy fixture {sid}: reference violates the root step ({rejected[0][1]})")
    assert not rejected, f"reference of {sid} unexpectedly rejected by the kinematic gate: {rejected}"
    res = score_layer3(scen["branches"], _echo_branches(scen), env["ontology"], env["config"],
                       env["branch_schema"], root=scen["observed_state"])
    # The ceiling is 1.0 only when every reference branch ends in a terminal
    # event: two matching non-terminal branches earn S_term = 0, so an exact
    # copy of a reference with k non-terminal branches out of n scores
    # 1 - 0.1 * k / n (pair weights 0.7/0.2/0.1; dissertation §5.6).
    terminal_types = set(env["ontology"]["terminal_event_types"])
    non_terminal = sum(1 for b in scen["branches"] if b["chain"][-1].get("event_type") not in terminal_types)
    assert res["Recall_cf"] == pytest.approx(1.0 - 0.1 * non_terminal / len(scen["branches"]), abs=1e-9)
    assert res["F1_cf"] == pytest.approx(res["Recall_cf"], abs=1e-9)  # precision equals recall for an exact copy


# AX5 — Early-error dominance: an identical node error placed EARLIER caps the
#       chain score at least as hard as later (forced by the prefix-min
#       bottleneck). We perturb the factual chain's ego x at depth 1 vs depth 3.
def test_ax5_early_error_dominance(gt, env):
    from ctms.ctms import canonicalize_branch, score_chain
    base = copy.deepcopy(gt["branches"][0])
    gt_canon = canonicalize_branch(copy.deepcopy(base), env["ontology"], env["config"])

    def perturb_at(depth, shift=6.0):
        b = copy.deepcopy(base)
        node = b["chain"][depth - 1]
        for a in node.get("actors_update", []):
            if a.get("id") == "ego":
                a["x"] = float(a.get("x", 0.0)) + shift
        return canonicalize_branch(b, env["ontology"], env["config"])

    if len(base["chain"]) < 3:
        pytest.skip("factual chain too short")
    early = score_chain(gt_canon, perturb_at(1), env["ontology"], env["config"])
    late = score_chain(gt_canon, perturb_at(3), env["ontology"], env["config"])
    assert early <= late + 1e-9


# AX6 — Validity necessity: a branch that violates a gate contributes zero to
#       the matched score (forced by the multiplicative gate G).
def test_ax6_validity_necessity(gt, env):
    from ctms.ctms import score_layer3
    bad = copy.deepcopy(gt["branches"][1])
    bad["chain"][0]["event_type"] = "quantum_leap"  # off-vocabulary -> schema gate
    res = score_layer3(gt["branches"], [bad], env["ontology"], env["config"],
                       env["branch_schema"], root=gt["observed_state"])
    assert res["F1_cf"] == 0.0
    assert res["diagnostics"]["gate_failure_counts"]["schema"] >= 1


# AX7 — Compliance tax non-negativity: the reported score never exceeds the
#       valid-only score (competence is an upper bound on what compliance can
#       cash in). reported <= valid_only.
def test_ax7_compliance_tax_nonnegative(gt, env):
    from ctms.ctms import score_layer3
    good = _echo_branches(gt)
    bad = copy.deepcopy(gt["branches"][1])
    bad["chain"][0]["event_type"] = "quantum_leap"
    res = score_layer3(gt["branches"], good + [bad], env["ontology"], env["config"],
                       env["branch_schema"], root=gt["observed_state"])
    assert res["F1_cf"] <= res["F1_cf_valid_only"] + 1e-9


# AX8 — Proportional overflow (no cliff): more than K_max unique branches scales
#       F1 by K_max/n rather than zeroing it, and is monotone decreasing in n.
def test_ax8_proportional_overflow_no_cliff(gt, env):
    from ctms.ctms import score_layer3
    base = gt["branches"]

    def variant(i):
        b = copy.deepcopy(base[i % len(base)])
        b["branch_id"] = f"v{i}"
        for node in b["chain"]:
            for a in node.get("actors_update", []) or []:
                if a.get("y") is not None:
                    a["y"] = float(a["y"]) + 3.0 * (i + 1)
        return b

    def f1_with(n):
        preds = [copy.deepcopy(b) for b in base] + [variant(i) for i in range(n - len(base))]
        return score_layer3(base, preds[:n], env["ontology"], env["config"],
                            env["branch_schema"], root=gt["observed_state"])["F1_cf"]

    f9, f16 = f1_with(9), f1_with(16)
    assert f9 > 0.0 and f16 > 0.0          # no cliff to zero
    assert f16 < f9                         # monotone decreasing beyond K_max


# ══ Coherence-family axioms ═══════════════════════════════════════════════

def _root():
    return {"actors": [{"id": "ego", "role_type": "ego_agent",
                        "x": 0.0, "y": 0.0, "speed": 8.0}]}


def _chain(x):
    return [{"depth": 1, "actors_update": [{"id": "ego", "role_type": "ego_agent",
                                            "x": x, "y": 0.0}]}]


# CX1 — Reference freedom: coherence is a function of the prediction alone; it
#       takes no ground truth (structural axiom — the signature has no GT arg).
def test_cx1_reference_freedom():
    import inspect
    from analysis import causal_coherence as cc
    params = inspect.signature(cc.coherence_from_pairs).parameters
    assert set(params) == {"pairs", "root"}  # no ground-truth / reference input


# CX2 — Collapse detection: responsiveness is zero iff all interventions give
#       the same ego outcome (the single-continuation failure).
def test_cx2_collapse_detection():
    from analysis import causal_coherence as cc
    collapsed = [("do_ego_panic_brake", _chain(20.0)), ("do_ego_no_action", _chain(20.0))]
    responsive = [("do_ego_panic_brake", _chain(4.0)), ("do_ego_no_action", _chain(24.0))]
    assert cc.responsiveness(collapsed, _root()) == pytest.approx(0.0)
    assert cc.responsiveness(responsive, _root()) > 0.0


# CX3 — Causal-order sensitivity: monotonicity is 1 for the correct
#       braking-displacement order and 0 for the inverted order.
def test_cx3_order_sensitivity():
    from analysis import causal_coherence as cc
    correct = [("do_ego_panic_brake", _chain(4.0)), ("do_ego_no_action", _chain(24.0))]
    inverted = [("do_ego_panic_brake", _chain(24.0)), ("do_ego_no_action", _chain(4.0))]
    assert cc.braking_monotonicity(correct, _root()) == pytest.approx(1.0)
    assert cc.braking_monotonicity(inverted, _root()) == pytest.approx(0.0)


# CX4 — Ground-truth ceiling on reference-tied axioms: the GT scores 1 on
#       braking monotonicity wherever it is defined (coherent by construction).
#       Property over all scenarios (audit #6).
@pytest.mark.parametrize("scen", _GTS, ids=_GT_IDS)
def test_cx4_gt_ceiling_monotonicity(scen):
    from analysis import causal_coherence as cc
    out = cc.coherence_from_pairs(cc.pairs_from_gt(scen), scen["observed_state"])
    if out["monotonicity"] is not None:
        assert out["monotonicity"] == pytest.approx(1.0)
