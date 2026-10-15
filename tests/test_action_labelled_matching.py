"""Layer 3 matching must respect the action label.

A label-agnostic Hungarian assignment would let a prediction that swapped the
chains of two declared actions earn full F1 because each chain finds its
look-alike reference. Protocol-1.0 branches carry explicit intervention
metadata, so a pair whose ego-action labels differ scores zero (config
``layer3.match_by_action``). Label-agnostic matching is available only behind
the toggle for pilot artifacts.
"""

import copy
import os
import sys

import pytest

ROOT = os.path.join(os.path.dirname(__file__), "..")
sys.path.insert(0, ROOT)

from analysis.rq3a import make_synthetic_gts, score_f1  # noqa: E402
from ctms.ctms import SCORER_VERSION, load_ctms_environment, score_layer3  # noqa: E402


def _env():
    return load_ctms_environment(os.path.join(ROOT, "ctms/ontology.yaml"),
                                 os.path.join(ROOT, "ctms/ctms_config.yaml"),
                                 os.path.join(ROOT, "ctms/schema_branch.json"),
                                 os.path.join(ROOT, "ctms/schema_counterfactual.json"))


def _swapped(branches):
    out = copy.deepcopy(branches)
    a = next(b for b in out if b["intervention"]["intervention_type"] == "do_ego_no_action")
    b = next(b for b in out if b["intervention"]["intervention_type"] == "do_ego_swerve_left")
    a["intervention"], b["intervention"] = b["intervention"], a["intervention"]
    return out


def test_scorer_version_bumped_for_label_constrained_matching():
    assert SCORER_VERSION == "ctms-scorer-1.3"
    assert _env()["config"]["version"] == "1.3.0"
    assert _env()["config"]["layer3"]["match_by_action"] is True


def test_swapping_action_labels_loses_credit_under_default_config():
    # An exact copy is the ceiling for this GT (non-terminating branches earn no
    # S_term by spec H2); attaching the right chains to the wrong labels must
    # fall well below that ceiling.
    gt = make_synthetic_gts(1)[0]
    env = _env()
    ceiling = score_f1(gt, copy.deepcopy(gt["branches"]), env)
    assert ceiling > 0.9
    assert score_f1(gt, _swapped(gt["branches"]), env) < ceiling - 0.3


def test_label_agnostic_matching_survives_only_behind_the_legacy_toggle():
    # With the toggle off: swapped labels earn exactly the exact-copy credit.
    gt = make_synthetic_gts(1)[0]
    env = _env()
    ceiling = score_f1(gt, copy.deepcopy(gt["branches"]), env)
    env["config"]["layer3"]["match_by_action"] = False
    assert score_f1(gt, _swapped(gt["branches"]), env) == pytest.approx(ceiling)


def test_diagnostics_record_label_constraint_and_zeroed_pairs():
    gt = make_synthetic_gts(1)[0]
    env = _env()
    r = score_layer3(gt["branches"], _swapped(gt["branches"]), env["ontology"], env["config"],
                     env["branch_schema"], root=gt["observed_state"])
    assert r["diagnostics"]["match_by_action"] is True
    assert r["diagnostics"]["label_mismatch_pairs_zeroed"] > 0


def test_branches_without_labels_are_not_constrained():
    # Predictions without an intervention object keep the unconstrained match so
    # pilot fixtures remain scoreable; the metadata gate handles protocol runs.
    gt = make_synthetic_gts(1)[0]
    env = _env()
    unlabelled = copy.deepcopy(gt["branches"])
    for b in unlabelled:
        b.pop("intervention", None)
    constrained = score_f1(gt, unlabelled, env)
    env["config"]["layer3"]["match_by_action"] = False
    assert constrained == pytest.approx(score_f1(gt, unlabelled, env))
    assert constrained > 0.9
