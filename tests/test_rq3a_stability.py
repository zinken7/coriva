"""
RQ3a — does the order-invariant structured metric (CTMS Layer 3 F1) produce
stable model rankings under controlled perturbations?

Self-contained: a synthetic GT is built via the kinematic twin (no nuScenes),
a quality ladder of predictions is derived from it, and we check:
  1. order-invariance  (shuffling branches -> identical score)
  2. entity-alias / format-invariance (synonyms canonicalized away)
  3. ranking stability under bounded coordinate jitter (Kendall tau == 1)
"""

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
    from analysis.rq3a import make_study_gt
    return make_study_gt()


def test_perfect_prediction_scores_high(gt, env):
    from analysis.rq3a import score_f1
    import copy
    assert score_f1(gt, copy.deepcopy(gt["branches"]), env) > 0.9


def test_order_invariance(gt, env):
    import random
    from analysis.rq3a import score_f1, shuffle_branches
    base = score_f1(gt, gt["branches"], env)
    for seed in range(5):
        shuffled = shuffle_branches(gt["branches"], random.Random(seed))
        assert score_f1(gt, shuffled, env) == pytest.approx(base)


def test_entity_alias_format_invariance(gt, env):
    from analysis.rq3a import score_f1, reformat_entity_aliases
    base = score_f1(gt, gt["branches"], env)
    aliased = reformat_entity_aliases(gt["branches"])
    assert score_f1(gt, aliased, env) == pytest.approx(base)


def test_quality_ladder_is_monotone(gt, env):
    from analysis.rq3a import build_quality_ladder, score_f1
    ladder = build_quality_ladder(gt)
    scores = [score_f1(gt, preds, env) for _name, preds in ladder]
    assert all(scores[i] >= scores[i + 1] for i in range(len(scores) - 1)), scores
    assert scores[0] > scores[-1], scores


def test_ranking_stable_under_jitter(gt, env):
    from analysis.rq3a import run_stability_study
    result = run_stability_study(gt, env, sigma=0.25, seeds=8)
    assert result["min_kendall_tau"] == pytest.approx(1.0), result
    assert result["rank_inversions"] == 0, result


# ── dataset-scale fleet ranking stability (the headline RQ3a result) ──
def test_make_synthetic_gts_are_distinct_and_valid(env):
    from analysis.rq3a import make_synthetic_gts
    from simulation.qc import qc_scenario_gt
    gts = make_synthetic_gts(6)
    assert len(gts) == 6
    for g in gts:
        ok, reasons = qc_scenario_gt(g, env["ontology"], env["config"], env["branch_schema"])
        assert ok, reasons


def test_fleet_ranking_stable_over_dataset(env):
    from analysis.rq3a import make_synthetic_gts, run_fleet_stability
    gts = make_synthetic_gts(6)
    r = run_fleet_stability(gts, env, seeds=10)
    assert r["base_order"] == ["perfect", "mild", "moderate", "heavy"], r["base_order"]
    assert r["min_kendall_tau"] == pytest.approx(1.0), r
    assert r["rank_inversions"] == 0, r


def test_invariance_holds_across_dataset(env):
    from analysis.rq3a import make_synthetic_gts, score_f1, shuffle_branches, reformat_entity_aliases
    import random
    for g in make_synthetic_gts(4):
        base = score_f1(g, g["branches"], env)
        assert score_f1(g, shuffle_branches(g["branches"], random.Random(0)), env) == pytest.approx(base)
        assert score_f1(g, reformat_entity_aliases(g["branches"]), env) == pytest.approx(base)
