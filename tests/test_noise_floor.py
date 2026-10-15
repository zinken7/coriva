"""Metric noise floor (outline VI.2b, dissertation 3.6.2): perturb ground-truth
trajectories with the empirical fidelity-error distribution, re-score a fixed
prediction set, and report delta_noise = std of the resulting scores.
Leaderboard differences below delta_noise are not ranking claims."""

import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from analysis.rq3a import make_synthetic_gts
from ctms.ctms import load_ctms_environment


@pytest.fixture(scope="module")
def env():
    return load_ctms_environment("ctms/ontology.yaml", "ctms/ctms_config.yaml",
                                 "ctms/schema_branch.json",
                                 "ctms/schema_counterfactual.json")


def test_zero_epsilon_gives_zero_noise_floor(env):
    from analysis.noise_floor import estimate_noise_floor
    gts = make_synthetic_gts(2)
    out = estimate_noise_floor(gts, epsilons=[0.0], env=env, n_trials=5, seed=0)
    assert out["delta_noise"] == pytest.approx(0.0)
    assert len(out["trial_scores"]) == 5


def test_positive_epsilon_produces_positive_noise_floor(env):
    from analysis.noise_floor import estimate_noise_floor
    gts = make_synthetic_gts(2)
    out = estimate_noise_floor(gts, epsilons=[1.0, 1.5], env=env,
                               n_trials=8, seed=0)
    assert out["delta_noise"] > 0.0


def test_deterministic_given_seed(env):
    from analysis.noise_floor import estimate_noise_floor
    gts = make_synthetic_gts(1)
    a = estimate_noise_floor(gts, epsilons=[0.8], env=env, n_trials=4, seed=42)
    b = estimate_noise_floor(gts, epsilons=[0.8], env=env, n_trials=4, seed=42)
    assert a["trial_scores"] == b["trial_scores"]


def test_perturbation_preserves_structure():
    from analysis.noise_floor import perturb_gt
    import random
    gt = make_synthetic_gts(1)[0]
    pert = perturb_gt(gt, epsilons=[2.0], rng=random.Random(0))
    assert len(pert["branches"]) == len(gt["branches"])
    for b0, b1 in zip(gt["branches"], pert["branches"]):
        assert len(b0["chain"]) == len(b1["chain"])
        for n0, n1 in zip(b0["chain"], b1["chain"]):
            ids0 = [a["id"] for a in n0["actors_update"]]
            ids1 = [a["id"] for a in n1["actors_update"]]
            assert ids0 == ids1
    # and it actually moved something
    moved = any(
        a0["x"] != a1["x"] or a0["y"] != a1["y"]
        for b0, b1 in zip(gt["branches"], pert["branches"])
        for n0, n1 in zip(b0["chain"], b1["chain"])
        for a0, a1 in zip(n0["actors_update"], n1["actors_update"]))
    assert moved
