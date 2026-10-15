"""Parameter-perturbation check (dissertation 5.4): perturb the metric's free
parameters (gamma, layer1.cat_weight, pairwise lambda/mu/nu) around their
defaults, re-rank the synthetic fleet, and report Kendall tau against the
default-config ranking. Leaderboard order must not hinge on the exact values."""

import copy
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


@pytest.fixture(scope="module")
def gts():
    return make_synthetic_gts(3)


def test_one_result_per_variant(env, gts):
    from analysis.param_perturbation import run_param_perturbation
    grid = {"layer2.gamma": [0.8, 0.95], "layer1.cat_weight": [0.3, 0.7]}
    out = run_param_perturbation(gts, env, grid=grid, seeds=2)
    assert len(out["variants"]) == 4
    for v in out["variants"]:
        assert 0.0 <= v["min_tau"] <= 1.0


def test_identity_variant_gives_tau_one(env, gts):
    from analysis.param_perturbation import run_param_perturbation
    out = run_param_perturbation(gts, env,
                                 grid={"layer2.gamma": [0.9]}, seeds=2)
    assert out["variants"][0]["min_tau"] == pytest.approx(1.0)


def test_env_config_is_not_mutated(env, gts):
    from analysis.param_perturbation import run_param_perturbation
    before = copy.deepcopy(env["config"])
    run_param_perturbation(gts, env, grid={"layer2.gamma": [0.5]}, seeds=1)
    assert env["config"] == before


def test_summary_min_tau_is_min_over_variants(env, gts):
    from analysis.param_perturbation import run_param_perturbation
    grid = {"layer2.gamma": [0.8, 0.9], "layer1.cat_weight": [0.3]}
    out = run_param_perturbation(gts, env, grid=grid, seeds=2)
    assert out["min_tau"] == pytest.approx(
        min(v["min_tau"] for v in out["variants"]))
