"""
Metric properties of the scorer.

- The observed_state root is INPUT, never scored: a leading observed_state
  node in any chain (GT or prediction) is stripped during canonicalization,
  so echoing the given root earns nothing and a format difference (root
  included vs not) never misaligns scoring.
- Split depth uses tolerance-based state comparison (reuse τ for position,
  the 1.0 m/s speed-similarity floor for speed) rather than exact equality
  on rounded values, so predicted branches do not all "split" at depth 1
  against the factual baseline.
- K_max overflow is a proportional penalty (× K_max / n_unique), not a
  submission-wide zero.
- Compliance vs reasoning separation: per-gate failure counts and
  valid-only scores are first-class outputs.
"""

import copy
import json
import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


@pytest.fixture(scope="module")
def ontology():
    import yaml
    with open(os.path.join("ctms", "ontology.yaml")) as f:
        return yaml.safe_load(f)


@pytest.fixture(scope="module")
def config():
    import yaml
    with open(os.path.join("ctms", "ctms_config.yaml")) as f:
        return yaml.safe_load(f)


@pytest.fixture(scope="module")
def branch_schema():
    with open(os.path.join("ctms", "schema_branch.json")) as f:
        return json.load(f)


@pytest.fixture(scope="module")
def cf_schema():
    with open(os.path.join("ctms", "schema_counterfactual.json")) as f:
        return json.load(f)


@pytest.fixture(scope="module")
def scenario_gt():
    with open(os.path.join("output", "scenario_1", "ground_truth.json")) as f:
        return json.load(f)


def _root_node(gt):
    """The scenario's observed_state root as a chain node."""
    return copy.deepcopy(gt["observed_state"])


# ── leading observed_state node is stripped, never scored ────────────────

def test_p12_canonicalize_strips_leading_observed_root(scenario_gt, ontology, config):
    from ctms.ctms import canonicalize_branch
    branch = copy.deepcopy(scenario_gt["branches"][0])
    with_root = {"chain": [_root_node(scenario_gt)] + branch["chain"]}
    canon = canonicalize_branch(with_root, ontology, config)
    assert canon["chain"][0].get("event_type") != "observed_state"
    assert len(canon["chain"]) == len(branch["chain"])
    # depths renumber from 1 after the strip
    assert canon["chain"][0].get("depth") == 1


def test_p12_pred_with_prepended_root_scores_full_f1(scenario_gt, ontology, config, branch_schema):
    """Echoing GT branches but prepending the given root must score F1=1.0
    (format-neutral), not be misaligned into a near-zero chain score."""
    from ctms.ctms import score_layer3
    preds = []
    for b in scenario_gt["branches"]:
        p = copy.deepcopy(b)
        p["chain"] = [_root_node(scenario_gt)] + p["chain"]
        preds.append(p)
    res = score_layer3(
        gt_branches=scenario_gt["branches"], pred_branches=preds,
        ontology=ontology, config=config, branch_schema=branch_schema,
        root=scenario_gt.get("observed_state"),
    )
    echo = score_layer3(gt_branches=scenario_gt["branches"], pred_branches=[copy.deepcopy(b) for b in scenario_gt["branches"]],
                        ontology=ontology, config=config, branch_schema=branch_schema, root=scenario_gt.get("observed_state"))
    # format-neutral: the echoed root changes nothing relative to the plain echo
    assert res["F1_cf"] == pytest.approx(echo["F1_cf"], abs=1e-9), res["diagnostics"]


def test_p12_layer4_pred_with_prepended_root_scores_one(scenario_gt, ontology, config, cf_schema):
    from ctms.ctms import score_layer4
    gt_cf = scenario_gt["counterfactuals"][0]
    pred_cf = {"counterfactual_chain": copy.deepcopy(gt_cf["counterfactual_chain"])}
    pred_cf["counterfactual_chain"]["chain"] = (
        [_root_node(scenario_gt)] + pred_cf["counterfactual_chain"]["chain"])
    res = score_layer4(
        gt_cf_chain=gt_cf, pred_cf_obj=pred_cf, ontology=ontology,
        config=config, counterfactual_schema=cf_schema,
        root=scenario_gt.get("observed_state"),
    )
    assert res["valid"] is True, res["diagnostics"]
    assert res["Score_interv"] == pytest.approx(1.0, abs=1e-9)


def test_p12_root_only_chain_scores_zero(scenario_gt, ontology, config):
    """The echo floor is exactly zero: a chain that only reproduces the
    given root has no scoreable content."""
    from ctms.ctms import canonicalize_branch, score_chain
    gt_canon = canonicalize_branch(
        copy.deepcopy(scenario_gt["branches"][0]), ontology, config)
    echo = canonicalize_branch(
        {"chain": [_root_node(scenario_gt)]}, ontology, config)
    assert echo["chain"] == []
    assert score_chain(gt_canon, echo, ontology, config) == 0.0


def test_p12_split_depth_unaffected_by_prepended_root(scenario_gt, ontology, config):
    from ctms.ctms import canonicalize_branch, compute_split_depth
    root = scenario_gt.get("observed_state")
    factual = canonicalize_branch(
        copy.deepcopy(scenario_gt["branches"][0]), ontology, config)
    with_root = canonicalize_branch(
        {"chain": [_root_node(scenario_gt)] + copy.deepcopy(scenario_gt["branches"][0]["chain"])},
        ontology, config)
    d_max = int(config.get("bounds", {}).get("D_max", 10))
    assert compute_split_depth(with_root, factual, config, ontology, root=root) == d_max + 1


# ── tolerance-based split-depth comparison ───────────────────────────────

def _jitter_positions(branch, delta):
    b = copy.deepcopy(branch)
    for node in b["chain"]:
        for actor in node.get("actors_update", []) or []:
            if actor.get("x") is not None:
                actor["x"] = float(actor["x"]) + delta
    return b


def test_p15_sub_tolerance_jitter_does_not_split(scenario_gt, ontology, config):
    """A prediction whose states sit within τ of the factual baseline has NOT
    diverged: sub-tolerance jitter must not force split depth 1."""
    from ctms.ctms import canonicalize_branch, compute_split_depth
    root = scenario_gt.get("observed_state")
    factual = canonicalize_branch(
        copy.deepcopy(scenario_gt["branches"][0]), ontology, config)
    jittered = canonicalize_branch(
        _jitter_positions(scenario_gt["branches"][0], 0.3), ontology, config)
    d_max = int(config.get("bounds", {}).get("D_max", 10))
    assert compute_split_depth(jittered, factual, config, ontology, root=root) == d_max + 1


def test_p15_super_tolerance_divergence_still_splits(scenario_gt, ontology, config):
    """Divergence beyond τ at depth 1 must still split at depth 1."""
    from ctms.ctms import canonicalize_branch, compute_split_depth
    root = scenario_gt.get("observed_state")
    factual = canonicalize_branch(
        copy.deepcopy(scenario_gt["branches"][0]), ontology, config)
    far = canonicalize_branch(
        _jitter_positions(scenario_gt["branches"][0], 5.0), ontology, config)
    assert compute_split_depth(far, factual, config, ontology, root=root) == 1


def test_p15_categorical_divergence_splits_regardless_of_tolerance(scenario_gt, ontology, config):
    """event_type change at depth 1 splits at depth 1 even with identical states."""
    from ctms.ctms import canonicalize_branch, compute_split_depth
    root = scenario_gt.get("observed_state")
    factual = canonicalize_branch(
        copy.deepcopy(scenario_gt["branches"][0]), ontology, config)
    other = copy.deepcopy(scenario_gt["branches"][0])
    other["chain"][0]["event_type"] = (
        "maintain_course" if other["chain"][0]["event_type"] != "maintain_course"
        else "ego_brake")
    other = canonicalize_branch(other, ontology, config)
    assert compute_split_depth(other, factual, config, ontology, root=root) == 1


def test_p15_gt_vs_gt_split_depths_unchanged(scenario_gt, ontology, config):
    """Tolerance comparison is backward compatible on GT: identical prefixes
    still match exactly, so GT-vs-factual split depths keep their values."""
    from ctms.ctms import canonicalize_branch, compute_split_depth
    root = scenario_gt.get("observed_state")
    branches = [canonicalize_branch(copy.deepcopy(b), ontology, config)
                for b in scenario_gt["branches"]]
    factual = branches[0]
    d_max = int(config.get("bounds", {}).get("D_max", 10))
    # factual vs itself: never splits
    assert compute_split_depth(factual, factual, config, ontology, root=root) == d_max + 1
    # non-factual GT branches diverge at some real depth ≥ 1, ≤ d_max
    for b in branches[1:]:
        d = compute_split_depth(b, factual, config, ontology, root=root)
        assert 1 <= d <= d_max


# ── K_max overflow → proportional penalty, not zero ──────────────────────

def _unique_variant(branch, idx):
    """A schema-valid, semantically distinct variant of a GT branch."""
    b = copy.deepcopy(branch)
    b["branch_id"] = f"variant_{idx}"
    for node in b["chain"]:
        for actor in node.get("actors_update", []) or []:
            if actor.get("y") is not None:
                actor["y"] = float(actor["y"]) + 3.0 * (idx + 1)
    return b


def _overflow_result(scenario_gt, ontology, config, branch_schema, n_pred):
    from ctms.ctms import score_layer3
    base = scenario_gt["branches"]
    preds = [copy.deepcopy(b) for b in base]
    i = 0
    while len(preds) < n_pred:
        preds.append(_unique_variant(base[i % len(base)], i))
        i += 1
    return score_layer3(
        gt_branches=base, pred_branches=preds[:n_pred],
        ontology=ontology, config=config, branch_schema=branch_schema,
        root=scenario_gt.get("observed_state"),
    )


def test_p13_at_k_max_no_penalty(scenario_gt, ontology, config, branch_schema):
    res = _overflow_result(scenario_gt, ontology, config, branch_schema, 8)
    assert res["F1_cf"] > 0.0
    assert res["diagnostics"].get("k_max_penalty", 1.0) == pytest.approx(1.0)


def test_p13_k_max_plus_one_scaled_not_zeroed(scenario_gt, ontology, config, branch_schema):
    res9 = _overflow_result(scenario_gt, ontology, config, branch_schema, 9)
    assert res9["F1_cf"] > 0.0, "one branch over K_max must not zero the submission"
    assert res9["diagnostics"]["k_max_penalty"] == pytest.approx(8 / 9)


def test_p13_spam_is_monotonically_worse(scenario_gt, ontology, config, branch_schema):
    """More overflow → strictly lower score: 16 branches score below 9."""
    f9 = _overflow_result(scenario_gt, ontology, config, branch_schema, 9)["F1_cf"]
    f16 = _overflow_result(scenario_gt, ontology, config, branch_schema, 16)["F1_cf"]
    assert 0.0 < f16 < f9


def test_p13_penalty_factor_is_k_over_n(scenario_gt, ontology, config, branch_schema):
    res16 = _overflow_result(scenario_gt, ontology, config, branch_schema, 16)
    assert res16["diagnostics"]["k_max_penalty"] == pytest.approx(0.5)


# ── compliance vs reasoning separation ───────────────────────────────────

def test_p11_gate_failure_counts_by_type(scenario_gt, ontology, config, branch_schema):
    """One off-vocabulary branch among good ones: the submission is scored,
    and the failure is counted by gate type."""
    from ctms.ctms import score_layer3
    preds = [copy.deepcopy(b) for b in scenario_gt["branches"]]
    bad = copy.deepcopy(scenario_gt["branches"][1])
    bad["branch_id"] = "bad"
    bad["chain"][0]["event_type"] = "quantum_leap"  # off-vocabulary
    preds.append(bad)
    res = score_layer3(
        gt_branches=scenario_gt["branches"], pred_branches=preds,
        ontology=ontology, config=config, branch_schema=branch_schema,
        root=scenario_gt.get("observed_state"),
    )
    diag = res["diagnostics"]
    assert diag["gate_failure_counts"]["schema"] >= 1
    assert diag["pred_valid_count"] == len(scenario_gt["branches"])
    assert res["F1_cf"] > 0.0


def test_p11_valid_only_f1_reported(scenario_gt, ontology, config, branch_schema):
    """F1_cf_valid_only conditions precision on gate-passing branches: with
    3 perfect + 1 gated-out branch, valid-only F1 is 1.0 while overall
    precision pays for the invalid branch."""
    from ctms.ctms import score_layer3
    preds = [copy.deepcopy(b) for b in scenario_gt["branches"]]
    bad = copy.deepcopy(scenario_gt["branches"][1])
    bad["branch_id"] = "bad"
    bad["chain"][0]["event_type"] = "quantum_leap"
    preds.append(bad)
    res = score_layer3(
        gt_branches=scenario_gt["branches"], pred_branches=preds,
        ontology=ontology, config=config, branch_schema=branch_schema,
        root=scenario_gt.get("observed_state"),
    )
    echo = score_layer3(gt_branches=scenario_gt["branches"], pred_branches=[copy.deepcopy(b) for b in scenario_gt["branches"]],
                        ontology=ontology, config=config, branch_schema=branch_schema, root=scenario_gt.get("observed_state"))
    assert res["F1_cf_valid_only"] == pytest.approx(echo["F1_cf"], abs=1e-9)
    assert res["F1_cf"] < echo["F1_cf"]


def test_p11_layer4_gate_flags_surfaced(scenario_gt, ontology, config, cf_schema):
    from ctms.ctms import score_layer4
    gt_cf = scenario_gt["counterfactuals"][0]
    pred_cf = {"counterfactual_chain": copy.deepcopy(gt_cf["counterfactual_chain"])}
    res = score_layer4(
        gt_cf_chain=gt_cf, pred_cf_obj=pred_cf, ontology=ontology,
        config=config, counterfactual_schema=cf_schema,
        root=scenario_gt.get("observed_state"),
    )
    gates = res["gates"]
    assert gates == {"schema": True, "semantic": True, "scene": True,
                     "kinematic": True, "metadata": True,
                     "intervention": True}


# ── run_benchmark aggregates a compliance block ──────────────────────────

def _fake_sample(sid, f1, cas, ctms, l3_diag, l4_results, f1_valid=None):
    return {
        "scenario_id": sid, "sample_id": sid,
        "scores": {"Causal_Foresight_F1": f1,
                   "Counterfactual_Alignment_Score": cas, "CTMS": ctms},
        "layer3": {"F1_cf": f1, "F1_cf_valid_only": f1_valid if f1_valid is not None else f1,
                   "diagnostics": l3_diag},
        "layer4": {"results": l4_results},
    }


def test_p11_summary_contains_compliance_block():
    from run_benchmark import aggregate_sample_results
    l3_diag = {"pred_unique_count": 4, "pred_valid_count": 3,
               "k_max_penalty": 1.0,
               "gate_failure_counts": {"schema": 1, "semantic": 0, "scene": 0}}
    l4_ok = {"Score_interv": 0.8, "valid": True, "matched": True,
             "gates": {"schema": True, "semantic": True, "kinematic": True, "intervention": True}}
    l4_bad = {"Score_interv": 0.0, "valid": False, "matched": True,
              "gates": {"schema": True, "semantic": True, "kinematic": False, "intervention": True}}
    samples = [
        _fake_sample("s1", 0.5, 0.4, 0.45, l3_diag, [l4_ok, l4_bad], f1_valid=0.7),
        _fake_sample("s2", 0.7, 0.8, 0.75,
                     {"pred_unique_count": 9, "pred_valid_count": 9,
                      "k_max_penalty": 8 / 9,
                      "gate_failure_counts": {"schema": 0, "semantic": 0, "scene": 0}},
                     [l4_ok], f1_valid=0.7),
    ]
    _per_scen, summary = aggregate_sample_results(samples, {}, "full")
    comp = summary["compliance"]
    assert comp["branches"]["total_unique"] == 13
    assert comp["branches"]["valid"] == 12
    assert comp["branches"]["gate_pass_rate"] == pytest.approx(12 / 13)
    assert comp["branches"]["gate_failure_counts"]["schema"] == 1
    assert comp["branches"]["k_max_violations"] == 1
    assert comp["counterfactuals"]["total"] == 3
    assert comp["counterfactuals"]["valid"] == 2
    assert comp["counterfactuals"]["gate_failure_counts"]["kinematic"] == 1
    assert summary["F1_cf_valid_only_mean"] == pytest.approx(0.7)
    assert summary["Score_interv_valid_only_mean"] == pytest.approx(0.8)


def test_protocol_depth_report_uses_one_two_three_second_horizons():
    from run_benchmark import aggregate_sample_results

    sample = _fake_sample("s1", 1.0, 1.0, 1.0, {}, [], f1_valid=1.0)
    sample["layer3"]["matches"] = [{"depth_scores": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6]}]
    sample["horizon_scores"] = {
        "1s": {"Causal_Foresight_F1": 0.25,
               "Counterfactual_Alignment_Score": 0.36, "CTMS": 0.30},
        "2s": {"Causal_Foresight_F1": 0.49,
               "Counterfactual_Alignment_Score": 0.64, "CTMS": 0.56},
        "3s": {"Causal_Foresight_F1": 0.81,
               "Counterfactual_Alignment_Score": 1.0, "CTMS": 0.90},
    }
    _per_scen, summary = aggregate_sample_results([sample], {}, "full")
    assert summary["per_horizon"]["1s"]["CTMS"] == pytest.approx(0.30)
    assert summary["per_horizon"]["2s"]["Causal_Foresight_F1"] == pytest.approx(0.49)
    assert summary["per_horizon"]["3s"]["Counterfactual_Alignment_Score"] == 1.0


def test_l4_breakdown_reports_ego_and_non_ego_targets_separately():
    from run_benchmark import aggregate_sample_results

    sample = _fake_sample("s1", 1.0, 0.5, 0.7, {}, [
        {"Score_interv": 0.8, "valid": True, "target_actor_id": "ego",
         "target_class": "ego_vehicle", "intervention_type": "ego_action",
         "gates": {"schema": True, "semantic": True, "kinematic": True,
                   "intervention": True}},
        {"Score_interv": 0.2, "valid": True, "target_actor_id": "npc_1",
         "target_class": "vehicle", "intervention_type": "npc_sudden_brake",
         "gates": {"schema": True, "semantic": True, "kinematic": True,
                   "intervention": True}},
    ])
    _per_scen, summary = aggregate_sample_results([sample], {}, "full")
    scopes = summary["l4_breakdown"]["target_scope"]
    assert scopes["ego"] == {"mean": pytest.approx(0.8), "n": 1}
    assert scopes["non_ego"] == {"mean": pytest.approx(0.2), "n": 1}


# ── baseline-normalised skill score ──────────────────────────────────────

def test_p16_skill_score_normalises_between_copy_factual_and_oracle():
    from analysis.skill_score import skill_scores
    oracle = {"Causal_Foresight_F1_mean": 1.0,
              "Counterfactual_Alignment_Score_mean": 1.0, "CTMS_mean": 1.0}
    copy_f = {"Causal_Foresight_F1_mean": 0.588,
              "Counterfactual_Alignment_Score_mean": 0.255, "CTMS_mean": 0.324}
    model = {"Causal_Foresight_F1_mean": 0.794,
             "Counterfactual_Alignment_Score_mean": 0.6275, "CTMS_mean": 0.662}
    s = skill_scores(model, oracle, copy_f)
    assert s["Causal_Foresight_F1_skill"] == pytest.approx(0.5)
    assert s["Counterfactual_Alignment_Score_skill"] == pytest.approx(0.5)
    assert s["CTMS_skill"] == pytest.approx(0.5)


def test_p16_copy_factual_itself_scores_zero_and_oracle_one():
    from analysis.skill_score import skill_scores
    oracle = {"CTMS_mean": 1.0}
    copy_f = {"CTMS_mean": 0.324}
    assert skill_scores(copy_f, oracle, copy_f)["CTMS_skill"] == pytest.approx(0.0)
    assert skill_scores(oracle, oracle, copy_f)["CTMS_skill"] == pytest.approx(1.0)


def test_p16_below_floor_is_negative_and_degenerate_denominator_is_none():
    from analysis.skill_score import skill_scores
    oracle = {"CTMS_mean": 1.0}
    copy_f = {"CTMS_mean": 0.4}
    assert skill_scores({"CTMS_mean": 0.1}, oracle, copy_f)["CTMS_skill"] < 0.0
    degenerate = skill_scores({"CTMS_mean": 0.5}, {"CTMS_mean": 0.4}, copy_f)
    assert degenerate["CTMS_skill"] is None
