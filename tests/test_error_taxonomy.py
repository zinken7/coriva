"""
Error taxonomy (TODO.md A3 / dissertation §6.4 RQ1+RQ2 analysis blocks).

Turns report.json (+ optionally the GT/pred dirs) into the per-model failure
breakdown the results chapter needs: RQ1 missed vs spurious branches,
terminal-state diversity / collapse; RQ2 counterfactual gate-failure rates and
chain-score decay by depth. Offline — no API/GPU.
"""

import json
import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from analysis import error_taxonomy as et  # noqa: E402


def _sample(sid, gt_count, matches, pred_unique, pred_valid,
            l4_gates=None, k_max_penalty=1.0):
    return {
        "scenario_id": sid, "sample_id": sid,
        "scores": {"Causal_Foresight_F1": 0.0, "Counterfactual_Alignment_Score": 0.0, "CTMS": 0.0},
        "layer3": {"matches": matches,
                   "diagnostics": {"gt_count": gt_count, "pred_unique_count": pred_unique,
                                   "pred_valid_count": pred_valid, "pred_scored_count": pred_unique,
                                   "k_max_penalty": k_max_penalty}},
        "layer4": {"results": [{"valid": g is not None and all(g.values()),
                                "matched": True, "gates": g} for g in (l4_gates or [])]},
    }


# ── RQ1 branch taxonomy (report-only) ────────────────────────────────────

def test_branch_taxonomy_missed_and_spurious():
    # scenario: 3 GT, model matched 1 (gt_index 0), submitted 4 unique, 3 valid
    rep = {"per_sample": [_sample("s1", gt_count=3,
                                  matches=[{"gt_index": 0, "pred_index": 0, "S_pair": 0.8}],
                                  pred_unique=4, pred_valid=3)]}
    out = et.branch_taxonomy(rep)
    assert out["n_gt"] == 3 and out["n_matched"] == 1
    assert out["missed_rate"] == pytest.approx(2 / 3)      # 2 of 3 GT uncovered
    assert out["spurious_rate"] == pytest.approx(3 / 4)    # 3 of 4 preds matched nothing
    assert out["invalid_rate"] == pytest.approx(1 / 4)     # 1 of 4 gate-failed


def test_branch_taxonomy_perfect_has_no_errors():
    rep = {"per_sample": [_sample("s1", gt_count=3,
                                  matches=[{"gt_index": i, "pred_index": i, "S_pair": 1.0} for i in range(3)],
                                  pred_unique=3, pred_valid=3)]}
    out = et.branch_taxonomy(rep)
    assert out["missed_rate"] == 0.0 and out["spurious_rate"] == 0.0 and out["invalid_rate"] == 0.0
    assert out["matched_s_pair_mean"] == pytest.approx(1.0)


def test_branch_taxonomy_aggregates_over_scenarios():
    rep = {"per_sample": [
        _sample("s1", 2, [{"gt_index": 0, "pred_index": 0, "S_pair": 0.5}], 2, 2),
        _sample("s2", 4, [], 1, 0),  # matched nothing, its 1 pred invalid
    ]}
    out = et.branch_taxonomy(rep)
    assert out["n_gt"] == 6 and out["n_matched"] == 1
    assert out["n_samples"] == 2


# ── RQ2 counterfactual taxonomy (report-only) ────────────────────────────

def test_counterfactual_taxonomy_gate_breakdown():
    rep = {"per_sample": [_sample("s1", 2, [], 2, 2, l4_gates=[
        {"schema": True, "semantic": True, "kinematic": True, "intervention": True},
        {"schema": True, "semantic": False, "kinematic": True, "intervention": True},
        {"schema": True, "semantic": True, "kinematic": False, "intervention": True},
    ])]}
    out = et.counterfactual_taxonomy(rep)
    assert out["n_cf"] == 3 and out["n_valid"] == 1
    assert out["gate_failure_counts"]["semantic"] == 1
    assert out["gate_failure_counts"]["kinematic"] == 1
    assert out["valid_rate"] == pytest.approx(1 / 3)


# ── Terminal diversity + depth decay (need GT + pred dirs) ───────────────

def _write_scenario(root, name, gt_branches, pred_branches):
    import os
    sd = os.path.join(root, "gt", name)
    pd = os.path.join(root, "pred", name)
    os.makedirs(sd, exist_ok=True)
    os.makedirs(pd, exist_ok=True)
    json.dump({"scenario_id": name, "observed_state": {"event_type": "observed_state",
              "actors": [{"id": "ego", "entity_type": "ego_vehicle", "role_type": "ego_agent",
                          "x": 0, "y": 0, "vx": 5, "vy": 0, "speed": 5}]},
               "branches": gt_branches, "counterfactuals": []},
              open(os.path.join(sd, "ground_truth.json"), "w"))
    json.dump({"predicted_branches": pred_branches},
              open(os.path.join(pd, "l3_pre.json"), "w"))


def _branch(term):
    return {"branch_id": term, "is_factual": term == "safe_stop",
            "intervention": {"intervention_type": "do_ego_no_action",
                             "target_actor_id": "ego", "trigger_depth": 1},
            "chain": [{"depth": 1, "timestamp": 0.5, "event_type": term, "terminal": True,
                       "actors_update": [{"id": "ego", "entity_type": "ego_vehicle",
                                          "role_type": "ego_agent", "x": 2, "y": 0}]}]}


def test_terminal_diversity_flags_collapse(tmp_path):
    root = str(tmp_path)
    # GT has 2 terminal types, pred collapses to 1
    _write_scenario(root, "scenario_1",
                    gt_branches=[_branch("safe_stop"), _branch("collision")],
                    pred_branches=[_branch("safe_stop"), _branch("safe_stop")])
    out = et.terminal_diversity(os.path.join(root, "gt"), os.path.join(root, "pred"))
    assert out["n_scenarios"] == 1
    assert out["mean_gt_terminal_types"] == pytest.approx(2.0)
    assert out["mean_pred_terminal_types"] == pytest.approx(1.0)
    assert out["collapse_rate"] == pytest.approx(1.0)   # pred < gt diversity


def test_terminal_diversity_no_collapse_when_matched(tmp_path):
    root = str(tmp_path)
    _write_scenario(root, "scenario_1",
                    gt_branches=[_branch("safe_stop"), _branch("collision")],
                    pred_branches=[_branch("safe_stop"), _branch("collision")])
    out = et.terminal_diversity(os.path.join(root, "gt"), os.path.join(root, "pred"))
    assert out["collapse_rate"] == 0.0
