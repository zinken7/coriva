"""
Leaderboard assembly: turn per-cell report.json files
into the dissertation's leaderboard — bootstrap CIs per axis, paired bootstrap
for model deltas, Δ_video (video − root_only, scenario-aligned), skill scores,
gate-failure breakdown, and a LaTeX table. Pure offline analysis of reports;
no API/GPU.
"""

import json
import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from analysis import leaderboard as lb  # noqa: E402


def _report(scores_by_scenario, summary_extra=None):
    """Minimal report.json shape: per_sample carries per-scenario scores."""
    per_sample = [{"scenario_id": sid, "sample_id": sid, "scores": sc}
                  for sid, sc in scores_by_scenario.items()]
    summary = {"CTMS_mean": sum(s["CTMS"] for s in scores_by_scenario.values()) / len(scores_by_scenario)}
    if summary_extra:
        summary.update(summary_extra)
    return {"summary": summary, "per_sample": per_sample}


def _cell(ct):
    return {sid: {"Causal_Foresight_F1": v, "Counterfactual_Alignment_Score": v, "CTMS": v}
            for sid, v in ct.items()}


# ── per-scenario extraction + alignment ──────────────────────────────────

def test_per_scenario_scores_keyed_by_id():
    rep = _report(_cell({"s1": 0.2, "s2": 0.4, "s3": 0.6}))
    out = lb.per_scenario_scores(rep, "CTMS")
    assert out == {"s1": 0.2, "s2": 0.4, "s3": 0.6}


def test_aligned_pairs_only_shared_scenarios():
    a = _report(_cell({"s1": 0.2, "s2": 0.4, "s3": 0.6}))
    b = _report(_cell({"s2": 0.1, "s3": 0.3, "s4": 0.9}))
    va, vb = lb.aligned_scores(a, b, "CTMS")
    assert va == [0.4, 0.6] and vb == [0.1, 0.3]  # s2,s3 in id order


# ── leaderboard rows with CI ─────────────────────────────────────────────

def test_leaderboard_row_has_ci_bracketing_mean():
    rep = _report(_cell({f"s{i}": 0.1 * (i % 5) for i in range(20)}))
    row = lb.leaderboard_row("modelX", rep, n_boot=500, seed=0)
    for axis in ("Causal_Foresight_F1", "Counterfactual_Alignment_Score", "CTMS"):
        ci = row["axes"][axis]
        assert ci["lo"] <= ci["mean"] <= ci["hi"]
    assert row["model"] == "modelX"


def test_leaderboard_includes_skill_and_gate_when_baselines_given():
    model = _report(_cell({f"s{i}": 0.3 for i in range(10)}),
                    summary_extra={"compliance": {"branches": {
                        "gate_pass_rate": 0.5, "valid": 5, "total_unique": 10,
                        "gate_failure_counts": {"schema": 0, "semantic": 4, "scene": 1}}}})
    oracle = _report(_cell({f"s{i}": 1.0 for i in range(10)}))
    copyf = _report(_cell({f"s{i}": 0.1 for i in range(10)}))
    row = lb.leaderboard_row("m", model, oracle=oracle, copy_factual=copyf,
                             n_boot=200, seed=0)
    # skill = (0.3 - 0.1)/(1.0 - 0.1) = 0.2222
    assert row["skill"]["CTMS_skill"] == pytest.approx(0.2222, abs=1e-3)
    assert row["gate_pass_rate"] == pytest.approx(0.5)
    assert row["gate_failure_counts"]["semantic"] == 4


# ── Δ_video paired bootstrap ─────────────────────────────────────────────

def test_delta_video_paired():
    video = _report(_cell({f"s{i}": 0.5 for i in range(12)}))
    root_only = _report(_cell({f"s{i}": 0.3 for i in range(12)}))
    dv = lb.delta_video(video, root_only, axis="CTMS", n_boot=500, seed=0)
    assert dv["mean_diff"] == pytest.approx(0.2, abs=1e-9)
    assert dv["significant"] is True  # constant +0.2 gap


def test_delta_video_not_significant_when_noisy():
    video = _report(_cell({f"s{i}": 0.5 + (0.3 if i % 2 else -0.3) for i in range(12)}))
    root_only = _report(_cell({f"s{i}": 0.5 - (0.3 if i % 2 else -0.3) for i in range(12)}))
    dv = lb.delta_video(video, root_only, axis="CTMS", n_boot=500, seed=0)
    assert dv["mean_diff"] == pytest.approx(0.0, abs=1e-9)
    assert dv["significant"] is False


def test_final_protocol_reports_all_three_paired_condition_deltas():
    video = _report(_cell({"s1": 0.4, "s2": 0.5}))
    history = _report(_cell({"s1": 0.3, "s2": 0.4}))
    both = _report(_cell({"s1": 0.7, "s2": 0.8}))
    out = lb.condition_deltas(video, history, both, n_boot=100, seed=0)
    assert set(out) == {"video_history_minus_video",
                        "video_history_minus_history_json",
                        "video_minus_history_json"}
    assert out["video_history_minus_video"]["mean_diff"] == pytest.approx(0.3)


# ── paired model comparison ──────────────────────────────────────────────

def test_compare_models_paired():
    a = _report(_cell({f"s{i}": 0.6 for i in range(10)}))
    b = _report(_cell({f"s{i}": 0.4 for i in range(10)}))
    cmp = lb.compare_models(a, b, axis="CTMS", n_boot=500, seed=0)
    assert cmp["mean_diff"] == pytest.approx(0.2) and cmp["significant"] is True


# ── LaTeX emission ───────────────────────────────────────────────────────

def test_latex_table_contains_models_and_ci():
    rows = [
        {"model": "Gemini 2.5 Pro", "axes": {
            "Causal_Foresight_F1": {"mean": 0.10, "lo": 0.05, "hi": 0.15},
            "Counterfactual_Alignment_Score": {"mean": 0.12, "lo": 0.08, "hi": 0.16},
            "CTMS": {"mean": 0.08, "lo": 0.04, "hi": 0.12}},
         "skill": {"CTMS_skill": -0.53}, "gate_pass_rate": 0.45},
    ]
    tex = lb.to_latex(rows, caption="Pilot leaderboard", label="tab:pilot")
    assert "Gemini 2.5 Pro" in tex
    assert "\\begin{table}" in tex and "tab:pilot" in tex
    assert "0.10" in tex and "0.05" in tex  # mean and CI bound rendered


# ── repeated-run averaging utility (mean over runs, then bootstrap
#    over scenarios) — placeholder tooling ready for the full run ──────────

def test_average_reports_means_per_scenario():
    r0 = _report(_cell({"s1": 0.2, "s2": 0.4}))
    r1 = _report(_cell({"s1": 0.4, "s2": 0.6}))
    merged = lb.average_reports([r0, r1])
    ps = lb.per_scenario_scores(merged, "CTMS")
    assert ps["s1"] == pytest.approx(0.3)   # mean(0.2, 0.4)
    assert ps["s2"] == pytest.approx(0.5)   # mean(0.4, 0.6)
    assert merged["summary"]["CTMS_mean"] == pytest.approx(0.4)


def test_average_reports_only_shared_scenarios():
    r0 = _report(_cell({"s1": 0.2, "s2": 0.4, "s3": 0.6}))
    r1 = _report(_cell({"s1": 0.4, "s2": 0.6}))  # missing s3
    merged = lb.average_reports([r0, r1])
    ps = lb.per_scenario_scores(merged, "CTMS")
    assert set(ps) == {"s1", "s2"}  # s3 dropped (not in every seed)


def test_average_reports_averages_summary_extras():
    r0 = _report(_cell({"s1": 0.2}), summary_extra={"F1_cf_valid_only_mean": 0.5,
                 "compliance": {"branches": {"gate_pass_rate": 0.4,
                                             "gate_failure_counts": {"schema": 2, "semantic": 0}}}})
    r1 = _report(_cell({"s1": 0.4}), summary_extra={"F1_cf_valid_only_mean": 0.7,
                 "compliance": {"branches": {"gate_pass_rate": 0.6,
                                             "gate_failure_counts": {"schema": 4, "semantic": 0}}}})
    merged = lb.average_reports([r0, r1])
    assert merged["summary"]["F1_cf_valid_only_mean"] == pytest.approx(0.6)
    assert merged["summary"]["compliance"]["branches"]["gate_pass_rate"] == pytest.approx(0.5)
    assert merged["summary"]["compliance"]["branches"]["gate_failure_counts"]["schema"] == pytest.approx(3.0)


def test_average_reports_single_is_identity():
    r0 = _report(_cell({"s1": 0.2, "s2": 0.4}))
    merged = lb.average_reports([r0])
    assert lb.per_scenario_scores(merged, "CTMS") == {"s1": 0.2, "s2": 0.4}


def test_leaderboard_row_over_seeds_end_to_end():
    # a cell given as several seed reports -> row over the per-scenario means
    seeds = [_report(_cell({f"s{i}": 0.3 for i in range(8)})),
             _report(_cell({f"s{i}": 0.5 for i in range(8)}))]
    row = lb.leaderboard_row("m", lb.average_reports(seeds), n_boot=200, seed=0)
    ci = row["axes"]["CTMS"]
    assert ci["mean"] == pytest.approx(0.4)      # mean of 0.3 and 0.5, constant
    assert ci["lo"] <= 0.4 <= ci["hi"]
