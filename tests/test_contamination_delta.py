"""
Contamination as a measurable quantity.

nuScenes is public and likely in VLM training data; the CARLA split is
clean-room. A raw nuScenes-minus-CARLA score gap conflates contamination with
the two splits' intrinsic difficulty difference. The fix: subtract the gap that
a NON-LEARNING baseline (kinematic / copy-factual) shows between the same
splits — a memorisation-free model captures pure difficulty, so a learning
model's EXCESS gap over the baseline is the contamination-attributable
advantage. Optionally corroborated by the scene-recognition probe rate.

Needs a report per split per model. Tested here on synthetic reports; the real
cross-split run awaits the CARLA model matrix (documented, no data yet).
"""

import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from analysis import contamination_delta as cd  # noqa: E402


def _rep(ctms):
    return {"summary": {"CTMS_mean": ctms, "Causal_Foresight_F1_mean": ctms}}


def _sample_rep(values):
    return {"summary": {"CTMS_mean": sum(values) / len(values)},
            "per_sample": [{"scenario_id": f"s{i}", "scores": {"CTMS": value}}
                           for i, value in enumerate(values)]}


def _coherence_rep(values, axis="responsiveness", uses_reference=False):
    return {
        "measurement_role": "primary",
        "uses_reference_answers": uses_reference,
        "per_scenario": [
            {"scenario_id": sid, axis: value}
            for sid, value in values.items()
        ],
    }


# ── raw split delta ──────────────────────────────────────────────────────

def test_split_delta_nuscenes_minus_carla():
    assert cd.split_delta(_rep(0.30), _rep(0.20), axis="CTMS_mean") == pytest.approx(0.10)


# ── contamination excess (difficulty-controlled) ─────────────────────────

def test_excess_is_model_gap_minus_baseline_gap():
    # model does +0.20 better on nuScenes; baseline only +0.05 (intrinsic
    # difficulty) -> excess 0.15 is the memorisation-attributable advantage
    out = cd.contamination_excess(model_ns=_rep(0.60), model_carla=_rep(0.40),
                                  baseline_ns=_rep(0.15), baseline_carla=_rep(0.10),
                                  axis="CTMS_mean")
    assert out["model_delta"] == pytest.approx(0.20)
    assert out["baseline_delta"] == pytest.approx(0.05)
    assert out["excess"] == pytest.approx(0.15)
    assert out["flag"] is True   # positive excess -> contamination-suspect


def test_no_excess_when_gap_matches_baseline():
    # model's advantage equals the baseline's -> fully explained by difficulty
    out = cd.contamination_excess(_rep(0.30), _rep(0.20), _rep(0.15), _rep(0.05),
                                  axis="CTMS_mean")
    assert out["excess"] == pytest.approx(0.0)
    assert out["flag"] is False


def test_negative_excess_not_flagged():
    # model does WORSE on nuScenes than difficulty predicts -> not contamination
    out = cd.contamination_excess(_rep(0.20), _rep(0.30), _rep(0.15), _rep(0.10),
                                  axis="CTMS_mean")
    assert out["excess"] < 0.0
    assert out["flag"] is False


def test_excess_reports_bootstrap_uncertainty_when_scenario_scores_exist():
    out = cd.contamination_excess(
        _sample_rep([0.8] * 8), _sample_rep([0.4] * 8),
        _sample_rep([0.2] * 8), _sample_rep([0.1] * 8), n_boot=100, seed=0)
    assert out["bootstrap_ci"]["excludes_zero"] is True


def test_secondary_reference_sensitivity_can_average_seed_reports():
    out = cd.contamination_excess(
        [_sample_rep([0.7, 0.9]), _sample_rep([0.9, 0.9])],
        [_sample_rep([0.3, 0.5]), _sample_rep([0.5, 0.5])],
        [_sample_rep([0.1, 0.3]), _sample_rep([0.3, 0.3])],
        [_sample_rep([0.0, 0.2]), _sample_rep([0.2, 0.2])],
        n_boot=100, seed=0)
    assert out["model_delta"] == pytest.approx(0.4)
    assert out["baseline_delta"] == pytest.approx(0.1)
    assert out["excess"] == pytest.approx(0.3)


def test_primary_coherence_excess_pairs_model_and_baseline_within_split():
    out = cd.coherence_contamination_excess(
        _coherence_rep({"n1": 0.8, "n2": 0.9}),
        _coherence_rep({"c1": 0.4, "c2": 0.5}),
        _coherence_rep({"n1": 0.2, "n2": 0.3}),
        _coherence_rep({"c1": 0.1, "c2": 0.2}),
        axis="responsiveness", n_boot=100, seed=0)
    assert out["measurement_role"] == "primary"
    assert out["n_nuscenes"] == 2 and out["n_carla"] == 2
    assert out["nuscenes_model_minus_baseline"] == pytest.approx(0.6)
    assert out["carla_model_minus_baseline"] == pytest.approx(0.3)
    assert out["excess"] == pytest.approx(0.3)
    assert out["bootstrap_ci"]["supports_positive_signal"] is True


def test_primary_coherence_excess_averages_seeds_before_scenario_bootstrap():
    model_ns = [_coherence_rep({"n1": 0.6}), _coherence_rep({"n1": 1.0})]
    model_carla = [_coherence_rep({"c1": 0.2}), _coherence_rep({"c1": 0.6})]
    baseline_ns = [_coherence_rep({"n1": 0.1}), _coherence_rep({"n1": 0.3})]
    baseline_carla = [_coherence_rep({"c1": 0.0}), _coherence_rep({"c1": 0.2})]
    out = cd.coherence_contamination_excess(
        model_ns, model_carla, baseline_ns, baseline_carla,
        axis="responsiveness", n_boot=10, seed=0)
    assert out["nuscenes_model_minus_baseline"] == pytest.approx(0.6)
    assert out["carla_model_minus_baseline"] == pytest.approx(0.3)
    assert out["excess"] == pytest.approx(0.3)


def test_primary_coherence_excess_reports_undefined_shared_axis():
    out = cd.coherence_contamination_excess(
        _coherence_rep({"n1": 0.8}), _coherence_rep({"c1": 0.4}),
        _coherence_rep({"other": 0.2}), _coherence_rep({"c1": 0.1}),
        axis="responsiveness", n_boot=10)
    assert out["excess"] is None
    assert out["n_nuscenes"] == 0
    assert out["bootstrap_ci"] is None


def test_primary_coherence_excess_rejects_reference_answer_report():
    with pytest.raises(ValueError, match="cannot use reference-answer"):
        cd.coherence_contamination_excess(
            _coherence_rep({"n1": 0.8}, uses_reference=True),
            _coherence_rep({"c1": 0.4}),
            _coherence_rep({"n1": 0.2}),
            _coherence_rep({"c1": 0.1}), axis="responsiveness", n_boot=10)


# ── report over several models ───────────────────────────────────────────

def test_report_ranks_models_and_folds_in_probe():
    models = {
        "gemini": {"nuscenes": _rep(0.60), "carla": _rep(0.40)},
        "qwen":   {"nuscenes": _rep(0.30), "carla": _rep(0.28)},
    }
    baseline = {"nuscenes": _rep(0.15), "carla": _rep(0.10)}
    probe = {"gemini": 0.8, "qwen": 0.1}   # recognition rate from the probe
    out = cd.contamination_report(models, baseline, axis="CTMS_mean", probe=probe)
    rows = {r["model"]: r for r in out["rows"]}
    assert rows["gemini"]["excess"] == pytest.approx(0.15)
    assert rows["qwen"]["excess"] == pytest.approx(-0.03, abs=1e-9)
    assert rows["gemini"]["probe_recognition"] == pytest.approx(0.8)
    # both signals point the same way for gemini
    assert rows["gemini"]["flag"] is True


def test_report_without_probe():
    models = {"m": {"nuscenes": _rep(0.5), "carla": _rep(0.3)}}
    baseline = {"nuscenes": _rep(0.1), "carla": _rep(0.1)}
    out = cd.contamination_report(models, baseline, axis="CTMS_mean")
    assert out["rows"][0]["probe_recognition"] is None
    assert out["rows"][0]["excess"] == pytest.approx(0.2)


def test_primary_coherence_report_carries_all_axes():
    def all_axes(prefix, value):
        return {
            "measurement_role": "primary", "uses_reference_answers": False,
            "per_scenario": [{"scenario_id": prefix,
                              "monotonicity": value,
                              "responsiveness": value,
                              "locality": value}],
        }

    models = {"m": {"nuscenes": all_axes("n", 0.8),
                     "carla": all_axes("c", 0.4)}}
    baseline = {"nuscenes": all_axes("n", 0.2),
                "carla": all_axes("c", 0.1)}
    out = cd.coherence_contamination_report(models, baseline, n_boot=10)
    assert out["measurement_role"] == "primary"
    assert set(out["rows"][0]["axes"]) == set(cd.COHERENCE_AXES)


# ── LaTeX ────────────────────────────────────────────────────────────────

def test_to_latex_has_models_and_excess():
    out = cd.contamination_report({"gemini": {"nuscenes": _rep(0.6), "carla": _rep(0.4)}},
                                  {"nuscenes": _rep(0.15), "carla": _rep(0.1)},
                                  axis="CTMS_mean", probe={"gemini": 0.8})
    tex = cd.to_latex(out["rows"])
    assert "gemini" in tex and "\\begin{tabular}" in tex and "0.15" in tex
