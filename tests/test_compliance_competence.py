"""
Compliance-vs-competence decomposition.

A structured-output score conflates two abilities: compliance (can the model
emit valid output at all) and competence (is the valid output good). Two models
with the same gated score can have opposite profiles. This module makes the
decomposition a first-class, domain-agnostic report: the core `decompose_axis`
takes plain numbers, so it applies to any structured-output benchmark;
`decompose_report` is the CORIVA adapter. It also classifies the compliance
loss as a FORMAT problem (schema) or a WORLD-MODEL problem (semantic/scene/
kinematic), which is the reframe's methodological payload.
"""

import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from analysis import compliance_competence as cckm  # noqa: E402


# ── Core decomposition (domain-agnostic) ─────────────────────────────────

def test_decompose_axis_additive_deficit():
    d = cckm.decompose_axis(reported=0.10, valid_only=0.12, gate_pass_rate=0.45,
                            gate_failure_counts={"schema": 0, "semantic": 41, "scene": 7})
    assert d["compliance"] == pytest.approx(0.45)
    assert d["competence"] == pytest.approx(0.12)
    assert d["competence_deficit"] == pytest.approx(0.88, abs=1e-9)  # 1 - K
    assert d["compliance_deficit"] == pytest.approx(0.02, abs=1e-9)  # K - R
    assert "compliance_tax" not in d  # no duplicate name for the same quantity
    # deficits partition the shortfall from the ceiling
    assert d["competence_deficit"] + d["compliance_deficit"] == pytest.approx(1 - 0.10, abs=1e-9)


def test_decompose_axis_kind_fraction_not_binary_label():
    # near-tie schema=40 vs semantic=41 must NOT be silently called one kind;
    # the mix is reported as fractions (audit #3, #small)
    d = cckm.decompose_axis(0.1, 0.12, 0.45, {"schema": 40, "semantic": 41, "kinematic": 0})
    frac = d["failure_kind_fraction"]
    assert frac["format"] == pytest.approx(40 / 81, abs=1e-6)
    assert frac["ontology"] == pytest.approx(41 / 81, abs=1e-6)
    assert frac["physics"] == pytest.approx(0.0)
    # semantic is ontology, NOT physics/world-model
    assert d["dominant_kind"] == "ontology"


def test_decompose_axis_physics_is_kinematic_only():
    # only the kinematic gate counts as physics (world-model) evidence
    d = cckm.decompose_axis(0.1, 0.12, 0.5, {"schema": 0, "semantic": 2, "kinematic": 8})
    assert d["failure_kind_fraction"]["physics"] == pytest.approx(0.8)
    assert d["failure_kind_fraction"]["ontology"] == pytest.approx(0.2)
    assert d["dominant_kind"] == "physics"


def test_decompose_axis_perfect_compliance_has_no_deficit():
    d = cckm.decompose_axis(0.8, 0.8, 1.0, {"schema": 0, "semantic": 0, "scene": 0})
    assert d["compliance_deficit"] == pytest.approx(0.0)
    assert d["dominant_kind"] is None


def test_same_reported_score_opposite_profiles():
    a = cckm.decompose_axis(0.30, 0.33, 0.90, {"schema": 5, "semantic": 0, "scene": 0})
    b = cckm.decompose_axis(0.30, 0.85, 0.35, {"schema": 0, "semantic": 40, "scene": 0})
    assert b["competence"] > a["competence"]
    assert b["compliance"] < a["compliance"]
    assert b["compliance_deficit"] > a["compliance_deficit"]


# ── CORIVA report adapter ────────────────────────────────────────────────

def _fake_report():
    return {"summary": {
        "Causal_Foresight_F1_mean": 0.10, "F1_cf_valid_only_mean": 0.12,
        "Counterfactual_Alignment_Score_mean": 0.116, "Score_interv_valid_only_mean": 0.145,
        "compliance": {
            "branches": {"gate_pass_rate": 0.45,
                         "gate_failure_counts": {"schema": 0, "semantic": 41, "scene": 7}},
            "counterfactuals": {"gate_pass_rate": 0.73,
                                "gate_failure_counts": {"schema": 0, "semantic": 11,
                                                        "kinematic": 6, "intervention": 0}}},
    }}


def test_decompose_report_both_axes():
    out = cckm.decompose_report(_fake_report())
    assert set(out) == {"branches", "counterfactuals"}
    assert out["branches"]["competence"] == pytest.approx(0.12)
    assert out["branches"]["compliance"] == pytest.approx(0.45)
    # branch failures are ontology (semantic/scene), not format or physics
    assert out["branches"]["dominant_kind"] == "ontology"
    assert out["counterfactuals"]["compliance"] == pytest.approx(0.73)


def test_decompose_report_on_real_pilot():
    import json, os
    path = "results/pilot_matrix/gemini-2.5-pro_video_seed0_report.json"
    if not os.path.exists(path):
        pytest.skip("pilot30 artifacts are not in the working tree")
    rep = json.load(open(path))
    out = cckm.decompose_report(rep)
    # Gemini's branch compliance loss is NOT a formatting problem; it is
    # ontology-adherence (semantic gate), with little/no physics (kinematic)
    assert out["branches"]["failure_kind_fraction"]["format"] == pytest.approx(0.0, abs=1e-9)
    assert out["branches"]["dominant_kind"] == "ontology"
    assert 0.0 <= out["branches"]["compliance"] <= 1.0


# ── LaTeX ────────────────────────────────────────────────────────────────

def test_to_latex_lists_models_and_both_numbers():
    rows = [{"model": "Gemini 2.5 Pro",
             "branches": cckm.decompose_axis(0.10, 0.12, 0.45,
                                             {"schema": 0, "semantic": 41, "scene": 7})}]
    tex = cckm.to_latex(rows)
    assert "Gemini 2.5 Pro" in tex and "\\begin{tabular}" in tex
    assert "0.45" in tex and "0.12" in tex
