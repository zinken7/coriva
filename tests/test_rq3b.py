"""
RQ3b human-correlation toolkit (dissertation §5.4/§6.5; TODO.md A4).

120 predictions stratified by task × split × CTMS tertile + 12 attention
checks (oracle → should rate high, random → low), ≥3 blinded raters on
anchored 1–5 scales. Validity = Spearman ρ (target ≥ 0.6) between each CTMS
axis and mean human rating on the 90 held-out items; agreement = Krippendorff
α (target ≥ 0.67); τ/ε_max calibrated on the 30-item subset. Offline: sampler
+ blinded sheet + correlation analysis of a filled sheet; no API/GPU.
"""

import csv
import json
import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from analysis import rq3b  # noqa: E402


def _report(scores_by_scenario):
    return {"per_sample": [{"scenario_id": sid, "sample_id": sid,
                            "scores": {"Causal_Foresight_F1": v,
                                       "Counterfactual_Alignment_Score": v, "CTMS": v}}
                           for sid, v in scores_by_scenario.items()]}


# ── Spearman ─────────────────────────────────────────────────────────────

def test_spearman_monotonic_is_one():
    assert rq3b.spearman([1, 2, 3, 4], [10, 20, 30, 40]) == pytest.approx(1.0)


def test_spearman_known_value():
    # scipy reference: rho = 0.8 for this pair
    assert rq3b.spearman([1, 2, 3, 4, 5], [2, 1, 4, 3, 5]) == pytest.approx(0.8, abs=1e-6)


def test_spearman_anticorrelated():
    assert rq3b.spearman([1, 2, 3, 4], [4, 3, 2, 1]) == pytest.approx(-1.0)


# ── Stratified sampling ──────────────────────────────────────────────────

def test_stratified_sample_balances_tertiles():
    # one cell, 30 scenarios spanning a CTMS gradient
    rep = _report({f"s{i}": i / 30.0 for i in range(30)})
    cells = [{"report": rep, "task": "full", "split": "nuscenes", "model": "gemini"}]
    items = rq3b.stratified_sample(cells, n=9, seed=0)
    assert len(items) == 9
    tertiles = sorted(it["tertile"] for it in items)
    # 9 items over 3 tertiles → 3 each
    assert tertiles.count("low") == 3 and tertiles.count("mid") == 3 and tertiles.count("high") == 3
    for it in items:
        assert set(it) >= {"item_id", "model", "scenario_id", "split", "task", "ctms", "tertile"}


def test_stratified_sample_deterministic():
    rep = _report({f"s{i}": i / 20.0 for i in range(20)})
    cells = [{"report": rep, "task": "full", "split": "nuscenes", "model": "m"}]
    a = rq3b.stratified_sample(cells, n=6, seed=1)
    b = rq3b.stratified_sample(cells, n=6, seed=1)
    assert [x["scenario_id"] for x in a] == [x["scenario_id"] for x in b]


# ── Attention checks ─────────────────────────────────────────────────────

def test_attention_checks_oracle_high_random_low():
    oracle = _report({f"s{i}": 1.0 for i in range(10)})
    rand = _report({f"s{i}": 0.02 for i in range(10)})
    checks = rq3b.attention_checks(oracle, rand, n=6, seed=0)
    assert len(checks) == 6
    assert sum(c["expected_high"] for c in checks) == 3
    assert sum(not c["expected_high"] for c in checks) == 3
    assert all(c["is_attention"] for c in checks)


# ── Blinded rating sheet + hidden key ────────────────────────────────────

def test_build_rating_sheet_is_blinded_with_key(tmp_path):
    rep = _report({f"s{i}": i / 12.0 for i in range(12)})
    cells = [{"report": rep, "task": "full", "split": "nuscenes", "model": "gemini"}]
    oracle = _report({f"o{i}": 1.0 for i in range(4)})
    rand = _report({f"r{i}": 0.01 for i in range(4)})
    sheet = tmp_path / "rq3b_sheet.csv"
    key = tmp_path / "rq3b_key.json"
    rq3b.build_rating_sheet(cells, oracle, rand, str(sheet), str(key),
                            n_items=6, n_attention=2, n_raters=3, seed=0)
    rows = list(csv.DictReader(open(sheet)))
    # blinded: no model / no ctms / no is_attention columns leak into the sheet
    for banned in ("model", "ctms", "is_attention", "expected_high", "tertile"):
        assert banned not in rows[0]
    # anchored scales per rater
    for scale in ("causal_plausibility", "physical_consistency", "coverage"):
        assert f"rater1_{scale}" in rows[0]
    # key maps every item_id to its hidden metadata
    keymap = json.load(open(key))
    assert set(keymap["items"].keys()) == {r["item_id"] for r in rows}
    assert len(rows) == 6 + 2  # items + attention checks
    # blinding: attention rows must NOT be distinguishable by task/split —
    # no "attention" label leaks into the visible sheet
    assert all(r["task"] != "attention" and r["split"] != "attention" for r in rows)


# ── Analysis of a filled sheet ───────────────────────────────────────────

def test_analyze_ratings_computes_rho_and_alpha(tmp_path):
    # 4 items whose mean human rating rises monotonically with CTMS → rho≈1
    key = {"items": {
        "i1": {"model": "m", "ctms": 0.1, "Causal_Foresight_F1": 0.1,
               "Counterfactual_Alignment_Score": 0.1, "is_attention": False, "expected_high": None},
        "i2": {"model": "m", "ctms": 0.4, "Causal_Foresight_F1": 0.4,
               "Counterfactual_Alignment_Score": 0.4, "is_attention": False, "expected_high": None},
        "i3": {"model": "m", "ctms": 0.7, "Causal_Foresight_F1": 0.7,
               "Counterfactual_Alignment_Score": 0.7, "is_attention": False, "expected_high": None},
        "i4": {"model": "m", "ctms": 0.9, "Causal_Foresight_F1": 0.9,
               "Counterfactual_Alignment_Score": 0.9, "is_attention": False, "expected_high": None},
        "a1": {"model": "oracle", "ctms": 1.0, "is_attention": True, "expected_high": True},
        "a2": {"model": "random", "ctms": 0.0, "is_attention": True, "expected_high": False},
    }, "holdout_ids": ["i1", "i2", "i3", "i4"], "calibration_ids": []}
    keyf = tmp_path / "key.json"
    json.dump(key, open(keyf, "w"))

    sheet = tmp_path / "filled.csv"
    cols = ["item_id"] + [f"rater{r}_{s}" for r in (1, 2)
                          for s in ("causal_plausibility", "physical_consistency", "coverage")]
    rows = [
        {"item_id": "i1", **_rate(2, 2)}, {"item_id": "i2", **_rate(3, 3)},
        {"item_id": "i3", **_rate(4, 4)}, {"item_id": "i4", **_rate(5, 5)},
        {"item_id": "a1", **_rate(5, 5)}, {"item_id": "a2", **_rate(1, 1)},
    ]
    with open(sheet, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        w.writerows(rows)

    out = rq3b.analyze_ratings(str(sheet), str(keyf))
    assert out["attention_pass_rate"] == pytest.approx(1.0)   # a1 high, a2 low
    assert out["spearman"]["CTMS"] == pytest.approx(1.0, abs=1e-6)
    assert 0.0 <= out["alpha"] <= 1.0
    assert out["n_holdout"] == 4
    assert isinstance(out["passes_rho"]["CTMS"], bool)


def _rate(a, b):
    """Two raters giving the same scores a,b across the three scales."""
    return {
        "rater1_causal_plausibility": a, "rater1_physical_consistency": a, "rater1_coverage": a,
        "rater2_causal_plausibility": b, "rater2_physical_consistency": b, "rater2_coverage": b,
    }
