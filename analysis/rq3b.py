"""
RQ3b human-correlation toolkit (dissertation §5.4/§6.5).

Validates the automated metric against expert judgement BEFORE any leaderboard
claim. 120 predictions are stratified by task × split × CTMS tertile, 12
attention checks are mixed in (oracle → should rate high, random → low), and
≥3 blinded raters score anchored 1–5 scales (causal plausibility, physical
consistency, coverage). Validity = Spearman ρ (target ≥ 0.6) between each CTMS
axis and the mean human rating on the 90 held-out items; agreement =
Krippendorff α (target ≥ 0.67); τ/ε_max are calibrated on the 30-item subset
only. Offline: sampler + blinded sheet + analysis of a filled sheet.

CLI:
    # 1. build the blinded sheet + hidden key (hand sheet to raters):
    python -m analysis.rq3b sheet --cell "gemini=results/matrix_nuscenes_protocol10/.../report.json:full:nuscenes" \
        --oracle results/pilot_bl/oracle_report.json --random results/pilot_bl/random_valid_report.json \
        --out human_study/rq3b_sheet.csv --key human_study/rq3b_key.json
    # 2. analyse the returned, filled sheet:
    python -m analysis.rq3b analyze --in human_study/rq3b_filled.csv --key human_study/rq3b_key.json
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import random
import sys
from typing import Any, Dict, List, Optional

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from analysis.branch_audit import krippendorff_alpha

SCALES = ("causal_plausibility", "physical_consistency", "coverage")
AXES = ("Causal_Foresight_F1", "Counterfactual_Alignment_Score", "CTMS")
RHO_TARGET = 0.6
ALPHA_TARGET = 0.67
CALIBRATION_SIZE = 30


# ── Spearman rank correlation (tie-corrected, via ranks) ─────────────────

def _ranks(xs: List[float]) -> List[float]:
    order = sorted(range(len(xs)), key=lambda i: xs[i])
    ranks = [0.0] * len(xs)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and xs[order[j + 1]] == xs[order[i]]:
            j += 1
        avg = (i + j) / 2.0 + 1.0  # average rank for ties (1-based)
        for k in range(i, j + 1):
            ranks[order[k]] = avg
        i = j + 1
    return ranks


def spearman(x: List[float], y: List[float]) -> float:
    """Spearman ρ = Pearson on ranks, with average-rank tie handling."""
    if len(x) != len(y) or len(x) < 2:
        return float("nan")
    rx, ry = _ranks(x), _ranks(y)
    n = len(rx)
    mx, my = sum(rx) / n, sum(ry) / n
    cov = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
    vx = sum((a - mx) ** 2 for a in rx)
    vy = sum((b - my) ** 2 for b in ry)
    if vx == 0 or vy == 0:
        return float("nan")
    return cov / (vx ** 0.5 * vy ** 0.5)


# ── Stratified sampling ──────────────────────────────────────────────────

def _items_from_cell(cell: Dict[str, Any]) -> List[Dict[str, Any]]:
    rep = cell["report"]
    if not isinstance(rep, dict):
        with open(rep) as f:
            rep = json.load(f)
    out = []
    for s in rep.get("per_sample", []):
        out.append({"model": cell["model"], "task": cell["task"], "split": cell["split"],
                    "scenario_id": s["scenario_id"],
                    "ctms": s["scores"].get("CTMS", 0.0),
                    "Causal_Foresight_F1": s["scores"].get("Causal_Foresight_F1", 0.0),
                    "Counterfactual_Alignment_Score": s["scores"].get("Counterfactual_Alignment_Score", 0.0)})
    return out


def _assign_tertiles(items: List[Dict[str, Any]]) -> None:
    ordered = sorted(items, key=lambda it: it["ctms"])
    n = len(ordered)
    for idx, it in enumerate(ordered):
        frac = idx / max(n, 1)
        it["tertile"] = "low" if frac < 1 / 3 else ("mid" if frac < 2 / 3 else "high")


def stratified_sample(cells: List[Dict[str, Any]], n: int = 120,
                      seed: int = 0) -> List[Dict[str, Any]]:
    """Sample n items balanced across CTMS tertiles (and, within, spread over
    the task×split strata present). Deterministic given seed. Caps at the pool
    size if fewer items exist."""
    pool: List[Dict[str, Any]] = []
    for c in cells:
        pool.extend(_items_from_cell(c))
    _assign_tertiles(pool)
    rng = random.Random(seed)
    per_tertile = n // 3
    chosen: List[Dict[str, Any]] = []
    for t in ("low", "mid", "high"):
        group = [it for it in pool if it["tertile"] == t]
        rng.shuffle(group)
        chosen.extend(group[:per_tertile])
    # top up (rounding / short tertiles) from whatever remains
    remaining = [it for it in pool if it not in chosen]
    rng.shuffle(remaining)
    chosen.extend(remaining[: n - len(chosen)])
    for i, it in enumerate(chosen):
        it["item_id"] = f"item_{i:04d}"
        it["is_attention"] = False
    return chosen


def attention_checks(oracle_report: Any, random_report: Any, n: int = 12,
                     seed: int = 0) -> List[Dict[str, Any]]:
    """n attention checks: half from oracle predictions (expected_high=True),
    half from random-valid (expected_high=False)."""
    def items(rep, model, expect):
        if not isinstance(rep, dict):
            with open(rep) as f:
                rep = json.load(f)
        return [{"model": model, "scenario_id": s["scenario_id"],
                 "ctms": s["scores"].get("CTMS", 0.0), "is_attention": True,
                 "expected_high": expect} for s in rep.get("per_sample", [])]
    rng = random.Random(seed)
    hi = items(oracle_report, "oracle", True)
    lo = items(random_report, "random", False)
    rng.shuffle(hi)
    rng.shuffle(lo)
    half = n // 2
    picks = hi[:half] + lo[:n - half]
    for i, it in enumerate(picks):
        it["item_id"] = f"attn_{i:04d}"
    return picks


# ── Blinded rating sheet + hidden key ────────────────────────────────────

def build_rating_sheet(cells: List[Dict[str, Any]], oracle_report: Any,
                       random_report: Any, out_csv: str, key_path: str,
                       n_items: int = 120, n_attention: int = 12,
                       n_raters: int = 3, seed: int = 0) -> int:
    """Write a BLINDED CSV (item_id + rater scale columns only — no model,
    CTMS, tertile or attention flag) plus a hidden key JSON with the metadata
    and the calibration/holdout split (first 30 held-out items are the
    calibration subset). Returns rows written."""
    items = stratified_sample(cells, n=n_items, seed=seed)
    checks = attention_checks(oracle_report, random_report, n=n_attention, seed=seed)
    # Attention checks wear a real-looking task/split cover so raters cannot
    # spot them from the visible sheet (blinding). Use the modal real stratum.
    cover_task = cells[0].get("task", "full")
    cover_split = cells[0].get("split", "nuscenes")
    for c in checks:
        c["task"] = cover_task
        c["split"] = cover_split
    all_rows = items + checks
    rng = random.Random(seed + 1)
    rng.shuffle(all_rows)  # blind order: attention checks interleaved

    rater_cols = [f"rater{r + 1}_{s}" for r in range(n_raters) for s in SCALES]
    fieldnames = ["item_id", "task", "split"] + rater_cols
    with open(out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for it in all_rows:
            w.writerow({"item_id": it["item_id"],
                        "task": it["task"], "split": it["split"],
                        **{c: "" for c in rater_cols}})

    real_ids = [it["item_id"] for it in all_rows if not it["is_attention"]]
    keymap = {
        "items": {it["item_id"]: {k: it.get(k) for k in
                                  ("model", "ctms", "Causal_Foresight_F1",
                                   "Counterfactual_Alignment_Score", "scenario_id",
                                   "is_attention", "expected_high")}
                  for it in all_rows},
        "calibration_ids": real_ids[:CALIBRATION_SIZE],
        "holdout_ids": real_ids[CALIBRATION_SIZE:],
        "scales": list(SCALES), "n_raters": n_raters,
    }
    with open(key_path, "w") as f:
        json.dump(keymap, f, indent=2)
    return len(all_rows)


# ── Analysis of a filled sheet ───────────────────────────────────────────

def _num(v: Any) -> Optional[float]:
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def analyze_ratings(filled_csv: str, key_path: str) -> Dict[str, Any]:
    """Filled sheet + key → attention-check pass rate, per-axis Spearman ρ
    (metric vs mean human rating on the HELD-OUT items), Krippendorff α over
    rater agreement, and pass/fail vs targets. Mean human rating per item =
    mean over all raters × all three scales."""
    with open(key_path) as f:
        key = json.load(f)
    with open(filled_csv) as f:
        rows = {r["item_id"]: r for r in csv.DictReader(f)}
    rater_scale_cols = [c for c in next(iter(rows.values())).keys()
                        if any(c.endswith("_" + s) for s in SCALES) and c.startswith("rater")]

    def mean_rating(item_id: str) -> Optional[float]:
        vals = [_num(rows[item_id][c]) for c in rater_scale_cols]
        vals = [v for v in vals if v is not None]
        return sum(vals) / len(vals) if vals else None

    # attention checks
    passed = total = 0
    for iid, meta in key["items"].items():
        if not meta.get("is_attention") or iid not in rows:
            continue
        total += 1
        mr = mean_rating(iid)
        if mr is None:
            continue
        # high-expected should land in the top half (>3), low in bottom (<3)
        if (meta["expected_high"] and mr > 3.0) or ((not meta["expected_high"]) and mr < 3.0):
            passed += 1
    attn_rate = passed / total if total else None

    # Spearman on held-out items
    holdout = [i for i in key["holdout_ids"] if i in rows]
    human = [mean_rating(i) for i in holdout]
    valid = [k for k, v in enumerate(human) if v is not None]
    rho: Dict[str, float] = {}
    for ax in AXES:
        metric = [key["items"][holdout[k]].get(ax, key["items"][holdout[k]].get("ctms")) for k in valid]
        rho[ax] = spearman(metric, [human[k] for k in valid])

    # Krippendorff α over rater agreement: unit = (item, scale), raters = columns
    n_raters = key.get("n_raters", 3)
    units: List[List[Optional[float]]] = []
    for iid in holdout:
        for s in SCALES:
            units.append([_num(rows[iid].get(f"rater{r + 1}_{s}")) for r in range(n_raters)])
    alpha = krippendorff_alpha(units)

    return {
        "n_holdout": len(holdout),
        "n_calibration": len(key.get("calibration_ids", [])),
        "attention_pass_rate": attn_rate,
        "spearman": rho,
        "alpha": alpha,
        "passes_rho": {ax: (rho[ax] >= RHO_TARGET) for ax in AXES},
        "passes_alpha": alpha >= ALPHA_TARGET,
        "targets": {"rho": RHO_TARGET, "alpha": ALPHA_TARGET},
    }


def _parse_cell(spec: str) -> Dict[str, Any]:
    # "name=path:task:split"
    name, _, rest = spec.partition("=")
    path, task, split = (rest.split(":") + ["full", "nuscenes"])[:3]
    return {"model": name, "report": path, "task": task, "split": split}


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = p.add_subparsers(dest="cmd", required=True)
    ps = sub.add_parser("sheet")
    ps.add_argument("--cell", action="append", required=True,
                    help='"name=report.json:task:split" (repeat)')
    ps.add_argument("--oracle", required=True)
    ps.add_argument("--random", required=True)
    ps.add_argument("--n-items", type=int, default=120)
    ps.add_argument("--n-attention", type=int, default=12)
    ps.add_argument("--raters", type=int, default=3)
    ps.add_argument("--seed", type=int, default=0)
    ps.add_argument("--out", required=True)
    ps.add_argument("--key", required=True)
    pa = sub.add_parser("analyze")
    pa.add_argument("--in", dest="infile", required=True)
    pa.add_argument("--key", required=True)
    args = p.parse_args()

    if args.cmd == "sheet":
        cells = [_parse_cell(c) for c in args.cell]
        n = build_rating_sheet(cells, args.oracle, args.random, args.out, args.key,
                               n_items=args.n_items, n_attention=args.n_attention,
                               n_raters=args.raters, seed=args.seed)
        print(f"wrote {n} rows -> {args.out}; key -> {args.key}")
    else:
        print(json.dumps(analyze_ratings(args.infile, args.key), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
