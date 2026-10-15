"""Baseline-normalised skill scores (dissertation §4.8).

Raw CTMS axes carry structural floors: echoing the factual branch alone buys
F1 ≈ 2/(n+1). The skill
score subtracts that floor and rescales by the oracle ceiling, so 0 means
"no better than copying the factual branch" and 1 means "indistinguishable
from ground truth":

    skill(M) = (score(M) − score(copy_factual)) / (score(oracle) − score(copy_factual))

Values below 0 (worse than the floor) are reported as-is; a degenerate
denominator (oracle ≤ copy_factual, i.e. a broken pipeline) yields None
rather than a misleading number.

Usage:
    .venv/bin/python -m analysis.skill_score \
        --model results/gemini_full_report.json \
        --oracle results/oracle_full_report.json \
        --copy-factual results/copy_factual_full_report.json \
        [--output results/gemini_skill.json]
"""

from __future__ import annotations

import argparse
import json
from typing import Any, Dict, Optional

AXES = (
    "Causal_Foresight_F1_mean",
    "Counterfactual_Alignment_Score_mean",
    "CTMS_mean",
)

_EPS = 1e-9


def skill_scores(
    model_summary: Dict[str, Any],
    oracle_summary: Dict[str, Any],
    copy_factual_summary: Dict[str, Any],
) -> Dict[str, Optional[float]]:
    """Normalise each axis of a model summary against the oracle/copy-factual
    corridor. Axes absent from the model summary are skipped."""
    out: Dict[str, Optional[float]] = {}
    for axis in AXES:
        if axis not in model_summary:
            continue
        skill_key = axis.replace("_mean", "_skill")
        try:
            model_v = float(model_summary[axis])
            oracle_v = float(oracle_summary[axis])
            floor_v = float(copy_factual_summary[axis])
        except (KeyError, TypeError, ValueError):
            out[skill_key] = None
            continue
        denom = oracle_v - floor_v
        out[skill_key] = None if denom <= _EPS else (model_v - floor_v) / denom
    return out


def _load_summary(path: str) -> Dict[str, Any]:
    with open(path) as f:
        report = json.load(f)
    return report.get("summary", report)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--model", required=True, help="model report JSON")
    parser.add_argument("--oracle", required=True, help="oracle report JSON")
    parser.add_argument("--copy-factual", required=True, help="copy_factual report JSON")
    parser.add_argument("--output", help="write result JSON here (default: stdout)")
    args = parser.parse_args()

    model = _load_summary(args.model)
    result = {
        "skill": skill_scores(model, _load_summary(args.oracle), _load_summary(args.copy_factual)),
        "raw": {axis: model.get(axis) for axis in AXES},
    }
    text = json.dumps(result, indent=2)
    if args.output:
        with open(args.output, "w") as f:
            f.write(text + "\n")
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
