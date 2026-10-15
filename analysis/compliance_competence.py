"""
Compliance-vs-competence decomposition — a general methodology.

A structured-output benchmark score conflates two abilities:

  compliance  C  — can the model emit VALID structured output at all
                   (fraction of output units passing the validity gates);
  competence  K  — how good is the output GIVEN it is valid
                   (the score conditioned on gate-passing units).

The reported (gated) score R hides which one is failing: two models with the
same R can have opposite (C, K). Reporting the pair, plus the additive deficit
from a ceiling of 1,

    (1 - R) = (1 - K) + (K - R) = competence_deficit + compliance_deficit,

tells you where the score went. The compliance loss is further classified as a
FORMAT problem (schema gate) or a WORLD-MODEL problem (semantic / scene /
kinematic gates) — the distinction a single gated number cannot make.

`decompose_axis` is domain-agnostic (plain numbers), so it applies to any
structured-output benchmark; `decompose_report` is the CORIVA adapter.

CLI:
    python -m analysis.compliance_competence \
        --cell "Gemini 2.5 Pro=results/pilot_matrix/gemini-2.5-pro_video_seed0_report.json" \
        --out results/compliance.json --tex results/compliance.tex
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Any, Dict, List, Optional

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Three kinds of compliance failure: a coarse format-vs-world-model split
# would conflate ontology-adherence with physics. A "semantic"
# failure (missing required role, action_type incompatible with event_type) is
# mostly an ONTOLOGY/instruction-following violation, not evidence of a broken
# physical world model; only the KINEMATIC gate (teleport, impossible
# displacement/acceleration) is unambiguously physics. The strong "world-model
# deficit" claim rests on the PHYSICS kind alone.
_GATE_KIND = {
    "schema": "format",
    "semantic": "ontology",
    "scene": "ontology",        # off-road / boundary: ontology/scene rules
    "intervention": "ontology",  # intervention-metadata mismatch
    "metadata": "ontology",
    "kinematic": "physics",
}


def decompose_axis(reported: float, valid_only: Optional[float], gate_pass_rate: float,
                   gate_failure_counts: Optional[Dict[str, int]] = None) -> Dict[str, Any]:
    """Decompose one score axis into compliance and competence (domain-agnostic).

    reported     R : the gated score actually obtained.
    valid_only   K : the score conditioned on gate-passing output (competence).
    gate_pass_rate C: fraction of output units passing the gates (compliance).
    """
    counts = {g: int(c) for g, c in (gate_failure_counts or {}).items()}
    total = sum(counts.values())
    dominant = max(counts, key=lambda g: counts[g]) if total else None
    # fraction of compliance failures by kind (format / ontology / physics),
    # reported instead of a single binary label so a schema=40 vs semantic=41
    # near-tie is visible rather than silently called one thing.
    kind_frac: Dict[str, float] = {"format": 0.0, "ontology": 0.0, "physics": 0.0}
    for g, c in counts.items():
        k = _GATE_KIND.get(g)
        if k and total:
            kind_frac[k] += c / total
    dominant_kind = _GATE_KIND.get(dominant) if dominant else None
    return {
        "reported": reported,
        "compliance": gate_pass_rate,
        "competence": valid_only,
        # the compliance deficit IS the score you have but cannot cash in;
        # one name for one quantity (was duplicated as compliance_tax, audit).
        "compliance_deficit": None if valid_only is None else valid_only - reported,
        "competence_deficit": None if valid_only is None else 1.0 - valid_only,
        "reason": 'no valid-output agreement available' if valid_only is None else None,
        "dominant_gate": dominant,
        "dominant_kind": dominant_kind,
        "failure_kind_fraction": kind_frac,              # {format, ontology, physics}
        "gate_failure_counts": dict(counts),
    }


def decompose_report(report: Dict[str, Any]) -> Dict[str, Any]:
    """CORIVA adapter: read the summary's two axes + compliance block."""
    s = report["summary"]
    comp = s.get("compliance", {})
    out: Dict[str, Any] = {}
    br = comp.get("branches")
    if br is not None:
        out["branches"] = decompose_axis(
            reported=s.get("Causal_Foresight_F1_mean", 0.0),
            valid_only=None if br.get('valid') == 0 else s.get("F1_cf_valid_only_mean"),
            gate_pass_rate=br.get("gate_pass_rate", 0.0),
            gate_failure_counts=br.get("gate_failure_counts"))
    cf = comp.get("counterfactuals")
    if cf is not None:
        out["counterfactuals"] = decompose_axis(
            reported=s.get("Counterfactual_Alignment_Score_mean", 0.0),
            valid_only=None if cf.get('valid') == 0 else s.get("Score_interv_valid_only_mean"),
            gate_pass_rate=cf.get("gate_pass_rate", 0.0),
            gate_failure_counts=cf.get("gate_failure_counts"))
    return out


def decompose_many(named_reports: Dict[str, Any]) -> List[Dict[str, Any]]:
    rows = []
    for name, rep in named_reports.items():
        if not isinstance(rep, dict):
            with open(rep) as f:
                rep = json.load(f)
        d = decompose_report(rep)
        d["model"] = name
        rows.append(d)
    return rows


def to_latex(rows: List[Dict[str, Any]], axis: str = "branches",
             caption: str = "Compliance vs competence (branch task)",
             label: str = "tab:compliance") -> str:
    """One row per model: reported score, compliance C, competence K, and the
    compliance-failure mix (format / ontology / physics)."""
    lines = [r"\begin{table}[t]", r"  \centering", f"  \\caption{{{caption}}}",
             f"  \\label{{{label}}}", r"  \begin{tabular}{@{}l ccc c@{}}",
             r"    \toprule",
             r"    Model & Reported & Compliance & Competence & Failure mix (fmt/ont/phys) \\",
             r"    \midrule"]
    for r in rows:
        d = r.get(axis)
        if d is None:
            continue
        kf = d["failure_kind_fraction"]
        mix = f"{kf['format']:.2f}/{kf['ontology']:.2f}/{kf['physics']:.2f}"
        agreement = '--' if d['competence'] is None else f"{d['competence']:.3f}"
        lines.append(f"    {r['model']} & {d['reported']:.3f} & {d['compliance']:.2f} "
                     f"& {agreement} & {mix} \\\\")
    lines += [r"    \bottomrule", r"  \end{tabular}", r"\end{table}"]
    return "\n".join(lines)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--cell", action="append", required=True,
                   help='"Display Name=path/to/report.json" (repeat)')
    p.add_argument("--axis", default="branches", choices=["branches", "counterfactuals"])
    p.add_argument("--out")
    p.add_argument("--tex")
    args = p.parse_args()
    named = {}
    for spec in args.cell:
        name, _, path = spec.partition("=")
        named[name] = path
    rows = decompose_many(named)
    print(json.dumps(rows, indent=2))
    if args.out:
        with open(args.out, "w") as f:
            json.dump(rows, f, indent=2)
    if args.tex:
        with open(args.tex, "w") as f:
            f.write(to_latex(rows, axis=args.axis) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
