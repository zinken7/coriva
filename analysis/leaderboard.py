"""
Leaderboard assembly (dissertation §5.3/§6).

Turns per-cell secondary reference-agreement `report.json` files into the
diagnostic table the dissertation needs: bootstrap CIs per axis, paired bootstrap
for model deltas, the three final condition contrasts, skill scores,
gate-failure breakdown, and a LaTeX table. Pure offline — no API/GPU.

CLI:
    python -m analysis.leaderboard \
        --cell "Gemini 2.5 Pro=results/matrix_nuscenes_protocol10/gemini-2.5-pro_video_seed0/report.json" \
        --oracle results/pilot_bl/oracle_report.json \
        --copy-factual results/pilot_bl/copy_factual_report.json \
        --out results/leaderboard_nuscenes_protocol10.json --tex results/leaderboard.tex
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Any, Dict, List, Optional, Tuple

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from analysis.bootstrap import bootstrap_ci, paired_bootstrap_diff
from analysis.skill_score import skill_scores

AXES = ("Causal_Foresight_F1", "Counterfactual_Alignment_Score", "CTMS")


def _load(path_or_report: Any) -> Dict[str, Any]:
    if isinstance(path_or_report, dict):
        return path_or_report
    with open(path_or_report) as f:
        return json.load(f)


def per_scenario_scores(report: Any, axis: str) -> Dict[str, float]:
    """Map scenario_id → axis score from a report's per_sample list."""
    rep = _load(report)
    return {s["scenario_id"]: s["scores"].get(axis, 0.0)
            for s in rep.get("per_sample", [])}


def _avg_nested(dicts: List[Any]) -> Any:
    """Average a list of matching JSON structures numerically. Numbers are
    meaned; dicts are recursed key-by-key over shared keys; anything else takes
    the first value. Retained for audits of repeated non-protocol runs."""
    first = dicts[0]
    if all(isinstance(d, (int, float)) and not isinstance(d, bool) for d in dicts):
        return sum(dicts) / len(dicts)
    if all(isinstance(d, dict) for d in dicts):
        keys = set(first)
        for d in dicts[1:]:
            keys &= set(d)
        return {k: _avg_nested([d[k] for d in dicts]) for k in keys}
    return first


def average_reports(reports: List[Any]) -> Dict[str, Any]:
    """Fold several seed reports into one by averaging per scenario, then
    (via the recomputed summary) supporting the dissertation's protocol:
    'mean over seeds, then bootstrap over scenarios'. per_sample is averaged
    per scenario_id over scenarios present in EVERY seed; the summary is
    averaged numerically (means, valid-only means, compliance rates, gate
    counts). A single report is returned unchanged. Feed the result to
    leaderboard_row / delta_video / decompose_report exactly like one report."""
    reps = [_load(r) for r in reports]
    if len(reps) == 1:
        return reps[0]
    for key in ("protocol_version", "scorer_version", "track", "source_split", "condition"):
        values = {rep.get(key) for rep in reps if rep.get(key) is not None}
        if len(values) > 1:
            raise ValueError(f"cannot average reports with mixed {key}: {sorted(values)}")

    # per-scenario axis means over scenarios shared by all seeds
    per_seed = [{s["scenario_id"]: s for s in r.get("per_sample", [])}
                for r in reps]
    shared = set(per_seed[0])
    for ps in per_seed[1:]:
        shared &= set(ps)
    merged_samples = []
    for sid in sorted(shared):
        axes = set(per_seed[0][sid].get("scores", {}))
        for ps in per_seed[1:]:
            axes &= set(ps[sid].get("scores", {}))
        scores = {ax: sum(ps[sid]["scores"][ax] for ps in per_seed) / len(per_seed)
                  for ax in axes}
        horizons = {}
        shared_horizons = set(per_seed[0][sid].get("horizon_scores", {}))
        for ps in per_seed[1:]:
            shared_horizons &= set(ps[sid].get("horizon_scores", {}))
        for horizon in shared_horizons:
            h_axes = set(per_seed[0][sid]["horizon_scores"][horizon])
            for ps in per_seed[1:]:
                h_axes &= set(ps[sid]["horizon_scores"][horizon])
            horizons[horizon] = {
                axis: sum(ps[sid]["horizon_scores"][horizon][axis]
                          for ps in per_seed) / len(per_seed)
                for axis in h_axes}
        merged_samples.append({"scenario_id": sid, "sample_id": sid,
                               "scores": scores, "horizon_scores": horizons})

    merged_summary = _avg_nested([r.get("summary", {}) for r in reps])
    return {"summary": merged_summary, "per_sample": merged_samples,
            "n_seeds": len(reps),
            **{key: reps[0].get(key) for key in
               ("protocol_version", "scorer_version", "track", "source_split", "condition")}}


def aligned_scores(a: Any, b: Any, axis: str) -> Tuple[List[float], List[float]]:
    """Two per-scenario score lists restricted to shared scenario_ids, in a
    common (sorted-id) order — the precondition for paired bootstrap."""
    ra, rb = _load(a), _load(b)
    for key in ("protocol_version", "scorer_version", "track", "source_split"):
        if ra.get(key) is not None and rb.get(key) is not None and ra.get(key) != rb.get(key):
            raise ValueError(f"paired reports differ in {key}")
    sa, sb = per_scenario_scores(ra, axis), per_scenario_scores(rb, axis)
    shared = sorted(set(sa) & set(sb))
    return [sa[i] for i in shared], [sb[i] for i in shared]


def leaderboard_row(model: str, report: Any, oracle: Optional[Any] = None,
                    copy_factual: Optional[Any] = None,
                    n_boot: int = 10_000, seed: int = 0) -> Dict[str, Any]:
    """One leaderboard row: per-axis mean + 95% bootstrap CI, plus skill
    (if oracle+copy_factual given) and the compliance/gate breakdown."""
    rep = _load(report)
    axes = {ax: bootstrap_ci(list(per_scenario_scores(rep, ax).values()),
                             n_boot=n_boot, seed=seed) for ax in AXES}
    row: Dict[str, Any] = {"model": model, "axes": axes,
                           "measurement_role": rep.get(
                               "measurement_role", "secondary_declared_reference_agreement"),
                           "reference_semantics": rep.get(
                               "reference_semantics", "declared_reference_agreement"),
                           "n_scenarios": len(rep.get("per_sample", []))}
    horizon_names = sorted({h for sample in rep.get("per_sample", [])
                            for h in (sample.get("horizon_scores") or {})})
    if horizon_names:
        row["per_horizon"] = {
            horizon: {
                axis: bootstrap_ci([
                    float(sample["horizon_scores"][horizon].get(axis, 0.0))
                    for sample in rep.get("per_sample", [])
                    if horizon in (sample.get("horizon_scores") or {})],
                    n_boot=n_boot, seed=seed)
                for axis in AXES}
            for horizon in horizon_names
        }

    if oracle is not None and copy_factual is not None:
        row["skill"] = skill_scores(rep["summary"], _load(oracle)["summary"],
                                    _load(copy_factual)["summary"])

    comp = rep.get("summary", {}).get("compliance", {}).get("branches", {})
    if comp:
        row["gate_pass_rate"] = comp.get("gate_pass_rate")
        row["gate_failure_counts"] = comp.get("gate_failure_counts")
    if "F1_cf_valid_only_mean" in rep.get("summary", {}):
        row["F1_cf_valid_only_mean"] = rep["summary"]["F1_cf_valid_only_mean"]
    return row


def delta_video(video: Any, root_only: Any, axis: str = "CTMS",
                n_boot: int = 10_000, seed: int = 0) -> Dict[str, Any]:
    """Δ_video = video − root_only, paired over shared scenarios (video utility
    gap). `significant` = the (1−α) interval excludes zero."""
    va, vb = aligned_scores(video, root_only, axis)
    if not va:
        return {"mean_diff": None, "lo": None, "hi": None,
                "significant": False, "n": 0, "reason": "no_shared_scenarios"}
    out = paired_bootstrap_diff(va, vb, n_boot=n_boot, seed=seed)
    out["n"] = len(va)
    return out


def condition_deltas(video: Any, history_json: Any, video_history: Any,
                     axis: str = "CTMS", n_boot: int = 10_000,
                     seed: int = 0) -> Dict[str, Any]:
    """The three paired representation contrasts."""
    return {
        "video_history_minus_video": delta_video(
            video_history, video, axis=axis, n_boot=n_boot, seed=seed),
        "video_history_minus_history_json": delta_video(
            video_history, history_json, axis=axis, n_boot=n_boot, seed=seed),
        "video_minus_history_json": delta_video(
            video, history_json, axis=axis, n_boot=n_boot, seed=seed),
    }


def compare_models(a: Any, b: Any, axis: str = "CTMS",
                   n_boot: int = 10_000, seed: int = 0) -> Dict[str, Any]:
    """Paired bootstrap of model a − model b on a shared scenario set."""
    va, vb = aligned_scores(a, b, axis)
    return paired_bootstrap_diff(va, vb, n_boot=n_boot, seed=seed)


# ── LaTeX ────────────────────────────────────────────────────────────────

_AXIS_HDR = {"Causal_Foresight_F1": "F1$_{cf}$",
             "Counterfactual_Alignment_Score": "CAS", "CTMS": "CTMS"}


def _fmt_ci(ci: Dict[str, float]) -> str:
    return f"{ci['mean']:.3f} [{ci['lo']:.3f}, {ci['hi']:.3f}]"


def to_latex(rows: List[Dict[str, Any]],
             caption: str = "Secondary declared-reference agreement",
             label: str = "tab:leaderboard") -> str:
    """Render leaderboard rows as a booktabs table with mean [CI] per axis,
    plus skill and gate-pass columns when present."""
    has_skill = any("skill" in r for r in rows)
    has_gate = any(r.get("gate_pass_rate") is not None for r in rows)
    cols = "l" + "c" * len(AXES) + ("c" if has_skill else "") + ("c" if has_gate else "")
    head = ["Model"] + [_AXIS_HDR[a] for a in AXES]
    if has_skill:
        head.append("CTMS skill")
    if has_gate:
        head.append("gate-pass")

    lines = [r"\begin{table}[t]", r"  \centering",
             f"  \\caption{{{caption}}}", f"  \\label{{{label}}}",
             f"  \\begin{{tabular}}{{@{{}}{cols}@{{}}}}", r"    \toprule",
             "    " + " & ".join(head) + r" \\", r"    \midrule"]
    for r in rows:
        cells = [r["model"]] + [_fmt_ci(r["axes"][a]) for a in AXES]
        if has_skill:
            sk = r.get("skill", {}).get("CTMS_skill")
            cells.append("--" if sk is None else f"{sk:.3f}")
        if has_gate:
            gp = r.get("gate_pass_rate")
            cells.append("--" if gp is None else f"{gp:.2f}")
        lines.append("    " + " & ".join(cells) + r" \\")
    lines += [r"    \bottomrule", r"  \end{tabular}", r"\end{table}"]
    return "\n".join(lines)


# ── CLI ──────────────────────────────────────────────────────────────────

def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--cell", action="append", default=[],
                   help='"Name=report.json" or, for repeated-run means, '
                        '"Name=seed0.json,seed1.json,seed2.json" (comma-separated, '
                        'averaged per scenario before bootstrap)')
    p.add_argument("--oracle")
    p.add_argument("--copy-factual")
    p.add_argument("--video", help="video-condition report (for Δ_video)")
    p.add_argument("--root-only", help="legacy root_only-condition report (not a final-protocol ablation)")
    p.add_argument("--history-json", help="history_json-condition report")
    p.add_argument("--video-history", help="video_history-condition report")
    p.add_argument("--n-boot", type=int, default=10_000)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--caption", default="CORIVA secondary reference agreement")
    p.add_argument("--label", default="tab:leaderboard")
    p.add_argument("--out", help="write leaderboard JSON here")
    p.add_argument("--tex", help="write LaTeX table here")
    args = p.parse_args()

    rows = []
    for spec in args.cell:
        name, _, path = spec.partition("=")
        seed_paths = [p for p in path.split(",") if p]
        report = average_reports(seed_paths) if len(seed_paths) > 1 else seed_paths[0]
        rows.append(leaderboard_row(name, report, oracle=args.oracle,
                                    copy_factual=args.copy_factual,
                                    n_boot=args.n_boot, seed=args.seed))
    result: Dict[str, Any] = {
        "measurement_role": "secondary_declared_reference_agreement",
        "note": ("These scores do not replace the primary ground-truth-free "
                 "causal-coherence axes."),
        "rows": rows,
    }
    if args.video and args.root_only:
        result["delta_video"] = {ax: delta_video(args.video, args.root_only, axis=ax,
                                                  n_boot=args.n_boot, seed=args.seed)
                                 for ax in AXES}
    if args.video and args.history_json and args.video_history:
        result["condition_deltas"] = {
            ax: condition_deltas(args.video, args.history_json, args.video_history,
                                 axis=ax, n_boot=args.n_boot, seed=args.seed)
            for ax in AXES
        }

    print(json.dumps(result, indent=2))
    if args.out:
        with open(args.out, "w") as f:
            json.dump(result, f, indent=2)
    if args.tex and rows:
        with open(args.tex, "w") as f:
            f.write(to_latex(rows, caption=args.caption, label=args.label) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
