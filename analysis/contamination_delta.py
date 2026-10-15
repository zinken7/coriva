"""
Contamination as a measurable quantity (hybrid protocol).

nuScenes is public and likely in current VLMs' training data; the CARLA split
is clean-room (freshly generated, cannot have been trained on). A model that
memorised nuScenes may score artificially higher on the real split than on
CARLA. But a raw nuScenes-minus-CARLA gap conflates memorisation with the two
splits' intrinsic difficulty difference. The primary analysis applies this
estimator separately to the three reference-free coherence axes:

    excess = (model_nuscenes - model_carla) - (baseline_nuscenes - baseline_carla)

A NON-LEARNING baseline (kinematic-extrapolation or copy-factual) cannot
memorise, so its cross-split gap is pure difficulty; a learning model's EXCESS
gap over the baseline is the contamination-attributable advantage. The
scene-recognition probe (`analysis/contamination_probe.py`) is folded in as a
corroborating signal.

Model and baseline outputs are paired by scenario inside each split. The two
splits contain different scenarios, so their paired-difference samples are
bootstrapped independently. CTMS reports remain available only as a secondary
sensitivity analysis because the split references have different meanings.

CAVEATS (state these wherever the number is reported):
1. `excess > 0` is a SIGN TEST, not a significance test. With finite scenarios
   the excess is almost never exactly zero; before calling a positive excess a
   "contamination signal" it needs a paired bootstrap confidence interval over
   scenarios (analysis/bootstrap.py) that excludes zero. `flag` here is a
   direction indicator only, not a claim of significance.
2. The estimator assumes the non-learning baseline's split-difficulty
   sensitivity equals the VLM's (additive separability / equal slopes across
   splits). If a VLM's difficulty response differs from the baseline's, the
   subtraction does not fully remove difficulty.
3. Excess conflates two things it cannot separate: having MEMORISED nuScenes
   specifically, and being better at REAL images than simulated ones in general
   (domain familiarity). A model strong on real-world imagery would show
   positive excess without having seen nuScenes. "Contamination-attributable"
   is therefore an upper bound on memorisation, not a point estimate.

CLI:
    python -m analysis.contamination_delta \
        --model "gemini=results/nuscenes/coherence.json:results/carla/coherence.json" \
        --baseline results/nuscenes/random_valid/coherence.json:results/carla/random_valid/coherence.json \
        --probe results/contamination_nuscenes_protocol10.json --out results/contamination.json
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
from typing import Any, Dict, List, Optional

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from analysis.causal_coherence import COHERENCE_AXES  # noqa: E402


def _summary(report: Any) -> Dict[str, Any]:
    value = _load_report(report)
    return value.get("summary", value)


def split_delta(nuscenes_report: Any, carla_report: Any,
                axis: str = "CTMS_mean") -> float:
    """nuScenes score minus CARLA score on `axis` (positive = better on the
    possibly-contaminated real split)."""
    return float(_summary(nuscenes_report).get(axis, 0.0)) - \
        float(_summary(carla_report).get(axis, 0.0))


def contamination_excess(model_ns: Any, model_carla: Any,
                         baseline_ns: Any, baseline_carla: Any,
                         axis: str = "CTMS_mean", n_boot: int = 10_000,
                         seed: int = 0) -> Dict[str, Any]:
    """Difficulty-controlled contamination signal: the model's cross-split gap
    minus a non-learning baseline's cross-split gap. Positive excess flags a
    real-split advantage beyond what intrinsic difficulty explains."""
    key = axis[:-5] if axis.endswith("_mean") else axis
    reports = [_load_report(x) for x in
               (model_ns, model_carla, baseline_ns, baseline_carla)]

    def score_map(report: Dict[str, Any]) -> Dict[str, float]:
        return {str(row["scenario_id"]): float(row.get("scores", {}).get(key, 0.0))
                for row in report.get("per_sample", [])
                if row.get("scenario_id") is not None}

    maps = [score_map(report) for report in reports]
    shared_ns = sorted(set(maps[0]) & set(maps[2]))
    shared_carla = sorted(set(maps[1]) & set(maps[3]))
    ns_differences = [maps[0][sid] - maps[2][sid] for sid in shared_ns]
    carla_differences = [maps[1][sid] - maps[3][sid] for sid in shared_carla]
    if ns_differences and carla_differences:
        ns_model = sum(maps[0][sid] for sid in shared_ns) / len(shared_ns)
        ns_baseline = sum(maps[2][sid] for sid in shared_ns) / len(shared_ns)
        carla_model = sum(maps[1][sid] for sid in shared_carla) / len(shared_carla)
        carla_baseline = sum(maps[3][sid] for sid in shared_carla) / len(shared_carla)
        model_delta = ns_model - carla_model
        baseline_delta = ns_baseline - carla_baseline
    else:
        model_delta = split_delta(model_ns, model_carla, axis)
        baseline_delta = split_delta(baseline_ns, baseline_carla, axis)
    excess = model_delta - baseline_delta
    # `flag` is a DIRECTION indicator (excess > 0), NOT a significance test and
    # NOT a memorisation point estimate — see the module caveats (needs a CI;
    # confounded with real-vs-sim domain familiarity).
    out = {"model_delta": model_delta, "baseline_delta": baseline_delta,
            "excess": excess, "flag": excess > 0.0,
            "note": "difficulty-controlled upper-bound signal; not proof of memorisation"}
    if ns_differences and carla_differences:
        rng = random.Random(seed)
        draws = []
        for _ in range(n_boot):
            ns_draw = sum(rng.choice(ns_differences)
                          for _ in ns_differences) / len(ns_differences)
            carla_draw = sum(rng.choice(carla_differences)
                             for _ in carla_differences) / len(carla_differences)
            draws.append(ns_draw - carla_draw)
        draws.sort()
        lo = draws[int(0.025 * (len(draws) - 1))]
        hi = draws[int(0.975 * (len(draws) - 1))]
        out["bootstrap_ci"] = {"lo": lo, "hi": hi, "n_boot": n_boot,
                               "excludes_zero": lo > 0.0 or hi < 0.0}
        out["n_nuscenes"] = len(ns_differences)
        out["n_carla"] = len(carla_differences)
    else:
        out["bootstrap_ci"] = None
    return out


def _load_report(report: Any) -> Dict[str, Any]:
    if isinstance(report, (list, tuple)):
        from analysis.leaderboard import average_reports
        return average_reports(list(report))
    if isinstance(report, dict):
        return report
    with open(report) as f:
        return json.load(f)


def _coherence_by_id(report: Any, axis: str) -> Dict[str, float]:
    """Load one reference-free coherence axis keyed by scenario id."""
    if axis not in COHERENCE_AXES:
        raise ValueError(f"unknown coherence axis {axis!r}")
    reports = list(report) if isinstance(report, (list, tuple)) else [report]
    per_seed: List[Dict[str, float]] = []
    for item in reports:
        value = _load_report(item)
        if value.get("uses_reference_answers") is True:
            raise ValueError("primary contamination cannot use reference-answer coherence")
        if value.get("measurement_role") not in (None, "primary"):
            raise ValueError("primary contamination requires model coherence reports")
        per_seed.append({
            str(row["scenario_id"]): float(row[axis])
            for row in value.get("per_scenario", [])
            if row.get("scenario_id") is not None and row.get(axis) is not None
        })
    if not per_seed:
        return {}
    shared = set(per_seed[0])
    for values in per_seed[1:]:
        shared &= set(values)
    return {
        sid: sum(values[sid] for values in per_seed) / len(per_seed)
        for sid in sorted(shared)
    }


def _paired_coherence_differences(model: Any, baseline: Any,
                                  axis: str) -> List[float]:
    model_values = _coherence_by_id(model, axis)
    baseline_values = _coherence_by_id(baseline, axis)
    shared = sorted(set(model_values) & set(baseline_values))
    return [model_values[sid] - baseline_values[sid] for sid in shared]


def coherence_contamination_excess(
        model_ns: Any, model_carla: Any, baseline_ns: Any, baseline_carla: Any,
        axis: str, n_boot: int = 10_000, seed: int = 0) -> Dict[str, Any]:
    """Primary excess on one coherence axis.

    Model and non-learning baseline are paired on scenario id within each
    split. nuScenes and CARLA contain different scenarios, so the two paired
    difference samples are resampled independently.
    """
    ns_differences = _paired_coherence_differences(model_ns, baseline_ns, axis)
    carla_differences = _paired_coherence_differences(
        model_carla, baseline_carla, axis)
    if not ns_differences or not carla_differences:
        return {
            "axis": axis,
            "measurement_role": "primary",
            "n_nuscenes": len(ns_differences),
            "n_carla": len(carla_differences),
            "nuscenes_model_minus_baseline": None,
            "carla_model_minus_baseline": None,
            "excess": None,
            "flag": False,
            "bootstrap_ci": None,
            "note": "axis undefined on a shared model-baseline subset",
        }

    ns_mean = sum(ns_differences) / len(ns_differences)
    carla_mean = sum(carla_differences) / len(carla_differences)
    excess = ns_mean - carla_mean
    rng = random.Random(seed)
    draws = []
    for _ in range(n_boot):
        ns_draw = sum(rng.choice(ns_differences)
                      for _ in ns_differences) / len(ns_differences)
        carla_draw = sum(rng.choice(carla_differences)
                         for _ in carla_differences) / len(carla_differences)
        draws.append(ns_draw - carla_draw)
    draws.sort()
    lo = draws[int(0.025 * (len(draws) - 1))]
    hi = draws[int(0.975 * (len(draws) - 1))]
    return {
        "axis": axis,
        "measurement_role": "primary",
        "n_nuscenes": len(ns_differences),
        "n_carla": len(carla_differences),
        "nuscenes_model_minus_baseline": ns_mean,
        "carla_model_minus_baseline": carla_mean,
        "excess": excess,
        "flag": excess > 0.0,
        "bootstrap_ci": {
            "lo": lo,
            "hi": hi,
            "n_boot": n_boot,
            "excludes_zero": lo > 0.0 or hi < 0.0,
            "supports_positive_signal": lo > 0.0,
        },
        "note": "difficulty-controlled upper-bound signal; not proof of memorisation",
    }


def coherence_contamination_report(
        models: Dict[str, Dict[str, Any]], baseline: Dict[str, Any],
        probe: Optional[Dict[str, float]] = None,
        sensitivity_baseline: Optional[Dict[str, Any]] = None,
        axes: tuple[str, ...] = COHERENCE_AXES,
        n_boot: int = 10_000, seed: int = 0) -> Dict[str, Any]:
    """Primary contamination report over all reference-free coherence axes."""
    rows: List[Dict[str, Any]] = []
    for name, split in models.items():
        axis_results = {
            axis: coherence_contamination_excess(
                split["nuscenes"], split["carla"], baseline["nuscenes"],
                baseline["carla"], axis, n_boot=n_boot, seed=seed)
            for axis in axes
        }
        row: Dict[str, Any] = {
            "model": name,
            "axes": axis_results,
            "probe_recognition": (probe or {}).get(name),
        }
        if sensitivity_baseline:
            row["sensitivity_axes"] = {
                axis: coherence_contamination_excess(
                    split["nuscenes"], split["carla"],
                    sensitivity_baseline["nuscenes"],
                    sensitivity_baseline["carla"], axis,
                    n_boot=n_boot, seed=seed)
                for axis in axes
            }
        rows.append(row)
    return {
        "measurement": "ground_truth_free_l3_causal_coherence",
        "measurement_role": "primary",
        "difficulty_baseline": "non_learning_coherence_baseline",
        "axes": list(axes),
        "rows": rows,
    }


def contamination_report(models: Dict[str, Dict[str, Any]],
                         baseline: Dict[str, Any], axis: str = "CTMS_mean",
                         probe: Optional[Dict[str, float]] = None,
                         sensitivity_baseline: Optional[Dict[str, Any]] = None,
                         n_boot: int = 10_000, seed: int = 0) -> Dict[str, Any]:
    """`models`: {name: {"nuscenes": report, "carla": report}}. `baseline`:
    {"nuscenes": report, "carla": report} for a non-learning control. `probe`:
    optional {name: scene-recognition rate}."""
    rows: List[Dict[str, Any]] = []
    for name, split in models.items():
        d = contamination_excess(split["nuscenes"], split["carla"],
                                 baseline["nuscenes"], baseline["carla"], axis,
                                 n_boot=n_boot, seed=seed)
        if sensitivity_baseline:
            d["copy_factual_sensitivity"] = contamination_excess(
                split["nuscenes"], split["carla"],
                sensitivity_baseline["nuscenes"], sensitivity_baseline["carla"],
                axis, n_boot=n_boot, seed=seed)
        d["model"] = name
        d["probe_recognition"] = (probe or {}).get(name)
        rows.append(d)
    rows.sort(key=lambda r: r["excess"], reverse=True)
    return {"measurement": "secondary_declared_reference_agreement_sensitivity",
            "measurement_role": "secondary",
            "axis": axis, "difficulty_baseline": "kinematic_extrapolation",
            "sensitivity_baseline": "copy_factual" if sensitivity_baseline else None,
            "rows": rows}


def to_latex(rows: List[Dict[str, Any]],
             caption: str = "Contamination signal: real-split advantage beyond difficulty",
             label: str = "tab:contamination") -> str:
    has_probe = any(r.get("probe_recognition") is not None for r in rows)
    header = r"    Model & Model $\Delta$ & Baseline $\Delta$ & Excess" + \
             (r" & Probe" if has_probe else "") + r" \\"
    cols = "l ccc" + ("c" if has_probe else "")
    lines = [r"\begin{table}[t]", r"  \centering", f"  \\caption{{{caption}}}",
             f"  \\label{{{label}}}", f"  \\begin{{tabular}}{{@{{}}{cols}@{{}}}}",
             r"    \toprule", header, r"    \midrule"]
    for r in rows:
        cell = (f"    {r['model']} & {r['model_delta']:+.3f} & {r['baseline_delta']:+.3f} "
                f"& {r['excess']:+.3f}")
        if has_probe:
            pr = r.get("probe_recognition")
            cell += f" & {'--' if pr is None else f'{pr:.2f}'}"
        lines.append(cell + r" \\")
    lines += [r"    \bottomrule", r"  \end{tabular}", r"\end{table}"]
    return "\n".join(lines)


def _parse_pair(spec: str):
    ns, _, carla = spec.partition(":")
    def paths(value: str):
        items = [item for item in value.split(",") if item]
        return items if len(items) > 1 else (items[0] if items else "")
    return {"nuscenes": paths(ns), "carla": paths(carla)}


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--model", action="append", required=True,
                   help='"name=nuscenes_report.json:carla_report.json" (repeat)')
    p.add_argument("--baseline", required=True,
                   help="non-learning nuScenes:CARLA report pair")
    p.add_argument("--sensitivity-baseline",
                   help="optional second non-learning nuScenes:CARLA pair")
    p.add_argument("--measurement", choices=("coherence", "reference"),
                   default="coherence",
                   help="primary coherence (default) or secondary reference sensitivity")
    p.add_argument("--axis",
                   help="one coherence axis, or a reference report summary axis")
    p.add_argument("--probe", help="contamination_probe JSON (per-model rate under summary)")
    p.add_argument("--out")
    p.add_argument("--n-boot", type=int, default=10_000)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    models = {}
    for spec in args.model:
        name, _, pair = spec.partition("=")
        models[name] = _parse_pair(pair)
    baseline = _parse_pair(args.baseline)
    sensitivity = _parse_pair(args.sensitivity_baseline) if args.sensitivity_baseline else None
    probe = None
    if args.probe:
        pj = json.load(open(args.probe))
        if pj and all(isinstance(v, (int, float)) for v in pj.values()):
            probe = pj
        elif len(models) == 1 and isinstance(pj.get("summary"), dict):
            summary = pj["summary"]
            rate = max(float(summary.get(key, 0.0)) for key in (
                "names_nuscenes_rate", "names_nuscenes_city_rate",
                "recognized_scene_rate"))
            probe = {next(iter(models)): rate}

    if args.measurement == "coherence":
        axes = (args.axis,) if args.axis else COHERENCE_AXES
        out = coherence_contamination_report(
            models, baseline, probe=probe, sensitivity_baseline=sensitivity,
            axes=axes, n_boot=args.n_boot, seed=args.seed)
    else:
        out = contamination_report(
            models, baseline, axis=args.axis or "CTMS_mean", probe=probe,
            sensitivity_baseline=sensitivity,
            n_boot=args.n_boot, seed=args.seed)
    print(json.dumps(out, indent=2))
    if args.out:
        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
        with open(args.out, "w") as f:
            json.dump(out, f, indent=2)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
