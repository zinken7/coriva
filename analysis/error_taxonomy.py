"""
Error taxonomy (dissertation §6.4 RQ1+RQ2 analysis).

Turns run_benchmark report.json (+ optionally the GT and prediction dirs) into
the per-model failure breakdown the results chapter needs, offline:

RQ1 (branch reconstruction):
  - branch_taxonomy(report): missed-branch rate (GT not covered), spurious
    rate (preds matching no GT), invalid rate (gate-failed preds), matched
    pair-score distribution — all from the report's matches + diagnostics.
  - terminal_diversity(gt_dir, pred_dir): does the model cover GT's terminal
    states or collapse onto one? (needs the branch files.)

RQ2 (counterfactual):
  - counterfactual_taxonomy(report): L4 valid rate + gate-failure breakdown
    (schema/semantic/kinematic/intervention).
  - depth_decay(gt_dir, pred_dir): mean node similarity by depth along matched
    chains — where in the rollout models lose credit (needs the branch files
    + the scorer's node similarity).

CLI:
    python -m analysis.error_taxonomy --report results/matrix_nuscenes_protocol10/.../report.json \
        --gt-dir output_nuscenes_protocol10 --pred-dir results/matrix_nuscenes_protocol10/<cell> \
        --out results/taxonomy_gemini.json
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import sys
from collections import Counter, defaultdict
from typing import Any, Dict, List, Optional

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

_L4_GATES = ("schema", "semantic", "kinematic", "intervention")


def _load(report: Any) -> Dict[str, Any]:
    if isinstance(report, dict):
        return report
    with open(report) as f:
        return json.load(f)


# ── RQ1: branch taxonomy (report-only) ───────────────────────────────────

def branch_taxonomy(report: Any) -> Dict[str, Any]:
    rep = _load(report)
    n_gt = n_matched = n_pred_unique = n_pred_valid = 0
    s_pairs: List[float] = []
    per_sample: List[Dict[str, Any]] = []
    for s in rep.get("per_sample", []):
        l3 = s.get("layer3") or {}
        d = l3.get("diagnostics") or {}
        gt_c = int(d.get("gt_count", 0))
        pu = int(d.get("pred_unique_count", 0))
        pv = int(d.get("pred_valid_count", 0))
        matches = [m for m in l3.get("matches", []) if m.get("S_pair", 0) > 0]
        matched = len({m["gt_index"] for m in matches})
        n_gt += gt_c
        n_matched += matched
        n_pred_unique += pu
        n_pred_valid += pv
        s_pairs.extend(m["S_pair"] for m in matches)
        per_sample.append({
            "scenario_id": s.get("scenario_id"),
            "n_gt": gt_c, "n_matched": matched,
            "missed": gt_c - matched,
            "spurious": pu - matched,
            "invalid": pu - pv,
        })
    return {
        "n_samples": len(rep.get("per_sample", [])),
        "n_gt": n_gt, "n_matched": n_matched,
        "n_pred_unique": n_pred_unique, "n_pred_valid": n_pred_valid,
        "missed_rate": (n_gt - n_matched) / n_gt if n_gt else 0.0,
        "spurious_rate": (n_pred_unique - n_matched) / n_pred_unique if n_pred_unique else 0.0,
        "invalid_rate": (n_pred_unique - n_pred_valid) / n_pred_unique if n_pred_unique else 0.0,
        "matched_s_pair_mean": sum(s_pairs) / len(s_pairs) if s_pairs else 0.0,
        "per_sample": per_sample,
    }


# ── RQ2: counterfactual taxonomy (report-only) ───────────────────────────

def counterfactual_taxonomy(report: Any) -> Dict[str, Any]:
    rep = _load(report)
    n_cf = n_valid = 0
    fails = Counter()
    for s in rep.get("per_sample", []):
        for cf in (s.get("layer4") or {}).get("results", []) or []:
            n_cf += 1
            if cf.get("valid"):
                n_valid += 1
            for g in _L4_GATES:
                if cf.get("gates") and not cf["gates"].get(g, True):
                    fails[g] += 1
    return {
        "n_cf": n_cf, "n_valid": n_valid,
        "valid_rate": n_valid / n_cf if n_cf else 0.0,
        "gate_failure_counts": {g: fails.get(g, 0) for g in _L4_GATES},
    }


# ── RQ1: terminal-state diversity (needs branch files) ───────────────────

def _terminal_types(branches: List[Dict[str, Any]]) -> set:
    from ctms.ctms import _branch_terminal_type
    terminals = {"safe_stop", "collision", "out_of_bounds"}
    out = set()
    for b in branches:
        t = _branch_terminal_type(b)
        if t in terminals:
            out.add(t)
    return out


def terminal_diversity(gt_dir: str, pred_dir: str) -> Dict[str, Any]:
    """Per scenario: distinct terminal types in GT vs prediction. 'collapse'
    = the prediction covers strictly fewer distinct terminals than GT (the
    model funnels every branch to one ending)."""
    gt_terms: List[int] = []
    pred_terms: List[int] = []
    collapses = 0
    n = 0
    for gt_path in sorted(glob.glob(os.path.join(gt_dir, "scenario_*", "ground_truth.json"))):
        name = os.path.basename(os.path.dirname(gt_path))
        pred_path = os.path.join(pred_dir, name, "l3_pre.json")
        if not os.path.exists(pred_path):
            continue
        gt = json.load(open(gt_path))
        pred = json.load(open(pred_path))
        g = _terminal_types(gt.get("branches", []))
        p = _terminal_types(pred.get("predicted_branches", []))
        gt_terms.append(len(g))
        pred_terms.append(len(p))
        if len(p) < len(g):
            collapses += 1
        n += 1
    return {
        "n_scenarios": n,
        "mean_gt_terminal_types": sum(gt_terms) / n if n else 0.0,
        "mean_pred_terminal_types": sum(pred_terms) / n if n else 0.0,
        "collapse_rate": collapses / n if n else 0.0,
    }


# ── RQ2: chain-score decay by depth (needs branch files + scorer) ────────

def depth_decay(gt_dir: str, pred_dir: str, report: Any,
                max_depth: int = 10) -> Dict[str, Any]:
    """Mean node similarity by depth along matched (gt_index, pred_index)
    branch pairs, using the scorer's own node similarity after canonicalizing
    both sides. Shows where in the rollout credit is lost."""
    from ctms.ctms import (canonicalize_branch, _node_similarity,
                           load_ctms_environment)
    env = load_ctms_environment("ctms/ontology.yaml", "ctms/ctms_config.yaml",
                                "ctms/schema_branch.json", "ctms/schema_counterfactual.json")
    onto, cfg = env["ontology"], env["config"]
    rep = _load(report)
    by_depth: Dict[int, List[float]] = defaultdict(list)
    n_pairs = 0

    for s in rep.get("per_sample", []):
        name_digits = "".join(ch for ch in str(s.get("scenario_id", "")) if ch.isdigit())
        # locate the GT + pred files by scenario dir digits
        gt_path = _find_by_digits(gt_dir, name_digits, "ground_truth.json")
        pred_path = _find_by_digits(pred_dir, name_digits, "l3_pre.json")
        if not gt_path or not pred_path:
            continue
        gt_branches = json.load(open(gt_path)).get("branches", [])
        pred_branches = json.load(open(pred_path)).get("predicted_branches", [])
        for m in (s.get("layer3") or {}).get("matches", []):
            gi, pi = m.get("gt_index"), m.get("pred_index")
            if gi is None or pi is None or gi >= len(gt_branches) or pi >= len(pred_branches):
                continue
            gc = canonicalize_branch(gt_branches[gi], onto, cfg).get("chain", [])
            pc = canonicalize_branch(pred_branches[pi], onto, cfg).get("chain", [])
            n_pairs += 1
            for depth in range(min(len(gc), len(pc), max_depth)):
                by_depth[depth + 1].append(_node_similarity(gc[depth], pc[depth], onto, cfg))

    return {
        "mean_node_sim_by_depth": {d: round(sum(v) / len(v), 4) for d, v in sorted(by_depth.items())},
        "n_by_depth": {d: len(v) for d, v in sorted(by_depth.items())},
        "n_matched_pairs": n_pairs,
    }


def _find_by_digits(root: str, digits: str, fname: str) -> Optional[str]:
    for p in glob.glob(os.path.join(root, "scenario_*", fname)):
        d = "".join(ch for ch in os.path.basename(os.path.dirname(p)) if ch.isdigit())
        if d == digits or d.lstrip("0") == digits.lstrip("0"):
            return p
    return None


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--report", required=True)
    p.add_argument("--gt-dir")
    p.add_argument("--pred-dir")
    p.add_argument("--out")
    args = p.parse_args()

    out: Dict[str, Any] = {
        "rq1_branches": branch_taxonomy(args.report),
        "rq2_counterfactuals": counterfactual_taxonomy(args.report),
    }
    if args.gt_dir and args.pred_dir:
        out["rq1_terminal_diversity"] = terminal_diversity(args.gt_dir, args.pred_dir)
        out["rq2_depth_decay"] = depth_decay(args.gt_dir, args.pred_dir, args.report)

    # print a compact summary (drop the per_sample list)
    compact = json.loads(json.dumps(out))
    compact["rq1_branches"].pop("per_sample", None)
    print(json.dumps(compact, indent=2))
    if args.out:
        with open(args.out, "w") as f:
            json.dump(out, f, indent=2)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
