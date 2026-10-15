"""
Branch audit v2: author review of scene context and supplied references.

Actions outside the scenario's declared vocabulary are outside task scope,
not physically inadmissible. Vocabulary/branch completeness is checked in
code; authors judge scene/actor correspondence and included-branch content.
No model, GPU or API is used. Multi-rater statistics remain available for
multi-rater sheets; they are not the dissertation's author audit.
"""

from __future__ import annotations

import argparse
import csv
import glob
import json
import os
import random
import sys
from collections import Counter, defaultdict
from typing import Any, Dict, List, Optional

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

INCLUSION_THRESHOLD = 0.90
ALPHA_THRESHOLD = 0.67
AUDIT_VERSION = "coriva-branch-audit-2"
SCENE_PROMPT = ("Do the target identity, visibility at the built T0, and scene context "
                "agree with the supplied root and query? (y/n/u)")
BRANCH_PROMPT = ("Is the supplied branch consistent with its named action, actor states "
                 "and visible scene constraints under the stated simulator/twin assumptions? (y/n/u)")
AUDIT_INSTRUCTIONS = (
    "Branch audit v2: rate scene_context and each included_branch; give evidence in notes. "
    "Use y (no evident inconsistency), n (identified inconsistency), or u (insufficient evidence). "
    "Review the built T0, not an earlier scene-selection image. A collision is not by itself a reason "
    "to reject a counterfactual. nuScenes alternatives are twin references, not observed futures. "
    "Out-of-query actions are not human-rated. Automatic checks concern protocol consistency, "
    "not physical completeness. This is descriptive author review, not independent validation.")


def audit_rows(gt, split_name, n_raters=1):
    """One scene decision plus one decision per distinct included action."""
    base = dict(scenario_id=gt['scenario_id'], split=split_name,
                **{f'rater{i+1}': '' for i in range(n_raters)}, notes='')
    rows = [dict(base, item_type='scene_context', item='scene_actor_correspondence',
                 prompt=SCENE_PROMPT)]
    actions = sorted({b['intervention']['intervention_type'] for b in gt['branches']})
    rows.extend(dict(base, item_type='included_branch', item=action, prompt=BRANCH_PROMPT)
                for action in actions)
    return rows


def check_branch_contract(gt, query, split_name):
    """Check published protocol separately from query/GT agreement.

    An unknown scenario type stays not_checked, never inferred from the very
    branch set being tested. The real split may append its one factual action.
    These are structural checks; no judgment of physical possibility is made.
    """
    from simulation.protocol import critical_lateral_offset, scenario_actions
    from simulation.kinematic_twin import EGO_PROFILES
    branches = gt['branches']
    actions = [b['intervention']['intervention_type'] for b in branches]
    declared = query.get('l3_ego_actions') if query else None
    factual = [b['intervention']['intervention_type'] for b in branches if b.get('is_factual')]
    # Same actor the builder used: the hazard the L3 template selected, else
    # the actor named by the L4 query.
    interventions = (query or {}).get('l4_interventions') or []
    target_id = ((gt.get('scenario_template') or {}).get('selected_actor_id')
                 or (interventions[0].get('target_actor_id') if interventions else None))
    template = scenario_actions(
        gt.get('scenario_type'),
        critical_lateral_offset(gt.get('observed_state'), target_id))
    expected = set(template) if template else None
    if expected is not None and split_name == 'nuscenes' and len(factual) == 1:
        expected.add(factual[0])
    checks = {
        'known_ego_actions': all(a in EGO_PROFILES for a in actions),
        'one_reference_per_action': len(actions) == len(set(actions)),
        'one_factual_branch': len(factual) == 1,
        'ego_target_and_trigger': all(b['intervention'].get('target_actor_id') == 'ego'
                                     and b['intervention'].get('trigger_depth') == 1 for b in branches),
        'query_matches_references': (len(declared) == len(set(declared))
                                    and set(declared) == set(actions)) if declared is not None else None,
        'query_matches_protocol': set(declared) == expected
            if declared is not None and expected is not None else None,
        'references_match_protocol': set(actions) == expected if expected is not None else None,
    }
    return dict(scenario_type=gt.get('scenario_type'),
                status='fail' if False in checks.values() else
                       'not_checked' if None in checks.values() else 'pass',
                checks=checks, template_actions=list(template) if template else None,
                expected_actions=sorted(expected) if expected is not None else None,
                declared_actions=declared, reference_actions=actions, factual_actions=factual,
                scope='Protocol consistency only; no claim that other actions are physically impossible.')


def summarize_author_audit(rows):
    """Descriptive v2 counts, separated by question type. Never return alpha."""
    allowed_types = {'scene_context', 'included_branch'}
    if any(r.get('item_type') not in allowed_types for r in rows):
        raise ValueError('Outdated branch audit rubric; rebuild the v2 package')
    if any(r.get('rater1', '').strip().lower() not in ('y', 'n', 'u', '') for r in rows):
        raise ValueError('Invalid branch audit judgment')
    complete = [r for r in rows if r.get('rater1', '').strip() and r.get('notes', '').strip()]
    def counts(items):
        return {v: sum(r['rater1'].strip().lower() == v for r in items) for v in ('y', 'n', 'u')}
    return dict(rubric_version=AUDIT_VERSION,
                status='reviewed_by_author' if rows and len(complete) == len(rows) else 'awaiting_author_review',
                reviewed=len(complete), total=len(rows), independent_validation='not_conducted',
                counts=counts(complete), by_item_type={kind: dict(
                    total=sum(r['item_type'] == kind for r in rows),
                    reviewed=sum(r['item_type'] == kind for r in complete),
                    counts=counts([r for r in complete if r['item_type'] == kind]))
                    for kind in sorted(allowed_types)}, observations=complete)


# ── Krippendorff's nominal α ─────────────────────────────────────────────

def krippendorff_alpha(units: List[List[Optional[Any]]]) -> float:
    """Nominal Krippendorff's α over reliability data.

    `units[i]` is the list of ratings for unit i, one per annotator, with
    None for a missing rating. Values are compared for equality only
    (nominal metric). Uses the coincidence-matrix method: a unit rated m ≥ 2
    times contributes 1/(m−1) to each ordered pair of its ratings.

        α = 1 − D_o / D_e,
        D_o = (1/n) Σ_{c≠k} o_ck,
        D_e = (1/(n(n−1))) Σ_{c≠k} n_c n_k,

    where o is the coincidence matrix, n_c its marginals, n its grand total.
    Returns 1.0 when there is no scoreable disagreement expectation.
    """
    coincidence: Dict[tuple, float] = defaultdict(float)
    for ratings in units:
        present = [r for r in ratings if r is not None]
        m = len(present)
        if m < 2:
            continue  # a single rating forms no pair
        w = 1.0 / (m - 1)
        for a in range(m):
            for b in range(m):
                if a != b:
                    coincidence[(present[a], present[b])] += w

    n_total = sum(coincidence.values())
    if n_total == 0:
        return 1.0  # nothing to disagree about

    marginals: Dict[Any, float] = defaultdict(float)
    for (c, k), v in coincidence.items():
        marginals[c] += v

    d_o = sum(v for (c, k), v in coincidence.items() if c != k) / n_total
    d_e = sum(marginals[c] * marginals[k]
              for c in marginals for k in marginals if c != k) / (n_total * (n_total - 1))
    if d_e == 0:
        return 1.0  # only one category observed → perfect by convention
    return 1.0 - d_o / d_e


# ── Sheet generation ─────────────────────────────────────────────────────

def build_audit_sheet(gt_dirs: List[str], out_csv: str, split_name: str = "nuscenes",
                      n_raters: int = 1, limit: int = 50, seed: int = 0) -> int:
    """Emit v2 scene and included-branch questions, sampled deterministically."""

    paths: List[str] = []
    for d in gt_dirs:
        paths.extend(sorted(glob.glob(os.path.join(d, "scenario_*", "ground_truth.json"))))
    rng = random.Random(seed)
    rng.shuffle(paths)
    paths = paths[:limit]

    rater_cols = [f"rater{i + 1}" for i in range(n_raters)]
    fieldnames = ["scenario_id", "split", "item_type", "item", "prompt"] + rater_cols + ['notes']
    rows: List[Dict[str, str]] = []
    for p in paths:
        gt = json.load(open(p))
        gt.setdefault('scenario_id', os.path.basename(os.path.dirname(p)))
        rows.extend(audit_rows(gt, split_name, n_raters))

    with open(out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)
    return len(rows)


# ── Summary of a filled sheet ────────────────────────────────────────────

def _norm(v: str) -> Optional[str]:
    v = (v or "").strip().lower()
    if v in ("y", "yes", "1", "true", "t"):
        return "y"
    if v in ("n", "no", "0", "false", "f"):
        return "n"
    return None  # blank / unrecognised → missing


def _majority_yes(ratings: List[Optional[str]]) -> Optional[bool]:
    votes = [r for r in ratings if r is not None]
    if not votes:
        return None
    c = Counter(votes)
    return c["y"] > c["n"]


def audit_summary(filled_csv: str) -> Dict[str, Any]:
    """Read a filled sheet → inclusion plausibility rate (majority vote over
    included branches), exclusion correctness rate, Krippendorff α over ALL
    binary judgments, and pass/fail vs the fixed thresholds."""
    with open(filled_csv) as f:
        rows = list(csv.DictReader(f))
    rater_cols = [c for c in rows[0].keys() if c.startswith("rater")] if rows else []
    if rater_cols == ['rater1']:
        return summarize_author_audit(rows)

    units: List[List[Optional[str]]] = []
    inc_maj: List[bool] = []
    exc_maj: List[bool] = []
    for r in rows:
        ratings = [_norm(r[c]) for c in rater_cols]
        units.append(ratings)
        maj = _majority_yes(ratings)
        if maj is None:
            continue
        if r["item_type"] == "included_branch":
            inc_maj.append(maj)
        elif r["item_type"] == "excluded_action":
            exc_maj.append(maj)

    alpha = krippendorff_alpha(units)
    inclusion = (sum(inc_maj) / len(inc_maj)) if inc_maj else None
    exclusion = (sum(exc_maj) / len(exc_maj)) if exc_maj else None
    return {
        "n_items": len(rows),
        "n_included": len(inc_maj),
        "n_excluded": len(exc_maj),
        "inclusion_plausibility": inclusion,
        "exclusion_correctness": exclusion,
        "alpha": alpha,
        "passes_inclusion": (inclusion is not None and inclusion >= INCLUSION_THRESHOLD),
        "passes_alpha": alpha >= ALPHA_THRESHOLD,
        "thresholds": {"inclusion": INCLUSION_THRESHOLD, "alpha": ALPHA_THRESHOLD},
    }


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = p.add_subparsers(dest="cmd", required=True)
    ps = sub.add_parser("sheet", help="generate a blank annotation sheet")
    ps.add_argument("--gt-dir", action="append", required=True,
                    help="GT dir (repeat for multiple splits)")
    ps.add_argument("--split", default="nuscenes")
    ps.add_argument("--limit", type=int, default=50)
    ps.add_argument("--raters", type=int, default=1)
    ps.add_argument("--seed", type=int, default=0)
    ps.add_argument("--out", required=True)
    pm = sub.add_parser("summary", help="score a filled sheet")
    pm.add_argument("--in", dest="infile", required=True)
    args = p.parse_args()

    if args.cmd == "sheet":
        n = build_audit_sheet(args.gt_dir, args.out, split_name=args.split,
                              n_raters=args.raters, limit=args.limit, seed=args.seed)
        print(f"wrote {n} rows -> {args.out} (fill rater1..rater{args.raters} and notes)")
    else:
        out = audit_summary(args.infile)
        print(json.dumps(out, indent=2))
        if 'passes_inclusion' in out:
            verdict = "PASS" if out["passes_inclusion"] and out["passes_alpha"] else "FAIL"
            print(f"legacy multi-rater verdict: {verdict}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
