"""Arm A versus arm B: the reported measures on
every finished cell, with scenario-level bootstrap intervals, the paired
A - B difference per configuration and the paired Cosmos - Qwen3-VL difference
per arm. Reads the panel roots and the baselines root only; no model is called.

  .venv/bin/python scripts/arm_report.py --arm A=results/api_panel --arm B=results/api_panel_b \\
      --baselines results/api_panel/baselines --out results/arm_report [--boot 10000] \\
      [--conditions video] [--only gemini-2.5-pro qwen3.5-9b-thinking]

Outputs in --out: rows.csv (every measure of every cell with its interval and
denominators), paired.csv (A - B per configuration; Cosmos - Qwen3-VL per
arm), per_scenario.csv (the per-scenario values the intervals rest on),
swerve_offsets.csv (the chosen lateral offsets) and summary.md.

Measures (one definition each):
  mean-type (per scenario value, mean over the scenarios where it is defined):
    the five coherence axes, the three L4 pair axes, compliance, CTMS/F1/CAS
    (from collected.json), and recipe/branch_share: for each ego branch matched
    to the published control's branch of the same action, the share of chain
    slots at the same depth at which both report the ego within 0.25 m, over
    the slots the shorter chain has (a slot the model leaves without the ego
    counts against it); per scenario the mean over its defined branches;
    collision/<task>_depth_error_signed and _abs: model minus reference depth
    of the collision node over the branches both label collision.
  ratio-type (sum of numerators over sum of denominators across scenarios,
    resampled by scenario):
    consequence/<category> over the evaluable chains; consequence/impossible
    over the collision labels judged at a complete node; example/ego_-2.4,
    example/target_-3.2 and example/published_profile over the branches whose
    first-step acceleration was estimable; own_effect/<reading> over the L4
    pairs; swerve/sign_ok_given_over_1m over the lateral branches whose offset
    exceeds 1 m and swerve/below_1m over the lateral branches with a
    comparison; collision/<task>_recall and _precision (overall, per ego action
    and per L4 intervention type) against the reference's terminal labels, a
    missing branch counting as a miss. A ratio is pooled over every scenario
    the cell covers: a covered scenario without a qualifying event contributes
    (0, 0) and stays in the resampling (`n` = scenarios with events).
Paired differences resample the scenarios both cells cover, jointly; a
scenario with events in one arm only is kept with (0, 0) in the other.
Condition factor (when an arm has both `video` and `history_json` for a
configuration): the paired video - history_json difference on every measure
within the arm (`video_minus_history_json`), the thinking effect (QT - QS) and
the decoding effect (QS - QD) within an arm, the interaction (A - B in video)
- (A - B in history_json) when all four cells exist, and the descriptive
representation identity (`identity/...`): the share of matched ego branches
whose node-wise ego distance over the common slots stays within 0.1 m (and
within 1 m; and the L4 treated target's track), which cannot separate image
use from the sensitivity of greedy decoding to any prompt change and is
therefore never read as image use on its own.
"""
from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import math
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from analysis.causal_coherence import _LAT, _ego_path, _matched_lateral_difference  # noqa: E402
from analysis.evaluation_campaign import BASELINES  # noqa: E402

_spec = importlib.util.spec_from_file_location("consequence", ROOT / "scripts" / "explore" / "consequence.py")
cq = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(cq)

COHERENCE = ("monotonicity", "responsiveness", "locality", "action_direction", "speed_response")
L4_PAIR = ("target_responsiveness", "direction_consistency", "ego_policy_consistency")
COMPLIANCE = ("L3_valid_only", "L3_valid_output_deficit", "L3_compliance_deficit", "L3_zero_valid",
              "L4_valid_only", "L4_valid_output_deficit", "L4_compliance_deficit", "L4_zero_valid",
              "L4_pair_valid", "parse_failure", "branch_shortfall_rate", "l3_branch_completeness")
REFERENCE = ("CTMS", "Causal_Foresight_F1", "Counterfactual_Alignment_Score")
MEAN_METRICS = COHERENCE + L4_PAIR + COMPLIANCE + REFERENCE
RECIPE_TOL_M = 0.25
EXAMPLE_TOL = 0.05
SWERVE_TOL_M = 1.0
CONTROL = "action_profile_published"
ARM_VARIANT = {"A": "full", "B": "recipe_withheld", "C": "ego_profiles_withheld"}
# Paired contrasts between arms, first minus second, on the scenarios both
# cover: A - B (the whole prompt effect), A - C (the part the ego profiles
# carry) and C - B (the part the other actors' stated motion carries).
ARM_CONTRASTS = (("A", "B"), ("A", "C"), ("C", "B"))
PAIRS = (("cosmos-reason2-8b", "qwen3-vl-8b", "cosmos_minus_qwen3vl"),
         ("qwen3.5-9b-thinking", "qwen3.5-9b-direct-sampled", "thinking_minus_sampled"),
         ("qwen3.5-9b-direct-sampled", "qwen3.5-9b-direct", "sampled_minus_greedy"))
CONSEQUENCE_UNDEFINED = {"insufficient_data", "empty_chain", "first_interval_contact"}
CONSEQUENCE_CATEGORIES = ("agree", "agree_extrapolated", "label_late", "label_early", "label_without_contact",
                          "label_unsupported", "label_beyond_horizon", "contact_unlabelled",
                          "contact_extrapolated", "no_collision_either")
IDENTITY = ("identity/branch_identical_0.1m", "identity/branch_within_1m", "identity/branch_same_length",
            "identity/target_identical_0.1m")
HEADLINE = ("recipe/branch_share", "example/ego_-2.4", "example/target_-3.2", "example/published_profile",
            "consequence/impossible", "consequence/contact_unlabelled", "consequence/label_unsupported",
            "consequence/label_without_contact", "own_effect/sign_ok", "swerve/sign_ok_given_over_1m",
            "swerve/below_1m", "collision/L3_recall", "collision/L3_precision", "collision/L4_treated_recall",
            "collision/L3_depth_error_abs", "monotonicity", "action_direction", "target_responsiveness",
            "L3_valid_only", "CTMS")


def _num(x):
    return cq.num(x)


def future_nodes(chain):
    """The chain without an echoed root (depth 0 / observed_state), as the scorer reads it."""
    if not isinstance(chain, list) or not chain:
        return chain if isinstance(chain, list) else []
    first = chain[0]
    if isinstance(first, dict) and (first.get("depth") == 0 or first.get("event_type") == "observed_state"):
        return chain[1:]
    return chain


def _branches(pred: Dict[str, Any]) -> Dict[str, Any]:
    """{intervention_type: chain} of the predicted ego branches (first one per action)."""
    out = {}
    for b in pred.get("predicted_branches") or []:
        if not isinstance(b, dict):
            continue
        iv = b.get("intervention") if isinstance(b.get("intervention"), dict) else {}
        action = iv.get("intervention_type")
        if isinstance(action, str) and action not in out:
            out[action] = future_nodes(b.get("chain"))
    return out


def _ego_xy(node, ego_id):
    actors = node.get("actors_update") if isinstance(node, dict) else None
    for a in actors if isinstance(actors, list) else []:
        if isinstance(a, dict) and (a.get("id") == ego_id or a.get("role_type") == "ego_agent"):
            x, y = _num(a.get("x")), _num(a.get("y"))
            return (x, y) if x is not None and y is not None else None
    return None


def _ego_track(chain, ego_id="ego"):
    """(x, y) or None per slot of a chain."""
    return [_ego_xy(n, ego_id) for n in chain] if isinstance(chain, list) else []


def _target_track(l4: Dict[str, Any]):
    """The treated target's (x, y) or None per slot of the first counterfactual."""
    for c in l4.get("predicted_counterfactuals") or []:
        if not isinstance(c, dict):
            continue
        iv = c.get("intervention") if isinstance(c.get("intervention"), dict) else {}
        tid = iv.get("target_actor_id")
        chain = future_nodes(c.get("treated_chain") or c.get("chain"))
        return [_ego_xy(n, tid) for n in chain] if isinstance(chain, list) and tid else []
    return []


def track_identity(a, b) -> Optional[float]:
    """Largest node-wise distance over the common slots at which both tracks
    report a position; None without such a slot."""
    ds = [math.hypot(p[0] - q[0], p[1] - q[1]) for p, q in zip(a or [], b or []) if p is not None and q is not None]
    return max(ds) if ds else None


def condition_identity(dv: Dict[str, Any], dh: Dict[str, Any]) -> Dict[str, Dict[str, Tuple[int, int]]]:
    """Representation identity between two cells of one arm and configuration:
    per scenario, matched ego branches (same action, at least one common
    reported slot) whose tracks stay within 0.1 m / 1 m at every common slot,
    and whose chains have the same length; and the L4 treated target's track."""
    out: Dict[str, Dict[str, Tuple[int, int]]] = defaultdict(dict)
    for scenario in sorted(set(dv["tracks"]) & set(dh["tracks"])):
        tv, th = dv["tracks"][scenario], dh["tracks"][scenario]
        n = ident = within = same_len = 0
        for action in sorted(set(tv) & set(th) - {"L4_treated_target"}):
            m = track_identity(tv[action], th[action])
            if m is None:
                continue
            n += 1; ident += m <= 0.1; within += m <= 1.0; same_len += len(tv[action]) == len(th[action])
        out["identity/branch_identical_0.1m"][scenario] = (ident, n)
        out["identity/branch_within_1m"][scenario] = (within, n)
        out["identity/branch_same_length"][scenario] = (same_len, n)
        m = track_identity(tv.get("L4_treated_target"), th.get("L4_treated_target"))
        out["identity/target_identical_0.1m"][scenario] = (int(m <= 0.1), 1) if m is not None else (0, 0)
    return dict(out)


def recipe_reproduction(model_chain, control_chain, ego_id="ego") -> Optional[float]:
    """Share of same-depth slots at which both chains report the ego within
    RECIPE_TOL_M, over the slots the shorter chain has (a slot at which the
    model does not report the ego counts against it); None when the shorter
    chain has no slot."""
    if not isinstance(model_chain, list) or not isinstance(control_chain, list):
        return None
    slots = min(len(model_chain), len(control_chain))
    if slots == 0:
        return None
    within = 0
    for m, c in zip(model_chain, control_chain):
        pm, pc = _ego_xy(m, ego_id), _ego_xy(c, ego_id)
        if pm is not None and pc is not None and math.hypot(pm[0] - pc[0], pm[1] - pc[1]) <= RECIPE_TOL_M:
            within += 1
    return within / slots


def swerve_offsets(branches: Dict[str, Any], root: Dict[str, Any]) -> List[Tuple[str, float]]:
    """(action, signed lateral offset) of every lateral branch against the
    road-following path of the farthest monotone neutral branch, at matched
    forward progression, exactly as the action_direction axis reads it;
    positive = on the branch's own side. Branches without a neutral
    comparison are left out."""
    lateral, neutral = [], []
    for action, chain in branches.items():
        lat = _LAT.get(action)
        if lat is None or not isinstance(chain, list):
            continue
        path = _ego_path([n for n in chain if isinstance(n, dict)], root)
        if len(path) < 2:
            continue
        (neutral if lat == 0.0 else lateral).append((action, 1.0 if lat > 0 else -1.0, path))
    if not neutral:
        return []

    def monotone(path):
        return len(path) >= 2 and all(b[0] >= a[0] for a, b in zip(path, path[1:]))
    traced = [p for _, _, p in neutral if monotone(p)] or [p for _, _, p in neutral]
    road = max(traced, key=lambda p: p[-1][0])
    return [(action, _matched_lateral_difference(path, road) * sign) for action, sign, path in lateral]


def _terminal(chain) -> Tuple[Optional[str], Optional[int]]:
    """(terminal label, depth of the labelled node) of a chain."""
    ev, idx = cq.label_terminal(future_nodes(chain))
    return ev, (idx + 1 if idx is not None else None)


def collision_events(gt: Dict[str, Any], l3: Dict[str, Any], l4: Dict[str, Any]
                     ) -> Tuple[Dict[str, Tuple[int, int]], Dict[str, List[float]]]:
    """Reference-versus-model terminal labels. Ratios {measure: (numerator,
    denominator)}: L3 recall and precision (overall and per action), safe-stop
    recall, L4 treated recall and precision (overall and per intervention
    type). Depth errors {measure: [model depth - reference depth]} over the
    branches both label collision (signed, and absolute). A reference branch
    without a model branch is a miss; a model branch without a reference
    branch is left out of precision."""
    out: Dict[str, List[int]] = defaultdict(lambda: [0, 0])
    depth: Dict[str, List[float]] = defaultdict(list)

    def tally(task: str, group: str, ref, mod, ref_depth, mod_depth) -> None:
        if ref == "collision":
            for key in (f"collision/{task}_recall", f"collision/{task}_recall/{group}"):
                out[key][1] += 1; out[key][0] += int(mod == "collision")
        if mod == "collision":
            for key in (f"collision/{task}_precision", f"collision/{task}_precision/{group}"):
                out[key][1] += 1; out[key][0] += int(ref == "collision")
        if ref == "collision" and mod == "collision" and ref_depth is not None and mod_depth is not None:
            depth[f"collision/{task}_depth_error_signed"].append(mod_depth - ref_depth)
            depth[f"collision/{task}_depth_error_abs"].append(abs(mod_depth - ref_depth))

    pred = _branches(l3)
    for b in gt.get("branches") or []:
        action = (b.get("intervention") or {}).get("intervention_type")
        (ref, rd), (mod, md) = _terminal(b.get("chain")), (_terminal(pred.get(action)) if action in pred else (None, None))
        tally("L3", action, ref, mod, rd, md)
        if ref == "safe_stop":
            out["collision/safe_stop_recall"][1] += 1; out["collision/safe_stop_recall"][0] += int(mod == "safe_stop")
    treated = {}
    for c in l4.get("predicted_counterfactuals") or []:
        if isinstance(c, dict):
            iv = c.get("intervention") if isinstance(c.get("intervention"), dict) else {}
            treated.setdefault(iv.get("intervention_type"), c.get("treated_chain") or c.get("chain"))
    for q in gt.get("l4_queries") or []:
        cc = q.get("counterfactual_chain") or {}
        itype = (cc.get("intervention") or {}).get("intervention_type") or q.get("intervention_type")
        (ref, rd), (mod, md) = _terminal(cc.get("chain")), (_terminal(treated.get(itype)) if itype in treated else (None, None))
        tally("L4_treated", itype, ref, mod, rd, md)
    return {k: (v[0], v[1]) for k, v in out.items()}, dict(depth)


# ── per cell ───────────────────────────────────────────────────────────────

def cell_measures(cell: Path, inputs: Path, config: str, condition: str, split: str, arm: str,
                  control_dir: Optional[Path]) -> Dict[str, Any]:
    """Per-scenario values of every measure of one cell: `means` {measure:
    {scenario: value}}, `ratios` {measure: {scenario: (num, den)}}, and the
    swerve offsets; `n_total` counts the cell's scenarios."""
    means: Dict[str, Dict[str, float]] = defaultdict(dict)
    ratios: Dict[str, Dict[str, Tuple[int, int]]] = defaultdict(dict)
    offsets: List[Tuple[str, str, float]] = []
    tracks: Dict[str, Dict[str, Any]] = {}
    collected = cell / "collected.json"
    scenarios = sorted(p.parent.name for p in cell.glob("scenario_*/l3_pre.json"))
    if collected.exists():
        metrics = json.loads(collected.read_text()).get("metrics") or {}
        for metric in MEAN_METRICS:
            for scenario, value in (metrics.get(metric) or {}).items():
                v = _num(value)
                if v is not None:
                    means[metric][scenario] = v
    rows = cq.analyse_cell(cell, inputs, config, condition, split, arm)
    by_scenario: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for r in rows:
        by_scenario[r["scenario"]].append(r)
    for scenario, rs in by_scenario.items():
        chains = [r for r in rs if r["kind"] != "L4pair"]
        evaluable = [r for r in chains if r["category"] not in CONSEQUENCE_UNDEFINED]
        for cat in CONSEQUENCE_CATEGORIES:
            ratios[f"consequence/{cat}"][scenario] = (sum(r["category"] == cat for r in evaluable), len(evaluable))
        ratios["consequence/undefined"][scenario] = (len(chains) - len(evaluable), len(chains))
        ratios["consequence/first_interval_contact"][scenario] = (
            sum(r["category"] == "first_interval_contact" for r in chains), len(chains))
        judged = [r for r in chains if r["impossible"] is not None]
        ratios["consequence/impossible"][scenario] = (sum(r["impossible"] == 1 for r in judged), len(judged))
        l3 = [r for r in rs if r["kind"] == "L3" and r["ego_accel"] is not None]
        ratios["example/ego_-2.4"][scenario] = (sum(abs(r["ego_accel"] - cq.EXAMPLE_EGO) <= EXAMPLE_TOL for r in l3), len(l3))
        prof = [r for r in l3 if r["action"] in cq.EGO_PROFILE_ACCEL]
        ratios["example/published_profile"][scenario] = (
            sum(abs(r["ego_accel"] - cq.EGO_PROFILE_ACCEL[r["action"]]) <= EXAMPLE_TOL for r in prof), len(prof))
        l4t = [r for r in rs if r["kind"] == "L4t" and r["target_accel"] is not None]
        ratios["example/target_-3.2"][scenario] = (sum(abs(r["target_accel"] - cq.EXAMPLE_TARGET) <= EXAMPLE_TOL for r in l4t), len(l4t))
        pairs = [r for r in rs if r["kind"] == "L4pair"]
        read = [r for r in pairs if r["category"] != "undefined"]
        for reading in ("sign_ok", "sign_wrong", "below_threshold"):
            ratios[f"own_effect/{reading}"][scenario] = (sum(r["category"] == reading for r in read), len(read))
        ratios["own_effect/undefined"][scenario] = (len(pairs) - len(read), len(pairs))
    for scenario in scenarios:
        try:
            l3 = json.loads((cell / scenario / "l3_pre.json").read_text())
            l4 = json.loads((cell / scenario / "l4_pre.json").read_text())
        except Exception:
            l3, l4 = {}, {}
        l3 = l3 if isinstance(l3, dict) else {}; l4 = l4 if isinstance(l4, dict) else {}
        branches = _branches(l3)
        tracks[scenario] = {action: _ego_track(chain) for action, chain in branches.items()}
        tracks[scenario]["L4_treated_target"] = _target_track(l4)
        root_path = inputs / scenario / "observed_state.json"
        root = json.loads(root_path.read_text()) if root_path.exists() else {}
        # recipe reproduction against the published control's branches
        if control_dir is not None and (control_dir / scenario / "l3_pre.json").exists():
            control = _branches(json.loads((control_dir / scenario / "l3_pre.json").read_text()))
            shares, undefined = [], 0
            for action, chain in control.items():
                share = recipe_reproduction(branches.get(action), chain)
                if share is None:
                    undefined += 1
                else:
                    shares.append(share)
            if shares:
                means["recipe/branch_share"][scenario] = float(np.mean(shares))
            ratios["recipe/branch_undefined"][scenario] = (undefined, len(control))
        # the chosen lateral offsets
        if root:
            found = swerve_offsets(branches, root)
            offsets += [(scenario, action, off) for action, off in found]
            over = [off for _, off in found if abs(off) > SWERVE_TOL_M]
            ratios["swerve/sign_ok_given_over_1m"][scenario] = (sum(off > 0 for off in over), len(over))
            ratios["swerve/below_1m"][scenario] = (len(found) - len(over), len(found))
        # terminal-label agreement with the reference
        gt_path = inputs / scenario / "ground_truth.json"
        if gt_path.exists():
            events, depth = collision_events(json.loads(gt_path.read_text()), l3, l4)
            for key, (n, d) in events.items():
                ratios[key][scenario] = (n, d)
            for key, values in depth.items():
                means[key][scenario] = float(np.mean(values))
    return dict(means=dict(means), ratios=dict(ratios), offsets=offsets, n_total=len(scenarios),
                scenarios=scenarios, tracks=tracks)


# ── statistics ─────────────────────────────────────────────────────────────

def _idx(rng, n, boot):
    return rng.integers(n, size=(boot, n))


def mean_ci(values: Dict[str, float], boot: int, seed: int) -> Dict[str, Any]:
    v = np.asarray(list(values.values()), dtype=float)
    if len(v) == 0:
        return dict(estimate=None, lo=None, hi=None, n=0)
    rng = np.random.default_rng(seed)
    m = v[_idx(rng, len(v), boot)].mean(axis=1)
    return dict(estimate=float(v.mean()), lo=float(np.quantile(m, .025)), hi=float(np.quantile(m, .975)), n=len(v))


def ratio_ci(pairs: Dict[str, Tuple[int, int]], scenarios, boot: int, seed: int) -> Dict[str, Any]:
    """Pooled ratio over the cell's covered `scenarios`, resampled by
    scenario: a covered scenario without a qualifying event contributes
    (0, 0) and stays in the resampling; `n` counts the covered scenarios,
    `n_events` those with a positive denominator."""
    scenarios = sorted(scenarios)
    num = np.asarray([pairs.get(k, (0, 0))[0] for k in scenarios], dtype=float)
    den = np.asarray([pairs.get(k, (0, 0))[1] for k in scenarios], dtype=float)
    if len(den) == 0 or den.sum() == 0:
        return dict(estimate=None, lo=None, hi=None, n=len(scenarios), n_events=0, num=0, den=0)
    rng = np.random.default_rng(seed)
    idx = _idx(rng, len(num), boot)
    with np.errstate(invalid="ignore", divide="ignore"):
        r = num[idx].sum(axis=1) / den[idx].sum(axis=1)
    r = r[np.isfinite(r)]
    return dict(estimate=float(num.sum() / den.sum()), lo=float(np.quantile(r, .025)), hi=float(np.quantile(r, .975)),
                n=len(scenarios), n_events=int((den > 0).sum()), num=int(num.sum()), den=int(den.sum()))


def paired_mean_ci(a: Dict[str, float], b: Dict[str, float], boot: int, seed: int) -> Dict[str, Any]:
    shared = sorted(set(a) & set(b))
    d = np.asarray([a[k] - b[k] for k in shared], dtype=float)
    if len(d) == 0:
        return dict(estimate=None, lo=None, hi=None, n=0)
    rng = np.random.default_rng(seed)
    m = d[_idx(rng, len(d), boot)].mean(axis=1)
    return dict(estimate=float(d.mean()), lo=float(np.quantile(m, .025)), hi=float(np.quantile(m, .975)), n=len(d))


def paired_ratio_ci(a: Dict[str, Tuple[int, int]], b: Dict[str, Tuple[int, int]], shared,
                    boot: int, seed: int) -> Dict[str, Any]:
    """Difference of the two pooled ratios over the scenarios BOTH cells
    cover (`shared`), resampled jointly; a covered scenario without a
    qualifying event in one cell contributes (0, 0) there, so a scenario
    with events in one arm only is never dropped. None when either ratio has
    no denominator on the shared scenarios."""
    shared = sorted(shared)
    na = np.asarray([a.get(k, (0, 0))[0] for k in shared], float); da = np.asarray([a.get(k, (0, 0))[1] for k in shared], float)
    nb = np.asarray([b.get(k, (0, 0))[0] for k in shared], float); db = np.asarray([b.get(k, (0, 0))[1] for k in shared], float)
    if not shared or da.sum() == 0 or db.sum() == 0:
        return dict(estimate=None, lo=None, hi=None, n=len(shared))
    rng = np.random.default_rng(seed)
    idx = _idx(rng, len(shared), boot)
    with np.errstate(invalid="ignore", divide="ignore"):
        r = na[idx].sum(axis=1) / da[idx].sum(axis=1) - nb[idx].sum(axis=1) / db[idx].sum(axis=1)
    r = r[np.isfinite(r)]
    return dict(estimate=float(na.sum() / da.sum() - nb.sum() / db.sum()),
                lo=float(np.quantile(r, .025)), hi=float(np.quantile(r, .975)), n=len(shared))


def interaction_mean_ci(av, bv, ah, bh, boot: int, seed: int) -> Dict[str, Any]:
    """(A - B in video) - (A - B in history_json), per scenario on the scenarios
    where all four cells define the measure, resampled by scenario."""
    shared = sorted(set(av) & set(bv) & set(ah) & set(bh))
    d = np.asarray([(av[k] - bv[k]) - (ah[k] - bh[k]) for k in shared], dtype=float)
    if len(d) == 0:
        return dict(estimate=None, lo=None, hi=None, n=0)
    rng = np.random.default_rng(seed)
    m = d[_idx(rng, len(d), boot)].mean(axis=1)
    return dict(estimate=float(d.mean()), lo=float(np.quantile(m, .025)), hi=float(np.quantile(m, .975)), n=len(d))


def interaction_ratio_ci(av, bv, ah, bh, shared, boot: int, seed: int) -> Dict[str, Any]:
    """The same for pooled ratios over the scenarios all four cells cover,
    (0, 0) where a cell has no qualifying event, resampled jointly."""
    shared = sorted(shared)
    if not shared:
        return dict(estimate=None, lo=None, hi=None, n=0)
    arrs = []
    for cell in (av, bv, ah, bh):
        arrs.append((np.asarray([cell.get(k, (0, 0))[0] for k in shared], float),
                     np.asarray([cell.get(k, (0, 0))[1] for k in shared], float)))
    if any(den.sum() == 0 for _, den in arrs):
        return dict(estimate=None, lo=None, hi=None, n=len(shared))
    rng = np.random.default_rng(seed)
    idx = _idx(rng, len(shared), boot)
    with np.errstate(invalid="ignore", divide="ignore"):
        r = [num[idx].sum(axis=1) / den[idx].sum(axis=1) for num, den in arrs]
        est = [num.sum() / den.sum() for num, den in arrs]
    diff = (r[0] - r[1]) - (r[2] - r[3]); diff = diff[np.isfinite(diff)]
    return dict(estimate=float((est[0] - est[1]) - (est[2] - est[3])), lo=float(np.quantile(diff, .025)),
                hi=float(np.quantile(diff, .975)), n=len(shared))


# ── report ─────────────────────────────────────────────────────────────────

def discover_arms(arms: Dict[str, Path], baselines: Optional[Path], only, conditions):
    """(arm, config, split, condition, cell, inputs) of every finished cell."""
    found = []
    for arm, root in arms.items():
        for cell, inputs, config, cond, split, manifest_arm in cq.discover(root):
            if config in BASELINES or (only and config not in only) or (conditions and cond not in conditions):
                continue
            if manifest_arm != arm:
                raise ValueError(f"{cell}: run under the {ARM_VARIANT.get(manifest_arm, '?')} prompt, "
                                 f"listed as arm {arm} ({ARM_VARIANT[arm]})")
            found.append((arm, config, split, cond, cell, inputs))
    if baselines is not None:
        for cell, inputs, config, cond, split, _ in cq.discover(baselines):
            if config in BASELINES and (not only or config in only):
                found.append(("control", config, split, cond, cell, inputs))
    return found


def fmt(stat, digits=3):
    if not stat or stat.get("estimate") is None:
        return "n/a"
    return f"{stat['estimate']:.{digits}f} [{stat['lo']:.{digits}f}, {stat['hi']:.{digits}f}]"


def build(arms: Dict[str, Path], baselines: Optional[Path], out: Path, boot: int, seed: int,
          only=None, conditions=None) -> Dict[str, Any]:
    out.mkdir(parents=True, exist_ok=True)
    cells = discover_arms(arms, baselines, only, conditions)
    if not cells:
        raise SystemExit("no finished cell found")
    data: Dict[Tuple[str, str, str, str], Dict[str, Any]] = {}
    for arm, config, split, cond, cell, inputs in cells:
        control_dir = (baselines / "cells" / split / CONTROL) if baselines is not None else None
        data[(arm, config, split, cond)] = cell_measures(cell, inputs, config, cond, split,
                                                         "A" if arm == "control" else arm, control_dir)
        print(f"{arm} {config:28s} {split:8s} {cond:13s} scenarios={data[(arm, config, split, cond)]['n_total']}", flush=True)
    rows, per_scenario, offsets = [], [], []
    for key, d in sorted(data.items()):
        arm, config, split, cond = key
        for measure, values in sorted(d["means"].items()):
            st = mean_ci(values, boot, seed)
            rows.append(dict(arm=arm, configuration=config, split=split, condition=cond, measure=measure, kind="mean",
                             **st, n_total=d["n_total"], num=None, den=None))
            per_scenario += [dict(arm=arm, configuration=config, split=split, condition=cond, measure=measure,
                                  scenario=s, value=v, num=None, den=None) for s, v in values.items()]
        for measure, pairs in sorted(d["ratios"].items()):
            st = ratio_ci(pairs, d["scenarios"], boot, seed)
            rows.append(dict(arm=arm, configuration=config, split=split, condition=cond, measure=measure, kind="ratio",
                             estimate=st["estimate"], lo=st["lo"], hi=st["hi"], n=st["n_events"], n_total=d["n_total"],
                             num=st["num"], den=st["den"]))
            per_scenario += [dict(arm=arm, configuration=config, split=split, condition=cond, measure=measure,
                                  scenario=s, value=None, num=n, den=dd) for s, (n, dd) in pairs.items()]
        offsets += [dict(arm=arm, configuration=config, split=split, condition=cond, scenario=s, action=a,
                         signed_offset_m=round(off, 3)) for s, a, off in d["offsets"]]
    paired = []
    for first, second in ARM_CONTRASTS:
        label, pair_arm = f"{first}_minus_{second}", f"{first}-{second}"
        for (arm, config, split, cond), d in sorted(data.items()):
            if arm != first or (second, config, split, cond) not in data:
                continue
            b = data[(second, config, split, cond)]
            shared = set(d["scenarios"]) & set(b["scenarios"])
            for measure in sorted(set(d["means"]) & set(b["means"])):
                paired.append(dict(comparison=label, arm=pair_arm, configuration=config, split=split, condition=cond,
                                   measure=measure, **paired_mean_ci(d["means"][measure], b["means"][measure], boot, seed)))
            for measure in sorted(set(d["ratios"]) | set(b["ratios"])):
                paired.append(dict(comparison=label, arm=pair_arm, configuration=config, split=split, condition=cond,
                                   measure=measure, **paired_ratio_ci(d["ratios"].get(measure, {}), b["ratios"].get(measure, {}),
                                                                      shared, boot, seed)))
    for (arm, config, split, cond), d in sorted(data.items()):
        for first, second, label in PAIRS:
            if config != first or (arm, second, split, cond) not in data:
                continue
            b = data[(arm, second, split, cond)]
            shared = set(d["scenarios"]) & set(b["scenarios"])
            for measure in sorted(set(d["means"]) & set(b["means"])):
                paired.append(dict(comparison=label, arm=arm, configuration=f"{first}-{second}", split=split,
                                   condition=cond, measure=measure,
                                   **paired_mean_ci(d["means"][measure], b["means"][measure], boot, seed)))
            for measure in sorted(set(d["ratios"]) | set(b["ratios"])):
                paired.append(dict(comparison=label, arm=arm, configuration=f"{first}-{second}", split=split,
                                   condition=cond, measure=measure,
                                   **paired_ratio_ci(d["ratios"].get(measure, {}), b["ratios"].get(measure, {}),
                                                     shared, boot, seed)))
    # Condition factor within an arm: video minus history_json, paired; the
    # representation identity between the two cells (descriptive rows).
    for (arm, config, split, cond), d in sorted(data.items()):
        if cond != "video" or (arm, config, split, "history_json") not in data:
            continue
        h = data[(arm, config, split, "history_json")]
        shared = set(d["scenarios"]) & set(h["scenarios"])
        for measure in sorted(set(d["means"]) & set(h["means"])):
            paired.append(dict(comparison="video_minus_history_json", arm=arm, configuration=config, split=split,
                               condition="video-history_json", measure=measure,
                               **paired_mean_ci(d["means"][measure], h["means"][measure], boot, seed)))
        for measure in sorted(set(d["ratios"]) | set(h["ratios"])):
            paired.append(dict(comparison="video_minus_history_json", arm=arm, configuration=config, split=split,
                               condition="video-history_json", measure=measure,
                               **paired_ratio_ci(d["ratios"].get(measure, {}), h["ratios"].get(measure, {}),
                                                 shared, boot, seed)))
        for measure, pairs in sorted(condition_identity(d, h).items()):
            st = ratio_ci(pairs, shared, boot, seed)
            rows.append(dict(arm=arm, configuration=config, split=split, condition="video~history_json", measure=measure,
                             kind="ratio", estimate=st["estimate"], lo=st["lo"], hi=st["hi"], n=st["n_events"],
                             n_total=len(shared), num=st["num"], den=st["den"]))
            per_scenario += [dict(arm=arm, configuration=config, split=split, condition="video~history_json",
                                  measure=measure, scenario=s, value=None, num=n, den=dd) for s, (n, dd) in pairs.items()]
    # Interaction recipe x condition: (A - B in video) - (A - B in history_json).
    for (arm, config, split, cond), av in sorted(data.items()):
        keys = [("A", config, split, "video"), ("B", config, split, "video"),
                ("A", config, split, "history_json"), ("B", config, split, "history_json")]
        if (arm, cond) != ("A", "video") or any(k not in data for k in keys):
            continue
        four = [data[k] for k in keys]
        shared = set.intersection(*(set(c["scenarios"]) for c in four))
        for measure in sorted(set.intersection(*(set(c["means"]) for c in four))):
            paired.append(dict(comparison="interaction_recipe_x_condition", arm="A-B", configuration=config, split=split,
                               condition="video-history_json", measure=measure,
                               **interaction_mean_ci(*(c["means"][measure] for c in four), boot, seed)))
        for measure in sorted(set.union(*(set(c["ratios"]) for c in four))):
            paired.append(dict(comparison="interaction_recipe_x_condition", arm="A-B", configuration=config, split=split,
                               condition="video-history_json", measure=measure,
                               **interaction_ratio_ci(*(c["ratios"].get(measure, {}) for c in four), shared, boot, seed)))
    _write_csv(out / "rows.csv", rows)
    _write_csv(out / "paired.csv", paired)
    _write_csv(out / "per_scenario.csv", per_scenario)
    _write_csv(out / "swerve_offsets.csv", offsets)
    (out / "summary.md").write_text(summary_md(rows, paired, offsets, boot), encoding="utf-8")
    result = dict(version="coriva-arm-report-1", n_boot=boot, seed=seed, arms={k: str(v) for k, v in arms.items()},
                  baselines=str(baselines) if baselines else None, control=CONTROL,
                  cells=[dict(arm=a, configuration=c, split=s, condition=k, n_total=d["n_total"])
                         for (a, c, s, k), d in sorted(data.items())],
                  rows=rows, paired=paired)
    (out / "results.json").write_text(json.dumps(result, indent=1), encoding="utf-8")
    return result


def _write_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    keys: List[str] = []
    for r in rows:
        keys += [k for k in r if k not in keys]
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=keys); w.writeheader(); w.writerows(rows)


def summary_md(rows, paired, offsets, boot) -> str:
    def table(records, columns, key_fields):
        by = defaultdict(dict)
        for r in records:
            by[tuple(r[k] for k in key_fields)][r["measure"]] = r
        lines = ["| " + " | ".join(key_fields + [c.split("/", 1)[-1] for c in columns]) + " |",
                 "|" + "---|" * (len(key_fields) + len(columns))]
        for key in sorted(by):
            lines.append("| " + " | ".join(list(key) + [fmt(by[key].get(c)) for c in columns]) + " |")
        return "\n".join(lines)
    arms_seen = sorted({r["arm"] for r in rows if r["arm"] in ARM_VARIANT})
    named = ", ".join(f"{a} = {ARM_VARIANT[a]}" for a in arms_seen)
    text = [f"# Arm comparison ({named})\n\nScenario bootstrap, {boot} resamples, pointwise 95% intervals; "
            f"ratio measures pool numerators over denominators; paired rows resample shared scenarios.\n"]
    for split in sorted({r["split"] for r in rows}):
        text.append(f"\n## {split}\n\n### Every cell (control rows are the calibration floors)\n")
        text.append(table([r for r in rows if r["split"] == split and "~" not in r["condition"]], HEADLINE,
                          ["arm", "configuration", "condition"]))
        for first, second in ARM_CONTRASTS:
            p = [r for r in paired if r["split"] == split and r["comparison"] == f"{first}_minus_{second}"]
            if p:
                text.append(f"\n### {first} minus {second}, paired on shared scenarios\n")
                text.append(table(p, HEADLINE, ["configuration", "condition"]))
        p = [r for r in paired if r["split"] == split and r["comparison"] == "video_minus_history_json"]
        if p:
            text.append("\n### video minus history_json within an arm, paired (the representation factor)\n")
            text.append(table(p, HEADLINE, ["arm", "configuration"]))
        p = [r for r in paired if r["split"] == split and r["comparison"] == "interaction_recipe_x_condition"]
        if p:
            text.append("\n### Interaction: (A - B in video) - (A - B in history_json)\n")
            text.append(table(p, HEADLINE, ["configuration"]))
        ident = [r for r in rows if r["split"] == split and "~" in r["condition"]]
        if ident:
            text.append("\n### Representation identity between video and history_json (descriptive; not image use)\n")
            text.append(table(ident, IDENTITY, ["arm", "configuration"]))
        for _, _, label in PAIRS:
            p = [r for r in paired if r["split"] == split and r["comparison"] == label]
            if p:
                text.append(f"\n### {label}, paired on shared scenarios\n")
                text.append(table(p, HEADLINE, ["arm", "condition"]))
        off = [o for o in offsets if o["split"] == split]
        if off:
            text.append("\n### Chosen lateral offsets (signed, + = own side), quantiles per cell\n")
            text.append("| arm | configuration | condition | n | q10 | median | q90 | share > 1 m |\n|---|---|---|---|---|---|---|---|")
            by = defaultdict(list)
            for o in off:
                by[(o["arm"], o["configuration"], o["condition"])].append(o["signed_offset_m"])
            for key in sorted(by):
                v = np.asarray(by[key])
                text.append(f"| {key[0]} | {key[1]} | {key[2]} | {len(v)} | {np.quantile(v, .1):.2f} | "
                            f"{np.median(v):.2f} | {np.quantile(v, .9):.2f} | {(np.abs(v) > SWERVE_TOL_M).mean():.2f} |")
    return "\n".join(text) + "\n"


def main(argv=None) -> Dict[str, Any]:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--arm", action="append", required=True,
                    help="A=<root>, B=<root> or C=<root> (full, recipe_withheld, ego_profiles_withheld); repeatable")
    ap.add_argument("--baselines", type=Path, help="root holding cells/<split>/<baseline> and inputs/")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--boot", type=int, default=10000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--only", nargs="*", help="configuration names to include")
    ap.add_argument("--conditions", nargs="*", help="conditions to include (default all)")
    a = ap.parse_args(argv)
    arms = {}
    for spec in a.arm:
        arm, _, root = spec.partition("=")
        if arm not in ARM_VARIANT or not root:
            raise SystemExit(f"--arm takes A=<root> or B=<root>, not {spec!r}")
        arms[arm] = Path(root)
    return build(arms, a.baselines, a.out, a.boot, a.seed, a.only, a.conditions)


if __name__ == "__main__":
    main()
