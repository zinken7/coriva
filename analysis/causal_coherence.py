"""
Causal-coherence metrics — the ground-truth-free contribution.

Scores whether a model's branch set obeys causal axioms, checked against the
model's OWN output: no twin trajectory or reference answer. This removes
twin-circularity from the primary measurement and prevents a memorised
reference answer from directly increasing the score. It does not prove that a
model was never trained on nuScenes.

Three axioms, each in [0,1] (1 = coherent) or None when not applicable. They are
deliberately separated so that the ground truth sits at the ceiling (it does:
GT is coherent by construction), which naive "all branches must differ" scoring
does NOT satisfy — two hard brakes legitimately converge, so requiring every
intervention to differ wrongly penalises the ground truth.

1. braking_monotonicity — among interventions whose ego displacements actually
   differ, harder braking must give not-longer forward displacement:
   do(panic) <= do(emergency) <= ... <= do(no_action) <= do(accelerate).
   Forward travel is compared AT THE SAME DEPTH (the deepest slot both
   branches report), so a branch cut short by a collision or a stop is not
   read as having travelled less than a branch that ran on. Score =
   concordant / comparable ordered pairs. None if no displacement spread (a
   collapsed set cannot earn a misleading 1.0).
2. responsiveness — the interventions must produce DISTINGUISHABLE futures at
   all. Score = normalised spread of ego forward displacement across
   interventions. A model that returns do(brake) ~ do(no_action) ~
   do(accelerate) has collapsed to a single continuation dressed as branches
   and scores ~0. This is the direct measurement of the paper's science
   question. (Separated from monotonicity: random output is responsive but not
   monotone; a collapsed model is monotone-vacuous but not responsive.)
3. intervention_locality — an ego intervention must not move a STATIC actor
   (barrier / cone / road boundary): those cannot react to the ego, so their
   position must be identical across interventions. Score = fraction of
   (static-actor, depth) cells stable across interventions. None if the scene
   has no static actors. (Restricted to static actors so GT's reactive dynamic
   agents are not counted as locality violations.)
4. action_direction — a lateral action (swerve left/right) must lie on its
   own side of the road-following path by more than TOL_M, compared at
   MATCHED FORWARD PROGRESSION. The road-following path is the lateral-
   neutral branch that progresses farthest (a brake that stops early traces
   no road beyond its stop and would defeat a swerve that has already
   developed); the comparison is made at the smaller of the two endpoint
   forward coordinates, the other branch interpolated along its own path.
   Comparing endpoints directly would let a shared road curvature masquerade
   as a wrong turn. None when no lateral action is paired with a reference
   branch.
5. speed_response — a braking action must end more than SPEED_TOL_MPS below
   the root ego speed and an accelerating action above it. Per-branch, so a
   single braking action is testable even where monotonicity has no pair.
   None when no longitudinal action is comparable (e.g. braking from rest).
   The ego speed is resolved exactly as the scorer's canonical form does:
   |(vx, vy)| whenever both components are reported, else the scalar
   `speed`, so two representations of one motion cannot give different
   verdicts.

Axes 1–3 are necessary conditions that do not test what the action label
means; axes 4–5 test the lateral and longitudinal
semantics of the requested action against the published profiles, still
without reading a future reference. All five can be satisfied by a rule-based
kinematic policy; the action-profile baseline in analysis/baselines.py makes
that floor explicit and must be reported next to every model.

Both adapters read Layer-3 ego-action branches only. Layer-4 named-actor
counterfactuals are deliberately inaccessible to this measurement.

CLI:
    python -m analysis.causal_coherence --scenario-dir pilot30_protocol10 \
        --pred-dir results/pilot_matrix/gemini-2.5-pro_video_seed0 \
        --out results/coherence_gemini.json
    python -m analysis.causal_coherence --scenario-dir pilot30_protocol10 --score-gt
"""

from __future__ import annotations

import argparse
import glob
import json
import math
import os
import sys
from typing import Any, Dict, List, Optional, Tuple

# Missing node slots retain their index for comparisons at the same depth.
Chain = List[Optional[Dict[str, Any]]]

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from simulation.protocol import PROTOCOL_VERSION

# Longitudinal accel per intervention (m/s^2), from the twin's EGO_PROFILES —
# the canonical ordering braking monotonicity is checked against.
try:
    from simulation.kinematic_twin import EGO_PROFILES
    _ACCEL = {k: v["accel"] for k, v in EGO_PROFILES.items()}
except Exception:  # keep the metric usable without the twin module
    _ACCEL = {
        "do_ego_panic_brake": -9.0, "do_ego_emergency_brake": -8.0,
        "do_ego_brake_and_swerve_left": -4.0, "do_ego_brake_and_swerve_right": -4.0,
        "do_ego_comfortable_brake": -3.0, "do_ego_decelerate_coast": -1.0,
        "do_ego_no_action": 0.0, "do_ego_swerve_left": 0.0, "do_ego_swerve_right": 0.0,
        "do_ego_accelerate_pass": 3.0, "do_ego_accelerate_escape": 3.5,
    }

TOL_M = 1.0  # position tolerance (m): reuse the metric's own resolution
SPEED_TOL_MPS = 1.0  # speed floor (m/s): the same floor Sim_speed uses

# Lateral velocity sign per intervention (+y is RIGHT in the fixed t0_ego
# frame; swerve-left profiles carry a negative lateral velocity).
try:
    _LAT = {k: v["lat"] for k, v in EGO_PROFILES.items()}
except NameError:  # twin module unavailable: mirror its published signs
    _LAT = {
        "do_ego_swerve_left": -2.5, "do_ego_swerve_right": 2.5,
        "do_ego_brake_and_swerve_left": -2.5, "do_ego_brake_and_swerve_right": 2.5,
    }

# Every reference-free L3 axis, in reporting order. `structure` (below) is an
# optional GT-referenced summary and is not an axis.
COHERENCE_AXES = ("monotonicity", "responsiveness", "locality",
                  "action_direction", "speed_response")

# Per-direction split of speed_response. Reported beside the axes, never as
# axes: they are one axis seen by action kind, not two more necessary
# conditions, and they must not enter the primary-axis floor and skill rows.
COHERENCE_BREAKDOWNS = ("speed_response_brake", "speed_response_accelerate")

# Measurement identifier, recorded in every coherence report. Bumped whenever
# an axis definition changes so that scores from different definitions are
# never pooled. 1.1: direction compared at matched forward progression; speed
# resolved as the scorer does. 1.2: monotonicity compares forward travel at
# the same depth (a collision-cut branch is not read as travelling less), and
# direction compares each lateral branch with the road-following path of the
# neutral branch that progresses farthest (an early stop no longer defeats a
# developed swerve). Under 1.1 the references themselves scored 0.81 and 0.67
# on CARLA-400; both effects were measurement artefacts.
COHERENCE_VERSION = "coriva-coherence-1.2"

# Actors that cannot causally react to an ego intervention (locality axiom).
_STATIC_ROLES = {"static_barrier", "road_boundary", "obstacle_actor"}
_STATIC_ENTITIES = {"barrier", "traffic_cone", "guardrail", "road_edge",
                    "curb", "traffic_sign"}


def _is_static(actor: Dict[str, Any]) -> bool:
    role, entity = actor.get("role_type"), actor.get("entity_type")
    return ((isinstance(role, str) and role in _STATIC_ROLES)
            or (isinstance(entity, str) and entity in _STATIC_ENTITIES))


def _actors(node: Any) -> List[Dict[str, Any]]:
    """Actor dicts of one node; [] for a malformed node or actor entry.
    Coherence reads the raw prediction, so a `null` node or actor is a
    missing value, never an exception."""
    if not isinstance(node, dict):
        return []
    items = node.get("actors_update") or node.get("actors") or []
    return [a for a in items if isinstance(a, dict)] if isinstance(items, list) else []


def _num(value: Any) -> Optional[float]:
    """A finite float, or None. Coherence reads predictions before any gate,
    so a malformed field ("unknown", null, NaN) must make that value missing
    rather than abort the scoring of a whole cell."""
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value) if math.isfinite(value) else None
    if isinstance(value, str):
        try:
            parsed = float(value)
        except ValueError:
            return None
        return parsed if math.isfinite(parsed) else None
    return None


def _ego_terminal(chain: Chain) -> Optional[Tuple[float, float]]:
    for node in reversed(chain):
        for a in _actors(node):
            if a.get("role_type") == "ego_agent" or a.get("id") == "ego":
                x, y = _num(a.get("x")), _num(a.get("y"))
                if x is not None and y is not None:
                    return x, y
    return None


def _actor_speed(a: Dict[str, Any]) -> Optional[float]:
    """One speed-resolution rule for the whole measurement, identical to the
    scorer's canonical form (`ctms._pre_normalize_branch`,
    `derive_speed_from_velocity`): |(vx, vy)| whenever both components are
    finite, otherwise the scalar `speed`, otherwise None. Reading the vector
    first, as the gates do, means a branch whose motion never braked cannot
    pass speed_response by declaring a smaller scalar."""
    vx, vy = _num(a.get("vx")), _num(a.get("vy"))
    if vx is not None and vy is not None:
        return math.hypot(vx, vy)
    return _num(a.get("speed"))


def _ego_terminal_state(chain: Chain
                        ) -> Optional[Tuple[float, float, Optional[float]]]:
    """Last reported ego (x, y, speed); speed by `_actor_speed`, None when
    neither a velocity vector nor a scalar speed is reported."""
    for node in reversed(chain):
        for a in _actors(node):
            if a.get("role_type") == "ego_agent" or a.get("id") == "ego":
                x, y = _num(a.get("x")), _num(a.get("y"))
                if x is None or y is None:
                    continue
                return x, y, _actor_speed(a)
    return None


def _root_ego_speed(root: Dict[str, Any]) -> float:
    for a in root.get("actors", []):
        if not isinstance(a, dict):
            continue
        if a.get("role_type") == "ego_agent" or a.get("id") == "ego":
            return _actor_speed(a) or 0.0
    return 0.0


def _root_ego(root: Dict[str, Any]) -> Tuple[float, float]:
    for a in root.get("actors", []):
        if a.get("role_type") == "ego_agent" or a.get("id") == "ego":
            return _num(a.get("x")) or 0.0, _num(a.get("y")) or 0.0
    return 0.0, 0.0


def _ego_forward_disp(chain: Chain, root: Dict[str, Any]) -> Optional[float]:
    term = _ego_terminal(chain)
    if term is None:
        return None
    rx, _ = _root_ego(root)
    return term[0] - rx  # ego-frame +x is forward


def _ego_positions_by_depth(chain: Chain, root: Dict[str, Any]
                            ) -> Dict[int, Tuple[float, float]]:
    """{depth: ego (x, y)}: the supplied root at depth 0, then every chain
    slot that reports a numeric ego position, keyed by its slot index. A
    missing or malformed slot keeps the later depths in place, as locality
    does, so two branches are always compared at the same depth."""
    out = {0: _root_ego(root)}
    for depth, node in enumerate(chain, start=1):
        for a in _actors(node):
            if a.get("role_type") == "ego_agent" or a.get("id") == "ego":
                x, y = _num(a.get("x")), _num(a.get("y"))
                if x is not None and y is not None:
                    out[depth] = (x, y)
                break
    return out


def _common_depth(a: Dict[int, Tuple[float, float]],
                  b: Dict[int, Tuple[float, float]]) -> Optional[int]:
    """The deepest depth >= 1, no later than either branch's last report, at
    which both report the ego; None when there is no such depth."""
    for depth in range(min(max(a), max(b)), 0, -1):
        if depth in a and depth in b:
            return depth
    return None


# ── Axiom 1: braking monotonicity ────────────────────────────────────────

def braking_monotonicity(pairs: List[Tuple[str, Chain]],
                         root: Dict[str, Any]) -> Optional[float]:
    """Harder braking must give strictly shorter forward travel, compared at
    the same depth: the deepest slot both branches report, so a branch that
    ends early (a collision, a stop) is compared where the other branch was
    at that time, never its endpoint against a later endpoint."""
    rx, _ = _root_ego(root)
    items = [(_ACCEL.get(iv), _ego_positions_by_depth(ch, root)) for iv, ch in pairs]
    items = [(ac, p) for ac, p in items if ac is not None and len(p) > 1]
    comparable = concordant = 0
    for ac_i, p_i in items:
        for ac_j, p_j in items:
            if not ac_i < ac_j:
                continue
            depth = _common_depth(p_i, p_j)
            if depth is None:
                continue
            d_i, d_j = p_i[depth][0] - rx, p_j[depth][0] - rx  # ego-frame +x is forward
            # only pairs that both differ in braking AND in displacement can
            # be ordered; equal displacements carry no ordering information
            if abs(d_i - d_j) > TOL_M:
                comparable += 1
                if d_i < d_j:  # harder brake (i) -> strictly shorter travel
                    concordant += 1
    if comparable == 0:
        return None
    return concordant / comparable


# ── Axiom 2: responsiveness (non-collapse) ───────────────────────────────

def responsiveness(pairs: List[Tuple[str, Chain]],
                   root: Dict[str, Any]) -> Optional[float]:
    # Spread of ego TERMINAL POSITIONS in 2D (max pairwise distance), so that
    # laterally-distinct interventions (swerve-left vs swerve-right) count as
    # distinct even when their forward displacement is equal — a longitudinal
    # -only measure would wrongly read them as collapsed.
    terms = [_ego_terminal(ch) for _, ch in pairs]
    terms = [t for t in terms if t is not None]
    if len(terms) < 2:
        return None
    spread = max(math.hypot(a[0] - b[0], a[1] - b[1])
                 for i, a in enumerate(terms) for b in terms[i + 1:])
    speed = max(_root_ego_speed(root), 1.0)
    ref = speed * 2.0  # ~2 s of ego travel = a meaningful spread scale
    return min(1.0, spread / ref) if ref > 0 else None


# ── Axiom 3: intervention locality ───────────────────────────────────────

def _static_positions_by_depth(chain: Chain) -> Dict[Tuple[str, int], Tuple[float, float]]:
    out = {}
    for depth, node in enumerate(chain, start=1):
        for a in _actors(node):
            if not _is_static(a):
                continue  # only static actors can test locality (they cannot react)
            x, y = _num(a.get("x")), _num(a.get("y"))
            aid = a.get("id")
            if isinstance(aid, (str, int)) and not isinstance(aid, bool) and x is not None and y is not None:
                out[(str(aid), depth)] = (x, y)
    return out


def intervention_locality(pairs: List[Tuple[str, Chain]],
                          root: Dict[str, Any]) -> Optional[float]:
    per_iv = [_static_positions_by_depth(ch) for _, ch in pairs]
    if len(per_iv) < 2:
        return None
    # gather (actor, depth) cells seen in >= 2 interventions
    keys: Dict[Tuple[str, int], List[Tuple[float, float]]] = {}
    for positions in per_iv:
        for k, xy in positions.items():
            keys.setdefault(k, []).append(xy)
    stable = total = 0
    for k, xys in keys.items():
        if len(xys) < 2:
            continue
        total += 1
        spread = max(math.hypot(a[0] - b[0], a[1] - b[1])
                     for i, a in enumerate(xys) for b in xys[i + 1:])
        if spread <= TOL_M:
            stable += 1
    return stable / total if total else None


# ── Axiom 4: action direction (lateral semantics) ────────────────────────

def _ego_path(chain: Chain, root: Dict[str, Any]
              ) -> List[Tuple[float, float]]:
    """Ego (x, y) from the supplied root through every node that reports the
    ego, in chain order."""
    path = [_root_ego(root)]
    for node in chain:
        for a in _actors(node):
            if a.get("role_type") == "ego_agent" or a.get("id") == "ego":
                x, y = _num(a.get("x")), _num(a.get("y"))
                if x is not None and y is not None:
                    path.append((x, y))
                break
    return path


def _lateral_at_progress(path: List[Tuple[float, float]], x: float) -> Optional[float]:
    """y of the path at forward coordinate x, linearly interpolated between
    the two nodes that bracket x. None when the path's forward coordinate is
    not non-decreasing (the ego reverses or the nodes are out of order) or
    x lies outside the path."""
    if len(path) < 2 or any(b[0] < a[0] for a, b in zip(path, path[1:])):
        return None
    if x < path[0][0] or x > path[-1][0]:
        return None
    for (x0, y0), (x1, y1) in zip(path, path[1:]):
        if x0 <= x <= x1:
            if x1 == x0:
                return y1
            return y0 + (y1 - y0) * (x - x0) / (x1 - x0)
    return None


def _matched_lateral_difference(path_a: List[Tuple[float, float]],
                                path_b: List[Tuple[float, float]]) -> float:
    """y_a - y_b at the smaller of the two endpoint forward coordinates; the
    branch that ends there contributes its endpoint, the other its lateral
    position interpolated along its own path. Falls back to the endpoint
    difference when a path cannot be interpolated."""
    def monotone(path):
        return len(path) >= 2 and all(b[0] >= a[0] for a, b in zip(path, path[1:]))
    # Both paths must be interpolable, including the one that ends at x*:
    # a shorter path that reverses has no defined progression either, and
    # taking its endpoint while interpolating the other would silently
    # change the verdict.
    if not monotone(path_a) or not monotone(path_b):
        return path_a[-1][1] - path_b[-1][1]
    x_star = min(path_a[-1][0], path_b[-1][0])
    ya = path_a[-1][1] if path_a[-1][0] == x_star else _lateral_at_progress(path_a, x_star)
    yb = path_b[-1][1] if path_b[-1][0] == x_star else _lateral_at_progress(path_b, x_star)
    if ya is None or yb is None:
        return path_a[-1][1] - path_b[-1][1]
    return ya - yb


def action_direction(pairs: List[Tuple[str, Chain]],
                     root: Dict[str, Any]) -> Optional[float]:
    """A lateral action must lie on its own side of the road-following path
    by more than TOL_M, compared at matched forward progression.

    The road-following path is the lateral-neutral branch whose forward
    progression reaches farthest (among the neutral branches whose forward
    coordinate is monotone; else the one that ends farthest). Let x* be the
    smaller of the lateral branch's and that path's endpoint forward
    coordinates. The branch that ends at x* contributes its endpoint y; the
    other its lateral position interpolated along its own path at x*. The
    lateral branch is correct when (y_lateral - y_road) has the profile's sign
    and exceeds TOL_M. Matching progression removes the confound of a road
    curvature that both branches follow but that a shorter branch has
    traversed less of; taking the farthest neutral branch removes the other
    confound, a brake that stops within the first metres, where no swerve
    can yet have moved TOL_M sideways although it is fully developed later
    (under 1.1 the references scored 0.5 on every crossing template for
    that reason). The assumption made explicit is that the branches follow
    the same road, so their road-following lateral positions agree at equal
    forward progression; that is how both references and the control are
    built. When a path's forward coordinate is not monotone the comparison
    falls back to endpoints. Without a neutral branch, opposite-signed
    lateral actions are compared with each other the same way. None when
    the set supplies no comparison. It tests the sign of the lateral
    response, not its magnitude."""
    lateral, neutral = [], []
    for iv, ch in pairs:
        lat = _LAT.get(iv)
        term = _ego_terminal_state(ch)
        if lat is None or term is None:
            continue
        path = _ego_path(ch, root)
        if lat == 0.0:
            neutral.append(path)
        else:
            lateral.append((1.0 if lat > 0 else -1.0, path))
    comparable = correct = 0
    if neutral:
        def monotone(path):
            return len(path) >= 2 and all(b[0] >= a[0] for a, b in zip(path, path[1:]))
        traced = [p for p in neutral if monotone(p)] or neutral
        road = max(traced, key=lambda p: p[-1][0])
        for sign, path in lateral:
            comparable += 1
            if _matched_lateral_difference(path, road) * sign > TOL_M:
                correct += 1
    if not neutral:
        for i, (s_i, p_i) in enumerate(lateral):
            for s_j, p_j in lateral[i + 1:]:
                if s_i == s_j:
                    continue
                comparable += 1
                right_p, left_p = (p_i, p_j) if s_i > 0 else (p_j, p_i)
                if _matched_lateral_difference(right_p, left_p) > TOL_M:
                    correct += 1
    if comparable == 0:
        return None
    return correct / comparable


# ── Axiom 5: speed response (longitudinal semantics) ─────────────────────

def _speed_response_counts(pairs: List[Tuple[str, Chain]],
                           root: Dict[str, Any]) -> Dict[str, List[int]]:
    """[correct, comparable] per longitudinal action kind.

    The single source of the speed_response comparison, so the blended axis and
    its per-direction breakdown can never disagree.
    """
    v0 = _root_ego_speed(root)
    counts = {"brake": [0, 0], "accelerate": [0, 0]}
    for iv, ch in pairs:
        accel = _ACCEL.get(iv)
        term = _ego_terminal_state(ch)
        if accel is None or accel == 0.0 or term is None or term[2] is None:
            continue
        if accel < 0:
            if v0 <= SPEED_TOL_MPS:
                continue
            counts["brake"][1] += 1
            if term[2] < v0 - SPEED_TOL_MPS:
                counts["brake"][0] += 1
        else:
            counts["accelerate"][1] += 1
            if term[2] > v0 + SPEED_TOL_MPS:
                counts["accelerate"][0] += 1
    return counts


def speed_response(pairs: List[Tuple[str, Chain]],
                   root: Dict[str, Any]) -> Optional[float]:
    """A braking action must end more than SPEED_TOL_MPS below the root ego
    speed; an accelerating action more than SPEED_TOL_MPS above it. Actions
    with zero longitudinal profile, branches without a usable ego speed, and
    braking from a root already at or below the floor supply no comparison.
    None when nothing is comparable.

    The two directions are very unequal work: in hazard scenes almost any
    continuation ends slower, so the braking half is close to tautological
    while the accelerating half is not. The blended value is dominated by
    whichever direction the query mix supplies more of, so read it beside the
    `speed_response_brake` / `speed_response_accelerate` breakdown, never alone.
    """
    counts = _speed_response_counts(pairs, root)
    correct = counts["brake"][0] + counts["accelerate"][0]
    comparable = counts["brake"][1] + counts["accelerate"][1]
    if comparable == 0:
        return None
    return correct / comparable


# ── Composite ─────────────────────────────────────────────────────────────

def coherence_from_pairs(pairs: List[Tuple[str, Chain]],
                         root: Dict[str, Any]) -> Dict[str, Optional[float]]:
    """Per-scenario coherence. Reports the three axioms SEPARATELY (each may be
    None when not applicable). There is deliberately NO single 'composite'
    number: monotonicity and locality are ceiling axioms (GT = 1) while
    responsiveness is a collapse detector with no GT target, and averaging
    axioms with different applicability across scenarios would produce an
    aggregate that does not match the per-axis means. Consumers must report
    each axiom with its available-n. `structure` = mean of the two ceiling
    axioms only (mono, locality), when at least one is present — an
    optional GT-referenced summary that excludes the no-target responsiveness."""
    m = braking_monotonicity(pairs, root)
    r = responsiveness(pairs, root)
    loc = intervention_locality(pairs, root)
    direction = action_direction(pairs, root)
    speed = speed_response(pairs, root)
    sc = _speed_response_counts(pairs, root)
    speed_by_kind = {f"speed_response_{k}": (c[0] / c[1] if c[1] else None)
                     for k, c in sc.items()}
    ceiling = [v for v in (m, loc) if v is not None]
    return {"monotonicity": m, "responsiveness": r, "locality": loc,
            "action_direction": direction, "speed_response": speed,
            **speed_by_kind,
            "structure": (sum(ceiling) / len(ceiling)) if ceiling else None,
            "n_interventions": len(pairs)}


# ── Adapters ──────────────────────────────────────────────────────────────

def pairs_from_gt(gt: Dict[str, Any]) -> List[Tuple[str, Chain]]:
    out = []
    for b in gt.get("branches", []):
        intervention = b.get("intervention") or {}
        iv = intervention.get("intervention_type")
        if iv and intervention.get("target_actor_id", "ego") == "ego" and b.get("chain"):
            out.append((iv, b["chain"]))
    return out


def pairs_from_prediction(pred: Dict[str, Any],
                          factual_action: Optional[str] = None
                          ) -> List[Tuple[str, Chain]]:
    """Read action identity from predicted Layer-3 branch metadata.

    ``factual_action`` supplies the factual branch's action only for
    predictions whose factual branch carries no intervention object.
    """
    out = []
    branches = pred.get("predicted_branches", []) if isinstance(pred, dict) else []
    for branch in (branches if isinstance(branches, list) else []):
        # The parser keeps `[null]` and other malformed units so the schema
        # gate can fail them; here they are simply units without a chain.
        if not isinstance(branch, dict):
            continue
        intervention = branch.get("intervention")
        intervention = intervention if isinstance(intervention, dict) else {}
        iv = intervention.get("intervention_type")
        if intervention.get("target_actor_id", "ego") != "ego":
            continue
        if not iv and factual_action and branch.get("is_factual"):
            iv = factual_action
        chain = branch.get("chain")
        # Preserve depth slots: _actors(None) supplies no observation, while
        # later nodes retain their index for cross-branch locality. A chain
        # with no object nodes still supplies no usable intervention.
        chain = [n if isinstance(n, dict) else None for n in chain] if isinstance(chain, list) else []
        if isinstance(iv, str) and iv and any(n is not None for n in chain):
            out.append((iv, chain))
    return out


# ── Dataset-level runner ──────────────────────────────────────────────────

def run_coherence(gt_dir: str, pred_dir: Optional[str] = None,
                  factual_from_gt: bool = False) -> Dict[str, Any]:
    """Score coherence per scenario and aggregate. If pred_dir is None, score
    the GT itself (validation: must be ~1). Otherwise score the model's
    predictions. Protocol predictions must carry their own intervention
    metadata. ``factual_from_gt`` is an explicit escape hatch for datasets
    without that metadata and is never enabled in the protocol-1.0 primary
    measurement."""
    rows: List[Dict[str, Any]] = []
    used_legacy_reference_metadata = False
    used_legacy_input_root = False
    scenario_dirs = sorted(glob.glob(os.path.join(gt_dir, "scenario_*")))
    for scenario_dir in scenario_dirs:
        if not os.path.isdir(scenario_dir):
            continue
        name = os.path.basename(scenario_dir)
        gt_path = os.path.join(scenario_dir, "ground_truth.json")
        if pred_dir is None:
            if not os.path.isfile(gt_path):
                continue
            gt = json.load(open(gt_path))
            root = gt["observed_state"]
            pairs = pairs_from_gt(gt)
            scenario_id = gt.get("scenario_id", name)
        else:
            l3_path = os.path.join(pred_dir, name, "l3_pre.json")
            if not os.path.exists(l3_path):
                continue
            state_path = os.path.join(scenario_dir, "observed_state.json")
            manifest_path = os.path.join(scenario_dir, "input_manifest.json")
            if os.path.isfile(state_path):
                root = json.load(open(state_path))
            elif os.path.isfile(gt_path):
                manifest = (json.load(open(manifest_path))
                            if os.path.isfile(manifest_path) else {})
                if manifest.get("protocol_version") == PROTOCOL_VERSION:
                    raise ValueError(
                        f"{name}: protocol {PROTOCOL_VERSION} requires observed_state.json; "
                        "primary coherence will not recover it from ground_truth.json")
                # Datasets without a standalone input root: read it from the GT.
                root = json.load(open(gt_path))["observed_state"]
                used_legacy_input_root = True
            else:
                continue
            pred = json.load(open(l3_path))
            fa = None
            if factual_from_gt:
                if not os.path.isfile(gt_path):
                    raise ValueError("legacy factual metadata requested but ground_truth.json is missing")
                gt = json.load(open(gt_path))
                fa = next((b["intervention"]["intervention_type"] for b in gt.get("branches", [])
                           if b.get("is_factual")), None)
                used_legacy_reference_metadata = used_legacy_reference_metadata or fa is not None
            pairs = pairs_from_prediction(pred, factual_action=fa)
            scenario_id = name
        out = coherence_from_pairs(pairs, root)
        out["scenario_id"] = scenario_id
        rows.append(out)

    def _mean_n(key):
        vals = [r[key] for r in rows if r.get(key) is not None]
        return {"mean": (sum(vals) / len(vals)) if vals else None, "n": len(vals)}

    # Report each axiom with its available-n; the aggregate is NOT a single
    # number averaged over axioms of differing applicability.
    validating_reference = pred_dir is None
    return {"measurement": ("reference_coherence_sanity_check" if validating_reference
                             else "ground_truth_free_l3_causal_coherence"),
            "coherence_version": COHERENCE_VERSION,
            "measurement_role": ("instrument_validation" if validating_reference
                                  else "primary"),
            "uses_reference_answers": validating_reference,
            "used_legacy_reference_metadata": used_legacy_reference_metadata,
            "used_legacy_input_root": used_legacy_input_root,
            "n_scenarios": len(rows),
            "axes": {k: _mean_n(k) for k in COHERENCE_AXES + ("structure",)},
            "axis_breakdown": {k: _mean_n(k) for k in COHERENCE_BREAKDOWNS},
            "per_scenario": rows}


def _check_version(report: Dict[str, Any]) -> None:
    """Every coherence report that is read, merged or compared must name the
    current axis definition; a report without a version is refused as well,
    so unversioned data can never be pooled and re-emitted as current."""
    version = report.get("coherence_version")
    if version != COHERENCE_VERSION:
        raise ValueError(f"coherence report version {version!r} is not {COHERENCE_VERSION!r}")


def _load_coherence(value: Any) -> Dict[str, Any]:
    if isinstance(value, (list, tuple)):
        reports = [_load_coherence(item) for item in value]
        if any(report.get("uses_reference_answers") for report in reports):
            raise ValueError("cannot average reference-answer coherence as a primary report")
        for report in reports:
            _check_version(report)
        per_seed = [{row["scenario_id"]: row
                     for row in report.get("per_scenario", [])}
                    for report in reports]
        if not per_seed:
            return {"uses_reference_answers": False, "per_scenario": []}
        shared = set(per_seed[0])
        for rows in per_seed[1:]:
            shared &= set(rows)
        merged = []
        for scenario_id in sorted(shared):
            row: Dict[str, Any] = {"scenario_id": scenario_id}
            for axis in COHERENCE_AXES + ("structure",):
                values = [seed_rows[scenario_id].get(axis) for seed_rows in per_seed]
                row[axis] = (sum(float(item) for item in values) / len(values)
                             if all(item is not None for item in values) else None)
            merged.append(row)
        return {
            "measurement": "ground_truth_free_l3_causal_coherence",
            "coherence_version": COHERENCE_VERSION,
            "measurement_role": "primary",
            "uses_reference_answers": False,
            "n_seeds": len(reports),
            "per_scenario": merged,
        }
    if isinstance(value, dict):
        _check_version(value)
        return value
    with open(value, encoding="utf-8") as f:
        report = json.load(f)
    _check_version(report)
    return report


def _paired_axis(a: Any, b: Any, axis: str) -> Tuple[List[float], List[float]]:
    ra, rb = _load_coherence(a), _load_coherence(b)
    if ra.get("uses_reference_answers") or rb.get("uses_reference_answers"):
        raise ValueError("primary coherence deltas cannot use reference-answer reports")
    aa = {r["scenario_id"]: r.get(axis) for r in ra.get("per_scenario", [])
          if r.get(axis) is not None}
    bb = {r["scenario_id"]: r.get(axis) for r in rb.get("per_scenario", [])
          if r.get(axis) is not None}
    shared = sorted(set(aa) & set(bb))
    return [float(aa[s]) for s in shared], [float(bb[s]) for s in shared]


def coherence_condition_deltas(video: Any, history_json: Any, video_history: Any,
                               n_boot: int = 10_000, seed: int = 0) -> Dict[str, Any]:
    """Paired input-condition effects for each primary coherence axis."""
    from analysis.bootstrap import paired_bootstrap_diff

    def delta(a: Any, b: Any, axis: str) -> Dict[str, Any]:
        va, vb = _paired_axis(a, b, axis)
        if not va:
            return {"mean_diff": None, "lo": None, "hi": None,
                    "significant": False, "n": 0}
        out = paired_bootstrap_diff(va, vb, n_boot=n_boot, seed=seed)
        out["n"] = len(va)
        return out

    return {
        "measurement": "ground_truth_free_l3_causal_coherence",
        "measurement_role": "primary",
        "axes": {
            axis: {
                "video_history_minus_video": delta(video_history, video, axis),
                "video_history_minus_history_json": delta(
                    video_history, history_json, axis),
                "video_minus_history_json": delta(video, history_json, axis),
            }
            for axis in COHERENCE_AXES
        },
    }


def summarize_coherence_reports(reports: Any, n_boot: int = 10_000,
                                seed: int = 0) -> Dict[str, Any]:
    """Average seeds per scenario, then bootstrap scenarios for each axis."""
    from analysis.bootstrap import bootstrap_ci

    values = _load_coherence(reports)
    if values.get("uses_reference_answers"):
        raise ValueError("primary coherence summary cannot use reference answers")
    rows = values.get("per_scenario", [])
    axes: Dict[str, Any] = {}
    for axis in COHERENCE_AXES:
        scores = [float(row[axis]) for row in rows if row.get(axis) is not None]
        if scores:
            axes[axis] = {**bootstrap_ci(scores, n_boot=n_boot, seed=seed),
                          "n": len(scores)}
        else:
            axes[axis] = {"mean": None, "lo": None, "hi": None, "n": 0}
    return {
        "measurement": "ground_truth_free_l3_causal_coherence",
        "measurement_role": "primary",
        "uses_reference_answers": False,
        "n_seeds": values.get("n_seeds", 1),
        "n_scenarios": len(rows),
        "axes": axes,
        "per_scenario": rows,
    }


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--scenario-dir", "--gt-dir", dest="scenario_dir",
                   help="scenario root containing observed_state.json and scenario_* dirs")
    p.add_argument("--pred-dir", help="omit (or use --score-gt) to score the GT itself")
    p.add_argument("--score-gt", action="store_true", help="score coherence of the GT (validation)")
    p.add_argument("--factual-from-gt", action="store_true", default=False,
                   help="legacy-only: recover missing factual action metadata from GT")
    p.add_argument("--out")
    p.add_argument("--video-coherence",
                   help="report path, or comma-separated seed report paths")
    p.add_argument("--history-json-coherence",
                   help="report path, or comma-separated seed report paths")
    p.add_argument("--video-history-coherence",
                   help="report path, or comma-separated seed report paths")
    p.add_argument("--average-coherence",
                   help="comma-separated seed reports to average and bootstrap")
    p.add_argument("--n-boot", type=int, default=10_000)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()
    compare_paths = (args.video_coherence, args.history_json_coherence,
                     args.video_history_coherence)
    if args.average_coherence:
        if any(compare_paths) or args.scenario_dir or args.pred_dir or args.score_gt:
            p.error("--average-coherence cannot be combined with scoring/comparison inputs")
        report_paths = [item for item in args.average_coherence.split(",") if item]
        out = summarize_coherence_reports(
            report_paths, n_boot=args.n_boot, seed=args.seed)
        print(json.dumps(out["axes"], indent=2),
              f"\n(scenarios={out['n_scenarios']}, seeds={out['n_seeds']})")
    elif any(compare_paths):
        if not all(compare_paths):
            p.error("coherence comparison requires all three condition reports")
        def paths(value: str):
            items = [item for item in value.split(",") if item]
            return items if len(items) > 1 else items[0]
        out = coherence_condition_deltas(
            paths(args.video_coherence), paths(args.history_json_coherence),
            paths(args.video_history_coherence), n_boot=args.n_boot, seed=args.seed)
        print(json.dumps(out, indent=2))
    else:
        if not args.scenario_dir:
            p.error("--scenario-dir is required when scoring coherence")
        pred_dir = None if args.score_gt else args.pred_dir
        if pred_dir is None and not args.score_gt:
            p.error("provide --pred-dir or --score-gt")
        out = run_coherence(args.scenario_dir, pred_dir=pred_dir,
                            factual_from_gt=args.factual_from_gt)
        print(json.dumps(out["axes"], indent=2), f"\n(scenarios={out['n_scenarios']})")
    if args.out:
        with open(args.out, "w") as f:
            json.dump(out, f, indent=2)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
