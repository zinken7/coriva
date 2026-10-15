"""Label-free decomposition of a scenario's predicted chains.

CTMS answers "how far is the prediction from the reference" with one number
per task. Three different failures land on the same low number: a model that emits one node and stops, a model whose
chain carries another actor's event label at depth 1 (scored zero by the
node gate), and a model whose trajectory is a metre off. The dissertation's
question is which of these happens, so this module reads the same predicted
and reference chains and reports the pieces separately. It never changes a
score; every value is a diagnostic reported next to CTMS.

Per scenario, `scenario_diagnostics(gt, l3_pre, l4_pre)` returns scalar
metrics (None = undefined for that scenario):

Structure
- l3_branch_completeness: distinct declared ego actions answered / declared.
- l3_chain_length_ratio: mean over action-matched branches of
  len(pred chain) / len(reference chain), capped at 1.
- l3_one_node_share: share of action-matched branches with exactly one node.
- l4_chain_length_ratio, l4_one_node: the same for the matched L4 chain.
- snapshot_completeness: mean over predicted nodes of 1[every root actor is
  listed in the node].

Semantics of the requested action / intervention
- l3_speed_sign_correct: over brake/accelerate branches, share whose last
  ego speed moved against/with the root speed as the action requires.
- l4_target_present: 1 if the intervened actor appears in every node of the
  matched L4 chain.
- l4_target_brake_observed (npc_sudden_brake only): 1 if the intervened
  actor's last predicted speed is below its root speed by more than
  0.5 m/s, else 0; None otherwise.
- l4_treated_lateral_sign_vs_reference (cut-in and road-entry types): 1 if
  the sign of the treated target's lateral displacement from the root, at
  the last common depth, agrees with the REFERENCE's (a displacement within
  0.2 m counts as none), else 0. This is a reference-anchored diagnostic
  (it reads the reference chain); the reference-free direction test of the
  pair protocol is `analysis/intervention_pair.direction_consistency`.
- l4_ego_speed_change: |last predicted ego speed - root ego speed| in the
  treated chain; a motion diagnostic, not a test of the declared ego policy.
- l4_control_present: 1 if the matched L4 unit carries a non-empty control
  chain (pair protocol), else 0.

Geometry (metres, position error against the reference node at the same
depth; depths are aligned by index as the scorer does, AFTER any echoed
leading observed_state node is dropped, exactly as the scorer drops it, so a
root echo changes no diagnostic)
- l3_ego_error_d1, l3_ego_error_d3, l3_ego_error_end: mean over matched
  branches at depth 1, depth 3 and the last common depth.
- l4_target_error_d1, l4_target_error_d3, l4_target_error_end,
  l4_ego_error_end: the intervened actor and the ego in the treated chain.
- l4_control_target_error_end, l4_control_ego_error_end: the same at the
  last common depth of the predicted control chain against the reference
  control chain (pair protocol; undefined on a legacy reference).
- static_actor_displacement: mean displacement between root and the last
  predicted L4 node of actors that are stationary at the root and are not the
  intervened actor (the reference keeps them still).

A malformed field (a non-numeric coordinate, a container of the wrong type,
a non-string label) is a MISSING value here, never a default of 0 and never
an exception: the scorer's schema gate is where such a unit fails.
"""
from __future__ import annotations

import math
import statistics
from typing import Any, Dict, List, Optional

BRAKE_SPEED_DROP_MPS = 0.5
LANE_APPROACH_M = 0.2
STATIC_SPEED_MPS = 0.05

METRICS = (
    "l3_branch_completeness", "l3_chain_length_ratio", "l3_one_node_share",
    "l3_speed_sign_correct", "l3_ego_error_d1", "l3_ego_error_d3", "l3_ego_error_end",
    "l4_chain_length_ratio", "l4_one_node", "l4_target_present",
    "l4_target_brake_observed", "l4_treated_lateral_sign_vs_reference",
    "l4_ego_speed_change", "l4_target_error_d1", "l4_target_error_d3", "l4_target_error_end",
    "l4_ego_error_end", "l4_control_present", "l4_control_target_error_end", "l4_control_ego_error_end",
    "snapshot_completeness", "static_actor_displacement",
)


_EVENT_SYNONYMS: Optional[Dict[str, str]] = None


def _event_synonyms() -> Dict[str, str]:
    global _EVENT_SYNONYMS
    if _EVENT_SYNONYMS is None:
        import os
        import yaml
        path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "ctms", "ontology.yaml")
        try:
            with open(path, encoding="utf-8") as f:
                _EVENT_SYNONYMS = dict(((yaml.safe_load(f) or {}).get("synonyms") or {}).get("event_type") or {})
        except OSError:
            _EVENT_SYNONYMS = {}
    return _EVENT_SYNONYMS


def _event(node: Dict[str, Any]) -> Optional[str]:
    ev = node.get("event_type")
    if not isinstance(ev, str):
        return None
    return _event_synonyms().get(ev, ev)


def _actors(node: Any) -> Dict[str, Dict[str, Any]]:
    if not isinstance(node, dict):
        return {}
    items = node.get("actors_update")
    if not isinstance(items, list) or not items:
        items = node.get("actors")
    if not isinstance(items, list):
        return {}
    return {str(a["id"]): a for a in items
            if isinstance(a, dict) and isinstance(a.get("id"), (str, int)) and not isinstance(a.get("id"), bool)}


def _ego_id(root: Dict[str, Dict[str, Any]]) -> Optional[str]:
    for aid, a in root.items():
        if a.get("role_type") == "ego_agent" or aid == "ego":
            return aid
    return None


def _num(a: Dict[str, Any], key: str) -> Optional[float]:
    """A finite number, or None for a missing or malformed field."""
    value = a.get(key)
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


def _speed(a: Dict[str, Any]) -> Optional[float]:
    """Same rule as the scorer's canonical form and the coherence axes
    (`causal_coherence._actor_speed`): the velocity vector when both
    components are present, else the scalar, else None."""
    vx, vy = _num(a, "vx"), _num(a, "vy")
    if vx is not None and vy is not None:
        return math.hypot(vx, vy)
    return _num(a, "speed")


def _dist(a: Dict[str, Any], b: Dict[str, Any]) -> Optional[float]:
    ax, ay, bx, by = _num(a, "x"), _num(a, "y"), _num(b, "x"), _num(b, "y")
    if None in (ax, ay, bx, by):
        return None
    return math.hypot(ax - bx, ay - by)


def _append(bucket: List[float], value: Optional[float]) -> None:
    if value is not None:
        bucket.append(value)


def _mean(values: List[float]) -> Optional[float]:
    return statistics.fmean(values) if values else None


def _chain(unit: Any) -> List[Optional[Dict[str, Any]]]:
    """The FUTURE nodes of a unit's chain, positions preserved: any leading
    echoed observed_state node(s) are dropped (the scorer's root rule,
    `ctms._pre_normalize_branch`), and a node that is not an object stays
    in place as None so that later nodes keep their depth."""
    if not isinstance(unit, dict):
        return []
    chain = unit.get("chain")
    if not isinstance(chain, list):
        return []
    nodes: List[Optional[Dict[str, Any]]] = [n if isinstance(n, dict) else None for n in chain]
    while nodes and nodes[0] is not None and _event(nodes[0]) == "observed_state":
        nodes = nodes[1:]
    return nodes


def _treated_unit(unit: Any) -> Optional[Dict[str, Any]]:
    """`{"chain": <treated chain>}` of a pair-protocol unit (`treated_chain`,
    alias `chain`), or None when the unit holds no future node."""
    if not isinstance(unit, dict):
        return None
    chain = unit.get("treated_chain") if "treated_chain" in unit else unit.get("chain")
    pseudo = {"chain": chain}
    return pseudo if _chain(pseudo) else None


def _intervention_type(unit: Any) -> Optional[str]:
    if not isinstance(unit, dict):
        return None
    iv = unit.get("intervention")
    value = iv.get("intervention_type") if isinstance(iv, dict) else unit.get("intervention_type")
    return value if isinstance(value, str) else None


LATERAL_TYPES = ("npc_cut_in_left", "npc_cut_in_right", "pedestrian_enters_road", "cyclist_enters_road")


def target_brake_observed(root_actor: Dict[str, Any], last_actor: Dict[str, Any]) -> Optional[int]:
    """1 if the target ends more than BRAKE_SPEED_DROP_MPS below its root
    speed; None when a speed is missing."""
    v0, v1 = _speed(root_actor), _speed(last_actor)
    if v0 is None or v1 is None:
        return None
    return int(v1 < v0 - BRAKE_SPEED_DROP_MPS)


def _lateral_sign(root_actor: Dict[str, Any], actor: Dict[str, Any]) -> Optional[int]:
    y0, y1 = _num(root_actor, "y"), _num(actor, "y")
    if y0 is None or y1 is None:
        return None
    d = y1 - y0
    return 0 if abs(d) <= LANE_APPROACH_M else (1 if d > 0 else -1)


def target_lateral_sign_agreement(root_actor: Dict[str, Any], pred_actor: Dict[str, Any],
                                  ref_actor: Dict[str, Any]) -> Optional[int]:
    """1 if the predicted and reference lateral displacement signs from the
    root agree (dead zone LANE_APPROACH_M); None when a value is missing."""
    sp, sr = _lateral_sign(root_actor, pred_actor), _lateral_sign(root_actor, ref_actor)
    if sp is None or sr is None:
        return None
    return int(sp == sr)


def scenario_diagnostics(gt: Dict[str, Any], l3_pre: Optional[Dict[str, Any]],
                         l4_pre: Optional[Dict[str, Any]]) -> Dict[str, Optional[float]]:
    out: Dict[str, Optional[float]] = {k: None for k in METRICS}
    root = _actors(gt.get("observed_state") or {})
    ego = _ego_id(root)
    if ego is None:
        return out
    snapshot: List[float] = []

    # ── Layer 3 ────────────────────────────────────────────────────────────
    declared = {}
    for b in gt.get("branches") or []:
        it = _intervention_type(b)
        if it and (b.get("intervention") or {}).get("target_actor_id", "ego") == "ego":
            declared[it] = b
    pred_branches = (l3_pre or {}).get("predicted_branches") if isinstance(l3_pre, dict) else None
    pred_branches = pred_branches if isinstance(pred_branches, list) else []
    answered = set()
    length_ratios: List[float] = []; one_node: List[float] = []
    err_d1: List[float] = []; err_d3: List[float] = []; err_end: List[float] = []
    sign_ok: List[float] = []
    for b in pred_branches:
        it = _intervention_type(b); chain = _chain(b)
        if not it or not chain:
            continue
        for node in chain:
            if node is not None:
                snapshot.append(1.0 if set(root) <= set(_actors(node)) else 0.0)
        ref = declared.get(it)
        if ref is None:
            continue
        answered.add(it)
        ref_chain = _chain(ref)
        if ref_chain:
            length_ratios.append(min(1.0, len(chain) / len(ref_chain)))
        one_node.append(1.0 if len(chain) == 1 else 0.0)
        for depth, bucket in ((1, err_d1), (3, err_d3)):
            if depth <= len(chain) and depth <= len(ref_chain):
                pa, ga = _actors(chain[depth - 1]), _actors(ref_chain[depth - 1])
                if ego in pa and ego in ga:
                    _append(bucket, _dist(pa[ego], ga[ego]))
        m = min(len(chain), len(ref_chain))
        if m:
            pa, ga = _actors(chain[m - 1]), _actors(ref_chain[m - 1])
            if ego in pa and ego in ga:
                _append(err_end, _dist(pa[ego], ga[ego]))
        if "brake" in it or "accelerate" in it:
            pa = _actors(chain[-1])
            if ego in pa:
                v0, v1 = _speed(root[ego]), _speed(pa[ego])
                if v0 is not None and v1 is not None:
                    want = -1 if "brake" in it else 1
                    sign_ok.append(1.0 if (v1 > v0) - (v1 < v0) == want else 0.0)
    if declared:
        out["l3_branch_completeness"] = len(answered & set(declared)) / len(declared)
    out["l3_chain_length_ratio"] = _mean(length_ratios)
    out["l3_one_node_share"] = _mean(one_node)
    out["l3_speed_sign_correct"] = _mean(sign_ok)
    out["l3_ego_error_d1"], out["l3_ego_error_d3"], out["l3_ego_error_end"] = _mean(err_d1), _mean(err_d3), _mean(err_end)

    # ── Layer 4 ────────────────────────────────────────────────────────────
    queries = gt.get("l4_queries") or gt.get("counterfactuals") or []
    pred_cfs = (l4_pre or {}).get("predicted_counterfactuals") if isinstance(l4_pre, dict) else None
    pred_cfs = pred_cfs if isinstance(pred_cfs, list) else []
    ratios: List[float] = []; ones: List[float] = []; present: List[float] = []
    brake: List[float] = []; lateral: List[float] = []
    policy: List[float] = []; t_d1: List[float] = []; t_d3: List[float] = []; t_end: List[float] = []
    e_end: List[float] = []; static_disp: List[float] = []
    control_present: List[float] = []; c_t_end: List[float] = []; c_e_end: List[float] = []
    for q in queries:
        cc = q.get("counterfactual_chain") if isinstance(q, dict) else None
        if not isinstance(cc, dict):
            continue
        iv = cc.get("intervention") or {}
        itype = iv.get("intervention_type") or q.get("intervention_type")
        target = str(iv.get("target_actor_id")) if iv.get("target_actor_id") is not None else None
        ref_chain = _chain(cc)
        # Pair protocol: the unit carries `treated_chain` (alias `chain`) and
        # `control_chain`; the treated chain feeds the diagnostics below.
        match = next((c for c in pred_cfs if _intervention_type(c) == itype and _treated_unit(c)), None)
        if match is None:
            continue
        chain = _chain(_treated_unit(match))
        control_unit = {"chain": match.get("control_chain")} if isinstance(match, dict) else {}
        control_chain = _chain(control_unit)
        control_present.append(1.0 if control_chain else 0.0)
        ref_control = _chain(q.get("control_chain")) if isinstance(q.get("control_chain"), dict) else []
        if control_chain and ref_control:
            m_c = min(len(control_chain), len(ref_control))
            pa, ga = _actors(control_chain[m_c - 1]), _actors(ref_control[m_c - 1])
            if target is not None and target in pa and target in ga:
                _append(c_t_end, _dist(pa[target], ga[target]))
            if ego in pa and ego in ga:
                _append(c_e_end, _dist(pa[ego], ga[ego]))
        nodes = [_actors(n) for n in chain]
        for n, na in zip(chain, nodes):
            if n is not None:
                snapshot.append(1.0 if set(root) <= set(na) else 0.0)
        if ref_chain:
            ratios.append(min(1.0, len(chain) / len(ref_chain)))
        ones.append(1.0 if len(chain) == 1 else 0.0)
        if target is not None and target in root:
            present.append(1.0 if all(target in na for na in nodes) else 0.0)
            last_with_target = next((na for na in reversed(nodes) if target in na), None)
            if last_with_target is not None and itype == "npc_sudden_brake":
                d = target_brake_observed(root[target], last_with_target[target])
                if d is not None:
                    brake.append(float(d))
            for depth, bucket in ((1, t_d1), (3, t_d3)):
                if depth <= len(chain) and depth <= len(ref_chain):
                    pa, ga = nodes[depth - 1], _actors(ref_chain[depth - 1])
                    if target in pa and target in ga:
                        _append(bucket, _dist(pa[target], ga[target]))
            m = min(len(chain), len(ref_chain))
            if m:
                pa, ga = nodes[m - 1], _actors(ref_chain[m - 1])
                if target in pa and target in ga:
                    _append(t_end, _dist(pa[target], ga[target]))
                    if itype in LATERAL_TYPES:
                        d = target_lateral_sign_agreement(root[target], pa[target], ga[target])
                        if d is not None:
                            lateral.append(float(d))
                if ego in pa and ego in ga:
                    _append(e_end, _dist(pa[ego], ga[ego]))
        if ego in nodes[-1]:
            v0, v1 = _speed(root[ego]), _speed(nodes[-1][ego])
            if v0 is not None and v1 is not None:
                policy.append(abs(v1 - v0))
        for aid, a in root.items():
            speed = _speed(a)
            if aid in (ego, target) or speed is None or speed > STATIC_SPEED_MPS:
                continue
            if aid in nodes[-1]:
                _append(static_disp, _dist(a, nodes[-1][aid]))
    out["l4_chain_length_ratio"] = _mean(ratios)
    out["l4_one_node"] = _mean(ones)
    out["l4_target_present"] = _mean(present)
    out["l4_target_brake_observed"] = _mean(brake)
    out["l4_treated_lateral_sign_vs_reference"] = _mean(lateral)
    out["l4_ego_speed_change"] = _mean(policy)
    out["l4_target_error_d1"], out["l4_target_error_d3"], out["l4_target_error_end"] = _mean(t_d1), _mean(t_d3), _mean(t_end)
    out["l4_ego_error_end"] = _mean(e_end)
    out["l4_control_present"] = _mean(control_present)
    out["l4_control_target_error_end"] = _mean(c_t_end)
    out["l4_control_ego_error_end"] = _mean(c_e_end)
    out["static_actor_displacement"] = _mean(static_disp)
    out["snapshot_completeness"] = _mean(snapshot)
    return out


def cell_diagnostics(split_dir: str, cell_dir: str) -> Dict[str, Dict[str, Optional[float]]]:
    """{scenario directory name: metrics} for every scenario of a cell that has
    prediction files, reading the reference from ``split_dir``."""
    import json
    import os

    out: Dict[str, Dict[str, Optional[float]]] = {}
    for name in sorted(os.listdir(cell_dir)):
        d = os.path.join(cell_dir, name)
        gt_path = os.path.join(split_dir, name, "ground_truth.json")
        if not name.startswith("scenario_") or not os.path.isfile(os.path.join(d, "l3_pre.json")) \
                or not os.path.isfile(gt_path):
            continue
        with open(gt_path, encoding="utf-8") as f:
            gt = json.load(f)
        with open(os.path.join(d, "l3_pre.json"), encoding="utf-8") as f:
            l3 = json.load(f)
        l4_path = os.path.join(d, "l4_pre.json")
        l4 = None
        if os.path.isfile(l4_path):
            with open(l4_path, encoding="utf-8") as f:
                l4 = json.load(f)
        out[name] = scenario_diagnostics(gt, l3, l4)
    return out
