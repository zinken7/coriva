"""
Reference-free L4 pair axes (Option B, docs/l4_pair_protocol.md §4).

Layer 3 varies the ego action and asks whether the model's predicted futures
change consistently across actions (`analysis/causal_coherence.py`). Layer 4
keeps the ego policy fixed, varies the behaviour of ONE named actor, and asks
whether the model's predicted control and treated futures differ consistently
with that intervention. This module measures that on the model's OWN pair:

    predicted control  <->  predicted treated

It never reads a reference chain. Its inputs are the two predicted chains, the
observed root and the PUBLIC query metadata (intervention type, target, the
per-scene profile the prompt prints). Reference agreement (predicted control
<-> reference control, predicted treated <-> reference treated) is the
scorer's separate business (`ctms.score_layer4_pair`).

Three axes, each per query in {0, 1} or None (undefined), never combined:

1. target_responsiveness — does the named target actually differ between the
   two chains (>= 1 m in position or >= 1 m/s in speed at some common depth)?
2. direction_consistency — is the predicted treatment effect in the direction
   the published intervention requires? Longitudinal (sudden brake): treated
   target at least 1 m/s slower than control at the last common depth.
   Lateral: signed target displacement in fixed T0 axes for nuScenes, or
   along a control-target heading normal proxy for CARLA. Steering intensity
   is never multiplied by time to claim an expected CARLA displacement.
   Curves, progress differences and fallback headings limit this proxy.
3. ego_policy_consistency — do both chains declare ego_no_action? Reads
   action_type (including an optional terminal declaration), never infers
   policy from identical trajectories. Missing declarations are undefined;
   ego speed/path checks are separate kinematic diagnostics.

Every axis is computed only for a pair-valid query (both chains pass the
scorer's own reference-free gates, `ctms.layer4_chain_gates`); the per-query
record carries the pair status so coverage is always reported next to the
axis. Malformed numbers are missing, never 0. A malformed chain never raises.

CLI:
    python -m analysis.intervention_pair --scenario-dir <split_dir> \
        --pred-dir <cell_dir> --out <cell_dir>/l4_pair.json
"""
from __future__ import annotations

import argparse
import copy
import glob
import json
import math
import os
import sys
from functools import lru_cache
from typing import Any, Dict, List, Optional, Tuple

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

L4_PAIR_AXES = ("target_responsiveness", "direction_consistency", "ego_policy_consistency")
L4_PAIR_VERSION = "coriva-l4-pair-1.0"

POSITION_TOL_M = 1.0          # admission threshold theta_p
SPEED_TOL_MPS = 1.0           # admission threshold theta_v
EGO_SPEED_STEP_TOL_MPS = 0.5  # theta_e: <= 1 m/s^2 over a 0.5 s step
EGO_HEADING_MIN_SPEED_MPS = 0.5
DT = 0.5

LONGITUDINAL_TYPES = ("npc_sudden_brake",)
LATERAL_TYPES = ("npc_cut_in_left", "npc_cut_in_right", "pedestrian_enters_road", "cyclist_enters_road")

Chain = List[Optional[Dict[str, Any]]]


# ── numeric helpers (malformed -> missing) ────────────────────────────────

def _num(container: Any, key: str) -> Optional[float]:
    if not isinstance(container, dict):
        return None
    v = container.get(key)
    if isinstance(v, bool) or not isinstance(v, (int, float)):
        return None
    return float(v) if math.isfinite(float(v)) else None


def _speed(actor: Any) -> Optional[float]:
    """|(vx, vy)| when both are reported, else the scalar speed (the scorer's rule)."""
    vx, vy = _num(actor, "vx"), _num(actor, "vy")
    if vx is not None and vy is not None:
        return math.hypot(vx, vy)
    return _num(actor, "speed")


def _xy(actor: Any) -> Optional[Tuple[float, float]]:
    x, y = _num(actor, "x"), _num(actor, "y")
    return None if x is None or y is None else (x, y)


def _actors(node: Any) -> Dict[str, Dict[str, Any]]:
    if not isinstance(node, dict):
        return {}
    items = node.get("actors_update")
    if not isinstance(items, list):
        items = node.get("actors")
    if not isinstance(items, list):
        return {}
    return {str(a.get("id")): a for a in items if isinstance(a, dict) and a.get("id") is not None}


@lru_cache(maxsize=1)
def _ontology():
    import yaml
    from pathlib import Path
    return yaml.safe_load((Path(__file__).resolve().parents[1] / 'ctms/ontology.yaml').read_text())


def _canonical_label(value, category):
    if not isinstance(value, str):
        return None
    return (_ontology().get('synonyms', {}).get(category) or {}).get(value, value)


def _event(node: Any) -> Optional[str]:
    if not isinstance(node, dict):
        return None
    e = node.get("event_type")
    return _canonical_label(e, 'event_type')


def future_nodes(chain: Any) -> Chain:
    """The future nodes of a predicted chain with depth slots preserved:
    leading echoed observed_state nodes are dropped (the scorer's root rule),
    non-object nodes stay as None so later nodes keep their depth."""
    if not isinstance(chain, list):
        return []
    nodes: Chain = [n if isinstance(n, dict) else None for n in chain]
    while nodes and nodes[0] is not None and _event(nodes[0]) == "observed_state":
        nodes = nodes[1:]
    return nodes


def terminal_index(nodes: Chain) -> int:
    """m_X: 1-based depth of the first terminal node, else the chain length."""
    for i, n in enumerate(nodes):
        if isinstance(n, dict) and (n.get("terminal") is True
                                    or _event(n) in ("collision", "safe_stop", "out_of_bounds")):
            return i + 1
    return len(nodes)


def _has_terminal(nodes: Chain) -> bool:
    return any(isinstance(n, dict) and (n.get("terminal") is True
                                        or _event(n) in ("collision", "safe_stop", "out_of_bounds"))
               for n in nodes)


def _ego_id(root: Dict[str, Any]) -> Optional[str]:
    for a in (root or {}).get("actors", []) or []:
        if isinstance(a, dict) and a.get("role_type") == "ego_agent":
            return str(a.get("id"))
    return None


def _root_actor(root: Dict[str, Any], actor_id: str) -> Optional[Dict[str, Any]]:
    for a in (root or {}).get("actors", []) or []:
        if isinstance(a, dict) and str(a.get("id")) == str(actor_id):
            return a
    return None


# ── common support ────────────────────────────────────────────────────────

def common_support(control: Chain, treated: Chain, target_id: str) -> List[int]:
    """K: depths 1..min(m_C, m_T) at which the target has a finite position
    and speed in BOTH chains."""
    limit = min(terminal_index(control), terminal_index(treated))
    depths = []
    for k in range(1, limit + 1):
        c, t = _actors(control[k - 1]).get(target_id), _actors(treated[k - 1]).get(target_id)
        if c is None or t is None:
            continue
        if _xy(c) is None or _xy(t) is None or _speed(c) is None or _speed(t) is None:
            continue
        depths.append(k)
    return depths


# ── axes ──────────────────────────────────────────────────────────────────

def target_responsiveness(control: Chain, treated: Chain, target_id: str
                          ) -> Tuple[Optional[float], Dict[str, Any]]:
    depths = common_support(control, treated, target_id)
    if not depths:
        return None, {"reason": "no_common_support", "common_depths": 0}
    max_pos = max_speed = 0.0
    for k in depths:
        c, t = _actors(control[k - 1])[target_id], _actors(treated[k - 1])[target_id]
        (cx, cy), (tx, ty) = _xy(c), _xy(t)
        max_pos = max(max_pos, math.hypot(tx - cx, ty - cy))
        max_speed = max(max_speed, abs(_speed(t) - _speed(c)))
    value = 1.0 if (max_pos >= POSITION_TOL_M or max_speed >= SPEED_TOL_MPS) else 0.0
    return value, {"common_depths": len(depths), "max_target_position_effect_m": round(max_pos, 3),
                   "max_target_speed_effect_mps": round(max_speed, 3)}


def road_normal_at(control: Chain, depth: int, root: Dict[str, Any]) -> Tuple[float, float]:
    """The +y-side road normal at `depth`: the CONTROL ego's velocity (vx, vy)
    rotated a quarter turn towards +y, (-vy, vx)/|v|; falls back to the root
    ego velocity, then to the T0 +y axis. At T0 it is exactly (0, 1), the
    axis the published lateral profile is declared on."""
    ego = _ego_id(root)
    candidates = []
    if ego is not None:
        candidates.append(_actors(control[depth - 1]).get(ego) if depth - 1 < len(control) else None)
        candidates.append(_root_actor(root, ego))
    for actor in candidates:
        vx, vy = _num(actor, "vx"), _num(actor, "vy")
        if vx is None or vy is None:
            continue
        norm = math.hypot(vx, vy)
        if norm >= EGO_HEADING_MIN_SPEED_MPS:
            return (-vy / norm, vx / norm)
    return (0.0, 1.0)


def target_normal_at(control: Chain, depth: int, root: Dict[str, Any], target_id: str):
    """CARLA route-normal proxy, not map truth. Prefer the control target's
    heading; a stationary target falls back to its root heading, then root
    ego heading, then the declared T0 axis. Record the source for review."""
    candidates = [(_actors(control[depth - 1]).get(target_id), 'control_target'),
                  (_root_actor(root, target_id), 'root_target'),
                  (_root_actor(root, _ego_id(root)), 'root_ego')]
    for actor, source in candidates:
        vx, vy = _num(actor, 'vx'), _num(actor, 'vy')
        if vx is not None and vy is not None and math.hypot(vx, vy) >= EGO_HEADING_MIN_SPEED_MPS:
            norm = math.hypot(vx, vy)
            return (-vy / norm, vx / norm), source
    return (0.0, 1.0), 't0_fallback'


def direction_consistency(control: Chain, treated: Chain, target_id: str, intervention_type: Any,
                          profile: Any, root: Dict[str, Any]) -> Tuple[Optional[float], Dict[str, Any]]:
    """Expected direction from the PUBLIC profile only (`profile` = {accel,
    lat} as the prompt prints it); undefined when the profile cannot have
    produced the threshold effect over the common support."""
    depths = common_support(control, treated, target_id)
    if not depths:
        return None, {"reason": "no_common_support", "common_depths": 0}
    k = max(depths)
    t_k = k * DT
    accel = _num(profile, "accel")
    lat = _num(profile, "lat")
    c, t = _actors(control[k - 1])[target_id], _actors(treated[k - 1])[target_id]
    if intervention_type in LONGITUDINAL_TYPES:
        if accel is None or accel >= 0.0:
            return None, {"reason": "no_published_longitudinal_profile", "common_depths": len(depths)}
        expected = abs(accel) * t_k
        if expected < SPEED_TOL_MPS:
            return None, {"reason": "insufficient_common_support_for_published_magnitude",
                          "common_depths": len(depths), "expected_magnitude": round(expected, 3)}
        effect = _speed(t) - _speed(c)
        return (1.0 if effect <= -SPEED_TOL_MPS else 0.0), {
            "family": "longitudinal", "common_depths": len(depths), "last_common_depth": k,
            "speed_effect_mps": round(effect, 3), "expected_magnitude": round(expected, 3)}
    if intervention_type in LATERAL_TYPES:
        if lat is None or lat == 0.0:
            return None, {"reason": "no_published_lateral_profile", "common_depths": len(depths)}
        realization = profile.get('lateral_realization', 'velocity')
        expected = None if realization == 'steering' else abs(lat) * t_k
        if expected is not None and expected < POSITION_TOL_M:
            return None, {"reason": "insufficient_common_support_for_published_magnitude",
                          "common_depths": len(depths), "expected_magnitude": round(expected, 3)}
        frame = profile.get('direction_frame', 't0')
        if frame == 't0':
            (nx, ny), source = (0.0, 1.0), 'declared_t0'
        elif frame == 'control_target_normal':
            (nx, ny), source = target_normal_at(control, k, root, target_id)
        else:
            return None, {'reason': 'unsupported_direction_frame', 'direction_frame': frame}
        (cx, cy), (tx, ty) = _xy(c), _xy(t)
        lateral_effect = nx * (tx - cx) + ny * (ty - cy)  # + = towards +y (right), - = towards -y (left)
        signed = math.copysign(1.0, lat) * lateral_effect
        return (1.0 if signed >= POSITION_TOL_M else 0.0), {
            "family": "lateral", "common_depths": len(depths), "last_common_depth": k,
            "lateral_effect_m": round(lateral_effect, 3), "declared_sign": (1 if lat > 0 else -1),
            "road_normal": [round(nx, 4), round(ny, 4)],
            "direction_frame": frame, "normal_source": source, "lateral_realization": realization,
            "expected_magnitude": round(expected, 3) if expected is not None else None}
    return None, {"reason": "intervention_type_without_direction_rule", "common_depths": len(depths)}


def _non_terminal_depths(nodes: Chain) -> List[int]:
    m = terminal_index(nodes)
    return list(range(1, m)) if _has_terminal(nodes) else list(range(1, m + 1))


def ego_kinematic_diagnostics(control: Chain, treated: Chain, root: Dict[str, Any]
                           ) -> Tuple[Optional[float], Dict[str, Any]]:
    ego = _ego_id(root)
    if ego is None:
        return None, {"reason": "no_ego_in_root"}
    u0 = _speed(_root_actor(root, ego))
    if u0 is None:
        return None, {"reason": "root_ego_speed_missing"}

    def speed_constancy(nodes: Chain) -> Optional[int]:
        depths = _non_terminal_depths(nodes)
        if not depths:
            return None
        prev = u0
        for k in depths:
            a = _actors(nodes[k - 1]).get(ego)
            u = _speed(a)
            if u is None:
                return None
            if abs(u - prev) > EGO_SPEED_STEP_TOL_MPS:
                return 0
            prev = u
        return 1

    s_c, s_t = speed_constancy(control), speed_constancy(treated)
    if s_c is None or s_t is None:
        return None, {"reason": "no_non_terminal_ego_state", "speed_constancy_control": s_c,
                      "speed_constancy_treated": s_t}
    common = sorted(set(_non_terminal_depths(control)) & set(_non_terminal_depths(treated)))
    agreement = 1
    max_gap = 0.0
    for k in common:
        c, t = _actors(control[k - 1]).get(ego), _actors(treated[k - 1]).get(ego)
        pc, pt = _xy(c), _xy(t)
        if pc is None or pt is None:
            return None, {"reason": "ego_position_missing", "depth": k}
        gap = math.hypot(pt[0] - pc[0], pt[1] - pc[1])
        max_gap = max(max_gap, gap)
        if gap > POSITION_TOL_M:
            agreement = 0
    value = float(s_c * s_t * agreement)
    return value, {"speed_constancy_control": s_c, "speed_constancy_treated": s_t,
                   "pair_path_agreement": agreement, "common_non_terminal_depths": len(common),
                   "max_ego_path_gap_m": round(max_gap, 3)}


def ego_policy_consistency(control: Chain, treated: Chain, root: Dict[str, Any]
                           ) -> Tuple[Optional[float], Dict[str, Any]]:
    """Compliance of declared ego actions, never an inference from motion.

    Every non-terminal node must declare the fixed action. A terminal may
    omit action_type; if supplied it is checked too. At least one usable
    action is needed in each condition. Known deviations return 0; otherwise
    missing required declarations leave the measurement undefined.
    """
    if _ego_id(root) is None:
        return None, {'reason': 'no_ego_in_root'}
    known = {label for family in _ontology()['action_type'].values() for label in family}
    counts = {}; missing = []; violations = []
    for name, nodes in (('control', control), ('treated', treated)):
        counts[name] = 0
        required = set(_non_terminal_depths(nodes))
        for k in range(1, terminal_index(nodes) + 1):
            node = nodes[k - 1]
            action = _canonical_label(node.get('action_type'), 'action_type') if isinstance(node, dict) else None
            if action is None and k not in required:
                continue
            if action not in known:
                missing.append({'condition': name, 'depth': k})
                continue
            counts[name] += 1
            if action != 'ego_no_action':
                violations.append({'condition': name, 'depth': k, 'action_type': action})
    detail = {'declarations_control': counts['control'], 'declarations_treated': counts['treated'],
              'missing_declarations': missing, 'policy_violations': violations,
              'measures': 'declared_action_only'}
    if violations:
        return 0.0, detail
    if missing or not all(counts.values()):
        detail['reason'] = 'insufficient_policy_declarations'
        return None, detail
    return 1.0, detail


def pair_axes(control_chain: Any, treated_chain: Any, root: Dict[str, Any],
              intervention: Dict[str, Any], profile: Any) -> Dict[str, Any]:
    """The three axes for one pair, from raw predicted chains (already known to
    be pair-valid by the caller). Never raises on a malformed chain."""
    control, treated = future_nodes(control_chain), future_nodes(treated_chain)
    target_id = str((intervention or {}).get("target_actor_id"))
    itype = (intervention or {}).get("intervention_type")
    out: Dict[str, Any] = {'details': {}}
    calculations = {
        'target_responsiveness': lambda: target_responsiveness(control, treated, target_id),
        'direction_consistency': lambda: direction_consistency(control, treated, target_id, itype, profile, root),
        'ego_policy_consistency': lambda: ego_policy_consistency(control, treated, root),
        'ego_kinematics': lambda: ego_kinematic_diagnostics(control, treated, root),
    }
    for name, calculate in calculations.items():
        try:
            value, details = calculate()
        except (TypeError, ValueError, OverflowError, KeyError, IndexError, AttributeError) as exc:
            value, details = None, {'reason': f'malformed_pair ({type(exc).__name__}: {exc})'}
        if name in L4_PAIR_AXES:
            out[name] = value
        out['details'][name] = details
    return out


# ── per-scenario record with pair validity ────────────────────────────────

def _undefined_record(reason: str, **extra) -> Dict[str, Any]:
    rec = {axis: None for axis in L4_PAIR_AXES}
    rec.update({"defined": False, "reason": reason, "details": {},
                "axis_status": {axis: {"defined": False, "reason": reason} for axis in L4_PAIR_AXES}})
    rec.update(extra)
    return rec


def scenario_pair_record(l4_pre: Any, root: Dict[str, Any], query: Dict[str, Any], env: Dict[str, Any],
                         gate_config: Dict[str, Any], env_behavior: Any = None,
                         intervention_profiles: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """One record per scenario for its (single) L4 query: pair status from the
    scorer's reference-free gates, then the axes when the pair is valid.
    Reads: the predicted L4 file, the public root, the released query and
    (CARLA) the scene configuration's declared parameters. Never a reference
    chain."""
    from ctms.ctms import control_intervention_for, layer4_chain_gates, split_l4_prediction_item
    from simulation.build_pilot import declared_intervention_profile

    queries = [q for q in (query.get("l4_interventions") or []) if isinstance(q, dict)]
    if len(queries) != 1:
        return _undefined_record("query_count_not_one", n_queries=len(queries))
    q = queries[0]
    itype = q.get("intervention_type")
    items = (l4_pre or {}).get("predicted_counterfactuals") if isinstance(l4_pre, dict) else None
    items = items if isinstance(items, list) else []
    matches = []
    for item in items:
        if not isinstance(item, dict) or not isinstance(item.get("intervention"), dict):
            continue
        if item["intervention"].get("intervention_type") == itype:
            matches.append(item)
    pair = {"treated_present": False, "control_present": False, "pair_complete": False,
            "treated_valid": False, "control_valid": False, "pair_valid": False}
    if not matches:
        return _undefined_record("no_matching_prediction", intervention_type=itype, pair=pair)
    if len(matches) > 1:
        return _undefined_record("multiple_chains_for_intervention", intervention_type=itype, pair=pair)
    parts = split_l4_prediction_item(matches[0])
    pair["treated_present"], pair["control_present"] = parts["treated_present"], parts["control_present"]
    pair["pair_complete"] = parts["treated_present"] and parts["control_present"]
    public_tuple = {"intervention_type": itype, "target_actor_id": q.get("target_actor_id"),
                    "trigger_depth": q.get("trigger_depth", 1)}
    treated_wrapper = {"counterfactual_chain": {"intervention": parts["intervention"], "chain": parts["treated_chain"]}}
    control_wrapper = {"counterfactual_chain": {"intervention": control_intervention_for(parts["intervention"]),
                                                "chain": parts["control_chain"]}}
    treated_gate = layer4_chain_gates(public_tuple, treated_wrapper, env["ontology"], gate_config,
                                      env["counterfactual_schema"], root) if parts["treated_present"] else None
    control_gate = layer4_chain_gates(control_intervention_for(public_tuple), control_wrapper, env["ontology"],
                                      gate_config, env["counterfactual_schema"], root) if parts["control_present"] else None
    pair["treated_valid"] = bool(treated_gate and treated_gate["valid"])
    pair["control_valid"] = bool(control_gate and control_gate["valid"])
    pair["pair_valid"] = pair["treated_valid"] and pair["control_valid"]
    if not pair["pair_valid"]:
        reason = ("pair_incomplete" if not pair["pair_complete"] else
                  "both_chains_invalid" if not (pair["treated_valid"] or pair["control_valid"]) else
                  "control_chain_invalid" if not pair["control_valid"] else "treated_chain_invalid")
        _, policy_detail = ego_policy_consistency(future_nodes(parts['control_chain']),
                                                  future_nodes(parts['treated_chain']), root)
        return _undefined_record(reason, intervention_type=itype, pair=pair,
                                 details={'ego_policy_consistency': policy_detail,
                                          'control_gates': {k: control_gate.get(k) for k in ('valid','gates','diagnostics')} if control_gate else None,
                                          'treated_gates': {k: treated_gate.get(k) for k in ('valid','gates','diagnostics')} if treated_gate else None})
    profile = ((intervention_profiles or {}).get(itype)
               or declared_intervention_profile({"observed_state": root}, dict(q), env_behavior))
    axes = pair_axes(parts["control_chain"], parts["treated_chain"], root, public_tuple, profile)
    axes['axis_status'] = {
        axis: {'defined': axes[axis] is not None,
               'reason': ((axes.get('details', {}).get(axis) or {}).get('reason')
                          or axes.get('details', {}).get('reason') or 'insufficient_observations')
               if axes[axis] is None else None}
        for axis in L4_PAIR_AXES}
    defined = all(axes[axis] is not None for axis in L4_PAIR_AXES)
    axes.update({"defined": defined, "reason": None if defined else 'one_or_more_axes_undefined',
                 "intervention_type": itype, "pair": pair,
                 "profile": dict(profile)})
    return axes


def run_l4_pair(gt_dir: str, pred_dir: str, env: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Per-scenario pair axes over a prediction directory. Reads only the
    public per-scene inputs (observed_state.json, query.json,
    input_manifest.json, scenario_config.json) and l4_pre.json; it never
    opens ground_truth.json."""
    from ctms.ctms import apply_reference_gate_policy, load_ctms_environment
    if env is None:
        here = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "ctms")
        env = load_ctms_environment(*(os.path.join(here, f) for f in (
            "ontology.yaml", "ctms_config.yaml", "schema_branch.json", "schema_counterfactual.json")))
    rows: List[Dict[str, Any]] = []
    for scenario_dir in sorted(glob.glob(os.path.join(gt_dir, "scenario_*"))):
        if not os.path.isdir(scenario_dir):
            continue
        name = os.path.basename(scenario_dir)
        l4_path = os.path.join(pred_dir, name, "l4_pre.json")
        if not os.path.exists(l4_path):
            continue
        state_path = os.path.join(scenario_dir, "observed_state.json")
        query_path = os.path.join(scenario_dir, "query.json")
        if not (os.path.isfile(state_path) and os.path.isfile(query_path)):
            raise ValueError(f"{name}: the pair axes need observed_state.json and query.json (public inputs)")
        root = json.load(open(state_path))
        query = json.load(open(query_path))
        manifest_path = os.path.join(scenario_dir, "input_manifest.json")
        manifest = json.load(open(manifest_path)) if os.path.isfile(manifest_path) else {}
        gate_config = apply_reference_gate_policy(env["config"], manifest)
        from simulation.build_pilot import public_intervention_profiles
        scenario_config = {}
        config_path = os.path.join(scenario_dir, "scenario_config.json")
        if os.path.isfile(config_path):
            scenario_config = json.load(open(config_path)) or {}
        try:
            l4_pre = json.load(open(l4_path))
        except (ValueError, OSError):
            l4_pre = None
        rec = scenario_pair_record(l4_pre, root, query, env, gate_config,
                                   intervention_profiles=public_intervention_profiles(
                                       root, query, scenario_config, manifest.get('split') or manifest.get('dataset')))
        rec["scenario_id"] = name
        rows.append(rec)

    def _mean_n(key):
        vals = [r[key] for r in rows if r.get(key) is not None]
        return {"mean": (sum(vals) / len(vals)) if vals else None, "n": len(vals), 'n_pairs': len(rows),
                'undefined_reasons': _count((r.get('axis_status', {}).get(key) or {}).get('reason')
                                            for r in rows if r.get(key) is None)}

    n_pairs = len(rows)
    return {"measurement": "reference_free_l4_pair_axes",
            "l4_pair_version": L4_PAIR_VERSION,
            "measurement_role": "primary",
            "uses_reference_answers": False,
            "n_scenarios": n_pairs,
            "coverage": {
                "pair_complete": sum(1 for r in rows if (r.get("pair") or {}).get("pair_complete")),
                "pair_valid": sum(1 for r in rows if (r.get("pair") or {}).get("pair_valid")),
                "treated_valid": sum(1 for r in rows if (r.get("pair") or {}).get("treated_valid")),
                "control_valid": sum(1 for r in rows if (r.get("pair") or {}).get("control_valid")),
                "undefined_reasons": _count(r.get("reason") for r in rows if r.get("reason")),
            },
            "axes": {k: _mean_n(k) for k in L4_PAIR_AXES},
            "per_scenario": rows}


def _count(values) -> Dict[str, int]:
    out: Dict[str, int] = {}
    for v in values:
        out[str(v)] = out.get(str(v), 0) + 1
    return out


def check_version(report: Dict[str, Any]) -> None:
    version = report.get("l4_pair_version")
    if version != L4_PAIR_VERSION:
        raise ValueError(f"l4 pair report version {version!r} is not {L4_PAIR_VERSION!r}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--scenario-dir", required=True)
    ap.add_argument("--pred-dir", required=True)
    ap.add_argument("--out")
    args = ap.parse_args()
    report = run_l4_pair(args.scenario_dir, args.pred_dir)
    text = json.dumps(report, indent=2)
    if args.out:
        with open(args.out, "w", encoding="utf-8") as f:
            f.write(text)
    else:
        print(text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
