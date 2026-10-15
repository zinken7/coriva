"""Deterministic admission measurements for protocol-1.0 CARLA scenarios."""

from __future__ import annotations

import itertools
import math
from typing import Any, Dict, List, Tuple

from simulation.build_ground_truth import ReferenceBuildError, scored_collision


MIN_ROOT_ACTORS = 5
MIN_BBOX_AREA_PX2 = 100.0
MIN_EFFECT_POSITION_M = 1.0
MIN_EFFECT_SPEED_MPS = 1.0
MAX_INTERACTION_EDGE_GAP_M = 5.0
MIN_L3_POSITION_DELTA_M = 1.0
MIN_L3_SPEED_DELTA_MPS = 1.0


def _closest(steps: List[Dict[str, Any]], timestamp: float) -> Dict[str, Any] | None:
    if not steps:
        return None
    step = min(steps, key=lambda item: abs(float(item.get("time_sec", 0.0)) - timestamp))
    return step if abs(float(step.get("time_sec", 0.0)) - timestamp) <= 0.06 else None


def _speed(state: Dict[str, Any]) -> float:
    if "vx" in state or "vy" in state:
        return math.hypot(float(state.get("vx", 0.0)), float(state.get("vy", 0.0)))
    return float(state.get("speed_kmh", 0.0)) / 3.6


def _actor_key(config: Dict[str, Any], canonical_id: str) -> str | None:
    return next((key for key, spec in config.get("actor_defs", {}).items()
                 if str(spec.get("canonical_id")) == str(canonical_id)), None)


def _bbox_area(state: Dict[str, Any]) -> float:
    bbox = state.get("bbox_2d")
    if not isinstance(bbox, (list, tuple)) or len(bbox) != 4:
        return 0.0
    x1, y1, x2, y2 = (float(value) for value in bbox)
    return max(0.0, x2 - x1) * max(0.0, y2 - y1)


def _extent(config: Dict[str, Any], key: str) -> Tuple[float, float]:
    spec = config["actor_defs"][key]
    defaults = {
        "ego_vehicle": (4.5, 2.0), "vehicle": (4.5, 2.0),
        "parked_vehicle": (4.5, 2.0), "truck": (7.0, 2.5),
        "bus": (12.0, 2.5), "motorcycle": (2.2, 0.8),
        "bicycle": (1.8, 0.6), "pedestrian": (0.6, 0.6),
    }
    length, width = defaults.get(spec.get("entity_type"), (1.5, 0.8))
    return float(spec.get("length", length)), float(spec.get("width", width))


def _edge_gap(ego: Dict[str, Any], target: Dict[str, Any],
              ego_extent: Tuple[float, float], target_extent: Tuple[float, float]) -> float:
    yaw = math.radians(float(ego.get("heading_deg", 0.0)))
    c, s = math.cos(yaw), math.sin(yaw)
    dx = float(target["x"]) - float(ego["x"])
    dy = float(target["y"]) - float(ego["y"])
    longitudinal = abs(c * dx + s * dy)
    lateral = abs(-s * dx + c * dy)
    gap_x = max(0.0, longitudinal - (ego_extent[0] + target_extent[0]) / 2.0)
    gap_y = max(0.0, lateral - (ego_extent[1] + target_extent[1]) / 2.0)
    return math.hypot(gap_x, gap_y)


def evaluate_carla_admission(raw: Dict[str, Any], config: Dict[str, Any],
                             gt: Dict[str, Any]) -> Tuple[bool, Dict[str, Any]]:
    """Return pass/fail plus auditable measurements for one generated slot."""
    reasons: List[str] = []
    t0 = float(config["intervention_time_sec"])
    critical_id = str(config["critical_actor_id"])
    critical_key = _actor_key(config, critical_id)
    ego_key = _actor_key(config, "ego")
    factual_name = next((name for name, meta in config.get("branches", {}).items()
                         if meta.get("is_factual")), None)
    factual_steps = (raw.get("branches", {}).get(factual_name, [])
                     if factual_name else [])

    root_count = len((gt.get("observed_state") or {}).get("actors", []))
    if root_count < MIN_ROOT_ACTORS:
        reasons.append(f"root_actor_count={root_count}<{MIN_ROOT_ACTORS}")

    bbox_areas = []
    if critical_key is None:
        reasons.append("critical_actor_key_missing")
    else:
        for offset in (-3.5, -3.0, -2.5, -2.0, -1.5, -1.0, -0.5, 0.0):
            step = _closest(factual_steps, t0 + offset)
            bbox_areas.append(_bbox_area((step or {}).get(critical_key, {})))
        if len(bbox_areas) != 8 or min(bbox_areas, default=0.0) < MIN_BBOX_AREA_PX2:
            reasons.append("critical_actor_projected_bbox_not_visible_all_8")

    intervention_steps = raw.get("_l4_steps") or []
    control_steps = raw.get("_l4_control_steps") or []
    max_position_effect = max_speed_effect = 0.0
    min_edge_gap = float("inf")
    # "Collision within 3 s" means the ego hits the critical actor inside the
    # scored horizon. A collision with another actor, or after the horizon,
    # does not show that the L4 target interacts with the ego.
    collision = False
    collision_time = None
    collision_partners: List[str] = []
    try:
        hit = scored_collision(intervention_steps, config.get("actor_defs", {}),
                               t0 + 3.0)
    except ReferenceBuildError:
        hit = None
    if hit is not None:
        collision_time, partner_keys = hit
        collision_partners = [str(config["actor_defs"][key].get("canonical_id", key))
                              for key in partner_keys]
        collision = critical_key in partner_keys
    if not intervention_steps or not control_steps or critical_key is None or ego_key is None:
        reasons.append("paired_l4_control_missing")
    else:
        for depth in range(1, 7):
            timestamp = t0 + depth * 0.5
            treated = _closest(intervention_steps, timestamp)
            control = _closest(control_steps, timestamp)
            if treated and control and critical_key in treated and critical_key in control:
                target_t, target_c = treated[critical_key], control[critical_key]
                max_position_effect = max(
                    max_position_effect,
                    math.hypot(float(target_t["x"]) - float(target_c["x"]),
                               float(target_t["y"]) - float(target_c["y"])))
                max_speed_effect = max(max_speed_effect, abs(_speed(target_t) - _speed(target_c)))
            if treated and ego_key in treated and critical_key in treated:
                min_edge_gap = min(
                    min_edge_gap,
                    _edge_gap(treated[ego_key], treated[critical_key],
                              _extent(config, ego_key), _extent(config, critical_key)))
        if (max_position_effect < MIN_EFFECT_POSITION_M
                and max_speed_effect < MIN_EFFECT_SPEED_MPS):
            reasons.append("l4_intervention_has_no_material_target_effect")
        if not collision and min_edge_gap > MAX_INTERACTION_EDGE_GAP_M:
            reasons.append("critical_actor_does_not_interact_with_ego_within_3s")

    branch_samples: Dict[str, List[Dict[str, Any]]] = {}
    for name, steps in raw.get("branches", {}).items():
        samples = []
        for depth in range(1, 7):
            step = _closest(steps, t0 + depth * 0.5)
            if step and ego_key in step:
                samples.append(step[ego_key])
        branch_samples[name] = samples
    indistinct_pairs = []
    for left, right in itertools.combinations(sorted(branch_samples), 2):
        max_pos = max_speed = 0.0
        for a, b in zip(branch_samples[left], branch_samples[right]):
            max_pos = max(max_pos, math.hypot(float(a["x"]) - float(b["x"]),
                                              float(a["y"]) - float(b["y"])))
            max_speed = max(max_speed, abs(_speed(a) - _speed(b)))
        if max_pos < MIN_L3_POSITION_DELTA_M and max_speed < MIN_L3_SPEED_DELTA_MPS:
            indistinct_pairs.append([left, right])
    if len(branch_samples) != 3 or indistinct_pairs:
        reasons.append("l3_branches_not_physically_distinct")

    metrics = {
        "passed": not reasons,
        "reasons": reasons,
        "root_actor_count": root_count,
        "critical_bbox_area_px2": [round(value, 1) for value in bbox_areas],
        "max_l4_target_position_effect_m": round(max_position_effect, 3),
        "max_l4_target_speed_effect_mps": round(max_speed_effect, 3),
        "min_ego_critical_edge_gap_m": (
            None if not math.isfinite(min_edge_gap) else round(min_edge_gap, 3)),
        "collision": collision,
        "collision_time_sec": collision_time,
        "collision_partners": collision_partners,
        "indistinct_l3_pairs": indistinct_pairs,
        "thresholds": {
            "min_root_actors": MIN_ROOT_ACTORS,
            "min_bbox_area_px2": MIN_BBOX_AREA_PX2,
            "min_effect_position_m": MIN_EFFECT_POSITION_M,
            "min_effect_speed_mps": MIN_EFFECT_SPEED_MPS,
            "max_interaction_edge_gap_m": MAX_INTERACTION_EDGE_GAP_M,
        },
    }
    return not reasons, metrics
