"""Role/state applicability and canonical-twin effect admission for nuScenes.

The real split cannot observe a counterfactual outcome.  This module therefore
does not claim to validate physical truth.  It performs two narrower checks:

* the named intervention is compatible with the selected actor's T0 role and
  past-derived motion state; and
* under the published deterministic twin, the intervention is non-degenerate
  and brings the selected actor into a measurable interaction with the ego.

The treated and control rollouts share the root state, fixed ego policy and
baseline behaviour of every actor.  The only difference is the named critical
actor intervention.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from typing import Any, Dict, List, Optional

from simulation.kinematic_twin import ENV_PROFILES, roll_branch


VEHICLE_ENTITIES = {"vehicle", "truck", "bus", "motorcycle"}
ROAD_ENTRY_ENTITIES = {
    "pedestrian": "pedestrian_enters_road",
    "bicycle": "cyclist_enters_road",
}


@dataclass(frozen=True)
class NuScenesAdmissionThresholds:
    """Auditable physical thresholds for the reviewed real split.

    The three effect thresholds start at the cross-split CARLA values.  They are
    CLI-configurable so the nuScenes pilot can calibrate and freeze them before
    the final build without changing code.
    """

    min_target_position_effect_m: float = 1.0
    min_target_speed_effect_mps: float = 1.0
    max_interaction_edge_gap_m: float = 5.0
    min_brake_forward_speed_mps: float = 1.0
    min_cut_in_forward_speed_mps: float = 1.0
    lane_halfwidth_m: float = 2.0
    max_entry_moving_away_speed_mps: float = 0.5

    def __post_init__(self) -> None:
        for name, value in asdict(self).items():
            if not math.isfinite(float(value)) or float(value) < 0.0:
                raise ValueError(f"{name} must be a finite non-negative number")

    def to_dict(self) -> Dict[str, float]:
        return {key: float(value) for key, value in asdict(self).items()}


DEFAULT_NUSCENES_ADMISSION_THRESHOLDS = NuScenesAdmissionThresholds()


def _state_summary(actor: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "id": actor.get("id"),
        "entity_type": actor.get("entity_type"),
        "role_type": actor.get("role_type"),
        "x": round(float(actor.get("x", 0.0)), 3),
        "y": round(float(actor.get("y", 0.0)), 3),
        "vx": round(float(actor.get("vx", 0.0)), 3),
        "vy": round(float(actor.get("vy", 0.0)), 3),
        "speed": round(float(actor.get("speed", 0.0)), 3),
        "length": round(float(actor.get("length", 0.0)), 3),
        "width": round(float(actor.get("width", 0.0)), 3),
    }


def resolve_nuscenes_intervention(
    actor: Dict[str, Any],
    requested_intervention: Optional[str] = None,
    *,
    horizon_seconds: float = 3.0,
    thresholds: NuScenesAdmissionThresholds = DEFAULT_NUSCENES_ADMISSION_THRESHOLDS,
) -> Dict[str, Any]:
    """Resolve the one intervention admissible for an actor's role/T0 state.

    T0 velocity is already computed by the adapter from the interval ending at
    T0, so this selection does not read the recorded future.  A supplied review
    intervention is checked against the deterministic result rather than
    trusted as free text.
    """
    entity = str(actor.get("entity_type", ""))
    role = str(actor.get("role_type", ""))
    x = float(actor.get("x", 0.0))
    y = float(actor.get("y", 0.0))
    vx = float(actor.get("vx", 0.0))
    vy = float(actor.get("vy", 0.0))
    width = max(0.0, float(actor.get("width", 0.0)))
    reasons: List[str] = []
    selected: Optional[str] = None
    matched_rule: Optional[str] = None
    resolved_profile: Optional[Dict[str, float]] = None
    derived: Dict[str, Any] = {}

    if entity in VEHICLE_ENTITIES:
        if role == "lead_vehicle":
            matched_rule = "moving_lead_vehicle_sudden_brake"
            if x <= 0.0:
                reasons.append("lead_actor_not_ahead_of_ego")
            if vx < thresholds.min_brake_forward_speed_mps:
                reasons.append("lead_actor_not_moving_forward_fast_enough")
            if not reasons:
                selected = "npc_sudden_brake"
                resolved_profile = dict(ENV_PROFILES[selected])
        elif role == "adjacent_vehicle":
            matched_rule = "moving_adjacent_vehicle_cut_in_toward_ego_lane"
            if x <= 0.0:
                reasons.append("adjacent_actor_not_ahead_of_ego")
            if vx < thresholds.min_cut_in_forward_speed_mps:
                reasons.append("adjacent_actor_not_moving_forward_fast_enough")
            if abs(y) <= thresholds.lane_halfwidth_m:
                reasons.append("adjacent_actor_inside_ego_lane_center_corridor")
            if not reasons:
                # In CORIVA's ego frame, negative y is left and positive y is
                # right.  Always choose the named cut-in that moves toward y=0.
                selected = "npc_cut_in_left" if y > 0.0 else "npc_cut_in_right"
                resolved_profile = dict(ENV_PROFILES[selected])
                derived["lateral_side_at_t0"] = "right" if y > 0.0 else "left"
        else:
            reasons.append(f"vehicle_role_not_intervention_eligible:{role or 'missing'}")
    elif entity in ROAD_ENTRY_ENTITIES:
        matched_rule = f"{entity}_outside_lane_enters_toward_ego_corridor"
        selected = ROAD_ENTRY_ENTITIES[entity]
        lateral_speed = abs(float(ENV_PROFILES[selected]["lat"]))
        lateral_edge_distance = abs(y) - width / 2.0 - thresholds.lane_halfwidth_m
        moving_away_speed = vy if y > 0.0 else -vy
        derived.update({
            "lateral_side_at_t0": "right" if y > 0.0 else "left",
            "lateral_edge_distance_to_ego_corridor_m": round(lateral_edge_distance, 3),
            "lateral_moving_away_speed_mps": round(moving_away_speed, 3),
            "maximum_intervention_lateral_reach_m": round(
                lateral_speed * horizon_seconds, 3),
        })
        if role != "crossing_actor":
            reasons.append(f"road_entry_actor_has_wrong_role:{role or 'missing'}")
        if x <= 0.0:
            reasons.append("road_entry_actor_not_ahead_of_ego")
        if lateral_edge_distance <= 0.0:
            reasons.append("road_entry_actor_already_in_ego_lane_corridor")
        if moving_away_speed > thresholds.max_entry_moving_away_speed_mps:
            reasons.append("road_entry_actor_moving_away_from_ego_lane")
        if lateral_edge_distance > lateral_speed * horizon_seconds:
            reasons.append("road_entry_actor_cannot_reach_ego_corridor_within_horizon")
        if reasons:
            selected = None
        else:
            resolved_profile = dict(ENV_PROFILES[selected])
            resolved_profile["lat"] = -lateral_speed if y > 0.0 else lateral_speed
    else:
        reasons.append(f"unsupported_nuscenes_intervention_actor_type:{entity or 'missing'}")

    if requested_intervention is not None:
        requested_intervention = str(requested_intervention)
        if requested_intervention not in ENV_PROFILES:
            reasons.append(f"unknown_environment_intervention:{requested_intervention}")
        elif selected is not None and requested_intervention != selected:
            reasons.append(
                "intervention_not_admissible_for_role_state:"
                f"expected={selected},got={requested_intervention}")

    passed = selected is not None and not reasons
    return {
        "passed": passed,
        "reasons": reasons,
        "requested_intervention": requested_intervention,
        "selected_intervention": selected,
        "admissible_interventions": [selected] if selected is not None else [],
        "matched_rule": matched_rule,
        "t0_state": _state_summary(actor),
        "derived_state": derived,
        "resolved_motion_profile": resolved_profile,
        "thresholds": thresholds.to_dict(),
        "uses_recorded_future": False,
    }


def observed_velocity_baseline(root: Dict[str, Any]) -> Dict[str, Dict[str, float]]:
    """Continue each non-ego actor's T0 velocity in the paired L4 control."""
    return {
        str(actor["id"]): {
            "accel": 0.0,
            "lat": float(actor.get("vy", 0.0)),
        }
        for actor in root.get("actors", [])
        if actor.get("role_type") != "ego_agent"
    }


def canonical_environment_query(actor_id: str, applicability: Dict[str, Any]) -> Dict[str, Any]:
    """Create the private twin query, including the resolved directional profile."""
    if not applicability.get("passed"):
        raise ValueError("cannot create a query from a failed role/state applicability check")
    profile = applicability.get("resolved_motion_profile") or {}
    return {
        "intervention_type": applicability["selected_intervention"],
        "target_actor_id": str(actor_id),
        "trigger_depth": 1,
        # These two fields are private generator controls.  The released model
        # query retains the ontology label and actor ID only; direction is
        # inferable from the visible/root state and is recorded in admission.
        "resolved_accel_mps2": float(profile.get("accel", 0.0)),
        "resolved_lateral_velocity_mps": float(profile.get("lat", 0.0)),
    }


def _speed(state: Dict[str, Any]) -> float:
    if "vx" in state or "vy" in state:
        return math.hypot(float(state.get("vx", 0.0)), float(state.get("vy", 0.0)))
    return float(state.get("speed", 0.0))


def _edge_gap(ego: Dict[str, Any], target: Dict[str, Any]) -> float:
    """Axis-aligned body-edge gap, matching the canonical twin collision model."""
    ego_length = float(ego.get("length", 4.0))
    ego_width = float(ego.get("width", 1.8))
    target_length = float(target.get("length", 1.5))
    target_width = float(target.get("width", 0.8))
    gap_x = max(
        0.0,
        abs(float(target["x"]) - float(ego["x"])) - (ego_length + target_length) / 2.0,
    )
    gap_y = max(
        0.0,
        abs(float(target["y"]) - float(ego["y"])) - (ego_width + target_width) / 2.0,
    )
    return math.hypot(gap_x, gap_y)


def evaluate_nuscenes_effect(
    root: Dict[str, Any],
    environment_query: Dict[str, Any],
    *,
    n_steps: int = 6,
    dt: float = 0.5,
    model: str = "bicycle",
    base_steer: float = 0.0,
    thresholds: NuScenesAdmissionThresholds = DEFAULT_NUSCENES_ADMISSION_THRESHOLDS,
) -> Dict[str, Any]:
    """Run matched control/treated twin rollouts and measure the L4 effect."""
    target_id = str(environment_query.get("target_actor_id", ""))
    actors = {str(actor.get("id")): actor for actor in root.get("actors", [])}
    ego_id = next(
        (str(actor.get("id")) for actor in root.get("actors", [])
         if actor.get("role_type") == "ego_agent"),
        None,
    )
    reasons: List[str] = []
    if ego_id is None:
        reasons.append("effect_gate_ego_missing")
    if target_id not in actors:
        reasons.append("effect_gate_target_missing")
    if reasons:
        return {
            "passed": False,
            "reasons": reasons,
            "target_actor_id": target_id,
            "thresholds": thresholds.to_dict(),
            "reference_semantics": "canonical_twin_applicability_not_physical_truth",
        }

    baseline = observed_velocity_baseline(root)
    control = roll_branch(
        root["actors"], "do_ego_no_action", n_steps, dt,
        model=model, env_behavior=baseline, base_steer=base_steer)
    treated = roll_branch(
        root["actors"], "do_ego_no_action", n_steps, dt,
        model=model, env_behavior=baseline, base_steer=base_steer,
        environment_intervention=environment_query)

    max_position_effect = 0.0
    max_speed_effect = 0.0
    min_treated_gap = float("inf")
    min_control_gap = float("inf")
    max_gap_reduction = 0.0
    treated_collision_depth: Optional[int] = None
    control_collision_depth: Optional[int] = None
    per_step: List[Dict[str, Any]] = []
    for index, (control_world, treated_world) in enumerate(zip(control, treated), start=1):
        target_c = control_world[target_id]
        target_t = treated_world[target_id]
        ego_c = control_world[ego_id]
        ego_t = treated_world[ego_id]
        position_effect = math.hypot(
            float(target_t["x"]) - float(target_c["x"]),
            float(target_t["y"]) - float(target_c["y"]),
        )
        speed_effect = abs(_speed(target_t) - _speed(target_c))
        treated_gap = _edge_gap(ego_t, target_t)
        control_gap = _edge_gap(ego_c, target_c)
        max_position_effect = max(max_position_effect, position_effect)
        max_speed_effect = max(max_speed_effect, speed_effect)
        min_treated_gap = min(min_treated_gap, treated_gap)
        min_control_gap = min(min_control_gap, control_gap)
        max_gap_reduction = max(max_gap_reduction, control_gap - treated_gap)
        if treated_gap <= 1e-9 and treated_collision_depth is None:
            treated_collision_depth = index
        if control_gap <= 1e-9 and control_collision_depth is None:
            control_collision_depth = index
        per_step.append({
            "depth": index,
            "timestamp_sec": round(index * dt, 3),
            "target_position_effect_m": round(position_effect, 3),
            "target_speed_effect_mps": round(speed_effect, 3),
            "treated_ego_target_edge_gap_m": round(treated_gap, 3),
            "control_ego_target_edge_gap_m": round(control_gap, 3),
        })

    if (max_position_effect < thresholds.min_target_position_effect_m
            and max_speed_effect < thresholds.min_target_speed_effect_mps):
        reasons.append("canonical_intervention_has_no_material_target_effect")
    if (treated_collision_depth is None
            and min_treated_gap > thresholds.max_interaction_edge_gap_m):
        reasons.append("critical_actor_does_not_interact_with_ego_within_3s")

    return {
        "passed": not reasons,
        "reasons": reasons,
        "target_actor_id": target_id,
        "intervention_type": environment_query.get("intervention_type"),
        "fixed_ego_policy": "do_ego_no_action",
        "control": "same_root_and_observed_velocity_baseline_without_named_intervention",
        "max_target_position_effect_m": round(max_position_effect, 3),
        "max_target_speed_effect_mps": round(max_speed_effect, 3),
        "min_treated_ego_target_edge_gap_m": round(min_treated_gap, 3),
        "min_control_ego_target_edge_gap_m": round(min_control_gap, 3),
        "max_ego_target_gap_reduction_m": round(max_gap_reduction, 3),
        "treated_collision_depth": treated_collision_depth,
        "control_collision_depth": control_collision_depth,
        "per_step": per_step,
        "thresholds": thresholds.to_dict(),
        "reference_semantics": "canonical_twin_applicability_not_physical_truth",
    }
