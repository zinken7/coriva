"""
nuScenes ingestion adapter.

Parses the nuScenes JSON tables directly (no nuscenes-devkit dependency) and
turns a single sample (keyframe) into a CORIVA `observed_state` root: the ego at
the origin facing +x, plus the nearest dynamic actors in ego-relative 2D
coordinates with velocities. The root feeds simulation.kinematic_twin.build_twin_gt.

Coordinates: nuScenes stores global positions; we translate to the T0 ego
origin, rotate by -ego_yaw, and flip the lateral axis into CORIVA's fixed
convention: +x is forward, +y is right, and -y is left.  Positions and
velocities at every historical and future step remain in these same T0 axes.
The observed_state timestamp is a small scenario-local value
(default 0.0), NOT the nuScenes epoch microseconds (which would break chain
timestamp monotonicity downstream).

Factual-branch anchoring: `extract_observed_trajectory` follows the real
`sample["next"]` chain to build the *observed* future (the factual branch),
so the model is scored against reality rather than a point-mass rollout. Only the
(unobservable, Pearl rung-3) counterfactual branches stay twin-generated.
`factual_fidelity_error` reports how far the twin drifts from that reality — the
supporting sanity check on the one branch where ground truth exists.
"""

import copy
import json
import math
import os
from typing import Any, Dict, List, Optional

# nuScenes category prefix -> CORIVA ontology entity_type (dynamic actors only).
_CATEGORY_PREFIXES = [
    ("vehicle.bicycle", "bicycle"),
    ("vehicle.motorcycle", "motorcycle"),
    ("vehicle.truck", "truck"),
    ("vehicle.bus", "bus"),
    ("vehicle.emergency", "vehicle"),
    ("vehicle.construction", "truck"),
    ("vehicle.trailer", "truck"),
    ("vehicle", "vehicle"),        # car and the rest
    ("human.pedestrian", "pedestrian"),
    ("movable_object.barrier", "barrier"),
    ("movable_object.trafficcone", "traffic_cone"),
]

_CROSSING_ENTITIES = {"pedestrian", "bicycle"}
_OBSTACLE_ENTITIES = {"barrier", "traffic_cone"}
_BOUNDARY_ENTITIES = {"guardrail", "curb", "road_edge"}
_SIGN_ENTITIES = {"traffic_sign", "traffic_light", "lane_marking", "crosswalk"}

# ── L3 scenario-template selection policy ──────────────────────────────
# `interaction-v3` chooses the scene's action set among root actors that are
# visible at T0 and pass the same role/state relevance rule the reviewed L4
# query uses, and then applies the interaction half of the L4 paired-effect
# gate to each of them: under its published intervention profile and the
# fixed no-action ego policy the actor must come within
# `max_interaction_edge_gap_m` of the ego (or collide) inside the horizon.
# Urgency is the first half-second step at which that happens; the reviewed
# critical actor breaks ties. Category alone never selects a template: a
# pedestrian or cyclist behind the ego, invisible at T0 or unable to reach
# the corridor is not a hazard, and the actor's own motion is part of the
# probe, so a lead pulling away at speed or a pedestrian 100 m down the
# corridor does not count as the most urgent hazard.
SCENARIO_TEMPLATE_POLICY = "interaction-v3"
_PROBE_EGO_POLICY = "do_ego_no_action"
_TEMPLATE_BY_FAMILY = {
    "pedestrian_enters_road": "pedestrian_collision_risk",
    "cyclist_enters_road": "cyclist_collision_risk",
    "npc_sudden_brake": "rear_end_collision_risk",
    "npc_cut_in_left": "cut_in_collision_risk",
    "npc_cut_in_right": "cut_in_collision_risk",
}
# L4 needs an intervention that changes something; L3 only needs a hazard.
# A stopped lead car and a pedestrian already inside the corridor are hazards
# even though "sudden brake" / "enters road" would be no-ops for them.
_L3_RELAXED_REASONS = {"lead_actor_not_moving_forward_fast_enough",
                       "road_entry_actor_already_in_ego_lane_corridor"}
_LEAD_TEMPLATE_ACCEL = -4.0  # the braking-lead profile of the rear-end template


def assign_role(entity_type: str, x: float, y: float, lane_halfwidth: float = 2.0) -> str:
    """Geometry-based role in the ego frame (ego at origin heading +x):
    pedestrians/cyclists cross; static objects map to obstacle / static_barrier /
    road_boundary; a vehicle ahead in-lane leads, ahead off-lane is adjacent,
    behind is rear."""
    if entity_type in _CROSSING_ENTITIES:
        return "crossing_actor"
    if entity_type in _BOUNDARY_ENTITIES:
        return "road_boundary"
    if entity_type in _OBSTACLE_ENTITIES:
        return "obstacle_actor" if (x > 0 and abs(y) < lane_halfwidth) else "static_barrier"
    if entity_type in _SIGN_ENTITIES:
        return "static_barrier"
    if x > 0:
        return "lead_vehicle" if abs(y) < lane_halfwidth else "adjacent_vehicle"
    return "rear_vehicle"

_TABLES = ["scene", "sample", "sample_annotation", "instance", "category",
           "ego_pose", "sample_data", "calibrated_sensor"]


def load_tables(dataroot: str, version: str = "v1.0-mini") -> Dict[str, Dict[str, Any]]:
    """Load the nuScenes JSON tables, each indexed by token."""
    base = os.path.join(dataroot, version)
    tables: Dict[str, Dict[str, Any]] = {}
    for name in _TABLES:
        with open(os.path.join(base, name + ".json"), "r", encoding="utf-8") as f:
            tables[name] = {r["token"]: r for r in json.load(f)}
    return tables


def quaternion_yaw(q: List[float]) -> float:
    """Yaw (rotation about z) from a nuScenes [w, x, y, z] quaternion."""
    w, x, y, z = q
    return math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


# Default body extent (m) for the ego vehicle, which has no annotation box.
# Roughly the nuScenes data-collection car (Renault Zoe): length × width.
EGO_LENGTH_M = 4.0
EGO_WIDTH_M = 1.8


def box_extent_in_ego_frame(size: List[float], box_yaw: float, ego_yaw: float) -> tuple:
    """Project a nuScenes 3D box onto the ego-aligned axes and return its
    axis-aligned extent (length along x, width along y).

    `size` is the nuScenes [width, length, height]; `box_yaw`/`ego_yaw` are global
    yaws. The box is rotated by (box_yaw - ego_yaw) into the ego frame, then we
    take the tight axis-aligned bounding box of that oriented rectangle:
        ext = |half_len·cos φ| + |half_wid·sin φ|   (and the transpose for y).
    This keeps the twin axis-aligned (no per-step yaw tracking) while still
    reflecting that a sideways vehicle is wide and a head-on one is long."""
    half_w, half_l = size[0] / 2.0, size[1] / 2.0
    phi = box_yaw - ego_yaw
    c, s = abs(math.cos(phi)), abs(math.sin(phi))
    length_x = 2.0 * (half_l * c + half_w * s)
    width_y = 2.0 * (half_l * s + half_w * c)
    return round(length_x, 3), round(width_y, 3)


def category_to_entity(name: str) -> Optional[str]:
    for prefix, entity in _CATEGORY_PREFIXES:
        if name.startswith(prefix):
            return entity
    return None


def _branch(branch_id, is_factual, intervention_type):
    return {"branch_id": branch_id, "is_factual": is_factual,
            "intervention_type": intervention_type, "target_actor_id": "ego", "trigger_depth": 1}


def t0_visible_actor_ids(root: Dict[str, Any], visibility_by_instance: Dict[str, Dict[str, bool]],
                         t0_sample_token: str) -> set:
    """Root actor ids whose camera visibility flag is true at the T0 keyframe.
    The ego is always included. Actors without a source instance token (e.g.
    synthetic roots) are treated as not visible."""
    ids = set()
    for actor in root.get("actors", []):
        if actor.get("role_type") == "ego_agent":
            ids.add(str(actor["id"]))
            continue
        per_frame = visibility_by_instance.get(actor.get("source_instance_token")) or {}
        if per_frame.get(t0_sample_token):
            ids.add(str(actor["id"]))
    return ids


def _hazard_family(actor: Dict[str, Any]) -> Optional[str]:
    """The named-intervention family an actor's role/entity would belong to."""
    from simulation.nuscenes_admission import ROAD_ENTRY_ENTITIES, VEHICLE_ENTITIES
    entity = str(actor.get("entity_type", ""))
    role = str(actor.get("role_type", ""))
    if entity in ROAD_ENTRY_ENTITIES:
        return ROAD_ENTRY_ENTITIES[entity]
    if entity in VEHICLE_ENTITIES and role == "lead_vehicle":
        return "npc_sudden_brake"
    if entity in VEHICLE_ENTITIES and role == "adjacent_vehicle":
        return "npc_cut_in_left" if float(actor.get("y", 0.0)) > 0.0 else "npc_cut_in_right"
    return None


def _probe_query(actor: Dict[str, Any], family: str) -> Dict[str, Any]:
    """The canonical L4 query for an actor's hazard family: the published
    profile with the road-entry direction resolved from the actor's T0 side,
    as `resolve_nuscenes_intervention` resolves it for the reviewed target
    (the cut-in families already encode the side)."""
    from simulation.kinematic_twin import ENV_PROFILES
    profile = dict(ENV_PROFILES[family])
    if family in ("pedestrian_enters_road", "cyclist_enters_road"):
        speed = abs(float(profile["lat"]))
        profile["lat"] = -speed if float(actor.get("y", 0.0)) > 0.0 else speed
    return {"intervention_type": family, "target_actor_id": str(actor.get("id")),
            "trigger_depth": 1,
            "resolved_accel_mps2": float(profile["accel"]),
            "resolved_lateral_velocity_mps": float(profile["lat"])}


def _hazard_candidate(actor: Dict[str, Any], root: Dict[str, Any], visible: bool,
                      n_steps: int, dt: float, thresholds, model: str,
                      base_steer: float) -> Dict[str, Any]:
    """Relevance and urgency of one root actor for the L3 template.

    Relevance reuses the L4 role/state rule (`resolve_nuscenes_intervention`)
    minus the two intervention-feasibility conditions in `_L3_RELAXED_REASONS`,
    and then the interaction condition of the L4 paired-effect gate
    (`evaluate_nuscenes_effect`): under the actor's published intervention
    profile and the fixed no-action ego policy, its body-edge gap to the ego
    must fall to `max_interaction_edge_gap_m` or below (a collision included)
    at some step within the horizon. Urgency is the first such step. A lead
    vehicle pulling away from the ego, or a pedestrian 100 m down the
    corridor, is therefore not a hazard whatever its distance. The probe
    decides relevance and order only; the L3
    alternative branches apply the template's own hazard profile to the
    selected actor (`_LEAD_TEMPLATE_ACCEL` for a lead, entry motion for a
    road-entry actor, observed motion for a cut-in candidate)."""
    from simulation.nuscenes_admission import (evaluate_nuscenes_effect,
                                               resolve_nuscenes_intervention)
    horizon = float(n_steps) * float(dt)
    limit = float(thresholds.max_interaction_edge_gap_m)
    out: Dict[str, Any] = {
        "id": str(actor.get("id")), "entity_type": actor.get("entity_type"),
        "role_type": actor.get("role_type"), "visible_at_t0": visible,
        "family": _hazard_family(actor), "relevant": False, "reasons": [],
        "urgency_seconds": None, "urgency_steps": None, "in_corridor": False,
        "probe_intervention_type": None, "probe_lateral_velocity_mps": None,
        "probe_min_edge_gap_m": None, "probe_interaction_step": None,
        "interaction_gap_threshold_m": limit,
        "distance_m": round(math.hypot(float(actor.get("x", 0.0)), float(actor.get("y", 0.0))), 3),
    }
    if not visible:
        out["reasons"] = ["not_visible_at_t0"]
        return out
    if out["family"] is None:
        out["reasons"] = [f"role_not_a_hazard_family:{actor.get('role_type') or 'missing'}"]
        return out
    res = resolve_nuscenes_intervention(actor, horizon_seconds=horizon, thresholds=thresholds)
    out["reasons"] = list(res["reasons"])
    if any(r not in _L3_RELAXED_REASONS for r in res["reasons"]):
        return out
    family = out["family"]
    if family != "npc_sudden_brake":
        edge = (abs(float(actor.get("y", 0.0))) - float(actor.get("width", 0.0)) / 2.0
                - float(thresholds.lane_halfwidth_m))
        out["in_corridor"] = edge <= 0.0
    query = _probe_query(actor, family)
    probe = evaluate_nuscenes_effect(root, query, n_steps=n_steps, dt=dt, model=model,
                                     base_steer=base_steer, thresholds=thresholds)
    gaps = [float(s["treated_ego_target_edge_gap_m"]) for s in probe.get("per_step", [])]
    step = next((i for i, g in enumerate(gaps, start=1) if g <= limit), None)
    out.update(probe_intervention_type=family,
               probe_lateral_velocity_mps=query["resolved_lateral_velocity_mps"],
               probe_min_edge_gap_m=round(min(gaps), 3) if gaps else None,
               probe_interaction_step=step)
    if step is None:
        out["reasons"].append("does_not_interact_with_ego_within_horizon")
        return out
    out.update(relevant=True, urgency_steps=int(step), urgency_seconds=round(step * dt, 3))
    return out


def infer_scenario_spec(scenario_id: str, root: Dict[str, Any], n_steps: int = 6,
                        dt: float = 0.5, *, t0_visible_ids: Optional[set] = None,
                        critical_actor_id: Optional[str] = None,
                        thresholds=None, model: str = "point_mass",
                        base_steer: float = 0.0) -> Dict[str, Any]:
    """Choose the L3 scenario template for a real scene (policy `interaction-v3`).

    Candidates are root actors visible at T0 (`t0_visible_ids`; None means all
    are visible, for roots without instance tokens) that pass the L3 relevance
    rule, including the L4 interaction probe (`_hazard_candidate`; `model` and
    `base_steer` are the builder's twin model and per-scenario steer so the
    probe is the same rollout as the paired-effect gate). The most urgent
    candidate selects the template; candidates with the same urgency step
    count go to the reviewed critical actor, then to the nearer actor. Only
    the selected actor follows its template profile in the L3 alternative
    branches; every other actor continues its observed T0 velocity, exactly
    as in the L4 control. With no relevant dynamic actor, an in-lane obstacle
    gives `obstacle_appearance_risk`, otherwise `generic_following`. The full
    candidate table is returned under `scenario_template` for provenance."""
    from simulation.kinematic_twin import ENV_PROFILES
    from simulation.nuscenes_admission import (DEFAULT_NUSCENES_ADMISSION_THRESHOLDS,
                                               observed_velocity_baseline)
    from simulation.protocol import scenario_actions
    thresholds = thresholds or DEFAULT_NUSCENES_ADMISSION_THRESHOLDS
    horizon = float(n_steps) * float(dt)
    non_ego = [a for a in root["actors"] if a.get("role_type") != "ego_agent"]
    visible_ids = None if t0_visible_ids is None else {str(i) for i in t0_visible_ids}

    candidates = [_hazard_candidate(
        a, root, visible_ids is None or str(a.get("id")) in visible_ids, n_steps, dt,
        thresholds, model, base_steer)
        for a in non_ego]
    relevant = [c for c in candidates if c["relevant"]]
    selected = None
    fallback = None
    if relevant:
        selected = min(relevant, key=lambda c: (
            c["urgency_steps"], 0 if c["id"] == str(critical_actor_id) else 1,
            c["distance_m"], c["id"]))
        stype = _TEMPLATE_BY_FAMILY[selected["family"]]
    else:
        obstacles = [a for a in non_ego if a.get("role_type") == "obstacle_actor"
                     and (visible_ids is None or str(a.get("id")) in visible_ids)]
        if obstacles:
            stype, fallback = "obstacle_appearance_risk", "obstacle_actor_in_lane"
        else:
            stype, fallback = "generic_following", "no_relevant_actor"

    # The lateral escape steers away from the hazard's side (protocol
    # `scenario_actions`); with no selected hazard the table default stands.
    selected_lateral = None
    if selected is not None:
        selected_lateral = next((a.get("y") for a in non_ego
                                 if str(a.get("id")) == str(selected["id"])), None)
    actions = scenario_actions(stype, selected_lateral)
    branches = [_branch(("Factual_" if i == 0 else "CF_") + action, i == 0, action)
                for i, action in enumerate(actions)]

    # Every non-ego actor continues its observed T0 velocity; only the actor
    # that selected the template follows that template's hazard profile.
    env_behavior = observed_velocity_baseline(root)
    if selected is not None:
        actor = next(a for a in non_ego if str(a.get("id")) == selected["id"])
        family = selected["family"]
        if family == "npc_sudden_brake":
            if float(actor.get("vx", 0.0)) >= float(thresholds.min_brake_forward_speed_mps):
                env_behavior[selected["id"]] = {"accel": _LEAD_TEMPLATE_ACCEL, "lat": 0.0}
        elif family in ("pedestrian_enters_road", "cyclist_enters_road") and not selected["in_corridor"]:
            speed = abs(float(ENV_PROFILES[family]["lat"]))
            env_behavior[selected["id"]] = {
                "accel": 0.0, "lat": -speed if float(actor.get("y", 0.0)) >= 0.0 else speed}
        # adjacent vehicles keep their observed motion in the L3 alternatives

    # The hazard manoeuvre the L3 references script for the selected actor
    # (None when the family keeps its observed motion, e.g. a cut-in
    # candidate). Published so the prompt can state it: the model cannot
    # infer from eight frames that a lead brakes at exactly -4 m/s^2 from
    # T=0.
    hazard_profile = None
    if selected is not None and env_behavior.get(selected["id"]) != observed_velocity_baseline(root).get(selected["id"]):
        hazard_profile = dict(env_behavior[selected["id"]])
    return {
        "scenario_id": scenario_id, "scenario_type": stype,
        "n_steps": n_steps, "dt": dt, "sample_interval": dt,
        "env_behavior": env_behavior,
        "branches": branches,
        "scenario_template": {
            "policy": SCENARIO_TEMPLATE_POLICY,
            "selected_actor_id": selected["id"] if selected else None,
            "selected_family": selected["family"] if selected else None,
            "hazard_profile": hazard_profile,
            "urgency_steps": selected["urgency_steps"] if selected else None,
            "urgency_seconds": selected["urgency_seconds"] if selected else None,
            "fallback": fallback,
            "critical_actor_id": critical_actor_id,
            "t0_visibility_applied": visible_ids is not None,
            "horizon_seconds": horizon, "dt": dt,
            "thresholds": thresholds.to_dict(),
            "interaction_probe": {
                "same_test_as": "l4_paired_effect_gate_interaction_condition",
                "ego_policy": _PROBE_EGO_POLICY, "twin_model": model,
                "base_steer": float(base_steer),
                "max_interaction_edge_gap_m": float(thresholds.max_interaction_edge_gap_m),
            },
            "candidates": candidates,
        },
    }


def _ego_pose_by_sample(tables: Dict[str, Dict[str, Any]]) -> Dict[str, str]:
    """Map sample_token -> ego_pose_token using any keyframe sample_data."""
    cache_key = "__coriva_ego_pose_by_sample"
    cached = tables.get(cache_key)
    if isinstance(cached, dict):
        return cached  # type: ignore[return-value]
    out: Dict[str, str] = {}
    for sd in tables["sample_data"].values():
        if sd.get("is_key_frame") and sd["sample_token"] not in out:
            out[sd["sample_token"]] = sd["ego_pose_token"]
    tables[cache_key] = out
    return out


def _annotation_indexes(tables: Dict[str, Dict[str, Any]]) -> tuple:
    """Build reusable annotation indexes once for large trainval workflows."""
    cache_key = "__coriva_annotation_indexes"
    cached = tables.get(cache_key)
    if isinstance(cached, tuple) and len(cached) == 2:
        return cached
    by_sample: Dict[str, List[Dict[str, Any]]] = {}
    by_instance: Dict[str, List[Dict[str, Any]]] = {}
    for ann in tables["sample_annotation"].values():
        by_sample.setdefault(ann["sample_token"], []).append(ann)
        by_instance.setdefault(ann["instance_token"], []).append(ann)
    cached = (by_sample, by_instance)
    tables[cache_key] = cached  # type: ignore[assignment]
    return cached


def annotations_for_sample(tables: Dict[str, Dict[str, Any]],
                           sample_token: str) -> List[Dict[str, Any]]:
    return _annotation_indexes(tables)[0].get(sample_token, [])


def annotations_for_instance(tables: Dict[str, Dict[str, Any]],
                             instance_token: str) -> List[Dict[str, Any]]:
    return _annotation_indexes(tables)[1].get(instance_token, [])


def _rotate(dx: float, dy: float, ang: float) -> tuple:
    c, s = math.cos(ang), math.sin(ang)
    return dx * c - dy * s, dx * s + dy * c


def _global_delta_to_t0_ego(dx: float, dy: float, root_yaw: float) -> tuple:
    """Express a global planar vector in CORIVA's fixed T0 ego axes.

    The ordinary nuScenes ego frame is right-handed in the horizontal plane:
    +x points forward and +y points left.  CORIVA shares CARLA's published
    benchmark convention instead: +x forward and +y right.  Negating only the
    rotated lateral component makes positions and velocity vectors obey the
    same convention without changing metric distances.
    """
    forward, left = _rotate(dx, dy, -root_yaw)
    return forward, -left


def _past_velocity_in_root_frame(previous_xy: tuple, current_xy: tuple,
                                 previous_timestamp: int, current_timestamp: int,
                                 root_yaw: float) -> tuple:
    """Finite-difference velocity for the interval ending at the current state.

    Both root and factual-future actors use this one convention: global
    displacement over the actor's preceding observed interval, rotated by the
    fixed T0 ego yaw. This avoids future leakage at T0 and keeps every emitted
    velocity in the same T0-aligned reference frame as its position.
    """
    dt = (current_timestamp - previous_timestamp) / 1e6
    if dt <= 0:
        return 0.0, 0.0
    gvx = (current_xy[0] - previous_xy[0]) / dt
    gvy = (current_xy[1] - previous_xy[1]) / dt
    return _global_delta_to_t0_ego(gvx, gvy, root_yaw)


def estimate_root_steer(tables: Dict[str, Dict[str, Any]], sample_token: str,
                        speed: float, wheelbase: float = 2.7,
                        min_speed: float = 0.5) -> float:
    """Equivalent steering angle of the ego at the root, from its PAST
    yaw rate only (ego_pose of sample.prev → root; the observed future the
    fidelity gate validates against is never touched, so the estimate cannot
    fit the twin to its own test data). Bicycle relation:
    yaw_rate = v·tan(steer)/L  →  steer = atan(L·yaw_rate / v).
    Returns 0.0 when there is no history or the ego is near-stationary."""
    if speed < min_speed:
        return 0.0
    s = tables["sample"].get(sample_token, {})
    prev = s.get("prev")
    if not prev:
        return 0.0
    pose_by_sample = _ego_pose_by_sample(tables)
    try:
        yaw_now = quaternion_yaw(tables["ego_pose"][pose_by_sample[sample_token]]["rotation"])
        yaw_prev = quaternion_yaw(tables["ego_pose"][pose_by_sample[prev]]["rotation"])
        dt = (s["timestamp"] - tables["sample"][prev]["timestamp"]) / 1e6
    except KeyError:
        return 0.0
    if dt <= 0:
        return 0.0
    dyaw = yaw_now - yaw_prev
    while dyaw > math.pi:
        dyaw -= 2 * math.pi
    while dyaw < -math.pi:
        dyaw += 2 * math.pi
    return math.atan(wheelbase * (dyaw / dt) / speed)


def _extract_root_full(tables: Dict[str, Dict[str, Any]], sample_token: str,
                       max_actors: int, radius: float, timestamp: float,
                       lane_halfwidth: float, priority_instance_token: Optional[str] = None,
                       allowed_instance_tokens: Optional[set[str]] = None):
    """Core root builder. Returns (root, npc_tokens, ego_ref) where npc_tokens maps
    each emitted npc id -> its nuScenes instance_token (so the same physical actors
    can be tracked into future frames) and ego_ref is (ex, ey, yaw, t0) — the root
    ego pose used to place everything in the ego frame."""
    sample = tables["sample"][sample_token]
    ego_pose_by_sample = _ego_pose_by_sample(tables)
    ego_pose = tables["ego_pose"][ego_pose_by_sample[sample_token]]
    ex, ey = ego_pose["translation"][0], ego_pose["translation"][1]
    yaw = quaternion_yaw(ego_pose["rotation"])
    t0 = sample["timestamp"]

    # T0 velocity is past-only: using sample.next here would leak the factual
    # future into the model's observed root state.
    ego_vx = ego_vy = 0.0
    if sample["prev"]:
        prev_sample = tables["sample"][sample["prev"]]
        prev_pose = tables["ego_pose"][ego_pose_by_sample[prev_sample["token"]]]
        ego_vx, ego_vy = _past_velocity_in_root_frame(
            (prev_pose["translation"][0], prev_pose["translation"][1]), (ex, ey),
            prev_sample["timestamp"], t0, yaw)
    ego_speed = round(math.hypot(ego_vx, ego_vy), 3)

    actors: List[Dict[str, Any]] = [{
        "id": "ego", "entity_type": "ego_vehicle", "role_type": "ego_agent",
        "x": 0.0, "y": 0.0, "vx": round(ego_vx, 3), "vy": round(ego_vy, 3),
        "speed": ego_speed,
        "length": EGO_LENGTH_M, "width": EGO_WIDTH_M,
    }]

    # Candidate dynamic actors in this sample (keep the instance_token for tracking).
    cands = []
    for ann in annotations_for_sample(tables, sample_token):
        if allowed_instance_tokens is not None and ann["instance_token"] not in allowed_instance_tokens:
            continue
        inst = tables["instance"][ann["instance_token"]]
        cat = tables["category"][inst["category_token"]]["name"]
        entity = category_to_entity(cat)
        if entity is None:
            continue
        gx, gy = ann["translation"][0], ann["translation"][1]
        rx, ry = _global_delta_to_t0_ego(gx - ex, gy - ey, yaw)
        dist = math.hypot(rx, ry)
        if dist > radius:
            continue
        length, width = box_extent_in_ego_frame(
            ann["size"], quaternion_yaw(ann["rotation"]), yaw)
        # T0 velocity from the preceding annotation of the same instance.
        vx = vy = 0.0
        if ann["prev"]:
            pann = tables["sample_annotation"][ann["prev"]]
            ps = tables["sample"][pann["sample_token"]]
            vx, vy = _past_velocity_in_root_frame(
                (pann["translation"][0], pann["translation"][1]), (gx, gy),
                ps["timestamp"], t0, yaw)
        cands.append((dist, entity, round(rx, 3), round(ry, 3), round(vx, 3),
                      round(vy, 3), length, width, ann["instance_token"]))

    cands.sort(key=lambda c: (0 if priority_instance_token and c[-1] == priority_instance_token else 1, c[0]))
    npc_tokens: Dict[str, str] = {}
    selected = cands[:max_actors]
    if priority_instance_token and not any(c[-1] == priority_instance_token for c in selected):
        critical = next((c for c in cands if c[-1] == priority_instance_token), None)
        if critical is None:
            raise ValueError(f"critical actor {priority_instance_token!r} is not present at T0")
        selected = [critical] + [c for c in cands if c[-1] != priority_instance_token][:max_actors - 1]
    for i, (_dist, entity, rx, ry, vx, vy, length, width, inst_tok) in enumerate(
            selected, start=1):
        npc_id = f"npc_{i}"
        npc_tokens[npc_id] = inst_tok
        actors.append({
            "id": npc_id,
            "entity_type": entity,
            "role_type": assign_role(entity, rx, ry, lane_halfwidth),
            "x": rx, "y": ry, "vx": vx, "vy": vy,
            "speed": round(math.hypot(vx, vy), 3),
            "length": length, "width": width,
            "source_instance_token": inst_tok,
        })

    root = {"depth": 0, "timestamp": timestamp, "event_type": "observed_state", "actors": actors}
    return root, npc_tokens, (ex, ey, yaw, t0)


def extract_observed_state(tables: Dict[str, Dict[str, Any]], sample_token: str,
                           max_actors: int = 5, radius: float = 60.0,
                           timestamp: float = 0.0, lane_halfwidth: float = 2.0,
                           priority_instance_token: Optional[str] = None,
                           allowed_instance_tokens: Optional[set[str]] = None) -> Dict[str, Any]:
    """Build a CORIVA observed_state root from one nuScenes sample."""
    root, _npc_tokens, _ego_ref = _extract_root_full(
        tables, sample_token, max_actors, radius, timestamp, lane_halfwidth,
        priority_instance_token=priority_instance_token,
        allowed_instance_tokens=allowed_instance_tokens)
    return root


# Observed-ego longitudinal accel (m/s^2) -> factual ego intervention label.
# Coarse bands on the MEAN acceleration over the recorded horizon; the label
# names the band, not the published profile value, so the recorded factual
# branch of a nuScenes scene follows the recording, not the profile (the
# prompt states this to the model).
FACTUAL_ACTION_BANDS = (
    ("do_ego_emergency_brake", "mean acceleration <= -6.0 m/s^2"),
    ("do_ego_comfortable_brake", "-6.0 < mean acceleration <= -1.5 m/s^2"),
    ("do_ego_accelerate_pass", "mean acceleration >= +1.5 m/s^2"),
    ("do_ego_no_action", "-1.5 < mean acceleration < +1.5 m/s^2"),
)


def infer_factual_action(ego_speeds: List[float], dt: float) -> str:
    """Label the ego's *observed* factual action from its speed profile, so the
    anchored factual branch carries an intervention consistent with reality
    (not a template guess). Coarse bands on mean longitudinal accel
    (`FACTUAL_ACTION_BANDS`)."""
    if len(ego_speeds) < 2 or dt <= 0:
        return "do_ego_no_action"
    accel = (ego_speeds[-1] - ego_speeds[0]) / ((len(ego_speeds) - 1) * dt)
    if accel <= -6.0:
        return "do_ego_emergency_brake"
    if accel <= -1.5:
        return "do_ego_comfortable_brake"
    if accel >= 1.5:
        return "do_ego_accelerate_pass"
    return "do_ego_no_action"


def extract_observed_trajectory(tables: Dict[str, Dict[str, Any]], sample_token: str,
                                n_steps: int, max_actors: int = 5, radius: float = 60.0,
                                timestamp: float = 0.0, lane_halfwidth: float = 2.0,
                                priority_instance_token: Optional[str] = None,
                                allowed_instance_tokens: Optional[set[str]] = None) -> Dict[str, Any]:
    """Build the root PLUS the real observed future (up to n_steps keyframes ahead),
    expressed in the root ego frame and tracking the same actors by instance_token.
    Returns {root, factual_states, factual_action}. `factual_states` mirrors the
    twin's per-step {actor_id: state} layout so build_twin_branch can consume it
    verbatim as override_states."""
    root, npc_tokens, (ex, ey, yaw, _t0) = _extract_root_full(
        tables, sample_token, max_actors, radius, timestamp, lane_halfwidth,
        priority_instance_token=priority_instance_token,
        allowed_instance_tokens=allowed_instance_tokens)
    identity = {a["id"]: a for a in root["actors"]}
    ego_pose_by_sample = _ego_pose_by_sample(tables)

    # Index each tracked instance's annotations by sample_token for O(1) future lookup.
    ann_by_inst: Dict[str, Dict[str, Any]] = {
        tok: {ann["sample_token"]: ann for ann in annotations_for_instance(tables, tok)}
        for tok in npc_tokens.values()
    }

    factual_states: List[Dict[str, Dict[str, Any]]] = []
    ego_speeds = [float(identity["ego"]["speed"])]
    prev_ex, prev_ey, prev_t = ex, ey, _t0
    # Previous annotation per tracked actor. The root annotation seeds step 1,
    # so root and every future state use the same past-ending difference.
    previous_ann = {
        npc_id: ann_by_inst[inst_tok].get(sample_token)
        for npc_id, inst_tok in npc_tokens.items()
    }
    tok = sample_token
    for _ in range(n_steps):
        tok = tables["sample"][tok]["next"]
        if not tok:
            break
        s = tables["sample"][tok]
        ep = tables["ego_pose"][ego_pose_by_sample[tok]]
        gex, gey = ep["translation"][0], ep["translation"][1]
        dt_real = (s["timestamp"] - prev_t) / 1e6
        # Ego in CORIVA's fixed ROOT frame (+x forward, +y right).
        erx, ery = _global_delta_to_t0_ego(gex - ex, gey - ey, yaw)
        if dt_real > 0:
            evx, evy = _past_velocity_in_root_frame(
                (prev_ex, prev_ey), (gex, gey), prev_t, s["timestamp"], yaw)
        else:
            evx = evy = 0.0
        ego_speed = math.hypot(evx, evy)
        ego_speeds.append(ego_speed)
        world: Dict[str, Dict[str, Any]] = {"ego": {
            "id": "ego", "entity_type": "ego_vehicle", "role_type": "ego_agent",
            "x": round(erx, 3), "y": round(ery, 3),
            "vx": round(evx, 3), "vy": round(evy, 3), "speed": round(ego_speed, 3),
            "length": EGO_LENGTH_M, "width": EGO_WIDTH_M,
        }}
        for npc_id, inst_tok in npc_tokens.items():
            ann = ann_by_inst[inst_tok].get(tok)
            if ann is None:  # actor not annotated in this frame -> despawned/out of scope
                continue
            base = identity[npc_id]
            gx, gy = ann["translation"][0], ann["translation"][1]
            rx, ry = _global_delta_to_t0_ego(gx - ex, gy - ey, yaw)
            pann = previous_ann.get(npc_id)
            if pann is not None:
                ps = tables["sample"][pann["sample_token"]]
                vx, vy = _past_velocity_in_root_frame(
                    (pann["translation"][0], pann["translation"][1]), (gx, gy),
                    ps["timestamp"], s["timestamp"], yaw)
            else:
                vx = vy = 0.0
            previous_ann[npc_id] = ann
            world[npc_id] = {
                "id": npc_id, "entity_type": base["entity_type"], "role_type": base["role_type"],
                "x": round(rx, 3), "y": round(ry, 3),
                "vx": round(vx, 3), "vy": round(vy, 3), "speed": round(math.hypot(vx, vy), 3),
                "length": base["length"], "width": base["width"],
            }
        factual_states.append(world)
        prev_ex, prev_ey, prev_t = gex, gey, s["timestamp"]

    action = infer_factual_action(ego_speeds, dt=0.5)
    return {"root": root, "factual_states": factual_states, "factual_action": action}


def observed_sample_tokens(tables: Dict[str, Dict[str, Any]], sample_token: str,
                           count: int = 8) -> List[str]:
    """Return the consecutive sample-token window ending at T0."""
    tokens = [sample_token]
    scene = tables["sample"][sample_token]["scene_token"]
    tok = sample_token
    for _ in range(count - 1):
        tok = tables["sample"][tok].get("prev")
        if not tok or tables["sample"][tok]["scene_token"] != scene:
            raise ValueError(f"sample {sample_token} has fewer than {count} history samples")
        tokens.append(tok)
    return list(reversed(tokens))


def extract_observed_history(tables: Dict[str, Dict[str, Any]], sample_token: str,
                             root: Dict[str, Any],
                             visibility_by_instance: Dict[str, Dict[str, bool]],
                             count: int = 8) -> Dict[str, Any]:
    """Extract eight past-only states in the T0 ego frame.

    Every emitted actor belongs to the closed root universe and must have an
    annotation at every observed keyframe.  The final record copies the root
    fields verbatim so rounding can never make T0 disagree with the prompt.
    """
    from simulation.protocol import HISTORY_TIMESTAMPS

    tokens = observed_sample_tokens(tables, sample_token, count)
    if count != len(HISTORY_TIMESTAMPS):
        raise ValueError("protocol history count must be eight")
    ego_pose_by_sample = _ego_pose_by_sample(tables)
    root_pose = tables["ego_pose"][ego_pose_by_sample[sample_token]]
    ex, ey = root_pose["translation"][:2]
    yaw = quaternion_yaw(root_pose["rotation"])
    actor_tokens = {a["id"]: a.get("source_instance_token")
                    for a in root["actors"] if a["id"] != "ego"}
    ann_index = {
        (a["instance_token"], a["sample_token"]): a
        for instance in set(actor_tokens.values())
        for a in annotations_for_instance(tables, instance)
    }
    records: List[Dict[str, Any]] = []
    root_identity = {a["id"]: a for a in root["actors"]}
    for index, tok in enumerate(tokens):
        if index == count - 1:
            actors = []
            for actor in root["actors"]:
                item = copy.deepcopy(actor)
                item["visibility"] = False if actor["id"] == "ego" else bool(
                    visibility_by_instance[actor["source_instance_token"]][tok])
                actors.append(item)
            records.append({"timestamp_sec": HISTORY_TIMESTAMPS[index], "actors": actors})
            continue
        sample = tables["sample"][tok]
        pose = tables["ego_pose"][ego_pose_by_sample[tok]]
        gx, gy = pose["translation"][:2]
        erx, ery = _global_delta_to_t0_ego(gx - ex, gy - ey, yaw)
        prev_tok = sample.get("prev")
        if prev_tok:
            prev_pose = tables["ego_pose"][ego_pose_by_sample[prev_tok]]
            evx, evy = _past_velocity_in_root_frame(
                prev_pose["translation"][:2], (gx, gy),
                tables["sample"][prev_tok]["timestamp"], sample["timestamp"], yaw)
        else:
            evx = evy = 0.0
        ego_base = root_identity["ego"]
        actors = [{"id": "ego", "entity_type": ego_base["entity_type"],
                   "role_type": ego_base["role_type"], "x": round(erx, 3),
                   "y": round(ery, 3), "vx": round(evx, 3), "vy": round(evy, 3),
                   "speed": round(math.hypot(evx, evy), 3),
                   "length": ego_base["length"], "width": ego_base["width"],
                   "visibility": False}]
        for actor in root["actors"][1:]:
            instance = actor_tokens[actor["id"]]
            ann = ann_index.get((instance, tok))
            if ann is None:
                raise ValueError(f"root actor {actor['id']} lacks an annotation in history")
            ax, ay = ann["translation"][:2]
            rx, ry = _global_delta_to_t0_ego(ax - ex, ay - ey, yaw)
            if ann.get("prev"):
                prev_ann = tables["sample_annotation"][ann["prev"]]
                prev_sample = tables["sample"][prev_ann["sample_token"]]
                vx, vy = _past_velocity_in_root_frame(
                    prev_ann["translation"][:2], (ax, ay), prev_sample["timestamp"],
                    sample["timestamp"], yaw)
            else:
                vx = vy = 0.0
            actors.append({"id": actor["id"], "entity_type": actor["entity_type"],
                           "role_type": actor["role_type"], "x": round(rx, 3),
                           "y": round(ry, 3), "vx": round(vx, 3), "vy": round(vy, 3),
                           "speed": round(math.hypot(vx, vy), 3),
                           "length": actor["length"], "width": actor["width"],
                           "source_instance_token": instance,
                           "visibility": bool(visibility_by_instance[instance][tok])})
        records.append({"timestamp_sec": HISTORY_TIMESTAMPS[index], "actors": actors})
    return {"schema_version": 1, "coordinate_frame": "t0_ego",
            "records": records}


def factual_fidelity_error(root: Dict[str, Any], observed_states: List[Dict[str, Dict[str, Any]]],
                           action: str, dt: float, model: str = "point_mass",
                           base_steer: float = 0.0) -> float:
    """Mean absolute ego position error (m) between the twin rolled
    under `action` and the real observed states — a sanity check that the twin is
    a faithful forward model on the one branch where reality exists. Only the
    factual branch can be validated this way; counterfactuals are unobservable.
    `model`/`base_steer` must match the configuration that generates the GT,
    so the gate validates the actual instrument."""
    from simulation.kinematic_twin import roll_branch
    n = len(observed_states)
    if n == 0:
        return 0.0
    twin = roll_branch(root["actors"], action, n_steps=n, dt=dt,
                       model=model, base_steer=base_steer)
    total = 0.0
    for tw, ob in zip(twin, observed_states):
        e = ob.get("ego", {})
        total += math.hypot(tw["ego"]["x"] - e.get("x", 0.0), tw["ego"]["y"] - e.get("y", 0.0))
    return round(total / n, 4)
