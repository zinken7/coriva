"""
Config-driven CARLA scenario runner.

One generic `run_scenario(config, world, api)` spawns the full cast from a
unified scenario config
(`scenarios/*.yaml`, single source of truth shared with build_ground_truth),
applies per-actor `env_behavior` at the environment-event trigger, applies
the branch's ego intervention at `intervention_time_sec`, records telemetry
per branch, and returns the `coriva_bench_examples.json` scenario shape.

CARLA is imported lazily (`_default_api`), so this module — and its unit
tests, which inject fake wrapper classes — work on machines without CARLA.
The real run happens on the server via `run_batch.py`.
"""

import json
import copy
import math
import os
import shutil
import time
from typing import Any, Dict, List, Optional, Tuple

from simulation.protocol import (CARLA_ROUTE_HALF_WIDTH_M,
                                 EGO_SWERVE_SPEED_MPS)
from simulation.build_ground_truth import _extent as _actor_extent

FPS = 10
STEP_LENGTH = 0.1
MAX_SPAWN_WAYPOINT_ATTEMPTS = 24
_CONTEXT_LATERAL_MAGNITUDES = (10.5, 8.5, 7.0, 5.5, 4.25, 3.5)
_CARLA_MAX_LATERAL_SPEED_MPS = 3.5
_SWERVE_INTENSITY = EGO_SWERVE_SPEED_MPS / _CARLA_MAX_LATERAL_SPEED_MPS
_MIN_VISIBLE_BBOX_AREA_PX2 = 100.0
_ROUTE_COVERAGE_MARGIN_M = 20.0


class ScenarioCandidateExhausted(RuntimeError):
    """All deterministic placement/preflight candidates rejected this sample."""


class PlacementRejected(RuntimeError):
    """A cast slot resolved onto ground that cannot host the actor: a moving
    vehicle whose lateral slot is not a same-direction driving lane (it would
    drive on the pavement or against traffic), or a parked vehicle with no
    same-side slot clear of every moving actor's lane (it would be driven
    through, because NPC physics are off). Treated like a spawn collision:
    the next deterministic waypoint candidate is tried."""


# A driving-lane waypoint projected farther than half a lane plus this
# tolerance from the requested point means the point is off the driving lanes
# (pavement, shoulder, building) in CARLA's map.
LANE_PROJECTION_TOLERANCE_M = 0.3
# Extra clearance between a parked vehicle and the lane centre of any moving
# actor on the same side, beyond the sum of their half-widths.
PARKED_CLEARANCE_MARGIN_M = 0.5


# intervention_type → (EgoActions method, kwargs). Magnitudes mirror the
# kinematic twin's EGO_PROFILES so the two splits share one canonical
# dynamics vocabulary (tests assert this).
INTERVENTION_TO_EGO_ACTION: Dict[str, Tuple[str, Dict[str, Any]]] = {
    "do_ego_emergency_brake":        ("brake", {"deceleration": 8.0}),
    "do_ego_comfortable_brake":      ("brake", {"deceleration": 3.0}),
    "do_ego_panic_brake":            ("brake", {"deceleration": 9.0}),
    "do_ego_decelerate_coast":       ("brake", {"deceleration": 1.0}),
    "do_ego_no_action":              ("maintain_speed", {}),
    "do_ego_accelerate_pass":        ("accelerate", {"acceleration": 3.0}),
    "do_ego_accelerate_escape":      ("accelerate", {"acceleration": 3.5}),
    "do_ego_swerve_left":            ("steer_left", {"intensity": _SWERVE_INTENSITY}),
    "do_ego_swerve_right":           ("steer_right", {"intensity": _SWERVE_INTENSITY}),
    "do_ego_brake_and_swerve_left":  ("brake_and_steer_left", {"deceleration": 4.0, "intensity": _SWERVE_INTENSITY}),
    "do_ego_brake_and_swerve_right": ("brake_and_steer_right", {"deceleration": 4.0, "intensity": _SWERVE_INTENSITY}),
}

_VEHICLE_FAMILY = {"vehicle", "car", "truck", "van", "bus", "motorcycle", "bicycle"}
_STATIC_PROP_TYPE = {  # entity_type → StaticProp.PROP_TYPES key (model overrides anyway)
    "guardrail": "street_barrier",
    "road_boundary": "street_barrier",
    "barrier": "street_barrier",
    "traffic_cone": "traffic_cone",
    "traffic_sign": "street_barrier",
}


def _default_api() -> Dict[str, Any]:
    """The real CARLA wrapper classes. Imported lazily so the module (and its
    mocked unit tests) load without a CARLA installation."""
    # The official ``carla`` wheel exposes Client/Actor/etc.  CORIVA's World,
    # actor and recorder wrappers live in ``simulation.carla`` and themselves
    # import that official client.  Importing wrappers from ``carla`` fails on
    # Vast because the official wheel correctly has no Ego/NPC wrapper names.
    from simulation.carla import (World, Ego, EgoActions, NPC, NPCActions,
                                  Pedestrian, PedestrianActions, StaticProp,
                                  TrajectoryRecorder, CollisionDetector)
    return {"World": World, "Ego": Ego, "EgoActions": EgoActions, "NPC": NPC,
            "NPCActions": NPCActions, "Pedestrian": Pedestrian,
            "PedestrianActions": PedestrianActions, "StaticProp": StaticProp,
            "TrajectoryRecorder": TrajectoryRecorder,
            "CollisionDetector": CollisionDetector}


def _context_lateral_candidates(lateral: float) -> List[float]:
    """Same-side deterministic fallbacks for non-causal scenery actors."""
    lateral = float(lateral)
    if lateral == 0.0:
        return [lateral]
    sign = 1.0 if lateral > 0.0 else -1.0
    magnitudes = [abs(lateral)]
    magnitudes.extend(value for value in _CONTEXT_LATERAL_MAGNITUDES
                      if value < abs(lateral))
    # Preserve order while removing the configured value if it is also a
    # standard fallback magnitude.
    unique = list(dict.fromkeys(magnitudes))
    return [sign * value for value in unique]


def _lane_class(world: Any, location: Any, reference_transform: Any) -> Optional[str]:
    """Classify the ground under ``location`` with the CARLA map:
    ``driving_same`` (a driving lane heading the way of ``reference_transform``),
    ``driving_opposite``, ``off_driving`` (no waypoint, or the nearest
    driving-lane waypoint is farther than half a lane away: pavement,
    shoulder, building). Returns None only for fakes whose locations or
    waypoints carry no lane geometry; a failing map query is an error and
    propagates, it is never turned into "unknown"."""
    try:
        lx, ly = float(location.x), float(location.y)
        lz = float(getattr(location, "z", 0.0))
        ref_yaw = float(reference_transform.rotation.yaw)
    except (AttributeError, TypeError, ValueError):
        return None  # fake location/transform without coordinates or yaw
    try:
        waypoint = world.get_waypoint(lx, ly, lz)
    except Exception as exc:
        raise RuntimeError(f"map query failed at ({lx:.1f}, {ly:.1f}): {exc}") from exc
    if waypoint is None:
        return "off_driving"
    try:
        wloc = waypoint.transform.location
        distance = math.hypot(float(wloc.x) - lx, float(wloc.y) - ly)
        width = float(getattr(waypoint, "lane_width", 3.5))
        yaw = float(waypoint.transform.rotation.yaw)
    except (AttributeError, TypeError, ValueError):
        return None  # fake waypoint without lane geometry
    if distance > width / 2.0 + LANE_PROJECTION_TOLERANCE_M:
        return "off_driving"
    delta = abs((yaw - ref_yaw + 180.0) % 360.0 - 180.0)
    return "driving_opposite" if delta > 90.0 else "driving_same"


_LANE_SEVERITY = {"driving_same": 0, "driving_opposite": 1, "off_driving": 2}
LANE_PATH_SPACING_M = 2.0
LANE_PATH_MARGIN_M = 5.0
# When an actor's lane-check budget does not fit on its offset polyline, the
# intended route is rebuilt this many times longer (each time ×GROWTH) at the
# same waypoint before the candidate is rejected.
ROUTE_COVERAGE_EXTENSIONS = 2
ROUTE_COVERAGE_GROWTH = 1.5


def _lane_path_class(world: Any, route_plan: Any, start_s: float, end_s: float,
                     lateral: float) -> Tuple[Optional[str], List[float], Optional[float]]:
    """Worst lane class along the offset polyline from ``start_s`` to
    ``end_s`` (route metres), sampled every LANE_PATH_SPACING_M; None when any
    sample is unknown (fakes). Returns (class, [start, end_checked],
    first_violation_s): the first sampled route metre whose ground is not a
    same-direction driving lane, None when the whole span is."""
    end_s = min(float(end_s), float(getattr(route_plan, "max_s", end_s)))
    worst: Optional[str] = "driving_same"
    first_violation: Optional[float] = None
    s = float(start_s)
    checked_end = s
    while True:
        cls = _lane_class(world, route_plan.offset_location_at(s, lateral),
                          route_plan.pose_at(s).transform)
        if cls is None:
            return None, [float(start_s), s], None
        if _LANE_SEVERITY[cls] > 0 and first_violation is None:
            first_violation = s
        if _LANE_SEVERITY[cls] > _LANE_SEVERITY[worst]:
            worst = cls
        checked_end = s
        if s >= end_s - 1e-9:
            break
        s = min(s + LANE_PATH_SPACING_M, end_s)
    return worst, [float(start_s), checked_end], first_violation


def _lane_check_laterals(config: Dict[str, Any], aid: str, lateral: float) -> List[float]:
    """Lateral offsets the actor may occupy across rollouts: its slot (kept
    for the whole L4 control rollout) and, for an actor whose environment
    behaviour moves it laterally, the ego lane (0) and the offset it reaches
    by the end of the rollout, clamped so it never passes the ego lane."""
    laterals = [float(lateral)]
    lat_v = float(((config.get("env_behavior") or {}).get(aid) or {}).get("lat", 0.0))
    if lat_v != 0.0:
        horizon = float(config.get("total_duration_sec", 8.0))
        trigger = float((config.get("environment_event") or {}).get(
            "trigger_time_sec", config.get("intervention_time_sec", 0.0)))
        reached = float(lateral) + lat_v * max(0.0, horizon - trigger)
        if (float(lateral) > 0 and reached < 0) or (float(lateral) < 0 and reached > 0):
            reached = 0.0
        laterals.extend(v for v in (0.0, reached) if v not in laterals)
    return laterals


def _lane_check_span(config: Dict[str, Any], aid: str, dist: float, speed: float,
                     lateral: float, route_plan: Any = None) -> Tuple[float, float, float]:
    """(start_s, end_s, travel_m) for the lane check of a moving vehicle.

    The travel budget is the whole recorded rollout at the actor's initial
    speed (plus any positive longitudinal acceleration and LANE_PATH_MARGIN_M),
    for every actor: the L4 control rollout drops the target's intervention,
    so even a cut-in car may keep its slot to the end. Physical metres are
    converted to route coordinate with the same ``advance_along_offset`` the
    actor itself uses, evaluated on every lateral offset the actor may occupy
    (`_lane_check_laterals`: on a curve the ego lane and the slot advance the
    route coordinate at different rates), and the farthest reach is the span
    end. Without that API the budget is added to the route coordinate."""
    horizon = float(config.get("total_duration_sec", 8.0))
    accel = float(((config.get("env_behavior") or {}).get(aid) or {}).get("accel", 0.0))
    travel = max(0.0, speed) * horizon + max(0.0, accel) * horizon ** 2 / 2.0 + LANE_PATH_MARGIN_M
    start = float(dist)
    if route_plan is not None and hasattr(route_plan, "advance_along_offset"):
        ends = []
        for lat_value in _lane_check_laterals(config, aid, lateral):
            try:
                ends.append(float(route_plan.advance_along_offset(start, travel, lat_value)))
            except ValueError as exc:
                # The offset polyline is shorter than the actor's budget
                # (inside of a curve, or a short route): the rollout itself
                # would fail with this error later. Never record an
                # unchecked budget.
                raise PlacementRejected(
                    f"actor {aid}: route offset coverage insufficient for {travel:.1f} m "
                    f"on the {lat_value:+g} m polyline from {start:g} m ({exc})") from exc
        end = max(ends)
    else:
        end = start + travel
    return start, end, travel


def _is_crossing_spec(spec: Dict[str, Any]) -> bool:
    """A road-entry actor (pedestrian or cyclist crossing) starts off the
    driving lanes by design and enters the road under its behaviour; its
    slot is never a driving lane and is not lane-checked."""
    return spec.get("default_role") == "crossing_actor"


def _is_moving_spec(spec: Dict[str, Any]) -> bool:
    entity = spec.get("entity_type")
    return (entity in _VEHICLE_FAMILY or entity in ("pedestrian", "walker", "ego_vehicle")) \
        and entity != "parked_vehicle"


def _parked_lateral_candidates(config: Dict[str, Any], aid: str, lat: float) -> List[float]:
    """Same-side fallbacks for a parked vehicle that keep it clear of the lane
    centre of every moving actor on that side (ego included), so the fallback
    cannot drop a parked car into an NPC's lane."""
    parked_w = _actor_extent(config["actor_defs"][aid])[1]
    movers = []
    for key, spec in config["actor_defs"].items():
        if key == aid or not _is_moving_spec(spec):
            continue
        movers.append((float(spec.get("lateral_offset", 0.0)), _actor_extent(spec)[1]))
    allowed = []
    for candidate in _context_lateral_candidates(lat):
        clear = True
        for mover_lat, mover_w in movers:
            same_side = mover_lat == 0.0 or mover_lat * candidate > 0.0
            if same_side and abs(candidate - mover_lat) < (parked_w + mover_w) / 2.0 + PARKED_CLEARANCE_MARGIN_M:
                clear = False
                break
        if clear:
            allowed.append(candidate)
    return allowed


def _spawn_cast_at_waypoint(config: Dict[str, Any], world: Any,
                            api: Dict[str, Any], wp: Any,
                            context_offsets: Optional[Dict[str, float]] = None,
                            route_plan: Any = None,
                            lane_checks: Optional[Dict[str, str]] = None):
    """Spawn one complete cast at a preselected base waypoint.

    ``lane_checks`` (optional dict) receives, per moving vehicle with a
    lateral slot, the map classification of the ground under that slot."""
    actors: Dict[str, Any] = {}
    controllers: Dict[str, Any] = {}
    dynamic: List[str] = []
    resolved_context_offsets: Dict[str, float] = {}
    context_offsets = context_offsets or {}

    def placement_waypoint(distance: float) -> Tuple[Any, float]:
        if route_plan is None:
            return wp, distance
        return route_plan.pose_at(distance), 0.0

    def slot_location(distance: float, lateral: float, base_wp: Any) -> Any:
        if route_plan is not None and hasattr(route_plan, "offset_location_at"):
            return route_plan.offset_location_at(distance, lateral)
        anchor = base_wp
        if distance > 0 and hasattr(base_wp, "next"):
            nxt = base_wp.next(distance)
            anchor = nxt[0] if nxt else base_wp
        return _offset_location(anchor, lateral)

    for aid, spec in config["actor_defs"].items():
        entity = spec["entity_type"]
        model = spec.get("model")
        dist = float(spec.get("distance_ahead", 0.0))
        lat = float(spec.get("lateral_offset", 0.0))
        speed = float(spec.get("initial_speed", 0.0))
        actor_waypoint, actor_distance = placement_waypoint(dist)

        try:
            if entity == "ego_vehicle":
                actors[aid] = api["Ego"](
                    world, model=model, waypoint=wp, initial_speed=speed,
                    route_plan=route_plan)
                controllers[aid] = api["EgoActions"](actors[aid])
                dynamic.append(aid)
            elif entity == "pedestrian":
                behavior = (config.get("env_behavior") or {}).get(aid, {})
                facing = ("cross_right" if float(behavior.get("lat", 0.0)) >= 0
                          else "cross_left")
                actors[aid] = api["Pedestrian"](
                    world, model=model, road_waypoint=actor_waypoint,
                    distance_ahead=actor_distance,
                    side=("right" if lat >= 0 else "left"), offset=abs(lat),
                    facing=facing)
                controllers[aid] = api["PedestrianActions"](actors[aid])
                dynamic.append(aid)
            elif entity == "parked_vehicle":
                lateral_candidates = ([float(context_offsets[aid])]
                                      if aid in context_offsets
                                      else _parked_lateral_candidates(config, aid, lat))
                if not lateral_candidates:
                    raise PlacementRejected(
                        f"actor {aid}: no same-side parked slot clear of the moving "
                        f"actors' lanes (configured lateral {lat:g} m)")
                last_collision: Optional[BaseException] = None
                for candidate_lat in lateral_candidates:
                    try:
                        actor = api["NPC"](
                            world, vehicle_type="car", model=model,
                            waypoint=actor_waypoint, initial_speed=0.0,
                            distance_ahead=actor_distance,
                            lane_offset=candidate_lat)
                    except Exception as exc:
                        if not _is_spawn_collision(exc):
                            raise
                        last_collision = exc
                        continue
                    actors[aid] = actor
                    resolved_context_offsets[aid] = candidate_lat
                    break
                else:
                    raise RuntimeError(
                        f"actor {aid} collided at all same-side lateral "
                        f"candidates {lateral_candidates}; last error: "
                        f"{last_collision}") from last_collision
                # A parked car is collidable but is neither updated nor controlled.
            elif entity in _VEHICLE_FAMILY:
                vehicle_type = "car" if entity == "vehicle" else entity
                if lat != 0.0 and not _is_crossing_spec(spec):
                    # A moving vehicle in a lateral slot must stay on a
                    # same-direction driving lane of the map for the whole
                    # span it keeps that slot (the adjacent lane may end);
                    # otherwise it would run on the pavement or against
                    # traffic. Without a route plan only the spawn point can
                    # be checked.
                    start_s, end_s, travel = _lane_check_span(config, aid, dist, speed, lat, route_plan)
                    first_violation: Optional[float] = None
                    if (route_plan is not None and hasattr(route_plan, "offset_location_at")
                            and hasattr(route_plan, "pose_at")):
                        lane, checked, first_violation = _lane_path_class(
                            world, route_plan, start_s, end_s, lat)
                    else:
                        reference = placement_waypoint(dist)[0]
                        lane = _lane_class(world, slot_location(dist, lat, wp),
                                           getattr(reference, "transform", None))
                        checked = [float(dist), float(dist)]
                        travel = 0.0
                        if lane in ("off_driving", "driving_opposite"):
                            first_violation = float(dist)
                    if lane_checks is not None:
                        lane_checks[aid] = {"class": lane or "unknown", "span_m": checked,
                                            "travel_m": round(travel, 3)}
                        if first_violation is not None:
                            lane_checks[aid]["first_violation_m"] = round(first_violation, 3)
                    if lane in ("off_driving", "driving_opposite"):
                        where = (f"from {first_violation:g} m" if first_violation is not None
                                 else "somewhere")
                        raise PlacementRejected(
                            f"actor {aid}: lateral slot {lat:g} m is {lane} ground {where} "
                            f"along the route (checked {checked[0]:g} to {checked[1]:g} m); "
                            f"a moving {entity} needs a same-direction driving lane for "
                            f"that whole span")
                actors[aid] = api["NPC"](
                    world, vehicle_type=vehicle_type, model=model, waypoint=wp,
                    initial_speed=speed, distance_ahead=dist, lane_offset=lat,
                    route_plan=route_plan)
                controllers[aid] = api["NPCActions"](actors[aid])
                dynamic.append(aid)
            else:
                # Static street furniture is contextual rather than causal. Try
                # deterministic same-side offsets without moving it into ego's
                # lane, and persist the selected offset across every rollout.
                prop_wp = actor_waypoint
                if route_plan is None:
                    if dist > 0:
                        prop_wp = wp.next(dist)[0]
                    elif dist < 0:
                        prop_wp = wp.previous(abs(dist))[0]
                lateral_candidates = ([float(context_offsets[aid])]
                                      if aid in context_offsets
                                      else _context_lateral_candidates(lat))
                last_collision = None
                for candidate_lat in lateral_candidates:
                    try:
                        actor = api["StaticProp"](
                            world,
                            prop_type=_STATIC_PROP_TYPE.get(entity, "street_barrier"),
                            model=model,
                            position=_offset_location(prop_wp, candidate_lat),
                            rotation=_road_rotation(prop_wp))
                    except Exception as exc:
                        if not _is_spawn_collision(exc):
                            raise
                        last_collision = exc
                        continue
                    actors[aid] = actor
                    resolved_context_offsets[aid] = candidate_lat
                    break
                else:
                    raise RuntimeError(
                        f"actor {aid} collided at all same-side lateral "
                        f"candidates {lateral_candidates}; last error: "
                        f"{last_collision}") from last_collision
        except PlacementRejected:
            raise
        except RuntimeError as exc:
            if str(exc).startswith(f"actor {aid} "):
                raise
            raise RuntimeError(f"actor {aid} spawn failed: {exc}") from exc
    return (actors, controllers, dynamic), resolved_context_offsets


def _is_spawn_collision(exc: BaseException) -> bool:
    if isinstance(exc, PlacementRejected):
        return True
    return (isinstance(exc, RuntimeError)
            and "collision at spawn position" in str(exc).lower())


def _required_route_distance(config: Dict[str, Any]) -> float:
    """Conservative forward route coverage for every ego rollout."""
    ego_spec = next(
        (spec for spec in config.get("actor_defs", {}).values()
         if spec.get("entity_type") == "ego_vehicle"
         or spec.get("canonical_id") == "ego"),
        {},
    )
    initial_speed = max(0.0, float(ego_spec.get("initial_speed", 0.0)))
    total_duration = max(0.0, float(config.get("total_duration_sec", 8.0)))
    intervention_time = max(
        0.0, float(config.get("intervention_time_sec", 0.0)))
    acceleration_duration = max(0.0, total_duration - intervention_time)
    max_acceleration = 0.0
    ego_policies = [
        branch.get("intervention_type")
        for branch in config.get("branches", {}).values()
    ]
    l4_query = config.get("l4_query") or {}
    if l4_query:
        ego_policies.append(l4_query.get("ego_policy", "do_ego_no_action"))
    for intervention_type in ego_policies:
        _method, params = INTERVENTION_TO_EGO_ACTION.get(
            intervention_type, ("maintain_speed", {}))
        max_acceleration = max(
            max_acceleration, float(params.get("acceleration", 0.0)))
    return (
        initial_speed * total_duration
        + 0.5 * max_acceleration * acceleration_duration ** 2
        + _ROUTE_COVERAGE_MARGIN_M
    )


def _required_cast_route_distance(config: Dict[str, Any]) -> float:
    """Forward coverage needed to place and advance every road vehicle."""
    total_duration = max(0.0, float(config.get("total_duration_sec", 8.0)))
    trigger_time = max(0.0, float(
        (config.get("environment_event") or {}).get(
            "trigger_time_sec", config.get("intervention_time_sec", 0.0))))
    post_trigger = max(0.0, total_duration - trigger_time)
    furthest = max(0.0, _required_route_distance(config)
                   - _ROUTE_COVERAGE_MARGIN_M)
    behavior = config.get("env_behavior") or {}
    for aid, spec in config.get("actor_defs", {}).items():
        distance = float(spec.get("distance_ahead", 0.0))
        entity = str(spec.get("entity_type", ""))
        travel = 0.0
        if entity in _VEHICLE_FAMILY:
            speed = max(0.0, float(spec.get("initial_speed", 0.0)))
            acceleration = max(
                0.0, float((behavior.get(aid) or {}).get("accel", 0.0)))
            travel = (
                speed * total_duration
                + 0.5 * acceleration * post_trigger ** 2
            )
        furthest = max(furthest, distance + travel)
    return furthest + _ROUTE_COVERAGE_MARGIN_M


def spawn_cast(config: Dict[str, Any], world: Any, api: Dict[str, Any],
               placement: Optional[Dict[str, Any]] = None):
    """Spawn every actor in config['actor_defs'] via the wrapper classes.

    Returns (actors: {id: instance}, controllers: {id: actions-instance},
    dynamic_ids: ids that get update()/collision checks). Static entities
    (guardrails, signs, cones) spawn as StaticProp and are never stepped;
    parked vehicles spawn as speed-0 NPCs so their bounding boxes still
    participate in collision checks.
    """
    distances = [float(spec.get("distance_ahead", 0.0))
                 for spec in config.get("actor_defs", {}).values()]
    forward_required = max([0.0, *distances]) + 5.0
    backward_required = abs(min([0.0, *distances])) + 5.0

    # A run_scenario call shares this state across all L3 and L4 rollouts. Once
    # one full cast succeeds, every other rollout uses that exact base waypoint
    # instead of repeating failed probes or silently changing scene geometry.
    candidate_offset = int((placement or {}).get("candidate_offset", 0))
    if placement is not None and placement.get("waypoint") is not None:
        candidates = [placement["waypoint"]]
        total_candidates = 1
        candidate_base = 0
        cached = True
    elif hasattr(world, "get_scenario_waypoints"):
        all_candidates = world.get_scenario_waypoints(
            config.get("seed", 0),
            forward_required=forward_required,
            backward_required=backward_required,
            limit=MAX_SPAWN_WAYPOINT_ATTEMPTS,
        )
        total_candidates = len(all_candidates)
        candidates = all_candidates[candidate_offset:]
        candidate_base = candidate_offset
        cached = False
    elif hasattr(world, "get_scenario_waypoint"):
        if candidate_offset:
            raise ScenarioCandidateExhausted(
                "world exposes only one scenario waypoint; deterministic "
                "preflight candidates are exhausted")
        candidates = [world.get_scenario_waypoint(
            config.get("seed", 0),
            forward_required=forward_required,
            backward_required=backward_required,
        )]
        total_candidates = 1
        candidate_base = 0
        cached = False
    else:
        if candidate_offset:
            raise ScenarioCandidateExhausted(
                "fallback world exposes only one waypoint; deterministic "
                "preflight candidates are exhausted")
        candidates = [world.get_waypoint(250, -370)]
        total_candidates = 1
        candidate_base = 0
        cached = False

    if not candidates:
        raise ScenarioCandidateExhausted(
            f"deterministic waypoint candidates exhausted after "
            f"{candidate_offset}/{total_candidates} preflight rejection(s)")

    route_required = _required_cast_route_distance(config)
    last_placement_error: Optional[BaseException] = None
    for index, wp in enumerate(candidates):
        route_evidence = ((placement or {}).get("route_evidence")
                          if cached else None)
        route_plan = ((placement or {}).get("route_plan") if cached else None)
        route_forward = route_required
        if not cached and hasattr(world, "build_intended_route"):
            try:
                route_plan = world.build_intended_route(
                    wp, forward_distance=route_forward,
                    backward_distance=backward_required)
                route_evidence = route_plan.evidence()
            except (RuntimeError, ValueError) as exc:
                last_placement_error = exc
                if index + 1 < len(candidates):
                    print(
                        f"[route retry] {config.get('scenario_id', '<unknown>')}: "
                        f"candidate {candidate_base + index + 1}/{total_candidates} "
                        f"rejected ({exc})"
                    )
                    continue
                break
        elif not cached and hasattr(world, "sample_route_evidence"):
            try:
                route_evidence = world.sample_route_evidence(
                    wp, forward_distance=route_required)
            except (RuntimeError, ValueError) as exc:
                last_placement_error = exc
                if index + 1 < len(candidates):
                    print(
                        f"[route retry] {config.get('scenario_id', '<unknown>')}: "
                        f"candidate {candidate_base + index + 1}/{total_candidates} "
                        f"rejected ({exc})"
                    )
                    continue
                break
        try:
            cached_offsets = ((placement or {}).get("context_lateral_offsets")
                              if cached else None)
            lane_checks: Dict[str, str] = {}
            for extension in range(ROUTE_COVERAGE_EXTENSIONS + 1):
                try:
                    cast, resolved_offsets = _spawn_cast_at_waypoint(
                        config, world, api, wp, context_offsets=cached_offsets,
                        route_plan=route_plan, lane_checks=lane_checks)
                    break
                except PlacementRejected as exc:
                    # A short offset polyline (inside of a curve) is fixed by
                    # a longer route at the same waypoint, not by giving up
                    # on the candidate; try up to ROUTE_COVERAGE_EXTENSIONS
                    # longer routes before rejecting it.
                    if ("offset coverage" not in str(exc) or cached
                            or extension >= ROUTE_COVERAGE_EXTENSIONS
                            or not hasattr(world, "build_intended_route")):
                        raise
                    world.cleanup()
                    route_forward *= ROUTE_COVERAGE_GROWTH
                    print(
                        f"[route extend] {config.get('scenario_id', '<unknown>')}: "
                        f"candidate {candidate_base + index + 1}/{total_candidates} "
                        f"route -> {route_forward:.0f} m ({exc})"
                    )
                    try:
                        route_plan = world.build_intended_route(
                            wp, forward_distance=route_forward,
                            backward_distance=backward_required)
                        route_evidence = route_plan.evidence()
                    except (RuntimeError, ValueError) as build_exc:
                        # The map has no longer route here: a deterministic
                        # property of this candidate, so reject it and move
                        # on (never an infrastructure error).
                        raise PlacementRejected(
                            f"route cannot be extended to {route_forward:.0f} m at this "
                            f"waypoint ({build_exc}); {exc}") from build_exc
        except Exception as exc:
            # A constructor may fail after several actors have spawned. Always
            # destroy that partial cast before retrying or propagating.
            world.cleanup()
            if not _is_spawn_collision(exc):
                raise
            last_placement_error = exc
            if cached:
                if isinstance(exc, PlacementRejected):
                    # Deterministic geometry: this sample cannot be a
                    # reference at its cached placement. Reject-and-resample,
                    # never an infrastructure error to replay at the same seed.
                    raise ScenarioCandidateExhausted(
                        f"cached placement rejected on re-check: {exc}") from exc
                raise RuntimeError(
                    "cached scenario waypoint became blocked while recreating "
                    f"the cast: {exc}") from exc
            if index + 1 < len(candidates):
                print(
                    f"[spawn retry] {config.get('scenario_id', '<unknown>')}: "
                    f"candidate {candidate_base + index + 1}/{total_candidates} "
                    f"collided ({exc})"
                )
                continue
            break
        if placement is not None and not cached:
            placement["waypoint"] = wp
            placement["candidate_number"] = candidate_base + index + 1
            placement["context_lateral_offsets"] = resolved_offsets
            placement["moving_actor_lane_checks"] = dict(lane_checks)
            if route_evidence is not None:
                placement["route_evidence"] = route_evidence
            if route_plan is not None:
                placement["route_plan"] = route_plan
        if not cached:
            for aid, selected_lat in resolved_offsets.items():
                configured_lat = float(
                    config["actor_defs"][aid].get("lateral_offset", 0.0))
                if selected_lat != configured_lat:
                    print(
                        f"[spawn fallback] {config.get('scenario_id', '<unknown>')}: "
                        f"{aid} lateral {configured_lat:g} -> {selected_lat:g} m"
                    )
        if index:
            print(
                f"[spawn retry] {config.get('scenario_id', '<unknown>')}: "
                f"selected candidate {candidate_base + index + 1}/{total_candidates}"
            )
        return cast

    raise ScenarioCandidateExhausted(
        f"unable to spawn complete cast at {len(candidates)} remaining deterministic "
        f"waypoint candidate(s); last error: {last_placement_error}"
    ) from last_placement_error


def _road_rotation(waypoint: Any) -> Any:
    """A prop's spawn rotation: aligned with the road at its waypoint, so its
    declared length runs along the lane and its telemetry heading is that
    yaw. None for fakes (StaticProp then defaults to yaw 0)."""
    try:
        import carla as _carla
        yaw = float(waypoint.transform.rotation.yaw)
        return _carla.Rotation(pitch=0.0, yaw=yaw, roll=0.0)
    except Exception:
        return None


def _offset_location(waypoint: Any, lateral: float) -> Any:
    """Location `lateral` metres to the right (+) / left (−) of a waypoint."""
    tf = waypoint.transform
    right = tf.rotation.get_right_vector()
    loc = tf.location
    try:
        import carla as _carla  # real CARLA: build a proper Location
        return _carla.Location(x=loc.x + right.x * lateral,
                               y=loc.y + right.y * lateral, z=loc.z)
    except Exception:
        pass
    try:  # fakes: a plain shifted point so lane checks still see the offset
        from types import SimpleNamespace
        return SimpleNamespace(x=float(loc.x) + float(right.x) * lateral,
                               y=float(loc.y) + float(right.y) * lateral,
                               z=float(getattr(loc, "z", 0.0)))
    except Exception:
        return loc  # position identity is irrelevant for such fakes


def _placement_record(config: Dict[str, Any], placement: Dict[str, Any]) -> Dict[str, Any]:
    """Build JSON-safe provenance for the effective runtime placement."""
    waypoint = placement["waypoint"]
    transform = waypoint.transform
    location = transform.location
    rotation = transform.rotation
    record = {
        "map": config.get("map"),
        "simulation_seed": config.get("seed"),
        "candidate_number": placement.get("candidate_number"),
        "base_transform": {
            "x": round(float(location.x), 6),
            "y": round(float(location.y), 6),
            "z": round(float(location.z), 6),
            "pitch": round(float(getattr(rotation, "pitch", 0.0)), 6),
            "yaw": round(float(getattr(rotation, "yaw", 0.0)), 6),
            "roll": round(float(getattr(rotation, "roll", 0.0)), 6),
        },
        "context_lateral_offsets_m": {
            aid: float(value) for aid, value in sorted(
                placement.get("context_lateral_offsets", {}).items())
        },
        "moving_actor_lane_checks": {
            aid: copy.deepcopy(value) for aid, value in sorted(
                (placement.get("moving_actor_lane_checks") or {}).items())
        },
    }
    route_evidence = placement.get("route_evidence")
    if isinstance(route_evidence, dict):
        record["route_id"] = route_evidence.get("route_id")
        record["route_mode"] = route_evidence.get("route_mode")
    return record


def _apply_ego_intervention(ego_actions: Any, intervention_type: str) -> None:
    method, params = INTERVENTION_TO_EGO_ACTION.get(
        intervention_type, ("maintain_speed", {}))
    if method == "brake_and_steer_left":
        ego_actions.brake(deceleration=params["deceleration"])
        ego_actions.steer_left(intensity=params["intensity"])
    elif method == "brake_and_steer_right":
        ego_actions.brake(deceleration=params["deceleration"])
        ego_actions.steer_right(intensity=params["intensity"])
    else:
        getattr(ego_actions, method)(**params)


def _apply_env_behavior(config: Dict[str, Any], actors: Dict[str, Any],
                        controllers: Dict[str, Any]) -> None:
    """At the environment-event trigger, start every non-ego actor's declared
    motion: longitudinal accel for vehicles, lateral crossing for walkers.
    Configs without env_behavior fall back to applying the event's
    deceleration to the actor carrying collision_role."""
    env_behavior = config.get("env_behavior")
    if env_behavior:
        for aid, b in env_behavior.items():
            ctl = controllers.get(aid)
            if ctl is None:
                continue
            accel = float(b.get("accel", 0.0))
            lat = float(b.get("lat", 0.0))
            entity = config["actor_defs"][aid]["entity_type"]
            if entity == "pedestrian" and lat != 0.0:
                ctl.start_crossing(speed=abs(lat))
                continue
            if accel < 0:
                ctl.brake(deceleration=-accel)
            elif accel > 0:
                ctl.accelerate(acceleration=accel)
            # Config lateral velocity is positive to the right.  Vehicle
            # controllers expose direction-specific methods and accept a
            # normalized intensity.
            if lat < 0:
                ctl.steer_left(intensity=min(1.0, abs(lat) / 3.5))
            elif lat > 0:
                ctl.steer_right(intensity=min(1.0, abs(lat) / 3.5))
        return
    # fallback for configs without env_behavior
    params = (config.get("environment_event") or {}).get("params", {})
    decel = float(params.get("deceleration", params.get("intensity", 0.0)))
    for aid, spec in config["actor_defs"].items():
        if spec.get("collision_role") and decel > 0 and aid in controllers:
            controllers[aid].brake(deceleration=decel)


def run_branch(config: Dict[str, Any], branch_name: str, world: Any,
               api: Dict[str, Any], output_dir: str,
               placement: Optional[Dict[str, Any]] = None):
    """Simulate ONE branch of a scenario. Returns (trajectory, controllers)."""
    branch_out = os.path.join(output_dir, config["output_subdir"], branch_name)
    try:
        actors, controllers, dynamic = spawn_cast(
            config, world, api, placement=placement)
        ego_id = next(aid for aid, s in config["actor_defs"].items()
                      if s["entity_type"] == "ego_vehicle")

        camera = config.get("camera") or {}
        recorder = api["TrajectoryRecorder"](
            world, attach_to=actors[ego_id].carla_actor, output_dir=branch_out,
            resolution=tuple(camera.get("resolution", (1024, 768))),
            fov=float(camera.get("fov_deg", 100.0)),
            cam_offset=tuple(camera.get("offset_m", (1.5, 0.0, 1.7))),
            cam_pitch=float(camera.get("pitch_deg", 0.0)))
        recorder.start()
        detector_ids = [aid for aid in actors if aid != ego_id]
        detectors = [api["CollisionDetector"](actors[ego_id], actors[aid])
                     for aid in detector_ids]
        # Capture the initial state before any kinematic update.  In protocol
        # configs this frame is T=-3.5 s relative to the intervention root, and it
        # keeps camera images aligned with telemetry timestamps.
        world.tick()
        recorder.wait_for_frame()
        time.sleep(0.2)

        total_steps = int(float(config.get("total_duration_sec", 8.0)) / STEP_LENGTH)
        env_step = int(float(config["environment_event"]["trigger_time_sec"]) / STEP_LENGTH)
        interv_step = int(float(config["intervention_time_sec"]) / STEP_LENGTH)
        interv_type = config["branches"][branch_name]["intervention_type"]

        env_applied = False
        for step in range(total_steps + 1):
            t = round(step * STEP_LENGTH, 2)
            if step > 0:
                for aid in dynamic:
                    actors[aid].update(STEP_LENGTH)
                world.tick()
                recorder.wait_for_frame()
            # The observed-input protocol needs static as well as dynamic root
            # actors.  Recording the full cast also lets the release overlay
            # and scene-boundary gate use the same stable actor universe.
            recorder.record_step(t, branch_name, **actors)
            # T0 is an observed pre-intervention state. Apply both the named
            # actor event and ego policy only after recording that step so all
            # branches share an identical root and depth 1 is the first effect.
            if step >= env_step and not env_applied:
                _apply_env_behavior(config, actors, controllers)
                env_applied = True
            if step == interv_step:
                _apply_ego_intervention(controllers[ego_id], interv_type)
            hits = [aid for aid, detector in zip(detector_ids, detectors)
                    if detector.check()]
            if hits:
                recorder.record_collision(t, branch_name, partners=hits)
                break
        time.sleep(0.2)
        return recorder.get_trajectory(), controllers
    finally:
        world.cleanup()


def _bbox_area_px2(value: Any) -> float:
    if not isinstance(value, (list, tuple)) or len(value) != 4:
        return 0.0
    try:
        x1, y1, x2, y2 = (float(item) for item in value)
    except (TypeError, ValueError):
        return 0.0
    return max(0.0, x2 - x1) * max(0.0, y2 - y1)


def _preflight_route_projection(
        point: Tuple[float, float], line: List[List[float]]) -> Dict[str, Any]:
    px, py = point
    best: Optional[Dict[str, Any]] = None
    last_segment = len(line) - 2
    for index, (start, end) in enumerate(zip(line, line[1:])):
        ax, ay = float(start[0]), float(start[1])
        bx, by = float(end[0]), float(end[1])
        dx, dy = bx - ax, by - ay
        length_sq = dx * dx + dy * dy
        if length_sq <= 1e-12:
            continue
        raw_fraction = ((px - ax) * dx + (py - ay) * dy) / length_sq
        fraction = max(0.0, min(1.0, raw_fraction))
        qx, qy = ax + fraction * dx, ay + fraction * dy
        distance = math.hypot(px - qx, py - qy)
        status = "supported"
        if index == 0 and raw_fraction < 0.0:
            status = "unsupported_before_start"
        elif index == last_segment and raw_fraction > 1.0:
            status = "unsupported_after_end"
        candidate = {"distance": distance, "status": status}
        if (best is None or distance < best["distance"] - 1e-9
                or (abs(distance - best["distance"]) <= 1e-9
                    and status == "supported"
                    and best["status"] != "supported")):
            best = candidate
    return best or {"distance": float("inf"), "status": "invalid_route"}


def _preflight_route_reasons(
        selected: List[Tuple[float, Dict[str, Any]]], ego_key: str,
        route_evidence: Optional[Dict[str, Any]]) -> List[str]:
    if not isinstance(route_evidence, dict):
        return []
    if route_evidence.get("coordinate_frame") != "world":
        return ["preflight intended route is not expressed in world coordinates"]
    raw_line = route_evidence.get("route_centerline")
    if not isinstance(raw_line, list) or len(raw_line) < 2:
        return ["preflight intended route has fewer than two points"]
    try:
        line = [[float(point[0]), float(point[1])] for point in raw_line]
    except (TypeError, ValueError, IndexError):
        return ["preflight intended route contains invalid geometry"]
    halfwidth = float(route_evidence.get(
        "route_half_width_m", CARLA_ROUTE_HALF_WIDTH_M))
    reasons = []
    for offset, step in selected:
        ego = step.get(ego_key)
        if not isinstance(ego, dict):
            continue
        try:
            point = (float(ego["x"]), float(ego["y"]))
        except (KeyError, TypeError, ValueError):
            reasons.append(
                f"preflight ego route position missing at T={offset:.1f}s")
            continue
        projection = _preflight_route_projection(point, line)
        if projection["status"] != "supported":
            reasons.append(
                f"preflight intended route does not cover ego at "
                f"T={offset:.1f}s ({projection['status']})")
        elif float(projection["distance"]) > halfwidth:
            reasons.append(
                f"preflight ego outside intended route at T={offset:.1f}s "
                f"(distance={projection['distance']:.2f}m, "
                f"halfwidth={halfwidth:.2f}m)")
    return reasons


def _validate_protocol_preflight(
        trajectory: List[Dict[str, Any]], config: Dict[str, Any],
        frames_dir: Optional[str] = None,
        route_evidence: Optional[Dict[str, Any]] = None
        ) -> Tuple[bool, List[str]]:
    """Validate one factual CARLA rollout before sharing its placement.

    The gate is deliberately waypoint-level: it requires the exact eight
    protocol history timestamps, a usable projected box for the declared
    critical actor in every history image, and no collision at/before T0.
    When ``frames_dir`` is supplied, each selected telemetry record must also
    be bound to a PNG written by the RGB sensor.
    """
    from simulation.protocol import HISTORY_TIMESTAMPS

    reasons: List[str] = []
    t0 = float(config.get("intervention_time_sec", 0.0))
    actor_defs = config.get("actor_defs", {})
    ego_key = next((key for key, spec in actor_defs.items()
                    if str(spec.get("canonical_id", key)) == "ego"), None)
    critical_id = config.get("critical_actor_id")
    if critical_id is None:
        critical_id = next((spec.get("canonical_id", key)
                            for key, spec in actor_defs.items()
                            if key != ego_key and spec.get("collision_role")), None)
    critical_key = next((key for key, spec in actor_defs.items()
                         if str(spec.get("canonical_id", key)) == str(critical_id)), None)
    if ego_key is None:
        reasons.append("preflight has no canonical ego actor")
    if critical_key is None:
        reasons.append(f"preflight cannot resolve critical actor {critical_id!r}")
    if reasons:
        return False, reasons

    selected: List[Tuple[float, Dict[str, Any]]] = []
    for offset in HISTORY_TIMESTAMPS:
        target = round(t0 + float(offset), 2)
        candidates = [step for step in trajectory
                      if abs(float(step.get("time_sec", -1e9)) - target) <= 0.06]
        if not candidates:
            reasons.append(f"preflight history is missing T={offset:.1f}s")
            continue
        step = min(candidates, key=lambda item: abs(
            float(item.get("time_sec", -1e9)) - target))
        selected.append((float(offset), step))

    for step in trajectory:
        step_time = float(step.get("time_sec", 1e9))
        collision_time = float(step.get("collision_time", step_time))
        if step.get("collision") and collision_time <= t0 + 1e-6:
            reasons.append(
                f"preflight collision at or before T0 (time={collision_time:.2f}s)")
            break

    camera_frames = set()
    for offset, step in selected:
        if ego_key not in step:
            reasons.append(f"preflight ego missing at T={offset:.1f}s")
        critical = step.get(critical_key)
        if not isinstance(critical, dict):
            reasons.append(f"preflight critical actor missing at T={offset:.1f}s")
        else:
            area = _bbox_area_px2(critical.get("bbox_2d"))
            if area < _MIN_VISIBLE_BBOX_AREA_PX2:
                reasons.append(
                    f"preflight critical actor not visible at T={offset:.1f}s "
                    f"(bbox_area={area:.1f}px2)")
        if frames_dir is not None:
            frame = step.get("camera_frame")
            if frame is None:
                reasons.append(f"preflight camera frame missing at T={offset:.1f}s")
                continue
            try:
                frame_number = int(frame)
            except (TypeError, ValueError):
                reasons.append(f"preflight camera frame invalid at T={offset:.1f}s")
                continue
            camera_frames.add(frame_number)
            if not os.path.isfile(os.path.join(frames_dir, f"{frame_number:06d}.png")):
                reasons.append(f"preflight camera PNG missing at T={offset:.1f}s")
    reasons.extend(_preflight_route_reasons(
        selected, ego_key, route_evidence))
    if frames_dir is not None and len(camera_frames) != len(selected):
        reasons.append(
            f"preflight history uses {len(camera_frames)} distinct camera frames; "
            f"expected {len(selected)}")
    return not reasons, reasons


def _discard_preflight_candidate(config: Dict[str, Any], output_dir: str,
                                 branch_name: str,
                                 placement: Dict[str, Any]) -> int:
    """Remove a rejected factual rollout and advance deterministic placement."""
    candidate_number = int(placement.get("candidate_number", 0))
    shutil.rmtree(
        os.path.join(output_dir, config["output_subdir"], branch_name),
        ignore_errors=True,
    )
    placement.clear()
    placement["candidate_offset"] = candidate_number
    return candidate_number


def run_scenario(config: Dict[str, Any], world: Any, api: Optional[Dict[str, Any]] = None,
                 output_dir: str = "/workspace/out",
                 only_branches: Optional[List[str]] = None) -> Dict[str, Any]:
    """Simulate every branch of one scenario config. Returns the
    coriva_bench_examples.json scenario shape:
    {"scenario": id, "branches": {name: telemetry_steps}} (plus a private
    _controllers key used by the mocked unit tests)."""
    api = api or _default_api()
    result: Dict[str, Any] = {"scenario": config["scenario_id"], "branches": {},
                              "_controllers": {}}
    placement: Dict[str, Any] = {}
    branch_names = list(config["branches"])
    protocol_preflight = bool(config.get("template_id")) and not only_branches
    if protocol_preflight:
        factual_name = next(
            name for name, meta in config["branches"].items()
            if meta.get("is_factual"))
        branch_names = [factual_name] + [
            name for name in branch_names if name != factual_name]

        while factual_name not in result["branches"]:
            trajectory, controllers = run_branch(
                config, factual_name, world, api, output_dir,
                placement=placement)
            branch_dir = os.path.join(
                output_dir, config["output_subdir"], factual_name)
            valid, reasons = _validate_protocol_preflight(
                trajectory, config, frames_dir=branch_dir,
                route_evidence=placement.get("route_evidence"))
            if valid:
                result["branches"][factual_name] = trajectory
                result["_controllers"][factual_name] = controllers
                break
            rejected_candidate = _discard_preflight_candidate(
                config, output_dir, factual_name, placement)
            print(
                f"[preflight retry] {config.get('scenario_id', '<unknown>')}: "
                f"candidate {rejected_candidate}/{MAX_SPAWN_WAYPOINT_ATTEMPTS} "
                f"rejected ({'; '.join(reasons)})"
            )

    for branch_name in branch_names:
        if only_branches and branch_name not in only_branches:
            continue
        if branch_name in result["branches"]:
            continue
        traj, controllers = run_branch(
            config, branch_name, world, api, output_dir, placement=placement)
        result["branches"][branch_name] = traj
        result["_controllers"][branch_name] = controllers
    # Final-protocol L4 is a distinct deterministic CARLA rollout: the named
    # critical-actor environment event is applied while the ego follows the
    # declared fixed policy.  It is stored separately so it can never enter
    # the Layer-3 branch denominator.
    l4_query = config.get("l4_query")
    if l4_query and not only_branches:
        l4_config = copy.deepcopy(config)
        l4_name = "__L4_FIXED_EGO__"
        l4_config["branches"] = {
            l4_name: {
                "is_factual": False,
                "intervention_type": l4_query.get("ego_policy", "do_ego_no_action"),
                "target_actor_id": "ego",
                "trigger_depth": 1,
            }
        }
        trajectory, controllers = run_branch(
            l4_config, l4_name, world, api, output_dir, placement=placement)
        result["_l4_steps"] = trajectory
        result["_controllers"][l4_name] = controllers
        # Paired control retained for admission and the Option-B reference:
        # identical fixed ego policy/root, but no named-actor intervention.
        control_config = copy.deepcopy(l4_config)
        control_name = "__L4_NO_INTERVENTION__"
        control_config["branches"] = {
            control_name: dict(l4_config["branches"][l4_name])
        }
        # Hold every other actor behaviour fixed.  The paired control removes
        # only the named critical-actor intervention; otherwise the measured
        # difference would confound that intervention with changes to the
        # whole traffic scene.
        target_id = str(l4_query.get("target_actor_id", ""))
        target_key = next(
            (aid for aid, spec in control_config.get("actor_defs", {}).items()
             if str(spec.get("canonical_id", aid)) == target_id),
            target_id,
        )
        if target_key in control_config.get("env_behavior", {}):
            control_config["env_behavior"][target_key] = {
                "accel": 0.0, "lat": 0.0
            }
        control_trajectory, control_controllers = run_branch(
            control_config, control_name, world, api, output_dir,
            placement=placement)
        result["_l4_control_steps"] = control_trajectory
        result["_controllers"][control_name] = control_controllers
    if placement.get("waypoint") is not None:
        result["_placement"] = _placement_record(config, placement)
        if placement.get("route_evidence") is not None:
            result["_route_evidence"] = copy.deepcopy(
                placement["route_evidence"])
    return result


def main():
    import argparse
    from simulation.scenario_config import load_scenario_configs

    parser = argparse.ArgumentParser(description="Config-driven CARLA scenario runner")
    parser.add_argument("--scenarios-dir", default="scenarios")
    parser.add_argument("--output-dir", default="/workspace/out")
    parser.add_argument("--host", default="localhost")
    parser.add_argument("--port", type=int, default=2000)
    parser.add_argument("--only", nargs="*", help="scenario_ids to run (default: all)")
    args = parser.parse_args()

    api = _default_api()
    World = api["World"]
    configs = load_scenario_configs(args.scenarios_dir)
    if args.only:
        configs = [c for c in configs if c["scenario_id"] in args.only]

    world = None
    current_map = None
    final_output = []
    for i, config in enumerate(configs):
        map_name = config.get("map", "Town04")
        if world is None:
            world = World(host=args.host, port=args.port, timeout=30.0,
                          map_name=map_name, fps=FPS, step_length=STEP_LENGTH)
            world.setup()
            current_map = map_name
        elif map_name != current_map:
            world.change_map(map_name)
            current_map = map_name
        print(f"[{i + 1}/{len(configs)}] {config['scenario_id']} on {map_name}")
        raw = run_scenario(config, world, api, output_dir=args.output_dir)
        raw.pop("_controllers", None)
        final_output.append(raw)

    out_path = os.path.join(args.output_dir, "coriva_bench_examples.json")
    with open(out_path, "w") as f:
        json.dump(final_output, f, indent=4)
    print(f"[SUCCESS] {len(final_output)} scenarios -> {out_path}")


if __name__ == "__main__":
    main()
