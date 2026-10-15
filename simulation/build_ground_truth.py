"""
build_ground_truth.py — Generate per-scenario ground truth from raw simulation data.

Reads coriva_bench_examples.json (raw 0.1s telemetry) and produces
per-scenario ground_truth.json files in the CORIVA actor-array schema
that ctms.py expects.

Output format matches ctms.py evaluate_ctms() contract:
  - branches[]: each with chain[] of actor-array nodes
  - counterfactuals[]: one per non-factual intervention
  - Node format: {depth, timestamp, event_type, action_type, actors[], terminal?}
  - Actor format: {id, entity_type, role_type, x, y, vx, vy, speed}

Usage:
    python3 simulation/build_ground_truth.py \\
        --input coriva_bench_examples.json \\
        --output-dir output
"""

import json
import argparse
import os
import math
import copy

try:
    from simulation.protocol import (STOP_SPEED_MPS, CARLA_ROUTE_HALF_WIDTH_M, PRIMARY_TRACK,
                                     OBSERVED_FRAME_COUNT, track_steps,
                                     protocol_manifest)
except ImportError:  # direct script execution from Coding/simulation/
    from protocol import (STOP_SPEED_MPS, CARLA_ROUTE_HALF_WIDTH_M, PRIMARY_TRACK,  # noqa: F401
                          OBSERVED_FRAME_COUNT, track_steps,
                          protocol_manifest)

try:
    from simulation.scenario_config import load_scenario_configs
except ImportError:  # when run as a script from inside simulation/
    from scenario_config import load_scenario_configs


# ── Intervention → action/event mapping ──────────────────────────────
# Maps do_* intervention labels to (action_type, event_type)
INTERVENTION_TO_ACTION = {
    "do_ego_emergency_brake":      ("ego_emergency_brake",      "ego_brake"),
    "do_ego_comfortable_brake":    ("ego_comfortable_brake",    "ego_brake"),
    "do_ego_panic_brake":          ("ego_panic_brake",          "ego_brake"),
    "do_ego_swerve_left":          ("ego_swerve_left",          "ego_steer"),
    "do_ego_swerve_right":         ("ego_swerve_right",         "ego_steer"),
    "do_ego_brake_and_swerve_left":  ("ego_brake_and_swerve_left",  "ego_combined_maneuver"),
    "do_ego_brake_and_swerve_right": ("ego_brake_and_swerve_right", "ego_combined_maneuver"),
    "do_ego_accelerate_pass":      ("ego_accelerate_pass",      "ego_accelerate"),
    "do_ego_accelerate_escape":    ("ego_accelerate_escape",    "ego_accelerate"),
    "do_ego_no_action":            ("ego_no_action",            "maintain_course"),
    "do_ego_decelerate_coast":     ("ego_decelerate_coast",     "maintain_course"),
}

# ── Entity type alias → canonical ────────────────────────────────────
ENTITY_TYPE_MAP = {
    "ego":    "ego_vehicle",
    "car":    "vehicle",
    "sedan":  "vehicle",
    "walker": "pedestrian",
    "person": "pedestrian",
    "bike":   "bicycle",
}

_ACTOR_EXTENTS = {
    "ego_vehicle": (4.5, 2.0), "vehicle": (4.5, 2.0), "parked_vehicle": (4.5, 2.0),
    "car": (4.5, 2.0), "truck": (7.0, 2.5), "bus": (12.0, 2.5),
    "motorcycle": (2.2, 0.8), "bicycle": (1.8, 0.6),
    "pedestrian": (0.6, 0.6), "walker": (0.6, 0.6),
    "barrier": (1.5, 0.5), "traffic_cone": (0.4, 0.4),
    "guardrail": (1.5, 0.5), "traffic_sign": (0.6, 0.6),
}


# ═══════════════════════════════════════════
# SCENARIO CONFIGURATIONS
# ═══════════════════════════════════════════
SCENARIOS = [
    {
        "scenario_id": "high_speed_following",
        "scenario_index": 0,
        "output_subdir": "scenario_1",
        "scenario_type": "rear_end_collision_risk",
        "actor_defs": {
            "ego": {
                "canonical_id": "ego",
                "entity_type": "ego_vehicle",
                "default_role": "ego_agent",
            },
            "npc": {
                "canonical_id": "npc_1",
                "entity_type": "vehicle",
                "default_role": "lead_vehicle",
                "collision_role": "impacted_actor",
            },
        },
        "environment_event": {
            "event_type": "npc_sudden_brake",
            "trigger_time_sec": 3.5,
            "params": {"deceleration": 8.0}
        },
        "critical_actor_id": "npc_1",
        "l4_query": {"intervention_type": "npc_sudden_brake",
                     "target_actor_id": "npc_1", "trigger_depth": 1,
                     "ego_policy": "do_ego_no_action"},
        "intervention_time_sec": 3.5,
        "sample_interval_sec": 0.45,
        "d_max": 10,
        "total_duration_sec": 8.0,
        "branches": {
            "Factual_AEB": {
                "is_factual": True,
                "intervention_type": "do_ego_emergency_brake",
                "target_actor_id": "ego",
                "trigger_depth": 1,
            },
            "Counterfactual_Distracted": {
                "is_factual": False,
                "intervention_type": "do_ego_no_action",
                "target_actor_id": "ego",
                "trigger_depth": 1,
            },
            "Counterfactual_Swerve": {
                "is_factual": False,
                "intervention_type": "do_ego_swerve_left",
                "target_actor_id": "ego",
                "trigger_depth": 1,
            },
        },
    },
    {
        "scenario_id": "pedestrian_crossing",
        "scenario_index": 1,
        "output_subdir": "scenario_2",
        "scenario_type": "pedestrian_collision_risk",
        "actor_defs": {
            "ego": {
                "canonical_id": "ego",
                "entity_type": "ego_vehicle",
                "default_role": "ego_agent",
            },
            "walker": {
                "canonical_id": "npc_1",
                "entity_type": "pedestrian",
                "default_role": "crossing_actor",
                "collision_role": "impacted_actor",
            },
        },
        "environment_event": {
            "event_type": "pedestrian_crossing",
            "trigger_time_sec": 3.5,
            "params": {"crossing_speed": 2.0}
        },
        "critical_actor_id": "npc_1",
        "l4_query": {"intervention_type": "pedestrian_enters_road",
                     "target_actor_id": "npc_1", "trigger_depth": 1,
                     "ego_policy": "do_ego_no_action"},
        "intervention_time_sec": 3.0,
        "sample_interval_sec": 0.5,
        "d_max": 10,
        "total_duration_sec": 8.0,
        "branches": {
            "Factual_Yield": {
                "is_factual": True,
                "intervention_type": "do_ego_comfortable_brake",
                "target_actor_id": "ego",
                "trigger_depth": 1,
            },
            "Counterfactual_Panic_Brake": {
                "is_factual": False,
                "intervention_type": "do_ego_panic_brake",
                "target_actor_id": "ego",
                "trigger_depth": 1,
            },
            "Counterfactual_Swerve": {
                "is_factual": False,
                "intervention_type": "do_ego_swerve_left",
                "target_actor_id": "ego",
                "trigger_depth": 1,
            },
        },
    },
]


def find_closest_step(steps, target_time, tolerance=0.05):
    """Find the step closest to target_time within tolerance."""
    best = None
    best_diff = float('inf')
    for step in steps:
        diff = abs(step["time_sec"] - target_time)
        if diff < best_diff:
            best_diff = diff
            best = step
    if best_diff <= tolerance:
        return best
    return None


def find_step_at_or_before(steps, target_time):
    """Return the latest step with time_sec <= target_time; fall back to first step."""
    best = None
    for step in steps:
        if step["time_sec"] <= target_time + 1e-9:
            if best is None or step["time_sec"] > best["time_sec"]:
                best = step
    if best is not None:
        return best
    return steps[0] if steps else None


def _root_reference(scenario_raw, scenario_config):
    factual_name = next(
        name for name, meta in scenario_config["branches"].items() if meta["is_factual"])
    step = find_step_at_or_before(
        scenario_raw["branches"][factual_name], scenario_config["intervention_time_sec"])
    ego_key = next(key for key, actor in scenario_config["actor_defs"].items()
                   if actor.get("canonical_id") == "ego")
    ego = step[ego_key]
    yaw = (math.radians(float(ego["heading_deg"])) if ego.get("heading_deg") is not None
           else math.atan2(float(ego.get("vy", 0.0)), float(ego.get("vx", 0.0))))
    return float(ego["x"]), float(ego["y"]), yaw


def _normalise_raw_actor(raw_actor, reference):
    ox, oy, yaw = reference
    dx, dy = float(raw_actor["x"]) - ox, float(raw_actor["y"]) - oy
    c, s = math.cos(yaw), math.sin(yaw)
    out = dict(raw_actor)
    out["x"], out["y"] = round(c * dx + s * dy, 3), round(-s * dx + c * dy, 3)
    vx, vy = float(raw_actor.get("vx", 0.0)), float(raw_actor.get("vy", 0.0))
    out["vx"], out["vy"] = round(c * vx + s * vy, 3), round(-s * vx + c * vy, 3)
    return out


def _build_route_scene_geometry(scenario_raw, scenario_config, reference):
    """Build curved CARLA drivable evidence in the T0 ego frame.

    Protocol CARLA runs provide a branch-independent route sampled directly
    from the selected map waypoint. Fixtures without that evidence fall
    back to per-step waypoint telemetry. The corridor remains a bounded
    two-lane envelope derived from CARLA lane widths.
    """
    actor_defs = scenario_config.get("actor_defs", {})
    ego_key = next((key for key, spec in actor_defs.items()
                    if str(spec.get("canonical_id", key)) == "ego"), None)
    if ego_key is None or reference is None:
        return None

    route_evidence = scenario_raw.get("_route_evidence")
    if isinstance(route_evidence, dict):
        raw_lines = route_evidence.get("route_centerlines")
        if raw_lines is None:
            raw_line = route_evidence.get("route_centerline")
            raw_lines = [raw_line] if raw_line is not None else []
        route_lines = []
        for raw_line in raw_lines or []:
            line = []
            for raw_point in raw_line or []:
                try:
                    if route_evidence.get("coordinate_frame") == "t0_ego":
                        point = [round(float(raw_point[0]), 3),
                                 round(float(raw_point[1]), 3)]
                    else:
                        normalized = _normalise_raw_actor({
                            "x": raw_point[0], "y": raw_point[1],
                        }, reference)
                        point = [round(float(normalized["x"]), 3),
                                 round(float(normalized["y"]), 3)]
                except (TypeError, ValueError, IndexError):
                    continue
                if not line or math.hypot(
                        point[0] - line[-1][0], point[1] - line[-1][1]) >= 0.05:
                    line.append(point)
            if len(line) >= 2:
                route_lines.append(line)
        if route_lines:
            geometry = {
                "schema_version": 1,
                "coordinate_frame": "t0_ego",
                "source": str(route_evidence.get(
                    "source", "carla_map_waypoint_route")),
                "route_centerlines": route_lines,
                "route_half_width_m": round(float(route_evidence.get(
                    "route_half_width_m", CARLA_ROUTE_HALF_WIDTH_M)), 3),
            }
            for key in ("route_id", "route_mode", "route_coverage_m",
                        "route_required_m", "route_backward_coverage_m",
                        "route_backward_required_m", "sample_spacing_m"):
                if route_evidence.get(key) is not None:
                    value = route_evidence[key]
                    geometry[key] = (value if key in {"route_id", "route_mode"}
                                     else float(value))
            if (geometry.get("route_coverage_m") is not None
                    and geometry.get("route_required_m") is not None):
                geometry["route_coverage_complete"] = (
                    geometry["route_coverage_m"]
                    >= geometry["route_required_m"])
            if (geometry.get("route_backward_coverage_m") is not None
                    and geometry.get("route_backward_required_m") is not None):
                geometry["route_backward_coverage_complete"] = (
                    geometry["route_backward_coverage_m"]
                    >= geometry["route_backward_required_m"])
            return geometry

    # Compatibility fallback for existing fixtures and older telemetry. These
    # lines are map-waypoint observations, but unlike the protocol route above
    # their coverage can depend on the branch trajectory.
    route_lines = []
    lane_widths = []
    seen_lines = set()
    for steps in (scenario_raw.get("branches") or {}).values():
        line = []
        for step in steps or []:
            ego = step.get(ego_key)
            if not isinstance(ego, dict):
                continue
            if ego.get("road_center_x") is None or ego.get("road_center_y") is None:
                continue
            normalized = _normalise_raw_actor({
                "x": ego["road_center_x"],
                "y": ego["road_center_y"],
            }, reference)
            point = [round(float(normalized["x"]), 3),
                     round(float(normalized["y"]), 3)]
            if not line or math.hypot(
                    point[0] - line[-1][0], point[1] - line[-1][1]) >= 0.05:
                line.append(point)
            try:
                lane_width = float(ego.get("road_lane_width", 0.0))
            except (TypeError, ValueError):
                lane_width = 0.0
            if lane_width > 0.0:
                lane_widths.append(lane_width)
        key = tuple((point[0], point[1]) for point in line)
        if len(line) >= 2 and key not in seen_lines:
            route_lines.append(line)
            seen_lines.add(key)
    if not route_lines:
        return None
    lane_width = max(lane_widths, default=3.5)
    return {
        "schema_version": 1,
        "coordinate_frame": "t0_ego",
        "source": "carla_waypoint_route",
        "route_centerlines": route_lines,
        "route_half_width_m": round(min(12.0, max(8.0, 2.0 * lane_width)), 3),
        "route_coverage_complete": None,
    }


def _extent(actor_def):
    entity = actor_def.get("entity_type", "vehicle")
    length, width = _ACTOR_EXTENTS.get(entity, (4.5, 2.0))
    return float(actor_def.get("length", length)), float(actor_def.get("width", width))


def _camera_visible(raw_actor, raw_ego, fov_deg=100.0):
    """Return visibility from the projected CARLA box, with a geometric
    fallback when the telemetry carries no box."""
    bbox = raw_actor.get("bbox_2d")
    if isinstance(bbox, (list, tuple)) and len(bbox) == 4:
        x1, y1, x2, y2 = (float(value) for value in bbox)
        return x2 > x1 and y2 > y1 and (x2 - x1) * (y2 - y1) >= 100.0
    dx = float(raw_actor["x"]) - float(raw_ego["x"])
    dy = float(raw_actor["y"]) - float(raw_ego["y"])
    yaw = (math.radians(float(raw_ego["heading_deg"]))
           if raw_ego.get("heading_deg") is not None
           else math.atan2(float(raw_ego.get("vy", 0.0)), float(raw_ego.get("vx", 0.0))))
    c, s = math.cos(yaw), math.sin(yaw)
    # Telemetry without a projected box.  Approximate the windshield
    # camera, whose optical centre is 1.5 m ahead of the ego origin.
    # Protocol-1.0 releases use the real projected CARLA box above.
    forward = c * dx + s * dy - 1.5
    lateral = -s * dx + c * dy
    return forward > 0.5 and abs(math.degrees(math.atan2(lateral, forward))) <= fov_deg / 2.0


def build_observed_inputs(scenario_raw, scenario_config):
    """Return the closed T0 root and the exact eight-record numeric history."""
    from simulation.protocol import HISTORY_TIMESTAMPS

    actor_defs = scenario_config["actor_defs"]
    factual_name = next(name for name, meta in scenario_config["branches"].items()
                        if meta["is_factual"])
    steps = scenario_raw["branches"][factual_name]
    t0 = float(scenario_config["intervention_time_sec"])
    selected = []
    for offset in HISTORY_TIMESTAMPS:
        step = find_closest_step(steps, round(t0 + offset, 2), tolerance=0.06)
        if step is None:
            raise ValueError(f"CARLA history is missing T={offset:.1f}s")
        selected.append(step)
    reference = _root_reference(scenario_raw, scenario_config)
    ego_key = next(key for key, actor in actor_defs.items()
                   if actor.get("canonical_id") == "ego")
    complete = [key for key in actor_defs if all(key in step for step in selected)]
    visible = {key: [_camera_visible(step[key], step[ego_key]) for step in selected]
               for key in complete if key != ego_key}
    allowed = [ego_key] + [key for key in complete if key != ego_key and any(visible[key])]
    critical_id = scenario_config.get("critical_actor_id")
    if not critical_id:
        critical_id = next((spec.get("canonical_id") for key, spec in actor_defs.items()
                            if key != ego_key and spec.get("collision_role")), None)
    critical_key = next((key for key, spec in actor_defs.items()
                         if spec.get("canonical_id") == critical_id), None)
    if critical_key and (critical_key not in allowed or not all(visible.get(critical_key, []))):
        raise ValueError("critical actor must be visible in all eight CARLA frames")
    records = []
    for index, step in enumerate(selected):
        actors = []
        for key in allowed:
            actor = build_actor(_normalise_raw_actor(step[key], reference), actor_defs[key])
            actor["visibility"] = False if key == ego_key else bool(visible[key][index])
            actors.append(actor)
        records.append({"timestamp_sec": HISTORY_TIMESTAMPS[index], "actors": actors})
    root_actors = []
    for actor in records[-1]["actors"]:
        item = copy.deepcopy(actor)
        item.pop("visibility", None)
        root_actors.append(item)
    root = {"depth": 0, "timestamp": 0.0, "event_type": "observed_state",
            "actors": root_actors}
    return root, {"schema_version": 1, "coordinate_frame": "t0_ego", "records": records}, reference


def build_observed_state(scenario_raw, scenario_config):
    """Build the shared T=0 root snapshot captured just before the intervention."""
    actor_defs = scenario_config["actor_defs"]
    factual_name = next(
        name for name, meta in scenario_config["branches"].items() if meta["is_factual"])
    step = find_step_at_or_before(
        scenario_raw["branches"][factual_name], scenario_config["intervention_time_sec"])
    if step is None:
        raise ValueError(f"no telemetry steps for factual branch {factual_name!r}")
    actors = [build_actor(step[key], actor_def) for key, actor_def in actor_defs.items()
              if key in step]
    return {"depth": 0, "timestamp": step["time_sec"],
            "event_type": "observed_state", "actors": actors}


class ReferenceBuildError(ValueError):
    """The raw rollout cannot be turned into a valid declared reference."""


# Tick-to-sample tolerance: raw telemetry is recorded on a 0.1 s grid.
COLLISION_TIME_TOLERANCE = 0.05

# Footprints used by the runtime CollisionDetector (simulation/carla wrappers),
# keyed by config entity_type. These are the extents the detector actually
# compared, so partner reconstruction from raw without recorded partners must
# use them, not the release extents in _ACTOR_EXTENTS (traffic_sign differs).
_DETECTOR_EXTENTS = {
    "ego_vehicle": (4.5, 2.0), "vehicle": (4.5, 2.0), "car": (4.5, 2.0),
    "parked_vehicle": (4.5, 2.0), "truck": (7.0, 2.5), "van": (6.0, 2.3),
    "bus": (12.0, 2.5), "motorcycle": (2.2, 0.8), "bicycle": (1.8, 0.6),
    "pedestrian": (0.6, 0.6), "walker": (0.6, 0.6),
    "guardrail": (1.5, 0.5), "road_boundary": (1.5, 0.5), "barrier": (1.5, 0.5),
    "traffic_sign": (1.5, 0.5), "traffic_cone": (0.4, 0.4),
}


def _ego_key(actor_defs):
    return next((key for key, spec in actor_defs.items()
                 if spec.get("entity_type") in {"ego", "ego_vehicle"}
                 or spec.get("canonical_id") == "ego"), None)


def find_collision_step(steps):
    """First raw step flagged as a collision, or None."""
    for step in steps:
        if step.get("collision", False):
            return step
    return None


def reconstruct_collision_partners(step, actor_defs):
    """Actors overlapping the ego footprint in this raw step, in config order.

    Mirrors CollisionDetector.check(): axis-aligned in the ego heading frame,
    per-axis threshold = sum of half extents, distances rounded to 3 dp and
    compared with '<'. Used only for raw without recorded partners.
    """
    ego_key = _ego_key(actor_defs)
    ego = step.get(ego_key) if ego_key else None
    if not ego:
        return []
    yaw = math.radians(float(ego.get("heading_deg", 0.0)))
    fx, fy = math.cos(yaw), math.sin(yaw)
    ego_len, ego_wid = _DETECTOR_EXTENTS.get(
        actor_defs[ego_key].get("entity_type"), (4.5, 2.0))
    hits = []
    for key, spec in actor_defs.items():
        if key == ego_key or key not in step:
            continue
        other = step[key]
        dx = float(ego["x"]) - float(other["x"])
        dy = float(ego["y"]) - float(other["y"])
        forward = round(abs(dx * fx + dy * fy), 3)
        lateral = round(abs(-dx * fy + dy * fx), 3)
        length, width = _DETECTOR_EXTENTS.get(spec.get("entity_type"), (4.5, 2.0))
        if (forward < (ego_len + length) / 2.0
                and lateral < (ego_wid + width) / 2.0):
            hits.append(key)
    return hits


def collision_partner_keys(step, actor_defs):
    """Raw actor keys the ego collided with in this step.

    Prefers the partners the recorder stored at the hit; falls back to
    footprint reconstruction for raw without recorded partners.
    """
    recorded = step.get("collision_partners")
    if recorded is not None:
        return [str(key) for key in recorded]
    return reconstruct_collision_partners(step, actor_defs)


def scored_collision(steps, actor_defs, horizon_end, root_actor_ids=None):
    """(collision_time, partner_keys) for the first raw collision at or before
    `horizon_end`, else None. Only such a collision is a reference event.

    Raises ReferenceBuildError when the partner cannot be resolved or, if
    `root_actor_ids` is given, when a partner is outside the observed root.
    """
    step = find_collision_step(steps)
    if step is None:
        return None
    collision_time = float(step.get("collision_time", step["time_sec"]))
    if collision_time > horizon_end + COLLISION_TIME_TOLERANCE:
        return None
    partner_keys = collision_partner_keys(step, actor_defs)
    if not partner_keys:
        raise ReferenceBuildError(
            f"collision at {collision_time:.2f}s has no resolvable partner")
    if root_actor_ids is not None:
        hidden = [actor_defs[key]["canonical_id"] for key in partner_keys
                  if actor_defs[key]["canonical_id"] not in root_actor_ids]
        if hidden:
            raise ReferenceBuildError(
                f"collision at {collision_time:.2f}s with actor(s) "
                f"{hidden} absent from the observed root")
    return collision_time, partner_keys


STATIC_ENTITY_TYPES = {"parked_vehicle", "guardrail", "barrier", "traffic_cone",
                       "traffic_sign", "road_edge", "curb", "pole"}


def _footprint_corners(x, y, yaw_deg, length, width):
    c, s = math.cos(math.radians(yaw_deg)), math.sin(math.radians(yaw_deg))
    hl, hw = length / 2.0, width / 2.0
    return [(x + c * dx - s * dy, y + s * dx + c * dy)
            for dx, dy in ((hl, hw), (hl, -hw), (-hl, -hw), (-hl, hw))]


def _footprints_overlap(a, b):
    """Separating-axis test for two oriented rectangles given as corner lists."""
    for poly in (a, b):
        for i in range(4):
            ex, ey = poly[(i + 1) % 4][0] - poly[i][0], poly[(i + 1) % 4][1] - poly[i][1]
            nx, ny = -ey, ex
            pa = [px * nx + py * ny for px, py in a]
            pb = [px * nx + py * ny for px, py in b]
            if max(pa) < min(pb) or max(pb) < min(pa):
                return False
    return True


def _state_yaw(state, fallback_yaw):
    heading = state.get("heading_deg")
    if heading is not None:
        return float(heading)
    vx, vy = float(state.get("vx", 0.0) or 0.0), float(state.get("vy", 0.0) or 0.0)
    if math.hypot(vx, vy) > 0.1:
        return math.degrees(math.atan2(vy, vx))
    return fallback_yaw


def static_pass_through(raw, actor_defs, substeps=2):
    """Every point of the raw rollouts at which a moving non-ego actor's
    declared body rectangle (length x width, at its recorded heading) overlaps
    a static actor's rectangle (parked vehicle or prop, at its recorded spawn
    heading; a prop without one is taken at world yaw 0). Movers are also
    interpolated linearly between consecutive ticks (``substeps`` positions
    per 0.1 s tick) so a body cannot jump over a static actor between two
    samples. CARLA NPCs run with physics off and the collision detector
    watches only the ego, so nothing else records such a pass-through. Scans
    every L3 branch and the L4 treated/control rollouts. Returns dicts with
    branch, time_sec, mover, static (canonical ids) and distance_m (centre
    distance)."""
    static_keys = [k for k, s in actor_defs.items()
                   if s.get("entity_type") in STATIC_ENTITY_TYPES]
    ego_key = _ego_key(actor_defs)
    mover_keys = [k for k, s in actor_defs.items()
                  if k not in static_keys and k != ego_key
                  and s.get("entity_type") not in ("ego", "ego_vehicle")]
    streams = list((raw.get("branches") or {}).items())
    for key in ("_l4_steps", "_l4_control_steps"):
        if raw.get(key):
            streams.append((key, raw[key]))
    hits = []
    for name, steps in streams:
        prev = None
        for step in steps or []:
            ego_yaw = _state_yaw(step.get(ego_key) or {}, 0.0) if ego_key else 0.0
            statics = []
            for s in static_keys:
                b = step.get(s)
                if not b or b.get("x") is None:
                    continue
                sl, sw = _extent(actor_defs[s])
                # A static actor's heading is its spawn yaw; props that do
                # not report one are taken at yaw 0 in world axes.
                statics.append((s, b, _footprint_corners(
                    float(b["x"]), float(b["y"]), _state_yaw(b, 0.0), sl, sw)))
            for m in mover_keys:
                a = step.get(m)
                if not a or a.get("x") is None:
                    continue
                ml, mw = _extent(actor_defs[m])
                yaw = _state_yaw(a, ego_yaw)
                a0 = (prev or {}).get(m) if prev else None
                fractions = ([k / substeps for k in range(1, substeps + 1)]
                             if a0 and a0.get("x") is not None else [1.0])
                for f in fractions:
                    if f < 1.0:
                        x = float(a0["x"]) + f * (float(a["x"]) - float(a0["x"]))
                        y = float(a0["y"]) + f * (float(a["y"]) - float(a0["y"]))
                        t = float(prev.get("time_sec", 0.0)) + f * (
                            float(step.get("time_sec", 0.0)) - float(prev.get("time_sec", 0.0)))
                    else:
                        x, y, t = float(a["x"]), float(a["y"]), float(step.get("time_sec", 0.0))
                    corners = _footprint_corners(x, y, yaw, ml, mw)
                    for s, b, s_corners in statics:
                        if _footprints_overlap(corners, s_corners):
                            hits.append(dict(
                                branch=name, time_sec=round(t, 2),
                                mover=actor_defs[m].get("canonical_id", m),
                                static=actor_defs[s].get("canonical_id", s),
                                distance_m=round(math.hypot(x - float(b["x"]), y - float(b["y"])), 2)))
                            break
            prev = step
    return hits


MOVING_VEHICLE_TYPES = {"vehicle", "car", "truck", "van", "bus", "motorcycle", "bicycle"}


def lane_evidence_gaps(raw, actor_defs):
    """Messages describing every way the placement's lane evidence
    (``raw['_placement']['moving_actor_lane_checks']``) fails to cover a
    moving vehicle with a lateral slot: entry missing or unstructured, class
    not ``driving_same``, non-finite or inverted span/budget, budget below
    the path the actor actually drove in some rollout, a tick of the actor's
    telemetry without a finite ``route_s``, or occupied route coordinates
    outside the checked span. Empty when the evidence covers every rollout
    (L3 branches, L4 treated, L4 control). Used before a scenario is
    published (run_batch) and again at release (verify_carla_dataset)."""
    checks = ((raw.get("_placement") or {}).get("moving_actor_lane_checks") or {})
    streams = list((raw.get("branches") or {}).items())
    for extra in ("_l4_steps", "_l4_control_steps"):
        if raw.get(extra):
            streams.append((extra, raw[extra]))
    gaps = []
    for key, spec in (actor_defs or {}).items():
        if spec.get("entity_type") not in MOVING_VEHICLE_TYPES:
            continue
        if spec.get("canonical_id") == "ego" or float(spec.get("lateral_offset", 0.0)) == 0.0:
            continue
        if spec.get("default_role") == "crossing_actor":
            continue  # a road-entry cyclist starts on the pavement by design
        label = f"moving actor {key} (lateral {float(spec.get('lateral_offset', 0.0)):g} m)"
        entry = checks.get(key)
        if not isinstance(entry, dict):
            gaps.append(f"{label}: lane evidence missing or not structured ({entry!r}); "
                        "release requires {class, span_m, travel_m}")
            continue
        cls, span, travel = entry.get("class"), entry.get("span_m"), entry.get("travel_m")
        if cls != "driving_same":
            gaps.append(f"{label}: lane evidence class {cls!r}; release requires 'driving_same'")
        if not isinstance(travel, (int, float)) or not math.isfinite(float(travel)):
            gaps.append(f"{label}: lane evidence has no finite travel_m budget ({travel!r})")
            continue
        span_ok = (isinstance(span, list) and len(span) == 2
                   and all(isinstance(v, (int, float)) and math.isfinite(float(v)) for v in span)
                   and float(span[1]) >= float(span[0]))
        if not span_ok:
            gaps.append(f"{label}: lane evidence span_m {span!r} is not a finite [start, end] route span")
        for name, steps in streams:
            path, prev, route_s, bad_ticks = 0.0, None, [], 0
            for step in steps or []:
                state = step.get(key)
                if not state or state.get("x") is None:
                    continue
                if prev is not None:
                    path += math.hypot(float(state["x"]) - prev[0], float(state["y"]) - prev[1])
                prev = (float(state["x"]), float(state["y"]))
                s_val = state.get("route_s")
                if isinstance(s_val, (int, float)) and math.isfinite(float(s_val)):
                    route_s.append(float(s_val))
                else:
                    bad_ticks += 1
            if path > float(travel) + 0.5:
                gaps.append(f"{label}: travelled {path:.1f} m in {name} but the lane "
                            f"evidence budget is only {float(travel):.1f} m")
            if bad_ticks:
                gaps.append(f"{label}: {bad_ticks} tick(s) in {name} without a finite route_s; "
                            "the lane evidence cannot be tied to the road actually used")
            if route_s and span_ok and (float(span[0]) > min(route_s) + 0.5
                                        or float(span[1]) < max(route_s) - 0.5):
                gaps.append(f"{label}: occupied route {min(route_s):.1f}-{max(route_s):.1f} m "
                            f"in {name} but the lane evidence checked only "
                            f"{float(span[0]):.1f}-{float(span[1]):.1f} m")
    return gaps


def _ego_stopped(step, actor_defs):
    """Raw ego at or below the shared stop threshold (STOP_SPEED_MPS, 0.5 m/s
    = 1.8 km/h; the same constant the twin and the prompt use)."""
    ego_key = _ego_key(actor_defs)
    ego = step.get(ego_key, {}) if ego_key is not None else {}
    return bool(ego) and ego.get("speed_kmh", float("inf")) <= STOP_SPEED_MPS * 3.6 + 1e-9


def _ego_stays_stopped(steps, from_time, horizon_end, actor_defs):
    """True iff the ego is stopped at every raw step from `from_time` through
    `horizon_end`: a stop that is followed by motion inside the horizon is
    not a `safe_stop` terminal."""
    if not steps or steps[-1]["time_sec"] < horizon_end - COLLISION_TIME_TOLERANCE:
        return False  # the raw does not cover the horizon: no evidence of holding
    window = [s for s in steps
              if from_time - COLLISION_TIME_TOLERANCE <= s["time_sec"] <= horizon_end + COLLISION_TIME_TOLERANCE]
    return bool(window) and all(_ego_stopped(s, actor_defs) for s in window)


def build_actor(raw_actor, actor_def, impacted=False):
    """Convert a raw flat actor dict to CORIVA actor-array format.

    `impacted` marks the actor the ego actually collided with in this node."""
    vx = raw_actor.get("vx", 0.0)
    vy = raw_actor.get("vy", 0.0)

    # Prefer computing speed from velocity components for consistency
    speed_from_v = math.sqrt(vx ** 2 + vy ** 2)
    # Fallback to speed_kmh conversion if vx/vy are absent
    if vx == 0.0 and vy == 0.0:
        speed_kmh = raw_actor.get("speed_kmh", 0.0)
        speed = round(speed_kmh / 3.6, 3)
    else:
        speed = round(speed_from_v, 3)

    role = "impacted_actor" if impacted else actor_def["default_role"]

    length, width = _extent(actor_def)
    return {
        "id": actor_def["canonical_id"],
        "entity_type": actor_def["entity_type"],
        "role_type": role,
        "x": raw_actor["x"],
        "y": raw_actor["y"],
        "vx": round(vx, 1),
        "vy": round(vy, 1),
        "speed": speed,
        "length": length,
        "width": width,
    }


def build_chain(branch_steps, scenario_config, branch_meta, track=PRIMARY_TRACK,
                reference=None, root_actor_ids=None, protocol=False):
    """Build the chain[] array from raw simulation steps."""
    actor_defs = scenario_config["actor_defs"]
    intervention_time = scenario_config["intervention_time_sec"]
    sample_interval = 0.5 if protocol else scenario_config["sample_interval_sec"]
    # The release contract is 3 s at 2 Hz.  Keep the scenario's longer raw
    # rollout for diagnostics, but clip scored CARLA chains to six nodes.
    d_max = min(int(scenario_config["d_max"]), track_steps(track))
    intervention_type = branch_meta["intervention_type"]

    # Resolve action_type and event_type from intervention
    action_type, default_event_type = INTERVENTION_TO_ACTION.get(
        intervention_type, ("ego_no_action", "maintain_course")
    )

    # Only a collision inside the scored horizon is a reference event. The
    # runner stops recording at the hit, so a later collision simply means
    # the branch reaches the horizon (a boundary, not a terminal).
    horizon_end = round(intervention_time + d_max * sample_interval, 2)
    hit = scored_collision(branch_steps, actor_defs, horizon_end, root_actor_ids)
    collision_time, partner_keys = hit if hit else (None, [])

    chain = []
    for depth in range(1, d_max + 1):
        target_time = round(intervention_time + depth * sample_interval, 2)

        if branch_steps and target_time > branch_steps[-1]["time_sec"] + COLLISION_TIME_TOLERANCE:
            break
        if collision_time is not None and target_time > collision_time + COLLISION_TIME_TOLERANCE:
            break

        step = find_closest_step(branch_steps, target_time)
        if step is None:
            break

        next_time = round(intervention_time + (depth + 1) * sample_interval, 2)
        # Collision is emitted at the last sample at/before the recorded hit
        # (the raw has no post-impact state); the twin uses the same
        # convention. safe_stop is emitted only when the
        # ego is stopped in this sample AND stays stopped through the end of
        # the horizon AND no collision follows within the horizon; a stop
        # followed by motion, or a stopped ego that is hit later, is not a
        # safe outcome.
        collision_terminal = (
            collision_time is not None
            and next_time > collision_time + COLLISION_TIME_TOLERANCE)
        stopped_terminal = (collision_time is None and _ego_stopped(step, actor_defs)
                            and _ego_stays_stopped(branch_steps, target_time, horizon_end, actor_defs))
        if collision_terminal:
            node_event_type = "collision"
        elif stopped_terminal:
            node_event_type = "safe_stop"
        else:
            node_event_type = default_event_type

        actors = []
        for raw_key, actor_def in actor_defs.items():
            if root_actor_ids is not None and actor_def["canonical_id"] not in root_actor_ids:
                continue
            raw_actor = step.get(raw_key)
            if raw_actor is None:
                continue
            impacted = collision_terminal and raw_key in partner_keys
            normalized = _normalise_raw_actor(raw_actor, reference) if reference else raw_actor
            actors.append(build_actor(normalized, actor_def, impacted=impacted))

        node = {
            "depth": depth,
            "timestamp": round(depth * sample_interval, 2) if protocol else target_time,
            "event_type": node_event_type,
        }

        if node_event_type not in ("collision",):
            node["action_type"] = action_type

        if node_event_type in ("collision", "safe_stop", "out_of_bounds"):
            node["terminal"] = True

        node["actors_update"] = actors
        chain.append(node)
        if node.get("terminal") is True:
            break

    return chain


def build_branch_gt(branch_name, branch_steps, scenario_config, track=PRIMARY_TRACK,
                    reference=None, root_actor_ids=None, protocol=False):
    """Build a single branch's GT data from raw steps."""
    branch_meta = scenario_config["branches"][branch_name]

    chain = build_chain(branch_steps, scenario_config, branch_meta, track=track,
                        reference=reference, root_actor_ids=root_actor_ids,
                        protocol=protocol)

    return {
        "branch_id": branch_name,
        "is_factual": branch_meta["is_factual"],
        "intervention": {
            "intervention_type": branch_meta["intervention_type"],
            "target_actor_id": branch_meta["target_actor_id"],
            "trigger_depth": branch_meta["trigger_depth"],
        },
        "chain": chain,
    }


def build_scenario_gt(scenario_raw, scenario_config, track=PRIMARY_TRACK,
                      protocol: bool = False):
    """Build full GT for one scenario."""

    from simulation.kinematic_twin import build_twin_branch, ENV_PROFILES
    if protocol:
        observed_state, observed_history, reference = build_observed_inputs(
            scenario_raw, scenario_config)
    else:
        observed_state, observed_history, reference = (
            build_observed_state(scenario_raw, scenario_config), None, None)
    root_actor_ids = {actor["id"] for actor in observed_state["actors"]}
    # Build branches (L3)
    branches = []
    for branch_name, branch_steps in scenario_raw["branches"].items():
        if branch_name not in scenario_config["branches"]:
            print(f"  [WARN] Branch '{branch_name}' not in config, skipping")
            continue
        branch_gt = build_branch_gt(
            branch_name, branch_steps, scenario_config, track=track,
            reference=reference, root_actor_ids=root_actor_ids, protocol=protocol)
        branches.append(branch_gt)

    # Build counterfactuals (L4): one entry per non-factual branch
    counterfactuals = []
    for branch_gt in branches:
        if branch_gt["is_factual"]:
            continue
        counterfactuals.append({
            "intervention_type": branch_gt["intervention"]["intervention_type"],
            "counterfactual_chain": {
                "intervention": branch_gt["intervention"],
                "chain": branch_gt["chain"],
            },
        })

    result = {
        "scenario_id": scenario_config["scenario_id"],
        "scenario_type": scenario_config["scenario_type"],
        "observed_state": observed_state,
        "environment_event": scenario_config["environment_event"],
        "intervention_time_sec": scenario_config["intervention_time_sec"],
        "sample_interval_sec": 0.5 if protocol else scenario_config["sample_interval_sec"],
        "d_max": scenario_config["d_max"],
        "total_duration_sec": scenario_config["total_duration_sec"],
        "branches": branches,
        "counterfactuals": counterfactuals,
    }
    if observed_history is not None:
        result["_observed_history"] = observed_history
    scene_geometry = _build_route_scene_geometry(
        scenario_raw, scenario_config, reference)
    if scene_geometry is not None:
        result["scene_geometry"] = scene_geometry
    # CARLA and nuScenes share the same scored horizon and versioned contract.
    # The CARLA recorder still keeps its full rollout for diagnostics, while
    # the released GT is clipped to the selected track by the branch builder.
    result["protocol"] = (protocol_manifest(
        track=track, split="carla", dataset="carla") if protocol else {
            "protocol_version": "legacy",
            "track": track,
            "future_steps": track_steps(track),
            "future_horizon_seconds": track_steps(track) * 0.5,
            "observed_frame_count": OBSERVED_FRAME_COUNT,
            "dataset": "carla",
            "split": "carla",
        })
    # CARLA has an explicit scenario event and a known primary actor. Final
    # protocol builds use a separate simulator rollout with a fixed ego policy;
    # this is exact CARLA ground truth and never enters the Layer-3 branch set.
    event_type = (scenario_config.get("environment_event") or {}).get("event_type")
    critical_id = scenario_config.get("critical_actor_id")
    if not critical_id:
        critical_id = next((spec.get("canonical_id") for key, spec in
                            scenario_config.get("actor_defs", {}).items()
                            if key != "ego" and spec.get("collision_role")), None)
    configured_query = scenario_config.get("l4_query") or {}
    if protocol and not scenario_raw.get("_l4_steps"):
        raise ValueError("protocol CARLA scenario is missing the fixed-policy L4 rollout")
    if scenario_raw.get("_l4_steps") and critical_id:
        query = {
            "intervention_type": configured_query.get("intervention_type", event_type),
            "target_actor_id": configured_query.get("target_actor_id", critical_id),
            "trigger_depth": int(configured_query.get("trigger_depth", 1)),
        }
        policy = configured_query.get("ego_policy", "do_ego_no_action")
        l4_chain = build_chain(
            scenario_raw["_l4_steps"], scenario_config,
            {"intervention_type": policy}, track=track, reference=reference,
            root_actor_ids=root_actor_ids, protocol=protocol)
        result["l4_queries"] = [{
            "intervention_type": query["intervention_type"],
            "counterfactual_chain": {
                "intervention": query,
                "chain": l4_chain,
            },
        }]
        result["protocol"].update({
            "critical_actor_id": critical_id,
            "intervention_type": query["intervention_type"],
            "l4_ego_policy": policy,
            "l4_reference_source": "carla_fixed_policy_rollout",
        })
        # Matched-pair protocol: the paired control rollout (same fixed ego
        # policy, target on its baseline) is built by the very same chain
        # rule and published next to the treated chain.
        if scenario_raw.get("_l4_control_steps"):
            from simulation.l4_control import control_query_entry, mark_pair_protocol
            control_chain = build_chain(
                scenario_raw["_l4_control_steps"], scenario_config,
                {"intervention_type": policy}, track=track, reference=reference,
                root_actor_ids=root_actor_ids, protocol=protocol)
            result["l4_queries"][0]["control_chain"] = control_query_entry(query, control_chain, "carla")
            mark_pair_protocol(result["protocol"], "carla")
        elif protocol:
            raise ValueError("protocol CARLA scenario is missing the paired L4 control rollout")
    elif not protocol and event_type in ENV_PROFILES and critical_id:
        query = {"intervention_type": event_type,
                 "target_actor_id": critical_id, "trigger_depth": 1}
        l4_branch = build_twin_branch(
            "L4_LegacyTwin", False, event_type, observed_state,
            track_steps(track), 0.5, 0.5, model="point_mass",
            environment_intervention=query, intervention_meta=query)
        result["l4_queries"] = [{"intervention_type": event_type,
                                 "counterfactual_chain": {
                                     "intervention": query,
                                     "chain": l4_branch["chain"]}}]
    declared = list(dict.fromkeys(
        b.get("intervention_type") for b in branches if b.get("intervention_type")))
    result["action_audit"] = {
        "declared_actions": declared,
        "enumerated_count": len(branches),
        "deduplicated": len(declared) != len(branches),
        "gate": "qc_scenario_gt",
    }
    return result


def resolve_scenarios(scenarios_dir="scenarios"):
    """Return scenario configs from the unified YAML source if present, else the
    in-code SCENARIOS fallback. The YAML configs are the single source of truth;
    SCENARIOS remains as a fallback and as the reference the configs were ported from."""
    if os.path.isdir(scenarios_dir):
        configs = load_scenario_configs(scenarios_dir)
        if configs:
            return configs
    return SCENARIOS


def main():
    parser = argparse.ArgumentParser(
        description='Build per-scenario ground truth from raw simulation data')
    parser.add_argument('--input', required=True,
                        help='Path to coriva_bench_examples.json')
    parser.add_argument('--output-dir', required=True,
                        help='Output directory (output)')
    parser.add_argument('--scenarios-dir', default='scenarios',
                        help='Directory of unified scenario YAML configs')
    args = parser.parse_args()
    scenarios = resolve_scenarios(args.scenarios_dir)

    with open(args.input) as f:
        raw_data = json.load(f)

    print(f"Loaded {len(raw_data)} scenarios from {args.input}")

    for scenario_config in scenarios:
        idx = scenario_config["scenario_index"]
        if idx >= len(raw_data):
            print(f"  [WARN] Scenario index {idx} not found in data, skipping")
            continue

        scenario_raw = raw_data[idx]
        subdir = scenario_config["output_subdir"]

        print(f"\n  Building GT for '{scenario_config['scenario_id']}' ({subdir})...")
        gt = build_scenario_gt(scenario_raw, scenario_config)
        observed_history = gt.pop("_observed_history", None)

        out_path = os.path.join(args.output_dir, subdir, "ground_truth.json")
        os.makedirs(os.path.dirname(out_path), exist_ok=True)

        with open(out_path, 'w') as f:
            json.dump(gt, f, indent=2)
        if observed_history is not None:
            from simulation.observed_history import write_observed_artifacts
            write_observed_artifacts(os.path.dirname(out_path), gt["observed_state"], observed_history)

        n_branches = len(gt["branches"])
        n_cfs = len(gt["counterfactuals"])
        print(f"  ✓ {out_path}")
        print(f"    {n_branches} branches, {n_cfs} counterfactuals")
        for b in gt["branches"]:
            last = b["chain"][-1] if b["chain"] else {}
            term = last.get("event_type", "?")
            print(f"    - {b['branch_id']}: {len(b['chain'])} nodes, terminal={term}")


if __name__ == '__main__':
    main()
