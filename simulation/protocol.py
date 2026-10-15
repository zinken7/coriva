"""Versioned CORIVA dataset/input protocol.

This module is deliberately dependency-light so builders, runners and reports
share one contract instead of silently carrying split-specific defaults.
"""

from __future__ import annotations

from typing import Any, Dict

PROTOCOL_VERSION = "coriva-protocol-1.0"
COORDINATE_FRAME = "t0_ego"
COORDINATE_CONVENTION = {
    "origin": "ego_position_at_t0",
    "x_axis": "positive_forward_at_t0",
    "y_axis": "positive_right_at_t0",
    "velocity_frame": "fixed_t0_ego_axes",
}
PRIMARY_TRACK = "primary"
CHALLENGE_TRACK = "challenge_6s"
TRACK_STEPS = {PRIMARY_TRACK: 6, CHALLENGE_TRACK: 12}
TRACK_DT = 0.5
# Ego speed at or below which it counts as stopped for `safe_stop`, in both
# references (twin samples in m/s; CARLA raw speed_kmh compared against
# STOP_SPEED_MPS * 3.6 = 1.8 km/h) and in the model prompt. The scripted
# CARLA ego holds at exactly 0 when stopped.
STOP_SPEED_MPS = 0.5
OBSERVED_FRAME_COUNT = 8
INPUT_CONDITIONS = ("video", "history_json", "video_history")
LEGACY_INPUT_CONDITIONS = ("root_only",)
PRIMARY_MEASUREMENT = "ground_truth_free_l3_causal_coherence"
REFERENCE_EVALUATION_ROLE = "secondary_declared_reference_agreement"
HISTORY_TIMESTAMPS = tuple(round(-3.5 + 0.5 * i, 3) for i in range(OBSERVED_FRAME_COUNT))
EGO_SWERVE_SPEED_MPS = 2.5
# A swerve moves the ego laterally at EGO_SWERVE_SPEED_MPS until it is one lane
# width from its road-following path, then holds that offset (kinematic-twin-5).
# CARLA's ego action realises the same lane change in the
# simulator (generate._SWERVE_INTENSITY on a 3.5 m lane); the twin and the
# action-profile control apply this rule, so both references share one swerve.
EGO_SWERVE_LANE_WIDTH_M = 3.5
CARLA_ROUTE_HALF_WIDTH_M = 8.0
HISTORY_FIELDS = ("timestamp_sec", "id", "entity_type", "role_type", "x", "y",
                  "vx", "vy", "speed", "length", "width", "visibility")

# The three currently declared ego branches are retained as the complete
# scenario-applicable action vocabulary.  The builder still rolls and gates
# every member; this avoids silently making an open-world completeness claim.
# A lateral escape steers AWAY from the critical actor. An actor entering from
# the ego's left therefore gets do_ego_swerve_right and one from the right gets
# do_ego_swerve_left; a critical actor within half a lane of the ego centreline
# has no near side, so the left default stands. This keeps the scorer's
# action-direction axis able to test both signs and never makes the only
# lateral option steer INTO a hazard entering from the left.
LATERAL_NEAR_SIDE_M = 1.75

SCENARIO_ACTIONS = {
    "rear_end_collision_risk": (
        "do_ego_emergency_brake", "do_ego_no_action", "do_ego_swerve_left"),
    "pedestrian_collision_risk": (
        "do_ego_comfortable_brake", "do_ego_panic_brake", "do_ego_swerve_left"),
    "cyclist_collision_risk": (
        "do_ego_comfortable_brake", "do_ego_panic_brake", "do_ego_swerve_left"),
    "generic_following": (
        "do_ego_comfortable_brake", "do_ego_no_action", "do_ego_swerve_left"),
    "cut_in_collision_risk": (
        "do_ego_comfortable_brake", "do_ego_no_action", "do_ego_accelerate_escape"),
    "obstacle_appearance_risk": (
        "do_ego_emergency_brake", "do_ego_no_action", "do_ego_swerve_left"),
}

def scenario_actions(scenario_type, critical_lateral=None):
    """The three declared L3 actions of a scene.

    `critical_lateral` is the critical actor's lateral offset at T0 in the
    t0_ego frame (+y right). When it is on the ego's left beyond
    LATERAL_NEAR_SIDE_M, the declared lateral escape becomes
    do_ego_swerve_right; otherwise the table stands. Returns the table
    unchanged for an unknown scenario type."""
    actions = SCENARIO_ACTIONS.get(scenario_type)
    if actions is None:
        return ()
    want = lateral_escape_action(critical_lateral)
    return tuple(want if a in ("do_ego_swerve_left", "do_ego_swerve_right") else a
                 for a in actions)


def lateral_escape_action(critical_lateral=None):
    """The lateral escape that steers away from a hazard at `critical_lateral`."""
    if critical_lateral is not None and float(critical_lateral) < -LATERAL_NEAR_SIDE_M:
        return "do_ego_swerve_right"
    return "do_ego_swerve_left"


def lateral_action_conflict(root, critical_actor_id, actions):
    """Reason string when a scene's declared lateral action steers TOWARD the
    critical actor as its own published root shows it, else None.

    The declared set is chosen before the rollout from the template's lateral
    offset along the route; on a curved route the realised T0 offset in the
    t0_ego frame can land on the other side. The published scene is
    therefore checked against its own root, and
    a scene that disagrees is rejected and resampled rather than published with
    an escape that steers into the hazard."""
    lateral = [a for a in actions if a in ("do_ego_swerve_left", "do_ego_swerve_right")]
    if not lateral:
        return None
    y = critical_lateral_offset(root, critical_actor_id)
    if y is None:
        return None
    want = lateral_escape_action(y)
    if lateral[0] == want:
        return None
    return (f"declared lateral action {lateral[0]} steers toward critical actor "
            f"{critical_actor_id} at y={y:.2f} m in the published root; the side-aware "
            f"rule wants {want} (threshold {LATERAL_NEAR_SIDE_M} m)")


def critical_lateral_offset(root, critical_actor_id):
    """The critical actor's lateral offset at T0, or None when unavailable."""
    if not isinstance(root, dict) or critical_actor_id is None:
        return None
    for actor in root.get("actors") or []:
        if str(actor.get("id")) == str(critical_actor_id):
            try:
                return float(actor.get("y"))
            except (TypeError, ValueError):
                return None
    return None


# One real-world L4 query is reviewed per scenario.  The twin applies this
# intervention to the named actor while the ego follows an open-loop policy.
NON_EGO_INTERVENTIONS = {
    "vehicle": "npc_sudden_brake",
    "truck": "npc_sudden_brake",
    "bus": "npc_sudden_brake",
    "motorcycle": "npc_sudden_brake",
    "bicycle": "cyclist_enters_road",
    "pedestrian": "pedestrian_enters_road",
    "barrier": "obstacle_sudden_appearance",
    "traffic_cone": "obstacle_sudden_appearance",
}

# The reviewed nuScenes track uses only dynamic actors.  These are the possible
# ontology labels per entity; the builder resolves exactly one label from the
# actor's role, lateral side and past-derived T0 motion state.  Entity type alone
# never authorises a reviewed intervention.
NUSCENES_REVIEW_INTERVENTIONS = {
    "vehicle": ("npc_sudden_brake", "npc_cut_in_left", "npc_cut_in_right"),
    "truck": ("npc_sudden_brake", "npc_cut_in_left", "npc_cut_in_right"),
    "bus": ("npc_sudden_brake", "npc_cut_in_left", "npc_cut_in_right"),
    "motorcycle": ("npc_sudden_brake", "npc_cut_in_left", "npc_cut_in_right"),
    "bicycle": ("cyclist_enters_road",),
    "pedestrian": ("pedestrian_enters_road",),
}


def reference_semantics(split: str) -> str:
    """Interpret, rather than overclaim, the secondary reference scores."""
    value = str(split or "unknown").lower()
    if value == "carla":
        return "exact_deterministic_simulator_agreement"
    if "nuscenes" in value or value.startswith("v1.0-"):
        return "observed_factual_and_canonical_twin_agreement"
    return "declared_reference_agreement"


def validate_protocol_manifest(manifest: Dict[str, Any]) -> Dict[str, Any]:
    """Validate released per-scenario input metadata fail-closed."""
    if manifest.get("protocol_version") != PROTOCOL_VERSION:
        raise ValueError("protocol_version mismatch")
    if manifest.get("observed_frame_count") != OBSERVED_FRAME_COUNT:
        raise ValueError("observed_frame_count must be exactly 8")
    if manifest.get("coordinate_frame") != COORDINATE_FRAME:
        raise ValueError(f"coordinate_frame must be {COORDINATE_FRAME!r}")
    if manifest.get("coordinate_convention") != COORDINATE_CONVENTION:
        raise ValueError("coordinate_convention mismatch")
    if manifest.get("primary_measurement") != PRIMARY_MEASUREMENT:
        raise ValueError("primary_measurement mismatch")
    if manifest.get("reference_evaluation_role") != REFERENCE_EVALUATION_ROLE:
        raise ValueError("reference_evaluation_role mismatch")
    expected_reference = reference_semantics(manifest.get("split", "unknown"))
    if manifest.get("reference_semantics") != expected_reference:
        raise ValueError("reference_semantics mismatch")
    track_steps(manifest.get("track", PRIMARY_TRACK))
    condition = manifest.get("condition", "video")
    if condition not in INPUT_CONDITIONS:
        raise ValueError("invalid input condition")
    available = manifest.get("available_conditions", INPUT_CONDITIONS)
    if tuple(available) != INPUT_CONDITIONS and set(available) != set(INPUT_CONDITIONS):
        raise ValueError("protocol manifest must expose all three input conditions")
    if manifest.get("history_timestamps_sec") is not None:
        got = tuple(round(float(v), 3) for v in manifest["history_timestamps_sec"])
        if got != HISTORY_TIMESTAMPS:
            raise ValueError("history timestamps must be -3.5 through 0.0 at 0.5 s")
    frame_ts = tuple(round(float(v), 3) for v in
                     manifest.get("observed_frame_timestamps_sec", HISTORY_TIMESTAMPS))
    if frame_ts != HISTORY_TIMESTAMPS:
        raise ValueError("frame timestamps must be -3.5 through 0.0 at 0.5 s")
    return manifest


def track_steps(track: str) -> int:
    try:
        return TRACK_STEPS[track]
    except KeyError as exc:
        raise ValueError(f"unknown CORIVA track: {track!r}") from exc


def validate_condition(condition: str) -> str:
    if condition not in INPUT_CONDITIONS and condition not in LEGACY_INPUT_CONDITIONS:
        raise ValueError(
            f"unknown input condition {condition!r}; expected one of {INPUT_CONDITIONS}")
    return condition


def protocol_manifest(track: str = PRIMARY_TRACK, condition: str = "video",
                      split: str = "unknown", **extra: Any) -> Dict[str, Any]:
    validate_condition(condition)
    steps = track_steps(track)
    value = {
        "protocol_version": PROTOCOL_VERSION,
        "track": track,
        "split": split,
        "condition": condition,
        "dt_seconds": TRACK_DT,
        "future_steps": steps,
        "future_horizon_seconds": steps * TRACK_DT,
        "observed_frame_count": OBSERVED_FRAME_COUNT,
        "coordinate_frame": COORDINATE_FRAME,
        "coordinate_convention": dict(COORDINATE_CONVENTION),
        "observed_frame_timestamps_sec": list(HISTORY_TIMESTAMPS),
        "history_timestamps_sec": list(HISTORY_TIMESTAMPS),
        "available_conditions": list(INPUT_CONDITIONS),
        "primary_measurement": PRIMARY_MEASUREMENT,
        "reference_evaluation_role": REFERENCE_EVALUATION_ROLE,
        "reference_semantics": reference_semantics(split),
    }
    value.update(extra)
    return value
