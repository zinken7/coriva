"""
Kinematic Digital Twin engine.

Deterministic, axis-aligned point-mass forward model used to roll a real-world
observed state (e.g. from nuScenes) forward under explicit ego interventions,
producing scene-admissible branching futures in the CORIVA structure. Pure
Python — no CARLA, no nuScenes dependency (those feed the root state in).

Model: fixed T0 ego axes with +x forward, +y right and -y left; longitudinal
acceleration acts along +x and lateral velocity along y (for point-mass
swerves). Speed is clamped to >= 0 (braking cannot reverse the vehicle).
Identity fields (id/entity_type/role_type) are preserved across steps, as is the
per-actor body extent (`length`/`width`, axis-aligned in the ego frame) when
present — collision detection sums the two actors' half-extents per axis so a
long truck is hit before a compact car, falling back to a fixed
collision_gap/lane_halfwidth when dims are absent (point-center actors stay
backward compatible).
"""

import math
from typing import Any, Dict, List, Optional, Tuple

try:
    from simulation.build_ground_truth import INTERVENTION_TO_ACTION
    from simulation.protocol import EGO_SWERVE_SPEED_MPS, EGO_SWERVE_LANE_WIDTH_M
except ImportError:  # when run as a script from inside simulation/
    from build_ground_truth import INTERVENTION_TO_ACTION
    from protocol import EGO_SWERVE_SPEED_MPS, EGO_SWERVE_LANE_WIDTH_M

# Ego intervention -> longitudinal accel (m/s^2) and lateral velocity (m/s).
# The lateral velocity is applied by BOTH ego models (capped at one lane
# width, see roll_branch); the `steer` entry is not read by the rollout and
# is kept only for readers of the table.
# Keys are ontology intervention_type labels (do_ego_*).
# Semantics version of the reference rollouts this engine produces. Bump it
# whenever `step_actor`, terminal detection or role relabelling changes the
# chains a build would write, so that campaign admission can tell two builds
# apart even when the protocol version is unchanged.
# kinematic-twin-5 semantics:
#   - `step_actor` keeps the sign of vx (braking reduces |v| to rest, never
#     flips it); only the collision partner is relabelled `impacted_actor`.
#   - First contact is located by interpolating every actor between
#     consecutive samples at CONTACT_SUBSTEPS sub-steps (0.1 s, the CARLA
#     tick), and the collision node is the LAST SAMPLED STATE AT OR BEFORE
#     CONTACT -- the same convention as the CARLA builder, which emits the
#     last 0.5 s sample at or before the recorded hit. Contact inside the
#     first interval has no pre-contact future sample and raises
#     TwinReferenceError.
#   - `safe_stop` requires the ego to stay at or below STOP_SPEED_MPS from
#     the stop through the end of the horizon; a stop followed by motion is
#     not terminal, and a collision anywhere in the horizon takes precedence.
#   - One swerve for both references. The ego's lateral action is a constant
#     EGO_SWERVE_SPEED_MPS in the fixed T0 frame until the offset from its
#     road-following path reaches EGO_SWERVE_LANE_WIDTH_M (one lane), then it
#     holds that offset; the reported vy carries the lateral component while
#     it moves and drops to the road-following value once held, speed =
#     hypot(vx, vy). The bicycle model keeps `base_steer` (the scene's own
#     curvature) and adds no steering angle for the action; the point-mass
#     model (the action-profile control) applies the same cap. This matches
#     CARLA's ego, which changes one 3.5 m lane and holds it.
TWIN_VERSION = "kinematic-twin-5"
# Sub-steps per sample interval when locating first contact (0.5 s / 5 = 0.1 s).
CONTACT_SUBSTEPS = 5
# Ego speed (m/s) below which it counts as stopped for `safe_stop`; shared
# with the CARLA builder, verifier and prompt (simulation.protocol).
from simulation.protocol import STOP_SPEED_MPS  # noqa: E402


class TwinReferenceError(ValueError):
    """The rolled branch cannot be represented under the reference protocol
    (e.g. contact before the first sampled future state)."""

EGO_PROFILES: Dict[str, Dict[str, float]] = {
    "do_ego_emergency_brake":  {"accel": -8.0, "lat": 0.0,  "steer": 0.0},
    "do_ego_comfortable_brake": {"accel": -3.0, "lat": 0.0, "steer": 0.0},
    "do_ego_panic_brake":      {"accel": -9.0, "lat": 0.0,  "steer": 0.0},
    "do_ego_decelerate_coast": {"accel": -1.0, "lat": 0.0,  "steer": 0.0},
    "do_ego_no_action":        {"accel": 0.0,  "lat": 0.0,  "steer": 0.0},
    "do_ego_accelerate_pass":  {"accel": 3.0,  "lat": 0.0,  "steer": 0.0},
    "do_ego_accelerate_escape": {"accel": 3.5, "lat": 0.0,  "steer": 0.0},
    "do_ego_swerve_left":      {"accel": 0.0,  "lat": -EGO_SWERVE_SPEED_MPS, "steer": 0.04},
    "do_ego_swerve_right":     {"accel": 0.0,  "lat": EGO_SWERVE_SPEED_MPS, "steer": -0.04},
    "do_ego_brake_and_swerve_left":  {"accel": -4.0, "lat": -EGO_SWERVE_SPEED_MPS, "steer": 0.04},
    "do_ego_brake_and_swerve_right": {"accel": -4.0, "lat": EGO_SWERVE_SPEED_MPS,  "steer": -0.04},
}

ENV_PROFILES: Dict[str, Dict[str, float]] = {
    "npc_sudden_brake": {"accel": -8.0, "lat": 0.0},
    "npc_cut_in_left": {"accel": 0.0, "lat": -2.0},
    "npc_cut_in_right": {"accel": 0.0, "lat": 2.0},
    "pedestrian_enters_road": {"accel": 0.0, "lat": -1.6},
    "cyclist_enters_road": {"accel": 0.0, "lat": -4.0},
    "obstacle_sudden_appearance": {"accel": 0.0, "lat": 0.0},
}

WHEELBASE_M = 2.7


def _carry_dims(src: Dict[str, Any], dst: Dict[str, Any]) -> None:
    """Carry the static body extent (`length`/`width`, axis-aligned in the ego
    frame) forward across a step when present. These are body dimensions, not
    motion state, so they pass through unchanged."""
    for k in ("length", "width"):
        if k in src:
            dst[k] = src[k]


def step_actor(actor: Dict[str, Any], accel: float, lat_v: float, dt: float) -> Dict[str, Any]:
    """Advance one actor by dt under a longitudinal accel and lateral velocity."""
    vx0 = float(actor.get("vx", 0.0))
    # Longitudinal motion keeps the actor's direction of travel: a braking
    # profile reduces the speed magnitude towards rest and never flips the
    # sign, so an actor travelling in -x (oncoming in the ego frame) keeps
    # moving instead of being frozen at T0.
    sign = -1.0 if vx0 < 0.0 else 1.0
    s0 = abs(vx0)
    s1 = s0 + accel * dt
    if s1 < 0.0:
        s1 = 0.0
        # Actor stops within this step; advance only to the stopping point.
        ds = (s0 * s0) / (2.0 * abs(accel)) if accel < 0 else 0.0
    else:
        ds = s0 * dt + 0.5 * accel * dt * dt
    vx1 = sign * s1
    x1 = float(actor.get("x", 0.0)) + sign * ds
    y1 = float(actor.get("y", 0.0)) + lat_v * dt
    vy1 = lat_v
    speed = math.sqrt(vx1 * vx1 + vy1 * vy1)
    out = {
        "id": actor.get("id"),
        "entity_type": actor.get("entity_type"),
        "role_type": actor.get("role_type"),
        "x": round(x1, 3),
        "y": round(y1, 3),
        "vx": round(vx1, 3),
        "vy": round(vy1, 3),
        "speed": round(speed, 3),
    }
    _carry_dims(actor, out)
    return out


def step_actor_bicycle(actor: Dict[str, Any], accel: float, steer: float, dt: float,
                       wheelbase: float = WHEELBASE_M) -> Dict[str, Any]:
    """Advance one actor by dt under a kinematic bicycle model: longitudinal
    accel along the heading + a steering angle that curves the heading. Heading
    is read from / written back to vx/vy (no extra field), so output is the same
    standard fields as the point-mass model and remains schema-compatible."""
    vx0 = float(actor.get("vx", 0.0))
    vy0 = float(actor.get("vy", 0.0))
    v0 = math.hypot(vx0, vy0)
    th0 = math.atan2(vy0, vx0) if v0 > 1e-6 else 0.0

    v1 = v0 + accel * dt
    if v1 < 0.0:
        v1 = 0.0
        ds = (v0 * v0) / (2.0 * abs(accel)) if accel < 0 else 0.0
    else:
        ds = v0 * dt + 0.5 * accel * dt * dt
    # Positive steering is a physical left turn.  CORIVA's lateral axis points
    # right, so leftward heading and velocity have a negative y component.
    th1 = th0 - (v0 / wheelbase) * math.tan(steer) * dt

    x1 = float(actor.get("x", 0.0)) + ds * math.cos(th0)
    y1 = float(actor.get("y", 0.0)) + ds * math.sin(th0)
    vx1 = v1 * math.cos(th1)
    vy1 = v1 * math.sin(th1)
    out = {
        "id": actor.get("id"),
        "entity_type": actor.get("entity_type"),
        "role_type": actor.get("role_type"),
        "x": round(x1, 3),
        "y": round(y1, 3),
        "vx": round(vx1, 3),
        "vy": round(vy1, 3),
        "speed": round(v1, 3),
    }
    _carry_dims(actor, out)
    return out


def _env_motion(aid: str, env_accels: Dict[str, float],
                env_behavior: Optional[Dict[str, Dict[str, float]]]) -> tuple:
    """Resolve a non-ego actor's (longitudinal accel, lateral velocity).
    env_behavior ({id: {accel, lat}}) takes precedence over env_accels
    ({id: accel}), which is also accepted."""
    if env_behavior and aid in env_behavior:
        b = env_behavior[aid]
        return float(b.get("accel", 0.0)), float(b.get("lat", 0.0))
    return float(env_accels.get(aid, 0.0)), 0.0


def roll_branch(
    root_actors: List[Dict[str, Any]],
    ego_intervention: str,
    n_steps: int,
    dt: float,
    env_accels: Optional[Dict[str, float]] = None,
    model: str = "point_mass",
    env_behavior: Optional[Dict[str, Dict[str, float]]] = None,
    base_steer: float = 0.0,
    environment_intervention: Optional[Dict[str, Any]] = None,
) -> List[Dict[str, Dict[str, Any]]]:
    """Roll the root state forward n_steps. The ego (role_type == 'ego_agent')
    follows EGO_PROFILES[ego_intervention] under `model` ('point_mass' or
    'bicycle'). Every other actor follows its per-actor behavior — `env_behavior`
    ({id: {accel, lat}}) or `env_accels` ({id: accel}) — using the
    point-mass update (a crossing pedestrian moves laterally, a parked car stays).
    `base_steer` (bicycle model only) is the scene's own curvature —
    estimated from the ego's PAST yaw rate — so a car rolled forward on a
    curved road keeps following the curve under every intervention.

    Swerve: the ego's road-following state is integrated
    separately (bicycle with `base_steer`, or point-mass with no lateral
    term) and the action's lateral velocity is added in the fixed T0 frame
    on top of it until the accumulated offset reaches EGO_SWERVE_LANE_WIDTH_M,
    after which the offset is held. The emitted vy includes the lateral
    velocity while the offset grows and the road-following vy once held.
    This reproduces the CARLA ego action (2.5 m/s to a 3.5 m lane, hold) on
    a straight road exactly; see tests/test_twin_carla_swerve_parity.py.
    Returns a list of per-step {actor_id: state}."""
    if ego_intervention not in EGO_PROFILES:
        raise KeyError(f"unknown ego intervention: {ego_intervention!r}")
    prof = EGO_PROFILES[ego_intervention]
    ego_lat_v = float(prof.get("lat", 0.0))
    ego_lat_offset = 0.0
    ego_base: Optional[Dict[str, Any]] = None  # road-following ego state
    env_accels = env_accels or {}
    env_intervention = environment_intervention or {}
    env_target = env_intervention.get("target_actor_id")
    env_prof = ENV_PROFILES.get(env_intervention.get("intervention_type"), {})

    current = {a["id"]: dict(a) for a in root_actors}
    states: List[Dict[str, Dict[str, Any]]] = []
    for _ in range(n_steps):
        nxt: Dict[str, Dict[str, Any]] = {}
        for aid, a in current.items():
            if a.get("role_type") == "ego_agent":
                if ego_base is None:
                    ego_base = dict(a)
                if model == "bicycle":
                    ego_base = step_actor_bicycle(ego_base, prof["accel"], base_steer, dt)
                else:
                    ego_base = step_actor(ego_base, prof["accel"], 0.0, dt)
                out = dict(ego_base)
                if ego_lat_v != 0.0:
                    remaining = max(0.0, EGO_SWERVE_LANE_WIDTH_M - abs(ego_lat_offset))
                    step_lat = math.copysign(min(abs(ego_lat_v) * dt, remaining), ego_lat_v)
                    ego_lat_offset += step_lat
                    moving = abs(ego_lat_offset) < EGO_SWERVE_LANE_WIDTH_M - 1e-9
                    vy = float(ego_base.get("vy", 0.0)) + (ego_lat_v if moving else 0.0)
                    out["y"] = round(float(ego_base.get("y", 0.0)) + ego_lat_offset, 3)
                    out["vy"] = round(vy, 3)
                    out["speed"] = round(math.hypot(float(ego_base.get("vx", 0.0)), vy), 3)
                nxt[aid] = out
            else:
                accel, lat = _env_motion(aid, env_accels, env_behavior)
                if aid == env_target and env_prof:
                    # Reviewed real-scene interventions may resolve the profile
                    # direction from the target's T0 side.  The public ontology
                    # label stays unchanged; these private controls prevent a
                    # pedestrian/cyclist on one side from being moved away from
                    # the ego lane by a globally fixed lateral sign.
                    accel = float(env_intervention.get(
                        "resolved_accel_mps2", env_prof.get("accel", accel)))
                    lat = float(env_intervention.get(
                        "resolved_lateral_velocity_mps", env_prof.get("lat", lat)))
                nxt[aid] = step_actor(a, accel, lat, dt)
        states.append(nxt)
        current = nxt
    return states


def _collision_gaps(ego, o, collision_gap, lane_halfwidth):
    """Per-axis center-distance thresholds for an ego↔other collision. When both
    actors carry a body extent (`length`/`width`, axis-aligned in the ego frame),
    the threshold is the sum of their half-extents — so a long truck is hit
    sooner than a compact car. Actors without dims fall back to the fixed
    collision_gap (x) / lane_halfwidth (y)."""
    gap_x = (ego["length"] + o["length"]) / 2.0 if "length" in ego and "length" in o else collision_gap
    gap_y = (ego["width"] + o["width"]) / 2.0 if "width" in ego and "width" in o else lane_halfwidth
    return gap_x, gap_y


def _first_contact_fraction(ego0, ego1, o0, o1, collision_gap, lane_halfwidth):
    """Smallest fraction f in {1/n, ..., 1} of the sample interval at which the
    linearly interpolated ego and other footprints overlap, else None."""
    gap_x, gap_y = _collision_gaps(ego1, o1, collision_gap, lane_halfwidth)
    for k in range(1, CONTACT_SUBSTEPS + 1):
        f = k / CONTACT_SUBSTEPS
        ex = ego0["x"] + (ego1["x"] - ego0["x"]) * f
        ey = ego0["y"] + (ego1["y"] - ego0["y"]) * f
        ox = o0["x"] + (o1["x"] - o0["x"]) * f
        oy = o0["y"] + (o1["y"] - o0["y"]) * f
        if abs(ox - ex) < gap_x and abs(oy - ey) < gap_y:
            return f
    return None


FIRST_INTERVAL_RAISE = "raise"          # references: not representable
FIRST_INTERVAL_LABEL_FIRST = "label_first"  # predictions/controls: best representable chain


def _detect_terminal_partner(states, root_actors, collision_gap, lane_halfwidth,
                             first_interval=FIRST_INTERVAL_RAISE):
    """Return (terminal_event_type, terminal_step_index, collision_partner_ids).

    Collision: every actor is interpolated between consecutive
    samples (the root before the first) at CONTACT_SUBSTEPS sub-steps; the
    first sub-step at which the ego footprint overlaps another actor's
    (per-axis sum of half-extents, or the fixed collision_gap/lane_halfwidth
    when dims are absent) is the contact. The collision node is the last
    sampled state at or before that contact -- the sample itself when contact
    falls exactly on it, otherwise the previous sample -- matching the CARLA
    builder. Every actor first contacted at that same sub-step is a partner
    (order-independent), so a simultaneous two-actor contact labels both.
    Contact inside the first interval has no pre-contact future sample:
    with `first_interval="raise"` (references) it raises TwinReferenceError;
    with `"label_first"` (predictions such as the action-profile control,
    which must always emit a chain) the first sample is the collision node.

    safe_stop: the ego is at or below STOP_SPEED_MPS at some sample
    and stays at or below it at every later sample of the horizon; the node is the first
    such sample. A stop followed by motion is not a terminal event, and a
    collision anywhere in the horizon takes precedence.

    Merely exhausting the observation horizon is not a physical terminal
    event and therefore returns ``(None, None, None)``."""
    ego_id = next(a["id"] for a in root_actors if a.get("role_type") == "ego_agent")
    others = [a["id"] for a in root_actors if a["id"] != ego_id]
    prev = {a["id"]: a for a in root_actors}
    for i, world in enumerate(states):
        ego1 = world[ego_id]
        ego0 = prev.get(ego_id, ego1)
        contacts = []
        for oid in others:
            o1 = world.get(oid)
            if o1 is None:
                continue
            o0 = prev.get(oid, o1)
            f = _first_contact_fraction(ego0, ego1, o0, o1, collision_gap, lane_halfwidth)
            if f is not None:
                contacts.append((f, oid))
        if contacts:
            f_min = min(f for f, _ in contacts)
            partners = sorted(oid for f, oid in contacts if abs(f - f_min) < 1e-9)
            idx = i if f_min >= 1.0 - 1e-9 else i - 1
            if idx < 0:
                if first_interval == FIRST_INTERVAL_LABEL_FIRST:
                    idx = 0
                else:
                    raise TwinReferenceError(
                        f"contact with {partners} inside the first sample interval "
                        f"(fraction {f_min:.1f}): no pre-contact future sample to represent it")
            return "collision", idx, partners
        prev = world
    speeds = [float(w[ego_id].get("speed", 0.0)) for w in states]
    # "at or below" STOP_SPEED_MPS counts as stopped, exactly as the CARLA
    # builder, the verifier and the prompt define it.
    moving = [j for j, s in enumerate(speeds) if s > STOP_SPEED_MPS]
    first_stop = (moving[-1] + 1) if moving else 0
    if first_stop < len(states):
        return "safe_stop", first_stop, []
    return None, None, []


def _detect_terminal(states, root_actors, collision_gap, lane_halfwidth):
    """(terminal_event_type, terminal_step_index); see _detect_terminal_partner."""
    term_type, term_idx, _partner = _detect_terminal_partner(
        states, root_actors, collision_gap, lane_halfwidth)
    return term_type, term_idx


def build_twin_branch(branch_id, is_factual, intervention_type, root, n_steps, dt,
                      sample_interval, env_accels=None, collision_gap=4.0,
                      lane_halfwidth=2.0, intervention_meta=None, model="point_mass",
                      env_behavior=None, override_states=None, base_steer=0.0,
                      environment_intervention=None, first_interval=FIRST_INTERVAL_RAISE):
    """Roll one branch from the root under intervention_type and emit a CORIVA
    branch. If `override_states` is given (a precomputed per-step
    {actor_id: state} list), those states are used verbatim instead of rolling
    the twin — this anchors the FACTUAL branch to the real observed future while
    the (unobservable) counterfactual branches stay twin-generated."""
    if intervention_type in EGO_PROFILES:
        action_type, default_event = INTERVENTION_TO_ACTION.get(
            intervention_type, ("ego_no_action", "maintain_course"))
        ego_action = intervention_type
    elif intervention_type in ENV_PROFILES:
        action_type, default_event = "ego_no_action", "maintain_course"
        ego_action = "do_ego_no_action"
        environment_intervention = environment_intervention or {
            "intervention_type": intervention_type,
            "target_actor_id": "ego", "trigger_depth": 1,
        }
    else:
        raise KeyError(f"unknown intervention_type={intervention_type!r}")
    states = override_states if override_states is not None else roll_branch(
        root["actors"], ego_action, n_steps, dt, env_accels,
        model=model, env_behavior=env_behavior, base_steer=base_steer,
        environment_intervention=environment_intervention)
    root_ts = float(root.get("timestamp", 0.0))
    term_type, term_idx, partners = _detect_terminal_partner(
        states, root["actors"], collision_gap, lane_halfwidth, first_interval=first_interval)
    last_idx = term_idx if term_idx is not None else len(states) - 1

    chain = []
    for i in range(last_idx + 1):
        depth = i + 1
        is_terminal = term_type is not None and i == term_idx
        ev = term_type if is_terminal else default_event
        node: Dict[str, Any] = {
            "depth": depth,
            "timestamp": round(root_ts + depth * sample_interval, 2),
            "event_type": ev,
        }
        if ev != "collision":
            node["action_type"] = action_type
        if ev in ("collision", "safe_stop", "out_of_bounds"):
            node["terminal"] = True
        actors = []
        for a in states[i].values():
            actor = dict(a)
            # Only the actor(s) the ego actually hits are impacted; a far-away
            # vehicle keeps its role.
            if ev == "collision" and a.get("id") in partners:
                actor["role_type"] = "impacted_actor"
            actors.append(actor)
        node["actors_update"] = actors
        chain.append(node)

    return {
        "branch_id": branch_id,
        "is_factual": is_factual,
        "intervention": intervention_meta or {
            "intervention_type": intervention_type, "target_actor_id": "ego", "trigger_depth": 1},
        "chain": chain,
    }


def build_twin_gt(root, scenario_spec):
    """Assemble a full CORIVA ground truth from a root observed state by rolling
    each branch through the kinematic twin. Branches + counterfactuals + the
    shared observed_state root. The motion model is point_mass unless the
    scenario_spec sets model: 'bicycle'."""
    model = scenario_spec.get("model", "point_mass")
    base_steer = float(scenario_spec.get("base_steer", 0.0))
    # Real observed future for the factual branch: if present, the
    # factual branch is anchored to reality; counterfactual branches stay twin.
    factual_states = scenario_spec.get("factual_states")
    rolled_branches = []
    for b in scenario_spec["branches"]:
        rolled_branches.append(build_twin_branch(
            b["branch_id"], b["is_factual"], b["intervention_type"], root,
            scenario_spec["n_steps"], scenario_spec["dt"], scenario_spec["sample_interval"],
            env_accels=scenario_spec.get("env_accels"),
            collision_gap=scenario_spec.get("collision_gap", 4.0),
            lane_halfwidth=scenario_spec.get("lane_halfwidth", 2.0),
            model=model,
            env_behavior=b.get("env_behavior", scenario_spec.get("env_behavior")),
            override_states=factual_states if b["is_factual"] else None,
            base_steer=base_steer,
            intervention_meta={
                "intervention_type": b["intervention_type"],
                "target_actor_id": b.get("target_actor_id", "ego"),
                "trigger_depth": b.get("trigger_depth", 1),
            },
            environment_intervention=b.get("environment_intervention"),
        ))
    # L3 is the declared future branch set: the factual branch plus every
    # scenario-applicable ego-action branch.  A named non-ego intervention is
    # an L4 query and must not change the L3 denominator.
    branches = [
        b for b in rolled_branches
        if b.get("intervention", {}).get("target_actor_id", "ego") == "ego"
        or b.get("is_factual")
    ]
    l4_branches = [
        b for b in rolled_branches
        if b.get("intervention", {}).get("target_actor_id", "ego") != "ego"
    ]
    # Keep the historical `counterfactuals` field for existing datasets and
    # scorer consumers.  Protocol-aware datasets additionally expose the
    # reviewed named-actor queries in `l4_queries`; the scorer and runner use
    # that field for the NeurIPS main track.
    counterfactuals = [
        {"intervention_type": b["intervention"]["intervention_type"],
         "counterfactual_chain": {"intervention": b["intervention"], "chain": b["chain"]}}
        for b in rolled_branches if not b["is_factual"]
    ]
    l4_queries = [
        {"intervention_type": b["intervention"]["intervention_type"],
         "counterfactual_chain": {"intervention": b["intervention"], "chain": b["chain"]}}
        for b in l4_branches
    ]
    # Matched-pair protocol (docs/l4_pair_protocol.md): every named-actor
    # query also publishes its control, the same twin roll with the same
    # fixed ego policy and per-actor baseline but no environment
    # intervention. Deterministic: the migration audit re-derives it.
    for query, b in zip(l4_queries, l4_branches):
        spec_b = next(sb for sb in scenario_spec["branches"]
                      if sb["branch_id"] == b["branch_id"])
        from simulation.l4_control import control_intervention, control_query_entry
        ego_policy = "do_ego_no_action"
        control = build_twin_branch(
            f"{b['branch_id']}_control", False, ego_policy, root,
            scenario_spec["n_steps"], scenario_spec["dt"], scenario_spec["sample_interval"],
            env_accels=scenario_spec.get("env_accels"),
            collision_gap=scenario_spec.get("collision_gap", 4.0),
            lane_halfwidth=scenario_spec.get("lane_halfwidth", 2.0),
            model=model,
            env_behavior=spec_b.get("env_behavior", scenario_spec.get("env_behavior")),
            base_steer=base_steer,
            intervention_meta=control_intervention(b["intervention"]))
        query["control_chain"] = control_query_entry(b["intervention"], control["chain"], "nuscenes",
                                                     twin_version=TWIN_VERSION)
    result = {
        "scenario_id": scenario_spec["scenario_id"],
        "scenario_type": scenario_spec.get("scenario_type", "twin"),
        "observed_state": root,
        "branches": branches,
        "counterfactuals": counterfactuals,
    }
    if l4_queries:
        result["l4_queries"] = l4_queries
    if scenario_spec.get("action_audit") is not None:
        result["action_audit"] = scenario_spec["action_audit"]
    return result
