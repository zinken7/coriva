"""Reference baselines — GT-derived controls for the CORIVA leaderboard.

Each function mirrors _oracle_text in analysis/run_model.py: it takes the
scenario's ground-truth dict and returns raw JSON text in the predicted format,
so predictions flow through the identical parse -> assemble -> score pipeline
as real model output.

These baselines may read the ground truth by design: they are diagnostic
controls (dissertation Chapter 5), not competitors. copy_factual reads the
factual branch; kinematic_extrapolation reads only the root state plus the
shape (depths/timestamps) of the factual chain; random_valid reads the root
state and the scenario's intervention list; action_profile_kinematic reads the
root state, the declared action/query list and the published motion profiles
(the strongest non-learning control; it defines every axis the scene allows
but is not guaranteed to reach the ceiling: branches that collide or stop at
different times can break the ordering even for this rule, so its measured
values are the floor); action_profile_published is the same rule extended
to the per-scene hazard and intervention parameters the prompt prints, so it
is the arithmetic floor over ALL the public information the model receives.
"""

import copy
from contextlib import contextmanager
import json
import os
import random
import sys
import zlib
from typing import Any, Dict, List

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from simulation.kinematic_twin import (EGO_PROFILES, ENV_PROFILES,  # noqa: E402
                                       build_twin_branch)
from simulation.nuscenes_admission import observed_velocity_baseline  # noqa: E402


def _l4_query_set(gt: Dict[str, Any]):
    return gt.get("l4_queries") or gt.get("counterfactuals", [])


L4_CONTROL_ACTION = {"intervention_type": "do_ego_no_action", "target_actor_id": "ego", "trigger_depth": 1}


def l4_pair_unit(intervention: Dict[str, Any], control_chain: list, treated_chain: list) -> Dict[str, Any]:
    """One `predicted_counterfactuals[]` item of the matched-pair protocol
    (docs/l4_pair_protocol.md §2): the query's intervention, the control
    chain (named target on its declared baseline) and the treated chain."""
    return {"intervention": copy.deepcopy(intervention),
            "control_chain": control_chain, "treated_chain": treated_chain}


def _public_chain_shape(gt: Dict[str, Any]):
    """(n_steps, dt) from the published protocol block, falling back to the
    top-level `sample_interval_sec` / `dt` / `n_steps` fields; never from the
    private reference chains beyond their length."""
    proto = gt.get("protocol") or {}
    dt = float(proto.get("dt_seconds") or gt.get("sample_interval_sec") or gt.get("dt") or 0.5)
    n_steps = proto.get("future_steps") or gt.get("n_steps")
    if not n_steps:
        n_steps = max((len(b.get("chain", [])) for b in gt.get("branches", [])), default=6)
    return int(n_steps), dt


def action_profile_kinematic_text(gt: Dict[str, Any]) -> str:
    """The action-aware non-learning control.

    Applies the PUBLISHED ego action profile (``EGO_PROFILES``: longitudinal
    acceleration and lateral velocity) to the ego under a point-mass update
    with no scene curvature, keeps every other actor at its root velocity, and
    for each L4 query applies the published intervention profile
    (``ENV_PROFILES``) to the named target while the ego follows the fixed
    ``do_ego_no_action`` policy. Under the matched-pair protocol
    (docs/l4_pair_protocol.md) every L4 answer is a control/treated pair: the
    control is the same rollout with the target at its root velocity (the
    declared baseline), the treated one applies the profile. Terminal
    events follow the same geometric rules as the twin. It reads only inputs the model also receives: the root
    state, the declared action list, the L4 query metadata and the chain
    length/step. It never reads reference chains, per-scenario twin steering
    or private actor behaviour.

    Both splits' references share the EGO profiles (CARLA's generator mirrors
    `EGO_PROFILES`, test-asserted), so this control shows how much reference
    agreement and how much cross-branch coherence a twenty-line kinematic
    rule obtains. The L4 INTERVENTION profile is shared with the nuScenes twin
    only: CARLA executes the scene configuration's own magnitude through the
    actor's controller (CARLA-400: sudden brake -6/-5 m/s^2, cut-in 1.8 m/s
    as a steering intensity, pedestrian 2.0 m/s, cyclist 2.5 m/s as a steering
    intensity) while this control applies `ENV_PROFILES` (-8, 2.0, 1.6, 4.0),
    so on CARLA its L4 chain differs from the reference by construction and
    its CAS there is bounded by that difference. A design property to report,
    not a scoring error; the prompt prints the scene's own values, the control
    keeps the twin's. It emits every declared branch, so
    each coherence axis the scene's action set and actor inventory allow is
    defined for it (a cut-in template has no lateral action, a scene without
    static actors has no locality cell); it is the floor a model must exceed
    on every axis that is defined before any reasoning claim.
    An intervention type outside the published profiles falls back to the
    no-action rollout and is recorded in the branch id. The control is a
    prediction, not a reference: when its point-mass rollout makes contact
    inside the first sample interval (a reference would be rejected) it
    labels the first sample as the collision node instead of failing, so the
    evaluation never stops on a control branch (twin `first_interval`
    policy "label_first")."""
    root = gt["observed_state"]
    n_steps, dt = _public_chain_shape(gt)
    # Every non-ego actor continues its observed T0 velocity, lateral component
    # included (`step_actor` overwrites vy with the lateral command, so an
    # empty behaviour map would freeze lateral motion).
    observed = observed_velocity_baseline(root)
    actors_by_id = {str(a.get("id")): a for a in root.get("actors", [])}

    def roll(branch_id: str, is_factual: bool, intervention: Dict[str, Any]):
        itype = intervention.get("intervention_type")
        target = intervention.get("target_actor_id", "ego")
        if itype in EGO_PROFILES and target == "ego":
            return build_twin_branch(branch_id, is_factual, itype, root, n_steps, dt, dt,
                                     intervention_meta=copy.deepcopy(intervention),
                                     model="point_mass", base_steer=0.0,
                                     env_behavior=observed, first_interval="label_first")
        if itype in ENV_PROFILES:
            env_iv = {"intervention_type": itype, "target_actor_id": target,
                      "trigger_depth": intervention.get("trigger_depth", 1)}
            if itype in ("pedestrian_enters_road", "cyclist_enters_road"):
                # The published entry profile is directed towards the ego lane
                # from the target's T0 side (public root information).
                ty = float(actors_by_id.get(str(target), {}).get("y", 0.0))
                speed = abs(float(ENV_PROFILES[itype]["lat"]))
                env_iv["resolved_lateral_velocity_mps"] = -speed if ty >= 0.0 else speed
            return build_twin_branch(
                branch_id, is_factual, itype, root, n_steps, dt, dt,
                intervention_meta=copy.deepcopy(intervention), model="point_mass",
                base_steer=0.0, env_behavior=observed, environment_intervention=env_iv,
                first_interval="label_first")
        return build_twin_branch(f"{branch_id}_unprofiled", is_factual, "do_ego_no_action",
                                 root, n_steps, dt, dt,
                                 intervention_meta=copy.deepcopy(intervention),
                                 model="point_mass", base_steer=0.0, env_behavior=observed,
                                 first_interval="label_first")

    branches: List[Dict[str, Any]] = []
    for declared in gt.get("branches", []):
        intervention = declared.get("intervention") or {}
        if intervention.get("target_actor_id", "ego") != "ego":
            continue
        branches.append(roll(f"APK_{intervention.get('intervention_type')}",
                             bool(declared.get("is_factual")), intervention))
    cfs = []
    for cf in _l4_query_set(gt):
        intervention = cf["counterfactual_chain"]["intervention"]
        # Control: the same fixed ego policy with every actor, the named
        # target included, at its root velocity (the declared baseline).
        # Treated: the same with the published profile on the target.
        control = roll("APK_L4_control", False, L4_CONTROL_ACTION)
        treated = roll("APK_L4", False, intervention)
        cfs.append(l4_pair_unit(intervention, control["chain"], treated["chain"]))
    return json.dumps({"predicted_branches": branches,
                       "predicted_counterfactuals": cfs})


# Author-chosen typical magnitudes for a driver who is told only each action's
# name. They are deliberately milder than the stated profiles (`EGO_PROFILES`)
# and are not the values any prompt states, so the program shows what a
# parameter-free strategy achieves under the all-withheld prompt (a calibration
# point, not a floor a model must beat).
TYPICAL_EGO_PROFILES: Dict[str, Dict[str, float]] = {
    "do_ego_emergency_brake":  {"accel": -6.0, "lat": 0.0,  "steer": 0.0},
    "do_ego_comfortable_brake": {"accel": -2.0, "lat": 0.0, "steer": 0.0},
    "do_ego_panic_brake":      {"accel": -8.0, "lat": 0.0,  "steer": 0.0},
    "do_ego_decelerate_coast": {"accel": -0.5, "lat": 0.0,  "steer": 0.0},
    "do_ego_no_action":        {"accel": 0.0,  "lat": 0.0,  "steer": 0.0},
    "do_ego_accelerate_pass":  {"accel": 2.0,  "lat": 0.0,  "steer": 0.0},
    "do_ego_accelerate_escape": {"accel": 2.5, "lat": 0.0,  "steer": 0.0},
    "do_ego_swerve_left":      {"accel": 0.0,  "lat": -1.5, "steer": 0.04},
    "do_ego_swerve_right":     {"accel": 0.0,  "lat": 1.5,  "steer": -0.04},
    "do_ego_brake_and_swerve_left":  {"accel": -3.0, "lat": -1.5, "steer": 0.04},
    "do_ego_brake_and_swerve_right": {"accel": -3.0, "lat": 1.5,  "steer": -0.04},
}


@contextmanager
def _ego_profiles(profiles: Dict[str, Dict[str, float]]):
    """Swap the twin's ego profiles in place for the duration of one rollout.
    `build_twin_branch` reads `EGO_PROFILES[action]` by name, so the values are
    updated in place and restored afterwards; the rollouts are sequential."""
    saved = {k: dict(v) for k, v in EGO_PROFILES.items()}
    try:
        for k, v in profiles.items():
            EGO_PROFILES[k].update(v)
        yield
    finally:
        for k, v in saved.items():
            EGO_PROFILES[k].clear(); EGO_PROFILES[k].update(v)


def action_profile_typical_text(gt: Dict[str, Any]) -> str:
    """The typical-rates program: `action_profile_kinematic_text` with the ego
    under `TYPICAL_EGO_PROFILES` instead of the stated `EGO_PROFILES`; the
    target keeps the fixed-profile program's `ENV_PROFILES`. It reads the same
    public inputs as the other programs and no prompt."""
    with _ego_profiles(TYPICAL_EGO_PROFILES):
        return action_profile_kinematic_text(gt)


def _declared_profile_values(profile: Any):
    """(accel, lat) from a published profile dict ({accel, lat, ...}), or None
    when the profile is absent or carries no usable numbers (e.g. the
    "no published profile" record of `declared_intervention_profile`)."""
    if not isinstance(profile, dict):
        return None
    accel, lat = profile.get("accel"), profile.get("lat")
    if accel is None or lat is None:
        return None
    return float(accel), float(lat)


def action_profile_published_text(gt: Dict[str, Any],
                                  hazard_profile: Any = None,
                                  intervention_profiles: Any = None) -> str:
    """The action-aware control on ALL the information the prompt publishes.

    Identical to `action_profile_kinematic_text` (point-mass ego under the
    published `EGO_PROFILES`, `base_steer` 0, every other actor at its root
    velocity, the twin's terminal rules, `first_interval="label_first"`, no
    reference chain read) with two differences, both driven by inputs the
    prompt prints:

    * L3 -- when `hazard_profile` ({actor_id, accel, lat}, the value of
      `simulation.build_pilot.declared_hazard_profile` that the prompt prints
      as the scene's scripted hazard manoeuvre) names a non-ego root actor,
      that actor follows the declared (accel, lat) from T=0 in every
      GENERATED L3 branch through the twin's per-actor `env_behavior` map,
      exactly as the references script it (CARLA `env_behavior[critical]`,
      nuScenes `scenario_template.hazard_profile`). The FACTUAL branch of a
      split whose factual reference is the recording
      (`observed_history.factual_branch_contract` -> `recorded_future`,
      nuScenes) gets no script: the prompt states that no manoeuvre is
      scripted there, so this control keeps every non-ego actor at its root
      velocity in that branch while the ego still follows the published
      profile of the factual label (the control has no other public rule
      for the recorded motion). On CARLA the
      simulator executed the script in the factual branch too, so it is
      applied there. With `hazard_profile=None`, or an actor id outside the
      root, the L3 branches equal the existing control.
    * L4 -- the named target follows `intervention_profiles[intervention_type]`
      ({accel, lat, source}, the value of
      `simulation.build_pilot.declared_intervention_profile` printed in the
      prompt: the CARLA scene configuration's own magnitude, or the twin's
      side-resolved profile on nuScenes) via the twin's
      `resolved_accel_mps2` / `resolved_lateral_velocity_mps` intervention
      controls. When no usable profile is declared for the type it falls back
      SILENTLY to the existing control's rule (`ENV_PROFILES`, road-entry
      directed towards the lane from the target's T=0 side); the output
      schema is identical to the existing control, so nothing records the
      fallback. An intervention type outside `ENV_PROFILES` falls back to the
      no-action rollout (branch id suffixed `_unprofiled`), as in the
      existing control.

    In L4 every actor other than the target keeps its root velocity: the
    prompt scripts the hazard manoeuvre for the L3 branches only. Reads: the
    root state, the declared action list, the L4 query metadata, the chain
    length/step, and the two published per-scene parameter records. Never
    the reference chains, `scenario_config.json` or twin steering.

    Beating `action_profile_kinematic` shows a model beats arithmetic on the
    published profiles alone; beating this control shows it beats arithmetic
    on everything the prompt states. Both are reported; neither is modified
    to reach a ceiling."""
    root = gt["observed_state"]
    n_steps, dt = _public_chain_shape(gt)
    observed = observed_velocity_baseline(root)
    actors_by_id = {str(a.get("id")): a for a in root.get("actors", [])}
    # L3 behaviour map: the root velocities, with the published hazard
    # manoeuvre substituted for the named actor (the twin reads
    # {id: {accel, lat}}; `lat` is a lateral VELOCITY, as in the references).
    l3_behavior = observed
    hazard = _declared_profile_values(hazard_profile)
    hazard_id = str(hazard_profile.get("actor_id")) if isinstance(hazard_profile, dict) else None
    if hazard is not None and hazard_id in observed:
        l3_behavior = {k: dict(v) for k, v in observed.items()}
        l3_behavior[hazard_id] = {"accel": hazard[0], "lat": hazard[1]}
    from simulation.observed_history import factual_branch_contract
    contract = factual_branch_contract(gt) or {}
    factual_is_recorded = contract.get("factual_reference") == "recorded_future"

    def roll(branch_id: str, is_factual: bool, intervention: Dict[str, Any],
             l4_control: bool = False):
        itype = intervention.get("intervention_type")
        target = intervention.get("target_actor_id", "ego")
        if itype in EGO_PROFILES and target == "ego":
            # The L4 control carries no hazard script (the prompt scripts it
            # for the L3 branches only): every actor at its root velocity.
            behaviour = observed if (l4_control or (is_factual and factual_is_recorded)) else l3_behavior
            return build_twin_branch(branch_id, is_factual, itype, root, n_steps, dt, dt,
                                     intervention_meta=copy.deepcopy(intervention),
                                     model="point_mass", base_steer=0.0,
                                     env_behavior=behaviour, first_interval="label_first")
        if itype in ENV_PROFILES:
            env_iv = {"intervention_type": itype, "target_actor_id": target,
                      "trigger_depth": intervention.get("trigger_depth", 1)}
            declared = _declared_profile_values((intervention_profiles or {}).get(itype))
            if declared is not None:
                env_iv["resolved_accel_mps2"] = declared[0]
                env_iv["resolved_lateral_velocity_mps"] = declared[1]
            elif itype in ("pedestrian_enters_road", "cyclist_enters_road"):
                # Fallback = the existing control: the published entry profile
                # directed towards the ego lane from the target's T0 side.
                ty = float(actors_by_id.get(str(target), {}).get("y", 0.0))
                speed = abs(float(ENV_PROFILES[itype]["lat"]))
                env_iv["resolved_lateral_velocity_mps"] = -speed if ty >= 0.0 else speed
            return build_twin_branch(
                branch_id, is_factual, itype, root, n_steps, dt, dt,
                intervention_meta=copy.deepcopy(intervention), model="point_mass",
                base_steer=0.0, env_behavior=observed, environment_intervention=env_iv,
                first_interval="label_first")
        return build_twin_branch(f"{branch_id}_unprofiled", is_factual, "do_ego_no_action",
                                 root, n_steps, dt, dt,
                                 intervention_meta=copy.deepcopy(intervention),
                                 model="point_mass", base_steer=0.0, env_behavior=observed,
                                 first_interval="label_first")

    branches: List[Dict[str, Any]] = []
    for declared in gt.get("branches", []):
        intervention = declared.get("intervention") or {}
        if intervention.get("target_actor_id", "ego") != "ego":
            continue
        branches.append(roll(f"APP_{intervention.get('intervention_type')}",
                             bool(declared.get("is_factual")), intervention))
    cfs = []
    for cf in _l4_query_set(gt):
        intervention = cf["counterfactual_chain"]["intervention"]
        # The control is the L4 baseline (root velocities, no hazard script:
        # the prompt scripts the hazard for L3 only), identical to the
        # `action_profile_kinematic` control; only the treated magnitude
        # differs between the two floors.
        control = roll("APP_L4_control", False, L4_CONTROL_ACTION, l4_control=True)
        treated = roll("APP_L4", False, intervention)
        cfs.append(l4_pair_unit(intervention, control["chain"], treated["chain"]))
    return json.dumps({"predicted_branches": branches,
                       "predicted_counterfactuals": cfs})


def copy_factual_text(gt: Dict[str, Any]) -> str:
    """Emit the factual chain as every predicted branch and as every
    counterfactual chain. Tests that the matching layer cannot be gamed by
    repetition: duplicates either collapse in canonicalisation or starve
    against dummy branches in the one-to-one assignment."""
    factual = next(b for b in gt["branches"] if b.get("is_factual"))
    branches = []
    for i, declared in enumerate(gt["branches"]):
        b = copy.deepcopy(factual)
        b["branch_id"] = f"CF_COPY_{i + 1}"
        b["is_factual"] = i == 0
        b["intervention"] = copy.deepcopy(declared.get("intervention"))
        branches.append(b)
    cfs = [l4_pair_unit(cf["counterfactual_chain"]["intervention"],
                        copy.deepcopy(factual["chain"]), copy.deepcopy(factual["chain"]))
           for cf in _l4_query_set(gt)]
    return json.dumps({"predicted_branches": branches,
                       "predicted_counterfactuals": cfs})


def kinematic_extrapolation_text(gt: Dict[str, Any]) -> str:
    """Constant-velocity rollout from the root observed state: every actor keeps
    its root velocity, nothing brakes, nothing branches, and the intervention is
    ignored. Reads only the root state plus the factual chain's depth/timestamp
    shape. Emits one branch (constant velocity admits exactly one future) and
    the same chain for every intervention."""
    factual = next(b for b in gt["branches"] if b.get("is_factual"))
    dt = float(gt.get("sample_interval_sec", 0.5))
    timestamps = [n.get("timestamp", dt * (i + 1))
                  for i, n in enumerate(factual["chain"])]

    chain = []
    elapsed_prev = 0.0
    positions = {a["id"]: (float(a.get("x", 0.0)), float(a.get("y", 0.0)))
                 for a in gt["observed_state"]["actors"]}
    for depth, ts in enumerate(timestamps, start=1):
        step_dt = (ts - elapsed_prev) if depth > 1 else dt
        elapsed_prev = ts
        updates = []
        for a in gt["observed_state"]["actors"]:
            x, y = positions[a["id"]]
            x += float(a.get("vx", 0.0)) * step_dt
            y += float(a.get("vy", 0.0)) * step_dt
            positions[a["id"]] = (x, y)
            updates.append({"id": a["id"], "entity_type": a["entity_type"],
                            "role_type": a.get("role_type"),
                            "x": round(x, 3), "y": round(y, 3),
                            "vx": a.get("vx", 0.0), "vy": a.get("vy", 0.0),
                            "speed": a.get("speed", 0.0)})
        chain.append({"depth": depth, "timestamp": ts,
                      "event_type": "maintain_course",
                      "action_type": "ego_no_action",
                      "actors_update": updates})

    cfs = [l4_pair_unit(cf["counterfactual_chain"]["intervention"],
                        copy.deepcopy(chain), copy.deepcopy(chain))
           for cf in _l4_query_set(gt)]
    action = copy.deepcopy((factual.get("intervention") or {
        "intervention_type": "do_ego_no_action", "target_actor_id": "ego",
        "trigger_depth": 1}))
    return json.dumps({"predicted_branches": [
        {"branch_id": "KIN_EXTRAP", "is_factual": True,
         "intervention": action, "chain": chain}],
        "predicted_counterfactuals": cfs})


# Non-terminal events a random rollout may emit without extra ontology slots.
_RANDOM_EVENTS = ["maintain_course", "lane_follow"]
_MAX_RANDOM_SPEED = 3.0  # m/s per axis; keeps steps inside the kinematic gate


def _random_chain(gt: Dict[str, Any], rng: random.Random) -> list:
    dt = float(gt.get("sample_interval_sec", 0.5))
    positions = {a["id"]: (float(a.get("x", 0.0)), float(a.get("y", 0.0)))
                 for a in gt["observed_state"]["actors"]}
    chain = []
    for depth in range(1, rng.randint(3, 6) + 1):
        updates = []
        for a in gt["observed_state"]["actors"]:
            vx = rng.uniform(-_MAX_RANDOM_SPEED, _MAX_RANDOM_SPEED)
            vy = rng.uniform(-_MAX_RANDOM_SPEED, _MAX_RANDOM_SPEED)
            x, y = positions[a["id"]]
            x, y = x + vx * dt, y + vy * dt
            positions[a["id"]] = (x, y)
            updates.append({"id": a["id"], "entity_type": a["entity_type"],
                            "role_type": a.get("role_type"),
                            "x": round(x, 3), "y": round(y, 3),
                            "vx": round(vx, 3), "vy": round(vy, 3),
                            "speed": round((vx * vx + vy * vy) ** 0.5, 3)})
        chain.append({"depth": depth, "timestamp": round(dt * depth, 3),
                      "event_type": rng.choice(_RANDOM_EVENTS),
                      "action_type": "ego_no_action",
                      "actors_update": updates})
    return chain


def random_valid_text(gt: Dict[str, Any], seed: int = 0) -> str:
    """Schema-valid random ontology rollouts: the chance floor under the gates.
    Actors come from the root universe, motions are small random walks inside
    the kinematic bounds, and event labels are sampled from the ontology, so
    the prediction passes the schema and semantic gates while carrying no
    information about the scene's actual future. Deterministic in (seed,
    scenario_id)."""
    rng = random.Random(zlib.crc32(str(gt.get("scenario_id", "")).encode())
                        ^ (seed + 0x9E3779B9))
    declared = [copy.deepcopy(b.get("intervention")) for b in gt.get("branches", [])
                if isinstance(b.get("intervention"), dict)]
    fallback_actions = [
        {"intervention_type": "do_ego_no_action", "target_actor_id": "ego", "trigger_depth": 1},
        {"intervention_type": "do_ego_emergency_brake", "target_actor_id": "ego", "trigger_depth": 1},
    ]
    for fallback in fallback_actions:
        if len(declared) >= 2:
            break
        if fallback not in declared:
            declared.append(fallback)
    rng.shuffle(declared)
    n_branches = rng.randint(2, min(4, len(declared))) if len(declared) >= 2 else 1
    branches = [{"branch_id": f"RND_{i + 1}", "is_factual": i == 0,
                 "intervention": action, "chain": _random_chain(gt, rng)}
                for i, action in enumerate(declared[:n_branches])]
    cfs = [l4_pair_unit(cf["counterfactual_chain"]["intervention"],
                        _random_chain(gt, rng), _random_chain(gt, rng))
           for cf in _l4_query_set(gt)]
    return json.dumps({"predicted_branches": branches,
                       "predicted_counterfactuals": cfs})
