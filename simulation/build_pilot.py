"""
Pilot packager — produce a VLM-ready package per nuScenes-derived scenario:
  <out>/<scenario_id>/ground_truth.json   (twin GT for scoring)
  <out>/<scenario_id>/observed.mp4        (optional review clip up to T=0)
  <out>/<scenario_id>/prompt.md           (task + observed_state + rules)

Lets you run the full CORIVA benchmark on real data with NO server: hand the
eight extracted frames + prompt.md to a VLM (API), drop its l3_pre.json / l4_pre.json into
results/<model>/<scenario_id>/, then score with the existing evaluation pipeline
against these ground_truth.json files.

CLI:
    python3 simulation/build_pilot.py --version v1.0-mini --limit 10 --out-dir output_pilot
"""

import argparse
import json
import os
import shutil
import sys
from typing import Any, Dict, List, Optional, Tuple

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from simulation.nuscenes_adapter import (load_tables, extract_observed_trajectory,
                                          extract_observed_history)
from simulation.kinematic_twin import build_twin_gt
from simulation.qc import qc_scenario_gt
from simulation.build_nuscenes_dataset import _usable_samples, _default_spec
from simulation.observed_clip import build_observed_clip, write_model_frames
from ctms.ctms import load_ctms_environment
from simulation.protocol import (PRIMARY_TRACK, OBSERVED_FRAME_COUNT, STOP_SPEED_MPS,
                                 EGO_SWERVE_SPEED_MPS, EGO_SWERVE_LANE_WIDTH_M,
                                 track_steps, validate_condition, protocol_manifest)
from simulation.kinematic_twin import EGO_PROFILES, ENV_PROFILES

# The ego intervention set the model must reason over (matches _default_spec).
# Fallback when a scenario's own GT intervention list is not passed in.
PILOT_INTERVENTIONS = ["do_ego_emergency_brake", "do_ego_no_action", "do_ego_swerve_left"]

PROMPT_VERSION = "coriva-prompt-2026-09-23d"

# Prompt variants: arm A is the full prompt, arm B the no-parameter prompt and
# arm C the target-only prompt.
# `full` prints the published motion profiles, the scene's scripted hazard
# manoeuvre and the L4 intervention magnitude; `recipe_withheld` names the
# actions and interventions only, so the model must decide every magnitude
# and predict every other actor from the observations; `ego_profiles_withheld`
# prints the scripted hazard manoeuvre and the L4 intervention magnitude as
# `full` does but withholds the ego actions' published profiles (and, like
# `recipe_withheld`, teaches the format with the non-published example rates),
# so arm A minus arm C is the part of the prompt effect that the ego profiles
# carry and arm C minus arm B the part that the other actors' stated motion
# carries. The variant is part of the prompt identity: it changes the prompt
# hash and the frozen `prompt_version` of a campaign root, so the arms never
# share a root.
PROMPT_VARIANTS = ("full", "recipe_withheld", "ego_profiles_withheld")


def validate_prompt_variant(variant: Optional[str]) -> str:
    variant = variant or "full"
    if variant not in PROMPT_VARIANTS:
        raise ValueError(f"unknown prompt variant {variant!r}; choose one of {PROMPT_VARIANTS}")
    return variant


def prompt_version(variant: Optional[str] = None) -> str:
    """The frozen prompt identity of a campaign root: the base version for
    `full`, the base version suffixed with the variant otherwise."""
    variant = validate_prompt_variant(variant)
    return PROMPT_VERSION if variant == "full" else f"{PROMPT_VERSION}+{variant}"

_ONTOLOGY_CACHE: Dict[str, Any] = {}


def _json_block(obj: Any, allow_nan: bool = True, _level: int = 0) -> str:
    """The prompt's JSON layout. An object or array that holds no nested
    object or array (an actor, an `intervention`, a label list) stays on one
    line without spaces; every other key and array element starts a new line.

    One actor per line keeps answers short. The line breaks between closing
    brackets matter: in single-line JSON, Qwen3.5 closed every branch twice,
    because its tokenizer has `},{` and `}}` as single tokens."""
    nested = isinstance(obj, (dict, list)) and any(
        isinstance(v, (dict, list)) for v in (obj.values() if isinstance(obj, dict) else obj))
    if not nested:
        return json.dumps(obj, separators=(",", ":"), allow_nan=allow_nan)
    pad, inner = "  " * _level, "  " * (_level + 1)
    if isinstance(obj, dict):
        items = [f"{inner}{json.dumps(k)}:{_json_block(v, allow_nan, _level + 1)}"
                 for k, v in obj.items()]
        return "{\n" + ",\n".join(items) + "\n" + pad + "}"
    items = [inner + _json_block(v, allow_nan, _level + 1) for v in obj]
    return "[\n" + ",\n".join(items) + "\n" + pad + "]"


def _worked_example_section(ego_decel: float = 3.0, target_decel: float = 8.0) -> str:
    """One fixed, synthetic demonstration for every backend/split/condition.

    `ego_decel` and `target_decel` are the toy braking rates (m/s^2). The
    defaults are the ones the `full` prompt has always used; the
    `recipe_withheld` variant passes rates that are no published profile
    value, so the example teaches the format without hinting at the recipe.

    No scene inputs or reference files enter this example. Its explicit
    straight-line toy dynamics illustrate serialization, complete branches,
    a matched pair, and a terminal node; they are not a scene predictor.
    Keep the complete six-step chains: an abbreviated example encourages
    the same omissions the real output contract forbids.
    """
    def actor(actor_id: str, x: float, speed: float) -> Dict[str, Any]:
        ego = actor_id == "ego"
        return {"id": actor_id, "entity_type": "ego_vehicle" if ego else "vehicle",
                "role_type": "ego_agent" if ego else "lead_vehicle",
                "x": x, "y": 0.0, "vx": speed, "vy": 0.0, "speed": speed}

    ego_v0, target_v0, target_x0 = 6.0, 4.0, 40.0
    ego_stop_t = ego_v0 / ego_decel          # 2.0 s at the default -3 m/s^2
    target_stop_t = target_v0 / target_decel  # 0.5 s at the default -8 m/s^2

    def braked(x0: float, v0: float, decel: float, t: float) -> Tuple[float, float]:
        """Constant deceleration from T=0 until the stop, then held. Rounded to
        3 decimals so a rate such as 2.4 prints 6.3, not 6.300000000000001; the
        default rates give exact values, which rounding leaves unchanged."""
        tt = min(t, v0 / decel)
        return (round(x0 + v0 * tt - 0.5 * decel * tt * tt, 3),
                round(max(0.0, v0 - decel * t), 3))

    target_stop_x = braked(target_x0, target_v0, target_decel, target_stop_t)[0]

    def chain(brake_ego: bool = False, brake_target: bool = False) -> List[Dict[str, Any]]:
        nodes = []
        for depth in range(1, 7):
            t = depth * 0.5
            # Ego starts at 6 m/s and, when queried, brakes at -ego_decel.
            ego_x, ego_v = braked(0.0, ego_v0, ego_decel, t) if brake_ego else (ego_v0 * t, ego_v0)
            # Target starts at 4 m/s at x=40; -target_decel stops it and it holds.
            target_x, target_v = (braked(target_x0, target_v0, target_decel, t) if brake_target
                                  else (target_x0 + target_v0 * t, target_v0))
            node = {"depth": depth, "timestamp": t,
                    "event_type": "ego_brake" if brake_ego else "maintain_course",
                    "action_type": "ego_comfortable_brake" if brake_ego else "ego_no_action",
                    "actors_update": [actor("ego", ego_x, ego_v), actor("npc_1", target_x, target_v)]}
            if brake_ego and t >= ego_stop_t:  # the first sample at or after the stop
                node.update(event_type="safe_stop", terminal=True)
                del node["action_type"]
            nodes.append(node)
            if node.get("terminal"):
                break
        return nodes

    actions = ["do_ego_no_action", "do_ego_comfortable_brake"]
    intervention = {"intervention_type": "npc_sudden_brake", "target_actor_id": "npc_1", "trigger_depth": 1}
    example_input = {
        "observed_state": {"depth": 0, "timestamp": 0.0, "event_type": "observed_state",
                           "actors": [actor("ego", 0.0, 6.0), actor("npc_1", 40.0, 4.0)]},
        "ego_actions": actions, "factual_action": "do_ego_no_action",
        "target_interventions": [intervention],
    }
    answer = {
        "predicted_branches": [
            {"branch_id": f"B{i + 1}", "is_factual": i == 0,
             "intervention": {"intervention_type": action, "target_actor_id": "ego", "trigger_depth": 1},
             "chain": chain(brake_ego=i == 1)} for i, action in enumerate(actions)],
        "predicted_counterfactuals": [
            {"intervention": intervention, "control_chain": chain(), "treated_chain": chain(brake_target=True)}],
    }
    answer_json = _json_block(answer, allow_nan=False)
    return f"""## Worked example (synthetic; not this scene)
This fixed example teaches the output contract only. It is not a prediction
for the images or observed state in your task. Do not copy its actors, action
list, factual flag, numbers or outcomes into your answer.

Toy assumptions: straight unobstructed road, two vehicles on y=0, no collision
within a 3-second horizon. Sample every 0.5 s. In BOTH ego-action branches,
npc_1 continues at 4 m/s. The factual ego action is no-action at 6 m/s;
the other ego action brakes at -{ego_decel:g} m/s^2 until stopped, then stays stopped.
For the target intervention, the ego stays at 6 m/s in BOTH chains. The
control target continues at 4 m/s; the treated target brakes at -{target_decel:g} m/s^2
from T=0, reaches zero speed at {target_stop_t:g} s, then stays at x={target_stop_x:g} m.

Example input:
```json
{_json_block(example_input, allow_nan=False)}
```
Example output (complete; no omitted nodes or actors):
```json
{answer_json}
```
The target stopping does NOT make an ego `safe_stop`: the intervention chains
continue through 3 s with `maintain_course` / `ego_no_action`. Only the ego
braking branch ends early, at its first `safe_stop` node at {ego_stop_t:g} s. Its optional
`action_type` is omitted, not null. Both target chains are in ONE object.
## End of worked example
The actual task follows. Use its actor inventory, queries, factual contract,
motion assumptions and horizon, even when they differ from the example.
"""


def _ontology_vocab() -> Dict[str, Any]:
    """Exact label vocabularies the schema gate enforces — stated verbatim in
    the prompt. The dissertation's recall argument requires the ontology to be
    model input ('a missed branch is always an action the model was told
    exists'); without this section a strong model fails schema gates on
    labels it was never given."""
    if not _ONTOLOGY_CACHE:
        import yaml
        with open(os.path.join("ctms", "ontology.yaml"), encoding="utf-8") as f:
            onto = yaml.safe_load(f)
        # Labels MUST come from the schema enums — the contract the gate
        # enforces. ontology.yaml's action_type holds action FAMILIES
        # (brake/steer/passive/...), not the 11 concrete labels; listing
        # those in the prompt made Gemini emit "passive" and fail every node.
        with open(os.path.join("ctms", "schema_branch.json"), encoding="utf-8") as f:
            schema = json.load(f)
        node_props = schema["$defs"]["node"]["properties"]
        _ONTOLOGY_CACHE["event_types"] = [e for e in node_props["event_type"]["enum"]
                                          if e != "observed_state"]
        _ONTOLOGY_CACHE["action_types"] = list(node_props["action_type"]["enum"])
        _ONTOLOGY_CACHE["terminal_types"] = list(onto["terminal_event_types"])
    return _ONTOLOGY_CACHE


def _lateral_words(lat: float) -> str:
    if lat == 0.0:
        return "no lateral velocity"
    side = "-y (left)" if lat < 0 else "+y (right)"
    return f"lateral velocity {abs(lat):g} m/s towards {side}"


# CARLA vehicle controllers turn a lateral command into a normalised steering
# intensity |lat| / 3.5 (simulation/generate._apply_env_behavior).
_CARLA_STEER_NORMALISER_MPS = 3.5


def declared_intervention_profile(gt: Dict[str, Any], intervention: Dict[str, Any],
                                  env_behavior: Optional[Dict[str, Dict[str, float]]] = None
                                  ) -> Dict[str, Any]:
    """The intervention parameters the SCENE's reference actually applies to
    the named actor. Sources, in order:

    1. `resolved_accel_mps2` / `resolved_lateral_velocity_mps` on the query
       item itself. NO released query carries them today: the nuScenes
       builder keeps them as private twin controls
       (`nuscenes_admission.canonical_environment_query`) and
       `build_twin_gt` copies only the (type, target, trigger) tuple into
       `l4_queries`, so `write_protocol_sidecars` never writes them. The
       branch is reserved for the planned sidecar change that publishes
       them for both splits.
    2. The CARLA scene configuration's executed `env_behavior[target]`
       (`scenario_config.json`, read by `run_model`; this file is in the
       release bundle, is NOT a manifest input asset, and is hashed into
       the frozen input fingerprint whenever present,
       `smoke_admission._OPTIONAL_PROMPT_SOURCE_ASSETS`).
    3. The CARLA `environment_event.params.intensity` mapped by event type,
       road-entry direction towards the ego lane from the target's T=0 side.
    4. The twin's `ENV_PROFILES`, road-entry direction resolved from the
       target's side as `nuscenes_adapter._probe_query` does. This is what
       every nuScenes scene gets today, and it equals the twin's own
       resolved profile, so the printed values are the reference's.

    The magnitudes differ between splits (CARLA-400: sudden brake -6 or -5
    m/s^2, cut-in steering input 1.8, pedestrian 2.0 m/s, cyclist steering
    input 2.5; twin: -8,
    2.0, 1.6, 4.0), so a prompt that printed the twin values on a CARLA scene
    would tell the model the wrong reference."""
    itype = intervention.get("intervention_type")
    target = str(intervention.get("target_actor_id"))
    root = gt.get("observed_state") or {}
    ty = next((float(a.get("y", 0.0)) for a in root.get("actors", []) if str(a.get("id")) == target), None)
    # Same side rule as nuscenes_adapter._probe_query: an actor on the +y
    # (right) side enters towards -y, otherwise towards +y.
    towards_lane = (lambda mag: -abs(mag) if (ty or 0.0) > 0.0 else abs(mag))
    if intervention.get("resolved_accel_mps2") is not None or \
            intervention.get("resolved_lateral_velocity_mps") is not None:
        return {"accel": float(intervention.get("resolved_accel_mps2", 0.0)),
                "lat": float(intervention.get("resolved_lateral_velocity_mps", 0.0)),
                "source": "released query (profile resolved for this scene)"}
    behaviour = (env_behavior or {}).get(target)
    if behaviour is not None:
        return {"accel": float(behaviour.get("accel", 0.0)), "lat": float(behaviour.get("lat", 0.0)),
                "source": "scene configuration executed by the simulator"}
    event = gt.get("environment_event") or {}
    params = event.get("params") or {}
    if event.get("event_type") == itype and params.get("intensity") is not None:
        mag = float(params["intensity"])
        if itype == "npc_sudden_brake":
            return {"accel": -mag, "lat": 0.0, "source": "scene configuration executed by the simulator"}
        if itype == "npc_cut_in_left":
            return {"accel": 0.0, "lat": -mag, "source": "scene configuration executed by the simulator"}
        if itype == "npc_cut_in_right":
            return {"accel": 0.0, "lat": mag, "source": "scene configuration executed by the simulator"}
        if itype in ("pedestrian_enters_road", "cyclist_enters_road"):
            return {"accel": 0.0, "lat": towards_lane(mag), "source": "scene configuration executed by the simulator"}
    prof = ENV_PROFILES.get(itype)
    if prof is None:
        return {"accel": None, "lat": None, "source": "no published profile"}
    lat = float(prof["lat"])
    if itype in ("pedestrian_enters_road", "cyclist_enters_road") and ty is not None:
        lat = towards_lane(lat)
    return {"accel": float(prof["accel"]), "lat": lat,
            "source": "published twin profile, direction resolved from the target's side"}


def public_intervention_profiles(root: Dict[str, Any], query: Dict[str, Any],
                                 scenario_config: Optional[Dict[str, Any]] = None,
                                 split: Optional[str] = None) -> Dict[str, Dict[str, Any]]:
    """Resolve the same public profiles for prompting and reference-free L4.

    No future reference is an input. CARLA's optional configuration supplies
    either the executed actor map or the declared event intensity; both are
    already covered by the frozen input fingerprint.
    """
    config = scenario_config or {}
    context = {"observed_state": root, "environment_event": config.get("environment_event")}
    profiles = {}
    actors = {a['id']: a for a in root.get('actors', []) if isinstance(a, dict)}
    carla = (split or '').lower() == 'carla'
    for item in (query.get('l4_interventions') or []):
        if not isinstance(item, dict):
            continue
        profile = declared_intervention_profile(context, item, config.get('env_behavior'))
        if carla and profile.get('source', '').startswith('published twin profile'):
            raise ValueError('CARLA L4 requires its published scene profile in query.json or scenario_config.json; '
                             'the twin default is not a CARLA intervention magnitude')
        target = actors.get(item.get('target_actor_id'), {})
        profile.update(direction_frame='control_target_normal' if carla else 't0',
                       lateral_realization='steering' if carla and target.get('entity_type') != 'pedestrian'
                       else 'velocity')
        profiles[item['intervention_type']] = profile
    return profiles


def declared_hazard_profile(gt: Dict[str, Any], critical_actor_id: Optional[str],
                            env_behavior: Optional[Dict[str, Dict[str, float]]] = None
                            ) -> Optional[Dict[str, Any]]:
    """The manoeuvre the L3 references script for the scene's hazard actor in
    EVERY branch. CARLA: the critical
    actor's executed `env_behavior` (scene configuration; e.g. the lead
    brakes at -6 m/s^2 from T=0 in all three branches). nuScenes: the
    template profile recorded as `scenario_template.hazard_profile` (lead
    -4 m/s^2, road entry at the published entry speed); a cut-in candidate
    keeps its observed motion and yields None. Returns {actor_id, accel,
    lat} or None when the scene scripts nothing."""
    template = gt.get("scenario_template") or {}
    actor_id = template.get("selected_actor_id") or critical_actor_id
    if actor_id is None:
        return None
    profile = None
    if template.get("hazard_profile") is not None:
        profile = template["hazard_profile"]
    elif env_behavior and str(actor_id) in env_behavior:
        profile = env_behavior[str(actor_id)]
    if profile is None:
        return None
    accel = float(profile.get("accel", 0.0)); lat = float(profile.get("lat", 0.0))
    if accel == 0.0 and lat == 0.0:
        return None
    return {"actor_id": str(actor_id), "accel": accel, "lat": lat}


def factual_branch_section(factual_action: Optional[str], factual_reference: Optional[str],
                           l3_actions: List[str], variant: str = "full") -> str:
    """The factual-branch contract.

    The L3 references of the two splits define the factual branch
    differently, and the model must be told which declared action it is and
    what it is scored against: on nuScenes it is the RECORDED continuation of
    every actor, labelled by the band of the recorded ego acceleration
    (`nuscenes_adapter.FACTUAL_ACTION_BANDS`), so the published profile and
    the scripted hazard manoeuvre do not apply to it; on CARLA the simulator
    executed the factual action with the published profile like every other
    branch. Without this section the prompt would ask for the profile in
    every branch while the nuScenes factual branch is scored against a
    recording that only lies in the label's band, so an answer that followed
    the instruction could score below one that copied the recording."""
    if not factual_action or factual_action not in l3_actions:
        return ""
    lines = ["## Factual branch",
             f"The declared action `{factual_action}` is the FACTUAL branch: mark that branch",
             "`is_factual: true` and every other branch `is_factual: false`."]
    variant = validate_prompt_variant(variant)
    withheld = variant == "recipe_withheld"
    ego_withheld = variant == "ego_profiles_withheld"
    if factual_reference == "recorded_future":
        from simulation.nuscenes_adapter import FACTUAL_ACTION_BANDS
        band = dict(FACTUAL_ACTION_BANDS).get(factual_action, "the band of that label")
        lines += [
            "Its reference is the RECORDED continuation of this scene: the ego's and every",
            "other actor's recorded states over the horizon, not a generated rollout. The",
            f"label names only the band of the ego's recorded mean acceleration ({band});"]
        if withheld:
            lines += [
                "for this branch predict what actually happened, from the observations. Every",
                "OTHER declared action is scored against a generated rollout of that action."]
        elif ego_withheld:
            lines += [
                "the hazard actor's recorded motion need not follow the scripted manoeuvre. For",
                "this branch predict what actually happened, from the observations. The scripted",
                "hazard manoeuvre applies to every OTHER declared action, each of which is scored",
                "against a generated rollout of that action."]
        else:
            lines += [
                "the recorded motion need not follow the published profile of that action, and the",
                "hazard actor's recorded motion need not follow the scripted manoeuvre. For this",
                "branch predict what actually happened, from the observations. The published",
                "profiles and the scripted hazard manoeuvre apply to every OTHER declared action,",
                "whose references are generated."]
    else:
        lines += (["It is scored like every other branch."] if (withheld or ego_withheld) else [
            "The simulator executed it with the published profile of that action, like every",
            "other declared action, so it is scored like every other branch."])
    return "\n" + "\n".join(lines) + "\n"


def withheld_motion_rules(split: Optional[str] = None) -> str:
    """The `recipe_withheld` counterpart of `motion_profile_rules`: the model
    is told that no magnitude is supplied and what the directions mean, and
    nothing about the published values, the lane width, the swerve speed or
    the scripted hazard behaviour."""
    carla = (split or "").lower() == "carla"
    lines = ["## Motion parameters: not supplied",
             "No published motion profile is given for any declared ego action, no",
             "magnitude for the named target intervention, and no scripted behaviour for",
             "any other actor. The declared actions and interventions are NAMED in the",
             "query only. Decide each ego action's longitudinal and lateral motion from",
             "the action name, the observed motion and your own knowledge of driving.",
             "A brake slows the ego along its direction of travel and speed never goes",
             "below 0 (a stopped ego holds). A swerve moves the ego sideways, left =",
             "towards -y and right = towards +y, on top of its road-following path;",
             "`vy` carries the lateral component while the offset grows and returns to",
             "the road-following value once held; `speed` = sqrt(vx^2 + vy^2) throughout.",
             *(["CARLA executes a swerve as a lane change on the road: the offset is",
                "perpendicular to the road centreline and turns with the road, so project",
                "it into BOTH fixed T0 coordinates as the coordinate convention states."]
               if carla else
               ["On nuScenes the ego follows the observed road curvature and the lateral",
                "motion is added in the fixed T=0 frame on top of that road-following path."]),
             "Predict every non-ego actor's behaviour from the observations; for the",
             "named target intervention choose the magnitude yourself."]
    return "\n".join(lines) + "\n"


def ego_profiles_withheld_motion_rules(split: Optional[str] = None) -> str:
    """The `ego_profiles_withheld` counterpart of `motion_profile_rules` (arm
    C): the model is told that no published profile is given for the ego
    actions and what the directions mean, in the words of
    `withheld_motion_rules`, while the named intervention's magnitude and the
    scripted hazard manoeuvre are printed under `## Motion profiles of this
    scene` exactly as in `full`."""
    carla = (split or "").lower() == "carla"
    lines = ["## Motion parameters of the ego actions: not supplied",
             "No published motion profile is given for any declared ego action: the",
             "declared ego actions are NAMED in the query only. Decide each ego action's",
             "longitudinal and lateral motion from the action name, the observed motion and",
             "your own knowledge of driving.",
             "A brake slows the ego along its direction of travel and speed never goes",
             "below 0 (a stopped ego holds). A swerve moves the ego sideways, left =",
             "towards -y and right = towards +y, on top of its road-following path;",
             "`vy` carries the lateral component while the offset grows and returns to",
             "the road-following value once held; `speed` = sqrt(vx^2 + vy^2) throughout.",
             *(["CARLA executes a swerve as a lane change on the road: the offset is",
                "perpendicular to the road centreline and turns with the road, so project",
                "it into BOTH fixed T0 coordinates as the coordinate convention states."]
               if carla else
               ["On nuScenes the ego follows the observed road curvature and the lateral",
                "motion is added in the fixed T=0 frame on top of that road-following path."]),
             "The named target intervention's magnitude and the scripted manoeuvre of the",
             "scene's hazard actor ARE given under `## Motion profiles of this scene`, applied",
             "from T=0 (the depth-1 state already includes 0.5 s of them): use them as stated.",
             "Predict every other non-ego actor's behaviour from the observations."]
    return "\n".join(lines) + "\n"


def motion_profile_rules(split: Optional[str] = None,
                         factual_reference: Optional[str] = None,
                         variant: str = "full") -> str:
    """How the published motion profiles are applied: the same for every
    scene of a split, so it belongs to the fixed instructions. The values of
    the scene's declared actions are in `motion_profile_section`."""
    variant = validate_prompt_variant(variant)
    if variant == "recipe_withheld":
        return withheld_motion_rules(split)
    if variant == "ego_profiles_withheld":
        return ego_profiles_withheld_motion_rules(split)
    carla = (split or "").lower() == "carla"
    recorded_factual = factual_reference == "recorded_future"
    lines = ["## Published motion profiles of the declared actions",
             *(["Every declared EGO action except the factual branch (whose reference is the",
                "recording, see `## Factual branch`) is produced with the published parameters",
                "listed under `## Motion profiles of this scene`,"]
               if recorded_factual else
               ["Every declared EGO action is produced with the published parameters listed",
                "under `## Motion profiles of this scene`,"]),
             "applied from T=0, so the depth-1 state already includes 0.5 s of it.",
             "Longitudinal acceleration acts along the ego's direction of travel; speed",
             "never goes below 0 (a braking ego reaches 0 and holds). A swerve moves the",
             f"ego sideways at {EGO_SWERVE_SPEED_MPS:g} m/s (left = towards -y, right = towards +y) until it is",
             f"one lane width ({EGO_SWERVE_LANE_WIDTH_M:g} m) from its road-following path, then holds that",
             "offset; `vy` carries the lateral component while the offset grows and returns",
             "to the road-following value once held; `speed` = sqrt(vx^2 + vy^2) throughout.",
             *(["CARLA executes the swerve in the simulator as a lane change on the road: the",
                "offset is measured perpendicular to the road centreline at the ego's current",
                "position and turns with the road. Project the offset along the route's",
                "right unit normal into BOTH fixed T0 coordinates: delta_x = -d * sin(theta),",
                "delta_y = d * cos(theta). Here d is the signed offset in metres (right positive,",
                "left negative); theta is the route heading relative to the T0 +x axis,",
                "positive towards +y. Add these components to the centreline position at the",
                "same route progress; do not add d directly to the T0 y coordinate.",
                "For example, at theta = 60 degrees a right offset d = 3.5 m contributes",
                "delta_x = -3.031 m and delta_y = 1.75 m."]
               if carla else
               ["On nuScenes the ego follows the observed road curvature (a bicycle model",
                "whose steering is estimated from the motion before T=0) and the lateral",
                "velocity is added in the fixed T=0 frame on top of that road-following path;",
                "the non-learning control applies the same rule."]),
             "Follow these values; a branch that brakes or steers at another rate is a",
             "different prediction from the reference."]
    return "\n".join(lines) + "\n"


WITHHELD_SCENE_SECTION = """## Motion profiles of this scene
Not supplied. Neither the ego actions' accelerations, nor the named actor's
intervention magnitude, nor any other actor's future behaviour is given for
this scene: decide them from the action names, the observations and your own
knowledge of driving, and predict every other actor's behaviour from the
observations.
"""


def motion_profile_section(l3_actions: List[str], interventions: List[str],
                           root: Dict[str, Any], critical_actor_id: Optional[str],
                           intervention_profiles: Optional[Dict[str, Dict[str, Any]]] = None,
                           split: Optional[str] = None,
                           hazard_profile: Optional[Dict[str, Any]] = None,
                           factual_reference: Optional[str] = None,
                           variant: str = "full") -> str:
    """The published parameters of the declared ego actions (`EGO_PROFILES`,
    which CARLA's generator mirrors and the control applies) and the
    intervention parameters declared for THIS scene (`intervention_profiles`,
    from `declared_intervention_profile`; the twin default when absent).
    Stating them removes the control's advantage of knowing the reference's
    deceleration while the model has to guess it. The CARLA intervention
    magnitudes are the scene's own,
    not the twin's, and CARLA realises them through its actor controllers
    rather than as a constant point-mass profile. How the values are applied
    is stated once in `motion_profile_rules`. The `recipe_withheld` variant
    states that nothing is supplied; `ego_profiles_withheld` states that the
    ego profiles are not supplied and prints the rest as `full` does."""
    variant = validate_prompt_variant(variant)
    if variant == "recipe_withheld":
        return WITHHELD_SCENE_SECTION
    ego_withheld = variant == "ego_profiles_withheld"
    carla = (split or "").lower() == "carla"
    lines = ["## Motion profiles of this scene"]
    if ego_withheld:
        lines.append("Ego actions: not supplied. No acceleration or lateral rule is given for any")
        lines.append("declared ego action; decide each from the action name, the observations and")
        lines.append("your own knowledge of driving (see `## Motion parameters of the ego actions`).")
    for action in ([] if ego_withheld else l3_actions):
        prof = EGO_PROFILES.get(action)
        if prof is None:
            lines.append(f"- `{action}`: no published profile")
            continue
        lateral = _lateral_words(prof['lat'])
        if prof['lat'] != 0.0:
            lateral += f" until one lane width ({EGO_SWERVE_LANE_WIDTH_M:g} m), then hold"
        lines.append(f"- `{action}`: longitudinal acceleration {prof['accel']:+g} m/s^2, {lateral}")
    if interventions:
        lines.append("")
        lines.append("Target-actor intervention parameters declared for the named actor in THIS scene, applied")
        lines.append("from T=0 (the depth-1 state already includes 0.5 s of it) while the ego follows")
        lines.append("`do_ego_no_action`:" if ego_withheld else "`do_ego_no_action` (acceleration 0):")
        actors_by_id = {str(a.get("id")): a for a in root.get("actors", [])}
        ty = actors_by_id.get(str(critical_actor_id), {}).get("y")
        for itype in interventions:
            prof = (intervention_profiles or {}).get(itype)
            if prof is None:
                prof = declared_intervention_profile({"observed_state": root},
                                                     {"intervention_type": itype, "target_actor_id": critical_actor_id})
            if prof.get("accel") is None:
                lines.append(f"- `{itype}`: no published profile")
                continue
            accel = float(prof["accel"]); lat = float(prof["lat"])
            if itype in ("pedestrian_enters_road", "cyclist_enters_road"):
                side = f" (y = {float(ty):g} at T=0)" if ty is not None else ""
                words = (f"{_lateral_words(lat)}, i.e. towards the ego lane from its side{side}; "
                         f"longitudinal velocity kept")
            else:
                words = f"longitudinal acceleration {accel:+g} m/s^2, {_lateral_words(lat)}"
            lines.append(f"- `{itype}`: {words}")
        if carla:
            lines.append("These are the values of the scene configuration that the CARLA simulator")
            lines.append("executed for the named actor: a vehicle's deceleration is applied by its")
            lines.append("controller, a walker's lateral value is its crossing speed, and a vehicle's")
            lines.append(f"or cyclist's lateral value is a steering manoeuvre of normalised intensity")
            lines.append(f"|lateral| / {_CARLA_STEER_NORMALISER_MPS:g} towards that side, not a constant lateral velocity.")
            lines.append("They are not the twin's default intervention profile.")
        else:
            lines.append("The nuScenes twin applies these values as a constant point-mass profile from")
            lines.append("the first step; the non-learning control does the same.")
    lines.append("")
    recorded = factual_reference == "recorded_future"
    scope = "every GENERATED (non-factual) ego-action branch" if recorded else "EVERY ego-action branch"
    if hazard_profile is not None:
        lines.append(f"In {scope} the reference gives the scene's hazard actor `{hazard_profile['actor_id']}`")
        lines.append(f"one scripted manoeuvre from T=0: longitudinal acceleration {hazard_profile['accel']:+g} m/s^2,")
        lines.append(f"{_lateral_words(hazard_profile['lat'])}. Use it in {'those' if recorded else 'all'} branches; it does not depend on")
        lines.append("the ego action. Every OTHER non-ego actor's behaviour must be predicted from")
        lines.append("the observations." + (" In the factual branch every actor is recorded (see" if recorded else ""))
        if recorded:
            lines.append("`## Factual branch`).")
    else:
        lines.append(f"The references script no manoeuvre for any other actor in {scope}:")
        lines.append("every non-ego actor's behaviour must be predicted from the observations.")
    return "\n".join(lines) + "\n"


def build_prompt_parts(root: Dict[str, Any], scenario_id: str,
                       interventions: Optional[List[str]] = None,
                       critical_actor_id: Optional[str] = None,
                       condition: Optional[str] = None, track: str = PRIMARY_TRACK,
                       observed_history: Optional[Dict[str, Any]] = None,
                       l3_actions: Optional[List[str]] = None,
                       ontology: Optional[Dict[str, Any]] = None,
                       query: Optional[Dict[str, Any]] = None,
                       intervention_profiles: Optional[Dict[str, Dict[str, Any]]] = None,
                       split: Optional[str] = None,
                       hazard_profile: Optional[Dict[str, Any]] = None,
                       factual_action: Optional[str] = None,
                       factual_reference: Optional[str] = None,
                       variant: str = "full") -> Tuple[str, str]:
    """Compose the condition-specific, input-only VLM prompt as (system, user).

    The system part holds the instructions that are the same for every scene
    of a split and condition (worked example, conventions, contract, rules);
    the user part holds this scene. Neither carries the scenario id or the
    prompt version: the model needs neither, and run manifests record both.

    `intervention_profiles` maps each queried intervention type to the
    parameters the scene's reference applies (`declared_intervention_profile`);
    `run_model` supplies them from the released query (nuScenes) or the CARLA
    scene configuration. Without them the released query's resolved values,
    then the twin defaults, are used. `variant` selects the prompt variant
    (`PROMPT_VARIANTS`): `recipe_withheld` prints no motion parameter, no
    hazard manoeuvre and no intervention magnitude; `ego_profiles_withheld`
    prints the hazard manoeuvre and the intervention magnitude but no ego
    profile, with the non-published example rates of `recipe_withheld`."""
    condition = condition or "video"
    validate_condition(condition)
    variant = validate_prompt_variant(variant)
    withheld = variant == "recipe_withheld"
    example_withheld = variant != "full"      # `recipe_withheld` and `ego_profiles_withheld`
    n_steps = track_steps(track)
    actor_ids = [a["id"] for a in root["actors"]]
    actor_identity = {a["id"]: {"entity_type": a.get("entity_type"),
                                "role_type": a.get("role_type")}
                      for a in root["actors"]}
    vocab = _ontology_vocab()
    interv_list = ", ".join(interventions or [])
    input_description = {
        "root_only": "Legacy text-only root condition; this condition is not part of the final protocol.",
        "video": f"Exactly {OBSERVED_FRAME_COUNT} consecutive overlaid CAM_FRONT images are provided; no numeric trajectory history is provided.",
        "history_json": "Exactly eight numeric observed states are provided below; no images are provided.",
        "video_history": f"Exactly {OBSERVED_FRAME_COUNT} consecutive overlaid CAM_FRONT images and the same eight numeric observed states are provided.",
    }[condition]
    history_section = ""
    if condition in {"history_json", "video_history"}:
        if observed_history is None:
            raise ValueError(f"condition {condition} requires observed_history")
        history_section = "\n## observed_history (T=-3.5...0.0)\n```json\n" + \
            _json_block(observed_history) + "\n```\n"
    action_list = l3_actions or (interventions or PILOT_INTERVENTIONS)
    ontology_section = ("\n## ontology\n```json\n" + _json_block(ontology)
                        + "\n```\n") if ontology is not None else ""
    query_section = ("\n## explicit_query\n```json\n" + _json_block(query)
                     + "\n```\n") if query is not None else ""
    if intervention_profiles is None and query is not None:
        intervention_profiles = {
            item.get("intervention_type"): declared_intervention_profile({"observed_state": root}, item)
            for item in (query.get("l4_interventions") or []) if isinstance(item, dict)}
    if query is not None:
        factual_action = factual_action or query.get("factual_action")
        factual_reference = factual_reference or query.get("factual_reference")
    if factual_action and not factual_reference:
        factual_reference = ("executed_action_profile" if (split or "").lower() == "carla"
                             else "recorded_future")
    profile_rules = motion_profile_rules(split, factual_reference, variant)
    profile_section = motion_profile_section(
        list(action_list), list(interventions or []), root, critical_actor_id,
        intervention_profiles, split, hazard_profile, factual_reference, variant)
    factual_section = factual_branch_section(factual_action, factual_reference, list(action_list), variant)
    # -2.4 and -3.2 m/s^2 are no published ego profile, twin intervention or
    # CARLA scene value, so the example cannot hint at the recipe.
    worked_example = (_worked_example_section(ego_decel=2.4, target_decel=3.2) if example_withheld
                      else _worked_example_section())
    brake_note = ("(a braking ego moves less than speed × 0.5 s)" if example_withheld else
                  "(with the published acceleration,\n   a braking ego moves less than speed × 0.5 s)")
    treated_clause = ("applies the named intervention (its magnitude is your judgement)" if withheld
                      else "applies the intervention declared above")
    control_baseline = (
        "retains its baseline speed along its road/lane, turning with the route, with no named manoeuvre"
        if (split or "").lower() == "carla" else
        "keeps its T=0 velocity vector (speed and heading) in the fixed T0 frame, with no named manoeuvre")
    system = f"""# CORIVA prediction task
Predict the requested futures, not a recommended driving action. Complete
both tasks for the scene that follows. The final answer must be one JSON
object containing `predicted_branches` and `predicted_counterfactuals`.

{worked_example}

## Input condition: {condition}
{input_description}

## Coordinate convention
All positions and velocities use one fixed T0 ego frame. The ego position at
T0 is the origin; +x is forward, +y is right, and -y is left. The axes do not
rotate during the predicted future. These signs name sides of the T0 frame;
a route-relative manoeuvre on a curve must be projected into both fixed axes.
{ontology_section}
{profile_rules}
## OUTPUT CONTRACT (strict — schema-validated, one JSON object)
Use the structure of the complete example, with the ACTUAL task's data.
- Top-level keys: `predicted_branches`, `predicted_counterfactuals`.
- Each ego-action branch: `branch_id`, `is_factual`, `intervention`, `chain`.
- Each target-intervention pair: `intervention`, `control_chain`, `treated_chain`.
- Each intervention: `intervention_type`, `target_actor_id`, `trigger_depth`.
- Each future node: `depth`, `timestamp`, `event_type`, `actors_update`, plus
  `action_type` for a non-terminal node or `terminal: true` for a terminal.
- Each `actors_update` is an ARRAY of full actor objects, not a dictionary
  keyed by actor id and not a list of changes. Include each actual actor id
  exactly once.
- For each actor output `id`, `entity_type`, `role_type`, `x`, `y`, `vx`,
  `vy`, `speed`. Do not copy input-only metadata such as `source_instance_token`.

All numeric values must be computed finite JSON numbers. For example, output
`1.35`, not `2.7 * 0.5`, `"1.35"`, NaN or Infinity. No formulas,
comments, placeholders, ellipses, "same as above", or null values. When an
optional field is not applicable, omit the key; never set `action_type` to
null or an object. Omit `action_type` on collision nodes; on other terminal
nodes it may be omitted. Do not echo the observed root as a future node.
Lay out the JSON like the JSON in this prompt: write every object that holds
no nested object or array (each actor, each `intervention`) on one line
without spaces, and put every other key and array element on its own line.
Never spread an actor's fields over several lines. Still include every
required branch, chain, node and actor. Return no explanation or reasoning
transcript in the final answer.

## Node label lookup (non-terminal nodes only)
Use the row for the branch's EGO action. Both target-intervention chains use
the no-action row. A target manoeuvre never changes this row. Terminal events
take precedence and follow rules 4-6 below.

| Declared ego action | `event_type` | `action_type` |
| --- | --- | --- |
| do_ego_no_action | maintain_course | ego_no_action |
| do_ego_decelerate_coast | maintain_course | ego_decelerate_coast |
| do_ego_emergency_brake | ego_brake | ego_emergency_brake |
| do_ego_comfortable_brake | ego_brake | ego_comfortable_brake |
| do_ego_panic_brake | ego_brake | ego_panic_brake |
| do_ego_swerve_left | ego_steer | ego_swerve_left |
| do_ego_swerve_right | ego_steer | ego_swerve_right |
| do_ego_brake_and_swerve_left | ego_combined_maneuver | ego_brake_and_swerve_left |
| do_ego_brake_and_swerve_right | ego_combined_maneuver | ego_brake_and_swerve_right |
| do_ego_accelerate_pass | ego_accelerate | ego_accelerate_pass |
| do_ego_accelerate_escape | ego_accelerate | ego_accelerate_escape |

Full schema `event_type` vocabulary (for reference, NOT a menu of labels to
choose freely for this task; the lookup above and terminal rules govern):
{", ".join(vocab["event_types"])}

Allowed `action_type` labels (exact strings, nothing else):
{", ".join(vocab["action_types"])}

Terminal `event_type` labels: {", ".join(vocab["terminal_types"])}. The references
only ever end with `collision` or `safe_stop`; `out_of_bounds` is in the
vocabulary but no reference uses it, so a chain ending in it can never match.
Describe an off-road outcome through the states and end the chain at the
horizon, or with the collision that follows.

## Rules
1. Reuse only the observed actor ids; never invent ids or change an actor's
   `entity_type`/`role_type`. One exception: on a `collision` node, the
   actor(s) the ego collides with carry `"role_type": "impacted_actor"`.
   Every actors_update entry carries `id`, `entity_type`, `role_type`, `x`,
   `y`, `vx`, `vy`, `speed`. EVERY node is a full snapshot: it lists every
   actor of the observed state, including actors that do not move (repeat
   their state). An actor missing from a node scores zero for that node,
   and the named target must be present from the first node onwards in both
   the control and the treated chain.
2. Depth increases by 1 and timestamps are exactly 0.5, 1.0, 1.5, …
   (depth × 0.5 s after T=0). A chain has {n_steps} steps unless a terminal
   event ends it earlier; a terminal event may occur at depth 1.
3. Keep positions and velocities consistent from the T=0 state onwards.
   `vx`, `vy` and `speed` (= sqrt(vx^2 + vy^2)) are the instantaneous values
   at the node's timestamp; the displacement over each 0.5 s step must agree
   with the average velocity over that step {brake_note}. The checked bounds are
   |acceleration| <= 10 m/s^2, speed <= 45 m/s and <= 25 m per step. All
   coordinates are the fixed T=0 ego frame, which does not rotate: a vehicle
   on a curved road or a swerving ego has nonzero `vy`; do not force `vy` to
   0. Crossing pedestrians/cyclists have nonzero `vy`.
4. Omit the `terminal` field on non-terminal nodes; set `"terminal": true`
   only on the final node when `event_type` is a terminal label. Reaching
   the end of the horizon is not a terminal event.
5. `collision`: the collision node is the last sampled state AT OR BEFORE
   contact. If contact falls between two samples, the node is the earlier
   sample: report the actors' states at that timestamp (not yet
   overlapping) and label it `collision` because contact happens within the
   following 0.5 s. If contact falls exactly on a sample, that sample is the
   collision node and its states are at contact. Mark the actor(s) hit as
   `impacted_actor` and omit `action_type` on collision nodes. Only a
   contact inside the horizon is a collision: if you expect contact after
   the last sample, do not label the last sample `collision`; end the chain
   at the horizon without a terminal event.
6. `safe_stop`: only when NO collision occurs within the horizon (a
   collision inside the horizon always takes precedence, also for a stopped
   ego that another actor hits: then the chain ends with `collision`), the
   ego is at or below {STOP_SPEED_MPS:g} m/s at that node and stays at or below it at
   every later step of the horizon. Label the FIRST such node `safe_stop`
   and end the chain there; do not keep emitting stopped nodes. A stop
   followed by motion is not a terminal event; keep predicting the motion
   instead.
7. `event_type` on every node names what the EGO is doing at that step, or a
   terminal event; it never names another actor's behaviour. On every
   NON-terminal node, use `maintain_course` with `action_type: ego_no_action`
   for the no-action branch and for BOTH intervention chains (the ego keeps its fixed
   no-action policy in the control and in the treated chain). A terminal node follows
   rules 4-6 instead: `collision`, `safe_stop` or `out_of_bounds`, with no
   `action_type` on a collision node. For the other ego actions the label of
   every non-terminal node follows the action: a brake action (`ego_emergency_brake`,
   `ego_comfortable_brake`, `ego_panic_brake`) is `ego_brake`; a swerve
   (`ego_swerve_left`, `ego_swerve_right`) is `ego_steer`; an acceleration
   (`ego_accelerate_pass`, `ego_accelerate_escape`) is `ego_accelerate`; a
   brake-and-swerve is `ego_combined_maneuver`; `ego_decelerate_coast` is
   `maintain_course`. Do not use `lane_follow`, `npc_brake`, `npc_cut_in`,
   `npc_yield`, `pedestrian_cross`, `cyclist_cross`, `obstacle_appear` or
   `near_miss`: the other actor's event is expressed only by the
   `intervention` object and by its states in `actors_update`. A node whose
   `event_type` differs from the reference convention scores zero.
8. Output ONLY the JSON object — no prose outside the single JSON block.
"""
    user = f"""## Scene
Expected output: {len(action_list)} ego-action branches and {len(interventions or [])} target-intervention pairs
({len(action_list) + 2 * len(interventions or [])} future chains in total).
Each non-terminal chain runs for {n_steps} steps at 0.5 s per step.

The observed state at T=0 is given below (the closed world of actors — you
must reuse ONLY these ids: {actor_ids}; do not introduce new ids).

## observed_state (T=0)
```json
{_json_block(root)}
```
{history_section}
{query_section}
{profile_section}
## Task — Ego-action branching
Predict exactly one future branch for every declared scenario-applicable ego
action in [{", ".join(action_list)}]. Do not add or omit actions. Each branch
is a root-to-leaf chain. Exactly one branch has `is_factual: true`.
Put each action in a separate `predicted_branches` object with a unique
`branch_id` and that action's intervention metadata (`target_actor_id: ego`).
The target-actor interventions belong ONLY in `predicted_counterfactuals`.
Return all branches for the queried actions, including those leading to
collision, rather than selecting only the most likely or safest branch.
{factual_section}
## Task — Target-actor intervention
For EACH intervention in [{interv_list}], predict TWO chains from the same
observed root under the same fixed ego policy `do_ego_no_action`:
- `control_chain`: the named actor `{critical_actor_id or 'ego'}` {control_baseline}.
  Every other actor: your prediction from the observations, under the same
  non-target assumptions in both conditions.
- `treated_chain`: `{critical_actor_id or 'ego'}` {treated_clause}
  from T=0 (`trigger_depth: 1`). Every other actor: the same assumptions as in
  `control_chain`.
The only intended difference between the two chains is `{critical_actor_id or 'ego'}`'s
behaviour. The ego keeps `do_ego_no_action` in BOTH chains, even where braking
or steering would be sensible; its state may differ between the chains only
through a physical interaction (a collision). Both chains are required: a
query with one chain missing is incomplete.
For each listed intervention, put `intervention`, `control_chain` and
`treated_chain` together in ONE `predicted_counterfactuals` object. Never
split control and treated into separate objects or duplicate the intervention.
If the intervention list is empty, return an empty array for
`predicted_counterfactuals`.

## Actor identity
Per-actor identity is FIXED — repeat these exact `entity_type`/`role_type`
values in every `actors_update` entry, and include each of these actor ids
exactly once in every node: {actor_ids}.
```json
{_json_block(actor_identity)}
```

Before finalizing, check: exactly {len(action_list)} ego-action branches;
exactly {len(interventions or [])} complete target-intervention pairs; correct factual flag;
all {len(actor_ids)} actors in every node; {n_steps} nodes unless a terminal occurs;
computed numbers and the node labels of the lookup table. Do not output this checklist.
"""
    return system.strip() + "\n", user.strip() + "\n"


def join_prompt_parts(system: str, user: str) -> str:
    """The single-message form, for backends without a system role."""
    return f"{system}\n\n{user}"


def build_prompt(*args, **kwargs) -> str:
    """`build_prompt_parts` joined into one prompt text."""
    return join_prompt_parts(*build_prompt_parts(*args, **kwargs))


def build_pilot(dataroot: str, version: str, limit: int, out_dir: str,
                env: Optional[Dict[str, Any]] = None, n_steps: int = 6, dt: float = 0.5,
                samples_per_scene: int = 3, max_actors: int = 5, radius: float = 60.0,
                with_clip: bool = True) -> Dict[str, Any]:
    """Package up to `limit` QC-accepted scenarios with GT + clip + prompt.

    The pilot emits the same primary-track artifacts as the reviewed nuScenes
    builder.
    """
    if env is None:
        env = load_ctms_environment("ctms/ontology.yaml", "ctms/ctms_config.yaml",
                                    "ctms/schema_branch.json", "ctms/schema_counterfactual.json")
    tables = load_tables(dataroot, version)
    os.makedirs(out_dir, exist_ok=True)

    scenarios: List[Dict[str, Any]] = []
    rejected = 0
    idx = 0
    can_clip = with_clip and shutil.which("ffmpeg") is not None
    candidate_tokens = _usable_samples(
        tables, samples_per_scene,
        history_frames=OBSERVED_FRAME_COUNT,
        future_steps=n_steps,
        channel="CAM_FRONT",
    )
    for sample_token in candidate_tokens:
        if len(scenarios) >= limit:
            break
        idx += 1
        scenario_id = f"nuscenes_{idx:04d}"
        from simulation.review_overlay import observable_root_universe
        universe = observable_root_universe(
            tables, sample_token, OBSERVED_FRAME_COUNT, radius=radius)
        traj = extract_observed_trajectory(
            tables, sample_token, n_steps, max_actors=max_actors, radius=radius,
            allowed_instance_tokens=set(universe["allowed_instance_tokens"]))
        root = traj["root"]
        history = extract_observed_history(
            tables, sample_token, root, universe["visibility_by_instance"])
        gt = build_twin_gt(root, _default_spec(
            scenario_id, root, n_steps, dt, traj["factual_states"], traj["factual_action"]))
        gt["protocol"] = protocol_manifest(track=PRIMARY_TRACK, split=version)
        ok, reasons = qc_scenario_gt(gt, env["ontology"], env["config"], env["branch_schema"])
        if not ok:
            rejected += 1
            continue
        # Dir name matches the eval pipeline glob (scenario_*); scenario_id stays informative.
        scen_dir = os.path.join(out_dir, f"scenario_{idx}")
        os.makedirs(scen_dir, exist_ok=True)
        with open(os.path.join(scen_dir, "ground_truth.json"), "w", encoding="utf-8") as f:
            json.dump(gt, f, indent=2)
        from simulation.observed_history import (write_observed_artifacts,
                                                   write_protocol_sidecars)
        write_observed_artifacts(scen_dir, root, history)
        write_protocol_sidecars(scen_dir, gt, env["ontology"])
        with open(os.path.join(scen_dir, "input_manifest.json"), "w", encoding="utf-8") as f:
            json.dump(protocol_manifest(
                track=PRIMARY_TRACK, condition="video", split=version,
                coordinate_frame="t0_ego", frame_asset="frames",
                raw_frame_asset="raw_frames", observed_state_asset="observed_state.json",
                observed_history_asset="observed_history.json",
                ontology_asset="ontology.json", query_asset="query.json",
                overlay_version="coriva-id-overlay-2"), f, indent=2)
        write_model_frames(tables, sample_token, dataroot, scen_dir, root,
                           num_frames=OBSERVED_FRAME_COUNT)
        with open(os.path.join(scen_dir, "prompt.md"), "w", encoding="utf-8") as f:
            f.write(build_prompt(root, scenario_id, condition="video"))
        clip = None
        if can_clip:
            clip = os.path.join(scen_dir, "observed.mp4")
            build_observed_clip(dataroot, tables, sample_token, clip, fps=2)
        scenarios.append({"scenario_id": scenario_id, "dir": scen_dir,
                          "sample_token": sample_token, "clip": clip})

    return {"accepted": len(scenarios), "rejected": rejected, "scenarios": scenarios}


def main():
    p = argparse.ArgumentParser(description="Package nuScenes scenarios for VLM inference")
    p.add_argument("--dataroot", default="data/nuscenes")
    p.add_argument("--version", default="v1.0-mini")
    p.add_argument("--limit", type=int, default=10)
    p.add_argument("--out-dir", default="output_pilot")
    p.add_argument("--no-clip", action="store_true", help="skip observed.mp4 (no ffmpeg)")
    args = p.parse_args()
    r = build_pilot(args.dataroot, args.version, args.limit, args.out_dir,
                    with_clip=not args.no_clip)
    print(f"accepted={r['accepted']} rejected={r['rejected']} -> {args.out_dir}")


if __name__ == "__main__":
    main()
