"""Prompt contract tests that need no dataset download."""
import os
import sys
import json
import re

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


def test_build_prompt_states_the_node_event_label_convention():
    """The references label every non-terminal L4 node and the
    no-action branch maintain_course/ego_no_action; the prompt must say so,
    because the scorer zeroes a node on any event_type mismatch."""
    from simulation.build_pilot import build_prompt_parts
    root = {"actors": [
        {"id": "ego", "entity_type": "ego_vehicle", "role_type": "ego_agent", "x": 0, "y": 0, "vx": 10, "vy": 0, "speed": 10},
        {"id": "npc_1", "entity_type": "pedestrian", "role_type": "crossing_actor", "x": 20, "y": -4, "vx": 0, "vy": 1, "speed": 1},
    ]}
    system, _ = build_prompt_parts(root, "nuscenes_0001", interventions=["pedestrian_enters_road"])
    rules = system.split("## Rules", 1)[1]  # the rules close the system message
    assert "names what the EGO is doing" in rules
    assert "On every\n   NON-terminal node" in rules and "for BOTH intervention chains" in rules
    assert "EVERY node of an L4 chain" not in rules  # terminal nodes keep collision/safe_stop
    assert "A terminal node follows\n   rules 4-6 instead" in rules
    assert "`maintain_course` with `action_type: ego_no_action`" in rules
    for action, label in (("ego_panic_brake", "ego_brake"), ("ego_swerve_right", "ego_steer"),
                          ("ego_accelerate_escape", "ego_accelerate"), ("ego_decelerate_coast", "maintain_course")):
        assert f"`{action}`" in rules and f"`{label}`" in rules
    for forbidden in ("lane_follow", "npc_brake", "pedestrian_cross", "near_miss"):
        assert f"`{forbidden}`" in rules
    assert rules.strip().splitlines()[-1].startswith("8. Output ONLY the JSON object")


def _prompt(**kw):
    from simulation.build_pilot import build_prompt
    root = {"actors": [
        {"id": "ego", "entity_type": "ego_vehicle", "role_type": "ego_agent", "x": 0, "y": 0, "vx": 10, "vy": 0, "speed": 10},
        {"id": "npc_1", "entity_type": "pedestrian", "role_type": "crossing_actor", "x": 20, "y": 4, "vx": 0, "vy": -1, "speed": 1},
    ]}
    return build_prompt(root, "nuscenes_0001", interventions=["pedestrian_enters_road"], critical_actor_id="npc_1",
                        l3_actions=["do_ego_comfortable_brake", "do_ego_no_action", "do_ego_swerve_right"], **kw)


def test_prompt_publishes_the_motion_profiles_the_references_and_control_use():
    """The control reads EGO_PROFILES/ENV_PROFILES, so the model must see
    the same values. The section is generated from the same constants, and
    the road-entry direction is resolved from the target's side."""
    from simulation.kinematic_twin import EGO_PROFILES, ENV_PROFILES
    text = _prompt()
    section = text.split("## Published motion profiles", 1)[1].split("## Task — Ego-action branching", 1)[0]
    assert f"`do_ego_comfortable_brake`: longitudinal acceleration {EGO_PROFILES['do_ego_comfortable_brake']['accel']:+g} m/s^2" in section
    assert "`do_ego_swerve_right`" in section and "2.5 m/s towards +y (right) until one lane width (3.5 m), then hold" in section
    assert "applied from T=0, so the depth-1 state already includes 0.5 s of it" in section
    assert "bicycle model" in section and "point-mass rollout" not in section  # nuScenes wording
    assert "`pedestrian_enters_road`" in section and f"{abs(ENV_PROFILES['pedestrian_enters_road']['lat']):g} m/s towards -y (left)" in section
    assert "every non-ego actor's behaviour must be predicted from the observations" in section


def test_intervention_profile_follows_the_scene_reference_not_the_twin_default():
    """CARLA executes the scene configuration (sudden brake
    -6/-5, cut-in 1.8, pedestrian 2.0, cyclist 2.5), the twin default differs
    (-8, 2.0, 1.6, 4.0). The prompt must print the scene's own values."""
    from simulation.build_pilot import declared_intervention_profile, build_prompt
    from simulation.kinematic_twin import ENV_PROFILES
    root = {"actors": [{"id": "ego", "y": 0.0}, {"id": "npc_1", "y": 3.0}]}
    carla_gt = {"observed_state": root, "protocol": {"split": "carla"},
                "environment_event": {"event_type": "npc_sudden_brake", "params": {"intensity": 6.0}}}
    q = {"intervention_type": "npc_sudden_brake", "target_actor_id": "npc_1", "trigger_depth": 1}
    assert declared_intervention_profile(carla_gt, q)["accel"] == -6.0
    assert declared_intervention_profile(carla_gt, q, {"npc_1": {"accel": -5.0, "lat": 0.0}})["accel"] == -5.0
    ped_gt = {"observed_state": root, "protocol": {"split": "carla"},
              "environment_event": {"event_type": "pedestrian_enters_road", "params": {"intensity": 2.0}}}
    ped = declared_intervention_profile(ped_gt, {"intervention_type": "pedestrian_enters_road", "target_actor_id": "npc_1"})
    assert ped["lat"] == -2.0  # towards the ego lane from the +y side
    nus = declared_intervention_profile({"observed_state": root}, {"intervention_type": "cyclist_enters_road", "target_actor_id": "npc_1",
                                        "resolved_accel_mps2": 0.0, "resolved_lateral_velocity_mps": -4.0})
    assert nus["lat"] == -4.0 and "released query" in nus["source"]
    default = declared_intervention_profile({"observed_state": root}, {"intervention_type": "cyclist_enters_road", "target_actor_id": "npc_1"})
    assert default["lat"] == -abs(ENV_PROFILES["cyclist_enters_road"]["lat"]) and "twin" in default["source"]
    text = build_prompt(root | {"actors": [dict(a, entity_type="vehicle", role_type="lead_vehicle", x=10, vx=8, vy=0, speed=8) for a in root["actors"]]},
                        "carla_0001", interventions=["npc_sudden_brake"], critical_actor_id="npc_1",
                        l3_actions=["do_ego_emergency_brake"], intervention_profiles={"npc_sudden_brake": declared_intervention_profile(carla_gt, q)},
                        split="carla")
    section = text.split("## Published motion profiles", 1)[1].split("## Task — Ego-action branching", 1)[0]
    l4 = section.split("Target-actor intervention parameters", 1)[1]
    assert "`npc_sudden_brake`: longitudinal acceleration -6 m/s^2" in l4
    assert "-8 m/s^2" not in l4 and "CARLA simulator" in l4 and "not the twin's default" in l4


def test_prompt_states_snapshot_stop_precedence_bounds_and_dead_terminal_label():
    """Snapshot semantics, stop precedence, per-step bounds and the dead
    terminal label are stated in the rules."""
    text = _prompt()
    rules = text.split("## Rules", 1)[1]
    assert "EVERY node is a full snapshot" in rules and "named target must be present from the first node" in rules
    assert "only when NO collision occurs within the horizon" in rules and "Label the FIRST such node `safe_stop`" in rules
    assert "average velocity over that step" in rules and "do not force `vy` to\n   0" in rules
    assert "displacement must equal speed" not in rules
    assert "`out_of_bounds` is in the\nvocabulary but no reference uses it" in text


def test_prompt_publishes_the_scripted_hazard_manoeuvre_of_the_l3_references():
    """CARLA's lead brakes -6 m/s^2 from T=0 in
    every L3 branch (scene env_behavior); nuScenes records the template
    profile as scenario_template.hazard_profile; a cut-in candidate keeps its
    observed motion, so nothing is published and the prompt says so."""
    from simulation.build_pilot import declared_hazard_profile, build_prompt
    root = {"actors": [
        {"id": "ego", "entity_type": "ego_vehicle", "role_type": "ego_agent", "x": 0, "y": 0, "vx": 12, "vy": 0, "speed": 12},
        {"id": "npc_1", "entity_type": "vehicle", "role_type": "lead_vehicle", "x": 8, "y": 0, "vx": 12, "vy": 0, "speed": 12}]}
    carla = declared_hazard_profile({"observed_state": root}, "npc_1", {"npc_1": {"accel": -6.0, "lat": 0.0}, "npc_2": {"accel": 0.0, "lat": 0.0}})
    assert carla == {"actor_id": "npc_1", "accel": -6.0, "lat": 0.0}
    nus = declared_hazard_profile({"scenario_template": {"selected_actor_id": "npc_1", "hazard_profile": {"accel": -4.0, "lat": 0.0}}}, "npc_1")
    assert nus["accel"] == -4.0
    assert declared_hazard_profile({"scenario_template": {"selected_actor_id": "npc_1", "hazard_profile": None}}, "npc_1") is None
    assert declared_hazard_profile({"observed_state": root}, "npc_1", {"npc_1": {"accel": 0.0, "lat": 0.0}}) is None
    text = build_prompt(root, "carla_0001", interventions=["npc_sudden_brake"], critical_actor_id="npc_1",
                        l3_actions=["do_ego_emergency_brake", "do_ego_no_action"], split="carla", hazard_profile=carla)
    section = text.split("## Published motion profiles", 1)[1].split("## Task — Ego-action branching", 1)[0]
    assert "In EVERY ego-action branch the reference gives the scene's hazard actor `npc_1`" in section
    assert "longitudinal acceleration -6 m/s^2" in section.split("In EVERY ego-action branch", 1)[1]
    assert "Every OTHER non-ego actor's behaviour must be predicted" in section
    text2 = build_prompt(root, "nuscenes_0001", interventions=["npc_cut_in_left"], critical_actor_id="npc_1",
                         l3_actions=["do_ego_no_action"], split="nuscenes", hazard_profile=None)
    assert "script no manoeuvre for any other actor" in text2


def _worked_example(text):
    section = text.split("## Worked example (synthetic; not this scene)", 1)[1].split(
        "## End of worked example", 1)[0]
    blocks = re.findall(r"```json\n(.*?)\n```", section, re.S)
    assert len(blocks) == 2
    return section, json.loads(blocks[0]), json.loads(blocks[1])


def test_worked_example_is_a_complete_parseable_and_gate_valid_answer():
    """The actual demonstration the model sees must pass the existing gates,
    including T0 -> first-node kinematics, not merely look like JSON."""
    from analysis.parse_prediction import parse_model_output
    from ctms.ctms import (load_ctms_environment, check_schema_validity,
                           check_semantic_validity, check_kinematic_validity)
    _, example, answer = _worked_example(_prompt())
    l3, l4, errors = parse_model_output(json.dumps(answer), "synthetic_example")
    assert errors == []
    branches = l3["predicted_branches"]
    pairs = l4["predicted_counterfactuals"]
    assert [b["intervention"]["intervention_type"] for b in branches] == example["ego_actions"]
    assert sum(b["is_factual"] for b in branches) == 1
    assert next(b for b in branches if b["is_factual"])["intervention"]["intervention_type"] == example["factual_action"]
    assert [p["intervention"] for p in pairs] == example["target_interventions"]
    assert all(set(p) == {"intervention", "control_chain", "treated_chain"} for p in pairs)
    env = load_ctms_environment("ctms/ontology.yaml", "ctms/ctms_config.yaml",
                                "ctms/schema_branch.json", "ctms/schema_counterfactual.json")
    env["config"]["timing"]["require_protocol_grid"] = True
    root = example["observed_state"]
    universe = {a["id"]: a["entity_type"] for a in root["actors"]}
    units = branches + [dict(branch_id=key, intervention=p["intervention"], chain=p[key])
                        for p in pairs for key in ("control_chain", "treated_chain")]
    for unit in units:
        for ok, messages in (
            check_schema_validity(unit, env["branch_schema"]),
            check_semantic_validity(unit, env["ontology"], root_universe=universe),
            check_kinematic_validity(unit, env["config"], root=root),
        ):
            assert ok, messages
        chain = unit["chain"]
        assert len(chain) == 6 or chain[-1].get("terminal") is True
        for depth, node in enumerate(chain, 1):
            assert (node["depth"], node["timestamp"]) == (depth, depth * 0.5)
            assert {a["id"] for a in node["actors_update"]} == set(universe)
            assert len(node["actors_update"]) == len(universe)
    stop = branches[1]["chain"][-1]
    assert stop["event_type"] == "safe_stop" and "action_type" not in stop
    assert stop["actors_update"][0]["speed"] == 0.0
    pair = pairs[0]
    assert pair["control_chain"][-1]["actors_update"][1]["x"] == 52.0
    assert pair["treated_chain"][-1]["actors_update"][1]["x"] == 41.0
    assert all(node["event_type"] == "maintain_course" and node["action_type"] == "ego_no_action"
               for key in ("control_chain", "treated_chain") for node in pair[key])


@pytest.mark.parametrize("condition", ["video", "history_json", "video_history"])
@pytest.mark.parametrize("split", ["carla", "nuscenes"])
@pytest.mark.parametrize("track, steps", [("primary", 6), ("challenge_6s", 12)])
def test_worked_example_is_identical_across_inputs_and_does_not_replace_the_query(condition, split, track, steps):
    from simulation.build_pilot import build_prompt
    section, _, _ = _worked_example(_prompt())
    root = {"actors": [{"id": "ego", "entity_type": "ego_vehicle", "role_type": "ego_agent",
                        "x": 0, "y": 0, "vx": 7, "vy": 0, "speed": 7},
                       {"id": "npc_9", "entity_type": "pedestrian", "role_type": "crossing_actor",
                        "x": 13, "y": 3, "vx": 0, "vy": -1, "speed": 1}]}
    text = build_prompt(root, "a_different_scene", l3_actions=["do_ego_no_action"],
                        interventions=["pedestrian_enters_road"], critical_actor_id="npc_9",
                        factual_action="do_ego_no_action", split=split, condition=condition,
                        track=track, observed_history={"states": [root] * 8})
    assert _worked_example(text)[0] == section
    actual_task = text.split("## End of worked example", 1)[1]
    assert "Expected output: 1 ego-action branches and 1 target-intervention pairs" in actual_task
    assert f"Each non-terminal chain runs for {steps} steps" in actual_task
    assert "npc_9" in actual_task
    assert "<same node" not in text and "<one more entry" not in text
    assert "source_instance_token" in actual_task  # input metadata must not be copied to output


def test_empty_intervention_list_does_not_request_ego_actions_as_target_interventions():
    from simulation.build_pilot import build_prompt
    text = build_prompt({"actors": [{"id": "ego"}]}, "no_l4",
                        l3_actions=["do_ego_no_action"], interventions=[])
    task = text.split("## Task — Target-actor intervention", 1)[1].split("## OUTPUT CONTRACT", 1)[0]
    assert "For EACH intervention in []" in task
    assert "Expected output: 1 ego-action branches and 0 target-intervention pairs" in text
    assert "empty array" in task


# ── prompt variants: arm A `full` vs arm B `recipe_withheld` ─────────────

def _variant_root():
    return {"actors": [
        {"id": "ego", "entity_type": "ego_vehicle", "role_type": "ego_agent", "x": 0.0, "y": 0.0,
         "vx": 10.0, "vy": 0.0, "speed": 10.0, "length": 4.5, "width": 2.0},
        {"id": "npc_1", "entity_type": "vehicle", "role_type": "lead_vehicle", "x": 12.0, "y": 0.0,
         "vx": 10.0, "vy": 0.0, "speed": 10.0, "length": 4.5, "width": 2.0}]}


def _variant_kwargs(split="carla"):
    query = {"l3_ego_actions": ["do_ego_emergency_brake", "do_ego_no_action", "do_ego_swerve_left"],
             "l4_interventions": [{"intervention_type": "npc_sudden_brake", "target_actor_id": "npc_1",
                                   "trigger_depth": 1}],
             "l4_ego_policy": "do_ego_no_action"}
    return dict(interventions=["npc_sudden_brake"], critical_actor_id="npc_1", condition="video",
                l3_actions=query["l3_ego_actions"], query=query,
                intervention_profiles={"npc_sudden_brake": {"accel": -6.0, "lat": 0.0}}, split=split,
                hazard_profile={"actor_id": "npc_1", "accel": -6.0, "lat": 0.0},
                factual_action="do_ego_emergency_brake" if split == "carla" else "do_ego_no_action",
                factual_reference="executed_action_profile" if split == "carla" else "recorded_future")


def test_recipe_withheld_prompt_names_the_actions_but_supplies_no_motion_parameter():
    from simulation.build_pilot import PROMPT_VERSION, build_prompt_parts, prompt_version
    root, kw = _variant_root(), _variant_kwargs()
    full = "".join(build_prompt_parts(root, "carla_0001", **kw))
    assert "".join(build_prompt_parts(root, "carla_0001", variant="full", **kw)) == full
    withheld = "".join(build_prompt_parts(root, "carla_0001", variant="recipe_withheld", **kw))
    for leak in ("-8 m/s^2", "-6 m/s^2", "-3 m/s^2", "2.5 m/s", "one lane width",
                 "Use it in all branches", "scripted manoeuvre", "published profile",
                 "Published motion profiles"):
        assert leak in full and leak not in withheld, leak
    for kept in ("do_ego_emergency_brake", "do_ego_swerve_left", "npc_sudden_brake", "## Rules",
                 "predicted_counterfactuals", "safe_stop", "## explicit_query", "## Actor identity",
                 "Motion parameters: not supplied", "Not supplied", "magnitude is your judgement",
                 "left =\ntowards -y"):
        assert kept in withheld, kept
    assert prompt_version() == PROMPT_VERSION
    assert prompt_version("recipe_withheld") == PROMPT_VERSION + "+recipe_withheld"
    with pytest.raises(ValueError):
        build_prompt_parts(root, "carla_0001", variant="nope", **kw)


def test_recipe_withheld_keeps_the_recorded_factual_contract_without_the_profiles():
    from simulation.build_pilot import build_prompt_parts
    root, kw = _variant_root(), _variant_kwargs("nuscenes")
    withheld = "".join(build_prompt_parts(root, "nuscenes_0001", variant="recipe_withheld", **kw))
    assert "RECORDED continuation" in withheld and "is_factual: true" in withheld
    assert "published profile" not in withheld and "scripted manoeuvre" not in withheld
    assert "Published motion profiles" not in withheld and "Motion profiles of this scene\n- " not in withheld
    assert "generated rollout of that action" in withheld


def test_withheld_worked_example_stops_on_the_grid_and_uses_no_published_rate():
    import re
    from simulation.build_pilot import _worked_example_section
    text = _worked_example_section(ego_decel=2.4, target_decel=3.2)
    answer = json.loads(re.findall(r"```json\n(.*?)\n```", text, re.S)[1])
    brake = answer["predicted_branches"][1]["chain"]
    assert brake[-1]["event_type"] == "safe_stop" and brake[-1]["terminal"] is True
    assert brake[-1]["depth"] == 5 and brake[-1]["actors_update"][0]["speed"] == 0.0
    assert abs(brake[-1]["actors_update"][0]["x"] - 7.5) < 1e-9
    treated = answer["predicted_counterfactuals"][0]["treated_chain"]
    assert len(treated) == 6 and treated[-1]["actors_update"][1]["speed"] == 0.0
    assert abs(treated[-1]["actors_update"][1]["x"] - 42.5) < 1e-9
    assert "-2.4 m/s^2" in text and "-3.2 m/s^2" in text
    assert "-3 m/s^2" not in text and "-8 m/s^2" not in text
    assert "`safe_stop` node at 2.5 s" in text
    # the default example is the one every arm-A prompt has carried
    default = _worked_example_section()
    assert "-3 m/s^2" in default and "-8 m/s^2" in default and "x=41 m" in default


def test_worked_example_prints_rounded_values_and_keeps_the_arm_a_example():
    """A rate such as 2.4 m/s^2 prints 6.3, not 6.300000000000001. The default
    example, which every arm-A prompt carried, stays byte-identical."""
    import hashlib
    from simulation.build_pilot import _worked_example_section
    withheld = _worked_example_section(ego_decel=2.4, target_decel=3.2)
    for text in (_worked_example_section(), withheld):
        assert not re.findall(r"\d+\.\d{4,}", text)
    assert '"x":6.3,' in withheld and '"vx":1.2,' in withheld and '"vx":0.8,' in withheld
    assert hashlib.sha256(_worked_example_section().encode()).hexdigest() == \
        "b628d15a0caba0fedc4fe0a03b4fa288e08e7ac118b3e7618cd604773f14593b"


def test_ego_profiles_withheld_prompt_keeps_the_other_actors_motion_but_no_ego_profile():
    """Arm C (the target-only prompt): the scripted hazard manoeuvre and the
    intervention magnitude are printed as in `full`; the ego profiles, the
    swerve speed, the lane width and the published example rates are not."""
    from simulation.build_pilot import PROMPT_VERSION, build_prompt_parts, prompt_version
    root, kw = _variant_root(), _variant_kwargs()
    full = "".join(build_prompt_parts(root, "carla_0001", **kw))
    withheld = "".join(build_prompt_parts(root, "carla_0001", variant="recipe_withheld", **kw))
    partial = "".join(build_prompt_parts(root, "carla_0001", variant="ego_profiles_withheld", **kw))
    assert partial != full and partial != withheld
    for leak in ("-8 m/s^2", "-3 m/s^2", "2.5 m/s", "one lane width", "Published motion profiles",
                 "with the published acceleration", "(acceleration 0)", "published profile"):
        assert leak in full and leak not in partial, leak
    for kept in ("-6 m/s^2", "Use it in all branches", "scripted manoeuvre",
                 "applies the intervention declared above", "Target-actor intervention parameters",
                 "Motion parameters of the ego actions: not supplied", "Ego actions: not supplied",
                 "do_ego_emergency_brake", "do_ego_swerve_left", "npc_sudden_brake", "## Rules",
                 "predicted_counterfactuals", "safe_stop", "## explicit_query", "## Actor identity",
                 "left =\ntowards -y"):
        assert kept in partial, kept
    # the worked example is the non-published one of arm B, so C - B isolates the stated motion of the other actors
    for example_rate in ("-2.4 m/s^2", "-3.2 m/s^2"):
        assert example_rate in withheld and example_rate in partial and example_rate not in full, example_rate
    assert "magnitude is your judgement" in withheld and "magnitude is your judgement" not in partial
    assert prompt_version("ego_profiles_withheld") == PROMPT_VERSION + "+ego_profiles_withheld"


def test_ego_profiles_withheld_keeps_the_recorded_factual_contract_and_the_hazard_manoeuvre():
    from simulation.build_pilot import build_prompt_parts
    root, kw = _variant_root(), _variant_kwargs("nuscenes")
    partial = "".join(build_prompt_parts(root, "nuscenes_0001", variant="ego_profiles_withheld", **kw))
    assert "RECORDED continuation" in partial and "is_factual: true" in partial
    assert "published profile" not in partial and "Published motion profiles" not in partial
    assert "Motion profiles of this scene\n- " not in partial and "Ego actions: not supplied" in partial
    assert "scripted manoeuvre" in partial and "-6 m/s^2" in partial and "generated rollout of that action" in partial
