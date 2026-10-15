"""Synthetic release-verifier coverage for protocol-1.0 CARLA artifacts."""

import json
from pathlib import Path

from PIL import Image

from scenarios.generate_configs import CARLA_CAMERA
from scenarios.verify_carla_dataset import verify
from simulation.protocol import HISTORY_TIMESTAMPS, PROTOCOL_VERSION, protocol_manifest
from simulation.run_batch import _config_hash


def _write_json(path: Path, value):
    path.write_text(json.dumps(value), encoding="utf-8")


def _observed_actor(actor_id: str, entity_type: str = "vehicle"):
    return {
        "id": actor_id,
        "entity_type": entity_type,
        "role_type": "ego_agent" if actor_id == "ego" else "lead_vehicle",
        "x": 0.0 if actor_id == "ego" else 10.0,
        "y": 0.0,
        "vx": 5.0,
        "vy": 0.0,
        "speed": 5.0,
        "length": 4.5,
        "width": 2.0,
    }


def _raw_state(x, speed=10.0):
    return {"x": x, "y": 0.0, "vx": speed, "vy": 0.0, "speed_kmh": speed * 3.6,
            "heading_deg": 0.0}


def _raw_steps():
    """Recorded 0.1 s rollout, no collision, all root actors far apart."""
    steps = []
    for index in range(66):
        t = round(index * 0.1, 1)
        step = {"time_sec": t, "ego": _raw_state(10.0 * t)}
        for offset in range(1, 10):
            step[f"npc_{offset}"] = _raw_state(10.0 * t + 40.0 * offset)
        steps.append(step)
    return steps


def _gt_chain():
    return [{"depth": depth, "timestamp": depth * 0.5,
             "event_type": "maintain_course", "action_type": "ego_no_action",
             "actors_update": [
                 {"id": "ego", "entity_type": "ego_vehicle", "role_type": "ego_agent"},
                 {"id": "npc_1", "entity_type": "vehicle", "role_type": "lead_vehicle"}]}
            for depth in range(1, 7)]


def _flag_collision(raw, stream, time_sec, partners):
    steps = raw["branches"][stream] if stream in raw["branches"] else raw[stream]
    step = min(steps, key=lambda item: abs(item["time_sec"] - time_sec))
    step["collision"] = True
    step["collision_time"] = time_sec
    step["collision_partners"] = partners
    del steps[steps.index(step) + 1:]


def _collision_node(depth, impacted):
    return {"depth": depth, "timestamp": depth * 0.5, "event_type": "collision",
            "terminal": True, "actors_update": [
                {"id": "ego", "entity_type": "ego_vehicle", "role_type": "ego_agent"},
                {"id": impacted, "entity_type": "vehicle", "role_type": "impacted_actor"}]}


def _fixture(root: Path):
    scenario = root / "scenario_1"
    (scenario / "frames").mkdir(parents=True)
    (scenario / "raw_frames").mkdir()
    for index in range(8):
        for subdir in ("frames", "raw_frames"):
            Image.new("RGB", tuple(CARLA_CAMERA["resolution"]), color=(index, 0, 0)).save(
                scenario / subdir / f"{index:03d}.png")

    root_actors = [_observed_actor("ego", "ego_vehicle")]
    root_actors.extend(_observed_actor(f"npc_{index}") for index in range(1, 5))
    observed_state = {
        "depth": 0,
        "timestamp": 0.0,
        "event_type": "observed_state",
        "actors": root_actors,
    }
    observed_history = {
        "schema_version": 1,
        "coordinate_frame": "t0_ego",
        "records": [
            {
                "timestamp_sec": timestamp,
                "actors": [dict(actor, visibility=actor["id"] != "ego")
                           for actor in root_actors],
            }
            for timestamp in HISTORY_TIMESTAMPS
        ],
    }
    actor_defs = {
        "ego": {
            "canonical_id": "ego", "entity_type": "ego_vehicle",
            "default_role": "ego_agent", "model": "vehicle.tesla.model3",
        }
    }
    for index in range(1, 10):
        actor_defs[f"npc_{index}"] = {
            "canonical_id": f"npc_{index}", "entity_type": "vehicle",
            "default_role": "lead_vehicle", "model": "vehicle.audi.a2",
        }
    config = {
        "scenario_id": "lead_car_brake_0000",
        "scenario_index": 0,
        "slot_id": 1,
        "template_id": "lead_car_brake",
        "attempt_seed": 0,
        "output_subdir": "scenario_1",
        "scenario_type": "rear_end_collision_risk",
        "map": "Town03",
        "weather": {"preset": "ClearNoon"},
        "seed": 123,
        "camera": CARLA_CAMERA,
        "critical_actor_id": "npc_1",
        "actor_defs": actor_defs,
        "intervention_time_sec": 3.5,
        "sample_interval_sec": 0.5,
        "d_max": 6,
        "branches": {
            "Factual_Brake": {"is_factual": True,
                              "intervention_type": "do_ego_emergency_brake",
                              "target_actor_id": "ego", "trigger_depth": 1},
            "CF_NoAction": {"is_factual": False,
                            "intervention_type": "do_ego_no_action",
                            "target_actor_id": "ego", "trigger_depth": 1},
            "CF_Swerve": {"is_factual": False,
                          "intervention_type": "do_ego_swerve_left",
                          "target_actor_id": "ego", "trigger_depth": 1},
        },
    }
    raw = {"branches": {name: _raw_steps() for name in config["branches"]},
           "_l4_steps": _raw_steps(), "_l4_control_steps": _raw_steps()}
    branches = [
        {"branch_id": name, "is_factual": meta["is_factual"],
         "intervention": {"intervention_type": meta["intervention_type"],
                          "target_actor_id": "ego", "trigger_depth": 1},
         "chain": _gt_chain()}
        for name, meta in config["branches"].items()
    ]
    l4_queries = [{"intervention_type": "npc_sudden_brake",
                   "counterfactual_chain": {
                       "intervention": {"intervention_type": "npc_sudden_brake",
                                        "target_actor_id": "npc_1",
                                        "trigger_depth": 1},
                       "chain": _gt_chain()}}]
    protocol = protocol_manifest(
        split="carla", dataset="carla", critical_actor_id="npc_1",
        intervention_type="npc_sudden_brake",
        l4_ego_policy="do_ego_no_action",
        l4_reference_source="carla_fixed_policy_rollout")
    _write_json(scenario / "ground_truth.json", {
        "scenario_id": config["scenario_id"],
        "observed_state": observed_state,
        "branches": branches,
        "l4_queries": l4_queries,
        "protocol": protocol,
        "scene_geometry": {
            "schema_version": 1,
            "coordinate_frame": "t0_ego",
            "source": "carla_map_intended_route",
            "route_mode": "single_deterministic_intended_route",
            "route_id": "0123456789abcdef",
            "route_centerlines": [[[0.0, 0.0], [20.0, 0.0]]],
            "route_half_width_m": 8.0,
            "route_coverage_m": 20.0,
            "route_required_m": 20.0,
            "route_coverage_complete": True,
            "route_backward_coverage_m": 6.0,
            "route_backward_required_m": 5.0,
            "route_backward_coverage_complete": True,
        },
    })
    _write_json(scenario / "raw_telemetry.json", raw)
    _write_json(scenario / "observed_state.json", observed_state)
    _write_json(scenario / "observed_history.json", observed_history)
    _write_json(scenario / "query.json", {})
    _write_json(scenario / "ontology.json", {})
    _write_json(scenario / "scenario_config.json", config)
    _write_json(scenario / "input_manifest.json", protocol_manifest(
        split="carla", dataset="carla", critical_actor_id="npc_1",
        intervention_type="npc_sudden_brake", camera=CARLA_CAMERA,
        frame_asset="frames", raw_frame_asset="raw_frames",
        observed_state_asset="observed_state.json",
        observed_history_asset="observed_history.json",
        ontology_asset="ontology.json", query_asset="query.json",
        overlay_version="coriva-id-overlay-2"))
    actor_blueprints = {
        actor_id: spec["model"] for actor_id, spec in actor_defs.items()
    }
    actor_entities = {
        actor_id: spec["entity_type"] for actor_id, spec in actor_defs.items()
    }
    _write_json(scenario / "scenario_manifest.json", {
        "schema_version": 1,
        "protocol_version": PROTOCOL_VERSION,
        "scenario_id": config["scenario_id"],
        "slot_id": 1,
        "template_id": "lead_car_brake",
        "scenario_type": config["scenario_type"],
        "attempt_seed": 0,
        "simulation_seed": 123,
        "config_sha256": _config_hash(config),
        "critical_actor_type": "vehicle",
        "critical_actor_id": "npc_1",
        "map": "Town03",
        "weather": "ClearNoon",
        "camera": CARLA_CAMERA,
        "simulated_actor_count": 10,
        "root_actor_count": 5,
        "runtime_placement": {
            "route_mode": "single_deterministic_intended_route",
            "route_id": "0123456789abcdef",
        },
        "protocol": protocol,
        "admission": {
            "passed": True,
            "reasons": [],
            "root_actor_count": 5,
            "critical_bbox_area_px2": [500.0] * 8,
            "max_l4_target_position_effect_m": 1.5,
            "max_l4_target_speed_effect_mps": 0.0,
            "min_ego_critical_edge_gap_m": 4.0,
            "collision": False,
            "indistinct_l3_pairs": [],
            "thresholds": {
                "min_root_actors": 5,
                "min_bbox_area_px2": 100.0,
                "min_effect_position_m": 1.0,
                "min_effect_speed_mps": 1.0,
                "max_interaction_edge_gap_m": 5.0,
            },
        },
        "actor_blueprints": actor_blueprints,
        "actor_entities": actor_entities,
    })
    _write_json(root / "dataset_manifest.json", {
        "protocol_version": PROTOCOL_VERSION,
        "accepted_scenarios": 1,
    })
    return scenario


def test_small_synthetic_release_passes_all_per_scenario_gates(tmp_path):
    _fixture(tmp_path)
    report = verify(str(tmp_path), expected=1)
    assert report["passed"], report["errors"]


def test_verifier_rejects_camera_calibration_drift(tmp_path):
    scenario = _fixture(tmp_path)
    manifest = json.loads((scenario / "input_manifest.json").read_text())
    manifest["camera"]["fov_deg"] = 90.0
    _write_json(scenario / "input_manifest.json", manifest)
    report = verify(str(tmp_path), expected=1)
    assert not report["passed"]
    assert any("CAM_FRONT calibration mismatch" in error for error in report["errors"])


def test_verifier_rejects_missing_carla_route_evidence(tmp_path):
    scenario = _fixture(tmp_path)
    gt = json.loads((scenario / "ground_truth.json").read_text())
    gt.pop("scene_geometry")
    _write_json(scenario / "ground_truth.json", gt)

    report = verify(str(tmp_path), expected=1)
    assert not report["passed"]
    assert any("scene_geometry" in error for error in report["errors"])


def test_verifier_rejects_legacy_route_not_bound_to_runtime(tmp_path):
    scenario = _fixture(tmp_path)
    gt = json.loads((scenario / "ground_truth.json").read_text())
    gt["scene_geometry"].update({
        "source": "carla_waypoint_route",
        "route_mode": None,
        "route_id": None,
    })
    _write_json(scenario / "ground_truth.json", gt)

    report = verify(str(tmp_path), expected=1)
    assert not report["passed"]
    assert any("route mode mismatch" in error for error in report["errors"])


def test_verifier_rejects_manifest_and_ground_truth_route_mismatch(tmp_path):
    scenario = _fixture(tmp_path)
    manifest = json.loads((scenario / "scenario_manifest.json").read_text())
    manifest["runtime_placement"]["route_id"] = "fedcba9876543210"
    _write_json(scenario / "scenario_manifest.json", manifest)

    report = verify(str(tmp_path), expected=1)
    assert not report["passed"]
    assert any("manifest route ID differs" in error for error in report["errors"])


def test_verifier_rejects_backward_coverage_below_exact_requirement(tmp_path):
    scenario = _fixture(tmp_path)
    gt = json.loads((scenario / "ground_truth.json").read_text())
    gt["scene_geometry"]["route_backward_coverage_m"] = 4.999
    gt["scene_geometry"]["route_backward_required_m"] = 5.0
    _write_json(scenario / "ground_truth.json", gt)

    report = verify(str(tmp_path), expected=1)
    assert not report["passed"]
    assert any("backward route coverage is below requirement" in error
               for error in report["errors"])


def test_verifier_rejects_corrupt_frame(tmp_path):
    scenario = _fixture(tmp_path)
    (scenario / "frames" / "007.png").write_bytes(b"not-a-png")
    report = verify(str(tmp_path), expected=1)
    assert not report["passed"]
    assert any("unreadable image" in error for error in report["errors"])


def test_verifier_rejects_config_hash_drift(tmp_path):
    scenario = _fixture(tmp_path)
    config = json.loads((scenario / "scenario_config.json").read_text())
    config["seed"] += 1
    _write_json(scenario / "scenario_config.json", config)
    report = verify(str(tmp_path), expected=1)
    assert not report["passed"]
    assert any("config SHA-256 mismatch" in error for error in report["errors"])


def test_verifier_rejects_observed_history_drift(tmp_path):
    scenario = _fixture(tmp_path)
    history = json.loads((scenario / "observed_history.json").read_text())
    history["records"][0]["timestamp_sec"] = -3.4
    _write_json(scenario / "observed_history.json", history)
    report = verify(str(tmp_path), expected=1)
    assert not report["passed"]
    assert any("invalid observed artifacts" in error for error in report["errors"])


def test_verifier_rejects_protocol_manifest_drift(tmp_path):
    scenario = _fixture(tmp_path)
    manifest = json.loads((scenario / "scenario_manifest.json").read_text())
    manifest["protocol"]["future_horizon_seconds"] = 6.0
    _write_json(scenario / "scenario_manifest.json", manifest)
    report = verify(str(tmp_path), expected=1)
    assert not report["passed"]
    assert any("protocol differs from ground truth" in error for error in report["errors"])


def test_verifier_rejects_incomplete_admission_metrics(tmp_path):
    scenario = _fixture(tmp_path)
    manifest = json.loads((scenario / "scenario_manifest.json").read_text())
    manifest["admission"]["critical_bbox_area_px2"] = [500.0] * 7
    _write_json(scenario / "scenario_manifest.json", manifest)
    report = verify(str(tmp_path), expected=1)
    assert not report["passed"]
    assert any("eight-frame critical bbox admission" in error
               for error in report["errors"])


def _load(scenario, name):
    return json.loads((scenario / name).read_text())


def test_verifier_accepts_reference_matching_raw_collision(tmp_path):
    scenario = _fixture(tmp_path)
    raw, gt = _load(scenario, "raw_telemetry.json"), _load(scenario, "ground_truth.json")
    _flag_collision(raw, "Factual_Brake", 5.5, ["npc_2"])
    gt["branches"][0]["chain"] = gt["branches"][0]["chain"][:3] + [_collision_node(4, "npc_2")]
    _write_json(scenario / "raw_telemetry.json", raw)
    _write_json(scenario / "ground_truth.json", gt)
    report = verify(str(tmp_path), expected=1)
    assert report["passed"], report["errors"]


def test_verifier_rejects_in_horizon_collision_missing_from_reference(tmp_path):
    scenario = _fixture(tmp_path)
    raw, gt = _load(scenario, "raw_telemetry.json"), _load(scenario, "ground_truth.json")
    _flag_collision(raw, "Factual_Brake", 5.5, ["npc_2"])
    gt["branches"][0]["chain"] = gt["branches"][0]["chain"][:2]
    gt["branches"][0]["chain"][-1].update(event_type="safe_stop", terminal=True)
    _write_json(scenario / "raw_telemetry.json", raw)
    _write_json(scenario / "ground_truth.json", gt)
    report = verify(str(tmp_path), expected=1)
    assert not report["passed"]
    assert any("Factual_Brake" in e and "missing from the reference" in e
               for e in report["errors"])


def test_verifier_rejects_wrong_collision_partner(tmp_path):
    scenario = _fixture(tmp_path)
    raw, gt = _load(scenario, "raw_telemetry.json"), _load(scenario, "ground_truth.json")
    _flag_collision(raw, "Factual_Brake", 5.5, ["npc_2"])
    gt["branches"][0]["chain"] = gt["branches"][0]["chain"][:3] + [_collision_node(4, "npc_1")]
    _write_json(scenario / "raw_telemetry.json", raw)
    _write_json(scenario / "ground_truth.json", gt)
    report = verify(str(tmp_path), expected=1)
    assert not report["passed"]
    assert any("impacted_actor" in e for e in report["errors"])


def test_verifier_rejects_collision_partner_outside_root(tmp_path):
    scenario = _fixture(tmp_path)
    raw, gt = _load(scenario, "raw_telemetry.json"), _load(scenario, "ground_truth.json")
    _flag_collision(raw, "_l4_steps", 5.5, ["npc_9"])
    chain = gt["l4_queries"][0]["counterfactual_chain"]["chain"]
    gt["l4_queries"][0]["counterfactual_chain"]["chain"] = chain[:3] + [_collision_node(4, "npc_9")]
    _write_json(scenario / "raw_telemetry.json", raw)
    _write_json(scenario / "ground_truth.json", gt)
    report = verify(str(tmp_path), expected=1)
    assert not report["passed"]
    assert any("npc_9" in e and "root" in e for e in report["errors"])


def test_verifier_rejects_post_horizon_collision_in_reference(tmp_path):
    scenario = _fixture(tmp_path)
    raw, gt = _load(scenario, "raw_telemetry.json"), _load(scenario, "ground_truth.json")
    raw["branches"]["CF_NoAction"] = _raw_steps() + [
        dict(_raw_steps()[-1], time_sec=6.8, collision=True, collision_time=6.8,
             collision_partners=["npc_1"])]
    gt["branches"][1]["chain"][-1] = _collision_node(6, "npc_1")
    _write_json(scenario / "raw_telemetry.json", raw)
    _write_json(scenario / "ground_truth.json", gt)
    report = verify(str(tmp_path), expected=1)
    assert not report["passed"]
    assert any("CF_NoAction" in e and "no raw collision" in e for e in report["errors"])


def test_verifier_rejects_short_non_terminal_chain(tmp_path):
    scenario = _fixture(tmp_path)
    gt = _load(scenario, "ground_truth.json")
    gt["branches"][2]["chain"] = gt["branches"][2]["chain"][:4]
    _write_json(scenario / "ground_truth.json", gt)
    report = verify(str(tmp_path), expected=1)
    assert not report["passed"]
    assert any("CF_Swerve" in e and "6 nodes" in e for e in report["errors"])


def test_verifier_rejects_missing_raw_rollout(tmp_path):
    scenario = _fixture(tmp_path)
    raw = _load(scenario, "raw_telemetry.json")
    del raw["branches"]["CF_Swerve"]
    _write_json(scenario / "raw_telemetry.json", raw)
    report = verify(str(tmp_path), expected=1)
    assert not report["passed"]
    assert any("CF_Swerve" in e and "raw" in e for e in report["errors"])


def test_verifier_rejects_a_slot_on_a_map_the_release_plan_does_not_give_it(tmp_path):
    """Slot 1 (lead_car_brake, occurrence 0) is planned on Town03; a scene of
    that slot on another map was not generated from the released plan."""
    scenario = _fixture(tmp_path)
    manifest = json.loads((scenario / "scenario_manifest.json").read_text())
    manifest["map"] = "Town10HD"
    _write_json(scenario / "scenario_manifest.json", manifest)
    report = verify(str(tmp_path), expected=1)
    assert not report["passed"]
    assert any("release plan places it on Town03" in error for error in report["errors"])


def test_blueprint_concentration_is_a_warning_not_an_error(tmp_path):
    """Blueprint balance is not a release quota: resampling rotates a slot to
    the next blueprint, so accepted scenes over-represent the blueprints that
    follow badly placing ones. The verifier reports it and still passes."""
    scenario = _fixture(tmp_path)
    manifest = json.loads((scenario / "scenario_manifest.json").read_text())
    entities = dict(manifest.get("actor_entities") or {})
    blueprints = dict(manifest.get("actor_blueprints") or {})
    parked = [aid for aid, ent in entities.items() if ent == "parked_vehicle"]
    if len(parked) < 2:  # fixture has no parked pair: synthesise one
        entities.update({"park_x1": "parked_vehicle", "park_x2": "parked_vehicle", "park_x3": "parked_vehicle"})
        parked = ["park_x1", "park_x2", "park_x3"]
    for aid in parked:
        blueprints[aid] = "vehicle.lincoln.mkz_2020"
    blueprints[parked[-1]] = "vehicle.audi.a2"
    manifest["actor_entities"] = entities
    manifest["actor_blueprints"] = blueprints
    _write_json(scenario / "scenario_manifest.json", manifest)
    report = verify(str(tmp_path), expected=1)
    assert any("blueprint concentration" in w for w in report["warnings"]), report
    assert not any("blueprint concentration" in e for e in report["errors"])


def test_verifier_rejects_a_lateral_action_that_steers_into_the_hazard(tmp_path):
    """Release gate for the curved-route case: the published root decides the
    hazard's side, so a scene whose lateral escape points at it must fail."""
    scenario = _fixture(tmp_path)
    gt = json.loads((scenario / "ground_truth.json").read_text())
    gt["observed_state"]["actors"] = [
        dict(a, y=(-5.0 if a.get("id") == "npc_1" else a.get("y", 0.0)))
        for a in gt["observed_state"]["actors"]]
    for branch in gt["branches"]:
        if branch["intervention"]["intervention_type"].startswith("do_ego_swerve"):
            branch["intervention"]["intervention_type"] = "do_ego_swerve_left"
    _write_json(scenario / "ground_truth.json", gt)
    report = verify(str(tmp_path), expected=1)
    assert not report["passed"]
    assert any("steers toward critical actor" in e for e in report["errors"])
