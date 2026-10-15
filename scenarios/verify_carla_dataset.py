"""Verify the released protocol-1.0 CARLA-400 composition and artifacts."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import glob
import json
import math
import os
import re
from typing import Any, Dict, List, Optional

from PIL import Image

from scenarios.generate_configs import CARLA_CAMERA, MAPS, TEMPLATES, WEATHERS, slot_map
from simulation.protocol import lateral_action_conflict
from simulation.carla_admission import (
    MAX_INTERACTION_EDGE_GAP_M,
    MIN_BBOX_AREA_PX2,
    MIN_EFFECT_POSITION_M,
    MIN_EFFECT_SPEED_MPS,
    MIN_ROOT_ACTORS,
)
from simulation.build_ground_truth import (ReferenceBuildError, lane_evidence_gaps, scored_collision,
                                           static_pass_through)
from simulation.observed_history import validate_observed_history
from simulation.protocol import (STOP_SPEED_MPS, CARLA_ROUTE_HALF_WIDTH_M, PROTOCOL_VERSION,
                                 validate_protocol_manifest)
from simulation.run_batch import _config_hash


EXPECTED_CRITICAL = {
    "vehicle": 100, "truck": 25, "bus": 25, "motorcycle": 50,
    "pedestrian": 100, "bicycle": 100,
}

EXPECTED_FRAME_NAMES = [f"{index:03d}.png" for index in range(8)]
EXPECTED_ASSETS = {
    "frame_asset": "frames",
    "raw_frame_asset": "raw_frames",
    "observed_state_asset": "observed_state.json",
    "observed_history_asset": "observed_history.json",
    "ontology_asset": "ontology.json",
    "query_asset": "query.json",
    "overlay_version": "coriva-id-overlay-2",
}
SHARED_PROTOCOL_FIELDS = (
    "protocol_version", "track", "split", "condition", "dt_seconds",
    "future_steps", "future_horizon_seconds", "observed_frame_count",
    "coordinate_frame", "coordinate_convention",
    "observed_frame_timestamps_sec", "history_timestamps_sec",
    "available_conditions", "primary_measurement",
    "reference_evaluation_role", "reference_semantics", "dataset",
    "critical_actor_id", "intervention_type",
)


def _load_json(path: str, errors: List[str]) -> Dict[str, Any] | None:
    try:
        with open(path, encoding="utf-8") as stream:
            value = json.load(stream)
    except (OSError, json.JSONDecodeError) as exc:
        errors.append(f"{path}: cannot load JSON: {exc}")
        return None
    if not isinstance(value, dict):
        errors.append(f"{path}: JSON root must be an object")
        return None
    return value


def _verify_frames(scenario_dir: str, subdir: str, errors: List[str]) -> None:
    frame_dir = os.path.join(scenario_dir, subdir)
    names = sorted(os.path.basename(path) for path in
                   glob.glob(os.path.join(frame_dir, "*.png")))
    if names != EXPECTED_FRAME_NAMES:
        errors.append(
            f"{scenario_dir}: {subdir} names={names}, expected={EXPECTED_FRAME_NAMES}")
        return
    expected_size = tuple(CARLA_CAMERA["resolution"])
    for name in names:
        path = os.path.join(frame_dir, name)
        try:
            with Image.open(path) as image:
                if image.format != "PNG":
                    errors.append(f"{path}: image format is not PNG")
                if image.size != expected_size:
                    errors.append(
                        f"{path}: image size={image.size}, expected={expected_size}")
                image.verify()
        except (OSError, SyntaxError, ValueError) as exc:
            errors.append(f"{path}: unreadable image: {exc}")


def _verify_config(item: Dict[str, Any], config: Dict[str, Any],
                   scenario_dir: str, errors: List[str]) -> None:
    if _config_hash(config) != item.get("config_sha256"):
        errors.append(f"{scenario_dir}: scenario config SHA-256 mismatch")
    expected_fields = {
        "scenario_id": config.get("scenario_id"),
        "slot_id": config.get("slot_id"),
        "template_id": config.get("template_id", config.get("scenario_type")),
        "scenario_type": config.get("scenario_type"),
        "attempt_seed": config.get("attempt_seed"),
        "simulation_seed": config.get("seed"),
        "map": config.get("map"),
        "weather": (config.get("weather") or {}).get("preset"),
        "camera": config.get("camera"),
        "critical_actor_id": config.get("critical_actor_id"),
        "simulated_actor_count": len(config.get("actor_defs", {})),
    }
    for field, expected in expected_fields.items():
        if item.get(field) != expected:
            errors.append(
                f"{scenario_dir}: scenario manifest {field} does not match config")
    expected_blueprints = {
        str(spec.get("canonical_id", key)): spec.get("model")
        for key, spec in config.get("actor_defs", {}).items()
    }
    expected_entities = {
        str(spec.get("canonical_id", key)): spec.get("entity_type")
        for key, spec in config.get("actor_defs", {}).items()
    }
    if item.get("actor_blueprints") != expected_blueprints:
        errors.append(f"{scenario_dir}: actor_blueprints do not match config")
    if item.get("actor_entities") != expected_entities:
        errors.append(f"{scenario_dir}: actor_entities do not match config")
    critical_type = expected_entities.get(str(config.get("critical_actor_id")))
    if item.get("critical_actor_type") != critical_type:
        errors.append(f"{scenario_dir}: critical_actor_type does not match config")


def _verify_admission(item: Dict[str, Any], scenario_dir: str,
                      errors: List[str]) -> None:
    admission = item.get("admission")
    if not isinstance(admission, dict) or not admission.get("passed"):
        errors.append(f"{scenario_dir}: admission did not pass")
        return
    if admission.get("reasons") != []:
        errors.append(f"{scenario_dir}: passed admission has non-empty reasons")
    if admission.get("root_actor_count") != item.get("root_actor_count"):
        errors.append(f"{scenario_dir}: admission root_actor_count mismatch")
    bbox = admission.get("critical_bbox_area_px2")
    if (not isinstance(bbox, list) or len(bbox) != 8
            or any(not isinstance(value, (int, float))
                   or float(value) < MIN_BBOX_AREA_PX2 for value in bbox)):
        errors.append(f"{scenario_dir}: invalid eight-frame critical bbox admission")
    position_effect = admission.get("max_l4_target_position_effect_m")
    speed_effect = admission.get("max_l4_target_speed_effect_mps")
    if (not isinstance(position_effect, (int, float))
            or not isinstance(speed_effect, (int, float))
            or (float(position_effect) < MIN_EFFECT_POSITION_M
                and float(speed_effect) < MIN_EFFECT_SPEED_MPS)):
        errors.append(f"{scenario_dir}: L4 material-effect admission mismatch")
    edge_gap = admission.get("min_ego_critical_edge_gap_m")
    if (not admission.get("collision")
            and (not isinstance(edge_gap, (int, float))
                 or float(edge_gap) > MAX_INTERACTION_EDGE_GAP_M)):
        errors.append(f"{scenario_dir}: L4 interaction-gap admission mismatch")
    if admission.get("indistinct_l3_pairs") != []:
        errors.append(f"{scenario_dir}: L3 branches are not physically distinct")
    expected_thresholds = {
        "min_root_actors": MIN_ROOT_ACTORS,
        "min_bbox_area_px2": MIN_BBOX_AREA_PX2,
        "min_effect_position_m": MIN_EFFECT_POSITION_M,
        "min_effect_speed_mps": MIN_EFFECT_SPEED_MPS,
        "max_interaction_edge_gap_m": MAX_INTERACTION_EDGE_GAP_M,
    }
    if admission.get("thresholds") != expected_thresholds:
        errors.append(f"{scenario_dir}: admission thresholds mismatch")


def _verify_scene_geometry(gt: Dict[str, Any], manifest: Dict[str, Any],
                           scenario_dir: str,
                           errors: List[str]) -> None:
    """Verify CARLA-only intended-route evidence and manifest provenance."""
    geometry = gt.get("scene_geometry")
    if not isinstance(geometry, dict):
        errors.append(f"{scenario_dir}: scene_geometry is missing")
        return
    if geometry.get("coordinate_frame") != "t0_ego":
        errors.append(f"{scenario_dir}: scene_geometry coordinate frame mismatch")
    if geometry.get("source") != "carla_map_intended_route":
        errors.append(f"{scenario_dir}: scene_geometry source mismatch")
    if geometry.get("route_mode") != "single_deterministic_intended_route":
        errors.append(f"{scenario_dir}: scene_geometry route mode mismatch")
    route_id = geometry.get("route_id")
    if (not isinstance(route_id, str)
            or re.fullmatch(r"[0-9a-f]{16}", route_id) is None):
        errors.append(f"{scenario_dir}: scene_geometry route ID is invalid")
    for label, coverage_key, required_key in (
            ("forward", "route_coverage_m", "route_required_m"),
            ("backward", "route_backward_coverage_m",
             "route_backward_required_m")):
        coverage = geometry.get(coverage_key)
        required = geometry.get(required_key)
        if (not isinstance(coverage, (int, float))
                or isinstance(coverage, bool)
                or not isinstance(required, (int, float))
                or isinstance(required, bool)):
            errors.append(
                f"{scenario_dir}: scene_geometry {label} route coverage "
                "values are invalid")
        elif float(coverage) < float(required):
            errors.append(
                f"{scenario_dir}: scene_geometry {label} route coverage "
                "is below requirement")
    if geometry.get("route_coverage_complete") is not True:
        errors.append(f"{scenario_dir}: scene_geometry route coverage is incomplete")
    if geometry.get("route_backward_coverage_complete") is not True:
        errors.append(
            f"{scenario_dir}: scene_geometry backward route coverage is incomplete")
    halfwidth = geometry.get("route_half_width_m")
    if (not isinstance(halfwidth, (int, float)) or isinstance(halfwidth, bool)
            or float(halfwidth) <= 0.0):
        errors.append(f"{scenario_dir}: invalid scene_geometry route half-width")
    elif abs(float(halfwidth) - CARLA_ROUTE_HALF_WIDTH_M) > 1e-6:
        errors.append(f"{scenario_dir}: scene_geometry route half-width is not 8 m")
    placement = manifest.get("runtime_placement")
    if not isinstance(placement, dict):
        errors.append(f"{scenario_dir}: runtime placement is missing")
    else:
        if placement.get("route_mode") != geometry.get("route_mode"):
            errors.append(f"{scenario_dir}: manifest route mode differs from ground truth")
        if placement.get("route_id") != geometry.get("route_id"):
            errors.append(f"{scenario_dir}: manifest route ID differs from ground truth")
    lines = geometry.get("route_centerlines")
    if not isinstance(lines, list) or not lines:
        errors.append(f"{scenario_dir}: scene_geometry has no route centerline")
        return
    for line in lines:
        if not isinstance(line, list) or len(line) < 2:
            errors.append(f"{scenario_dir}: malformed scene_geometry route centerline")
            return
        for point in line:
            if (not isinstance(point, list) or len(point) != 2
                    or any(not isinstance(value, (int, float))
                           or isinstance(value, bool) for value in point)):
                errors.append(f"{scenario_dir}: malformed scene_geometry route point")
                return


def _raw_ego_key(actor_defs: Dict[str, Any]) -> Optional[str]:
    return next((key for key, spec in actor_defs.items()
                 if spec.get("entity_type") in {"ego", "ego_vehicle"}
                 or spec.get("canonical_id") == "ego"), None)


def _expected_stop_depth(steps: List[Dict[str, Any]], t0: float, horizon_end: float,
                         actor_defs: Dict[str, Any]) -> Optional[int]:
    """First 0.5 s sample depth at which the raw ego is stopped (at or below
    STOP_SPEED_MPS, 0.5 m/s = 1.8 km/h, the threshold shared with the twin and
    the prompt) and stays stopped at every later raw step through the horizon; None when
    the ego never holds a stop. Re-derived from raw, independent of the
    builder (reference rule 2026-09-13: a stop followed by motion is not a
    `safe_stop`)."""
    ego_key = _raw_ego_key(actor_defs)
    if ego_key is None:
        return None
    if not steps or float(steps[-1]["time_sec"]) < horizon_end - 1e-3:
        return None  # raw does not reach the horizon: no evidence of holding

    def stopped(step: Dict[str, Any]) -> bool:
        ego = step.get(ego_key) or {}
        return bool(ego) and float(ego.get("speed_kmh", float("inf"))) <= STOP_SPEED_MPS * 3.6 + 1e-9

    for depth in range(1, 7):
        t = t0 + 0.5 * depth
        window = [s for s in steps if t - 1e-3 <= float(s["time_sec"]) <= horizon_end + 1e-3]
        if window and all(stopped(s) for s in window):
            return depth
    return None


_MOVING_VEHICLE_TYPES = {"vehicle", "car", "truck", "van", "bus", "motorcycle", "bicycle"}


def _verify_lane_evidence(raw: Dict[str, Any], config: Dict[str, Any],
                          scenario_dir: str, errors: List[str]) -> None:
    """Every moving vehicle spawned in a lateral slot must carry map evidence
    that its slot is a same-direction driving lane over the road it actually
    used (`_placement.moving_actor_lane_checks`, written by the generator
    since 2026-09-14). The shared check `lane_evidence_gaps` is the same one
    run_batch applies before publishing."""
    for gap in lane_evidence_gaps(raw, config.get("actor_defs") or {}):
        errors.append(f"{scenario_dir}: {gap}")


def _verify_reference_collisions(gt: Dict[str, Any], raw: Dict[str, Any],
                                 config: Dict[str, Any], scenario_dir: str,
                                 errors: List[str]) -> None:
    """Re-derive every collision from raw telemetry and compare with the
    released reference: an in-horizon hit must appear as a collision node
    whose impacted actor is the recorded partner and lies in the observed
    root; a hit after the horizon must not; a chain without a terminal
    event must reach all six nodes. This does not trust the builder."""
    t0 = config.get("intervention_time_sec")
    if not isinstance(t0, (int, float)):
        errors.append(f"{scenario_dir}: scenario config has no intervention_time_sec")
        return
    horizon_end = float(t0) + 3.0
    actor_defs = config.get("actor_defs") or {}
    root_ids = {str(actor.get("id")) for actor in
                (gt.get("observed_state") or {}).get("actors", [])}
    # No moving NPC may drive through a parked vehicle or prop anywhere in
    # the raw rollouts (orientation-free centre-distance test, 2026-09-14).
    seen = set()
    for hit in static_pass_through(raw, actor_defs):
        key = (hit["branch"], hit["mover"], hit["static"])
        if key in seen:
            continue
        seen.add(key)
        errors.append(f"{scenario_dir}: {hit['branch']}: moving actor {hit['mover']} passes "
                      f"through static {hit['static']} at {hit['time_sec']:.2f}s "
                      f"(centre distance {hit['distance_m']:.2f} m)")
    raw_branches = raw.get("branches") or {}
    streams = [(branch.get("branch_id"), raw_branches.get(branch.get("branch_id")),
                branch.get("chain") or [])
               for branch in gt.get("branches", [])]
    for query in gt.get("l4_queries", []):
        streams.append(("L4_fixed_policy", raw.get("_l4_steps"),
                        (query.get("counterfactual_chain") or {}).get("chain") or []))
        # Matched-pair protocol: the control reference is verified against
        # the retained paired control rollout by the same rules.
        control = query.get("control_chain")
        if (gt.get("protocol") or {}).get("l4_pair_protocol") is not None or control is not None:
            if not isinstance(control, dict):
                errors.append(f"{scenario_dir}: L4 control reference missing from ground truth")
            else:
                streams.append(("L4_control", raw.get("_l4_control_steps"),
                                control.get("chain") or []))
    if not streams:
        errors.append(f"{scenario_dir}: ground truth has no branches to verify")
    for name, steps, chain in streams:
        label = f"{scenario_dir}: {name}"
        if not steps:
            errors.append(f"{label}: raw rollout missing from raw_telemetry")
            continue
        if not chain:
            errors.append(f"{label}: reference chain is empty")
            continue
        try:
            hit = scored_collision(steps, actor_defs, horizon_end)
        except ReferenceBuildError as exc:
            errors.append(f"{label}: {exc}")
            continue
        collision_nodes = [node for node in chain if node.get("event_type") == "collision"]
        if hit is not None:
            collision_time, partner_keys = hit
            partners = sorted(str(actor_defs[key].get("canonical_id", key))
                              for key in partner_keys)
            if not collision_nodes:
                errors.append(f"{label}: raw collision at {collision_time:.2f}s "
                              "is missing from the reference")
                continue
            impacted = sorted(str(actor.get("id")) for actor in
                              collision_nodes[-1].get("actors_update", [])
                              if actor.get("role_type") == "impacted_actor")
            if impacted != partners:
                errors.append(f"{label}: impacted_actor {impacted} differs from "
                              f"raw collision partner {partners}")
            hidden = [actor for actor in partners if actor not in root_ids]
            if hidden:
                errors.append(f"{label}: collision partner {hidden} is not in the observed root")
            expected_depth = int((collision_time - float(t0) + 1e-6) // 0.5)
            if collision_nodes[-1].get("depth") != expected_depth:
                errors.append(f"{label}: collision node depth "
                              f"{collision_nodes[-1].get('depth')} != {expected_depth} "
                              f"for raw collision at {collision_time:.2f}s")
        elif collision_nodes:
            errors.append(f"{label}: reference has a collision node but no raw "
                          "collision inside the horizon")
        if hit is None:
            last_t = float(steps[-1].get("time_sec", float("nan")))
            if not last_t >= horizon_end - 1e-3:
                errors.append(f"{label}: raw rollout ends at {last_t:.2f}s before the "
                              f"horizon {horizon_end:.2f}s without a collision")
            expected_stop = _expected_stop_depth(steps, float(t0), horizon_end, actor_defs)
            stop_nodes = [node for node in chain if node.get("event_type") == "safe_stop"]
            if expected_stop is None and stop_nodes:
                errors.append(f"{label}: reference has a safe_stop node at depth "
                              f"{stop_nodes[-1].get('depth')} but the raw ego does not "
                              "stay stopped through the horizon")
            elif expected_stop is not None:
                if not stop_nodes:
                    errors.append(f"{label}: raw ego stops at depth {expected_stop} and "
                                  "stays stopped, but the reference has no safe_stop node")
                elif (stop_nodes[-1].get("depth") != expected_stop
                      or chain[-1] is not stop_nodes[-1]):
                    errors.append(f"{label}: safe_stop node depth "
                                  f"{stop_nodes[-1].get('depth')} != {expected_stop} "
                                  "re-derived from the raw ego speed")
        if not chain[-1].get("terminal") and len(chain) != 6:
            errors.append(f"{label}: non-terminal chain has {len(chain)} nodes, expected 6 nodes")


def verify(root: str, expected: int = 400) -> Dict[str, Any]:
    errors: List[str] = []
    paths = sorted(glob.glob(os.path.join(root, "scenario_*", "scenario_manifest.json")))
    records = []
    for path in paths:
        item = _load_json(path, errors)
        if item is not None:
            records.append((item, path))
    manifests = [item for item, _path in records]
    if len(manifests) != expected:
        errors.append(f"accepted scenario manifests={len(manifests)}, expected={expected}")

    ids = [str(item.get("scenario_id")) for item in manifests]
    slots = [item.get("slot_id") for item in manifests]
    if len(ids) != len(set(ids)):
        errors.append("scenario_id values are not unique")
    if len(slots) != len(set(slots)):
        errors.append("slot_id values are not unique")
    if expected == 400 and set(slots) != set(range(1, 401)):
        errors.append("slot_id values are not exactly 1..400")

    templates = Counter(str(item.get("template_id")) for item in manifests)
    expected_templates = {str(item["template_id"]): expected // len(TEMPLATES)
                          for item in TEMPLATES}
    if expected == 400 and dict(templates) != expected_templates:
        errors.append(f"template quota mismatch: {dict(templates)}")

    critical = Counter(str(item.get("critical_actor_type")) for item in manifests)
    if expected == 400 and dict(critical) != EXPECTED_CRITICAL:
        errors.append(f"critical actor quota mismatch: {dict(critical)}")
    maps = Counter(str(item.get("map")) for item in manifests)
    if expected == 400 and maps != Counter({name: 100 for name in MAPS}):
        errors.append(f"map quota mismatch: {dict(maps)}")
    # Every slot's map is fixed by the release plan (templates with an
    # adjacent-lane vehicle never sit on a single-carriageway map), so a
    # scene on another map was not generated from the released plan.
    for item in manifests:
        slot_id = item.get("slot_id")
        if isinstance(slot_id, int) and slot_id >= 1:
            planned = slot_map(slot_id - 1)
            if str(item.get("map")) != planned:
                errors.append(
                    f"slot {slot_id} ({item.get('template_id')}) is on map "
                    f"{item.get('map')}; the release plan places it on {planned}")
    weather = Counter(str(item.get("weather")) for item in manifests)
    if expected == 400 and (set(weather) != set(WEATHERS)
                            or min(weather.values()) < 66 or max(weather.values()) > 67):
        errors.append(f"weather balance mismatch: {dict(weather)}")

    dataset_manifest_path = os.path.join(root, "dataset_manifest.json")
    if not os.path.isfile(dataset_manifest_path):
        errors.append("dataset_manifest.json is missing")
    else:
        dataset_manifest = _load_json(dataset_manifest_path, errors)
        if dataset_manifest is not None:
            if dataset_manifest.get("protocol_version") != PROTOCOL_VERSION:
                errors.append("dataset manifest protocol version mismatch")
            if dataset_manifest.get("accepted_scenarios") != len(manifests):
                errors.append("dataset manifest accepted_scenarios mismatch")

    model_counts: Dict[str, Counter] = defaultdict(Counter)
    for item, manifest_path in records:
        scenario_dir = os.path.dirname(manifest_path)
        required = ("ground_truth.json", "raw_telemetry.json", "observed_state.json",
                    "observed_history.json", "query.json", "ontology.json",
                    "input_manifest.json", "scenario_manifest.json",
                    "scenario_config.json")
        missing = [name for name in required if not os.path.isfile(os.path.join(scenario_dir, name))]
        if missing:
            errors.append(f"{scenario_dir}: missing {missing}")
        _verify_frames(scenario_dir, "frames", errors)
        _verify_frames(scenario_dir, "raw_frames", errors)

        loaded: Dict[str, Dict[str, Any] | None] = {}
        for name in required:
            if name == "scenario_manifest.json":
                loaded[name] = item
                continue
            path = os.path.join(scenario_dir, name)
            loaded[name] = _load_json(path, errors) if os.path.isfile(path) else None

        input_manifest = loaded["input_manifest.json"]
        gt = loaded["ground_truth.json"]
        observed_state = loaded["observed_state.json"]
        observed_history = loaded["observed_history.json"]
        config = loaded["scenario_config.json"]

        if input_manifest is not None:
            try:
                validate_protocol_manifest(input_manifest)
            except ValueError as exc:
                errors.append(f"{scenario_dir}: invalid input manifest: {exc}")
            if input_manifest.get("camera") != CARLA_CAMERA:
                errors.append(f"{scenario_dir}: CAM_FRONT calibration mismatch")
            for field, value in EXPECTED_ASSETS.items():
                if input_manifest.get(field) != value:
                    errors.append(
                        f"{scenario_dir}: input manifest {field} mismatch")

        scenario_protocol = item.get("protocol")
        gt_protocol = gt.get("protocol") if gt is not None else None
        for label, value in (("scenario manifest", scenario_protocol),
                             ("ground truth", gt_protocol)):
            if not isinstance(value, dict):
                errors.append(f"{scenario_dir}: {label} protocol is missing")
                continue
            try:
                validate_protocol_manifest(value)
            except ValueError as exc:
                errors.append(f"{scenario_dir}: invalid {label} protocol: {exc}")
        if isinstance(scenario_protocol, dict) and isinstance(gt_protocol, dict):
            if scenario_protocol != gt_protocol:
                errors.append(
                    f"{scenario_dir}: scenario manifest protocol differs from ground truth")
        if isinstance(gt_protocol, dict) and input_manifest is not None:
            for field in SHARED_PROTOCOL_FIELDS:
                if field not in gt_protocol or field not in input_manifest:
                    errors.append(f"{scenario_dir}: shared protocol field {field} is missing")
                elif gt_protocol[field] != input_manifest[field]:
                    errors.append(f"{scenario_dir}: shared protocol field {field} mismatch")

        if observed_state is not None and observed_history is not None:
            try:
                validate_observed_history(observed_history, observed_state)
            except (TypeError, ValueError) as exc:
                errors.append(f"{scenario_dir}: invalid observed artifacts: {exc}")
        if gt is not None and observed_state is not None:
            if gt.get("observed_state") != observed_state:
                errors.append(
                    f"{scenario_dir}: ground truth observed_state differs from sidecar")
            if gt.get("scenario_id") != item.get("scenario_id"):
                errors.append(f"{scenario_dir}: ground truth scenario_id mismatch")
            _verify_scene_geometry(gt, item, scenario_dir, errors)

        if config is not None:
            _verify_config(item, config, scenario_dir, errors)
        raw = loaded["raw_telemetry.json"]
        if gt is not None and raw is not None and config is not None:
            _verify_reference_collisions(gt, raw, config, scenario_dir, errors)
            _verify_lane_evidence(raw, config, scenario_dir, errors)
        if item.get("protocol_version") != PROTOCOL_VERSION:
            errors.append(f"{scenario_dir}: scenario manifest protocol version mismatch")
        if item.get("camera") != CARLA_CAMERA:
            errors.append(f"{scenario_dir}: scenario manifest camera mismatch")
        if item.get("simulated_actor_count") != 10:
            errors.append(f"{scenario_dir}: simulated actor count is not 10")
        if config is not None:
            behind = [aid for aid, spec in (config.get("actor_defs") or {}).items()
                      if float(spec.get("distance_ahead", 0.0)) < 0.0]
            if behind:
                errors.append(f"{scenario_dir}: actors spawned behind the ego: {behind}")
        root_count = item.get("root_actor_count")
        if not isinstance(root_count, int) or root_count < MIN_ROOT_ACTORS:
            errors.append(f"{scenario_dir}: root actor count below 5")
        if observed_state is not None:
            actors = observed_state.get("actors")
            if not isinstance(actors, list) or len(actors) != root_count:
                errors.append(f"{scenario_dir}: root_actor_count does not match observed_state")
        _verify_admission(item, scenario_dir, errors)
        if gt is not None:
            conflict = lateral_action_conflict(
                gt.get("observed_state"), item.get("critical_actor_id"),
                [b.get("intervention", {}).get("intervention_type")
                 for b in gt.get("branches") or []])
            if conflict:
                errors.append(f"{scenario_dir}: {conflict}")
        entities = item.get("actor_entities") or {}
        for actor_id, model in (item.get("actor_blueprints") or {}).items():
            model_counts[str(entities.get(actor_id))][str(model)] += 1

    # Blueprint (mesh) frequencies are reported, not enforced. The configs
    # assign blueprints in a balanced rotation, but the rotation index moves
    # with the attempt seed, so a slot that is resampled changes blueprint;
    # blueprints that place badly (mercedes.coupe as a parked car in the
    # 2026-09-15 release) and everything sharing their rotation index are
    # therefore under-represented among accepted scenes. No claim depends on
    # blueprint balance (entity types, templates, maps and weather are the
    # quotas), and the generator cannot change for a published release
    # without breaking config reproducibility, so this is a warning.
    warnings: List[str] = []
    for entity, counts in model_counts.items():
        total = sum(counts.values())
        if len(counts) > 1 and max(counts.values()) / total > 0.5:
            warnings.append(
                f"blueprint concentration exceeds 50% for {entity}: {dict(counts)} "
                "(resampled slots rotate to the next blueprint; not a release quota)")

    report = {
        "protocol_version": PROTOCOL_VERSION,
        "passed": not errors,
        "accepted_scenarios": len(manifests),
        "errors": errors,
        "warnings": warnings,
        "distributions": {
            "template_id": dict(sorted(templates.items())),
            "critical_actor_type": dict(sorted(critical.items())),
            "map": dict(sorted(maps.items())),
            "weather": dict(sorted(weather.items())),
            "blueprints_by_entity": {
                entity: dict(sorted(counts.items()))
                for entity, counts in sorted(model_counts.items())
            },
        },
    }
    with open(os.path.join(root, "verification_report.json"), "w", encoding="utf-8") as stream:
        json.dump(report, stream, indent=2)
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True)
    parser.add_argument("--expected", type=int, default=400)
    args = parser.parse_args()
    report = verify(args.root, args.expected)
    print(json.dumps({key: report[key] for key in
                      ("protocol_version", "passed", "accepted_scenarios", "errors", "warnings")}, indent=2))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
