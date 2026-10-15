"""Stable-ID visibility helpers for manual nuScenes review.

Review overlays use the projected-centre visibility rule (center-v1). The
builder can select a separate versioned bounding-box policy (bbox-area-v2)
for its root universe. Review and build share stable nuScenes instance identities.
"""

from __future__ import annotations

import json
import math
import os
import shutil
import subprocess
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from simulation.observed_clip import collect_consecutive_cam_frames
from simulation.protocol import COORDINATE_CONVENTION, COORDINATE_FRAME


def overlay_font_path() -> str:
    """Return the project-bundled font used for reproducible overlay labels."""
    path = os.path.abspath(os.path.join(
        os.path.dirname(__file__), "..", "assets", "fonts", "NotoSans-Regular.ttf"))
    if not os.path.isfile(path):
        raise RuntimeError(
            "bundled overlay font is missing: " + path
            + "; restore Coding/assets/fonts/NotoSans-Regular.ttf")
    return path


def _quat_matrix(q: Sequence[float]) -> List[List[float]]:
    w, x, y, z = [float(v) for v in q]
    return [
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ]


def _mat_vec(m: List[List[float]], v: Sequence[float]) -> Tuple[float, float, float]:
    return tuple(sum(m[i][j] * float(v[j]) for j in range(3)) for i in range(3))  # type: ignore[return-value]


def _sub(a: Sequence[float], b: Sequence[float]) -> Tuple[float, float, float]:
    return tuple(float(a[i]) - float(b[i]) for i in range(3))  # type: ignore[return-value]


def project_global_point(point: Sequence[float], ego_pose: Dict[str, Any],
                        calibrated_sensor: Dict[str, Any]) -> Optional[Tuple[float, float, float]]:
    """Project a global point into a nuScenes camera (u, v, depth)."""
    ego_r = _quat_matrix(ego_pose["rotation"])
    sensor_r = _quat_matrix(calibrated_sensor["rotation"])
    ego_point = _mat_vec([[ego_r[j][i] for j in range(3)] for i in range(3)],
                          _sub(point, ego_pose["translation"]))
    sensor_point = _mat_vec([[sensor_r[j][i] for j in range(3)] for i in range(3)],
                             _sub(ego_point, calibrated_sensor["translation"]))
    # nuScenes calibrated camera coordinates are x right, y down, z forward.
    x, y, z = sensor_point
    intrinsic = calibrated_sensor.get("camera_intrinsic") or []
    if z <= 1e-6 or len(intrinsic) != 3:
        return None
    u = intrinsic[0][0] * x / z + intrinsic[0][2]
    v = intrinsic[1][1] * y / z + intrinsic[1][2]
    return float(u), float(v), float(z)


def _camera_records(tables: Dict[str, Dict[str, Any]], sample_tokens: Iterable[str],
                    channel: str = "CAM_FRONT") -> Dict[str, Dict[str, Any]]:
    wanted = set(sample_tokens)
    cache_key = "__coriva_camera_keyframes_by_channel"
    caches = tables.get(cache_key)
    if not isinstance(caches, dict):
        caches = {}
        tables[cache_key] = caches
    channel_records = caches.get(channel)
    if channel_records is None:
        prefix = f"samples/{channel}/"
        channel_records = {
            record["sample_token"]: record
            for record in tables["sample_data"].values()
            if record.get("is_key_frame")
            and record.get("filename", "").startswith(prefix)
        }
        caches[channel] = channel_records
    return {token: channel_records[token] for token in wanted if token in channel_records}


def _candidate_annotations(tables: Dict[str, Dict[str, Any]], sample_token: str,
                           ego_pose: Dict[str, Any], radius: float = 60.0) -> List[Dict[str, Any]]:
    from simulation.nuscenes_adapter import (
        annotations_for_sample, category_to_entity, quaternion_yaw, _rotate)
    candidates = []
    ex, ey = ego_pose["translation"][:2]
    yaw = quaternion_yaw(ego_pose["rotation"])
    for ann in annotations_for_sample(tables, sample_token):
        inst = tables["instance"].get(ann.get("instance_token"))
        if not inst:
            continue
        cat = tables["category"].get(inst.get("category_token"), {}).get("name", "")
        entity = category_to_entity(cat)
        if entity is None:
            continue
        rx, ry = _rotate(ann["translation"][0] - ex, ann["translation"][1] - ey, -yaw)
        dist = math.hypot(rx, ry)
        if dist <= radius:
            candidates.append({"annotation": ann, "entity_type": entity, "distance": dist})
    candidates.sort(key=lambda x: (x["distance"], x["annotation"]["instance_token"]))
    return candidates


def actor_visibility(tables: Dict[str, Dict[str, Any]], sample_token: str,
                     instance_token: str, channel: str = "CAM_FRONT") -> Dict[str, Any]:
    """Return visibility and projected center for one actor at one keyframe."""
    sample = tables["sample"][sample_token]
    sd = _camera_records(tables, [sample_token], channel).get(sample_token)
    if sd is None:
        return {"visible": False, "reason": "missing_camera_keyframe"}
    ego_pose = tables["ego_pose"][sd["ego_pose_token"]]
    from simulation.nuscenes_adapter import annotations_for_sample
    ann = next((a for a in annotations_for_sample(tables, sample_token)
                if a.get("instance_token") == instance_token), None)
    if ann is None:
        return {"visible": False, "reason": "missing_annotation"}
    cal = tables["calibrated_sensor"][sd["calibrated_sensor_token"]]
    projected = project_global_point(ann["translation"], ego_pose, cal)
    if projected is None:
        return {"visible": False, "reason": "behind_camera"}
    u, v, depth = projected
    # A projected center is a conservative, reproducible proxy for a visible
    # box.  Keep a small margin so actors touching the image boundary are not
    # treated as stable visible labels.
    margin = 2.0
    visible = margin <= u < float(sd["width"]) - margin and margin <= v < float(sd["height"]) - margin
    return {"visible": visible, "reason": None if visible else "outside_image",
            "u": round(u, 2), "v": round(v, 2), "depth": round(depth, 2),
            "sample_token": sample_token, "instance_token": instance_token}


def observable_root_universe(tables: Dict[str, Dict[str, Any]], sample_token: str,
                             history_frames: int = 8, channel: str = "CAM_FRONT",
                             radius: float = 60.0, visibility_policy: str = "center-v1") -> Dict[str, Any]:
    """Resolve camera-observable actors with complete observed-window states."""
    from simulation.nuscenes_adapter import annotations_for_instance, observed_sample_tokens
    from simulation.nuscenes_visibility import policy_actor_visibility, visibility_policy_manifest
    visibility_policy_manifest(visibility_policy)

    tokens = observed_sample_tokens(tables, sample_token, history_frames)
    root_sd = _camera_records(tables, [sample_token], channel).get(sample_token)
    if root_sd is None:
        raise ValueError(f"missing {channel} keyframe at T0")
    root_pose = tables["ego_pose"][root_sd["ego_pose_token"]]
    candidates = _candidate_annotations(tables, sample_token, root_pose, radius=radius)
    allowed, visibility = [], {}
    for candidate in candidates:
        instance = candidate["annotation"]["instance_token"]
        annotated_samples = {
            ann["sample_token"] for ann in annotations_for_instance(tables, instance)}
        if not all(tok in annotated_samples for tok in tokens):
            continue
        per_frame = {tok: bool(policy_actor_visibility(
            tables, tok, instance, channel, policy=visibility_policy).get("visible")) for tok in tokens}
        if not any(per_frame.values()):
            continue
        allowed.append(instance)
        visibility[instance] = per_frame
    return {"sample_tokens": tokens, "allowed_instance_tokens": allowed,
            "visibility_by_instance": visibility}


def render_saved_candidate_labels(manifest: dict, source_dir, out_dir) -> None:
    """Label an empty-admission scene using saved projections and unlabelled JPEGs."""
    from pathlib import Path

    if manifest.get("admissible_stable_candidates") or manifest.get("rendered_candidate_labels") != []:
        raise ValueError("Diagnostic relabelling requires saved frames with no rendered labels")
    if shutil.which("magick") is None:
        raise RuntimeError("ImageMagick 'magick' is required for diagnostic overlays")
    font = overlay_font_path()
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    for i in range(8):
        source = Path(source_dir) / f"{i:03d}.jpg"
        target = Path(out_dir) / f"{i:03d}.jpg"
        args = ["magick", str(source), "-font", font, "-fill", "red",
                "-stroke", "white", "-strokewidth", "2", "-pointsize", "26"]
        for label, candidate in manifest["candidates"].items():
            frame = candidate["frames"][i]
            if frame.get("visible"):
                args.extend(["-annotate", f"+{int(frame['u'])}+{int(frame['v'])}", label])
        args.append(str(target))
        subprocess.run(args, check=True)


def build_review_overlay(tables: Dict[str, Dict[str, Any]], sample_token: str,
                         dataroot: str, out_dir: str, num_frames: int = 8,
                         channel: str = "CAM_FRONT") -> Dict[str, Any]:
    """Create labelled review images and return candidate/visibility manifest."""
    if shutil.which("magick") is None:
        raise RuntimeError("ImageMagick 'magick' is required for review overlays")
    font_path = overlay_font_path()
    frame_files = collect_consecutive_cam_frames(tables, sample_token, num_frames, channel)
    sample_tokens = []
    tok = sample_token
    for _ in range(num_frames):
        sample_tokens.append(tok)
        tok = tables["sample"][tok].get("prev")
    sample_tokens.reverse()
    os.makedirs(out_dir, exist_ok=True)
    root_sd = _camera_records(tables, [sample_token], channel)[sample_token]
    root_pose = tables["ego_pose"][root_sd["ego_pose_token"]]
    candidates = _candidate_annotations(tables, sample_token, root_pose)
    labels = {c["annotation"]["instance_token"]: f"cand_{i + 1:03d}"
              for i, c in enumerate(candidates)}
    # Compute role and past-derived T0 state for every labelled actor.  The
    # review artifact then recommends the only intervention allowed by code,
    # instead of asking the reviewer to infer it from entity type alone.
    from simulation.nuscenes_adapter import extract_observed_state, estimate_root_steer
    from simulation.nuscenes_admission import (
        canonical_environment_query, evaluate_nuscenes_effect,
        resolve_nuscenes_intervention)
    allowed_instances = set(labels)
    root = extract_observed_state(
        tables, sample_token, max_actors=max(1, len(candidates)), radius=60.0,
        priority_instance_token=None, allowed_instance_tokens=allowed_instances)
    actor_by_instance = {
        actor.get("source_instance_token"): actor
        for actor in root.get("actors", [])
        if actor.get("source_instance_token")
    }
    ego = next(actor for actor in root["actors"] if actor.get("role_type") == "ego_agent")
    base_steer = estimate_root_steer(tables, sample_token, float(ego.get("speed", 0.0)))
    visibility = {}
    for instance_token, label in labels.items():
        actor = actor_by_instance.get(instance_token)
        applicability = (resolve_nuscenes_intervention(actor)
                         if actor is not None else {
                             "passed": False,
                             "reasons": ["actor_missing_from_t0_root"],
                             "selected_intervention": None,
                         })
        effect = None
        if actor is not None and applicability.get("passed"):
            # Other actors do not influence the point-mass target/ego distance,
            # so this two-actor preview is numerically identical to the final
            # paired effect gate while avoiding repeated large root builds.
            effect = evaluate_nuscenes_effect(
                {"actors": [ego, actor]},
                canonical_environment_query(actor["id"], applicability),
                n_steps=6, dt=0.5, model="bicycle", base_steer=base_steer)
        visibility[label] = {
            "instance_token": instance_token,
            "entity_type": next(c["entity_type"] for c in candidates
                                 if c["annotation"]["instance_token"] == instance_token),
            "role_type": actor.get("role_type") if actor else None,
            "t0_state": ({key: actor.get(key) for key in
                          ("x", "y", "vx", "vy", "speed", "length", "width")}
                         if actor else None),
            "role_state_applicability": applicability,
            "recommended_intervention": applicability.get("selected_intervention"),
            "paired_canonical_twin_effect_gate": effect,
            "frames": [actor_visibility(tables, st, instance_token, channel)
                       for st in sample_tokens],
        }
    stable_candidates = [label for label, item in visibility.items()
                         if all(frame.get("visible") for frame in item["frames"])]
    role_state_eligible = [
        label for label in stable_candidates
        if visibility[label]["role_state_applicability"].get("passed")]
    admissible_candidates = [
        label for label in role_state_eligible
        if (visibility[label].get("paired_canonical_twin_effect_gate") or {}).get("passed")]
    for i, (st, source) in enumerate(zip(sample_tokens, frame_files)):
        labels_for_frame = []
        for label in admissible_candidates:
            rec = visibility[label]["frames"][i]
            if rec.get("visible"):
                labels_for_frame.append((int(rec["u"]), int(rec["v"]), label))
        dst = os.path.join(out_dir, f"{i:03d}.jpg")
        args = ["magick", os.path.join(dataroot, source), "-font", font_path,
                "-fill", "red",
                "-stroke", "white", "-strokewidth", "2", "-pointsize", "26"]
        for u, v, label in labels_for_frame:
            args += ["-annotate", f"+{u}+{v}", label]
        args += [dst]
        subprocess.run(args, check=True)
    return {
        "schema_version": 2,
        "overlay_version": "coriva-id-overlay-2",
        "coordinate_frame": COORDINATE_FRAME,
        "coordinate_convention": dict(COORDINATE_CONVENTION),
        "sample_token": sample_token,
        "channel": channel,
        "frames": [{"frame_index": i, "sample_token": st, "filename": frame_files[i]}
                   for i, st in enumerate(sample_tokens)],
        "candidates": visibility,
        "stable_candidates": stable_candidates,
        "role_state_eligible_stable_candidates": role_state_eligible,
        "admissible_stable_candidates": admissible_candidates,
        "rendered_candidate_labels": admissible_candidates,
        "intervention_selection_policy":
            "entity_type_plus_role_plus_past_derived_t0_state",
        "effect_gate_semantics": "canonical_twin_applicability_not_physical_truth",
    }
