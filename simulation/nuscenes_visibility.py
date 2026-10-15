"""Versioned nuScenes camera visibility shared by audit and dataset builder."""

import itertools
import math

from simulation.review_overlay import _quat_matrix, _mat_vec, _sub, project_global_point

REFERENCE_WIDTH = 1024
REFERENCE_HEIGHT = 768
NEAR_PLANE = 0.1
MIN_AREA_REFERENCE_PX2 = 100.0
VISIBILITY_POLICIES = ("bbox-area-v2", "center-v1")
DEFAULT_VISIBILITY_POLICY = "bbox-area-v2"


def visibility_policy_manifest(policy):
    if policy not in VISIBILITY_POLICIES:
        raise ValueError(f"unknown nuScenes visibility policy: {policy}")
    common = {"name": policy, "channel": "CAM_FRONT", "critical_required_frames": 8,
              "occlusion_measured": False}
    if policy == "center-v1":
        return {**common, "center_margin_px": 2.0}
    return {**common, "min_area_reference_px2": MIN_AREA_REFERENCE_PX2,
            "reference_width": REFERENCE_WIDTH, "reference_height": REFERENCE_HEIGHT,
            "near_plane_m": NEAR_PLANE, "near_plane_crossing_policy": "reject",
            "area_measure": "clipped_projected_rectangle_scaled_by_image_area"}


def policy_actor_visibility(tables, sample_token, instance_token, channel="CAM_FRONT",
                            *, policy=DEFAULT_VISIBILITY_POLICY):
    from simulation.review_overlay import actor_visibility, _camera_records
    from simulation.nuscenes_adapter import annotations_for_sample
    visibility_policy_manifest(policy)
    if policy == "center-v1":
        return actor_visibility(tables, sample_token, instance_token, channel)
    result = {"sample_token": sample_token, "instance_token": instance_token,
              "visible": False, "reason": None}
    try:
        camera = _camera_records(tables, [sample_token], channel).get(sample_token)
        if camera is None:
            return {**result, "reason": "missing_camera_keyframe"}
        ann = next((r for r in annotations_for_sample(tables, sample_token)
                    if r.get("instance_token") == instance_token), None)
        if ann is None:
            return {**result, "reason": "missing_annotation"}
        measurement = measure_box(ann, tables["ego_pose"][camera["ego_pose_token"]],
                                  tables["calibrated_sensor"][camera["calibrated_sensor_token"]],
                                  camera["width"], camera["height"])
        result.update(measurement, width=camera["width"], height=camera["height"])
        if measurement["crosses_near_plane"]:
            result["reason"] = "bbox_crosses_camera_near_plane"
        elif measurement["area_reference_px2"] < MIN_AREA_REFERENCE_PX2:
            result["reason"] = "projected_bbox_area_below_threshold"
        else:
            x1, y1, x2, y2 = measurement["bbox_clipped"]
            result.update(visible=True, u=(x1 + x2) / 2, v=(y1 + y2) / 2)
    except (KeyError, TypeError, ValueError, IndexError, ZeroDivisionError) as exc:
        result.update(reason="invalid_bbox_metadata", detail=str(exc))
    return result


def _camera_point(point, pose, cal):
    def inverse_rotation(q, vector):
        matrix = _quat_matrix(q)
        return _mat_vec([[matrix[j][i] for j in range(3)] for i in range(3)], vector)

    ego = inverse_rotation(pose["rotation"], _sub(point, pose["translation"]))
    return inverse_rotation(cal["rotation"], _sub(ego, cal["translation"]))


def measure_box(annotation, pose, cal, width, height):
    """Project a nuScenes box (size = width, length, height), clipped to image.

    Edges crossing the camera near plane are clipped before projection. Such
    boxes are flagged for review, since large projected areas can mislead.
    """
    if width <= 1 or height <= 1:
        raise ValueError("invalid image dimensions")
    bw, length, bh = map(float, annotation["size"])
    if not all(math.isfinite(v) and v > 0 for v in (bw, length, bh)):
        raise ValueError("invalid box dimensions")
    rotation = _quat_matrix(annotation["rotation"])
    signs = list(itertools.product((-1, 1), repeat=3))
    corners = []
    for sx, sy, sz in signs:
        rotated = _mat_vec(rotation, (sx * length / 2, sy * bw / 2, sz * bh / 2))
        world = [rotated[i] + float(annotation["translation"][i]) for i in range(3)]
        corners.append(_camera_point(world, pose, cal))
    if not all(math.isfinite(v) for corner in corners for v in corner):
        raise ValueError("non-finite box coordinates")
    points = [p for p in corners if p[2] >= NEAR_PLANE]
    crosses_near = bool(points) and len(points) != 8
    for i, j in itertools.combinations(range(8), 2):
        if sum(a != b for a, b in zip(signs[i], signs[j])) != 1:
            continue
        a, b = corners[i], corners[j]
        if (a[2] < NEAR_PLANE) != (b[2] < NEAR_PLANE):
            fraction = (NEAR_PLANE - a[2]) / (b[2] - a[2])
            points.append(tuple(a[k] + fraction * (b[k] - a[k]) for k in range(3)))
    intrinsic = cal["camera_intrinsic"]
    if len(intrinsic) != 3 or any(len(row) != 3 for row in intrinsic):
        raise ValueError("invalid camera intrinsic matrix")
    projected = []
    for point in points:
        u, v, z = _mat_vec(intrinsic, point)
        projected.append((u / z, v / z))
    clipped, area, fraction = None, 0.0, 0.0
    if projected:
        x1, y1 = min(p[0] for p in projected), min(p[1] for p in projected)
        x2, y2 = max(p[0] for p in projected), max(p[1] for p in projected)
        full_area = max(0, x2 - x1) * max(0, y2 - y1)
        left, top = max(0, x1), max(0, y1)
        right, bottom = min(width - 1, x2), min(height - 1, y2)
        area = max(0, right - left) * max(0, bottom - top)
        if area:
            clipped = [left, top, right, bottom]
        fraction = area / full_area if full_area > 0 else 0.0
    center = project_global_point(annotation["translation"], pose, cal)
    return {
        "bbox_clipped": clipped, "area_px2": area,
        "area_reference_px2": area * REFERENCE_WIDTH * REFERENCE_HEIGHT / (width * height),
        "in_frame_fraction": fraction, "crosses_near_plane": crosses_near,
        "center_visible": bool(center and 2 <= center[0] < width - 2 and 2 <= center[1] < height - 2),
    }

