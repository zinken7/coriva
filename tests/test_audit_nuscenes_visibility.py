import json
from pathlib import Path
import subprocess
import sys

import pytest


def test_box_at_image_edge_is_measured_even_when_center_is_outside():
    from scripts.audit_nuscenes_visibility import measure_box

    pose = {"translation": [0, 0, 0], "rotation": [1, 0, 0, 0]}
    cal = {**pose, "camera_intrinsic": [[100, 0, 50], [0, 100, 50], [0, 0, 1]]}
    ann = {"translation": [6, 0, 10], "size": [2, 4, 2], "rotation": [1, 0, 0, 0]}
    result = measure_box(ann, pose, cal, 100, 100)
    assert result["center_visible"] is False
    assert result["area_px2"] == pytest.approx((99 - (50 + 400 / 11)) * (200 / 9))
    assert 0 < result["in_frame_fraction"] < 1


def test_rotation_translation_and_width_length_convention():
    from scripts.audit_nuscenes_visibility import measure_box

    pose = {"translation": [10, 0, 0], "rotation": [1, 0, 0, 0]}
    cal = {"translation": [1, 0, 0], "rotation": [1, 0, 0, 0],
           "camera_intrinsic": [[100, 0, 50], [0, 100, 50], [0, 0, 1]]}
    ann = {"translation": [11, 0, 10], "size": [2, 4, 2],
           "rotation": [2**-0.5, 0, 0, 2**-0.5]}
    result = measure_box(ann, pose, cal, 100, 100)
    x1, y1, x2, y2 = result["bbox_clipped"]
    assert x2 - x1 == pytest.approx(200 / 9)
    assert y2 - y1 == pytest.approx(400 / 9)
    assert result["center_visible"] is True


def test_reference_area_is_invariant_to_image_scaling():
    from scripts.audit_nuscenes_visibility import measure_box

    pose = {"translation": [0, 0, 0], "rotation": [1, 0, 0, 0]}
    ann = {"translation": [0, 0, 20], "size": [2, 4, 2], "rotation": [1, 0, 0, 0]}
    cal = {**pose, "camera_intrinsic": [[500, 0, 512], [0, 500, 384], [0, 0, 1]]}
    first = measure_box(ann, pose, cal, 1024, 768)
    cal["camera_intrinsic"] = [[1000, 0, 1024], [0, 1000, 768], [0, 0, 1]]
    second = measure_box(ann, pose, cal, 2048, 1536)
    assert second["area_px2"] == pytest.approx(4 * first["area_px2"])
    assert second["area_reference_px2"] == pytest.approx(first["area_reference_px2"])


def test_rotated_camera_and_ego_pose_match_known_projection():
    from scripts.audit_nuscenes_visibility import measure_box

    # Ego yaw +90 degrees; camera +90 around Y (camera forward = ego +X).
    q = 2**-0.5
    pose = {"translation": [10, 20, 0], "rotation": [q, 0, 0, q]}
    cal = {"translation": [1, 0, 0], "rotation": [q, 0, q, 0],
           "camera_intrinsic": [[100, 0, 50], [0, 100, 50], [0, 0, 1]]}
    ann = {"translation": [10, 31, 0], "size": [2, 4, 2], "rotation": [q, 0, 0, q]}
    result = measure_box(ann, pose, cal, 100, 100)
    assert result["bbox_clipped"] == pytest.approx([37.5, 37.5, 62.5, 62.5])
    assert result["center_visible"] is True


@pytest.mark.parametrize("depth,area,near", [(-10, 0, False), (0.5, 9801, True)])
def test_behind_camera_and_near_plane(depth, area, near):
    from scripts.audit_nuscenes_visibility import measure_box

    pose = {"translation": [0, 0, 0], "rotation": [1, 0, 0, 0]}
    cal = {**pose, "camera_intrinsic": [[100, 0, 50], [0, 100, 50], [0, 0, 1]]}
    ann = {"translation": [0, 0, depth], "size": [2, 4, 2], "rotation": [1, 0, 0, 0]}
    result = measure_box(ann, pose, cal, 100, 100)
    assert result["area_px2"] == pytest.approx(area)
    assert result["crosses_near_plane"] is near


@pytest.mark.parametrize("contents", ['[{},]', '[{}', '{}', '[{}] garbage'])
def test_stream_rejects_invalid_json(tmp_path, contents):
    from scripts.audit_nuscenes_visibility import iter_records

    path = tmp_path / "table.json"
    path.write_text(contents)
    with pytest.raises(ValueError):
        list(iter_records(path, chunk_size=2))


def test_stream_handles_chunk_boundaries_and_braces_in_strings(tmp_path):
    from scripts.audit_nuscenes_visibility import iter_records

    records = [{"token": "a", "text": 'brace } comma , quote " tiếng Việt'}, {"token": "b"}]
    path = tmp_path / "table.json"
    path.write_text(json.dumps(records))
    assert list(iter_records(path, chunk_size=3)) == records


@pytest.mark.parametrize("missing_annotation", [False, True])
def test_cli_uses_metadata_only_and_reports_missing_actor(tmp_path, missing_annotation):
    meta = tmp_path / "metadata"
    source = tmp_path / "results"
    meta.mkdir()
    (source / "videos").mkdir(parents=True)
    (source / "review_selections_approved.csv").write_text('scene_name,t0_sec\nscene-0001,3.5\n')
    (source / "review_manifest_approved.json").write_text(json.dumps({"selections": [{
        "scene_name": "scene-0001", "t0_sec": 3.5, "critical_instance_token": "actor",
        "audit_status": "approved", "interaction_within_3s": True,
        "intervention_type": "npc_sudden_brake"}]}))
    (source / "videos/scene-0001_CAM_FRONT.json").write_text(json.dumps({
        "scene": {"name": "scene-0001"}, "video": {"fps": 2},
        "frames": [{"is_key_frame": True, "source_sample_token": "s7", "video_time_sec": 3.5}]}))
    identity = {"translation": [0, 0, 0], "rotation": [1, 0, 0, 0]}
    tables = {
        "scene": [{"token": "scene", "name": "scene-0001"}],
        "sample": [{"token": f"s{i}", "prev": f"s{i-1}" if i else "", "scene_token": "scene"}
                   for i in range(8)],
        "sample_data": [{"token": f"d{i}", "sample_token": f"s{i}", "is_key_frame": True,
                         "filename": f"samples/CAM_FRONT/missing-{i}.jpg", "width": 1024, "height": 768,
                         "ego_pose_token": "pose", "calibrated_sensor_token": "cal"} for i in range(8)],
        "ego_pose": [{"token": "pose", **identity}],
        "calibrated_sensor": [{"token": "cal", **identity,
                               "camera_intrinsic": [[500, 0, 512], [0, 500, 384], [0, 0, 1]]}],
        "sample_annotation": [{"token": f"a{i}", "sample_token": f"s{i}", "instance_token": "actor",
                               "translation": [0, 0, 20], "size": [2, 4, 2], "rotation": [1, 0, 0, 0]}
                              for i in range(7 if missing_annotation else 8)],
    }
    for name, rows in tables.items():
        (meta / f"{name}.json").write_text(json.dumps(rows))
    originals = {p: p.read_bytes() for p in tmp_path.rglob("*") if p.is_file()}
    output = tmp_path / "audit"
    script = Path(__file__).resolve().parents[1] / "scripts/audit_nuscenes_visibility.py"
    run = subprocess.run([sys.executable, "-B", str(script), "--metadata-dir", str(meta),
                          "--source", str(source), "--output", str(output)],
                         cwd=tmp_path, capture_output=True, text=True)
    assert run.returncode == (2 if missing_annotation else 0), run.stderr
    report = json.loads((output / "report.json").read_text())
    assert report["summary"]["error" if missing_annotation else "pass"] == 1
    assert len(report["scenes"][0]["frames"]) == 8
    assert all(path.read_bytes() == content for path, content in originals.items())
    assert not list(tmp_path.rglob("*.jpg"))
