"""Tests for the nuScenes full-scene review-video exporter."""

import json
import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


def test_iter_json_array_streams_with_tiny_chunks(tmp_path):
    from scripts.export_nuscenes_review_videos import _iter_json_array

    rows = [
        {"token": "a", "description": "contains [brackets], commas, and unicode: đường"},
        {"token": "b", "nested": {"value": 2}},
    ]
    path = tmp_path / "records.json"
    path.write_text(json.dumps(rows), encoding="utf-8")

    assert list(_iter_json_array(str(path), chunk_size=7)) == rows


def test_select_scenes_accepts_name_or_token_and_preserves_order():
    from scripts.export_nuscenes_review_videos import _select_scenes

    scenes = [
        {"token": "t1", "name": "scene-0001"},
        {"token": "t2", "name": "scene-0002"},
        {"token": "t3", "name": "scene-0003"},
    ]
    selected = _select_scenes(scenes, ["scene-0003", "t1", "t1"], offset=0, limit=None)
    assert [scene["token"] for scene in selected] == ["t3", "t1"]

    sliced = _select_scenes(scenes, None, offset=1, limit=1)
    assert [scene["token"] for scene in sliced] == ["t2"]

    with pytest.raises(ValueError, match="unknown scene"):
        _select_scenes(scenes, ["scene-missing"], offset=0, limit=None)


def test_collect_scene_frames_filters_channel_scene_and_mode(tmp_path):
    from scripts.export_nuscenes_review_videos import _collect_scene_frames

    records = [
        {"token": "sweep", "sample_token": "sample-a", "timestamp": 200,
         "filename": "sweeps/CAM_FRONT/sweep.jpg", "is_key_frame": False},
        {"token": "key", "sample_token": "sample-a", "timestamp": 100,
         "filename": "samples/CAM_FRONT/key.jpg", "is_key_frame": True},
        {"token": "left", "sample_token": "sample-a", "timestamp": 100,
         "filename": "samples/CAM_FRONT_LEFT/key.jpg", "is_key_frame": True},
        {"token": "other-scene", "sample_token": "sample-b", "timestamp": 100,
         "filename": "samples/CAM_FRONT/key.jpg", "is_key_frame": True},
    ]
    path = tmp_path / "sample_data.json"
    path.write_text(json.dumps(records), encoding="utf-8")
    sample_to_scene = {"sample-a": "scene-a", "sample-b": "scene-b"}

    all_frames = _collect_scene_frames(
        str(path), sample_to_scene, ["scene-a"], "CAM_FRONT", keyframes_only=False)
    assert [record["token"] for record in all_frames["scene-a"]] == ["key", "sweep"]

    keyframes = _collect_scene_frames(
        str(path), sample_to_scene, ["scene-a"], "CAM_FRONT", keyframes_only=True)
    assert [record["token"] for record in keyframes["scene-a"]] == ["key"]


def test_sidecar_maps_sweep_to_nearest_keyframe():
    from scripts.export_nuscenes_review_videos import _build_sidecar

    scene = {"token": "scene", "name": "scene-0001", "description": "test", "nbr_samples": 2}
    samples = {
        "sample-1": {"token": "sample-1", "scene_token": "scene", "timestamp": 1_000_000},
        "sample-2": {"token": "sample-2", "scene_token": "scene", "timestamp": 1_500_000},
    }
    records = [
        {"token": "sd-1", "sample_token": "sample-1", "timestamp": 1_000_000,
         "filename": "samples/CAM_FRONT/1.jpg", "is_key_frame": True},
        {"token": "sd-2", "sample_token": "sample-2", "timestamp": 1_400_000,
         "filename": "sweeps/CAM_FRONT/2.jpg", "is_key_frame": False},
    ]

    sidecar = _build_sidecar(
        scene, records, samples, "v1.0-mini", "CAM_FRONT", 2.0, "scene.mp4", 0)
    assert sidecar["frames"][1]["nearest_keyframe_sample_token"] == "sample-2"
    assert sidecar["frames"][1]["nearest_keyframe_delta_sec"] == -0.1
    assert sidecar["video"]["frame_count"] == 2


def test_infer_fps_preserves_timestamp_span():
    from scripts.export_nuscenes_review_videos import _infer_fps

    records = [
        {"timestamp": 1_000_000},
        {"timestamp": 1_100_000},
        {"timestamp": 1_250_000},
    ]
    # Two encoded frame intervals span 0.25 seconds, despite irregular source
    # intervals of 0.10 and 0.15 seconds.
    assert _infer_fps(records) == 8.0
