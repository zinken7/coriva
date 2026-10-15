import json
import os
from pathlib import Path
import subprocess
import sys

import pytest


SCRIPT = Path(__file__).resolve().parents[1] / "scripts/package_nuscenes_review.py"


@pytest.fixture
def review(tmp_path):
    source = tmp_path / "results"
    videos = source / "videos"
    videos.mkdir(parents=True)
    (source / "review_selections_approved.csv").write_bytes(
        b"scene_name,t0_sec\r\nscene-0001,5\r\n")
    (source / "review_manifest_approved.json").write_text(json.dumps({
        "selections": [{"scene_name": "scene-0001", "t0_sec": 5,
                        "critical_instance_token": "actor-token",
                        "intervention_type": "npc_sudden_brake",
                        "audit_status": "approved", "interaction_within_3s": True}]}))
    for name in ("scene-0001", "scene-0002"):
        (videos / f"{name}_CAM_FRONT.json").write_text(json.dumps({
            "scene": {"name": name}, "video": {"fps": 12},
            "frames": [{"is_key_frame": True, "source_sample_token": "sample",
                        "video_time_sec": 5, "video_frame_index": 60}]}))
        (videos / f"{name}_CAM_FRONT.mp4").write_bytes(b"video fixture")
    return source


def run(source, output, *extra):
    env = dict(os.environ)
    env.pop("PYTHONPATH", None)
    return subprocess.run(
        [sys.executable, "-B", str(SCRIPT), "--source", str(source),
         "--output", str(output), *extra], cwd=source.parent,
        env=env, capture_output=True, text=True)


@pytest.mark.parametrize("include_videos", [False, True])
def test_standalone_package_copies_only_approved_files(review, tmp_path, include_videos):
    output = tmp_path / "upload"
    result = run(review, output, *(["--include-videos"] if include_videos else []))
    assert result.returncode == 0, result.stderr
    expected = {"review_selections_approved.csv", "review_manifest_approved.json",
                "videos/scene-0001_CAM_FRONT.json"}
    if include_videos:
        expected.add("videos/scene-0001_CAM_FRONT.mp4")
    assert {str(p.relative_to(output)) for p in output.rglob("*") if p.is_file()} == expected
    for name in expected:
        assert (output / name).read_bytes() == (review / name).read_bytes()
    assert (review / "videos/scene-0002_CAM_FRONT.mp4").is_file()


@pytest.mark.parametrize("problem", ["missing_json", "missing_video", "t0", "scene", "approval"])
def test_invalid_input_leaves_no_package(review, tmp_path, problem):
    if problem.startswith("missing_"):
        extension = "json" if problem == "missing_json" else "mp4"
        (review / f"videos/scene-0001_CAM_FRONT.{extension}").unlink()
    elif problem == "scene":
        (review / "videos/scene-0001_CAM_FRONT.json").write_text('{"scene":{"name":"scene-0002"}}')
    else:
        path = review / "review_manifest_approved.json"
        data = json.loads(path.read_text())
        data["selections"][0]["t0_sec" if problem == "t0" else "audit_status"] = (
            6 if problem == "t0" else "pending")
        path.write_text(json.dumps(data))
    output = tmp_path / "upload"
    result = run(review, output, "--include-videos")
    assert result.returncode != 0
    assert "ERROR:" in result.stderr
    assert not output.exists()


def test_existing_destination_is_preserved(review, tmp_path):
    output = tmp_path / "upload"
    output.mkdir()
    marker = output / "keep.txt"
    marker.write_text("untouched")
    assert run(review, output).returncode != 0
    assert list(output.iterdir()) == [marker]
    assert marker.read_text() == "untouched"


def test_copy_failure_cleans_partial_package(review, tmp_path, monkeypatch):
    from scripts import package_nuscenes_review as package

    def fail(*args, **kwargs):
        raise OSError("disk full")

    monkeypatch.setattr(package.shutil, "copy2", fail)
    output = tmp_path / "upload"
    before = set(tmp_path.iterdir())
    with pytest.raises(OSError, match="disk full"):
        package.package_review(review, output)
    assert set(tmp_path.iterdir()) == before
