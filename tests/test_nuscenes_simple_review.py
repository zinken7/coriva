import csv
import json
from pathlib import Path

import pytest


def test_export_command_runs_directly_without_pythonpath(tmp_path):
    import os
    import shutil
    import subprocess
    import sys
    from scripts import review_nuscenes_simple as simple

    scripts = tmp_path / "scripts"
    scripts.mkdir()
    for name in ("review_nuscenes_simple.py", "nuscenes_review_queue.py"):
        shutil.copy2(simple.PROJECT_ROOT / "scripts" / name, scripts / name)
    overlays = tmp_path / "results/overlays"
    overlays.mkdir(parents=True)
    (overlays / "index.json").write_text('{"valid": [], "rejected": []}')
    env = dict(os.environ)
    env.pop("PYTHONPATH", None)
    result = subprocess.run(
        [sys.executable, "-B", "scripts/review_nuscenes_simple.py", "export-no-candidates"],
        cwd=tmp_path, env=env, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert (tmp_path / "results/overlays_no_candidates/index.json").is_file()


def test_actor_csv_prefills_single_candidate_and_leaves_multiple_choice(tmp_path, monkeypatch):
    from scripts import review_nuscenes_simple as simple

    target = tmp_path / "actor_review.csv"
    monkeypatch.setattr(simple, "ACTOR_REVIEW_CSV", target)
    simple._write_actor_review_csv({
        "scene-0097": ["cand_003", "cand_008"],
        "scene-0098": ["cand_012"],
    })
    assert target.read_text() == (
        'scene_name,candidate_label,interaction_within_3s,notes\n'
        'scene-0097,cand_,yes,""\n'
        'scene-0098,cand_012,yes,""\n')
    with pytest.raises(ValueError, match="đã tồn tại"):
        simple._write_actor_review_csv({"scene-0099": ["cand_001"]})


def test_export_no_candidates_copies_only_current_empty_scenes(tmp_path):
    from scripts.review_nuscenes_simple import export_no_candidate_overlays

    source = tmp_path / "overlays"
    rows = []
    for name, allowed in (("scene-0001", []), ("scene-0002", ["cand_001"]),
                          ("scene-stale", [])):
        scene = source / name
        scene.mkdir(parents=True)
        for frame in range(8):
            (scene / f"{frame:03d}.jpg").write_bytes(b"image")
        (scene / "overlay_manifest.json").write_text(json.dumps({
            "admissible_stable_candidates": allowed}))
        rows.append({"scene_name": name, "admissible_stable_candidates": allowed})
    (source / "index.json").write_text(json.dumps({"valid": rows[:2], "rejected": []}))
    target = tmp_path / "overlays_no_candidates"
    assert export_no_candidate_overlays(source, target) == 1
    assert sorted(p.name for p in target.iterdir()) == ["index.json", "scene-0001"]
    for path in (source / "scene-0001").iterdir():
        assert (target / "scene-0001" / path.name).read_bytes() == path.read_bytes()
    index = json.loads((target / "index.json").read_text())
    assert index["valid"][0]["manifest"] == str((target / "scene-0001" / "overlay_manifest.json").resolve())
    with pytest.raises(ValueError, match="đã tồn tại"):
        export_no_candidate_overlays(source, target)


def test_export_no_candidates_rejects_missing_frames_without_partial_output(tmp_path):
    from scripts.review_nuscenes_simple import export_no_candidate_overlays

    source = tmp_path / "overlays"
    scene = source / "scene-0001"
    scene.mkdir(parents=True)
    (scene / "overlay_manifest.json").write_text('{"admissible_stable_candidates": []}')
    (source / "index.json").write_text(json.dumps({
        "valid": [{"scene_name": "scene-0001", "admissible_stable_candidates": []}],
        "rejected": [],
    }))
    target = tmp_path / "overlays_no_candidates"
    with pytest.raises(ValueError, match="000.jpg"):
        export_no_candidate_overlays(source, target)
    assert not target.exists()


def test_scene_names_for_downloaded_blob_parts():
    from scripts.review_nuscenes_simple import scene_names_for_parts

    scenes = [{"name": f"scene-{index:04d}"} for index in range(1, 851)]
    names = scene_names_for_parts(scenes, [1, 3])

    assert len(names) == 170
    assert names[:2] == ["scene-0001", "scene-0002"]
    assert names[84:87] == ["scene-0085", "scene-0171", "scene-0172"]


def _write_actor_csv(path: Path, rows: list[list[str]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow([
            "scene_name", "candidate_label", "interaction_within_3s", "notes"])
        writer.writerows(rows)


def test_actor_review_csv_becomes_approved_or_skipped_decisions(tmp_path):
    from scripts.review_nuscenes_simple import actor_decisions_from_csv

    overlay_dir = tmp_path / "overlays"
    for scene_name in ("scene-0001", "scene-0002"):
        scene_dir = overlay_dir / scene_name
        scene_dir.mkdir(parents=True)
        (scene_dir / "overlay_manifest.json").write_text("{}\n", encoding="utf-8")
    csv_path = tmp_path / "actor_review.csv"
    _write_actor_csv(csv_path, [
        ["scene-0001", "cand_007", "yes", "Stable in all eight frames."],
        ["scene-0002", "", "", ""],
    ])

    decisions = actor_decisions_from_csv(
        csv_path, overlay_dir, ["scene-0001", "scene-0002"])

    assert decisions[0]["status"] == "approved"
    assert decisions[0]["candidate_label"] == "cand_007"
    assert decisions[1]["status"] == "skipped"

    _write_actor_csv(csv_path, [
        ["scene-0001", "cand_007", "no", "Not plausible."],
        ["scene-0002", "", "", ""],
    ])
    with pytest.raises(ValueError, match="phải là yes"):
        actor_decisions_from_csv(
            csv_path, overlay_dir, ["scene-0001", "scene-0002"])
