"""Collaborative, two-pass nuScenes human-review queue."""

import csv
import json
import os
import sys

import pytest


sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


def _write_sidecar(path, scene_name, keyframe_count=14):
    frames = [
        {
            "video_frame_index": index,
            "video_time_sec": index * 0.5,
            "is_key_frame": True,
            "source_sample_token": f"sample-{index}",
        }
        for index in range(keyframe_count)
    ]
    path.write_text(json.dumps({
        "schema_version": 1,
        "scene": {"name": scene_name},
        "video": {
            "filename": f"{scene_name}_CAM_FRONT.mp4",
            "duration_sec": frames[-1]["video_time_sec"],
            "skipped_missing_source_frames": 0,
        },
        "frames": frames,
    }), encoding="utf-8")


def _write_csv(path, rows):
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["scene_name", "t0_sec"])
        writer.writeheader()
        writer.writerows(rows)


def test_discovery_partition_and_t0_window_validation(tmp_path):
    from scripts.nuscenes_review_queue import (
        discover_review_scenes,
        partition_assignments,
        validate_t0,
    )

    review_dir = tmp_path / "videos"
    review_dir.mkdir()
    for number in range(1, 5):
        scene = f"scene-{number:04d}"
        (review_dir / f"{scene}_CAM_FRONT.mp4").write_bytes(b"video")
        _write_sidecar(review_dir / f"{scene}_CAM_FRONT.json", scene)

    scenes = discover_review_scenes(review_dir)
    assert [row["scene_name"] for row in scenes] == [
        "scene-0001", "scene-0002", "scene-0003", "scene-0004"]

    assignments = partition_assignments(
        [row["scene_name"] for row in scenes], ["alice", "bob"])
    assert assignments == {
        "alice": ["scene-0001", "scene-0003"],
        "bob": ["scene-0002", "scene-0004"],
    }

    resolved = validate_t0(review_dir / "scene-0001_CAM_FRONT.json", 3.5)
    assert resolved["resolved_t0_sec"] == 3.5
    assert resolved["history_keyframes"] == 8
    assert resolved["future_keyframes"] == 6

    with pytest.raises(ValueError, match="at least 8 history keyframes"):
        validate_t0(review_dir / "scene-0001_CAM_FRONT.json", 1.0)


def test_prepare_trainval_dataroot_merges_cam_front_blob_parts(tmp_path):
    from scripts.nuscenes_review_queue import prepare_trainval_dataroot

    source = tmp_path / "trainval"
    metadata = source / "v1.0-trainval_meta"
    (metadata / "v1.0-trainval").mkdir(parents=True)
    (metadata / "maps").mkdir()
    (metadata / "v1.0-trainval" / "scene.json").write_text("[]", encoding="utf-8")
    (metadata / "maps" / "map.png").write_bytes(b"map")

    for part, filename in (("01", "one.jpg"), ("02", "two.jpg")):
        blob = source / f"v1.0-trainval{part}_blobs"
        for tree in ("samples", "sweeps"):
            camera = blob / tree / "CAM_FRONT"
            camera.mkdir(parents=True)
            (camera / filename).write_bytes(f"{tree}-{part}".encode())

    out = tmp_path / "data" / "nuscenes_trainval_full"
    summary = prepare_trainval_dataroot(source, out)

    assert summary == {
        "blob_parts": 2,
        "sample_links": 2,
        "sweep_links": 2,
    }
    assert (out / "v1.0-trainval" / "scene.json").read_text() == "[]"
    assert (out / "maps" / "map.png").read_bytes() == b"map"
    assert (out / "samples" / "CAM_FRONT" / "one.jpg").is_symlink()
    assert (out / "samples" / "CAM_FRONT" / "two.jpg").is_symlink()
    assert (out / "sweeps" / "CAM_FRONT" / "one.jpg").is_symlink()
    assert (out / "sweeps" / "CAM_FRONT" / "two.jpg").is_symlink()


def test_merge_t0_contributions_writes_minimal_csv_and_rejects_conflicts(tmp_path):
    from scripts.nuscenes_review_queue import merge_t0_contributions

    alice = tmp_path / "alice.json"
    bob = tmp_path / "bob.json"
    alice.write_text(json.dumps({
        "schema_version": 1,
        "stage": "t0_review",
        "reviewer": "alice",
        "decisions": [{
            "scene_name": "scene-0001", "status": "selected", "t0_sec": 4.0,
        }],
    }), encoding="utf-8")
    bob.write_text(json.dumps({
        "schema_version": 1,
        "stage": "t0_review",
        "reviewer": "bob",
        "decisions": [{
            "scene_name": "scene-0002", "status": "selected", "t0_sec": 5.0,
        }],
    }), encoding="utf-8")

    out = tmp_path / "review_selections.csv"
    rows = merge_t0_contributions([alice, bob], out)
    assert rows == [
        {"scene_name": "scene-0001", "t0_sec": 4.0},
        {"scene_name": "scene-0002", "t0_sec": 5.0},
    ]
    assert out.read_text(encoding="utf-8") == (
        "scene_name,t0_sec\nscene-0001,4\nscene-0002,5\n")

    bob.write_text(json.dumps({
        "schema_version": 1,
        "stage": "t0_review",
        "reviewer": "bob",
        "decisions": [{
            "scene_name": "scene-0001", "status": "selected", "t0_sec": 6.0,
        }],
    }), encoding="utf-8")
    with pytest.raises(ValueError, match="duplicate scene decision"):
        merge_t0_contributions([alice, bob], out)


def test_merge_actor_contributions_derives_authority_from_current_overlay(tmp_path):
    from scripts.nuscenes_review_queue import (
        file_sha256,
        merge_actor_contributions,
    )

    overlay_dir = tmp_path / "overlays"
    scene_dir = overlay_dir / "scene-0001"
    scene_dir.mkdir(parents=True)
    manifest_path = scene_dir / "overlay_manifest.json"
    manifest_path.write_text(json.dumps({
        "schema_version": 2,
        "overlay_version": "coriva-id-overlay-2",
        "coordinate_convention": {"y_axis": "positive_right_at_t0"},
        "selection": {
            "scene_name": "scene-0001",
            "requested_t0_sec": 4.0,
            "sample_token": "sample-token",
        },
        "admissible_stable_candidates": ["cand_001"],
        "candidates": {
            "cand_001": {
                "instance_token": "instance-token",
                "recommended_intervention": "pedestrian_enters_road",
                "role_state_applicability": {
                    "passed": True,
                    "selected_intervention": "pedestrian_enters_road",
                },
                "paired_canonical_twin_effect_gate": {"passed": True},
            },
            "cand_002": {
                "instance_token": "not-authorized",
                "recommended_intervention": None,
                "role_state_applicability": {
                    "passed": False,
                    "selected_intervention": None,
                },
                "paired_canonical_twin_effect_gate": None,
            },
        },
    }), encoding="utf-8")
    (overlay_dir / "index.json").write_text(json.dumps({
        "valid": [{
            "scene_name": "scene-0001",
            "admissible_stable_candidates": ["cand_001"],
        }],
        "rejected": [],
    }), encoding="utf-8")

    selections = tmp_path / "selected.csv"
    _write_csv(selections, [{"scene_name": "scene-0001", "t0_sec": 4.0}])
    contribution = tmp_path / "alice-actors.json"
    contribution.write_text(json.dumps({
        "schema_version": 1,
        "stage": "actor_review",
        "reviewer": "alice",
        "decisions": [{
            "scene_name": "scene-0001",
            "status": "approved",
            "candidate_label": "cand_001",
            "interaction_within_3s": True,
            "notes": "Stable identity and plausible road entry.",
            "overlay_manifest_sha256": file_sha256(manifest_path),
        }],
    }), encoding="utf-8")

    approved_csv = tmp_path / "approved.csv"
    review_manifest = tmp_path / "review_manifest.json"
    rows = merge_actor_contributions(
        overlay_dir,
        selections,
        [contribution],
        approved_csv,
        review_manifest,
    )
    assert rows == [{"scene_name": "scene-0001", "t0_sec": 4.0}]
    manifest = json.loads(review_manifest.read_text(encoding="utf-8"))
    assert manifest == {
        "schema_version": 1,
        "selections": [{
            "scene_name": "scene-0001",
            "t0_sec": 4.0,
            "critical_instance_token": "instance-token",
            "intervention_type": "pedestrian_enters_road",
            "audit_status": "approved",
            "interaction_within_3s": True,
            "reviewer": "alice",
            "notes": "Stable identity and plausible road entry.",
        }],
    }

    decision = json.loads(contribution.read_text(encoding="utf-8"))
    decision["decisions"][0]["candidate_label"] = "cand_002"
    contribution.write_text(json.dumps(decision), encoding="utf-8")
    with pytest.raises(ValueError, match="not admissible"):
        merge_actor_contributions(
            overlay_dir,
            selections,
            [contribution],
            approved_csv,
            review_manifest,
        )


def test_merge_approved_rounds_keeps_canonical_join_and_rejects_duplicates(tmp_path):
    from scripts.nuscenes_review_queue import merge_approved_rounds

    selections = []
    manifests = []
    for round_number, scene_name in ((1, "scene-0001"), (2, "scene-0002")):
        selection = tmp_path / f"round{round_number}.csv"
        manifest = tmp_path / f"round{round_number}.json"
        _write_csv(selection, [{"scene_name": scene_name, "t0_sec": round_number + 3}])
        manifest.write_text(json.dumps({
            "schema_version": 1,
            "selections": [{
                "scene_name": scene_name,
                "t0_sec": round_number + 3,
                "critical_instance_token": f"token-{round_number}",
                "intervention_type": "pedestrian_enters_road",
                "audit_status": "approved",
                "interaction_within_3s": True,
                "reviewer": "alice",
                "notes": "Reviewed.",
            }],
        }), encoding="utf-8")
        selections.append(selection)
        manifests.append(manifest)

    out_csv = tmp_path / "all.csv"
    out_manifest = tmp_path / "all.json"
    rows = merge_approved_rounds(
        selections, manifests, out_csv, out_manifest)
    assert [row["scene_name"] for row in rows] == ["scene-0001", "scene-0002"]
    assert len(json.loads(out_manifest.read_text())["selections"]) == 2

    with pytest.raises(ValueError, match="duplicate approved scene"):
        merge_approved_rounds(
            [selections[0], selections[0]],
            [manifests[0], manifests[0]],
            out_csv,
            out_manifest,
        )
