import json
import argparse
import csv

import pytest


def test_nearest_passing_t0_is_chosen_not_lowest_error():
    from scripts.recover_nuscenes_t0 import search_scene
    calls = []
    def evaluate(frame):
        calls.append(frame["video_time_sec"])
        return {"passed": frame["video_time_sec"] != 9.5,
                "epsilon": 1.8 if frame["video_time_sec"] == 10.5 else 0.1}
    frames = [{"video_time_sec": t, "sample_token": str(t)} for t in (8, 9.5, 10, 10.5, 12)]
    result = search_scene(frames, 10, "10", evaluate)
    assert calls == [9.5, 10.5]
    assert result["proposal"]["video_time_sec"] == 10.5
    assert result["proposal"]["epsilon"] == 1.8


def test_exhausted_search_records_failures():
    from scripts.recover_nuscenes_t0 import search_scene
    result = search_scene([{"video_time_sec": 3, "sample_token": "s"}], 5, "old",
                          lambda _: {"passed": False, "stage": "fidelity", "epsilon": 3})
    assert result["status"] == "no_valid_t0"
    assert result["attempts"][0]["stage"] == "fidelity"


def test_mechanical_preview_does_not_require_or_create_human_approval():
    from simulation.build_nuscenes_dataset import _validate_reviewed_actor, _check_actor_geometry
    from test_nuscenes_visibility_policy import edge_tables
    root = {"actors": [{"id": "npc_1", "source_instance_token": "actor", "entity_type": "vehicle",
                         "role_type": "lead_vehicle", "x": 10, "y": 0, "vx": 5, "vy": 0}]}
    row = {"critical_instance_token": "actor", "intervention_type": "npc_sudden_brake"}
    assert not _validate_reviewed_actor(edge_tables(), "s7", root, row)["ok"]
    result = _check_actor_geometry(edge_tables(), "s7", root, row)
    assert result["ok"]
    assert "interaction_within_3s" not in result
    assert "audit_status" not in result


def test_only_fidelity_rejections_are_selected():
    from scripts.recover_nuscenes_t0 import select_failures
    resolved = {"selections": [
        {"scene_name": "scene-0001", "status": "accepted", "sample_token": "s1"},
        {"scene_name": "scene-0002", "status": "rejected", "sample_token": "s2",
         "reasons": ["fidelity_gate: eps=3 > epsilon_max=2"]},
        {"scene_name": "scene-0003", "status": "rejected", "sample_token": "s3",
         "reasons": [{"kinematic_msgs": ["bad acceleration"]}]},
    ]}
    fidelity = {"per_scenario": [
        {"scene_name": "scene-0001", "admitted": True, "source_sample_token": "s1"},
        {"scene_name": "scene-0002", "admitted": False, "source_sample_token": "s2"},
        {"scene_name": "scene-0003", "admitted": True, "source_sample_token": "s3"},
    ]}
    assert [r["scene_name"] for r in select_failures(resolved, fidelity)] == ["scene-0002"]


def test_review_requires_explicit_decision_and_matching_proposal(tmp_path):
    from scripts.recover_nuscenes_t0 import read_decisions
    proposal = {"scene_name": "scene-0001", "proposal_id": "digest", "video_time_sec": 5}
    path = tmp_path / "review.csv"
    header = "scene_name,proposal_id,t0_sec,decision,notes\n"
    path.write_text(header + 'scene-0001,digest,5,,\n')
    with pytest.raises(ValueError, match="decision"):
        read_decisions(path, [proposal])
    path.write_text(header + 'scene-0001,old,5,yes,checked\n')
    with pytest.raises(ValueError, match="proposal"):
        read_decisions(path, [proposal])
    path.write_text(header + 'scene-0001,digest,5,yes,checked all frames\n')
    assert read_decisions(path, [proposal])["scene-0001"]["decision"] == "yes"


def test_portable_search_resume_and_finalize_preserve_originals(tmp_path, monkeypatch):
    from scripts import recover_nuscenes_t0 as recovery
    from simulation.nuscenes_visibility import visibility_policy_manifest
    from simulation.nuscenes_admission import DEFAULT_NUSCENES_ADMISSION_THRESHOLDS
    from test_nuscenes_visibility_policy import edge_tables
    from PIL import Image

    source, build, raw = (tmp_path / name for name in ("source", "build", "raw"))
    (source / "videos").mkdir(parents=True)
    build.mkdir()
    (raw / "v1.0-mini").mkdir(parents=True)
    names = ["scene-0001", "scene-0002", "scene-0003"]
    approved = [{"scene_name": name, "t0_sec": 3.0, "critical_instance_token": "actor",
                 "intervention_type": "npc_sudden_brake", "audit_status": "approved",
                 "interaction_within_3s": True, "reviewer": "original", "notes": "original notes"} for name in names]
    recovery.write_json(source / recovery.MANIFEST, {"schema_version": 2, "selections": approved})
    (source / recovery.SELECTIONS).write_text("scene_name,t0_sec\n" + ''.join(f"{name},3\n" for name in names))
    for name in names:
        recovery.write_json(source / "videos" / f"{name}_CAM_FRONT.json", {
            "scene": {"name": name}, "video": {"fps": 2}, "frames": [
                {"is_key_frame": True, "source_sample_token": f"s{i}", "video_time_sec": i * .5}
                for i in (5, 6, 7)]})
    policy = visibility_policy_manifest("bbox-area-v2")
    recovery.write_json(build / "resolved_selections.json", {
        "track": "primary", "dt_seconds": .5, "future_steps": 6, "visibility_policy": policy,
        "selections": [{"scene_name": name, "requested_t0_sec": 3., "sample_token": "s6",
                        "status": "accepted" if i == 0 else "rejected",
                        "reasons": ["fidelity_gate: eps=3"] if i == 1 else ["qc"]}
                       for i, name in enumerate(names)]})
    recovery.write_json(build / "fidelity_report.json", {"epsilon_max": 2., "twin_model": "bicycle",
        "per_scenario": [{"scene_name": name, "admitted": i != 1, "epsilon": 3 if i == 1 else 1,
                          "source_sample_token": "s6"} for i, name in enumerate(names)]})
    recovery.write_json(build / "nuscenes_effect_report.json", {
        "visibility_policy": policy, "thresholds": DEFAULT_NUSCENES_ADMISSION_THRESHOLDS.to_dict()})
    tables = edge_tables()
    monkeypatch.setattr(recovery, "load_tables", lambda *args: tables)
    calls = []
    def preview(tables, token, instance, intervention, env, **kwargs):
        calls.append((token, instance, intervention, kwargs["epsilon_max"]))
        if token == "s6":
            return {"passed": False, "stage": "fidelity", "epsilon": 3}
        if token == "s5":
            return {"passed": False, "stage": "fidelity", "epsilon": 4}
        return {"passed": True, "epsilon": 1.5, "critical_actor_id": "npc_1",
                "human_review_status": "pending", "root": {"actors": [{"id": "npc_1", "source_instance_token": "actor"}]}}
    monkeypatch.setattr(recovery, "preview_reviewed_candidate", preview)
    for i in range(8):
        path = raw / f"samples/CAM_FRONT/{i}.jpg"
        path.parent.mkdir(parents=True, exist_ok=True)
        Image.new("RGB", (100, 100), "white").save(path)
    originals = {p: p.read_bytes() for folder in (source, build) for p in folder.rglob("*") if p.is_file()}
    output = tmp_path / "recovery"
    args = argparse.Namespace(build_dir=build, review_root=source, dataroot=raw, version="v1.0-mini", output=output, resume=False)
    assert recovery.run_search(args) == 0
    assert [c[0] for c in calls] == ["s6", "s5", "s7"]
    assert all(c[1:] == ("actor", "npc_sudden_brake", 2.) for c in calls)
    assert len(list((output / "review").rglob("*.jpg"))) == 8
    report = recovery.read_json(output / "report.json")
    assert report["summary"]["preserved_accepted"] == 1
    assert report["ignored_rejected_names"] == ["scene-0003"]
    final = tmp_path / "final"
    with pytest.raises(ValueError, match="decision"):
        recovery.finalize(output, final, "Tyrone")
    assert not final.exists()
    with (output / "review.csv").open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    rows[0].update(decision="yes", notes="checked new eight frames")
    with (output / "review.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    saved_review = (output / "review.csv").read_bytes()
    args.resume = True
    assert recovery.run_search(args) == 0
    assert len(calls) == 3  # Completed scenes were not searched again.
    assert (output / "review.csv").read_bytes() == saved_review
    assert recovery.finalize(output, final, "Tyrone") == 2
    released = recovery.read_json(final / recovery.MANIFEST)["selections"]
    assert released[0] == approved[0]
    assert released[1]["t0_sec"] == 3.5
    assert released[1]["reviewer"] == "Tyrone"
    assert released[1]["notes"] != approved[1]["notes"]
    assert released[1]["audit_status"] == "approved"
    assert all(p.read_bytes() == content for p, content in originals.items())
    # A changed image cannot be approved by a subsequent finalize.
    (output / "review/scene-0002/frames/000.jpg").write_bytes(b"changed")
    with pytest.raises(ValueError, match="image changed"):
        recovery.finalize(output, tmp_path / "bad-final", "Tyrone")
