import json

import pytest


def edge_tables():
    identity = {"translation": [0, 0, 0], "rotation": [1, 0, 0, 0]}
    return {
        "sample": {f"s{i}": {"token": f"s{i}", "prev": f"s{i-1}" if i else "",
                               "scene_token": "scene", "timestamp": i * 500000} for i in range(8)},
        "sample_data": {f"d{i}": {"token": f"d{i}", "sample_token": f"s{i}", "is_key_frame": True,
                                    "filename": f"samples/CAM_FRONT/{i}.jpg", "width": 100, "height": 100,
                                    "ego_pose_token": "pose", "calibrated_sensor_token": "cal"} for i in range(8)},
        "ego_pose": {"pose": identity},
        "calibrated_sensor": {"cal": {**identity, "camera_intrinsic": [[100, 0, 50], [0, 100, 50], [0, 0, 1]]}},
        "sample_annotation": {f"a{i}": {"token": f"a{i}", "sample_token": f"s{i}", "instance_token": "actor",
                                        "translation": [6, 0, 10], "size": [2, 4, 2],
                                        "rotation": [1, 0, 0, 0]} for i in range(8)},
        "instance": {"actor": {"category_token": "car"}},
        "category": {"car": {"name": "vehicle.car"}},
    }


def test_bbox_admits_edge_actor_and_center_mode_preserves_old_rule():
    from simulation.build_nuscenes_dataset import _validate_reviewed_actor
    tables = edge_tables()
    root = {"actors": [{"id": "npc_1", "source_instance_token": "actor", "entity_type": "vehicle",
                         "role_type": "lead_vehicle", "x": 10, "y": 0, "vx": 5, "vy": 0}]}
    review = {"critical_instance_token": "actor", "audit_status": "approved",
              "interaction_within_3s": True, "intervention_type": "npc_sudden_brake"}
    result = _validate_reviewed_actor(tables, "s7", root, review)
    assert result["ok"], result
    assert len(result["visibility"]) == 8
    assert all(f["area_reference_px2"] >= 100 for f in result["visibility"])
    assert not _validate_reviewed_actor(tables, "s7", root, review, visibility_policy="center-v1")["ok"]


def test_bbox_policy_applies_to_root_universe_and_frame_labels(tmp_path):
    from PIL import Image
    from simulation.review_overlay import observable_root_universe
    from simulation.observed_clip import write_model_frames
    tables = edge_tables()
    assert observable_root_universe(tables, "s7")["allowed_instance_tokens"] == []
    universe = observable_root_universe(tables, "s7", visibility_policy="bbox-area-v2")
    assert universe["allowed_instance_tokens"] == ["actor"]
    assert all(universe["visibility_by_instance"]["actor"].values())
    for i in range(8):
        path = tmp_path / f"samples/CAM_FRONT/{i}.jpg"
        path.parent.mkdir(parents=True, exist_ok=True)
        Image.new("RGB", (100, 100), "white").save(path)
    root = {"actors": [{"id": "npc_1", "source_instance_token": "actor"}]}
    write_model_frames(tables, "s7", str(tmp_path), str(tmp_path / "bbox"), root,
                       visibility_policy="bbox-area-v2")
    write_model_frames(tables, "s7", str(tmp_path), str(tmp_path / "center"), root,
                       visibility_policy="center-v1")
    assert (tmp_path / "bbox/frames/007.jpg").read_bytes() != (tmp_path / "center/frames/007.jpg").read_bytes()


def test_small_centered_box_and_missing_geometry_do_not_pass_bbox():
    from simulation.nuscenes_visibility import policy_actor_visibility
    from simulation.review_overlay import actor_visibility
    tables = edge_tables()
    for ann in tables["sample_annotation"].values():
        ann.update(translation=[0, 0, 10], size=[0.001, 0.001, 0.001])
    assert policy_actor_visibility(tables, "s7", "actor", policy="center-v1") == actor_visibility(tables, "s7", "actor")
    assert not policy_actor_visibility(tables, "s7", "actor", policy="bbox-area-v2")["visible"]
    del tables["sample_annotation"]["a7"]["size"]
    assert not policy_actor_visibility(tables, "s7", "actor", policy="bbox-area-v2")["visible"]


@pytest.mark.parametrize("policy", [None, "center-v1"])
def test_cli_default_and_explicit_legacy_forward_policy(monkeypatch, policy):
    from simulation import build_nuscenes_dataset as builder
    received = {}
    def fake_build(*args, **kwargs):
        received.update(kwargs)
        return {"accepted": 0, "rejected": 0, "fidelity": {"candidates": 0},
                "effect_gate": {"candidates": 0, "passed": 0, "rejected": 0}}
    monkeypatch.setattr(builder, "build_dataset", fake_build)
    monkeypatch.setattr("sys.argv", ["builder"] + (["--visibility-policy", policy] if policy else []))
    builder.main()
    assert received["visibility_policy"] == (policy or "bbox-area-v2")


def test_builder_refuses_to_mix_output_runs(tmp_path):
    from simulation.build_nuscenes_dataset import build_dataset
    marker = tmp_path / "resolved_selections.json"
    marker.write_text('{"visibility_policy": {"name":"center-v1"}}')
    with pytest.raises(ValueError, match="output"):
        build_dataset("missing-raw", "v1.0-mini", 1, str(tmp_path))
    assert json.loads(marker.read_text())["visibility_policy"]["name"] == "center-v1"
