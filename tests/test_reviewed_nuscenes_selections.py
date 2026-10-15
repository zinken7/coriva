"""Manual nuScenes scene/T0 selections integrated into the main builder."""

import csv
import json
import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

DATAROOT = "data/nuscenes"
MINI = os.path.join(DATAROOT, "v1.0-mini")
HAS_NUSCENES = os.path.isdir(MINI)


def _write_csv(path, rows):
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["scene_name", "t0_sec"])
        writer.writeheader()
        writer.writerows(rows)


def test_read_review_selections_accepts_minimal_csv(tmp_path):
    from simulation.build_nuscenes_dataset import _read_review_selections

    path = tmp_path / "selected.csv"
    _write_csv(path, [{"scene_name": "scene-0061", "t0_sec": "9.5"}])
    rows = _read_review_selections(str(path))
    assert rows == [{
        "scene_name": "scene-0061",
        "requested_t0_sec": 9.5,
        "csv_line": 2,
    }]


def test_read_review_selections_rejects_bad_header(tmp_path):
    from simulation.build_nuscenes_dataset import _read_review_selections

    path = tmp_path / "selected.csv"
    path.write_text("scene,t0\nscene-0061,9.5\n", encoding="utf-8")
    with pytest.raises(ValueError, match="missing required column"):
        _read_review_selections(str(path))


def test_reviewed_actor_requires_explicit_three_second_interaction_audit():
    from simulation.build_nuscenes_dataset import _validate_reviewed_actor

    result = _validate_reviewed_actor({}, "sample", {"actors": []}, {
        "critical_instance_token": "instance",
        "intervention_type": "npc_sudden_brake",
        "audit_status": "approved",
    })
    assert result == {
        "ok": False,
        "error": "review_manifest_missing:interaction_within_3s",
    }

    result = _validate_reviewed_actor({}, "sample", {"actors": []}, {
        "critical_instance_token": "instance",
        "intervention_type": "npc_sudden_brake",
        "audit_status": "approved",
        "interaction_within_3s": False,
    })
    assert result["error"] == "review_manifest_interaction_within_3s_not_approved"


def test_reviewed_nuscenes_actor_rejects_static_sudden_appearance():
    from simulation.build_nuscenes_dataset import _validate_reviewed_actor

    result = _validate_reviewed_actor({}, "sample", {"actors": [{
        "id": "npc_1", "entity_type": "barrier",
        "source_instance_token": "barrier-instance",
    }]}, {
        "critical_instance_token": "barrier-instance",
        "intervention_type": "obstacle_sudden_appearance",
        "audit_status": "approved",
        "interaction_within_3s": True,
    })
    assert result["error"] == "unsupported_nuscenes_intervention_actor_type:barrier"


def test_reviewed_candidate_rejects_t0_without_complete_window(tmp_path):
    from simulation.build_nuscenes_dataset import _reviewed_candidates

    tables = {
        "scene": {"scene-token": {"token": "scene-token", "name": "scene-0001"}},
        "sample": {
            "sample-0": {
                "token": "sample-0", "scene_token": "scene-token",
                "prev": "", "next": "sample-1", "timestamp": 0,
            },
            "sample-1": {
                "token": "sample-1", "scene_token": "scene-token",
                "prev": "sample-0", "next": "", "timestamp": 500_000,
            },
        },
        "sample_data": {
            "sd-0": {
                "token": "sd-0", "sample_token": "sample-0", "is_key_frame": True,
                "filename": "samples/CAM_FRONT/0.jpg",
            },
            "sd-1": {
                "token": "sd-1", "sample_token": "sample-1", "is_key_frame": True,
                "filename": "samples/CAM_FRONT/1.jpg",
            },
        },
    }
    review_dir = tmp_path / "review"
    review_dir.mkdir()
    sidecar = {
        "scene": {"name": "scene-0001"},
        "video": {"fps": 2.0},
        "frames": [{
            "video_frame_index": 0, "video_time_sec": 0.0, "is_key_frame": True,
            "source_sample_token": "sample-0",
        }],
    }
    (review_dir / "scene-0001_CAM_FRONT.json").write_text(
        json.dumps(sidecar), encoding="utf-8")
    csv_path = tmp_path / "selected.csv"
    _write_csv(csv_path, [{"scene_name": "scene-0001", "t0_sec": 0.0}])

    rows = _reviewed_candidates(tables, str(csv_path), str(review_dir))
    assert len(rows) == 1
    assert "fewer than 8" in rows[0]["error"]


@pytest.mark.skipif(not HAS_NUSCENES, reason="nuScenes v1.0-mini not present")
def test_reviewed_candidates_snap_and_validate_window(tmp_path):
    from scripts.export_nuscenes_review_videos import _build_sidecar
    from simulation.build_nuscenes_dataset import _reviewed_candidates
    from simulation.nuscenes_adapter import load_tables

    tables = load_tables(DATAROOT, "v1.0-mini")
    scene = next(iter(tables["scene"].values()))
    tokens = []
    tok = scene["first_sample_token"]
    while tok:
        tokens.append(tok)
        tok = tables["sample"][tok]["next"]
    # Index 10 has >= 8 history samples and >= 12 successors.
    records = []
    for index, token in enumerate(tokens):
        records.append({
            "token": f"sd-{index}", "sample_token": token,
            "timestamp": tables["sample"][token]["timestamp"],
            "filename": f"samples/CAM_FRONT/{index}.jpg", "is_key_frame": True,
        })
    sidecar = _build_sidecar(
        scene, records, tables["sample"], "v1.0-mini", "CAM_FRONT", 2.0,
        f"{scene['name']}_CAM_FRONT.mp4", 0)
    review_dir = tmp_path / "review"
    review_dir.mkdir()
    (review_dir / f"{scene['name']}_CAM_FRONT.json").write_text(
        json.dumps(sidecar), encoding="utf-8")
    csv_path = tmp_path / "selected.csv"
    t0_sec = sidecar["frames"][10]["video_time_sec"]
    _write_csv(csv_path, [{"scene_name": scene["name"], "t0_sec": t0_sec + 0.01}])

    rows = _reviewed_candidates(tables, str(csv_path), str(review_dir))
    assert len(rows) == 1
    assert rows[0]["sample_token"] == tokens[10]
    assert rows[0]["error"] is None


@pytest.mark.skipif(not HAS_NUSCENES, reason="nuScenes v1.0-mini not present")
@pytest.mark.parametrize("visibility_policy", ["bbox-area-v2", "center-v1"])
def test_reviewed_builder_writes_8_frames_and_primary_6_step_contract(tmp_path, visibility_policy):
    from scripts.export_nuscenes_review_videos import _build_sidecar
    from simulation.build_nuscenes_dataset import build_dataset
    from simulation.nuscenes_adapter import load_tables, extract_observed_trajectory
    from simulation.nuscenes_admission import (
        canonical_environment_query, evaluate_nuscenes_effect,
        resolve_nuscenes_intervention)
    from simulation.review_overlay import observable_root_universe

    tables = load_tables(DATAROOT, "v1.0-mini")
    scene = next(iter(tables["scene"].values()))
    tokens = []
    tok = scene["first_sample_token"]
    while tok:
        tokens.append(tok)
        tok = tables["sample"][tok]["next"]
    cam = {
        sd["sample_token"]: sd
        for sd in tables["sample_data"].values()
        if sd.get("is_key_frame") and sd["filename"].startswith("samples/CAM_FRONT/")
    }
    records = [cam[token] for token in tokens]
    sidecar = _build_sidecar(
        scene, records, tables["sample"], "v1.0-mini", "CAM_FRONT", 2.0,
        f"{scene['name']}_CAM_FRONT.mp4", 0)
    review_dir = tmp_path / "review"
    review_dir.mkdir()
    (review_dir / f"{scene['name']}_CAM_FRONT.json").write_text(
        json.dumps(sidecar), encoding="utf-8")
    # Find one mini candidate that satisfies the same role/state and paired
    # effect gates used by the production builder.  Do not hard-code the first
    # visible vehicle: it may be parked or semantically irrelevant.
    selected_index = None
    stable = None
    intervention = None
    for index in range(7, len(tokens) - 6):
        universe = observable_root_universe(tables, tokens[index], 8, visibility_policy=visibility_policy)
        allowed = set(universe["allowed_instance_tokens"])
        for instance_token in allowed:
            if not all(universe["visibility_by_instance"][instance_token].values()):
                continue
            trajectory = extract_observed_trajectory(
                tables, tokens[index], 6, priority_instance_token=instance_token,
                allowed_instance_tokens=allowed)
            actor = next((a for a in trajectory["root"]["actors"]
                          if a.get("source_instance_token") == instance_token), None)
            if actor is None:
                continue
            applicability = resolve_nuscenes_intervention(actor)
            if not applicability["passed"]:
                continue
            effect = evaluate_nuscenes_effect(
                trajectory["root"], canonical_environment_query(actor["id"], applicability),
                model="point_mass")
            if effect["passed"]:
                selected_index = index
                stable = instance_token
                intervention = applicability["selected_intervention"]
                break
        if stable is not None:
            break
    assert selected_index is not None and stable is not None and intervention is not None
    t0_token = tokens[selected_index]
    selected_t0 = sidecar["frames"][selected_index]["video_time_sec"]
    csv_path = tmp_path / "selected.csv"
    _write_csv(csv_path, [{"scene_name": scene["name"], "t0_sec": selected_t0}])
    manifest_path = tmp_path / "review_manifest.json"
    manifest_path.write_text(json.dumps({"selections": [{
        "scene_name": scene["name"], "t0_sec": selected_t0,
        "critical_instance_token": stable, "critical_actor_id": "npc_1",
        "audit_status": "approved",
        "interaction_within_3s": True,
        "intervention_type": intervention,
    }]}), encoding="utf-8")

    out_dir = tmp_path / "out"
    result = build_dataset(
        DATAROOT, "v1.0-mini", limit=1, out_dir=str(out_dir),
        selections_csv=str(csv_path), review_dir=str(review_dir),
        review_manifest=str(manifest_path),
        epsilon_max=None, visibility_policy=visibility_policy)
    assert result["accepted"] == 1, result
    frames = sorted((out_dir / "scenario_1" / "frames").glob("*.jpg"))
    assert len(frames) == 8
    resolved = json.loads((out_dir / "resolved_selections.json").read_text())
    assert resolved["future_steps"] == 6
    assert resolved["visibility_policy"]["name"] == visibility_policy
    gt = json.loads((out_dir / "scenario_1" / "ground_truth.json").read_text())
    assert gt["source_scene_name"] == scene["name"]
    assert gt["source_sample_token"] == t0_token
    assert max(len(branch["chain"]) for branch in gt["branches"]) <= 6
    assert gt["protocol"]["future_horizon_seconds"] == 3.0
    assert gt["protocol"]["human_interaction_within_3s"] is True
    assert gt["protocol"]["role_state_applicability_gate"] == "passed"
    # L3 template selection (interaction-v3) is recorded with its candidate table,
    # applied T0 visibility, and only a visible, relevant actor can drive it.
    template = gt["scenario_template"]
    assert template["policy"] == "interaction-v3"
    assert gt["protocol"]["scenario_template_policy"] == "interaction-v3"
    assert template["t0_visibility_applied"] is True
    assert template["critical_actor_id"] == "npc_1"
    if template["selected_actor_id"] is not None:
        chosen = next(c for c in template["candidates"] if c["id"] == template["selected_actor_id"])
        assert chosen["visible_at_t0"] and chosen["relevant"]
    manifest = json.loads((out_dir / "scenario_1" / "input_manifest.json").read_text())
    assert manifest["scenario_template_policy"] == "interaction-v3"
    assert manifest["scenario_template_actor_id"] == template["selected_actor_id"]
    # The reference generator is identified beyond the protocol version so that
    # campaign admission can tell builds apart: template
    # policy, twin semantics and the builder's code commit, in GT and manifest.
    from simulation.kinematic_twin import TWIN_VERSION
    assert TWIN_VERSION == "kinematic-twin-5"
    assert template["interaction_probe"]["twin_model"] == gt["twin_model"]
    for block in (gt["protocol"], manifest):
        assert block["scenario_template_policy"] == "interaction-v3"
        assert block["twin_version"] == TWIN_VERSION
        assert "build_code_commit" in block and "build_worktree_clean" in block
        assert block["build_code_commit"] is None or len(block["build_code_commit"]) == 40
    assert gt["protocol"]["build_code_commit"] == manifest["build_code_commit"]
    assert gt["protocol"]["canonical_twin_effect_gate"] == "passed"
    assert gt["nuscenes_admission"]["paired_effect_gate"]["passed"] is True
    from simulation.build_nuscenes_dataset import preview_reviewed_candidate
    from ctms.ctms import load_ctms_environment
    preview_env = load_ctms_environment("ctms/ontology.yaml", "ctms/ctms_config.yaml",
                                        "ctms/schema_branch.json", "ctms/schema_counterfactual.json")
    preview = preview_reviewed_candidate(tables, t0_token, stable, intervention,
        preview_env, epsilon_max=100.0, visibility_policy=visibility_policy)
    assert preview["passed"], preview
    assert preview["epsilon"] == gt["factual_fidelity_error"]
    assert preview["root"] == gt["observed_state"]
    assert preview["human_review_status"] == "pending"
    failed_preview = preview_reviewed_candidate(tables, t0_token, stable, intervention,
        preview_env, epsilon_max=1e-9, visibility_policy=visibility_policy)
    assert not failed_preview["passed"] and failed_preview["stage"] == "fidelity"
    assert gt["protocol"]["visibility_policy"]["name"] == visibility_policy
    measurements = gt["nuscenes_admission"]["critical_actor_visibility_measurements"]
    assert len(measurements) == 8 and all(row["visible"] for row in measurements)
    if visibility_policy == "bbox-area-v2":
        assert all(row["area_reference_px2"] >= 100 for row in measurements)
    else:
        assert all("area_reference_px2" not in row for row in measurements)
    inputs = json.loads((out_dir / "scenario_1/input_manifest.json").read_text())
    assert inputs["visibility_policy"] == gt["protocol"]["visibility_policy"]
    assert inputs["critical_actor_visibility_measurements"] == measurements
    assert resolved["selections"][0]["critical_actor_visibility_measurements"] == measurements
    assert gt["protocol"]["primary_measurement"] == "ground_truth_free_l3_causal_coherence"
    assert gt["protocol"]["reference_evaluation_role"] == "secondary_declared_reference_agreement"
    assert gt["protocol"]["reference_semantics"] == "observed_factual_and_canonical_twin_agreement"
    assert gt["protocol"]["l4_reference_source"] == "deterministic_canonical_twin"
    effect_report = json.loads((out_dir / "nuscenes_effect_report.json").read_text())
    assert effect_report["summary"]["passed"] == 1
    assert effect_report["summary"]["released_after_all_gates"] == 1
    assert effect_report["semantics"] == "canonical_twin_applicability_not_physical_truth"
    assert effect_report["visibility_policy"]["name"] == visibility_policy
