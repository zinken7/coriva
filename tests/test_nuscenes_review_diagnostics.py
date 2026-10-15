import json

import pytest

from scripts import review_nuscenes_simple as simple


def _manifest():
    return {
        "selection": {"requested_t0_sec": 9, "resolved_t0_sec": 9.1},
        "stable_candidates": ["cand_002"],
        "role_state_eligible_stable_candidates": ["cand_002"],
        "admissible_stable_candidates": [],
        "candidates": {
            "cand_001": {
                "frames": [{"visible": True}] * 7 + [
                    {"visible": False, "reason": "outside_image"}],
                "role_state_applicability": {"passed": True},
                "paired_canonical_twin_effect_gate": {"passed": True},
            },
            "cand_002": {
                "frames": [{"visible": True}] * 8,
                "t0_state": {"vx": 4.582},
                "role_state_applicability": {"passed": True},
                "paired_canonical_twin_effect_gate": {
                    "passed": False,
                    "reasons": ["critical_actor_does_not_interact_with_ego_within_3s"],
                    "min_treated_ego_target_edge_gap_m": 9.778,
                    "thresholds": {"max_interaction_edge_gap_m": 5.0},
                },
            },
        },
    }


def test_diagnostic_reports_first_failed_gate_and_exact_evidence():
    from scripts.nuscenes_review_diagnostics import format_scene_diagnostics

    output = format_scene_diagnostics("scene-0001", _manifest(), details=True)
    assert "admissible=0" in output
    assert "visibility=1, role_state=0, effect=1" in output
    assert "007:outside_image" in output
    assert "7/8" in output
    assert "9.778" in output and "5.0" in output
    assert "critical_actor_does_not_interact_with_ego_within_3s" in output


def test_diagnose_uses_current_index_without_changing_files(tmp_path, capsys):
    from scripts.nuscenes_review_diagnostics import diagnose_round

    scene = tmp_path / "scene-0001"
    scene.mkdir()
    (scene / "overlay_manifest.json").write_text(json.dumps(_manifest()))
    accepted = tmp_path / "scene-0003"
    accepted.mkdir()
    accepted_manifest = _manifest()
    accepted_manifest["admissible_stable_candidates"] = ["cand_002"]
    (accepted / "overlay_manifest.json").write_text(json.dumps(accepted_manifest))
    (tmp_path / "index.json").write_text(json.dumps({
        "valid": [{"scene_name": "scene-0001"}, {"scene_name": "scene-0003"}],
        "rejected": [{"scene_name": "scene-0002", "error": "incomplete history"}],
    }))
    stale = tmp_path / "scene-stale"
    stale.mkdir()
    (stale / "overlay_manifest.json").write_text("not JSON")
    before = {p: p.read_bytes() for p in tmp_path.rglob("*") if p.is_file()}
    diagnose_round(tmp_path)
    output = capsys.readouterr().out
    assert "scene-0001" in output
    assert "scene-0003" not in output
    assert "scene-0002" not in output
    diagnose_round(tmp_path, "scene-0002")
    assert "incomplete history" in capsys.readouterr().out
    diagnose_round(tmp_path, "scene-0003")
    assert "scene-0003" in capsys.readouterr().out
    assert before == {p: p.read_bytes() for p in tmp_path.rglob("*") if p.is_file()}
    with pytest.raises(ValueError, match="scene-stale"):
        diagnose_round(tmp_path, "scene-stale")


def test_diagnose_cli_does_not_bootstrap_or_render(monkeypatch, tmp_path):
    from scripts import nuscenes_review_diagnostics as diagnostics

    called = []
    monkeypatch.setattr(simple, "OVERLAYS_DIR", tmp_path)
    monkeypatch.setattr(simple.sys, "argv", ["review_nuscenes_simple.py", "diagnose", "--scene", "scene-0001"])
    monkeypatch.setattr(simple, "_bootstrap_or_reexec", lambda _: pytest.fail("must not bootstrap"))
    monkeypatch.setattr(diagnostics, "diagnose_round", lambda *args: called.append(args))
    assert simple.main() == 0
    assert called == [(tmp_path, "scene-0001")]


def test_role_rejections_and_empty_manifest_are_not_reported_as_effect_failures():
    from scripts.nuscenes_review_diagnostics import format_scene_diagnostics

    manifest = _manifest()
    manifest["role_state_eligible_stable_candidates"] = []
    manifest["candidates"]["cand_002"]["role_state_applicability"] = {
        "passed": False, "reasons": ["unsupported_nuscenes_intervention_actor_type:barrier"]}
    manifest["candidates"]["cand_002"]["paired_canonical_twin_effect_gate"] = None
    output = format_scene_diagnostics("scene-0001", manifest)
    assert "visibility=1, role_state=1, effect=0" in output
    assert "unsupported_nuscenes_intervention_actor_type:barrier" in output
    manifest.update(candidates={}, stable_candidates=[])
    assert "Không có annotation candidate tại T0" in format_scene_diagnostics("empty", manifest)
