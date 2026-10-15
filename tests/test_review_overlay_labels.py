import pytest


def test_saved_diagnostic_frames_show_all_visible_candidates_without_raw(tmp_path, monkeypatch):
    from simulation import review_overlay as overlay

    source = tmp_path / "source"
    target = tmp_path / "target"
    source.mkdir()
    for i in range(8):
        (source / f"{i:03d}.jpg").write_bytes(b"unlabelled image")
    manifest = {"admissible_stable_candidates": [], "rendered_candidate_labels": [],
                "candidates": {
                    "cand_001": {"frames": [{"visible": True, "u": 10, "v": 20}] * 8},
                    "cand_002": {"frames": [{"visible": False}] + [
                        {"visible": True, "u": 30, "v": 40}] * 7}}}
    calls = []
    monkeypatch.setattr(overlay, "overlay_font_path", lambda: "font.ttf")
    monkeypatch.setattr(overlay.shutil, "which", lambda _: "magick")
    monkeypatch.setattr(overlay.subprocess, "run", lambda args, **kwargs: calls.append(args))
    overlay.render_saved_candidate_labels(manifest, source, target)
    assert len(calls) == 8
    for i, args in enumerate(calls):
        assert args[1] == str(source / f"{i:03d}.jpg")
        assert "cand_001" in args
        assert ("cand_002" in args) == (i > 0)
        assert args[-1] == str(target / f"{i:03d}.jpg")


@pytest.mark.parametrize("effect_passed", [True, False])
def test_overlay_draws_only_admissible_labels_but_keeps_all_diagnostics(tmp_path, monkeypatch, effect_passed):
    from simulation import review_overlay as overlay
    from simulation import nuscenes_adapter as adapter
    from simulation import nuscenes_admission as admission

    tokens = [f"s{i}" for i in range(8)]
    tables = {"sample": {token: {"prev": tokens[i - 1] if i else None}
                         for i, token in enumerate(tokens)}, "ego_pose": {"pose": {}}}
    monkeypatch.setattr(overlay.shutil, "which", lambda _: "magick")
    monkeypatch.setattr(overlay, "overlay_font_path", lambda: "font.ttf")
    monkeypatch.setattr(overlay, "collect_consecutive_cam_frames", lambda *args: [f"{i}.jpg" for i in range(8)])
    monkeypatch.setattr(overlay, "_camera_records", lambda *args: {"s7": {"ego_pose_token": "pose"}})
    monkeypatch.setattr(overlay, "_candidate_annotations", lambda *args: [
        {"annotation": {"instance_token": token}, "entity_type": "vehicle"}
        for token in ("accepted", "bad_role", "unstable")])
    monkeypatch.setattr(adapter, "extract_observed_state", lambda *args, **kwargs: {
        "actors": [{"id": "ego", "role_type": "ego_agent"}] + [
            {"id": token, "source_instance_token": token} for token in ("accepted", "bad_role", "unstable")]})
    monkeypatch.setattr(adapter, "estimate_root_steer", lambda *args: 0)
    monkeypatch.setattr(admission, "resolve_nuscenes_intervention", lambda actor: {
        "passed": actor["id"] != "bad_role", "selected_intervention": "npc_brake"})
    monkeypatch.setattr(admission, "canonical_environment_query", lambda *args: {})
    monkeypatch.setattr(admission, "evaluate_nuscenes_effect", lambda *args, **kwargs: {"passed": effect_passed})
    monkeypatch.setattr(overlay, "actor_visibility", lambda tables, sample, instance, channel: {
        "visible": not (instance == "unstable" and sample == "s0"), "u": 10, "v": 20})
    commands = []
    monkeypatch.setattr(overlay.subprocess, "run", lambda args, **kwargs: commands.append(args))

    result = overlay.build_review_overlay(tables, "s7", str(tmp_path), str(tmp_path / "out"))
    assert len(commands) == 8
    for args in commands:
        assert ("cand_001" in args) == effect_passed
        assert "cand_002" not in args and "cand_003" not in args
    assert set(result["candidates"]) == {"cand_001", "cand_002", "cand_003"}
    assert result["admissible_stable_candidates"] == (["cand_001"] if effect_passed else [])
