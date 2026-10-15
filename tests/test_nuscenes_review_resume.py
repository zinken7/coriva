import csv
import json
from pathlib import Path

import pytest


def setup_round(tmp_path):
    from scripts import nuscenes_review_queue as queue
    source = tmp_path / "overlays"
    scene = source / "scene-0001"
    scene.mkdir(parents=True)
    selection = {"scene_name": "scene-0001", "requested_t0_sec": 4.0,
                 "resolved_t0_sec": 4.0, "sample_token": "old"}
    manifest = {"selection": selection, "sample_token": "old", "candidates": {},
                "stable_candidates": [], "role_state_eligible_stable_candidates": [],
                "admissible_stable_candidates": []}
    (scene / "overlay_manifest.json").write_text(json.dumps(manifest))
    for i in range(8):
        (scene / f"{i:03d}.jpg").write_bytes(b"old image")
    queue._write_json_atomic(source / "index.json", {"valid": [
        {**selection, **{k: manifest[k] for k in (
            "stable_candidates", "role_state_eligible_stable_candidates", "admissible_stable_candidates")}}], "rejected": []})
    selections = tmp_path / "selections.csv"
    queue._write_minimal_csv(selections, [{"scene_name": "scene-0001", "t0_sec": 4.0}])
    return source, selections


def test_resume_without_old_raw_data_or_rendering(tmp_path, monkeypatch):
    from scripts import nuscenes_review_queue as queue
    from simulation import nuscenes_adapter as adapter
    source, selections = setup_round(tmp_path)
    before = {p: p.read_bytes() for p in source.rglob('*') if p.is_file()}
    monkeypatch.setattr(adapter, "load_tables", lambda *args: pytest.fail("no raw data needed"))
    summary = queue.make_overlay_round(tmp_path / "deleted_raw", tmp_path, selections, source, resume=True)
    assert summary["skipped"] == 1 and summary["created"] == 0
    assert before == {p: p.read_bytes() for p in source.rglob('*') if p.is_file()}


@pytest.mark.parametrize('has_candidate', [False, True])
def test_changed_t0_allowed_only_for_previously_empty_scene(tmp_path, has_candidate):
    from scripts import nuscenes_review_queue as queue
    source, selections = setup_round(tmp_path)
    if has_candidate:
        path = source/'scene-0001/overlay_manifest.json'
        manifest = json.loads(path.read_text())
        manifest['admissible_stable_candidates'] = ['cand_001']
        path.write_text(json.dumps(manifest))
        path = source/'index.json'
        index = json.loads(path.read_text())
        index['valid'][0]['admissible_stable_candidates'] = ['cand_001']
        path.write_text(json.dumps(index))
    selections.write_text('scene_name,t0_sec\nscene-0001,5\n')
    if has_candidate:
        with pytest.raises(ValueError, match='T0'):
            queue.plan_overlay_resume(selections, source, allow_empty_t0_changes=True)
    else:
        saved, pending = queue.plan_overlay_resume(selections, source, allow_empty_t0_changes=True)
        assert saved[0]['requested_t0_sec'] == 4.0
        assert pending == [{'scene_name': 'scene-0001', 't0_sec': 5.0}]


@pytest.mark.parametrize('has_candidate', [False, True])
def test_remove_only_empty_scene_without_raw_and_archive_inspection(tmp_path, monkeypatch, has_candidate):
    from scripts import nuscenes_review_queue as queue, review_nuscenes_simple as simple
    from simulation import nuscenes_adapter as adapter
    source, selections = setup_round(tmp_path)
    target = tmp_path/'empty'
    simple.export_no_candidate_overlays(source, target)
    if has_candidate:
        manifest_path = source/'scene-0001/overlay_manifest.json'
        manifest = json.loads(manifest_path.read_text())
        manifest['admissible_stable_candidates'] = ['cand_001']
        manifest_path.write_text(json.dumps(manifest))
        index_path = source/'index.json'
        index = json.loads(index_path.read_text())
        index['valid'][0]['admissible_stable_candidates'] = ['cand_001']
        index_path.write_text(json.dumps(index))
    old_hash = queue.file_sha256(source/'scene-0001/overlay_manifest.json')
    selections.write_text('scene_name,t0_sec\n')
    monkeypatch.setattr(adapter, 'load_tables', lambda *args: pytest.fail('removal needs no raw'))
    if has_candidate:
        with pytest.raises(ValueError, match='scene-0001'):
            queue.make_overlay_round(tmp_path, tmp_path, selections, source, resume=True,
                                     allow_empty_removals=True)
        assert (source/'scene-0001').exists()
        return
    summary = queue.make_overlay_round(tmp_path, tmp_path, selections, source, resume=True,
                                       allow_empty_removals=True)
    assert summary['removed'] == 1 and summary['valid'] == 0
    assert not (source/'scene-0001').exists()
    assert not (tmp_path/'overlays_history').exists()
    simple.export_no_candidate_overlays(source, target, resume=True)
    assert not (target/'scene-0001').exists()
    assert not (tmp_path/'empty_history').exists()
    summary = queue.make_overlay_round(tmp_path, tmp_path, selections, source, resume=True,
                                       allow_empty_removals=True)
    assert summary['removed'] == 0
    simple.export_no_candidate_overlays(source, target, resume=True)


def test_removal_restores_frames_when_index_write_fails(tmp_path, monkeypatch):
    from scripts import nuscenes_review_queue as queue
    source, selections = setup_round(tmp_path)
    selections.write_text('scene_name,t0_sec\n')
    before = {p: p.read_bytes() for p in source.rglob('*') if p.is_file()}
    def fail(*args):
        raise OSError('index write failed')
    monkeypatch.setattr(queue, '_write_json_atomic', fail)
    with pytest.raises(OSError, match='index write failed'):
        queue.make_overlay_round(tmp_path, tmp_path, selections, source, resume=True, allow_empty_removals=True)
    assert before == {p: p.read_bytes() for p in source.rglob('*') if p.is_file()}
    assert not list(source.glob('.remove-*'))


def test_inspection_restores_old_folder_when_index_write_fails(tmp_path, monkeypatch):
    import shutil
    from scripts import nuscenes_review_queue as queue, review_nuscenes_simple as simple
    source, selections = setup_round(tmp_path)
    target = tmp_path/'empty'
    simple.export_no_candidate_overlays(source, target)
    before = {p: p.read_bytes() for p in target.rglob('*') if p.is_file()}
    shutil.copytree(source/'scene-0001', source/'scene-0002')
    index = json.loads((source/'index.json').read_text())
    index['valid'].append({**index['valid'][0], 'scene_name': 'scene-0002'})
    (source/'index.json').write_text(json.dumps(index))
    def fail(*args):
        raise OSError('index write failed')
    monkeypatch.setattr(queue, '_write_json_atomic', fail)
    with pytest.raises(OSError, match='index write failed'):
        simple.export_no_candidate_overlays(source, target, resume=True)
    assert before == {p: p.read_bytes() for p in target.rglob('*') if p.is_file()}
    assert not (target/'scene-0002').exists()


@pytest.mark.parametrize('result', ['empty', 'admissible', 'missing_raw', 'index_failure'])
def test_rebuild_empty_scene_updates_inspection_and_preserves_old_evidence(tmp_path, monkeypatch, result):
    from scripts import nuscenes_review_queue as queue, review_nuscenes_simple as simple
    from simulation import nuscenes_adapter as adapter, build_nuscenes_dataset as builder, review_overlay as overlay
    source, selections = setup_round(tmp_path)
    target = tmp_path/'empty'
    simple.export_no_candidate_overlays(source, target)
    before = {p: p.read_bytes() for p in source.rglob('*') if p.is_file()}
    old_hash = queue.file_sha256(source/'scene-0001/overlay_manifest.json')
    selections.write_text('scene_name,t0_sec\nscene-0001,5\n')
    monkeypatch.setattr(adapter, 'load_tables', lambda *args: {})
    monkeypatch.setattr(builder, '_reviewed_candidates', lambda *args: [{
        'scene_name': 'scene-0001', 'requested_t0_sec': 5.0,
        'resolved_t0_sec': 5.0, 'sample_token': 'new'}])
    def render(tables, token, dataroot, out_dir, **kwargs):
        if result == 'missing_raw':
            raise FileNotFoundError('raw frame missing')
        Path(out_dir).mkdir()
        for i in range(8):
            (Path(out_dir)/f'{i:03d}.jpg').write_bytes(b'new frame')
        allowed = ['cand_002'] if result == 'admissible' else []
        return {'sample_token': 'new', 'candidates': {}, 'stable_candidates': allowed,
                'role_state_eligible_stable_candidates': allowed, 'admissible_stable_candidates': allowed}
    monkeypatch.setattr(overlay, 'build_review_overlay', render)
    if result == 'index_failure':
        original_write = queue._write_json_atomic
        def fail_new_index(path, value):
            if path == source/'index.json' and any(r['sample_token'] == 'new' for r in value['valid']):
                raise OSError('index write failed')
            return original_write(path, value)
        monkeypatch.setattr(queue, '_write_json_atomic', fail_new_index)
    if result in ('missing_raw', 'index_failure'):
        with pytest.raises(OSError):
            queue.make_overlay_round(tmp_path, tmp_path, selections, source, resume=True, allow_empty_t0_changes=True)
        assert before == {p: p.read_bytes() for p in source.rglob('*') if p.is_file()}
        assert not (tmp_path/'overlays_history').exists()
        return
    summary = queue.make_overlay_round(tmp_path, tmp_path, selections, source, resume=True, allow_empty_t0_changes=True)
    assert summary['rebuilt'] == 1 and summary['created'] == 0
    assert not (tmp_path/'overlays_history').exists()
    simple.export_no_candidate_overlays(source, target, resume=True)
    assert not (tmp_path/'empty_history').exists()
    if result == 'empty':
        assert (target/'scene-0001/000.jpg').read_bytes() == b'new frame'
    else:
        assert not (target/'scene-0001').exists()
        assert json.loads((target/'index.json').read_text())['valid'] == []
        monkeypatch.setattr(simple, 'ACTOR_REVIEW_CSV', tmp_path/'actor_review.csv')
        simple._write_actor_review_csv({'scene-0001': ['cand_002']}, resume=True)
        assert 'scene-0001,cand_002,yes,""' in simple.ACTOR_REVIEW_CSV.read_text()
    simple.export_no_candidate_overlays(source, target, resume=True)
    summary = queue.make_overlay_round(tmp_path, tmp_path, selections, source, resume=True, allow_empty_t0_changes=True)
    assert summary['rebuilt'] == 0 and summary['skipped'] == 1


def test_step2_noop_needs_no_raw_or_external_tools(tmp_path, monkeypatch):
    from scripts import review_nuscenes_simple as simple
    from simulation import nuscenes_adapter as adapter
    source, selections = setup_round(tmp_path)
    target = tmp_path / 'empty'
    simple.export_no_candidate_overlays(source, target)
    for attr, value in {'OVERLAYS_DIR': source, 'SELECTIONS_CSV': selections,
                        'NO_CANDIDATES_DIR': target, 'DATAROOT': tmp_path/'deleted',
                        'VIDEOS_DIR': tmp_path/'no_videos'}.items():
        monkeypatch.setattr(simple, attr, value)
    monkeypatch.setattr(simple, '_require_external_tools', lambda: pytest.fail('no renderer needed'))
    monkeypatch.setattr(adapter, 'load_tables', lambda *args: pytest.fail('no raw needed'))
    simple.step2()


@pytest.mark.parametrize("change", ["t0", "remove", "missing_frame"])
def test_resume_conflicts_fail_before_render_or_writes(tmp_path, monkeypatch, change):
    from scripts import nuscenes_review_queue as queue
    from simulation import nuscenes_adapter as adapter
    source, selections = setup_round(tmp_path)
    if change == "t0":
        selections.write_text("scene_name,t0_sec\nscene-0001,5\n")
    elif change == "remove":
        selections.write_text("scene_name,t0_sec\nscene-0002,4\n")
    else:
        (source / "scene-0001/000.jpg").unlink()
    before = {p: p.read_bytes() for p in source.rglob('*') if p.is_file()}
    monkeypatch.setattr(adapter, "load_tables", lambda *args: pytest.fail("must fail before raw load"))
    with pytest.raises(ValueError):
        queue.make_overlay_round(tmp_path, tmp_path, selections, source, resume=True)
    assert before == {p: p.read_bytes() for p in source.rglob('*') if p.is_file()}


def test_resume_renders_only_new_scene_and_keeps_old_manifest(tmp_path, monkeypatch):
    from scripts import nuscenes_review_queue as queue
    from simulation import nuscenes_adapter as adapter, build_nuscenes_dataset as builder, review_overlay as overlay
    source, selections = setup_round(tmp_path)
    selections.write_text("scene_name,t0_sec\nscene-0001,4\nscene-0002,5\n")
    old = (source / "scene-0001/overlay_manifest.json").read_bytes()
    monkeypatch.setattr(adapter, "load_tables", lambda *args: {})
    def resolve(tables, csv_path, review_dir):
        rows = list(csv.DictReader(open(csv_path)))
        assert [r['scene_name'] for r in rows] == ['scene-0002']
        return [{"scene_name": "scene-0002", "requested_t0_sec": 5.0,
                 "resolved_t0_sec": 5.0, "sample_token": "new"}]
    monkeypatch.setattr(builder, "_reviewed_candidates", resolve)
    def render(tables, token, dataroot, out_dir, **kwargs):
        assert token == 'new'
        Path(out_dir).mkdir(parents=True, exist_ok=True)
        for i in range(8):
            (Path(out_dir) / f'{i:03d}.jpg').write_bytes(b'new image')
        return {"sample_token": token, "candidates": {}, "stable_candidates": [],
                "role_state_eligible_stable_candidates": [], "admissible_stable_candidates": []}
    monkeypatch.setattr(overlay, "build_review_overlay", render)
    summary = queue.make_overlay_round(tmp_path, tmp_path, selections, source, resume=True)
    assert summary['created'] == 1 and summary['skipped'] == 1
    assert (source / "scene-0001/overlay_manifest.json").read_bytes() == old
    assert len(json.loads((source/'index.json').read_text())['valid']) == 2


def test_csv_append_preserves_entered_bytes(tmp_path, monkeypatch):
    from scripts import review_nuscenes_simple as simple
    path = tmp_path / 'actor_review.csv'
    original = b'scene_name,candidate_label,interaction_within_3s,notes\r\nscene-0001,cand_007,yes,"Checked, keep this"\r\n'
    path.write_bytes(original)
    monkeypatch.setattr(simple, 'ACTOR_REVIEW_CSV', path)
    candidates = {'scene-0001': ['cand_007'], 'scene-0002': ['cand_012']}
    simple._write_actor_review_csv(candidates, resume=True)
    assert path.read_bytes() == original + b'scene-0002,cand_012,yes,""\n'
    simple._write_actor_review_csv(candidates, resume=True)
    assert path.read_bytes() == original + b'scene-0002,cand_012,yes,""\n'


def test_step1_handles_new_part_after_old_raw_removed(tmp_path, monkeypatch):
    import shutil
    from scripts import review_nuscenes_simple as simple, nuscenes_review_queue as queue
    from scripts import export_nuscenes_review_videos as exporter
    source = tmp_path / 'trainval'
    metadata = source / 'v1.0-trainval_meta'
    (metadata / 'v1.0-trainval').mkdir(parents=True)
    (metadata / 'maps').mkdir()
    (metadata / 'v1.0-trainval/scene.json').write_text(json.dumps([
        {'name': f'scene-{i:04d}'} for i in range(1, 851)]))
    for number in (1, 2):
        for tree in ('samples', 'sweeps'):
            camera = source / f'v1.0-trainval{number:02d}_blobs' / tree / 'CAM_FRONT'
            camera.mkdir(parents=True)
            (camera / f'part{number}.jpg').write_bytes(b'image')
    data = tmp_path / 'data'
    queue.prepare_trainval_dataroot(source, data)
    shutil.rmtree(source / 'v1.0-trainval01_blobs')
    assert (data / 'samples/CAM_FRONT/part1.jpg').is_symlink()
    assert not (data / 'samples/CAM_FRONT/part1.jpg').exists()
    results = tmp_path / 'results'
    videos = results / 'videos'
    videos.mkdir(parents=True)
    for suffix in ('mp4', 'json'):
        (videos / f'scene-0001_CAM_FRONT.{suffix}').write_bytes(b'old output')
    for attr, value in {'SOURCE_DIR': source, 'DATAROOT': data, 'RESULTS_DIR': results,
                        'VIDEOS_DIR': videos, 'SELECTIONS_CSV': results/'review_selections.csv'}.items():
        monkeypatch.setattr(simple, attr, value)
    monkeypatch.setattr(simple, '_require_external_tools', lambda: None)
    calls = []
    def export(args):
        assert args.scene == [f'scene-{i:04d}' for i in range(86, 171)]
        calls.append(args)
        for name in args.scene:
            for suffix in ('mp4', 'json'):
                (videos / f'{name}_CAM_FRONT.{suffix}').write_bytes(b'new output')
        return 0
    monkeypatch.setattr(exporter, 'export_review_videos', export)
    simple.step1()
    simple.step1()
    assert len(calls) == 1
    assert (videos/'scene-0001_CAM_FRONT.mp4').read_bytes() == b'old output'


def test_export_incrementally_preserves_old_copy(tmp_path):
    from scripts import review_nuscenes_simple as simple
    source, selections = setup_round(tmp_path)
    target = tmp_path / 'empty'
    simple.export_no_candidate_overlays(source, target, resume=True)
    old = {p: p.read_bytes() for p in (target/'scene-0001').iterdir()}
    import shutil
    shutil.copytree(source/'scene-0001', source/'scene-0002')
    index = json.loads((source/'index.json').read_text())
    index['valid'].append({**index['valid'][0], 'scene_name': 'scene-0002'})
    (source/'index.json').write_text(json.dumps(index))
    assert simple.export_no_candidate_overlays(source, target, resume=True) == 2
    assert simple.export_no_candidate_overlays(source, target, resume=True) == 2
    assert all(p.read_bytes() == content for p, content in old.items())


@pytest.mark.parametrize('already_copied', [False, True])
def test_diagnostic_export_adds_all_labels_once_and_preserves_source(tmp_path, monkeypatch, already_copied):
    import shutil
    from scripts import review_nuscenes_simple as simple
    from simulation import review_overlay as overlay
    source, selections = setup_round(tmp_path)
    path = source/'scene-0001/overlay_manifest.json'
    manifest = json.loads(path.read_text())
    manifest.update(rendered_candidate_labels=[], candidates={
        'cand_007': {'frames': [{'visible': True, 'u': 10, 'v': 20}] * 8}})
    path.write_text(json.dumps(manifest))
    before = {p: p.read_bytes() for p in source.rglob('*') if p.is_file()}
    target = tmp_path/'empty'
    if already_copied:
        shutil.copytree(source/'scene-0001', target/'scene-0001')
    calls = []
    def render(saved, original, destination):
        assert saved['admissible_stable_candidates'] == []
        assert list(saved['candidates']) == ['cand_007']
        calls.append(original)
        for i in range(8):
            (destination/f'{i:03d}.jpg').write_bytes(b'cand_007 label')
    monkeypatch.setattr(overlay, 'render_saved_candidate_labels', render)
    simple.export_no_candidate_overlays(source, target, resume=True)
    simple.export_no_candidate_overlays(source, target, resume=True)
    assert calls == [source/'scene-0001']
    assert (target/'scene-0001/000.jpg').read_bytes() == b'cand_007 label'
    assert json.loads((target/'scene-0001/review_render.json').read_text())['label_policy'] == 'all_candidates'
    assert before == {p: p.read_bytes() for p in source.rglob('*') if p.is_file()}
    (target/'scene-0001/000.jpg').write_bytes(b'modified')
    with pytest.raises(ValueError, match='không ghi đè'):
        simple.export_no_candidate_overlays(source, target, resume=True)
    assert len(calls) == 1


def test_failed_render_can_retry_without_touching_completed_scene(tmp_path, monkeypatch):
    from scripts import nuscenes_review_queue as queue
    from simulation import nuscenes_adapter as adapter, build_nuscenes_dataset as builder, review_overlay as overlay
    source, selections = setup_round(tmp_path)
    selections.write_text('scene_name,t0_sec\nscene-0001,4\nscene-0002,5\nscene-0003,6\n')
    monkeypatch.setattr(adapter, 'load_tables', lambda *args: {})
    def resolve(tables, csv_path, review_dir):
        with open(csv_path) as handle:
            return [{'scene_name': r['scene_name'], 'requested_t0_sec': float(r['t0_sec']),
                     'resolved_t0_sec': float(r['t0_sec']), 'sample_token': r['scene_name']}
                    for r in csv.DictReader(handle)]
    monkeypatch.setattr(builder, '_reviewed_candidates', resolve)
    calls = []
    fail = True
    def render(tables, token, dataroot, out_dir, **kwargs):
        calls.append(token)
        Path(out_dir).mkdir()
        if token == 'scene-0003' and fail:
            (Path(out_dir)/'000.jpg').write_bytes(b'partial')
            raise RuntimeError('render interrupted')
        for i in range(8):
            (Path(out_dir)/f'{i:03d}.jpg').write_bytes(b'image')
        return {'sample_token': token, 'candidates': {}, 'stable_candidates': [],
                'role_state_eligible_stable_candidates': [], 'admissible_stable_candidates': []}
    monkeypatch.setattr(overlay, 'build_review_overlay', render)
    with pytest.raises(RuntimeError, match='interrupted'):
        queue.make_overlay_round(tmp_path, tmp_path, selections, source, resume=True)
    assert not (source/'scene-0003').exists()
    assert len(json.loads((source/'index.json').read_text())['valid']) == 2
    fail = False
    queue.make_overlay_round(tmp_path, tmp_path, selections, source, resume=True)
    assert calls == ['scene-0002', 'scene-0003', 'scene-0003']
