"""Filesystem-only model inventory: no downloads or model loading."""
import json

import pytest
from pathlib import Path


def snapshot_files(path):
    path.mkdir(parents=True)
    (path/'config.json').write_text('{}')
    (path/'model.safetensors.index.json').write_text(json.dumps({'weight_map':{'a':'part1.safetensors','b':'part2.safetensors'}}))
    (path/'part1.safetensors').write_bytes(b'a')
    (path/'part2.safetensors').write_bytes(b'b')


def test_inventory_finds_exact_revision_in_flat_cache_without_calling_it_active(tmp_path):
    from scripts.inspect_model_storage import inspect_storage
    revision='a'*40
    cfg={'hf_home':str(tmp_path/'hf_cache'),'hf_hub_cache':str(tmp_path/'hf_cache/hub')}
    path=tmp_path/'hf_cache/models--Org--Model/snapshots'/revision
    snapshot_files(path)
    out=inspect_storage(cfg,[{'key':'model','repo_id':'Org/Model','revision':revision}],tmp_path/'hf_models')
    item=out['checkpoints'][0]
    assert not item['active_cache_snapshot']['exists']
    found=next(r for r in item['other_cache_snapshots'] if r['exists'])
    assert found['weights']['status']=='files_present'
    assert out['runtime_loading_tested'] is False


def test_inventory_detects_missing_shard_and_dangling_symlink(tmp_path):
    from scripts.inspect_model_storage import inspect_snapshot
    snapshot_files(tmp_path/'snapshot')
    (tmp_path/'snapshot/part2.safetensors').unlink()
    (tmp_path/'snapshot/part2.safetensors').symlink_to(tmp_path/'missing-blob')
    out=inspect_snapshot(tmp_path/'snapshot')
    assert out['weights']['status']=='missing_or_empty_files'
    assert out['weights']['missing']==['part2.safetensors']


def test_local_dir_revision_is_only_metadata_evidence(tmp_path):
    from scripts.inspect_model_storage import inspect_storage
    local=tmp_path/'hf_models/downloaded';snapshot_files(local)
    metadata=local/'.cache/huggingface/download/config.json.metadata'
    metadata.parent.mkdir(parents=True);metadata.write_text('b'*40+'\netag\n1.0\n')
    cfg={'hf_home':str(tmp_path/'cache'),'hf_hub_cache':str(tmp_path/'cache/hub')}
    out=inspect_storage(cfg,[],tmp_path/'hf_models')
    row=out['local_directories'][0]
    assert row['download_metadata_revisions']==['b'*40]
    assert row['identity_verified'] is False
    assert row['weights']['status']=='files_present'


@pytest.mark.skip(reason="needs the cluster campaign configuration, which is not part of this release")
def test_cache_configuration_uses_explicit_hub_path(tmp_path,monkeypatch):
    import yaml
    from analysis.evaluation_campaign import load_config,configure_model_cache
    cfg=yaml.safe_load(Path('evaluation/config.example.yaml').read_text())
    cfg['hf_home']=str(tmp_path/'home');cfg['hf_hub_cache']=str(tmp_path/'flat')
    p=tmp_path/'cfg.yaml';p.write_text(yaml.safe_dump(cfg))
    loaded=load_config(p)
    monkeypatch.setenv('HF_HUB_CACHE','/wrong')
    monkeypatch.setenv('HUGGINGFACE_HUB_CACHE','/wrong')
    monkeypatch.setenv('TRANSFORMERS_CACHE','/wrong')
    monkeypatch.setenv('HF_HOME','/wrong')
    configure_model_cache(loaded)
    import os
    assert os.environ['HF_HUB_CACHE']==str(tmp_path/'flat')
    assert os.environ['HUGGINGFACE_HUB_CACHE']==str(tmp_path/'flat')
    assert 'TRANSFORMERS_CACHE' not in os.environ
    cfg.pop('hf_hub_cache');p.write_text(yaml.safe_dump(cfg))
    assert load_config(p)['hf_hub_cache']==str(tmp_path/'home/hub')


def test_inspection_finds_symlinked_local_model_and_leaves_storage_unchanged(tmp_path):
    from scripts.inspect_model_storage import inspect_storage
    target=tmp_path/'elsewhere/snapshot';snapshot_files(target)
    models=tmp_path/'hf_models';models.mkdir()
    (models/'my-model').symlink_to(target,target_is_directory=True)
    before={str(p.relative_to(tmp_path)) for p in tmp_path.rglob('*')}
    cfg={'hf_home':str(tmp_path/'cache'),'hf_hub_cache':str(tmp_path/'cache/hub')}
    out=inspect_storage(cfg,[],models)
    assert len(out['local_directories'])==1
    assert out['local_directories'][0]['weights']['status']=='files_present'
    assert before=={str(p.relative_to(tmp_path)) for p in tmp_path.rglob('*')}
