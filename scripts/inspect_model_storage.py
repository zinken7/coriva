"""Read-only inspection of pinned snapshots and separate model directories.

Uses filesystem metadata and small JSON/download-metadata files only. Does not
import model libraries, access the network, hash weights or change storage.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import re
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def weight_inventory(path):
    for name in ('model.safetensors.index.json', 'pytorch_model.bin.index.json'):
        index = path/name
        if index.is_file():
            try:
                mapping = json.loads(index.read_text())['weight_map']
                if not isinstance(mapping, dict) or not mapping:
                    raise ValueError('empty or invalid weight_map')
                files = sorted(set(mapping.values()))
                if any(not isinstance(f, str) or Path(f).is_absolute() or '..' in Path(f).parts for f in files):
                    raise ValueError('invalid shard path')
                missing = [f for f in files if not (path/f).is_file() or (path/f).stat().st_size == 0]
                return dict(status='missing_or_empty_files' if missing else 'files_present',
                            index=name,expected_shards=len(files),missing=missing)
            except (OSError,ValueError,KeyError,TypeError) as exc:
                return dict(status='invalid_index',index=name,error=str(exc))
    for name in ('model.safetensors', 'pytorch_model.bin'):
        if (path/name).is_file() and (path/name).stat().st_size > 0:
            return dict(status='files_present',file=name)
    return dict(status='no_standard_weight_file_or_index')


def inspect_snapshot(path):
    path = Path(path)
    return dict(path=str(path),exists=path.is_dir(),config_present=(path/'config.json').is_file(),
                weights=weight_inventory(path))


def local_model_directories(root, max_depth=3):
    """Bounded scan; never descend into blobs or download caches."""
    root = Path(root)
    if not root.is_dir(): return []
    found=[]
    for current, dirs, files in os.walk(root):
        p=Path(current)
        if 'config.json' in files:
            found.append(p);dirs[:]=[];continue
        if len(p.relative_to(root).parts) >= max_depth:
            dirs[:]=[]
        else:
            dirs[:]=sorted(d for d in dirs if not d.startswith('.') and d not in ('blobs','node_modules'))
            # A model directory may itself point to a cache snapshot. Inspect
            # that leaf, without recursively following symlink trees/cycles.
            for name in dirs:
                child=p/name
                if child.is_symlink() and (child/'config.json').is_file():
                    found.append(child)
    return sorted(found)


def inspect_storage(cfg, snapshots, models_dir):
    home=Path(cfg['hf_home'])
    active=Path(cfg.get('hf_hub_cache') or home/'hub')
    # A previous prefetch --cache-dir "$HF_HOME" can create a flat cache.
    roots=[active,home,home/'hub',Path(models_dir),Path(models_dir)/'hub']
    inherited={name:os.environ.get(name) for name in (
        'HF_HOME','HF_HUB_CACHE','HUGGINGFACE_HUB_CACHE','TRANSFORMERS_CACHE')}
    roots.extend(Path(v).expanduser() for k,v in inherited.items() if v and k!='HF_HOME')
    roots=list(dict.fromkeys(roots))
    checkpoints=[]
    for item in snapshots:
        relative=Path('models--'+item['repo_id'].replace('/','--'))/'snapshots'/item['revision']
        checkpoints.append(dict(item,active_cache_snapshot=inspect_snapshot(active/relative),
                           other_cache_snapshots=[inspect_snapshot(r/relative) for r in roots if r!=active]))
    local=[]
    for path in local_model_directories(models_dir):
        revisions=set()
        metadata=path/'.cache/huggingface/download'
        if metadata.is_dir():
            for record in metadata.rglob('*.metadata'):
                with record.open() as stream: first=stream.readline(200).strip()
                if re.fullmatch(r'[0-9a-f]{40}',first): revisions.add(first)
        local.append(dict(inspect_snapshot(path),download_metadata_revisions=sorted(revisions),
                          identity_verified=False))
    datasets={k:dict(path=v['path'],exists=Path(v['path']).is_dir(),expected=v['expected'])
              for k,v in cfg.get('splits',{}).items()}
    data_roots={Path(v['path']).parent for v in cfg.get('splits',{}).values()}
    return dict(configured_hf_home=str(home),configured_hub_cache=str(active),
                inherited_cache_environment=inherited,checkpoints=checkpoints,local_directories=local,
                models_dir=str(models_dir),local_scan_max_depth=3,datasets=datasets,
                dataset_parent_directories={str(p):sorted(d.name for d in p.iterdir() if d.is_dir())
                                            for p in sorted(data_roots) if p.is_dir()},
                runtime_loading_tested=False,weight_contents_verified=False,
                note='File presence is not a model-load test. Local download metadata is provenance evidence only; it does not prove current file identity. No files were changed.')


def main():
    from analysis.evaluation_campaign import load_config
    from scripts.prefetch_weights import panel_snapshots
    parser=argparse.ArgumentParser(description=__doc__.split('\n')[0])
    parser.add_argument('--config',required=True,help='evaluation config.yaml or config.example.yaml')
    parser.add_argument('--models-dir',default='hf_models')
    parser.add_argument('--panel',default=str(ROOT/'analysis/model_panel.yaml'))
    parser.add_argument('--json',action='store_true',help='full machine-readable report instead of the concise summary')
    args=parser.parse_args()
    result=inspect_storage(load_config(args.config),panel_snapshots(args.panel),Path(args.models_dir).expanduser())
    if args.json:
        print(json.dumps(result,indent=2))
    else:
        print(f"Configured HF_HOME: {result['configured_hf_home']}")
        print(f"Configured HF_HUB_CACHE: {result['configured_hub_cache']}")
        print('Inherited cache variables: '+json.dumps(result['inherited_cache_environment']))
        for item in result['checkpoints']:
            active=item['active_cache_snapshot']
            print(f"\n{item['repo_id']} @ {item['revision']}")
            print(f"  ACTIVE: {active['weights']['status'] if active['exists'] else 'snapshot_absent'}; config={active['config_present']}")
            print(f"  Path: {active['path']}")
            for other in item['other_cache_snapshots']:
                if other['exists']:
                    print(f"  OTHER CACHE: {other['path']} ({other['weights']['status']})")
            if active['weights'].get('missing'):
                print('  Missing/empty shards: '+', '.join(active['weights']['missing']))
        print(f"\nSeparate model directories under {result['models_dir']} (depth <=3):")
        for item in result['local_directories']:
            print(f"  {item['path']}: {item['weights']['status']}")
            print('    Download metadata revisions: '+(', '.join(item['download_metadata_revisions']) or 'not recorded'))
        if not result['local_directories']: print('  No config.json found within the scan depth.')
        print('\nConfigured datasets:')
        for split,item in result['datasets'].items():
            print(f"  {split}: {item['path']} (directory_exists={item['exists']})")
        for parent,names in result['dataset_parent_directories'].items():
            print(f"  Directories in {parent}: "+', '.join(names))
        print('\n'+result['note'])
    return 0


if __name__=='__main__':
    raise SystemExit(main())
