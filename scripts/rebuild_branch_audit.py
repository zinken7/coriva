#!/usr/bin/env python3
"""Rebuild an unstarted branch audit, without inference or dataset mutation.

Campaign mode verifies frozen data/settings, permits only the three review
code paths listed below to change, and records the previous freeze. Portable
mode renews a public-only export; checks needing absent GT context stay unknown.
"""
from __future__ import annotations

import argparse
import copy
from datetime import datetime, timezone
from html import unescape
import json
from pathlib import Path
import re
import shutil
import sys
import tempfile

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from analysis.branch_audit import AUDIT_VERSION, AUDIT_INSTRUCTIONS, audit_rows, check_branch_contract
from analysis.evaluation_campaign import (ROOT, code_inventory, load_config, locked,
                                          read_json, sha, write_json)
from analysis.evaluation_human import csv_write, package_index, prepare_branch_audit, read_rows

REVIEW_CODE_PATHS = frozenset({
    'analysis/branch_audit.py', 'analysis/evaluation_human.py',
    'scripts/rebuild_branch_audit.py',
})


def review_code_changes(frozen):
    current = dict(code_inventory())
    previous = dict(frozen['code_files'])
    changed = {p for p in previous.keys() | current.keys() if previous.get(p) != current.get(p)}
    unexpected = changed - REVIEW_CODE_PATHS
    if unexpected:
        raise ValueError('Non-audit runtime code changed; use a new campaign and smoke: '
                         + ', '.join(sorted(unexpected)))
    if any(p not in current for p in previous):
        raise ValueError('Frozen audit code is missing; restore it before rebuilding')
    return sorted(changed), current


def verify_campaign_data(cfg, frozen):
    for key in ('splits', 'smoke', 'statistics', 'probe', 'human', 'validation'):
        if frozen['settings'][key] != cfg[key]:
            raise ValueError(f'Frozen {key} changed; use a new campaign')
    for split, rows in frozen['datasets'].items():
        base = Path(cfg['splits'][split]['path'])
        if {p.name for p in base.glob('scenario_*') if p.is_dir()} != {r['name'] for r in rows}:
            raise ValueError(f'Frozen {split} membership changed')
        for row in rows:
            for f in row['files']:
                if sha(base/row['name']/f['path']) != f['sha256']:
                    raise ValueError(f"Frozen data changed: {split}/{row['name']}/{f['path']}")
    from analysis.smoke_admission import scenario_input_fingerprint
    fingerprint, _ = scenario_input_fingerprint(str(Path(cfg['smoke']['path'])/cfg['smoke']['scenario']))
    if fingerprint != frozen['smoke_fingerprint']:
        raise ValueError('Frozen smoke data changed')


def replace_directory(staged, destination):
    """Keep the old tree until its replacement is in place; restore on failure."""
    if destination.is_symlink():
        raise ValueError('Refusing to reset a symlinked audit directory')
    backup = staged.parent/'previous-audit'
    if destination.exists():
        destination.rename(backup)
    try:
        staged.rename(destination)
    except BaseException:
        if backup.exists():
            backup.rename(destination)
        raise
    if backup.exists():
        shutil.rmtree(backup)


def rebuild_campaign(cfg, reset=False):
    campaign = Path(cfg['campaign_dir'])
    destination = campaign/'human/branch_audit'
    if destination.exists() and not reset:
        raise ValueError('Audit already exists; --reset replaces this audit only')
    if destination.is_symlink():
        raise ValueError('Refusing to reset a symlinked audit directory')
    # Same lock as campaign submissions; never alter scoring/inference outputs.
    with locked(campaign/'.submit.lock'):
        frozen = read_json(campaign/'frozen.json')
        changed, current_code = review_code_changes(frozen)
        verify_campaign_data(cfg, frozen)
        destination.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(prefix='.branch-audit-', dir=destination.parent) as tmp:
            stage_cfg = copy.deepcopy(cfg); stage_cfg['campaign_dir'] = tmp
            prepare_branch_audit(stage_cfg, frozen['datasets'])
            staged = Path(tmp)/'human/branch_audit'
            if (destination/'private_key.json').exists():
                if read_json(destination/'private_key.json') != read_json(staged/'private_key.json'):
                    raise ValueError('Rebuilt audit changed scene IDs; original package retained')
            revision = campaign/'audit_revisions'/datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
            write_json(revision/'frozen_before.json', frozen)
            write_json(revision/'revision.json', dict(
                rubric_version=AUDIT_VERSION, reason='Replace out-of-scope action questions with scene/reference review',
                changed_review_code=changed, reset_requested=reset,
                package=read_json(staged/'package.json'), data_hashes_verified=True,
                prior_package=read_json(destination/'package.json') if (destination/'package.json').exists() else None))
            replace_directory(staged, destination)
            # Explicit, narrow review-only revision; frozen runtime drift checks stay enabled.
            revised = copy.deepcopy(frozen); revised['code_files'] = list(current_code.items())
            write_json(campaign/'frozen.json', revised)
            status_path = campaign/'human/status.json'
            if status_path.exists():
                status = read_json(status_path)
                status['branch_audit'] = dict(status='awaiting_author_review', rubric_version=AUDIT_VERSION)
                write_json(status_path, status)
    return read_json(destination/'package.json'), destination/'public'


def rebuild_public(public, reset=False):
    """Preserve the existing public scene IDs and images, but clear all ratings."""
    public = Path(public).resolve()
    if not reset:
        raise ValueError('Public refresh requires --reset; it replaces the existing questions and ratings')
    source_rows = read_rows(public/'ratings.csv')
    ids = list(dict.fromkeys(r['scenario_id'] for r in source_rows))
    if any(not re.fullmatch(r'audit_\d+', sid) for sid in ids):
        raise ValueError('Unexpected public scene ID')
    splits = {sid: {r['split'] for r in source_rows if r['scenario_id'] == sid} for sid in ids}
    if any(len(v) != 1 for v in splits.values()):
        raise ValueError('Ambiguous split per scene')
    with tempfile.TemporaryDirectory(prefix='.audit-refresh-', dir=public.parent) as tmp:
        staged = Path(tmp)/'public'; staged.mkdir()
        rows=[]; checks={}
        for sid in ids:
            split = next(iter(splits[sid]))
            frames = list((public/sid).glob('frame*.jpg')) + list((public/sid).glob('frame*.png'))
            if len(frames) != 8:
                raise ValueError(f'{sid}: expected eight source frames; old package retained')
            original = (public/sid/'index.html').read_text()
            sections = {name: json.loads(unescape(body)) for name, body in
                        re.findall(r'<h2>(.*?)</h2><pre>(.*?)</pre>', original, re.S)}
            gt = dict(scenario_id=sid, observed_state=sections['Observed state'],
                      branches=sections['Included branches'],
                      scenario_type=sections.get('Scene context', {}).get('scenario_type'))
            checks[sid] = check_branch_contract(gt, sections['Declared query'], split)
            rows.extend(audit_rows(gt, split))
            shutil.copytree(public/sid, staged/sid)
            from html import escape
            # Replace any previous check block, retaining original source images/content.
            body = re.sub(r'<h2>Automatic protocol checks</h2><pre>.*?</pre>', '', original, flags=re.S)
            body += '<h2>Automatic protocol checks</h2><pre>'+escape(json.dumps(checks[sid],indent=2))+'</pre>'
            (staged/sid/'index.html').write_text(body)
        csv_write(staged/'ratings.csv', ['scenario_id','split','item_type','item','prompt','rater1','notes'], rows)
        package_index(staged, ids, AUDIT_INSTRUCTIONS + ' Public-only refresh: absent scenario_type is not inferred; regenerate on Kaya for the full protocol check.')
        write_json(staged/'automatic_checks.json', dict(rubric_version=AUDIT_VERSION,scenes=checks,
            counts={s:sum(v['status']==s for v in checks.values()) for s in ('pass','fail','not_checked')}))
        package = dict(rubric_version=AUDIT_VERSION,scenarios=len(ids),items=len(rows),
                       scene_items=len(ids),branch_items=len(rows)-len(ids),
                       source='public_only_refresh',ratings_sha256=sha(staged/'ratings.csv'))
        write_json(staged/'package.json',package)
        replace_directory(staged, public)
    return package, public


def main():
    parser=argparse.ArgumentParser(description=__doc__.splitlines()[0])
    group=parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--config', help='Campaign config on Kaya')
    group.add_argument('--public-dir', type=Path, help='Local public-only package; no dataset required')
    parser.add_argument('--reset', action='store_true', help='Replace this audit and discard its ratings')
    args=parser.parse_args()
    try:
        result,path = rebuild_public(args.public_dir,args.reset) if args.public_dir else rebuild_campaign(load_config(args.config),args.reset)
    except (ValueError,FileNotFoundError,KeyError) as exc:
        parser.exit(1,f'ERROR: {exc}\n')
    print(f"rubric={AUDIT_VERSION} scenes={result['scenarios']} scene_items={result['scene_items']} branch_items={result['branch_items']} total={result['items']}")
    print(f'Open: {path}/index.html')
    print(f'Check: {path}/automatic_checks.json')
    print('No dataset build, inference or API calls were run.')


if __name__=='__main__':
    main()
