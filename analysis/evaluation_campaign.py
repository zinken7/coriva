"""Kaya evaluation campaign: frozen inputs, resumable jobs and complete reports.

Entry points live in evaluation/01_prepare.sh through 04_report.sh. No model,
API or scheduler is touched when importing this module or using --dry-run.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import copy
import fcntl
import hashlib
import json
import os
import re
from pathlib import Path
import shlex
import subprocess
import sys
import time
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[1]
PANEL = ROOT / 'analysis/model_panel.yaml'
BASELINES = ('oracle', 'copy_factual', 'kinematic_extrapolation', 'action_profile_kinematic', 'action_profile_published', 'action_profile_typical', 'random_valid')
CONDITIONS = ('video', 'history_json', 'video_history')


def read_json(path):
    return json.loads(Path(path).read_text())


def write_json(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + f'.{os.getpid()}.tmp')
    temp.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + '\n')
    temp.replace(path)


def sha(path):
    h = hashlib.sha256()
    with open(path, 'rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


@contextmanager
def locked(path):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('a') as stream:
        try:
            fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError(f'Another job is writing {path.parent}; wait for it to finish') from exc
        try:
            yield
        finally:
            fcntl.flock(stream, fcntl.LOCK_UN)


def load_config(path):
    path = Path(path).resolve()
    cfg = yaml.safe_load(path.read_text())
    if not isinstance(cfg, dict) or set(cfg.get('splits', {})) != {'carla', 'nuscenes'}:
        raise ValueError('config requires carla and nuscenes splits')
    def resolve(p):
        p = Path(os.path.expandvars(os.path.expanduser(str(p))))
        return str((path.parent / p).resolve())
    for key in ('campaign_dir', 'python', 'hf_home'):
        cfg[key] = resolve(cfg[key])
    cfg['hf_hub_cache'] = resolve(cfg.get('hf_hub_cache') or str(Path(cfg['hf_home'])/'hub'))
    for split, count in (('carla', 400), ('nuscenes', 200)):
        cfg['splits'][split]['path'] = resolve(cfg['splits'][split]['path'])
        if cfg['splits'][split]['expected'] != count:
            raise ValueError(f'final campaign requires {count} {split} scenarios')
    cfg['smoke']['path'] = resolve(cfg['smoke']['path'])
    if Path(cfg['smoke']['scenario']).name != cfg['smoke']['scenario']:
        raise ValueError('smoke.scenario must be a directory basename')
    cfg['admission'] = resolve(cfg['admission']) if cfg.get('admission') else str(
        Path(cfg['campaign_dir'])/'smoke/admission.json')
    if cfg['statistics']['n_boot'] < 10000:
        raise ValueError('final reporting requires at least 10000 bootstrap resamples')
    if not 1 <= cfg['probe']['limit'] <= 200:
        raise ValueError('probe.limit must be in 1..200')
    human = cfg['human']
    mode = human.get('mode', 'external_raters')  # Preserve existing campaign configs.
    if mode == 'author_review':
        if (human['raters'], human['items'], human['attention'], human['calibration']) != (1, 120, 0, 0):
            raise ValueError('author review requires one author, 120 items, no attention or calibration subsets')
    elif mode == 'external_raters':
        if (human['items'], human['attention'], human['calibration']) != (120, 12, 30) or human['raters'] < 3:
            raise ValueError('external study requires >=3 raters, 120 items, 12 checks, 30 calibration items')
    else:
        raise ValueError('human.mode must be author_review or external_raters')
    if human['audit_per_split'] < 25:
        raise ValueError('branch audit requires >=25 scenarios per split')
    paths = [Path(s['path']) for s in cfg['splits'].values()]
    if paths[0] == paths[1] or any(Path(cfg['campaign_dir']).is_relative_to(p) for p in paths):
        raise ValueError('datasets and campaign outputs must have separate roots')
    cfg['_path'] = str(path)
    return cfg


def matrix_cells(cfg):
    from analysis.model_panel import load_model_panel, resolve_configurations
    panel = load_model_panel(str(PANEL))
    models = resolve_configurations(panel, [m['name'] for m in panel['configurations']])
    return [dict(split=split, configuration=m['name'], backend=m['backend'], condition=c,
                 out_dir=str(Path(cfg['campaign_dir'])/'matrix'/split/f"{m['name']}_{c}_seed0"))
            for split in sorted(cfg['splits']) for m in models for c in CONDITIONS]


def code_inventory():
    files = []
    for directory in ('analysis', 'ctms', 'simulation', 'scripts', 'slurm', 'evaluation'):
        for p in sorted((ROOT/directory).rglob('*')):
            if p.is_file() and p.suffix in {'.py', '.yaml', '.sh', '.sbatch', '.json'}:
                if directory == 'evaluation' and p.name != 'config.example.yaml' and p.suffix == '.yaml':
                    continue
                files.append((str(p.relative_to(ROOT)), sha(p)))
    files.append(('run_benchmark.py', sha(ROOT/'run_benchmark.py')))
    files.extend((p.name, sha(p)) for p in sorted(ROOT.glob('requirements*.txt')))
    return files


def freeze(path, value):
    path = Path(path)
    value = json.loads(json.dumps(value))
    if path.exists():
        if read_json(path) != value:
            raise ValueError('Frozen campaign inputs/config/code changed; use a new campaign directory')
    else:
        write_json(path, value)


def expected_reference_generator(split):
    """What the current nuScenes builder records about itself; a dataset that
    records anything else was built by another generator (a different template
    policy or twin semantics under the same protocol version) and is not
    admitted to this campaign. CARLA references come from the simulator."""
    if split != 'nuscenes':
        return None
    from simulation.nuscenes_adapter import SCENARIO_TEMPLATE_POLICY
    from simulation.kinematic_twin import TWIN_VERSION
    return dict(scenario_template_policy=SCENARIO_TEMPLATE_POLICY, twin_version=TWIN_VERSION)


def _check_reference_generator(d, gt, manifest, expected):
    """Fail closed unless GT protocol, GT candidate table and the released
    manifest all name the expected generator; return the recorded provenance."""
    proto = gt.get('protocol') or {}; template = gt.get('scenario_template') or {}
    for key, want in expected.items():
        found = {'input_manifest': manifest.get(key), 'ground_truth.protocol': proto.get(key)}
        if key == 'scenario_template_policy':
            found['ground_truth.scenario_template'] = template.get('policy')
        wrong = {k: v for k, v in found.items() if v != want}
        if wrong:
            raise ValueError(f'{d}: {key} must be {want!r} in every artifact, found {wrong}; '
                             'a build from another reference generator is not admitted: archive it and rebuild')
    # The build commit is verified, not copied: recorded as a full SHA in both
    # artifacts, identical, and taken from a clean worktree, so the reference
    # can be reproduced from that commit (two missing commits do not agree).
    commits = {'input_manifest': manifest.get('build_code_commit'),
               'ground_truth.protocol': proto.get('build_code_commit')}
    if len(set(commits.values())) != 1:
        raise ValueError(f'{d}: build_code_commit differs between GT and input manifest: {commits}')
    commit = manifest.get('build_code_commit')
    if not (isinstance(commit, str) and re.fullmatch(r'[0-9a-f]{40}', commit)):
        raise ValueError(f'{d}: build_code_commit must be a full git SHA in GT and input manifest, '
                         f'found {commit!r}; a build without a recorded commit cannot be reproduced')
    clean = {'input_manifest': manifest.get('build_worktree_clean'),
             'ground_truth.protocol': proto.get('build_worktree_clean')}
    if set(clean.values()) != {True}:
        raise ValueError(f'{d}: build_worktree_clean must be true in GT and input manifest, found {clean}; '
                         'rebuild from a clean checkout of the recorded commit')
    return dict(expected, build_code_commit=commit, build_worktree_clean=True)


def l4_query_mismatches(gt, query, manifest=None):
    """Every way the released L4 query can disagree with the reference it is
    scored against. The scorer pairs an answer
    with the reference by the full intervention object, so a query.json that
    names another intervention, target or trigger makes every honest answer
    unmatched (CAS 0). Checked before inference by `dataset_inventory` and
    again by `run_model`. Compares the ordered tuple
    (intervention_type, target_actor_id, trigger_depth) of every query, the
    fixed ego policy, and the input manifest's critical actor when present."""
    problems = []
    expected = [((q.get('counterfactual_chain') or {}).get('intervention') or {})
                for q in (gt.get('l4_queries') or [])]
    expected = [dict(intervention_type=i.get('intervention_type'), target_actor_id=i.get('target_actor_id'),
                     trigger_depth=i.get('trigger_depth')) for i in expected]
    released = [dict(intervention_type=(i or {}).get('intervention_type'), target_actor_id=(i or {}).get('target_actor_id'),
                     trigger_depth=(i or {}).get('trigger_depth')) for i in (query.get('l4_interventions') or [])]
    if not expected:
        problems.append('GT declares no l4_queries')
    elif released != expected:
        problems.append(f'L4 query differs from GT reference: released={released} expected={expected}')
    gt_policy = (gt.get('protocol') or {}).get('l4_ego_policy', 'do_ego_no_action')
    if query.get('l4_ego_policy', 'do_ego_no_action') != gt_policy:
        problems.append(f"L4 ego policy differs from GT protocol: {query.get('l4_ego_policy')!r} != {gt_policy!r}")
    critical = (manifest or {}).get('critical_actor_id')
    targets = {i['target_actor_id'] for i in expected}
    if critical is not None and targets and critical not in targets:
        problems.append(f'input manifest critical_actor_id {critical!r} is not an L4 target {sorted(targets)}')
    return problems


def dataset_inventory(path, expected, split):
    from simulation.protocol import validate_protocol_manifest
    from simulation.observed_history import load_observed_artifacts
    from analysis.smoke_admission import scenario_input_fingerprint
    expected_generator = expected_reference_generator(split)
    dirs = sorted(p for p in Path(path).glob('scenario_*') if p.is_dir())
    if len(dirs) != expected:
        raise ValueError(f'{split}: expected {expected} scenarios, found {len(dirs)}')
    rows = []; ids = set(); sources = set()
    for d in dirs:
        gt = read_json(d/'ground_truth.json'); manifest = read_json(d/'input_manifest.json')
        validate_protocol_manifest(manifest)
        validate_protocol_manifest(gt.get('protocol', {}))
        if gt['protocol'].get('split') != manifest.get('split'):
            raise ValueError(f'{d}: GT and input manifest split differ')
        if manifest.get('future_steps') != 6 or manifest.get('future_horizon_seconds') != 3.0:
            raise ValueError(f'{d}: primary evaluation requires six states through 3 seconds')
        if manifest.get('track') != 'primary':
            raise ValueError(f'{d}: not the primary track')
        source_split = manifest.get('split') or manifest.get('dataset')
        if source_split not in ({'carla'} if split == 'carla' else {'nuscenes', 'v1.0-trainval'}):
            raise ValueError(f'{d}: expected split {split}, found {source_split}')
        if set(manifest.get('available_conditions', [])) != set(CONDITIONS):
            raise ValueError(f'{d}: all three input conditions must be released')
        generator = (_check_reference_generator(d, gt, manifest, expected_generator)
                     if expected_generator else None)
        root, history = load_observed_artifacts(str(d))
        if root != gt['observed_state']:
            raise ValueError(f'{d}: standalone root differs from GT')
        query = read_json(d/'query.json')
        actions = [(b.get('intervention') or {}).get('intervention_type') for b in gt['branches']]
        if query.get('l3_ego_actions') != actions:
            raise ValueError(f'{d}: L3 query differs from GT metadata')
        if query.get('factual_action') is not None:
            from simulation.observed_history import factual_branch_contract
            if {k: query.get(k) for k in ('factual_action', 'factual_reference')} != factual_branch_contract(gt):
                raise ValueError(f'{d}: factual-branch contract in query differs from GT')
        if not query.get('l4_interventions'):
            raise ValueError(f'{d}: missing L4 query')
        from simulation.l4_control import validate_pair_reference
        try:
            validate_pair_reference(gt, manifest)
        except ValueError as exc:
            raise ValueError(f'{d}: {exc}') from exc
        for problem in l4_query_mismatches(gt, query, manifest):
            raise ValueError(f'{d}: {problem}')
        from simulation.build_pilot import public_intervention_profiles
        public_intervention_profiles(root, query,
                                     read_json(d/'scenario_config.json') if (d/'scenario_config.json').exists() else {},
                                     split)
        sid = gt['scenario_id']
        if sid in ids:
            raise ValueError(f'{split}: duplicate scenario ID {sid}')
        ids.add(sid)
        source = gt.get('source_scene_name') or gt.get('source_scene_token')
        if split == 'nuscenes' and not source:
            raise ValueError(f'{d}: missing source scene identity for statistical independence')
        if source and source in sources:
            raise ValueError(f'{split}: multiple roots from source scene {source}; clustered analysis required')
        if source:
            sources.add(source)
        if generator and rows and rows[0]['generator'] != generator:
            raise ValueError(f'{split}: a dataset must come from a single build; {rows[0]["name"]} records '
                             f'{rows[0]["generator"]} but {d.name} records {generator}')
        fingerprint, files = scenario_input_fingerprint(str(d))
        rows.append(dict(name=d.name, scenario_id=sid, source_scene=source,
                         source_sample=gt.get('source_sample_token'), fingerprint=fingerprint,
                         files=files, fidelity=gt.get('factual_fidelity_error'), generator=generator,
                         l3_branches=len(gt['branches']), l4_queries=len(query['l4_interventions'])))
    return rows


def prepare(cfg):
    from analysis.smoke_admission import repository_state, scenario_input_fingerprint
    from analysis.evaluation_human import prepare_branch_audit
    root = Path(cfg['campaign_dir'])
    inventories = {split: dataset_inventory(spec['path'], spec['expected'], split)
                   for split, spec in cfg['splits'].items()}
    smoke_dir = Path(cfg['smoke']['path'])/cfg['smoke']['scenario']
    smoke_hash, _ = scenario_input_fingerprint(str(smoke_dir))
    smoke_gt = read_json(smoke_dir/'ground_truth.json')
    from simulation.l4_control import validate_pair_reference
    smoke_manifest = read_json(smoke_dir/'input_manifest.json')
    validate_pair_reference(smoke_gt, smoke_manifest)
    problems = l4_query_mismatches(smoke_gt, read_json(smoke_dir/'query.json'), smoke_manifest)
    if problems:
        raise ValueError(f'Smoke L4 query is incompatible: {problems}')
    scientific = {k: cfg[k] for k in ('splits', 'smoke', 'statistics', 'probe', 'human', 'validation')}
    frozen = dict(version='coriva-evaluation-campaign-2', settings=scientific,
                  code_files=code_inventory(), datasets=inventories, smoke_fingerprint=smoke_hash)
    freeze(root/'frozen.json', frozen)
    write_json(root/'provenance.json', dict(repository=repository_state(str(ROOT)),
               executable=sys.executable, python=sys.version, config=cfg['_path']))
    reports = {}
    for split, spec in cfg['splits'].items():
        reports[split] = {p.name: {'path': str(p), 'sha256': sha(p)}
                          for p in sorted(Path(spec['path']).glob('*'))
                          if p.is_file() and p.suffix in {'.json', '.jsonl', '.csv'}}
    from collections import Counter
    import statistics
    import numpy as np
    accepted = {}
    for split, rows in inventories.items():
        fidelity = [r['fidelity'] for r in rows if isinstance(r['fidelity'], (int, float))]
        classes = Counter(); interventions = Counter()
        for r in rows:
            source = Path(cfg['splits'][split]['path'])/r['name']
            gt = read_json(source/'ground_truth.json')
            actors = {a['id']:a for a in gt['observed_state']['actors']}
            for query in read_json(source/'query.json')['l4_interventions']:
                interventions[query['intervention_type']] += 1
                classes[actors.get(query['target_actor_id'], {}).get('entity_type', 'unknown')] += 1
        accepted[split] = dict(scenarios=len(rows), l3_branches=sum(r['l3_branches'] for r in rows),
                              l4_queries=sum(r['l4_queries'] for r in rows), target_classes=dict(classes),
                              interventions=dict(interventions), fidelity_mean=statistics.fmean(fidelity) if fidelity else None,
                              fidelity_median=statistics.median(fidelity) if fidelity else None,
                              fidelity_p95=float(np.quantile(fidelity,.95)) if fidelity else None)
    write_json(root/'dataset_summary.json', dict(splits=inventories, accepted=accepted, source_reports=reports))
    evidence = root/'author_evidence.yaml'
    if not evidence.exists():
        ethics = ('external_rater_study: not_conducted\n'
                  'reason: Ethics approval was not obtained; review is by the author only.\n'
                  if cfg['human'].get('mode') == 'author_review' else 'ethics_and_consent: null\n')
        evidence.write_text("# Fill from actual records; never infer missing attempts from accepted counts.\n" + ethics +
                            "api_expenditure_and_invoice: null\n"
                            "generation_runtime_and_attempts: null\nrejected_candidate_threshold_sensitivity: null\n"
                            "release_identifier_and_location: null\nauthor_interpretation: null\n")
    prepare_branch_audit(cfg, inventories)
    cells = matrix_cells(cfg)
    scenario_runs = sum(len(inventories[c['split']]) for c in cells)
    print(f'Prepared frozen campaign: {root}; {len(cells)} cells, {scenario_runs} main scenario runs')


def frozen_inputs(cfg, verify_files=False):
    frozen = read_json(Path(cfg['campaign_dir'])/'frozen.json')
    for key in ('splits', 'smoke', 'statistics', 'probe', 'human', 'validation'):
        if frozen['settings'][key] != cfg[key]:
            raise ValueError(f'Frozen {key} changed; use a new campaign directory')
    if frozen['code_files'] != [list(x) for x in code_inventory()]:
        raise ValueError('Frozen runtime code changed; use a new campaign directory and smoke admission')
    if verify_files:
        # Recompute the COMPLETE prompt-source inventory of every scenario and
        # compare membership and hashes with the frozen one, so an optional
        # dependency that appears or disappears after the freeze (the CARLA
        # scene configuration) is caught, not only a change to a file that
        # was already listed.
        from analysis.smoke_admission import scenario_input_fingerprint
        for split, rows in frozen['datasets'].items():
            base = Path(cfg['splits'][split]['path'])
            if {p.name for p in base.glob('scenario_*') if p.is_dir()} != {r['name'] for r in rows}:
                raise ValueError(f'{split}: dataset scenario membership changed')
            for row in rows:
                fingerprint, files = scenario_input_fingerprint(str(base/row['name']))
                current = {f['path']: f['sha256'] for f in files}
                expected = {f['path']: f['sha256'] for f in row['files']}
                if set(current) != set(expected):
                    added = sorted(set(current) - set(expected)); removed = sorted(set(expected) - set(current))
                    raise ValueError(f"Frozen dataset inventory changed: {split}/{row['name']}: added {added}, removed {removed}")
                for path_, digest in expected.items():
                    if current[path_] != digest:
                        raise ValueError(f"Frozen dataset changed: {split}/{row['name']}/{path_}")
                if fingerprint != row['fingerprint']:
                    raise ValueError(f"Frozen input fingerprint changed: {split}/{row['name']}")
    return frozen


def check_cell(out_dir, rows, condition=None, configuration=None, backend=None, sampling=None,
               *, allow_rescore=False):
    out = Path(out_dir); errors = []; missing = []; present = []
    expected = {r['name'] for r in rows}
    extra = sorted(p.name for p in out.glob('scenario_*') if p.is_dir() and p.name not in expected)
    for row in rows:
        d = out/row['name']
        if not d.is_dir():
            missing.append(row['name']); continue
        for filename in ('l3_pre.json', 'l4_pre.json', 'run_manifest.json'):
            try:
                obj = read_json(d/filename)
                if not isinstance(obj, dict):
                    raise ValueError('expected a JSON object')
                if filename != 'run_manifest.json' and obj.get('scenario_id') != row['scenario_id']:
                    raise ValueError('scenario ID does not match dataset')
            except (OSError, ValueError) as exc:
                errors.append(f'{d.name}/{filename}: {exc}')
        try:
            m = read_json(d/'run_manifest.json')
            for key, value in (('condition', condition), ('configuration_name', configuration), ('backend', backend)):
                if value is not None and m.get(key) != value:
                    raise ValueError(f'{key} differs from expected cell')
            if backend is not None:
                from simulation.protocol import PROTOCOL_VERSION
                from ctms.ctms import SCORER_VERSION, RESCORABLE_SCORER_VERSIONS
                from analysis.parse_prediction import PARSER_VERSION
                # Greedy is the protocol; a configuration that cannot decode
                # greedily declares its model card's sampling, seed 0.
                decoding = ({'decoding': 'sampling', 'temperature': float(sampling['temperature']),
                             'generation_seed': 0} if sampling else
                            {'decoding': 'greedy', 'temperature': 0.0})
                expected_metadata = {'protocol_version': PROTOCOL_VERSION, 'scorer_version': SCORER_VERSION,
                                     'parser_version': PARSER_VERSION, 'scorer_input_track': 'primary',
                                     'seed': 0, **decoding}
                for key, value in expected_metadata.items():
                    # A saved answer keeps the scorer version recorded at inference.
                    # Version 1.3 changes gates only; 1.2 prediction JSON has the
                    # same input contract and can be rescored without new inference.
                    if (key == 'scorer_version' and allow_rescore
                            and m.get(key) in RESCORABLE_SCORER_VERSIONS):
                        continue
                    if m.get(key) != value:
                        raise ValueError(f'{key} differs from final protocol')
                expected_frames = [f['sha256'] for f in row.get('files', []) if f['path'].startswith('frames/')]
                if condition == 'history_json':
                    expected_frames = []
                if 'files' in row and m.get('frame_sha256') != expected_frames:
                    raise ValueError('ordered frame hashes differ from frozen input')
            if configuration:
                if (d/'prompt.txt').exists() and sha(d/'prompt.txt') != m.get('prompt_sha256'):
                    raise ValueError('prompt.txt hash differs from manifest')
                if m.get('system_prompt_sha256') is not None and (
                        not (d/'system_prompt.txt').exists()
                        or sha(d/'system_prompt.txt') != m['system_prompt_sha256']):
                    raise ValueError('system_prompt.txt missing or its hash differs from manifest')
                for filename, key in (('raw_response.txt', 'raw_response_sha256'),
                                      ('final_response.txt', 'final_response_sha256')):
                    if sha(d/filename) != m.get(key):
                        raise ValueError(f'{filename} hash differs from manifest')
                read_json(d/'generation_metadata.json')
            present.append(row['name'])
        except (OSError, ValueError) as exc:
            errors.append(f'{d.name}/metadata: {exc}')
    return dict(complete=not missing and not errors and not extra,
                expected=len(rows), present=len(present), missing_scenarios=missing,
                unexpected_scenarios=extra, errors=errors)


def require_cell(cell, rows, *, allow_rescore=False):
    result = check_cell(cell['out_dir'], rows, cell['condition'],
                        cell.get('configuration'), cell.get('backend'), cell.get('sampling'),
                        allow_rescore=allow_rescore)
    write_json(Path(cell['out_dir'])/'completeness.json', result)
    if not result['complete']:
        raise ValueError(f"Incomplete cell {cell['out_dir']}: {result}")
    return result


def execute_cell(cfg, cell, frozen, *, limit=None):
    from analysis.run_model import run_model, _existing_output_complete
    from analysis.smoke_admission import admit_panel
    from analysis.model_panel import load_model_panel, resolve_configurations
    out = Path(cell['out_dir']); rows = frozen['datasets'][cell['split']]
    if limit is not None and (not isinstance(limit, int) or limit < 1):
        raise ValueError('scenario limit must be a positive integer')
    selected = rows if limit is None else rows[:limit]
    with locked(out/'.run.lock'):
        if not (out/'campaign_identity.json').exists() and any(out.glob('scenario_*')):
            raise ValueError(f'Existing predictions have no campaign identity: {out}; use an empty campaign root')
        write_campaign_identity(cfg, cell)
        model = None
        if cell.get('configuration'):
            panel = admit_panel(load_model_panel(str(PANEL)), cfg['admission'], repo_dir=str(ROOT))
            model = resolve_configurations(panel, [cell['configuration']])[0]
            model.update(model_panel_version=panel['panel_version'], protocol_version=panel['protocol_version'],
                         scorer_version=panel['scorer_version'])
        # Resume only missing/incomplete attempts. Completed malformed model
        # answers are retained as benchmark outcomes, not regenerated.
        pending = []
        for row in selected:
            d = out/row['name']
            if d.is_dir() and (d/'run_manifest.json').exists():
                _existing_output_complete(str(d), model, cell['condition'])
            check = check_cell(out, [row], cell['condition'], cell.get('configuration'), cell['backend'])
            if check['errors'] and all((d/f).exists() for f in ('l3_pre.json','l4_pre.json','run_manifest.json')):
                raise ValueError(f'Completed response has conflicting/corrupt artifacts: {check["errors"]}')
            if check['missing_scenarios'] or check['errors']:
                pending.append(row['name'])
        if pending:
            run_model(cfg['splits'][cell['split']]['path'], str(out), backend=cell['backend'],
                      model_id=model['model_id'] if model else '', model_config=model,
                      condition=cell['condition'], track='primary', seed=0, num_frames=8,
                      temperature=0.0, skip_existing=False, scenario_names=pending)
        if limit is None:
            require_cell(cell, rows)
        else:
            # Keep the full-campaign completion marker honest during a pilot.
            result = check_cell(out, rows, cell['condition'], cell.get('configuration'), cell['backend'])
            write_json(out/'completeness.json', result)
            partial = check_cell(out, selected, cell['condition'], cell.get('configuration'), cell['backend'])
            if partial['missing_scenarios'] or partial['errors'] or result['unexpected_scenarios']:
                raise ValueError(f'Incomplete selected scenarios: {partial}')
            print(f"Completed selected {len(selected)}/{len(rows)} scenarios: {out}", flush=True)


def write_campaign_identity(cfg, cell):
    """The identity run_cell records before inference; `score_cell` refuses a
    cell whose predictions carry another campaign's identity."""
    out = Path(cell['out_dir']); out.mkdir(parents=True, exist_ok=True)
    freeze(out/'campaign_identity.json', {'frozen_sha256': sha(Path(cfg['campaign_dir'])/'frozen.json'),
                                         'split': cell['split'], 'cell': cell})


def score_cell(cfg, cell, frozen):
    from scripts.assemble_dataset import assemble_gt_dataset, assemble_pred_dataset, find_pred_dir
    rows = frozen['datasets'][cell['split']]; out = Path(cell['out_dir'])
    require_cell(cell, rows, allow_rescore=True)
    # The predictions must have been generated under THIS campaign: the
    # identity written by run_cell before inference is compared with the
    # current frozen inputs, so a rescore never relabels older predictions.
    frozen_sha = sha(Path(cfg['campaign_dir'])/'frozen.json')
    identity_path = out/'campaign_identity.json'
    if not identity_path.exists():
        raise ValueError(f'{out}: no campaign_identity.json; predictions were not generated by run_cell of this campaign')
    if read_json(identity_path).get('frozen_sha256') != frozen_sha:
        raise ValueError(f'{out}: predictions were generated under another campaign identity; rerun inference in a new campaign root')
    gt = assemble_gt_dataset(cfg['splits'][cell['split']]['path'])
    # Guard the digit/index-based assembler fallback against mismatching directory IDs.
    for index, row in enumerate(rows):
        actual = find_pred_dir(str(out), row['scenario_id'], index, scenario_dir=row['name'])
        if actual is None or Path(actual).resolve() != (out/row['name']).resolve():
            raise ValueError(f"Ambiguous scenario/directory mapping for {row['scenario_id']}")
    pred = assemble_pred_dataset(str(out), gt)
    write_json(out/'gt.json', gt); write_json(out/'pred.json', pred)
    subprocess.run([sys.executable, str(ROOT/'run_benchmark.py'), '--gt', str(out/'gt.json'),
                    '--pred', str(out/'pred.json'), '--output', str(out/'report.json'),
                    '--allow-rescore'], cwd=ROOT, check=True)
    from analysis.causal_coherence import run_coherence
    write_json(out/'coherence.json', run_coherence(cfg['splits'][cell['split']]['path'], str(out)))
    # Reference-free L4 pair axes, computed from the public inputs and the
    # predictions only (never ground_truth.json), next to L3 coherence.
    from analysis.intervention_pair import run_l4_pair
    write_json(out/'l4_pair.json', run_l4_pair(cfg['splits'][cell['split']]['path'], str(out)))
    files = {str(p.relative_to(out)): sha(p) for row in rows for p in
             (out/row['name']/'l3_pre.json', out/row['name']/'l4_pre.json', out/row['name']/'run_manifest.json')}
    write_json(out/'score_provenance.json', dict(frozen_sha256=frozen_sha,
               scorer_sha256=sha(ROOT/'ctms/ctms.py'),
               scorer_config_sha256=sha(ROOT/'ctms/ctms_config.yaml'),
               prediction_files=files, report_sha256=sha(out/'report.json'), coherence_sha256=sha(out/'coherence.json'),
               l4_pair_sha256=sha(out/'l4_pair.json')))


def baseline_cells(cfg):
    return [dict(split=split, backend=b, condition='video',
                 out_dir=str(Path(cfg['campaign_dir'])/'baselines'/split/b))
            for split in sorted(cfg['splits']) for b in BASELINES]


def configure_model_cache(cfg):
    """Select one explicit hub cache before importing model/backend libraries."""
    cache = cfg.get('hf_hub_cache') or str(Path(cfg['hf_home'])/'hub')
    os.environ['HF_HOME'] = cfg['hf_home']
    os.environ['HF_HUB_CACHE'] = cache
    os.environ['HUGGINGFACE_HUB_CACHE'] = cache
    os.environ.pop('TRANSFORMERS_CACHE', None)


def run_task(cfg, task, index=0, backend=None, *, skip_gemini_smoke=False, limit=None):
    if skip_gemini_smoke and task != 'admit':
        raise ValueError('--skip-gemini-smoke applies only to task admit')
    if limit is not None and (task != 'infer' or limit < 1):
        raise ValueError('--limit requires task infer and a positive integer')
    os.chdir(ROOT)
    configure_model_cache(cfg)
    if backend == 'local':
        os.environ['HF_HUB_OFFLINE'] = '1'; os.environ['TRANSFORMERS_OFFLINE'] = '1'
    if task == 'prepare':
        with locked(Path(cfg['campaign_dir'])/'.prepare.lock'):
            prepare(cfg)
        return
    frozen = frozen_inputs(cfg, verify_files=task in {'infer', 'baselines', 'score', 'probe', 'smoke', 'report', 'validate'})
    if task == 'smoke':
        from analysis.smoke_admission import run_configuration_smoke
        from analysis.model_panel import load_model_panel, resolve_configurations
        panel = load_model_panel(str(PANEL))
        models = [m for m in resolve_configurations(panel, [m['name'] for m in panel['configurations']]) if m['backend'] == backend]
        m = models[index]
        run_configuration_smoke(str(PANEL), cfg['smoke']['path'],
                                str(Path(cfg['campaign_dir'])/'smoke/evidence'),
                                cfg['smoke']['scenario'], m['name'], repo_dir=str(ROOT))
    elif task == 'admit':
        from analysis.smoke_admission import build_admission, _load_evidence, validate_admission
        from analysis.model_panel import load_model_panel
        panel = load_model_panel(str(PANEL))
        admission = build_admission(panel, _load_evidence(str(Path(cfg['campaign_dir'])/'smoke/evidence')),
                                    skip_gemini_smoke=skip_gemini_smoke)
        validate_admission(panel, admission)
        write_json(cfg['admission'], admission)
        print(f"Admitted {admission['observed_cell_count']}/{admission['expected_cell_count']} smoke cells; "
              f"Gemini smoke: {'not_run (user requested)' if skip_gemini_smoke else 'passed'}")
    elif task == 'infer':
        cells = [c for c in matrix_cells(cfg) if c['backend'] == backend]
        execute_cell(cfg, cells[index], frozen, limit=limit)
    elif task == 'baselines':
        for cell in baseline_cells(cfg):
            execute_cell(cfg, cell, frozen)
            score_cell(cfg, cell, frozen)
        from analysis.causal_coherence import run_coherence
        for split, spec in cfg['splits'].items():
            write_json(Path(cfg['campaign_dir'])/'baselines'/split/'reference_coherence.json', run_coherence(spec['path']))
    elif task == 'score':
        # Check the entire expected matrix first, including absent cell directories.
        results = {c['out_dir']: check_cell(c['out_dir'], frozen['datasets'][c['split']],
                   c['condition'], c['configuration'], c['backend']) for c in matrix_cells(cfg)}
        write_json(Path(cfg['campaign_dir'])/'completeness.json', results)
        if not all(r['complete'] for r in results.values()):
            raise ValueError('Matrix incomplete; see completeness.json. Resume step 03.')
        for cell in matrix_cells(cfg):
            with locked(Path(cell['out_dir'])/'.run.lock'):
                score_cell(cfg, cell, frozen)
    elif task == 'probe':
        from analysis.evaluation_probe import run_panel_probe
        from analysis.model_panel import load_model_panel, resolve_configurations
        panel = load_model_panel(str(PANEL))
        models = [m for m in resolve_configurations(panel, [m['name'] for m in panel['configurations']]) if m['backend'] == backend]
        run_panel_probe(cfg, models[index]['name'], frozen)
    elif task == 'report':
        from analysis.evaluation_report import build_report
        from analysis.evaluation_validation import run_validation
        from analysis.evaluation_human import prepare_rating_package, analyze_packages
        with locked(Path(cfg['campaign_dir'])/'.report.lock'):
            build_report(cfg, frozen)
            run_validation(cfg, frozen)
            prepare_rating_package(cfg, frozen)
            analyze_packages(cfg)
            status(cfg)
    else:
        raise ValueError(task)


# Wall-time requested per task. Short measured defaults keep light CPU tasks
# out of the long queue (prepare takes ~6 min on Kaya). Tasks
# without a measured default fall back to slurm.cpu_time / slurm.gpu_time.
# Override per campaign with slurm.task_time in config.yaml, or per submit
# with `--time TASK=HH:MM:SS` / `--time HH:MM:SS`.
TASKS = ('prepare', 'smoke', 'admit', 'infer', 'baselines', 'probe', 'score', 'report')
DEFAULT_TASK_TIME = {'prepare': '00:10:00', 'admit': '00:10:00'}
CPU_STEPS = {'prepare': ['prepare'], 'report': ['baselines', 'score', 'report']}
_WALLTIME = re.compile(r'^(\d+-)?\d{1,2}:\d{2}(:\d{2})?$|^\d+$')


def parse_time_overrides(values):
    """`TASK=HH:MM:SS` (one task) or `HH:MM:SS` (every task) -> {task: time}."""
    out = {}
    for value in values or ():
        task, sep, walltime = value.partition('=')
        if not sep:
            task, walltime = '*', value
        if task != '*' and task not in TASKS:
            raise ValueError(f'unknown task {task!r}; choose from {", ".join(TASKS)}')
        if not _WALLTIME.match(walltime):
            raise ValueError(f'invalid SLURM time {walltime!r} (use minutes, HH:MM:SS or D-HH:MM:SS)')
        out[task] = walltime
    return out


def task_walltime(cfg, task, backend=None):
    s = cfg['slurm']; configured = s.get('task_time') or {}
    if '*' in configured:
        return configured['*']
    if task in configured:
        return configured[task]
    if task in DEFAULT_TASK_TIME:
        return DEFAULT_TASK_TIME[task]
    return s['gpu_time'] if backend == 'local' else s['cpu_time']


def run_step_local(cfg, step):
    """Run a CPU-only step in this process, without sbatch. For light work
    (prepare ~6 min) this avoids waiting a day in the queue; GPU/API steps
    still need SLURM. Runs are logged to local_runs.jsonl, not jobs.jsonl."""
    if step not in CPU_STEPS:
        raise ValueError(f'step {step!r} needs GPU/API array jobs; submit it through SLURM')
    if step == 'report':
        frozen_inputs(cfg)
    log = Path(cfg['campaign_dir'])/'local_runs.jsonl'
    log.parent.mkdir(parents=True, exist_ok=True)
    for task in CPU_STEPS[step]:
        started = time.time(); row = dict(task=task, step=step, host=os.uname().nodename)
        print(f'Running {task} locally (no SLURM)...', flush=True)
        try:
            run_task(cfg, task)
            row.update(status='completed')
        except BaseException as exc:
            row.update(status='failed', error=f'{type(exc).__name__}: {exc}')
            raise
        finally:
            row['duration_sec'] = round(time.time() - started, 1)
            with log.open('a') as stream:
                stream.write(json.dumps(row) + '\n')
        print(f'Completed {task} locally in {row["duration_sec"]} s', flush=True)


def submit_job(cfg, task, *, backend=None, count=None, dependencies=(), dry_run=False):
    s = cfg['slurm']; gpu = backend == 'local'
    walltime = task_walltime(cfg, task, backend)
    log_dir = Path(cfg['campaign_dir'])/'logs'
    command = ['sbatch', '--parsable', '--job-name', f'coriva-{task}', '--chdir', str(ROOT),
               '--partition', s['gpu_partition'] if gpu else s['cpu_partition'],
               '--mem', s['gpu_memory'] if gpu else s['cpu_memory'],
               '--time', walltime,
               '--nodes', '1', '--ntasks', '1', '--cpus-per-task', str(s['cpus']), '--output', str(log_dir/f'{task}-%A_%a.out')]
    if s.get('account'):
        command += ['--account', s['account']]
    if gpu:
        command += ['--gres', s['gpu_gres']]
    if count:
        parallel = s.get('gpu_parallel', 3) if gpu else s.get('api_parallel', 1)
        command += ['--array', f'0-{count-1}%{parallel}']
    if dependencies:
        command += ['--dependency', 'afterok:' + ':'.join(dependencies), '--kill-on-invalid-dep=yes']
    command += [str(ROOT/'evaluation/job.sbatch'), cfg['python'], '-m', 'analysis.evaluation_campaign',
                '--config', cfg['_path'], 'task', task]
    if backend:
        command += ['--backend', backend]
    print(shlex.join(command), flush=True)
    if dry_run:
        return f'DRY_{task}_{backend or "cpu"}'
    log_dir.mkdir(parents=True, exist_ok=True)
    completed = subprocess.run(command, check=True, text=True, capture_output=True)
    job_id = completed.stdout.strip().split(';')[0]
    if not job_id.isdigit():
        raise RuntimeError(f'Unexpected sbatch response: {completed.stdout!r}')
    with (Path(cfg['campaign_dir'])/'jobs.jsonl').open('a') as stream:
        stream.write(json.dumps(dict(task=task, backend=backend, job_id=job_id, dependencies=list(dependencies),
                                     walltime=walltime)) + '\n')
    print(f'Submitted {task}: {job_id}', flush=True)
    return job_id


def submit_step(cfg, step, dry_run=False):
    # No automatic cancellation: re-submitting while jobs run would race resume.
    log = Path(cfg['campaign_dir'])/'jobs.jsonl'
    if log.exists() and not dry_run:
        ids = [json.loads(line)['job_id'] for line in log.read_text().splitlines() if line.strip()]
        import getpass
        queue = subprocess.run(['squeue', '--noheader', '--user', getpass.getuser(), '--format', '%i %T'],
                               check=True, text=True, capture_output=True).stdout
        # Query the live queue, then filter our recorded IDs. Passing expired
        # IDs to squeue --jobs can fail after Slurm purges completed jobs.
        active = '\n'.join(line for line in queue.splitlines()
                           if line.split() and line.split()[0].split('_')[0] in set(ids))
        if active:
            raise RuntimeError(f'Campaign still has queued/running jobs; wait or inspect squeue:\n{active}')
    from analysis.model_panel import load_model_panel, resolve_configurations
    panel = load_model_panel(str(PANEL))
    configurations = resolve_configurations(panel, [c["name"] for c in panel["configurations"]])
    backend_counts = {backend: sum(c['backend'] == backend for c in configurations)
                      for backend in ('local', 'gemini')}
    split_count = len(cfg.get('splits', ('carla', 'nuscenes')))
    submit = lambda task, **kw: submit_job(cfg, task, dry_run=dry_run, **kw)
    if step == 'prepare':
        submit('prepare')
    elif step == 'smoke':
        if not dry_run:
            frozen_inputs(cfg)
        local = submit('smoke', backend='local', count=backend_counts['local'])
        api = submit('smoke', backend='gemini', count=backend_counts['gemini'])
        submit('admit', dependencies=[local, api])
    elif step == 'evaluate':
        if not dry_run:
            frozen_inputs(cfg)
            from analysis.model_panel import load_model_panel
            from analysis.smoke_admission import admit_panel
            admit_panel(load_model_panel(str(PANEL)), cfg['admission'], repo_dir=str(ROOT))
        local = submit('infer', backend='local', count=backend_counts['local'] * len(CONDITIONS) * split_count)
        api = submit('infer', backend='gemini', count=backend_counts['gemini'] * len(CONDITIONS) * split_count)
        baseline = submit('baselines')
        probe_local = submit('probe', backend='local', count=backend_counts['local'], dependencies=[local])
        probe_api = submit('probe', backend='gemini', count=backend_counts['gemini'], dependencies=[api])
        score = submit('score', dependencies=[local, api, baseline])
        submit('report', dependencies=[score, probe_local, probe_api])
    elif step == 'report':
        if not dry_run:
            frozen_inputs(cfg)
        baseline = submit('baselines')
        score = submit('score', dependencies=[baseline])
        submit('report', dependencies=[score])


def status(cfg):
    root = Path(cfg['campaign_dir'])
    frozen_path = root/'frozen.json'
    if not frozen_path.exists():
        print('NOT PREPARED: run step 01'); return
    frozen = read_json(frozen_path)
    cells = matrix_cells(cfg)
    incomplete = []
    for c in cells:
        marker = Path(c['out_dir'])/'completeness.json'
        check = read_json(marker) if marker.exists() else {'complete': False, 'reason': 'no completed-cell record'}
        incomplete.append(dict(split=c['split'], configuration=c['configuration'], condition=c['condition'], **check))
    count = sum(c['complete'] for c in incomplete)
    required = [root/'report/results.json', root/'validation/results.json']
    required += [Path(c['out_dir'])/f for c in cells+baseline_cells(cfg) for f in ('report.json','coherence.json','score_provenance.json')]
    required += [root/'probes'/name/'report.json' for name in {c['configuration'] for c in cells}]
    missing = [str(p.relative_to(root)) for p in required if not p.exists()]
    human = read_json(root/'human/status.json') if (root/'human/status.json').exists() else {'status':'pending'}
    evidence = yaml.safe_load((root/'author_evidence.yaml').read_text()) if (root/'author_evidence.yaml').exists() else {}
    state = dict(expected_cells=len(cells), complete_cells=count, cells=incomplete,
                 completeness_source='cached job records; scoring verifies live files before producing a report',
                 automated_complete=count==len(cells) and not missing, missing_automated_artifacts=missing,
                 human=human, author_evidence=evidence,
                 dissertation_requires_author_review=True)
    write_json(root/'status.json', state)
    print(f"Completed predictions: {count}/{len(cells)} cells. Status: {root/'status.json'}")
    print('Human ratings and author evidence are tracked separately; predictions alone do not complete the dissertation.')


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('--config', default=str(ROOT/'evaluation/config.yaml'))
    sub = parser.add_subparsers(dest='command', required=True)
    s = sub.add_parser('submit'); s.add_argument('step', choices=['prepare', 'smoke', 'evaluate', 'report'])
    s.add_argument('--dry-run', action='store_true')
    s.add_argument('--time', action='append', default=[], metavar='[TASK=]WALLTIME',
                   help='SLURM wall-time override: TASK=HH:MM:SS for one task, HH:MM:SS for every task of the step')
    s.add_argument('--local', action='store_true',
                   help='run a CPU-only step (prepare, report) in this process instead of submitting to SLURM')
    t = sub.add_parser('task'); t.add_argument('task'); t.add_argument('--backend', choices=['local', 'gemini'])
    t.add_argument('--index', type=int, default=None)
    t.add_argument('--skip-gemini-smoke', action='store_true',
                   help='task admit only: require all local smoke cells; record Gemini smoke as not_run')
    t.add_argument('--limit', type=int,
                   help='task infer only: run the first N frozen scenarios; retain results for full resume')
    sub.add_parser('status')
    args = parser.parse_args(); cfg = load_config(args.config)
    if args.command == 'submit':
        overrides = parse_time_overrides(args.time)
        if overrides:
            cfg['slurm']['task_time'] = {**(cfg['slurm'].get('task_time') or {}), **overrides}
        if args.local:
            if args.dry_run:
                print(f'DRY RUN: would run {" -> ".join(CPU_STEPS.get(args.step, []))} in this process')
            else:
                with locked(Path(cfg['campaign_dir'])/'.submit.lock'):
                    run_step_local(cfg, args.step)
        elif args.dry_run:
            submit_step(cfg, args.step, True)
        else:
            with locked(Path(cfg['campaign_dir'])/'.submit.lock'):
                submit_step(cfg, args.step)
    elif args.command == 'status':
        status(cfg)
    else:
        os.environ.setdefault('CORIVA_API_USAGE_LOG', str(Path(cfg['campaign_dir'])/'logs'/f'api-{os.environ.get("SLURM_JOB_ID", "manual")}.jsonl'))
        options = {'skip_gemini_smoke': True} if args.skip_gemini_smoke else {}
        if args.limit is not None:
            options['limit'] = args.limit
        run_task(cfg, args.task, args.index if args.index is not None else int(os.environ.get('SLURM_ARRAY_TASK_ID', 0)), args.backend, **options)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
