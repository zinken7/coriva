"""Check one CARLA and one nuScenes scenario across the four local models.

Requires campaign prepare and admission. Predictions stay in the real campaign
for full resume. A portable copy of inputs, predictions and partial scores is
written to two_scenario_check; it is not a full-campaign report.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import shutil
import subprocess
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from analysis.evaluation_campaign import (
    PANEL, ROOT, baseline_cells, check_cell, configure_model_cache, execute_cell,
    freeze, frozen_inputs, load_config, locked, matrix_cells, read_json,
    score_cell, sha, write_campaign_identity, write_json,
)
from analysis.evaluation_report import collect_cell, export_tables, mean_stat
from analysis.model_panel import load_model_panel
from analysis.smoke_admission import admit_panel


def run(cfg):
    configure_model_cache(cfg)
    frozen = frozen_inputs(cfg, verify_files=True)
    # The same gate and prediction metadata are used by the full campaign.
    admit_panel(load_model_panel(str(PANEL)), cfg['admission'], repo_dir=str(ROOT))
    source = Path(cfg['campaign_dir'])
    out = source/'two_scenario_check'
    cells = [c for c in matrix_cells(cfg) if c['backend'] == 'local']
    selected = {split: rows[:1] for split, rows in frozen['datasets'].items()}
    if set(selected) != {'carla', 'nuscenes'} or any(len(rows) != 1 for rows in selected.values()):
        raise ValueError('two-scenario check requires one frozen scenario from each split')

    with locked(out/'.run.lock'):
        preview_frozen = dict(
            purpose='partial two-scenario pipeline check, not a full-campaign report',
            campaign_frozen_sha256=sha(source/'frozen.json'),
            smoke_admission_sha256=sha(Path(cfg['admission'])),
            datasets=selected, configurations=sorted({c['configuration'] for c in cells}),
        )
        freeze(out/'frozen.json', preview_frozen)
        shutil.copy2(source/'frozen.json', out/'campaign_frozen.json')
        shutil.copy2(cfg['admission'], out/'admission.json')
        if (source/'provenance.json').exists():
            shutil.copy2(source/'provenance.json', out/'campaign_provenance.json')
        preview_cfg = dict(campaign_dir=str(out), splits={})
        for split, rows in selected.items():
            subset = out/'inputs'/split
            preview_cfg['splits'][split] = {'path': str(subset)}
            name = rows[0]['name']
            shutil.copytree(Path(cfg['splits'][split]['path'])/name, subset/name, dirs_exist_ok=True)
            print(f'SELECTED {split}/{name}', flush=True)

        records = []
        status = dict(purpose=preview_frozen['purpose'], complete=False,
                      expected_model_cells=len(cells), completed_model_cells=0,
                      expected_baseline_cells=len(baseline_cells(cfg)), completed_baseline_cells=0,
                      selected_scenarios={s: r[0]['name'] for s, r in selected.items()})
        write_json(out/'status.json', status)

        def score_copy(cell):
            split = cell['split']; name = selected[split][0]['name']
            destination = out/Path(cell['out_dir']).relative_to(source)
            # Validate the completed source before copying it for review.
            checked = check_cell(cell['out_dir'], selected[split], cell['condition'],
                                 cell.get('configuration'), cell['backend'])
            if checked['missing_scenarios'] or checked['errors']:
                raise ValueError(f'Cannot review incomplete campaign output: {checked}')
            shutil.copytree(Path(cell['out_dir'])/name, destination/name, dirs_exist_ok=True)
            original_identity = read_json(Path(cell['out_dir'])/'campaign_identity.json')
            if original_identity['frozen_sha256'] != preview_frozen['campaign_frozen_sha256']:
                raise ValueError('Source predictions belong to another campaign')
            write_json(destination/'source_campaign_identity.json', original_identity)
            review_cell = dict(cell, out_dir=str(destination))
            write_campaign_identity(preview_cfg, review_cell)
            score_cell(preview_cfg, review_cell, preview_frozen)
            data = collect_cell(review_cell, selected[split],
                                split_path=preview_cfg['splits'][split]['path'],
                                frozen_sha256=sha(out/'frozen.json'))
            write_json(destination/'collected.json', data)
            for metric, values in data['metrics'].items():
                records.append(dict(split=split,
                                    configuration=cell.get('configuration', cell['backend']),
                                    condition=cell['condition'], comparison='scores', metric=metric,
                                    **mean_stat(list(values.values()))))
            export_tables(out, records)

        for index, cell in enumerate(cells):
            status['current_cell'] = f"{cell['split']}/{cell['configuration']}/{cell['condition']}"
            write_json(out/'status.json', status)
            print(f"RUN {index + 1}/{len(cells)}: {status['current_cell']}", flush=True)
            # A fresh process per cell releases all GPU weights/cache between models.
            subprocess.run([
                sys.executable, '-u', '-m', 'analysis.evaluation_campaign',
                '--config', cfg['_path'], 'task', 'infer', '--backend', 'local',
                '--index', str(index), '--limit', '1',
            ], cwd=ROOT, check=True)
            score_copy(cell)
            status['completed_model_cells'] += 1
            write_json(out/'status.json', status)

        for cell in baseline_cells(cfg):
            status['current_cell'] = f"{cell['split']}/{cell['backend']}"
            write_json(out/'status.json', status)
            print(f"BASELINE {status['current_cell']}", flush=True)
            execute_cell(cfg, cell, frozen, limit=1)
            score_copy(cell)
            status['completed_baseline_cells'] += 1
            write_json(out/'status.json', status)
        status.update(complete=True, current_cell=None)
        write_json(out/'status.json', status)
        print(f'DONE: {out}; full inference will reuse these campaign predictions', flush=True)
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', default=str(ROOT/'evaluation/config.yaml'))
    args = parser.parse_args()
    run(load_config(args.config))


if __name__ == '__main__':
    main()
