"""Partial live-model runs use campaign resume and the real scoring pipeline."""
import copy
import json
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tests.test_evaluation_campaign import make_protocol_dataset
from tests.test_smoke_admission import _repository, _valid_evidence


def test_two_scenario_check_is_portable_and_full_run_reuses_predictions(tmp_path, monkeypatch):
    import analysis.evaluation_campaign as campaign
    import analysis.run_model as runner
    import analysis.smoke_admission as smoke
    from analysis.backends.result import result_from_raw
    from analysis.model_panel import load_model_panel
    from scripts import check_two_scenarios as pilot

    source = tmp_path/'campaign'
    cfg = dict(campaign_dir=str(source), hf_home=str(tmp_path/'cache'),
               admission=str(source/'smoke/admission.json'), _path=str(tmp_path/'config.yaml'), splits={})
    frozen = {'datasets': {}}
    for split in ('carla', 'nuscenes'):
        data = make_protocol_dataset(tmp_path, split=split, n=2)
        cfg['splits'][split] = {'path': str(data)}
        frozen['datasets'][split] = campaign.dataset_inventory(data, 2, split)
    campaign.write_json(source/'frozen.json', frozen)
    panel = load_model_panel(str(campaign.PANEL))
    records = [r for r in _valid_evidence(panel) if r['backend'] == 'local']
    campaign.write_json(cfg['admission'], smoke.build_admission(panel, records, skip_gemini_smoke=True))
    monkeypatch.setattr(smoke, 'repository_state', lambda *a, **kw: _repository())
    monkeypatch.setattr(pilot, 'frozen_inputs', lambda *a, **kw: copy.deepcopy(frozen))
    monkeypatch.setattr(pilot, 'configure_model_cache', lambda cfg: None)
    calls = []
    def generate(prompt, frames, model_id, opts):
        calls.append((model_id, opts['inference_mode'], len(frames)))
        # Bad model answers are measured and retained, never retried for quality.
        return result_from_raw('I cannot produce the requested JSON.')
    monkeypatch.setitem(runner.BACKENDS, 'local', generate)
    cells = [c for c in campaign.matrix_cells(cfg) if c['backend'] == 'local']
    original_subprocess = pilot.subprocess.run
    interrupt = [True]
    invocations = []
    def fake_child(command, **kwargs):
        if 'analysis.evaluation_campaign' not in command:
            return original_subprocess(command, **kwargs)  # real scorer subprocess
        assert command[command.index('--limit') + 1] == '1'
        index = int(command[command.index('--index') + 1])
        invocations.append(index)
        if index == 1 and interrupt[0]:
            raise RuntimeError('simulated inference interruption')
        campaign.execute_cell(cfg, cells[index], frozen, limit=1)
    monkeypatch.setattr(pilot.subprocess, 'run', fake_child)

    with pytest.raises(RuntimeError, match='simulated inference interruption'):
        pilot.run(cfg)
    out = source/'two_scenario_check'
    assert len(calls) == 1
    assert campaign.read_json(out/'status.json')['complete'] is False
    first_response = Path(cells[0]['out_dir'])/'scenario_1/raw_response.txt'
    original_hash = campaign.sha(first_response)
    interrupt[0] = False
    pilot.run(cfg)
    status = campaign.read_json(out/'status.json')
    assert status['complete'] is True
    assert status['completed_model_cells'] == status['expected_model_cells'] == 24
    assert status['completed_baseline_cells'] == status['expected_baseline_cells'] == 14
    assert status['selected_scenarios'] == {'carla': 'scenario_1', 'nuscenes': 'scenario_1'}
    assert len(calls) == 24  # first successful answer survived the interrupted pilot
    assert len({call[0] for call in calls}) == 4
    assert sum(call[2] == 0 for call in calls) == 8
    assert sum(call[2] == 8 for call in calls) == 16
    assert campaign.sha(first_response) == original_hash
    assert campaign.read_json(source/'frozen.json') == frozen
    assert not any(p.is_symlink() for p in out.rglob('*'))
    assert (out/'inputs/carla/scenario_1/ground_truth.json').is_file()
    assert (out/'inputs/nuscenes/scenario_1/frames/000.jpg').is_file()
    assert (out/'tables.csv').is_file()
    assert (out/'admission.json').is_file()
    for cell in cells:
        proof = campaign.read_json(Path(cell['out_dir'])/'completeness.json')
        assert proof['expected'] == 2 and proof['complete'] is False
        preview = out/Path(cell['out_dir']).relative_to(source)
        assert (preview/'collected.json').is_file()
        assert (preview/'report.json').is_file()
        assert (preview/'coherence.json').is_file()
        assert (preview/'l4_pair.json').is_file()
        manifest = campaign.read_json(preview/'scenario_1/run_manifest.json')
        assert manifest['execution_purpose'] == 'benchmark'
        assert campaign.sha(preview/'scenario_1/prompt.txt') == manifest['prompt_sha256']

    # A full inference call computes only the second scene and preserves the first.
    campaign.execute_cell(cfg, cells[0], frozen)
    assert len(calls) == 25
    assert campaign.sha(first_response) == original_hash
    assert campaign.read_json(Path(cells[0]['out_dir'])/'completeness.json')['complete'] is True
    # Repeating the pilot after more campaign work must neither fail nor rerun models.
    pilot.run(cfg)
    assert len(calls) == 25
    assert campaign.read_json(Path(cells[0]['out_dir'])/'completeness.json')['complete'] is True


def test_partial_inference_rejects_invalid_limit_before_creating_output(tmp_path):
    from analysis.evaluation_campaign import execute_cell, run_task
    cell = dict(split='carla', out_dir=str(tmp_path/'output'))
    with pytest.raises(ValueError, match='positive integer'):
        execute_cell({}, cell, {'datasets': {'carla': []}}, limit=0)
    assert not (tmp_path/'output').exists()
    with pytest.raises(ValueError, match='requires task infer'):
        run_task({}, 'smoke', limit=1)
