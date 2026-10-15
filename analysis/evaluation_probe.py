"""Panel-wide observed-frame recognition probe with per-response resume."""
import hashlib
from pathlib import Path
import random
import time

from analysis.evaluation_campaign import (PANEL, read_json, write_json, freeze, sha, locked)
from analysis.contamination_probe import build_probe_prompt, parse_probe_response, summarize_probe


def run_panel_probe(cfg, name, frozen, *, generator=None, resolved_model=None):
    from analysis.model_panel import load_model_panel, resolve_configurations
    from analysis.smoke_admission import admit_panel
    from analysis.run_model import BACKENDS
    from analysis.backends.result import normalize_generation_result
    if resolved_model is None:
        panel=admit_panel(load_model_panel(str(PANEL)),cfg['admission'])
        resolved_model=resolve_configurations(panel,[name])[0]
    model=resolved_model
    if generator is None:
        generator=BACKENDS[model['backend']]
    rows=list(frozen['datasets']['nuscenes'])
    random.Random(cfg['statistics']['seed']).shuffle(rows)
    rows=rows[:cfg['probe']['limit']]
    out=Path(cfg['campaign_dir'])/'probes'/name
    prompt=build_probe_prompt().replace('{{','{').replace('}}','}')
    with locked(out/'.probe.lock'):
        identity=dict(configuration=name,model_id=model['model_id'],revision=model.get('revision'),
                      mode=model.get('inference_mode'),prompt=prompt,
                      selected_inputs={r['name']:r['fingerprint'] for r in rows})
        freeze(out/'identity.json',identity)
        parsed_rows=[]; versions=set()
        for row in rows:
            d=out/row['name']; record=d/'response.json'
            if record.exists():
                saved=read_json(record)
                if sha(d/'raw_response.txt')!=saved['raw_sha256'] or sha(d/'final_response.txt')!=saved['final_sha256']:
                    raise ValueError(f'Probe response changed: {d}')
            else:
                base=Path(cfg['splits']['nuscenes']['path'])/row['name']
                frames=[str(base/f['path']) for f in row['files'] if f['path'].startswith('frames/')]
                if len(frames)!=8:
                    raise ValueError('Recognition probe requires the same eight observed frames')
                started=time.monotonic()
                result=normalize_generation_result(generator(prompt,frames,model['model_id'],dict(model,seed=None,temperature=0.0)))
                metadata=dict(result.metadata);metadata.setdefault('latency_seconds',time.monotonic()-started)
                d.mkdir(parents=True,exist_ok=True)
                (d/'raw_response.txt').write_text(result.raw_text)
                (d/'final_response.txt').write_text(result.final_text)
                if result.thinking_trace is not None:
                    (d/'thinking_trace.txt').write_text(result.thinking_trace)
                saved=dict(parsed=parse_probe_response(result.final_text),metadata=metadata,
                           raw_sha256=sha(d/'raw_response.txt'),final_sha256=sha(d/'final_response.txt'))
                write_json(record,saved)
            version=saved['metadata'].get('model_version')
            if version: versions.add(version)
            if model['backend']=='gemini' and not version:
                raise ValueError('Gemini recognition probe is missing provider model_version')
            parsed_rows.append(dict(scenario_id=row['name'],**saved['parsed']))
        if len(versions)>1:
            raise ValueError('Recognition probe mixes provider model versions')
        report=dict(configuration=name,model_id=model['model_id'],identity=identity,
                    summary=summarize_probe(parsed_rows),per_scenario=parsed_rows,
                    criterion='dataset naming and city naming; scene recognition is unverified self-report',
                    note='Neither city naming nor recognized_scene=true confirms training exposure or exact scene identity.')
        write_json(out/'report.json',report)
        return report
