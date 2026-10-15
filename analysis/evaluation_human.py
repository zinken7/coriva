"""Portable human-review packages and conservative analysis of returned ratings.

No ratings are generated. Private model/score keys never enter public stimuli.
"""
from collections import defaultdict
import copy
import csv
import html
import json
import math
from pathlib import Path
import random
import shutil
import statistics
from analysis.report_labels import TASK_LABELS


import numpy as np

from analysis.evaluation_campaign import read_json, write_json, freeze, matrix_cells, sha
from analysis.rq3b import SCALES, spearman
from analysis.branch_audit import krippendorff_alpha, audit_summary


def csv_write(path, fields, rows):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=fields,extrasaction='ignore');w.writeheader();w.writerows(rows)


def read_rows(path):
    with Path(path).open(newline='') as f:
        return list(csv.DictReader(f))


def public_value(value):
    if isinstance(value,dict):
        return {k:public_value(v) for k,v in value.items()
                if k not in {'scenario_id','model','model_id','configuration_name','branch_id','query_id','node_id'}}
    if isinstance(value,list): return [public_value(v) for v in value]
    return value


def stimulus(public_dir,item_id,source,content):
    """Copy eight images under anonymous names and emit a self-contained page."""
    d=Path(public_dir)/item_id; d.mkdir(parents=True,exist_ok=True)
    frames=sorted(list((Path(source)/'frames').glob('*.jpg'))+list((Path(source)/'frames').glob('*.png')))
    if len(frames)!=8: raise ValueError('Human stimulus requires eight frames')
    images=[]
    for i,frame in enumerate(frames):
        name=f'frame{i:02d}{frame.suffix}';shutil.copyfile(frame,d/name)
        images.append(f'<figure><img src="{name}"><figcaption>T={-3.5+i*.5:.1f}s</figcaption></figure>')
    body=['<!doctype html><meta charset="utf-8"><title>Driving prediction review</title>',
          '<style>body{max-width:1100px;margin:2em auto;font:16px sans-serif} .frames{display:grid;grid-template-columns:repeat(4,1fr)}figure{margin:5px}img{width:100%}pre{white-space:pre-wrap;overflow-wrap:anywhere;background:#f5f5f5;padding:1em}</style>',
          f'<h1>{html.escape(item_id)}</h1><div class="frames">'+''.join(images)+'</div>']
    for heading,value in content.items():
        body += [f'<h2>{html.escape(heading)}</h2><pre>{html.escape(json.dumps(public_value(value),indent=2))}</pre>']
    (d/'index.html').write_text('\n'.join(body))


def package_index(public_dir, ids, instructions):
    p=Path(public_dir);p.mkdir(parents=True,exist_ok=True)
    links=''.join(f'<li><a href="{i}/index.html">{i}</a></li>' for i in ids)
    (p/'index.html').write_text('<!doctype html><meta charset="utf-8"><title>Review items</title><h1>Review items</h1><p>'+html.escape(instructions)+'</p><ol>'+links+'</ol>')


def prepare_branch_audit(cfg,inventories):
    from analysis.branch_audit import (build_audit_sheet, check_branch_contract,
                                       AUDIT_VERSION, AUDIT_INSTRUCTIONS)
    root=Path(cfg['campaign_dir'])/'human/branch_audit';root.mkdir(parents=True,exist_ok=True)
    if (root/'package.json').exists():
        if read_json(root/'package.json').get('rubric_version') != AUDIT_VERSION:
            raise ValueError('Outdated branch audit: run scripts/rebuild_branch_audit.py --config CONFIG --reset')
        return
    rows=[];keys={};ids=[];checks={}
    for split in sorted(cfg['splits']):
        temp=root/f'{split}_source.csv'
        build_audit_sheet([cfg['splits'][split]['path']],str(temp),split_name=split,
                          n_raters=cfg['human']['raters'],limit=cfg['human']['audit_per_split'],seed=cfg['statistics']['seed'])
        source_by_id={r['scenario_id']:r for r in inventories[split]}
        names={}
        for row in read_rows(temp):
            sid=row['scenario_id']
            if sid not in names:
                iid=f'audit_{len(ids):04d}';ids.append(iid);names[sid]=iid
                source=Path(cfg['splits'][split]['path'])/source_by_id[sid]['name']
                gt=read_json(source/'ground_truth.json')
                query=read_json(source/'query.json')
                checks[iid]=check_branch_contract(gt,query,split)
                stimulus(root/'public',iid,source,{'Observed state':gt['observed_state'],
                          'Declared query':query,'Included branches':gt['branches'],
                          'Scene context':dict(split=split,scenario_type=gt.get('scenario_type'),
                              reference_source='CARLA simulator' if split=='carla' else 'Recorded factual / kinematic-twin alternatives'),
                          'Automatic protocol checks':checks[iid]})
                keys[iid]=dict(split=split,scenario_id=sid,directory=source_by_id[sid]['name'])
            row['scenario_id']=names[sid];rows.append(row)
    author = cfg['human'].get('mode') == 'author_review'
    if author:
        for row in rows: row['notes'] = ''
    csv_write(root/'public/ratings.csv',list(rows[0]),rows)
    instructions = AUDIT_INSTRUCTIONS
    if not author:
        instructions = AUDIT_INSTRUCTIONS + ' External-rater mode: use the configured rater columns.'
    package_index(root/'public',ids,instructions)
    write_json(root/'private_key.json',keys)
    write_json(root/'public/automatic_checks.json',dict(
        rubric_version=AUDIT_VERSION, scenes=checks,
        counts={status:sum(v['status']==status for v in checks.values()) for status in ('pass','fail','not_checked')}))
    write_json(root/'package.json',dict(rubric_version=AUDIT_VERSION,scenarios=len(ids),items=len(rows),
        scene_items=sum(r['item_type']=='scene_context' for r in rows),
        branch_items=sum(r['item_type']=='included_branch' for r in rows),
        ratings_sha256=sha(root/'public/ratings.csv')))


def stratified_items(pool,n=120,seed=0):
    """Equal task/split/score-tertile quotas, without repeated source scenes."""
    grouped=defaultdict(list)
    for item in pool: grouped[(item['task'],item['split'])].append(item)
    strata={}
    for (task,split),items in sorted(grouped.items()):
        ordered=sorted(items,key=lambda it:(it['score'],it['name'],it['model']))
        for i,item in enumerate(ordered):
            t=min(2,i*3//len(ordered));entry=dict(item,tertile=t)
            strata.setdefault((task,split,t),[]).append(entry)
    if len(strata)!=12 or n%12:
        raise ValueError('Need all 12 task/split/tertile strata and equal quotas')
    rng=random.Random(seed)
    for items in strata.values(): rng.shuffle(items)
    chosen=[];used=set()
    for _ in range(n//12):
        for key,items in sorted(strata.items()):
            item=next((x for x in items if (x['split'],x['name']) not in used),None)
            if item is None: raise ValueError(f'Insufficient distinct scenarios for human stratum {key}')
            chosen.append(item);used.add((item['split'],item['name']))
    return chosen


def prepare_rating_package(cfg,frozen):
    author = cfg['human'].get('mode') == 'author_review'
    root=Path(cfg['campaign_dir'])/'human'/('author_review' if author else 'ctms');root.mkdir(parents=True,exist_ok=True)
    identity={'frozen_sha256':sha(Path(cfg['campaign_dir'])/'frozen.json'),
              'reports':{str(Path(c['out_dir'])/'report.json'):sha(Path(c['out_dir'])/'report.json')
                         for c in matrix_cells(cfg) if c['condition']=='video'}}
    if (root/'package.json').exists():
        freeze(root/'identity.json',identity);return
    freeze(root/'identity.json',identity)
    pool=[]
    for cell in matrix_cells(cfg):
        if cell['condition']!='video': continue
        byid={r['scenario_id']:r['name'] for r in frozen['datasets'][cell['split']]}
        rep=read_json(Path(cell['out_dir'])/'report.json')
        for row in rep['per_sample']:
            for task,axis in [('L3','Causal_Foresight_F1'),('L4','Counterfactual_Alignment_Score')]:
                pool.append(dict(model=cell['configuration'],split=cell['split'],task=task,
                                 name=byid[row['scenario_id']],score=row['scores'][axis],out_dir=cell['out_dir']))
    items=stratified_items(pool,cfg['human']['items'],cfg['statistics']['seed'])
    if author:
        prepare_author_items(root, cfg, items)
        return
    # Round-robin sampled order spans all strata. Calibration draws are taken
    # evenly across that order; scenes never recur in holdout.
    for i,item in enumerate(items):
        item['is_attention']=False;item['calibration']=i<cfg['human']['calibration']
    used={(i['split'],i['name']) for i in items};checks=[]
    rng=random.Random(cfg['statistics']['seed']+2)
    for split in sorted(cfg['splits']):
        rows=list(frozen['datasets'][split]);rng.shuffle(rows)
        for backend,expect in [('oracle','high'),('random_valid','low')]:
            picked=[r for r in rows if (split,r['name']) not in used][:3]
            if len(picked)!=3: raise ValueError('Insufficient independent attention candidates')
            for row in picked:
                used.add((split,row['name']))
                checks.append(dict(model=backend,split=split,name=row['name'],task='L3' if len(checks)%2==0 else 'L4',
                                   out_dir=str(Path(cfg['campaign_dir'])/'baselines'/split/backend),
                                   is_attention=True,calibration=False,expected=expect))
    items+=checks;rng.shuffle(items)
    public=root/'public';public.mkdir(exist_ok=True)
    keys={};sheet=[];review=[]
    fields=['item_id','task']+[f'rater{i+1}_{s}' for i in range(cfg['human']['raters']) for s in SCALES]
    for index,item in enumerate(items):
        iid=f'item_{index:04d}';keys[iid]=item
        sheet.append(dict(item_id=iid,task=TASK_LABELS[item['task']]))
        source=Path(cfg['splits'][item['split']]['path'])/item['name']
        pred=Path(item['out_dir'])/item['name']
        query=read_json(source/'query.json')
        if item['task']=='L3':
            query={k:v for k,v in query.items() if k=='l3_ego_actions'}
            predictions=read_json(pred/'l3_pre.json').get('predicted_branches',[])
        else:
            query={k:v for k,v in query.items() if k.startswith('l4_')}
            predictions=read_json(pred/'l4_pre.json').get('predicted_counterfactuals',[])
        stimulus(public,iid,source,{'Task':TASK_LABELS[item['task']],'Observed state':read_json(source/'observed_state.json'),
                                    'Query':query,'Prediction':predictions})
        if item['is_attention']: review.append(dict(item_id=iid,suggested_expectation=item['expected'],approved='',expectation='',notes=''))
    csv_write(public/'ratings.csv',fields,sheet)
    csv_write(root/'attention_review.csv',['item_id','suggested_expectation','approved','expectation','notes'],review)
    package_index(public,list(keys),'Use ratings.csv and the anchored scales in INSTRUCTIONS.md. Judge the displayed prediction, including empty predictions. Model identities and scores are hidden.')
    (public/'INSTRUCTIONS.md').write_text('''# Rating instructions

Rate each item independently on three 1–5 scales. The task is L3 ego-action branches or L4 target-actor consequences. Positions remain in the fixed T0 ego frame, x forward and y right. Actor updates are state values, not increments. The observer receives an exact T0 state.

- Causal plausibility: 1 = consequences contradict the specified action; 3 = mixed or uncertain consistency; 5 = consequences consistently follow the requested action under the shown context.
- Physical consistency: 1 = clear impossible motion or interactions; 3 = mixed consistency or substantial ambiguity; 5 = coherent motion and interactions throughout the shown horizon.
- Coverage: 1 = requested branches/queries or consequences are largely absent; 3 = partial coverage; 5 = all requested actions/queries and their relevant consequences are represented.

Use 2 and 4 for intermediate judgments. Do not guess a model identity. Leave an item blank if you cannot judge it and record your reason separately. Empty predictions are actual outputs to assess, not missing study files. Do not discuss scores with other raters before returning independent ratings. Consent/ethics and rater recruitment are managed by the researcher.
''')
    write_json(root/'private_key.json',dict(items=keys,n_raters=cfg['human']['raters'],scales=list(SCALES)))
    write_json(root/'package.json',dict(real_items=120,attention_candidates=12,calibration=30,holdout=90,
               attention_review='Researcher must approve all candidates and expectations before distributing the package.',
               thresholds='Scorer and dataset thresholds are frozen; no automated calibration or tuning occurs.'))


def l4_review_context(source, query):
    """Show the same public intervention assumptions as the model prompt."""
    from simulation.build_pilot import public_intervention_profiles
    observed = read_json(source/'observed_state.json')
    manifest = read_json(source/'input_manifest.json') if (source/'input_manifest.json').exists() else {}
    config = read_json(source/'scenario_config.json') if (source/'scenario_config.json').exists() else {}
    return {'Target-actor intervention contract': 'One control/treated pair, same root and ego_no_action policy; only the named target behaviour is manipulated. Ego states may differ through interactions. Check declarations and motion separately.',
            'Published intervention profiles': public_intervention_profiles(
                observed, query, config, manifest.get('split') or manifest.get('dataset'))}


def prepare_author_items(root, cfg, items):
    public=root/'public';keys={};sheet=[]
    for index,item in enumerate(items):
        iid=f'item_{index:04d}';keys[iid]=item
        source=Path(cfg['splits'][item['split']]['path'])/item['name']
        pred=Path(item['out_dir'])/item['name']
        query=read_json(source/'query.json')
        task=item['task']
        query={k:v for k,v in query.items() if (k=='l3_ego_actions' if task=='L3' else k.startswith('l4_'))}
        predictions=read_json(pred/('l3_pre.json' if task=='L3' else 'l4_pre.json')).get(
            'predicted_branches' if task=='L3' else 'predicted_counterfactuals',[])
        stimulus(public,iid,source,{'Task':TASK_LABELS[task],'Observed state':read_json(source/'observed_state.json'),
                                   'Query':query,'Prediction':predictions,
                                   **(l4_review_context(source,query) if task=='L4' else {})})
        sheet.append(dict(item_id=iid,task=TASK_LABELS[task],judgment='',issue_categories='',notes=''))
    csv_write(public/'ratings.csv',list(sheet[0]),sheet)
    instructions=('Author review only. Use judgment=acceptable/unacceptable/uncertain. Record action inconsistencies, motion anomalies or coverage issues in issue_categories (separate with |), and specific evidence in notes for every item. '
                  'Acceptable means no issue identified in this inspection, not verified real-world correctness. Empty predictions are outputs to inspect. '
                  'For target-actor intervention inspect both conditions, target response and direction, shared assumptions, and the declared fixed ego action; identical ego states are not required. '
                  'Model names and scores are omitted from these pages, but the author designed the benchmark and can access the key; this is not independent blinded validation. '
                  'Do not tune scoring or dataset thresholds from these judgments. Counts describe this stratified sample, not model-wide error rates.')
    package_index(public,list(keys),instructions)
    (public/'INSTRUCTIONS.md').write_text(instructions+'\n')
    write_json(root/'private_key.json',dict(items=keys))
    write_json(root/'package.json',dict(mode='author_review',real_items=len(items),attention_candidates=0,
               calibration=0,independent_validation='not_conducted'))


def analyze_author_review(filled,key):
    rows=read_rows(filled);items=key['items']
    if len(rows)!=len(items) or {r['item_id'] for r in rows}!=set(items):
        raise ValueError('Author sheet must contain each item ID exactly once')
    counts=defaultdict(int);groups=defaultdict(lambda:defaultdict(int));reviewed=[]
    for row in rows:
        judgment=row.get('judgment','').strip().lower()
        if judgment not in ('','acceptable','unacceptable','uncertain'):
            raise ValueError('Author judgment must be acceptable/unacceptable/uncertain')
        if not judgment or not row.get('notes','').strip(): continue
        item=items[row['item_id']];counts[judgment]+=1
        groups[f"{item['split']}/{item['task']}"][judgment]+=1
        reviewed.append(dict(row,task=item['task'],split=item['split']))
    return dict(status='reviewed_by_author' if len(reviewed)==len(items) else 'awaiting_author_review',
                reviewed=len(reviewed),total=len(items),counts=dict(counts),by_split_task=dict(groups),
                observations=reviewed,independent_validation='not_conducted',
                note='Single-author descriptive inspection; no independent reliability or metric-validity estimate.')


def rating_number(value):
    if value is None or str(value).strip()=='': return None
    try: number=float(value)
    except (ValueError,TypeError) as exc: raise ValueError(f'Invalid rating {value!r}') from exc
    if not math.isfinite(number) or number not in (1,2,3,4,5):
        raise ValueError('Ratings must be integer values 1..5')
    return number


def sample_interval(point,draws,n):
    vals=[v for v in draws if math.isfinite(v)]
    return dict(mean=point if math.isfinite(point) else None,n=n,
                lo=float(np.quantile(vals,.025)) if vals else None,
                hi=float(np.quantile(vals,.975)) if vals else None,defined_bootstrap_draws=len(vals))


def analyze_ctms(filled,key,review,n_boot=10000,seed=0,attention_fraction=.75):
    items=key['items']; rows=read_rows(filled)
    if len({r['item_id'] for r in rows})!=len(rows) or {r['item_id'] for r in rows}!=set(items):
        raise ValueError('Rating sheet must contain each blinded item ID exactly once')
    byid={r['item_id']:r for r in rows}; n_raters=key['n_raters']
    required=[f'rater{i+1}_{s}' for i in range(n_raters) for s in SCALES]
    values={iid:{c:rating_number(r.get(c)) for c in required} for iid,r in byid.items()}
    missing=sum(v is None for r in values.values() for v in r.values())
    if missing:
        return dict(status='awaiting_ratings',missing_ratings=missing,total_ratings=len(items)*len(required))
    attention={r['item_id']:r for r in read_rows(review)} if Path(review).exists() else {}
    expected={iid for iid,it in items.items() if it['is_attention']}
    if set(attention)!=expected or any(r.get('approved')!='y' or r.get('expectation') not in ('high','low') for r in attention.values()):
        return dict(status='awaiting_attention_review',reason='All attention candidates need researcher approval and high/low expectation')
    retained=[];rater_checks={}
    for r in range(n_raters):
        name=f'rater{r+1}';passed=0
        for iid,check in attention.items():
            rating=statistics.fmean(values[iid][f'{name}_{s}'] for s in SCALES)
            passed+=int(rating>3 if check['expectation']=='high' else rating<3)
        fraction=passed/len(attention);rater_checks[name]=dict(passed=passed,total=len(attention),fraction=fraction)
        if fraction>=attention_fraction: retained.append(name)
    if len(retained)<3:
        return dict(status='insufficient_retained_raters',retained=retained,attention=rater_checks,
                    reason='At least three raters must remain after the prespecified attention rule')
    held=[iid for iid,it in items.items() if not it['is_attention'] and not it['calibration']]
    means={iid:statistics.fmean(values[iid][f'{r}_{s}'] for r in retained for s in SCALES) for iid in held}
    rng=random.Random(seed); correlations={}
    for task in ('L3','L4'):
        ids=[i for i in held if items[i]['task']==task]
        x=[items[i]['score'] for i in ids];y=[means[i] for i in ids]
        draws=[]
        for _ in range(n_boot):
            ix=rng.choices(range(len(ids)),k=len(ids));draws.append(spearman([x[i] for i in ix],[y[i] for i in ix]))
        correlations[task]=sample_interval(spearman(x,y),draws,len(ids))
    def alpha(ids):
        units=[[values[iid][f'{r}_{s}'] for r in retained] for iid in ids for s in SCALES]
        # All-identical ratings have no disagreement variance; report undefined.
        if len({v for row in units for v in row})<2:return float('nan')
        return krippendorff_alpha(units)
    draws=[alpha(rng.choices(held,k=len(held))) for _ in range(n_boot)]
    return dict(status='analyzed',n_holdout=len(held),n_calibration=sum(it['calibration'] for it in items.values()),
                retained=retained,attention=rater_checks,attention_pass_fraction=attention_fraction,
                spearman=correlations,alpha_nominal=sample_interval(alpha(held),draws,len(held)),
                note='Item bootstrap preserves raters and scales; task correlations use held-out items only. Thresholds are not tuned.')


def analyze_packages(cfg):
    root=Path(cfg['campaign_dir'])/'human';statuses={}
    author = cfg['human'].get('mode') == 'author_review'
    if author:
        statuses.update(mode='author_review',independent_validation='not_conducted',
                        reason='Ethics approval for the external-rater study was not obtained.')
    for kind in ('branch_audit','author_review' if author else 'ctms'):
        p=root/kind;filled=p/'filled.csv'
        if not filled.exists():
            statuses[kind]={'status':'awaiting_author_review' if author else 'awaiting_ratings','expected_file':str(filled)};continue
        if kind=='author_review':
            result=analyze_author_review(filled,read_json(p/'private_key.json'))
        elif kind=='ctms':
            result=analyze_ctms(filled,read_json(p/'private_key.json'),p/'attention_review.csv',
                                cfg['statistics']['n_boot'],cfg['statistics']['seed'],cfg['human']['attention_pass_fraction'])
        else:
            from analysis.branch_audit import AUDIT_VERSION, summarize_author_audit
            package=read_json(p/'package.json')
            if package.get('rubric_version') != AUDIT_VERSION:
                raise ValueError('Outdated branch audit rubric; rebuild the v2 package')
            if package.get('ratings_sha256') != sha(p/'public/ratings.csv'):
                raise ValueError('Branch audit template changed; fill a copy as filled.csv')
            rows=read_rows(filled);template=read_rows(p/'public/ratings.csv')
            key=lambda r:(r['scenario_id'],r['split'],r['item_type'],r['item'])
            if len(rows)!=len(template) or {key(r) for r in rows}!={key(r) for r in template}:
                raise ValueError('Branch audit item membership differs from template')
            columns=[f'rater{i+1}' for i in range(cfg['human']['raters'])]
            vals=[r.get(c,'').strip().lower() for r in rows for c in columns]
            allowed=('y','n','u','') if author else ('y','n','')
            if any(v not in allowed for v in vals): raise ValueError('Invalid branch audit judgment')
            if author:
                result=summarize_author_audit(rows)
                result['automatic_protocol_checks']=read_json(p/'public/automatic_checks.json')['counts']
            elif any(not v for v in vals): result=dict(status='awaiting_ratings',missing_ratings=vals.count(''))
            else: result=dict(status='analyzed',**audit_summary(str(filled)))
        write_json(p/'analysis.json',result);statuses[kind]=result
    write_json(root/'status.json',statuses)
