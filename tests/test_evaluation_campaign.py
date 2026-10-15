"""Campaign regression tests use temporary synthetic data and no GPU/network."""
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


@pytest.mark.skip(reason="needs the cluster campaign configuration, which is not part of this release")
def test_author_config_needs_no_external_raters(tmp_path):
    import yaml
    from analysis.evaluation_campaign import load_config
    cfg=yaml.safe_load(Path('evaluation/config.example.yaml').read_text())
    cfg['human'].update(mode='author_review',raters=1,attention=0,calibration=0)
    p=tmp_path/'config.yaml';p.write_text(yaml.safe_dump(cfg))
    assert load_config(p)['human']['mode']=='author_review'
    cfg['human']['raters']=3;p.write_text(yaml.safe_dump(cfg))
    with pytest.raises(ValueError,match='one author'):
        load_config(p)


def test_author_review_is_descriptive_and_requires_notes(tmp_path):
    from analysis.evaluation_human import analyze_author_review,csv_write
    key={'items':{'a':{'task':'L3','split':'carla'},'b':{'task':'L4','split':'nuscenes'}}}
    rows=[dict(item_id='a',task='L3',judgment='acceptable',notes=''),
          dict(item_id='b',task='L4',judgment='uncertain',notes='Target is occluded.')]
    p=tmp_path/'filled.csv';csv_write(p,list(rows[0]),rows)
    assert analyze_author_review(p,key)['status']=='awaiting_author_review'
    rows[0]['notes']='Both queried branches are present.';csv_write(p,list(rows[0]),rows)
    out=analyze_author_review(p,key)
    assert out['status']=='reviewed_by_author' and out['reviewed']==2
    assert out['independent_validation']=='not_conducted'
    assert 'alpha' not in str(out) and 'spearman' not in str(out)
    rows[1]['item_id']='a';csv_write(p,list(rows[0]),rows)
    with pytest.raises(ValueError,match='exactly once'):
        analyze_author_review(p,key)


def test_expected_matrix_includes_both_splits_and_all_conditions():
    from analysis.evaluation_campaign import matrix_cells
    cells = matrix_cells({'splits': {'carla': {}, 'nuscenes': {}}, 'campaign_dir': '/tmp/run'})
    assert len(cells) == 30
    assert sum(c['backend'] == 'local' for c in cells) == 24
    assert sum(c['backend'] == 'gemini' for c in cells) == 6
    assert len({c['out_dir'] for c in cells}) == 30


def test_completeness_does_not_accept_an_empty_matrix(tmp_path):
    from analysis.evaluation_campaign import check_cell
    result = check_cell(tmp_path, [{'name': 'scenario_1', 'scenario_id': 'ns_1'}])
    assert result['complete'] is False
    assert result['missing_scenarios'] == ['scenario_1']


def test_missing_l4_is_incomplete_even_with_l3_and_old_report(tmp_path):
    from analysis.evaluation_campaign import check_cell
    d = tmp_path / 'scenario_1'; d.mkdir()
    (d/'l3_pre.json').write_text('{}')
    (tmp_path/'report.json').write_text('{}')
    result = check_cell(tmp_path, [{'name': 'scenario_1', 'scenario_id': 'ns_1'}])
    assert not result['complete']
    assert any('l4_pre.json' in e for e in result['errors'])


def test_freeze_rejects_dataset_or_config_drift(tmp_path):
    from analysis.evaluation_campaign import freeze
    p = tmp_path/'freeze.json'
    freeze(p, {'dataset': 'a'})
    freeze(p, {'dataset': 'a'})
    with pytest.raises(ValueError, match='changed'):
        freeze(p, {'dataset': 'b'})
    assert json.loads(p.read_text()) == {'dataset': 'a'}


def test_paired_stat_uses_intersection_and_preserves_undefined():
    from analysis.evaluation_report import paired_stat
    out = paired_stat({'s1': 0.8, 's2': None}, {'s1': 0.3, 's3': 0.9}, 100, 0)
    assert out['n'] == 1
    assert out['mean'] == pytest.approx(0.5)
    assert paired_stat({'s1': None}, {'s1': 0.2}, 100, 0)['mean'] is None


def test_cross_split_excess_is_independently_resampled():
    from analysis.evaluation_report import excess_stat
    out = excess_stat({'a': 0.8}, {'b': 0.4}, {'a': 0.2}, {'b': 0.1}, 100, 0)
    assert out['mean'] == pytest.approx(0.3)
    assert out['n_nuscenes'] == out['n_carla'] == 1
    empty = excess_stat({'a': 0.8}, {'b': 0.4}, {'x': 0.2}, {'b': 0.1}, 100, 0)
    assert empty['mean'] is None


def test_slurm_failure_does_not_create_fake_job_id(tmp_path, monkeypatch):
    from analysis.evaluation_campaign import submit_job
    import subprocess
    def fail(*args, **kwargs):
        raise subprocess.CalledProcessError(1, args[0], stderr='invalid partition')
    monkeypatch.setattr(subprocess, 'run', fail)
    cfg = {'campaign_dir': str(tmp_path), 'python': sys.executable, '_path': '/tmp/cfg',
           'slurm': {'cpu_partition': 'work', 'cpu_memory': '2G', 'cpu_time': '1:00:00', 'cpus': 1}}
    with pytest.raises(subprocess.CalledProcessError):
        submit_job(cfg, 'prepare')
    assert not (tmp_path/'jobs.jsonl').exists()


def make_protocol_dataset(tmp_path, split='nuscenes', n=2):
    import copy
    import yaml
    from analysis.rq3a import make_synthetic_gts
    from simulation.protocol import protocol_manifest, HISTORY_TIMESTAMPS
    from simulation.observed_history import write_observed_artifacts
    base=tmp_path/split
    for i,gt in enumerate(make_synthetic_gts(n),1):
        d=base/f'scenario_{i}';(d/'frames').mkdir(parents=True)
        gt['scenario_id']=f'{split}_{i:04d}'
        gt['source_scene_name']=f'{split}-scene-{i}'
        gt['source_sample_token']=f'{split}-sample-{i}'
        gt['factual_fidelity_error']=0.1*i
        generator={}
        if split=='nuscenes':
            # what the nuScenes builder records about its reference generator
            from simulation.nuscenes_adapter import SCENARIO_TEMPLATE_POLICY
            from simulation.kinematic_twin import TWIN_VERSION
            generator=dict(scenario_template_policy=SCENARIO_TEMPLATE_POLICY,twin_version=TWIN_VERSION,
                           build_code_commit='f'*40,build_worktree_clean=True)
            gt['scenario_template']={'policy':SCENARIO_TEMPLATE_POLICY,'selected_actor_id':'npc_1'}
        manifest=protocol_manifest(split='v1.0-trainval' if split=='nuscenes' else split,**generator)
        gt['protocol']=manifest
        root=gt['observed_state']
        for a in root['actors']:a.update(length=4.5,width=2.0)
        # Exercise the current campaign contract all the way through report.
        # Old synthetic ego counterfactuals remain useful for standalone
        # scorer tests, but are not an Option-B named-actor query.
        from simulation.kinematic_twin import build_twin_branch
        from simulation.nuscenes_admission import observed_velocity_baseline
        from simulation.l4_control import control_query_entry, mark_pair_protocol
        iv=dict(intervention_type='npc_sudden_brake',target_actor_id='npc_1',trigger_depth=1)
        kw=dict(model='point_mass',base_steer=0.0,env_behavior=observed_velocity_baseline(root),
                first_interval='label_first')
        treated=build_twin_branch('T',False,iv['intervention_type'],root,6,.5,.5,
                                  intervention_meta=iv,environment_intervention=iv,**kw)
        control=build_twin_branch('C',False,'do_ego_no_action',root,6,.5,.5,**kw)
        gt.pop('counterfactuals',None)
        gt['l4_queries']=[dict(counterfactual_chain=dict(intervention=iv,chain=treated['chain']),
                               control_chain=control_query_entry(iv,control['chain'],'carla' if split=='carla' else 'nuscenes'))]
        mark_pair_protocol(manifest,'carla' if split=='carla' else 'nuscenes')
        records=[]
        for t in HISTORY_TIMESTAMPS:
            actors=copy.deepcopy(root['actors'])
            for a in actors:a['visibility']=True
            records.append({'timestamp_sec':t,'actors':actors})
        write_observed_artifacts(str(d),root,{'records':records})
        query={'l3_ego_actions':[b['intervention']['intervention_type'] for b in gt['branches']],
               'l4_interventions':[{'intervention_type':'npc_sudden_brake','target_actor_id':'npc_1','trigger_depth':1}],
               'l4_ego_policy':'do_ego_no_action'}
        (d/'ground_truth.json').write_text(json.dumps(gt))
        (d/'input_manifest.json').write_text(json.dumps(manifest))
        (d/'query.json').write_text(json.dumps(query))
        if split=='carla':
            (d/'scenario_config.json').write_text(json.dumps({'env_behavior':{'npc_1':{'accel':-8.0,'lat':0.0}}}))
        (d/'ontology.json').write_text(json.dumps(yaml.safe_load(Path('ctms/ontology.yaml').read_text())))
        from PIL import Image
        for j in range(8):Image.new('RGB',(8,8)).save(d/'frames'/f'{j:03d}.jpg')
    return base


def test_dataset_inventory_and_repeated_source_detection(tmp_path):
    from analysis.evaluation_campaign import dataset_inventory
    base=make_protocol_dataset(tmp_path)
    rows=dataset_inventory(base,2,'nuscenes')
    assert len(rows)==2 and len(rows[0]['files'])==14
    gt=json.loads((base/'scenario_2/ground_truth.json').read_text())
    gt['source_scene_name']='nuscenes-scene-1'
    (base/'scenario_2/ground_truth.json').write_text(json.dumps(gt))
    with pytest.raises(ValueError,match='multiple roots'):
        dataset_inventory(base,2,'nuscenes')


def test_dataset_inventory_requires_the_current_reference_generator(tmp_path):
    from analysis.evaluation_campaign import dataset_inventory
    from simulation.nuscenes_adapter import SCENARIO_TEMPLATE_POLICY
    from simulation.kinematic_twin import TWIN_VERSION
    base=make_protocol_dataset(tmp_path)
    rows=dataset_inventory(base,2,'nuscenes')
    assert rows[0]['generator']=={'scenario_template_policy':SCENARIO_TEMPLATE_POLICY,'twin_version':TWIN_VERSION,
                                  'build_code_commit':'f'*40,'build_worktree_clean':True}
    d=base/'scenario_1'
    def rewrite(name,fn):
        doc=json.loads((d/name).read_text());fn(doc);(d/name).write_text(json.dumps(doc))
    def refused(pattern):
        with pytest.raises(ValueError,match=pattern):dataset_inventory(base,2,'nuscenes')
    # a build under another template policy, or one that never recorded a policy, is not admitted
    rewrite('input_manifest.json',lambda m:m.__setitem__('scenario_template_policy','interaction-v2'));refused('scenario_template_policy')
    rewrite('input_manifest.json',lambda m:m.pop('scenario_template_policy'));refused('scenario_template_policy')
    rewrite('input_manifest.json',lambda m:m.__setitem__('scenario_template_policy',SCENARIO_TEMPLATE_POLICY))
    # GT protocol, GT candidate table and input manifest must all agree
    rewrite('ground_truth.json',lambda g:g['protocol'].__setitem__('scenario_template_policy','interaction-v2'));refused('scenario_template_policy')
    rewrite('ground_truth.json',lambda g:g['protocol'].__setitem__('scenario_template_policy',SCENARIO_TEMPLATE_POLICY))
    rewrite('ground_truth.json',lambda g:g['scenario_template'].__setitem__('policy','interaction-v2'));refused('scenario_template_policy')
    rewrite('ground_truth.json',lambda g:g['scenario_template'].__setitem__('policy',SCENARIO_TEMPLATE_POLICY))
    # references rolled by the previous twin semantics are refused as well
    rewrite('ground_truth.json',lambda g:g['protocol'].__setitem__('twin_version','kinematic-twin-1'));refused('twin_version')
    rewrite('ground_truth.json',lambda g:g['protocol'].pop('twin_version'));refused('twin_version')
    rewrite('ground_truth.json',lambda g:g['protocol'].__setitem__('twin_version',TWIN_VERSION))
    # the build commit is verified, not merely copied: it must be recorded (a
    # missing commit in both artifacts is not agreement), be a full SHA, agree
    # between GT and manifest, and come from a clean worktree in both
    rewrite('input_manifest.json',lambda m:m.__setitem__('build_code_commit','a'*40));refused('build_code_commit')
    rewrite('input_manifest.json',lambda m:m.__setitem__('build_code_commit','f'*40))
    def both(fn):
        rewrite('input_manifest.json',fn);rewrite('ground_truth.json',lambda g:fn(g['protocol']))
    both(lambda m:m.pop('build_code_commit'));refused('build_code_commit')
    both(lambda m:m.__setitem__('build_code_commit',None));refused('build_code_commit')
    both(lambda m:m.__setitem__('build_code_commit','abc123'));refused('build_code_commit')
    both(lambda m:m.__setitem__('build_code_commit','f'*40))
    both(lambda m:m.__setitem__('build_worktree_clean',False));refused('build_worktree_clean')
    both(lambda m:m.pop('build_worktree_clean'));refused('build_worktree_clean')
    both(lambda m:m.__setitem__('build_worktree_clean',True))
    rewrite('ground_truth.json',lambda g:g['protocol'].__setitem__('build_worktree_clean',False));refused('build_worktree_clean')
    rewrite('ground_truth.json',lambda g:g['protocol'].__setitem__('build_worktree_clean',True))
    assert len(dataset_inventory(base,2,'nuscenes'))==2
    # one dataset comes from one build: scenarios built at different commits are refused
    d=base/'scenario_2'
    both(lambda m:m.__setitem__('build_code_commit','e'*40));refused('single build')
    both(lambda m:m.__setitem__('build_code_commit','f'*40))
    assert len(dataset_inventory(base,2,'nuscenes'))==2
    # CARLA references come from the simulator, not this generator
    carla=make_protocol_dataset(tmp_path,split='carla')
    assert dataset_inventory(carla,2,'carla')[0]['generator'] is None


def test_oracle_score_and_statistics_real_pipeline(tmp_path):
    from analysis.evaluation_campaign import dataset_inventory,score_cell,write_json,write_campaign_identity
    from analysis.run_model import run_model
    from analysis.evaluation_report import collect_cell,mean_stat
    base=make_protocol_dataset(tmp_path)
    rows=dataset_inventory(base,2,'nuscenes')
    out=tmp_path/'run';out.mkdir();write_json(out/'frozen.json',{})
    cell=dict(split='nuscenes',backend='oracle',condition='video',out_dir=str(out/'oracle'))
    cfg={'campaign_dir':str(out),'splits':{'nuscenes':{'path':str(base)}}}
    write_campaign_identity(cfg,cell)
    run_model(str(base),cell['out_dir'],backend='oracle')
    score_cell(cfg,cell,{'datasets':{'nuscenes':rows}})
    data=collect_cell(cell,rows)
    assert set(data['metrics']['CTMS'])=={'scenario_1','scenario_2'}
    assert mean_stat(list(data['metrics']['CTMS'].values()),100)['n']==2
    p=Path(cell['out_dir'])/'scenario_1/l4_pre.json';p.write_text(p.read_text()+' ')
    with pytest.raises(ValueError,match='changed after scoring'):
        collect_cell(cell,rows)


def test_submission_dependency_graph(monkeypatch,tmp_path):
    import analysis.evaluation_campaign as c
    calls=[]
    def submit(cfg,task,**kwargs):
        jid=str(len(calls)+1);calls.append((jid,task,kwargs));return jid
    monkeypatch.setattr(c,'submit_job',submit)
    c.submit_step({'campaign_dir':str(tmp_path)},'evaluate',dry_run=True)
    assert [v[2].get('count') for v in calls[:2]]==[24,6]
    score=next(v for v in calls if v[1]=='score')
    assert score[2]['dependencies']==['1','2','3']
    report=next(v for v in calls if v[1]=='report')
    assert report[2]['dependencies']==[score[0],'4','5']


def test_probe_resume_does_not_repeat_calls(tmp_path):
    from analysis.evaluation_campaign import dataset_inventory
    from analysis.evaluation_probe import run_panel_probe
    from analysis.backends.result import GenerationResult
    base=make_protocol_dataset(tmp_path);rows=dataset_inventory(base,2,'nuscenes');calls=[]
    def gen(prompt,frames,model,opts):
        calls.append(prompt)
        assert 'nuscenes_0001' not in prompt and len(frames)==8
        text='{"dataset":"nuScenes","city":"Singapore","recognized_scene":true}'
        return GenerationResult(raw_text=text,final_text=text,metadata={})
    cfg={'campaign_dir':str(tmp_path/'run'),'statistics':{'seed':0},'probe':{'limit':2},
         'splits':{'nuscenes':{'path':str(base)}}}
    model={'backend':'local','model_id':'test','revision':'abc','name':'test'}
    frozen={'datasets':{'nuscenes':rows}}
    first=run_panel_probe(cfg,'test',frozen,generator=gen,resolved_model=model)
    second=run_panel_probe(cfg,'test',frozen,generator=gen,resolved_model=model)
    assert first==second and len(calls)==2
    assert 'unverified self-report' in first['criterion']


def test_human_stratification_and_unique_scenarios():
    from analysis.evaluation_human import stratified_items
    pool=[dict(task=task,split=split,model=model,name=f'scenario_{i}',score=i/200)
          for task in ('L3','L4') for split in ('carla','nuscenes') for model in ('a','b') for i in range(200)]
    selected=stratified_items(pool)
    from collections import Counter
    counts=Counter((r['task'],r['split'],r['tertile']) for r in selected)
    assert set(counts.values())=={10}
    assert len({(r['split'],r['name']) for r in selected})==120


def test_blank_human_sheet_cannot_report_perfect_agreement(tmp_path):
    from analysis.evaluation_human import analyze_ctms,csv_write
    p=tmp_path/'ratings.csv';csv_write(p,['item_id','rater1_causal_plausibility'],[{'item_id':'x'}])
    key={'items':{'x':{'is_attention':False,'calibration':False}},'n_raters':3}
    result=analyze_ctms(p,key,tmp_path/'missing_review.csv',100)
    assert result['status']=='awaiting_ratings'
    assert 'alpha_nominal' not in result


def test_full_noise_floor_reports_ctms_not_just_f1():
    from analysis.evaluation_validation import full_noise_floor
    from analysis.rq3a import make_synthetic_gts
    from ctms.ctms import load_ctms_environment
    env=load_ctms_environment('ctms/ontology.yaml','ctms/ctms_config.yaml','ctms/schema_branch.json','ctms/schema_counterfactual.json')
    gts=make_synthetic_gts(1)
    predictions=[dict(branches=g['branches'],predicted_counterfactuals=g['counterfactuals']) for g in gts]
    result=full_noise_floor(gts,predictions,[0.0],env,n_trials=2)
    assert set(result['axes'])=={'Causal_Foresight_F1','Counterfactual_Alignment_Score','CTMS'}
    assert result['axes']['CTMS']['delta_noise']==0


@pytest.mark.skip(reason="needs the cluster campaign configuration, which is not part of this release")
@pytest.mark.parametrize('smoke_source', ['separate', 'benchmark', 'benchmark_copy'])
def test_prepare_freezes_inputs_and_builds_portable_audit(tmp_path, smoke_source):
    import shutil
    import analysis.evaluation_campaign as c
    cfg={'campaign_dir':str(tmp_path/'run'),'_path':str(tmp_path/'cfg.yaml'),
         'splits':{s:{'path':str(make_protocol_dataset(tmp_path,s)), 'expected':2} for s in ('carla','nuscenes')},
         'smoke':{'path':str(make_protocol_dataset(tmp_path,'smoke',1)),'scenario':'scenario_1'},
         'statistics':{'n_boot':100,'seed':0},'probe':{'limit':2},
         'human':{'raters':3,'audit_per_split':1},'validation':{}}
    if smoke_source == 'benchmark':
        cfg['smoke']['path'] = cfg['splits']['nuscenes']['path']
    elif smoke_source == 'benchmark_copy':
        cfg['smoke']['path'] = str(tmp_path/'smoke_copy')
        shutil.copytree(Path(cfg['splits']['nuscenes']['path'])/'scenario_1',
                        Path(cfg['smoke']['path'])/'scenario_1')
    c.prepare(cfg);c.prepare(cfg)
    frozen=c.frozen_inputs(cfg,verify_files=True)
    assert len(frozen['datasets']['carla'])==2
    assert len(frozen['datasets']['nuscenes'])==2
    if smoke_source != 'separate':
        assert frozen['smoke_fingerprint'] == frozen['datasets']['nuscenes'][0]['fingerprint']
    assert (tmp_path/'run/human/branch_audit/public/index.html').exists()
    root=Path(cfg['splits']['carla']['path'])/'scenario_1/observed_state.json'
    root.write_text(root.read_text()+' ')
    with pytest.raises(ValueError,match='Frozen dataset changed'):
        c.frozen_inputs(cfg,verify_files=True)


@pytest.mark.skip(reason="needs the cluster campaign configuration, which is not part of this release")
def test_full_matrix_report_exports_all_comparisons(tmp_path,monkeypatch):
    """Two synthetic scenarios per split, all 30 cells; real scoring, no inference."""
    import shutil
    import analysis.evaluation_campaign as c
    from analysis.evaluation_report import build_report
    from analysis.run_model import run_model
    from analysis.model_panel import load_model_panel,resolve_configurations
    cfg={'campaign_dir':str(tmp_path/'run'),'splits':{},'statistics':{'n_boot':100,'seed':0}}
    frozen={'datasets':{}}
    root=Path(cfg['campaign_dir']);root.mkdir();c.write_json(root/'frozen.json',{})
    for split in ('carla','nuscenes'):
        base=make_protocol_dataset(tmp_path,split)
        cfg['splits'][split]={'path':str(base),'expected':2}
        frozen['datasets'][split]=c.dataset_inventory(base,2,split)
        source=root/'source'/split
        cell={'split':split,'condition':'video','backend':'oracle','out_dir':str(source)}
        c.write_campaign_identity(cfg,cell)
        run_model(str(base),str(source),backend='oracle')
        c.score_cell(cfg,cell,frozen)
    panel=load_model_panel(str(c.PANEL))
    models={m['name']:m for m in resolve_configurations(panel,[m['name'] for m in panel['configurations']])}
    for cell in c.matrix_cells(cfg)+c.baseline_cells(cfg):
        dst=Path(cell['out_dir']);shutil.copytree(root/'source'/cell['split'],dst)
        model=models.get(cell.get('configuration'))
        for row in frozen['datasets'][cell['split']]:
            d=dst/row['name'];manifest=c.read_json(d/'run_manifest.json')
            manifest.update(backend=cell['backend'],condition=cell['condition'],configuration_name=cell.get('configuration'))
            if cell['condition']=='history_json':
                manifest['frame_sha256']=[]
                manifest['frame_count']=0
            if model:
                response='{"predicted_branches": [], "predicted_counterfactuals": []}'
                (d/'raw_response.txt').write_text(response);(d/'final_response.txt').write_text(response)
                c.write_json(d/'generation_metadata.json',{'output_tokens':10,'latency_seconds':1.0})
                manifest.update(raw_response_sha256=c.sha(d/'raw_response.txt'),final_response_sha256=c.sha(d/'final_response.txt'),
                                generation_metadata={'output_tokens':10,'latency_seconds':1.0})
            c.write_json(d/'run_manifest.json',manifest)
        proof=c.read_json(dst/'score_provenance.json')
        proof['prediction_files']={f:c.sha(dst/f) for f in proof['prediction_files']}
        c.write_json(dst/'score_provenance.json',proof)
    build_report(cfg,frozen)
    result=c.read_json(root/'report/results.json')
    assert len(result['cells'])==42  # 30 matrix + 6 baselines x 2 splits
    assert {'scores','posttraining_effect','history_added','images_added','representation_difference','contamination_excess','scores_common_subset'} <= {r['comparison'] for r in result['rows']}
    assert (root/'report/tables.csv').exists() and (root/'report/scores.tex').exists()
    assert any(r['mean'] is None for r in result['rows'])
    # Five coherence axes, the action-profile control,
    # collapse/shortfall/floor rates, and a common-subset ranking per axis.
    # (Every cell here is a copy of the oracle scores, so rates are 0.)
    assert set(result['primary_axes'])=={'monotonicity','responsiveness','locality','action_direction','speed_response'}
    assert 'carla/action_profile_kinematic/video' in result['cells']
    scores={(r['split'],r['configuration'],r['condition'],r['metric']):r for r in result['rows'] if r['comparison']=='scores'}
    for metric in ('collapse_rate','endpoint_shortfall_rate','branch_shortfall_rate','Causal_Foresight_F1_floor_rate','CTMS_floor_rate','action_direction','speed_response'):
        assert (('carla','oracle','video',metric) in scores), metric
    assert scores[('carla','oracle','video','collapse_rate')]['mean']==0.0
    assert scores[('carla','oracle','video','branch_shortfall_rate')]['mean']==0.0
    assert scores[('carla','oracle','video','Causal_Foresight_F1_floor_rate')]['mean']==0.0
    common=[r for r in result['rows'] if r['comparison']=='scores_common_subset' and r['split']=='carla' and r['condition']=='video' and r['metric']=='responsiveness']
    assert len(common)==5 and len({r['n_common'] for r in common})==1 and common[0]['n_common']==2
    assert any(r['metric']=='responsiveness/action_profile_kinematic' for r in result['rows'] if r['comparison']=='contamination_excess')
    # Floor-relative rows feed tab:floor-coherence / tab:floor-reference: every
    # model, split and condition gets model-minus-control paired deltas on every
    # axis, and a skill score on the reference axes. All cells copy the oracle
    # here, so deltas are 0 and the skill denominator is degenerate (None+reason).
    floor=[r for r in result['rows'] if r['comparison']=='floor_relative']
    assert {(r['split'],r['configuration'],r['condition']) for r in floor}=={(s,m,c) for s in ('carla','nuscenes') for m in model_names_in(result) for c in ('video','history_json','video_history')}
    fr=[r for r in floor if r['split']=='carla' and r['condition']=='video' and r['metric']=='responsiveness'][0]
    assert fr['n']==2 and fr['mean']==0.0 and fr['baseline']=='action_profile_kinematic'
    skill=[r for r in result['rows'] if r['comparison']=='skill_vs_floor']
    assert {r['metric'] for r in skill}=={'Causal_Foresight_F1','Counterfactual_Alignment_Score','CTMS'}
    assert all(r['mean'] is None and r.get('reason') for r in skill)
    assert (root/'report/floor_relative.tex').exists()


def model_names_in(result):
    return {r['configuration'] for r in result['rows'] if r['comparison']=='scores' and r['configuration'] not in
            ('oracle','copy_factual','kinematic_extrapolation','action_profile_kinematic','action_profile_published','random_valid')}


def test_skill_stat_is_ratio_of_paired_means_with_degenerate_guard():
    from analysis.evaluation_report import skill_stat
    model={'a':0.5,'b':0.7,'c':None}; floor={'a':0.2,'b':0.2,'c':0.2}; oracle={'a':1.0,'b':1.0,'c':1.0}
    out=skill_stat(model,floor,oracle,n_boot=200,seed=0)
    assert out['n']==2 and abs(out['mean']-0.5)<1e-9 and out['lo']<=out['mean']<=out['hi']
    degenerate=skill_stat(model,oracle,oracle,n_boot=200,seed=0)
    assert degenerate['mean'] is None and degenerate['reason']


def test_collect_cell_reports_collapse_shortfall_and_floor_rates(tmp_path):
    """A single-branch control collapses and falls short of the declared branch
    count; the action-profile control does neither."""
    import analysis.evaluation_campaign as c
    from analysis.evaluation_report import collect_cell
    from analysis.run_model import run_model
    cfg={'campaign_dir':str(tmp_path/'run'),'splits':{},'statistics':{'n_boot':100,'seed':0}}
    root=Path(cfg['campaign_dir']);root.mkdir();c.write_json(root/'frozen.json',{})
    base=make_protocol_dataset(tmp_path,'carla')
    cfg['splits']['carla']={'path':str(base),'expected':2}
    inventory=c.dataset_inventory(base,2,'carla')
    frozen={'datasets':{'carla':inventory}}
    out={}
    for backend in ('kinematic_extrapolation','action_profile_kinematic'):
        cell={'split':'carla','condition':'video','backend':backend,'out_dir':str(root/'baselines/carla'/backend)}
        c.write_campaign_identity(cfg,cell)
        run_model(str(base),cell['out_dir'],backend=backend)
        c.score_cell(cfg,cell,frozen)
        out[backend]=collect_cell(cell,inventory)['metrics']
    single=out['kinematic_extrapolation']; profile=out['action_profile_kinematic']
    # one branch: too few endpoints to compare (shortfall), not coincident endpoints (collapse)
    assert set(single['endpoint_shortfall_rate'].values())=={1} and set(single['branch_shortfall_rate'].values())=={1}
    assert all(v is None for v in single['collapse_rate'].values())
    assert set(profile['collapse_rate'].values())=={0} and set(profile['endpoint_shortfall_rate'].values())=={0}
    assert set(profile['branch_shortfall_rate'].values())=={0}
    assert set(profile['Causal_Foresight_F1_floor_rate'].values())=={0}
    assert all(v is not None for v in profile['action_direction'].values())
    assert all(v is not None for v in profile['speed_response'].values())


def test_human_heldout_analysis_does_not_include_calibration(tmp_path):
    from analysis.evaluation_human import analyze_ctms,csv_write,SCALES
    items={};rows=[];review=[]
    for i in range(12):
        iid=f'i{i}';attention=i>=8;cal=i<2
        score=(i%4)+1
        items[iid]=dict(task='L3' if i%2==0 else 'L4',score=score,is_attention=attention,calibration=cal)
        rows.append(dict(item_id=iid,**{f'rater{r}_{s}':score for r in (1,2,3) for s in SCALES}))
        if attention:
            expectation='high' if i%2==0 else 'low'
            for r in (1,2,3):
                for scale in SCALES: rows[-1][f'rater{r}_{scale}']=5 if expectation=='high' else 1
            review.append(dict(item_id=iid,approved='y',expectation=expectation))
    csv_write(tmp_path/'filled.csv',list(rows[0]),rows)
    csv_write(tmp_path/'review.csv',list(review[0]),review)
    result=analyze_ctms(tmp_path/'filled.csv',{'items':items,'n_raters':3},tmp_path/'review.csv',100)
    assert result['status']=='analyzed' and result['n_holdout']==6 and result['n_calibration']==2
    assert result['spearman']['L3']['n']==3 and result['spearman']['L3']['mean']==pytest.approx(1.0)


def test_human_stimulus_does_not_leak_model_or_score(tmp_path):
    from analysis.evaluation_human import stimulus
    base=make_protocol_dataset(tmp_path,n=1)
    stimulus(tmp_path/'public','item_0000',base/'scenario_1',{'Prediction':{'model_id':'secret-model','scenario_id':'nuscenes_0001','predicted_branches':[]}})
    page=(tmp_path/'public/item_0000/index.html').read_text()
    assert 'secret-model' not in page and 'nuscenes_0001' not in page
    assert len(list((tmp_path/'public/item_0000').glob('frame*.jpg')))==8


@pytest.mark.parametrize('author', [False, True])
def test_complete_rating_package_blinds_and_separates_holdout(tmp_path, author):
    from analysis.evaluation_campaign import dataset_inventory,matrix_cells,write_json
    from analysis.evaluation_human import prepare_rating_package,read_rows
    cfg={'campaign_dir':str(tmp_path/'run'),'statistics':{'seed':0},
         'human':{'items':120,'attention':12,'calibration':30,'raters':3},'splits':{}}
    if author: cfg['human'].update(mode='author_review',raters=1,attention=0,calibration=0)
    root=Path(cfg['campaign_dir']);write_json(root/'frozen.json',{})
    frozen={'datasets':{}}
    for split in ('carla','nuscenes'):
        base=make_protocol_dataset(tmp_path,split,n=100)
        cfg['splits'][split]={'path':str(base)}
        frozen['datasets'][split]=dataset_inventory(base,100,split)
    for cell in matrix_cells(cfg):
        if cell['condition']!='video':continue
        samples=[dict(scenario_id=row['scenario_id'],scores={'Causal_Foresight_F1':i/100,'Counterfactual_Alignment_Score':(99-i)/100})
                 for i,row in enumerate(frozen['datasets'][cell['split']])]
        write_json(Path(cell['out_dir'])/'report.json',dict(per_sample=samples))
    for split in cfg['splits']:
        for model in [c['configuration'] for c in matrix_cells(cfg) if c['split']==split and c['condition']=='video']+['oracle','random_valid']:
            target=root/'matrix'/split/f'{model}_video_seed0' if model not in ('oracle','random_valid') else root/'baselines'/split/model
            for row in frozen['datasets'][split]:
                write_json(target/row['name']/'l3_pre.json',{'predicted_branches':[]})
                write_json(target/row['name']/'l4_pre.json',{'predicted_counterfactuals':[]})
    prepare_rating_package(cfg,frozen)
    if author:
        from analysis.evaluation_human import analyze_packages,csv_write
        folder=root/'human/author_review'
        sheet=read_rows(folder/'public/ratings.csv')
        assert len(sheet)==120 and set(sheet[0])=={'item_id','task','judgment','issue_categories','notes'}
        assert not (root/'human/ctms').exists() and not (folder/'attention_review.csv').exists()
        key=json.loads((folder/'private_key.json').read_text())['items']
        assert len({(it['split'],it['name']) for it in key.values()})==120
        analyze_packages(cfg)
        assert json.loads((root/'human/status.json').read_text())['author_review']['status']=='awaiting_author_review'
        for row in sheet:row.update(judgment='uncertain',notes='Synthetic test only.')
        csv_write(folder/'filled.csv',list(sheet[0]),sheet)
        analyze_packages(cfg)
        result=json.loads((folder/'analysis.json').read_text())
        assert result['reviewed']==120 and result['independent_validation']=='not_conducted'
        prepare_rating_package(cfg,frozen)  # Resume must preserve the returned sheet.
        assert read_rows(folder/'filled.csv')==sheet
        return
    key=json.loads((root/'human/ctms/private_key.json').read_text())['items']
    real=[r for r in key.values() if not r['is_attention']]
    assert len(real)==120 and sum(r['calibration'] for r in real)==30
    cal={(r['split'],r['name']) for r in real if r['calibration']}
    held={(r['split'],r['name']) for r in real if not r['calibration']}
    assert not cal&held and len(held)==90
    sheet=read_rows(root/'human/ctms/public/ratings.csv')
    assert len(sheet)==132 and set(sheet[0])=={'item_id','task'}|{f'rater{i}_{s}' for i in (1,2,3) for s in ('causal_plausibility','physical_consistency','coverage')}
    assert (root/'human/ctms/public/index.html').exists()


def test_single_author_branch_audit_never_reports_agreement(tmp_path):
    from analysis.evaluation_campaign import dataset_inventory
    from analysis.evaluation_human import prepare_branch_audit,analyze_packages,read_rows,csv_write
    base=make_protocol_dataset(tmp_path,n=1)
    cfg={'campaign_dir':str(tmp_path/'run'),'splits':{'nuscenes':{'path':str(base)}},
         'human':{'mode':'author_review','raters':1,'audit_per_split':1},'statistics':{'seed':0}}
    prepare_branch_audit(cfg,{'nuscenes':dataset_inventory(base,1,'nuscenes')})
    folder=tmp_path/'run/human/branch_audit'
    rows=read_rows(folder/'public/ratings.csv')
    for row in rows:row.update(rater1='y',notes='Synthetic check of included/excluded action.')
    csv_write(folder/'filled.csv',list(rows[0]),rows)
    analyze_packages(cfg)
    result=json.loads((folder/'analysis.json').read_text())
    assert result['status']=='reviewed_by_author'
    assert result['counts']['y']==len(rows) and result['independent_validation']=='not_conducted'
    assert 'alpha' not in str(result)


def test_offline_validation_writes_current_axis_evidence(tmp_path):
    from analysis.evaluation_campaign import dataset_inventory,write_json
    from analysis.evaluation_validation import run_validation
    from analysis.run_model import run_model
    base=make_protocol_dataset(tmp_path,n=2);root=tmp_path/'run';write_json(root/'frozen.json',{})
    cfg={'campaign_dir':str(root),'statistics':{'seed':0},'splits':{'nuscenes':{'path':str(base)}},
         'validation':{'synthetic_scenarios':2,'seeds':2,'noise_scenarios':2,'noise_trials':2}}
    frozen={'datasets':{'nuscenes':dataset_inventory(base,2,'nuscenes')}}
    run_model(str(base),str(root/'baselines/nuscenes/oracle'),backend='oracle')
    run_validation(cfg,frozen)
    result=json.loads((root/'validation/results.json').read_text())
    checks=result['coherence_implementation_checks']
    assert checks['returncode']==0
    # the archived evidence must exercise every current coherence axis, the
    # action-profile control and action-labelled matching
    assert checks['tests']==['tests/test_causal_coherence.py','tests/test_coherence_action_axes.py',
                             'tests/test_baselines.py','tests/test_action_labelled_matching.py',
                             'tests/test_research_review_20260917.py','tests/test_action_profile_published.py',
                             'tests/test_chain_diagnostics.py',
                             'tests/test_intervention_pair.py','tests/test_l4_pair_scoring.py',
                             'tests/test_baseline_pairs.py','tests/test_l4_control_audit.py',
                             'tests/test_option_b_completion.py']
    assert all(t in checks['command'] for t in checks['tests'])
    assert result['identity']['tests']==checks['tests']
    assert 'CTMS' in result['noise_floor']['axes']
    assert any(r['param']=='node_similarity.thresholds.position_distance_m' for r in result['parameter_sensitivity']['variants'])


def test_resume_ignores_expired_job_ids_without_querying_them(tmp_path,monkeypatch):
    import analysis.evaluation_campaign as c
    import subprocess
    (tmp_path/'jobs.jsonl').write_text(json.dumps({'job_id':'123','task':'prepare'})+'\n')
    seen=[];submitted=[]
    def queue(command,**kwargs):
        seen.append(command)
        return subprocess.CompletedProcess(command,0,stdout='999 RUNNING\n',stderr='')
    monkeypatch.setattr(subprocess,'run',queue)
    monkeypatch.setattr(c,'submit_job',lambda cfg,task,**kw: submitted.append(task))
    c.submit_step({'campaign_dir':str(tmp_path)},'prepare')
    assert '--jobs' not in seen[0] and submitted==['prepare']


def test_active_campaign_job_blocks_duplicate_submission(tmp_path,monkeypatch):
    import analysis.evaluation_campaign as c
    import subprocess
    (tmp_path/'jobs.jsonl').write_text(json.dumps({'job_id':'123','task':'infer'})+'\n')
    monkeypatch.setattr(subprocess,'run',lambda command,**kw:subprocess.CompletedProcess(command,0,stdout='123_4 RUNNING\n'))
    with pytest.raises(RuntimeError,match='queued/running'):
        c.submit_step({'campaign_dir':str(tmp_path)},'prepare')


def test_baseline_execution_resumes_only_incomplete_scenarios(tmp_path,monkeypatch):
    import analysis.evaluation_campaign as c
    import analysis.run_model as runner
    base=make_protocol_dataset(tmp_path);root=tmp_path/'run';c.write_json(root/'frozen.json',{})
    cfg={'campaign_dir':str(root),'splits':{'nuscenes':{'path':str(base)}}}
    frozen={'datasets':{'nuscenes':c.dataset_inventory(base,2,'nuscenes')}}
    cell=dict(split='nuscenes',backend='oracle',condition='video',out_dir=str(root/'oracle'))
    c.execute_cell(cfg,cell,frozen)
    original=runner.run_model;calls=[]
    def spy(*args,**kwargs):
        calls.append(kwargs['scenario_names']);return original(*args,**kwargs)
    monkeypatch.setattr(runner,'run_model',spy)
    c.execute_cell(cfg,cell,frozen)
    assert not calls
    (root/'oracle/scenario_2/l4_pre.json').unlink()
    c.execute_cell(cfg,cell,frozen)
    assert calls==[['scenario_2']]


def test_parse_failure_is_a_complete_measured_attempt(tmp_path,monkeypatch):
    import analysis.run_model as runner
    from analysis.evaluation_campaign import dataset_inventory,check_cell
    base=make_protocol_dataset(tmp_path,n=1);rows=dataset_inventory(base,1,'nuscenes')
    monkeypatch.setitem(runner.BACKENDS,'mock',lambda *args:'I cannot provide JSON.')
    result=runner.run_model(str(base),str(tmp_path/'output'),backend='mock')
    assert result['failed']
    assert check_cell(tmp_path/'output',rows,condition='video',backend='mock')['complete']


# ── Per-task walltime and local (no-SLURM) execution ─────────────────────
def _slurm_cfg(tmp_path, **slurm):
    base = {'cpu_partition': 'work', 'gpu_partition': 'gpu', 'gpu_gres': 'gpu:1',
            'cpu_memory': '2G', 'gpu_memory': '8G', 'cpu_time': '12:00:00',
            'gpu_time': '3-00:00:00', 'cpus': 1}
    base.update(slurm)
    return {'campaign_dir': str(tmp_path), 'python': sys.executable, '_path': '/tmp/cfg',
            'slurm': base}


def _time_arg(command):
    return command[command.index('--time') + 1]


def test_prepare_requests_ten_minutes_by_default(tmp_path, capsys):
    from analysis.evaluation_campaign import submit_job, task_walltime
    cfg = _slurm_cfg(tmp_path)
    assert task_walltime(cfg, 'prepare') == '00:10:00'
    assert task_walltime(cfg, 'admit') == '00:10:00'
    # tasks without a measured default keep the partition-wide setting
    assert task_walltime(cfg, 'score') == '12:00:00'
    assert task_walltime(cfg, 'infer', backend='local') == '3-00:00:00'
    submit_job(cfg, 'prepare', dry_run=True)
    printed = capsys.readouterr().out
    assert '--time 00:10:00' in printed


def test_config_task_time_overrides_default(tmp_path):
    from analysis.evaluation_campaign import task_walltime
    cfg = _slurm_cfg(tmp_path, task_time={'prepare': '00:20:00', 'score': '01:30:00'})
    assert task_walltime(cfg, 'prepare') == '00:20:00'
    assert task_walltime(cfg, 'score') == '01:30:00'
    assert task_walltime(cfg, 'report') == '12:00:00'


def test_cli_time_override_applies_to_named_task_or_all(tmp_path):
    from analysis.evaluation_campaign import parse_time_overrides, task_walltime
    cfg = _slurm_cfg(tmp_path)
    cfg['slurm']['task_time'] = parse_time_overrides(['prepare=00:05:00'])
    assert task_walltime(cfg, 'prepare') == '00:05:00'
    assert task_walltime(cfg, 'score') == '12:00:00'
    cfg['slurm']['task_time'] = parse_time_overrides(['00:45:00'])
    assert task_walltime(cfg, 'prepare') == '00:45:00'
    assert task_walltime(cfg, 'score') == '00:45:00'
    with pytest.raises(ValueError):
        parse_time_overrides(['prepare=soon'])
    with pytest.raises(ValueError):
        parse_time_overrides(['unknown_task=00:10:00'])


def test_submit_records_requested_walltime(tmp_path, monkeypatch):
    import subprocess
    from analysis.evaluation_campaign import submit_job
    monkeypatch.setattr(subprocess, 'run', lambda *a, **k: subprocess.CompletedProcess(a, 0, stdout='4242;cluster\n', stderr=''))
    cfg = _slurm_cfg(tmp_path)
    submit_job(cfg, 'prepare')
    row = json.loads((tmp_path/'jobs.jsonl').read_text().splitlines()[-1])
    assert row['job_id'] == '4242' and row['walltime'] == '00:10:00'


def test_local_step_runs_prepare_in_process_without_sbatch(tmp_path, monkeypatch):
    import subprocess
    import analysis.evaluation_campaign as c
    ran = []
    monkeypatch.setattr(c, 'run_task', lambda cfg, task, index=0, backend=None: ran.append(task))
    monkeypatch.setattr(subprocess, 'run', lambda *a, **k: (_ for _ in ()).throw(AssertionError('sbatch must not be called')))
    cfg = _slurm_cfg(tmp_path)
    c.run_step_local(cfg, 'prepare')
    assert ran == ['prepare']
    assert not (tmp_path/'jobs.jsonl').exists()
    row = json.loads((tmp_path/'local_runs.jsonl').read_text().splitlines()[-1])
    assert row['task'] == 'prepare' and row['status'] == 'completed' and row['duration_sec'] >= 0


def test_local_step_report_runs_cpu_chain_in_order(tmp_path, monkeypatch):
    import analysis.evaluation_campaign as c
    ran = []
    monkeypatch.setattr(c, 'run_task', lambda cfg, task, index=0, backend=None: ran.append(task))
    monkeypatch.setattr(c, 'frozen_inputs', lambda cfg, **k: {})
    c.run_step_local(_slurm_cfg(tmp_path), 'report')
    assert ran == ['baselines', 'score', 'report']


def test_local_step_refuses_gpu_steps(tmp_path):
    import analysis.evaluation_campaign as c
    for step in ('smoke', 'evaluate'):
        with pytest.raises(ValueError, match='SLURM'):
            c.run_step_local(_slurm_cfg(tmp_path), step)


def test_local_step_records_failure(tmp_path, monkeypatch):
    import analysis.evaluation_campaign as c
    def boom(cfg, task, index=0, backend=None):
        raise RuntimeError('prepare exploded')
    monkeypatch.setattr(c, 'run_task', boom)
    with pytest.raises(RuntimeError):
        c.run_step_local(_slurm_cfg(tmp_path), 'prepare')
    row = json.loads((tmp_path/'local_runs.jsonl').read_text().splitlines()[-1])
    assert row['status'] == 'failed' and 'prepare exploded' in row['error']


def test_collect_cell_counts_node_label_departures_without_changing_scores(tmp_path):
    """A depth-1 event label that is in the vocabulary but departs
    from the reference convention passes the gates and zeroes the chain, so the
    report counts it separately (count only; scores come from the scorer)."""
    import copy
    from analysis.evaluation_campaign import dataset_inventory,score_cell,write_json,read_json,write_campaign_identity
    from analysis.run_model import run_model
    from analysis.evaluation_report import collect_cell
    base=make_protocol_dataset(tmp_path)
    rows=dataset_inventory(base,2,'nuscenes')
    out=tmp_path/'run';out.mkdir();write_json(out/'frozen.json',{})
    cfg={'campaign_dir':str(out),'splits':{'nuscenes':{'path':str(base)}}}
    frozen={'datasets':{'nuscenes':rows}}
    cell=dict(split='nuscenes',backend='oracle',condition='video',out_dir=str(out/'oracle'))
    write_campaign_identity(cfg,cell)
    run_model(str(base),cell['out_dir'],backend='oracle')
    # Corrupt only the labels of scenario_1: the L4 first node takes another
    # actor's event, the no-action branch (if declared) takes lane_follow.
    d=Path(cell['out_dir'])/'scenario_1'
    l4=read_json(d/'l4_pre.json'); l4['predicted_counterfactuals'][0]['treated_chain'][0]['event_type']='npc_brake'
    write_json(d/'l4_pre.json',l4)
    l3=read_json(d/'l3_pre.json'); noact=[b for b in l3['predicted_branches'] if b['intervention']['intervention_type']=='do_ego_no_action']
    for b in noact: b['chain'][0]['event_type']='lane_follow'
    write_json(d/'l3_pre.json',l3)
    score_cell(cfg,cell,frozen)
    data=collect_cell(cell,rows); m=data['metrics']
    # One matched pair per scenario; this diagnostic reads its treated chain.
    assert m['L4_first_node_label_match']=={'scenario_1':0.0,'scenario_2':1.0}
    assert data['label_diagnostics']['l4_first_node_labels_used_instead']=={'npc_brake':1}
    if noact:
        assert m['L3_no_action_label_match']=={'scenario_1':0,'scenario_2':1}
        assert data['label_diagnostics']['l3_no_action_labels_used_instead']=={'lane_follow':1}
    else:
        assert set(m['L3_no_action_label_match'].values())=={None}
    # the diagnostic reads gt.json/pred.json that scoring wrote; the score itself is the scorer's
    cas=m['Counterfactual_Alignment_Score']
    assert cas['scenario_1']<cas['scenario_2'] and cas['scenario_2']>0.0


@pytest.mark.skip(reason="needs the cluster campaign configuration, which is not part of this release")
def test_report_carries_ceiling_accounting_and_both_floors(tmp_path,monkeypatch):
    """Valid-output accounting is
    read against the oracle ceiling, and every model gets floor-relative rows
    against both non-learning controls."""
    test_full_matrix_report_exports_all_comparisons(tmp_path,monkeypatch)
    import analysis.evaluation_campaign as c
    result=c.read_json(Path(tmp_path/'run'/'report/results.json'))
    ceiling=[r for r in result['rows'] if r['comparison']=='valid_output_vs_ceiling']
    assert {r['metric'] for r in ceiling}>={'L3_ceiling_deficit','L4_ceiling_deficit','L3_valid_output_gap_to_ceiling','L4_valid_output_gap_to_ceiling'}
    l3=[r for r in ceiling if r['metric']=='L3_ceiling_deficit' and r['split']=='carla'][0]
    assert l3['configuration']=='oracle' and 0<l3['mean']<0.2
    scores={(r['split'],r['configuration'],r['condition'],r['metric']):r for r in result['rows'] if r['comparison']=='scores'}
    assert scores[('carla','oracle','video','L4_zero_valid')]['mean']==0.0
    floors={r['baseline'] for r in result['rows'] if r['comparison']=='floor_relative'}
    assert floors=={'action_profile_kinematic','action_profile_published'}
    assert {r['floor'] for r in result['rows'] if r['comparison']=='skill_vs_floor'}==floors
    assert 'valid_output_agreement' in result


def test_export_uses_research_labels_without_changing_metric_keys_or_values(tmp_path):
    import copy
    import csv
    from analysis.evaluation_report import export_tables
    rows = [dict(metric='Causal_Foresight_F1', mean=.75, lo=.5, hi=1., n=4, n_total=5),
            dict(metric='L3_zero_valid', mean=.2, lo=0., hi=.4, n=5, n_total=5)]
    original = copy.deepcopy(rows)
    export_tables(tmp_path, rows)
    exported = list(csv.DictReader((tmp_path / 'tables.csv').open()))
    assert exported[0]['metric'] == 'Causal_Foresight_F1'
    assert exported[0]['metric_label'] == 'Branch-set reference F1'
    assert float(exported[0]['mean']) == .75
    assert exported[1]['metric'] == 'L3_zero_valid'
    text = (tmp_path / 'scores.tex').read_text()
    assert 'Branch-set reference F1' in text and 'Ego-action branching' in text
    assert 'Causal\\_Foresight' not in text and 'L3\\_zero' not in text
    assert rows == original
