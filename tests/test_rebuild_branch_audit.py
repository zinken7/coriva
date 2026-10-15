"""Audit reset preserves input identity and refuses unrelated frozen drift."""
import copy
import json
from pathlib import Path
import pytest

from scripts import rebuild_branch_audit as rebuild
from analysis.branch_audit import AUDIT_VERSION
from analysis.evaluation_human import read_rows, csv_write


def test_review_revision_rejects_runtime_changes(monkeypatch):
    frozen={'code_files':[['analysis/branch_audit.py','old'],['ctms/ctms.py','same']]}
    monkeypatch.setattr(rebuild,'code_inventory',lambda:[('analysis/branch_audit.py','new'),('ctms/ctms.py','same')])
    assert rebuild.review_code_changes(frozen)[0]==['analysis/branch_audit.py']
    monkeypatch.setattr(rebuild,'code_inventory',lambda:[('analysis/branch_audit.py','new'),('ctms/ctms.py','changed')])
    with pytest.raises(ValueError,match='Non-audit runtime'):
        rebuild.review_code_changes(frozen)


def test_public_reset_keeps_images_ids_and_clears_judgments(tmp_path):
    from html import escape
    public=tmp_path/'public';scene=public/'audit_0004';scene.mkdir(parents=True)
    frame=scene/'frame00.jpg'
    for i in range(8):(scene/f'frame{i:02d}.jpg').write_bytes(b'original-image-bytes')
    sections={'Observed state':{},'Declared query':{'l3_ego_actions':['do_ego_no_action']},
              'Included branches':[{'is_factual':True,'intervention':{
                  'intervention_type':'do_ego_no_action','target_actor_id':'ego','trigger_depth':1}}]}
    (scene/'index.html').write_text(''.join('<h2>'+k+'</h2><pre>'+escape(json.dumps(v))+'</pre>' for k,v in sections.items()))
    rows=[dict(scenario_id='audit_0004',split='nuscenes',item_type='excluded_action',
               item='do_ego_swerve_left',prompt='old',rater1='y',notes='old rating')]
    csv_write(public/'ratings.csv',list(rows[0]),rows)
    with pytest.raises(ValueError,match='requires --reset'):
        rebuild.rebuild_public(public)
    package,_=rebuild.rebuild_public(public,True)
    assert package['items']==2 and package['rubric_version']==AUDIT_VERSION
    new=read_rows(public/'ratings.csv')
    assert {r['scenario_id'] for r in new}=={'audit_0004'}
    assert {r['item_type'] for r in new}=={'scene_context','included_branch'}
    assert all(not r['rater1'] and not r['notes'] for r in new)
    assert frame.read_bytes()==b'original-image-bytes'
    checks=json.loads((public/'automatic_checks.json').read_text())
    assert checks['counts']['not_checked']==1
    frame.unlink()
    before=(public/'ratings.csv').read_bytes()
    with pytest.raises(ValueError,match='eight source frames'):
        rebuild.rebuild_public(public,True)
    assert (public/'ratings.csv').read_bytes()==before


def test_campaign_reset_records_review_revision_and_keeps_freeze_usable(tmp_path):
    from test_evaluation_campaign import make_protocol_dataset
    from analysis.evaluation_campaign import prepare,frozen_inputs,read_json
    cfg={'campaign_dir':str(tmp_path/'run'),'_path':str(tmp_path/'cfg.yaml'),
         'splits':{s:{'path':str(make_protocol_dataset(tmp_path,s)), 'expected':2} for s in ('carla','nuscenes')},
         'smoke':{'path':str(make_protocol_dataset(tmp_path,'smoke',1)),'scenario':'scenario_1'},
         'statistics':{'n_boot':100,'seed':0},'probe':{'limit':2},
         'human':{'mode':'author_review','raters':1,'audit_per_split':1},'validation':{}}
    prepare(cfg)
    campaign=Path(cfg['campaign_dir']);folder=campaign/'human/branch_audit'
    prior=read_json(campaign/'frozen.json');keys=read_json(folder/'private_key.json')
    # Simulate the previous audit implementation's hash; all scoring code is unchanged.
    for entry in prior['code_files']:
        if entry[0]=='analysis/branch_audit.py':entry[1]='old-review-hash'
    (campaign/'frozen.json').write_text(json.dumps(prior))
    (folder/'filled.csv').write_text('obsolete judgments')
    package,_=rebuild.rebuild_campaign(cfg,True)
    assert read_json(folder/'private_key.json')==keys
    assert not (folder/'filled.csv').exists()
    assert package['rubric_version']==AUDIT_VERSION
    assert list((campaign/'audit_revisions').glob('*/frozen_before.json'))
    frozen_inputs(cfg,verify_files=True)  # also checks code inventory ordering
    assert read_json(campaign/'frozen.json')['datasets']==prior['datasets']
    gt=Path(cfg['splits']['carla']['path'])/'scenario_1/ground_truth.json'
    gt.write_text(gt.read_text()+' ')
    with pytest.raises(ValueError,match='Frozen data changed'):
        rebuild.rebuild_campaign(cfg,True)
    assert read_json(folder/'private_key.json')==keys
