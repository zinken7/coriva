"""Offline instrument checks; results identify the tested axis and sample."""
import copy
from pathlib import Path
import random
import statistics

from analysis.evaluation_campaign import ROOT, read_json, write_json, sha


def full_noise_floor(gts, predictions, epsilons, env, n_trials=20, seed=0):
    from analysis.noise_floor import perturb_gt
    from run_benchmark import score_sample
    axes=('Causal_Foresight_F1','Counterfactual_Alignment_Score','CTMS')
    if not gts or len(gts)!=len(predictions) or not epsilons or n_trials<2:
        raise ValueError('Noise study requires paired samples, empirical errors and >=2 trials')
    rng=random.Random(seed); trials={axis:[] for axis in axes}
    for _ in range(n_trials):
        values={axis:[] for axis in axes}
        for gt,pred in zip(gts,predictions):
            pert=perturb_gt(gt,epsilons,rng)
            # `perturb_gt` only perturbs L3. Extend the same displacement
            # sampling to L4 references, without altering query metadata/root.
            for cf in pert.get('l4_queries', pert.get('counterfactuals', [])):
                for key in ('counterfactual_chain', 'control_chain'):
                    if key in cf:
                        changed=perturb_gt({'branches':[cf[key]]},epsilons,rng)
                        cf[key]=changed['branches'][0]
            scores=score_sample(pert,pred,env)
            for axis in axes: values[axis].append(scores[axis])
        for axis in axes: trials[axis].append(statistics.fmean(values[axis]))
    return dict(axes={a:dict(delta_noise=statistics.stdev(v),trial_means=v) for a,v in trials.items()},
                n_scenarios=len(gts),n_trials=n_trials,epsilon_sample_size=len(epsilons),
                interpretation='sensitivity to the declared spatial perturbation model; not a proven error bound')


def run_validation(cfg,frozen):
    from ctms.ctms import load_ctms_environment
    from analysis.rq3a import (make_synthetic_gts,make_study_gt,run_stability_study,
                               run_fleet_stability,score_f1,shuffle_branches,reformat_entity_aliases)
    from analysis.param_perturbation import run_param_perturbation,DEFAULT_GRID
    from analysis.evaluation_report import finite
    root=Path(cfg['campaign_dir']);out=root/'validation';out.mkdir(parents=True,exist_ok=True)
    v=cfg['validation'];seed=cfg['statistics']['seed']
    # Property tests archived as implementation evidence: the five coherence
    # axes, the action-profile control and action-labelled matching.
    # The list is part of the identity so a changed list reruns.
    tests=['tests/test_causal_coherence.py','tests/test_coherence_action_axes.py',
           'tests/test_baselines.py','tests/test_action_labelled_matching.py',
           # factual contract, RQ2 accounting, coherence 1.1, malformed-output
           # domain, freeze, export, version, the published-parameter control
           # and the chain diagnostics.
           'tests/test_research_review_20260917.py','tests/test_action_profile_published.py',
           'tests/test_chain_diagnostics.py',
           'tests/test_intervention_pair.py','tests/test_l4_pair_scoring.py',
           'tests/test_baseline_pairs.py','tests/test_l4_control_audit.py',
           'tests/test_option_b_completion.py']
    identity=dict(frozen_sha256=sha(root/'frozen.json'),settings=v,tests=tests)
    if (out/'results.json').exists() and read_json(out/'results.json').get('identity')==identity:
        return
    env=load_ctms_environment(str(ROOT/'ctms/ontology.yaml'),str(ROOT/'ctms/ctms_config.yaml'),
                              str(ROOT/'ctms/schema_branch.json'),str(ROOT/'ctms/schema_counterfactual.json'))
    gt=make_study_gt();branches=gt['branches'];base=score_f1(gt,branches,env)
    import json
    invariance=dict(original=base,branch_order=score_f1(gt,shuffle_branches(branches,random.Random(seed)),env),
                    aliases=score_f1(gt,reformat_entity_aliases(branches),env),
                    whitespace=score_f1(gt,json.loads(json.dumps(branches,indent=4)),env))
    gts=make_synthetic_gts(v['synthetic_scenarios'])
    grid=copy.deepcopy(DEFAULT_GRID)
    grid['node_similarity.thresholds.position_distance_m']=[1.0,3.0]
    result=dict(identity=identity,invariance=invariance,
                quality_ladder=run_stability_study(gt,env,seeds=v['seeds']),
                fleet=run_fleet_stability(gts,env,seeds=v['seeds']),
                parameter_sensitivity=run_param_perturbation(gts,env,grid=grid,seeds=v['seeds']))
    rows=list(frozen['datasets']['nuscenes']); random.Random(seed).shuffle(rows)
    eps=[r['fidelity'] for r in rows if finite(r.get('fidelity'))]
    selected=rows[:v['noise_scenarios']]
    if eps:
        observed=[];preds=[]
        for row in selected:
            observed.append(read_json(Path(cfg['splits']['nuscenes']['path'])/row['name']/'ground_truth.json'))
            d=root/'baselines/nuscenes/oracle'/row['name']
            l3=read_json(d/'l3_pre.json'); l4=read_json(d/'l4_pre.json')
            preds.append(dict(branches=l3['predicted_branches'],predicted_counterfactuals=l4['predicted_counterfactuals']))
        result['noise_floor']=full_noise_floor(observed,preds,eps,env,v['noise_trials'],seed)
        result['noise_floor']['scenario_names']=[r['name'] for r in selected]
        result['noise_floor']['prediction_control']='fixed unperturbed oracle'
        result['noise_floor']['epsilon_population']='accepted nuScenes roots; rejected candidates are not inferred'
    else:
        result['noise_floor']=dict(status='pending',reason='No empirical factual_fidelity_error values in released GT')
    # Execute the current property tests; stdout and exit status are
    # archived as the evidence.
    import subprocess,sys
    command=[sys.executable,'-m','pytest','-q',*tests]
    checked=subprocess.run(command,cwd=ROOT,text=True,capture_output=True)
    (out/'coherence_checks.txt').write_text(checked.stdout+checked.stderr)
    result['coherence_implementation_checks']=dict(command=command,tests=tests,returncode=checked.returncode)
    if checked.returncode:
        raise RuntimeError('Coherence property checks failed; see validation/coherence_checks.txt')
    result['coherence_human_construct_validity']='future work; not tested by these checks'
    write_json(out/'results.json',result)
