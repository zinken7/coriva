"""Scenario-level dissertation tables from frozen campaign artifacts, offline."""
from __future__ import annotations

from collections import defaultdict
import csv
import math
from pathlib import Path
import statistics

import numpy as np

from analysis.causal_coherence import COHERENCE_AXES, COHERENCE_BREAKDOWNS
from analysis.intervention_pair import L4_PAIR_AXES, L4_PAIR_VERSION
from analysis.evaluation_campaign import (BASELINES, CONDITIONS, matrix_cells, baseline_cells,
    require_cell, read_json, write_json, sha)
from analysis.power_analysis import min_detectable_difference
from analysis.report_labels import TASK_LABELS, metric_label

COHERENCE = tuple(COHERENCE_AXES)
# Reported as ordinary score rows, deliberately NOT in COHERENCE: they are one
# axis split by action kind, and must not enter the primary-axis floor rows.
COHERENCE_SPLIT = tuple(COHERENCE_BREAKDOWNS)
# Reference-free L4 pair axes (docs/l4_pair_protocol.md §4): predicted control
# vs predicted treated, defined only on pair-valid queries; reported per axis
# with coverage, never combined with each other, with coherence or with CTMS.
L4_PAIR = tuple(L4_PAIR_AXES)
REFERENCE = ('Causal_Foresight_F1', 'Counterfactual_Alignment_Score', 'CTMS')
# Own-chain reference agreement, separate from pair-gated CAS and CTMS.
L4_CHAIN_REFERENCE = ('L4_control_similarity', 'L4_treated_similarity')
# Non-learning controls used for cross-split sensitivity; the action-profile
# controls are measured on their actual defined subsets, never assumed ceilings.
EXCESS_BASELINES = ('action_profile_kinematic', 'action_profile_published', 'kinematic_extrapolation', 'copy_factual', 'random_valid')
# Floors for the paired model-minus-control rows. `action_profile_kinematic`
# is the primary floor (published EGO profiles only; other actors at root
# velocity; twin intervention profile). `action_profile_published` additionally applies
# the per-scene hazard manoeuvre and intervention magnitudes the prompt prints,
# i.e. every number the model is given; exceeding it is what "beats
# non-learning arithmetic on the supplied information" means. Exceeding
# only the first shows more than the
# ego-profile rule, not more than the full printed arithmetic.
FLOORS = ('action_profile_kinematic', 'action_profile_published')


def finite(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def mean_stat(values, n_boot=10000, seed=0):
    values = list(values)
    n_total = len(values)
    vals = np.asarray([v for v in values if finite(v)], dtype=float)
    n = len(vals)
    if not n:
        return dict(mean=None, lo=None, hi=None, n=0, n_total=n_total, mdd=None, reason='no defined observations')
    rng = np.random.default_rng(seed)
    draws = np.concatenate([rng.choice(vals, size=(min(256, n_boot-i), n)).mean(axis=1)
                            for i in range(0, n_boot, 256)])
    return dict(mean=float(vals.mean()), lo=float(np.quantile(draws, .025)), hi=float(np.quantile(draws, .975)),
                n=n, n_total=n_total, mdd=min_detectable_difference(float(vals.std(ddof=1)), n) if n > 1 else None)


def paired_stat(a, b, n_boot=10000, seed=0):
    ids = sorted(set(a) & set(b))
    return mean_stat([a[k]-b[k] if finite(a[k]) and finite(b[k]) else None for k in ids], n_boot, seed)


def excess_stat(model_ns, model_c, baseline_ns, baseline_c, n_boot=10000, seed=0):
    def differences(a, b):
        return np.asarray([a[k]-b[k] for k in sorted(set(a)&set(b)) if finite(a[k]) and finite(b[k])])
    ns, c = differences(model_ns, baseline_ns), differences(model_c, baseline_c)
    out = dict(n_nuscenes=len(ns), n_carla=len(c), mean=None, lo=None, hi=None,
               interpretation='baseline-adjusted domain difference; not a memorisation estimate')
    if not len(ns) or not len(c):
        out['reason'] = 'axis undefined on a shared model-baseline subset'
        return out
    rng = np.random.default_rng(seed)
    draws = np.concatenate([rng.choice(ns, (min(256,n_boot-i),len(ns))).mean(axis=1)
                            - rng.choice(c, (min(256,n_boot-i),len(c))).mean(axis=1)
                            for i in range(0,n_boot,256)])
    out.update(mean=float(ns.mean()-c.mean()), lo=float(np.quantile(draws,.025)), hi=float(np.quantile(draws,.975)))
    return out


def skill_stat(model, floor, oracle, n_boot=10000, seed=0):
    """Skill relative to the non-learning floor: mean(model-floor)/mean(oracle-floor)
    over scenarios where all three are defined, bootstrapped over scenarios as a
    ratio of paired means. Zero = no better than the floor, one = at the oracle.
    None with a reason when the floor already matches the oracle."""
    ids=sorted(k for k in set(model)&set(floor)&set(oracle)
               if finite(model[k]) and finite(floor[k]) and finite(oracle[k]))
    out=dict(mean=None, lo=None, hi=None, n=len(ids), floor='action_profile_kinematic')
    if not ids:
        out['reason']='axis undefined on a shared model-floor-oracle subset'; return out
    gain=np.asarray([model[k]-floor[k] for k in ids]); room=np.asarray([oracle[k]-floor[k] for k in ids])
    if room.mean()<=1e-9:
        out['reason']='degenerate floor: control already matches the oracle'; return out
    rng=np.random.default_rng(seed); n=len(ids); ratios=[]
    for i in range(0,n_boot,256):
        idx=rng.integers(0,n,size=(min(256,n_boot-i),n))
        r=room[idx].mean(axis=1); g=gain[idx].mean(axis=1)
        ratios.append(np.where(r>1e-9, g/np.where(r>1e-9,r,1.0), np.nan))
    draws=np.concatenate(ratios); draws=draws[np.isfinite(draws)]
    out.update(mean=float(gain.mean()/room.mean()),
               lo=float(np.quantile(draws,.025)) if len(draws) else None,
               hi=float(np.quantile(draws,.975)) if len(draws) else None)
    return out


def common_defined_ids(metric_maps):
    """Scenario IDs on which every given {scenario_id: value} map is defined."""
    ids=None
    for m in metric_maps:
        defined={k for k,v in m.items() if finite(v)}
        ids=defined if ids is None else ids&defined
    return sorted(ids or [])


def ratio_stat(numerators, denominators, n_boot=10000, seed=0):
    ids = sorted(k for k in set(numerators)&set(denominators)
                 if finite(numerators[k]) and finite(denominators[k]))
    a = np.array([numerators[k] for k in ids], dtype=float)
    b = np.array([denominators[k] for k in ids], dtype=float)
    out = dict(mean=None, lo=None, hi=None, n=len(ids), numerator=float(a.sum()), denominator=float(b.sum()))
    if not len(ids) or b.sum() == 0:
        out['reason'] = 'zero denominator or no paired observations'; return out
    rng = np.random.default_rng(seed); draws = []
    for i in range(0,n_boot,256):
        ix = rng.integers(len(ids), size=(min(256,n_boot-i),len(ids)))
        den = b[ix].sum(axis=1); num = a[ix].sum(axis=1)
        draws.extend((num[den!=0]/den[den!=0]).tolist())
    out.update(mean=float(a.sum()/b.sum()), lo=float(np.quantile(draws,.025)), hi=float(np.quantile(draws,.975)))
    out['defined_bootstrap_draws'] = len(draws)
    return out


def collect_cell(cell, inventory, split_path=None, frozen_sha256=None):
    """Unify directory IDs and scorer IDs without matching by position/digits.

    ``split_path`` (the split's scenario root) enables the label-free chain
    diagnostics of ``analysis.chain_diagnostics``: per scenario,
    structure (chain length, branch completeness, snapshot), semantics of the
    requested action/intervention (speed sign, target motion direction, ego
    policy adherence) and geometry (position error by depth). They are
    reported as further metrics next to CTMS and change no score."""
    require_cell(cell, inventory, allow_rescore=True)
    path=Path(cell['out_dir']); report=read_json(path/'report.json'); coherence=read_json(path/'coherence.json')
    proof=read_json(path/'score_provenance.json'); l4_pair=read_json(path/'l4_pair.json')
    from ctms.ctms import SCORER_VERSION, L4_PAIR_PROTOCOL
    if report.get('scorer_version') != SCORER_VERSION or report.get('l4_pair_protocol') != L4_PAIR_PROTOCOL:
        raise ValueError(f'{path}: report scorer/L4 protocol version differs; rescore the cell')
    # Final campaign artifacts must name the axis definition they were scored
    # under and the campaign they belong to.
    from analysis.causal_coherence import COHERENCE_VERSION
    if coherence.get('coherence_version')!=COHERENCE_VERSION:
        raise ValueError(f"{path}: coherence.json version {coherence.get('coherence_version')!r} is not {COHERENCE_VERSION!r}; rescore the cell")
    if l4_pair.get('l4_pair_version')!=L4_PAIR_VERSION:
        raise ValueError(f"{path}: l4_pair.json version {l4_pair.get('l4_pair_version')!r} is not {L4_PAIR_VERSION!r}; rescore the cell")
    if frozen_sha256 is not None and proof.get('frozen_sha256')!=frozen_sha256:
        raise ValueError(f"{path}: score_provenance.frozen_sha256 belongs to another campaign")
    if (proof['report_sha256'] != sha(path/'report.json') or proof['coherence_sha256'] != sha(path/'coherence.json')
            or proof.get('l4_pair_sha256') != sha(path/'l4_pair.json')):
        raise ValueError(f'Score artifacts changed: {path}; rerun scoring')
    for filename, expected in proof['prediction_files'].items():
        if sha(path/filename)!=expected:
            raise ValueError(f'Prediction changed after scoring: {path/filename}')
    idmap={r['scenario_id']:r['name'] for r in inventory}
    if len(report['per_sample'])!=len(inventory) or {s['scenario_id'] for s in report['per_sample']}!=set(idmap):
        raise ValueError(f'Report scenario membership differs: {path}')
    metrics=defaultdict(dict); compliance={'L3':{},'L4':{}}; groups=defaultdict(lambda:defaultdict(dict))
    if len(coherence['per_scenario']) != len(inventory) or {r['scenario_id'] for r in coherence['per_scenario']} != set(idmap.values()):
        raise ValueError(f'Coherence report scenario membership differs: {path}')
    for row in coherence['per_scenario']:
        sid=row['scenario_id']
        for axis in COHERENCE+COHERENCE_SPLIT:
            metrics[axis][sid]=row.get(axis)
        # Two different failures, reported separately:
        # endpoint shortfall = too few usable ego endpoints to compare (axis
        # undefined); collapse = endpoints supplied but coincident (axis = 0).
        resp=row.get('responsiveness')
        metrics['endpoint_shortfall_rate'][sid]=int(resp is None)
        metrics['collapse_rate'][sid]=(None if resp is None else int(resp<=0.0))
    if len(l4_pair['per_scenario']) != len(inventory) or {r['scenario_id'] for r in l4_pair['per_scenario']} != set(idmap.values()):
        raise ValueError(f'L4 pair report scenario membership differs: {path}')
    for row in l4_pair['per_scenario']:
        sid=row['scenario_id']; pair=row.get('pair') or {}
        for axis in L4_PAIR:
            metrics[axis][sid]=row.get(axis)
        # Unit accounting of the pair (docs/l4_pair_protocol.md §3): indicator
        # means with intervals; an axis undefined on a scenario is reported
        # through these, never as 0.
        for key in ('pair_complete','pair_valid','control_valid','treated_valid'):
            metrics[f'L4_{key}'][sid]=int(bool(pair.get(key)))
        metrics['L4_pair_axes_defined'][sid]=int(bool(row.get('defined')))
        details = row.get('details') or {}
        kinematics = details.get('ego_kinematics') or {}
        for key in ('speed_constancy_control', 'speed_constancy_treated', 'pair_path_agreement', 'max_ego_path_gap_m'):
            metrics[f'L4_kinematic_{key}'][sid] = kinematics.get(key)
        policy = details.get('ego_policy_consistency') or {}
        metrics['L4_declared_policy_violation'][sid] = (
            int(bool(policy['policy_violations'])) if 'policy_violations' in policy else None)
    for sample in report['per_sample']:
        sid=idmap[sample['scenario_id']]
        for axis in REFERENCE:
            metrics[axis][sid]=sample['scores'][axis]
            metrics[f'{axis}_floor_rate'][sid]=int(finite(sample['scores'][axis]) and sample['scores'][axis]<=0.0)
        l3=sample.get('layer3') or {}; l4=sample.get('layer4') or {}
        d=l3.get('diagnostics') or {}; outcomes=l4.get('results') or []
        metrics['branch_shortfall_rate'][sid]=(int(d.get('pred_unique_count',0)<d['gt_count'])
                                               if finite(d.get('gt_count')) else None)
        # Valid-output agreement K is CONDITIONAL on at least one valid unit.
        # With none, the scorer's numerical fallback (0) is not an agreement
        # measured on any prediction, and 1-K = 1 would book the whole loss as
        # disagreement among valid outputs when none existed. Such scenarios
        # are reported through the
        # `{task}_zero_valid` indicator and leave K and both deficits
        # undefined, so the deficit means describe scenarios that had a valid
        # unit, next to the count of those that had none.
        valid=[o['Score_interv'] for o in outcomes if o.get('valid')]
        # Control-chain reference agreement (pair protocol): reported on its
        # own validity, separately from CAS; undefined on a legacy reference.
        control=[o.get('control_similarity') for o in outcomes if (o.get('pair') or {}).get('control_reference_present')]
        control_valid=[o.get('control_similarity') for o in outcomes if (o.get('pair') or {}).get('control_valid')]
        metrics['L4_control_similarity'][sid]=(statistics.fmean(control) if control else None)
        metrics['L4_control_valid_only'][sid]=(statistics.fmean(control_valid) if control_valid else None)
        treated=[(o.get('treated') or {}).get('Score_interv', 0.0) for o in outcomes]
        treated_valid=[o['treated']['Score_interv'] for o in outcomes
                       if (o.get('treated') or {}).get('valid')]
        metrics['L4_treated_similarity'][sid]=(statistics.fmean(treated) if treated else None)
        metrics['L4_treated_valid_only'][sid]=(statistics.fmean(treated_valid) if treated_valid else None)
        l3_zero_valid=int(d.get('pred_valid_count',0)==0) if finite(d.get('pred_unique_count')) else None
        l4_zero_valid=(int(not valid) if outcomes else None)
        metrics['L3_zero_valid'][sid]=l3_zero_valid
        metrics['L4_zero_valid'][sid]=l4_zero_valid
        metrics['L3_valid_only'][sid]=(None if l3_zero_valid in (1,None) else l3.get('F1_cf_valid_only'))
        metrics['L4_valid_only'][sid]=(statistics.fmean(valid) if valid else None)
        compliance['L3'][sid]=(d.get('pred_valid_count',0),d.get('pred_unique_count',0))
        compliance['L4'][sid]=(len(valid),len(outcomes))
        for task, axis in (('L3',REFERENCE[0]),('L4',REFERENCE[1])):
            k=metrics[f'{task}_valid_only'][sid]
            metrics[f'{task}_compliance_deficit'][sid]=(None if k is None else k-sample['scores'][axis])
            metrics[f'{task}_valid_output_deficit'][sid]=(None if k is None else 1-k)
        for horizon, scores in sample.get('horizon_scores',{}).items():
            for axis, value in scores.items():
                metrics[f'{horizon}/{axis}'][sid]=value
        for dimension in ('intervention_type','target_class'):
            grouped=defaultdict(list)
            for o in outcomes:
                grouped[o.get(dimension,'unknown')].append(o)
            for name, vals in grouped.items():
                groups[dimension][name][sid]=dict(score=statistics.fmean(o['Score_interv'] for o in vals),
                                                  valid=sum(bool(o.get('valid')) for o in vals), total=len(vals))
    label_diagnostics=node_label_diagnostics(read_json(path/'gt.json'),read_json(path/'pred.json'),idmap,metrics)
    if split_path is not None:
        from analysis.chain_diagnostics import cell_diagnostics, METRICS as CHAIN_METRICS
        # The diagnostics read the reference from the split directory; that
        # file must be the frozen one, so its hash is checked against the
        # inventory row before anything is read from it.
        for row in inventory:
            expected={f['path']:f['sha256'] for f in row.get('files',[])}.get('ground_truth.json')
            if expected is not None and sha(Path(split_path)/row['name']/'ground_truth.json')!=expected:
                raise ValueError(f"{split_path}/{row['name']}/ground_truth.json differs from the frozen inventory")
        per_scene=cell_diagnostics(str(split_path),str(path))
        for row in inventory:
            values=per_scene.get(row['name'],{})
            for key in CHAIN_METRICS:
                metrics[key][row['name']]=values.get(key)
    provider_versions=set()
    for row in inventory:
        sid=row['name']; m=read_json(path/sid/'run_manifest.json'); meta=m.get('generation_metadata') or {}
        if meta.get('model_version'):
            provider_versions.add(meta['model_version'])
        for k in ('input_tokens','output_tokens','thinking_tokens','latency_seconds','peak_cuda_memory_bytes'):
            metrics[k][sid]=meta.get(k)
        tokens=meta.get('output_tokens'); cap=meta.get('max_output_tokens')
        finish=meta.get('finish_reason')
        # A length-cap hit is observable; it does not by itself prove invalid JSON.
        metrics['output_cap_hit'][sid]=(int(finish=='MAX_TOKENS') if finish is not None else
                                       int(tokens>=cap) if finite(tokens) and finite(cap) else None)
        if cell.get('configuration'):
            from analysis.parse_prediction import parse_model_output
            _,_,errors=parse_model_output((path/sid/'final_response.txt').read_text(),row['scenario_id'])
            metrics['parse_failure'][sid]=int(bool(errors))
    return dict(metrics=dict(metrics), compliance=compliance, groups=groups, raw_summary=report['summary'],
                provider_versions=sorted(provider_versions), label_diagnostics=label_diagnostics)


def node_label_diagnostics(gt, pred, idmap, metrics):
    """Count-only diagnostic; it changes no score.

    The references label every non-terminal L4 node and the no-action L3
    branch `maintain_course`/`ego_no_action`; `ctms._node_similarity` zeroes a
    node on any `event_type` mismatch and the prefix minimum then zeroes the
    chain, so a wrong depth-1 label lands in the valid-output deficit (1-K),
    not in compliance. Per scenario: `L4_first_node_label_match` is the
    fraction of predicted L4 chains (paired with the query by
    intervention_type) whose depth-1 event_type equals the reference's, and
    `L3_no_action_label_match` is 1/0 for the no-action branch (None when the
    scenario declares no such branch or the model supplied none). The labels
    used instead are tallied for the report."""
    # Same root rule as the scorer (`ctms._pre_normalize_branch`): leading
    # observed_state nodes are the echoed input, never scored, so the first
    # FUTURE node is compared.
    import yaml
    from analysis.evaluation_campaign import ROOT as _ROOT
    ev_syn = ((yaml.safe_load((_ROOT/'ctms/ontology.yaml').read_text()) or {}).get('synonyms') or {}).get('event_type', {})
    def event(node):
        ev=node.get('event_type')
        return ev_syn.get(ev, ev) if isinstance(ev, str) else None   # a non-string label is no label
    def first_event(chain):
        # positions are kept: a malformed first future node reads as no label
        nodes=[n if isinstance(n, dict) else None for n in chain] if isinstance(chain, list) else []
        while nodes and nodes[0] is not None and event(nodes[0])=='observed_state':
            nodes=nodes[1:]
        return event(nodes[0]) if nodes and nodes[0] is not None else None
    gt_by_id={s['scenario_id']:s for s in gt['scenarios']}
    l4_wrong=defaultdict(int); l3_wrong=defaultdict(int)
    for s in pred['scenarios']:
        sid=idmap[s['scenario_id']]
        g=(gt_by_id[s['scenario_id']]['samples'][0]['ground_truth'])
        p=(s['samples'][0]['prediction'] if s.get('samples') else {})
        # protocol-1.0 datasets carry the target-actor query under l4_queries;
        # datasets that carry ego-target chains under counterfactuals are also accepted.
        queries={((q.get('counterfactual_chain') or {}).get('intervention') or {}).get(
            'intervention_type', q.get('intervention_type')):q
            for q in (g.get('l4_queries') or g.get('counterfactuals') or []) if isinstance(q,dict)}
        matched=[]
        for c in (p.get('predicted_counterfactuals') if isinstance(p.get('predicted_counterfactuals'), list) else []):
            # `[null]` survives the parser so the schema gate can fail it; the
            # diagnostic skips such units.
            if not isinstance(c, dict) or not isinstance(c.get('intervention'), dict):
                continue
            itype=c['intervention'].get('intervention_type')
            q=queries.get(itype) if isinstance(itype, str) else None
            # pair protocol: the treated chain is `treated_chain` (alias `chain`)
            chain=c.get('treated_chain') if 'treated_chain' in c else c.get('chain')
            if not q or not isinstance(chain, list) or not chain:
                continue
            ref=first_event((q.get('counterfactual_chain') or {}).get('chain')); got=first_event(chain)
            if ref is not None and got is not None:
                matched.append(int(ref==got))
                if ref!=got:
                    l4_wrong[got]+=1
        metrics['L4_first_node_label_match'][sid]=(sum(matched)/len(matched) if matched else None)
        ref_noact=next((b for b in g.get('branches') or []
                        if (b.get('intervention') or {}).get('intervention_type')=='do_ego_no_action' and b.get('chain')),None)
        pred_branches=p.get('branches') if isinstance(p.get('branches'), list) else []
        pred_noact=next((b for b in pred_branches
                         if isinstance(b, dict) and isinstance(b.get('intervention'), dict)
                         and b['intervention'].get('intervention_type')=='do_ego_no_action'
                         and isinstance(b.get('chain'), list) and b['chain']),None)
        if ref_noact is None or pred_noact is None:
            metrics['L3_no_action_label_match'][sid]=None
        else:
            ref=first_event(ref_noact['chain']); got=first_event(pred_noact['chain'])
            metrics['L3_no_action_label_match'][sid]=(int(ref==got)
                if ref is not None and got is not None else None)
            if ref is not None and got is not None and ref!=got:
                l3_wrong[got]+=1
    return dict(l4_first_node_labels_used_instead=dict(l4_wrong), l3_no_action_labels_used_instead=dict(l3_wrong))


def tex_escape(text):
    return str(text).replace('\\',r'\textbackslash{}').replace('_',r'\_\allowbreak{}').replace('%',r'\%').replace('&',r'\&').replace('#',r'\#')


def export_tables(out, rows):
    # `baseline` names the control of a floor_relative / skill_vs_floor row;
    # without it the two floors would export as identical rows.
    fields=['split','configuration','condition','comparison','metric','metric_label','baseline','mean','lo','hi','n','n_total','n_common','n_nuscenes','n_carla','mdd','reason']
    with (out/'tables.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=fields,extrasaction='ignore');w.writeheader();w.writerows(dict(r, metric_label=metric_label(r["metric"])) for r in rows)
    sections=defaultdict(list)
    for r in rows:
        sections[r.get('comparison','scores')].append(r)
    for label, entries in sections.items():
        text=[r'\begin{longtable}{p{.30\textwidth}p{.23\textwidth}rrr}',
              r'Configuration / split / input & Measurement & Mean & 95\% CI & $n/N$ \\',r'\hline',r'\endhead']
        def num(v): return '--' if v is None else f'{v:.3f}'
        for r in entries:
            identity=' / '.join(str(r.get(k,'')) for k in ('configuration','split','condition'))
            if r.get('baseline'):
                identity+=f" vs {r['baseline']}"
            ci='--' if r.get('lo') is None else f"[{num(r['lo'])}, {num(r['hi'])}]"
            n=r.get('n',f"{r.get('n_nuscenes','--')}/{r.get('n_carla','--')}")
            if 'n_total' in r:
                n=f"{n}/{r['n_total']}"
            text.append(f"{tex_escape(identity)} & {tex_escape(metric_label(r['metric']))} & {num(r.get('mean'))} & {ci} & {n} \\\\")
        text.append(r'\end{longtable}')
        (out/f'{label}.tex').write_text('\n'.join(text)+'\n')


def cell_rows(key, data, boot, seed):
    """The `scores` and `l4_groups` rows and the summary of one collected cell."""
    split, name, condition = key
    identity=dict(split=split,configuration=name,condition=condition); records=[]
    stats={axis:mean_stat(list(values.values()),boot,seed) for axis,values in data['metrics'].items()}
    for task, units in data['compliance'].items():
        stats[f'{task}_compliance']=ratio_stat({k:v[0] for k,v in units.items()},{k:v[1] for k,v in units.items()},boot,seed)
    for axis,result in stats.items():
        records.append(dict(**identity,comparison='scores',metric=axis,**result))
    for dimension, groups in data['groups'].items():
        for name_, units in groups.items():
            result=mean_stat([v['score'] for v in units.values()],boot,seed)
            result['query_count']=sum(v['total'] for v in units.values())
            records.append(dict(**identity,comparison='l4_groups',metric=f'{dimension}/{name_}/CAS',**result))
            records.append(dict(**identity,comparison='l4_groups',metric=f'{dimension}/{name_}/compliance',
                                **ratio_stat({k:v['valid'] for k,v in units.items()},{k:v['total'] for k,v in units.items()},boot,seed)))
    return records, dict(statistics=stats,diagnostics=data['raw_summary'],node_labels=data['label_diagnostics'])


def comparison_rows(cells, model_names, splits, boot, seed, model_pairs):
    """Rows comparing the models with each other, with the two floors, across
    inputs and across splits. `cells` maps (split, configuration, condition) to
    a collected cell, baselines under condition 'video'. `model_pairs` lists
    (a, b, comparison): paired differences a - b per split and condition."""
    records=[]
    for name in model_names:
        versions={v for (split,model,condition),data in cells.items() if model==name for v in data['provider_versions']}
        if len(versions)>1:
            raise ValueError(f'Provider model version changed across splits/conditions for {name}: {sorted(versions)}')
    # Common-subset ranking: per split/condition/axis, every model configuration
    # is summarised on the scenarios where ALL configurations define the axis, so a
    # cross-model ranking never compares means over different surviving subsets.
    # A cell that was not run (arm B runs the video condition only; a model may
    # be absent from a root) contributes nothing: every comparison below uses
    # the cells that exist and names them, never a placeholder.
    for split in splits:
        for condition in CONDITIONS:
            present=[model for model in model_names if (split,model,condition) in cells]
            if not present:
                continue
            for axis in COHERENCE+L4_PAIR:
                ids=common_defined_ids([cells[(split,model,condition)]['metrics'][axis] for model in present])
                for model in present:
                    values=[cells[(split,model,condition)]['metrics'][axis][k] for k in ids]
                    records.append(dict(split=split,configuration=model,condition=condition,comparison='scores_common_subset',
                                        metric=axis,n_common=len(ids),**mean_stat(values,boot,seed)))
    # Floor-relative rows (dissertation tab:floor-coherence / tab:floor-reference):
    # every model cell minus the action-profile control of the same split, paired
    # on shared defined scenarios; skill on the reference axes uses the oracle.
    # A root without a control or the oracle (a partial arm) keeps every row
    # that does not need it; the report's `cells_missing` names what is absent.
    for split in splits:
        oracle=cells[(split,'oracle','video')]['metrics'] if (split,'oracle','video') in cells else None
        for model in model_names:
            for condition in CONDITIONS:
                if (split,model,condition) not in cells:
                    continue
                mm=cells[(split,model,condition)]['metrics']
                for floor_name in FLOORS:
                    if (split,floor_name,'video') not in cells:
                        continue
                    floor=cells[(split,floor_name,'video')]['metrics']
                    for axis in COHERENCE+L4_PAIR+REFERENCE+L4_CHAIN_REFERENCE:
                        records.append(dict(split=split,configuration=model,condition=condition,comparison='floor_relative',
                                            metric=axis,baseline=floor_name,**paired_stat(mm[axis],floor[axis],boot,seed)))
                    if oracle is None:
                        continue
                    for axis in REFERENCE:
                        records.append(dict(split=split,configuration=model,condition=condition,comparison='skill_vs_floor',
                                            metric=axis,**{**skill_stat(mm[axis],floor[axis],oracle[axis],boot,seed),'baseline':floor_name,'floor':floor_name}))
                if oracle is None:
                    continue
                # Ceiling-aware valid-output accounting: the reference scored
                # as its own
                # prediction does not reach K = 1 (non-terminal branches
                # score 0 on the terminal component), so part of every 1-K is
                # the scale's own ceiling, not disagreement. `ceiling_deficit`
                # = 1 - K(oracle); `valid_output_gap_to_ceiling` = K(oracle) -
                # K(model), paired on scenarios where both define K.
                for task in ('L3','L4'):
                    records.append(dict(split=split,configuration=model,condition=condition,comparison='valid_output_vs_ceiling',
                                        metric=f'{task}_valid_output_gap_to_ceiling',**paired_stat(oracle[f'{task}_valid_only'],mm[f'{task}_valid_only'],boot,seed)))
        for task in ('L3','L4'):
            if oracle is None:
                continue
            records.append(dict(split=split,configuration='oracle',condition='video',comparison='valid_output_vs_ceiling',
                                metric=f'{task}_ceiling_deficit',**mean_stat(list(oracle[f'{task}_valid_output_deficit'].values()),boot,seed)))
    contrasts=(('video_history','video','history_added'),('video_history','history_json','images_added'),
               ('video','history_json','representation_difference'))
    for split in splits:
        for model in model_names:
            for a,b,label in contrasts:
                if (split,model,a) not in cells or (split,model,b) not in cells:
                    continue
                ma=cells[(split,model,a)]['metrics'];mb=cells[(split,model,b)]['metrics']
                for axis in COHERENCE+L4_PAIR+REFERENCE+L4_CHAIN_REFERENCE:
                    records.append(dict(split=split,configuration=model,condition=f'{a}-{b}',comparison=label,
                                        metric=axis,**paired_stat(ma[axis],mb[axis],boot,seed)))
        for a, b, comparison in model_pairs:
            for condition in CONDITIONS:
                if (split,a,condition) not in cells or (split,b,condition) not in cells:
                    continue
                da=cells[(split,a,condition)]; db=cells[(split,b,condition)]
                for axis in COHERENCE+L4_PAIR+REFERENCE+L4_CHAIN_REFERENCE:
                    records.append(dict(split=split,configuration=f'{a}-{b}',condition=condition,
                                        comparison=comparison,metric=axis,
                                        **paired_stat(da['metrics'][axis],db['metrics'][axis],boot,seed)))
    for model in model_names:
        for condition in CONDITIONS:
            for baseline in EXCESS_BASELINES:
                keys=[('nuscenes',model,condition),('carla',model,condition),
                      ('nuscenes',baseline,'video'),('carla',baseline,'video')]
                if any(k not in cells for k in keys):
                    continue
                for axis in COHERENCE+L4_PAIR+REFERENCE:
                    maps=[cells[k]['metrics'][axis] for k in keys]
                    records.append(dict(split='nuscenes-minus-carla',configuration=model,condition=condition,
                                        comparison='contamination_excess',metric=f'{axis}/{baseline}',**excess_stat(*maps,boot,seed)))
    return records


CONTROL_NOTES={
    'full': ('the prompt prints the published profiles, the scripted hazard manoeuvre and the L4 magnitude, '
             'so action_profile_published is an information-matched floor and action_profile_kinematic '
             'the published-profile floor'),
    'recipe_withheld': ('the prompt withholds the published profiles, the scripted hazard manoeuvre and the L4 '
                        'magnitude, so both controls are information-ADVANTAGED anchors that read values the '
                        'model never saw; a model is not expected to reach them, and floor_relative and '
                        'skill_vs_floor rows are read as distance from what the recipe gives'),
    'ego_profiles_withheld': ('the prompt prints the scripted hazard manoeuvre and the L4 magnitude but withholds '
                              'the published ego profiles, so both controls are information-ADVANTAGED on the ego '
                              'actions only; floor_relative and skill_vs_floor rows are read as distance from what '
                              'the ego recipe gives'),
}


def write_report(out, frozen_sha256, records, summaries, probes, boot, seed, prompt_variant='full', cells_missing=(),
                 baselines_root=None):
    """results.json, the tables and the README of a report directory.
    `prompt_variant` names the arm every reported cell was run under and
    selects how the controls are described; `cells_missing` lists the
    (split, configuration, condition) cells the design has and this root lacks;
    `baselines_root` is where the controls and the oracle were read from."""
    result=dict(version='coriva-dissertation-report-2',frozen_sha256=frozen_sha256,
                prompt_variant=prompt_variant,controls=CONTROL_NOTES.get(prompt_variant,'unknown prompt variant'),
                cells_missing=[list(k) for k in cells_missing],baselines_root=baselines_root,
                l4_pair_version=L4_PAIR_VERSION,
                n_boot=boot,seed=seed,primary_axes=list(COHERENCE),primary_l4_axes=list(L4_PAIR),
                secondary_axes=list(REFERENCE),l4_chain_reference=list(L4_CHAIN_REFERENCE),
                task_labels=TASK_LABELS, metric_labels={r['metric']: metric_label(r['metric']) for r in records},
                l4_pair_protocol=('the target-actor intervention unit is a matched control/treated pair; `target_responsiveness`, '
                                  '`direction_consistency`, `ego_policy_consistency` are reference-free and defined '
                                  'only on pair-valid queries with the required support; `L4_pair_axes_defined` indicates all three are defined; `L4_pair_complete`, '
                                  '`L4_pair_valid`, `L4_control_valid`, `L4_treated_valid` are indicator means; '
                                  'CAS is the treated-chain agreement of pair-valid answers and '
                                  "`L4_control_similarity` and `L4_treated_similarity` retain each chain's own gated agreement separately"),
                intervals='pointwise scenario bootstrap; no family-wise correction',
                mdd='normal approximation using observed variance, alpha=.05/power=.8; descriptive sensitivity only',
                compliance='pooled gate counts with scenario bootstrap; undefined for zero denominator',
                valid_output_agreement=('K, 1-K and K-R are defined only on scenarios with at least one valid unit of '
                                        'that task; `<task>_zero_valid` is the indicator mean of scenarios with none. '
                                        '`valid_output_vs_ceiling` rows give 1-K of the reference scored as its own '
                                        'prediction (the scale ceiling) and K(oracle)-K(model), paired'),
                l4_groups='scenario means within family/class, with query counts separately',
                cells=summaries,rows=records,probes=probes)
    write_json(out/'results.json',result);export_tables(out,records)
    (out/'README.md').write_text(f'''# Evaluation outputs

Prompt variant of every cell in this report: `{prompt_variant}`. Controls: {CONTROL_NOTES.get(prompt_variant, "unknown")}.
Cells the design has and this root lacks: {len(cells_missing)}.

''' + '''
`tables.csv` is the numeric source for the dissertation tables; each comparison also has a LaTeX longtable. `results.json` retains counts, intervals, diagnostics and definitions. Import only the required rows, not every auxiliary diagnostic. Undefined means remain null/--, with a reason.

Primary (ego-action branching): monotonicity, responsiveness, locality, action_direction, speed_response. Primary (target-actor intervention, reference-free, matched control/treated pair): target_responsiveness, direction_consistency, ego_policy_consistency, defined only on pair-valid queries; `L4_pair_complete`, `L4_pair_valid`, `L4_control_valid`, `L4_treated_valid` and `L4_pair_axes_defined` are indicator means (the latter means all three axes are defined). Secondary: Branch-set reference F1, CAS (treated-chain agreement of pair-valid answers), and supplementary CTMS; `L4_control_similarity` and `L4_treated_similarity` are own-chain gated reference scores; their valid-only variants retain each chain independently of the other. The control score is never folded into CTMS. Policy reads declared action_type; kinematic diagnostics are separate. Intervention direction uses T0 axes on nuScenes and a control-target heading proxy on CARLA; steering is not interpreted as a constant lateral velocity. Each mean reports defined n and total N. Valid-output agreement `L3_valid_only`/`L4_valid_only` (K), `<task>_valid_output_deficit` (1-K) and `<task>_compliance_deficit` (K-R) are defined only for scenarios with at least one valid unit of that task; `<task>_zero_valid` reports the share of scenarios with none (their whole loss is a compliance loss, not measured disagreement). `valid_output_vs_ceiling` rows report `<task>_ceiling_deficit` = 1-K of the reference scored as its own prediction (non-terminal branches cannot reach 1 on the terminal component) and `<task>_valid_output_gap_to_ceiling` = K(oracle)-K(model), paired; read 1-K against that ceiling, not against 1. Passing the configured gates is not compliance with every prompt rule: a missing branch, a missing actor or an in-vocabulary but unconventional label passes the gates and lands in 1-K; the chain diagnostics and node-label counts separate those. Paired differences use shared defined scenario IDs. `scores_common_subset` rows summarise every model configuration on the scenarios where all of them define the axis (`n_common`); use those rows for cross-model rankings, and the plain `scores` rows for each system's own coverage. `collapse_rate` (endpoints supplied but coincident, over scenarios where responsiveness is defined), `endpoint_shortfall_rate` (too few usable endpoints, responsiveness undefined), `branch_shortfall_rate` and `<axis>_floor_rate` are indicator means with intervals. Cross-split differences bootstrap each split independently. MDD uses observed variance and is descriptive, not evidence of a prior power calculation. Length-cap hits are not automatically parser failures. API output tokens and thinking tokens are reported separately.

The `action_profile_kinematic` control applies the published action profiles to the supplied root and emits every declared branch, so every axis the scene allows is defined for it (a cut-in template has no lateral action; a scene without static actors has no locality); report it beside every model as the non-learning floor with its defined-on counts. `floor_relative` rows are each model cell minus a control (paired, every axis; `baseline` names it) and `skill_vs_floor` rows are mean(model-control)/mean(oracle-control) on the reference axes, None with a reason when the control already matches the oracle. Two floors are reported: `action_profile_kinematic` (published ego profiles only; other actors at root velocity; twin intervention profile) and `action_profile_published`, which also applies the per-scene hazard manoeuvre and intervention magnitudes printed in the prompt. A model above the first but not the second has beaten the ego-profile rule, not arithmetic on all the information it was given; claims of exceeding non-learning arithmetic refer to the second. Gate diagnostic counts in results.json include all-triggered and first-diagnostic failures; optional intervention scene failures do not imply invalidity. Locality/monotonicity may be unavailable for the single-branch and copy controls. Cross-split excess rows are descriptive domain differences: the controls read no images, so they cannot control for visual domain shift, and the rows do not identify contamination. Probe self-reports do not verify scene identity or memorisation.

## Evidence that cannot be invented

Consult ../human/status.json and ../status.json. Author review, interpretation, release URL, invoices and missing dataset-generation or rejected-candidate sensitivity records require actual evidence. In author_review mode the external-rater study was not conducted; descriptive observations do not establish independent human validation. Legacy external-rater campaigns retain their separate study requirements. No automatic command edits final_dissertation.tex or fills missing evidence with zero.
''')
    print(f'Wrote {len(records)} result rows to {out}')


def build_report(cfg, frozen):
    root=Path(cfg['campaign_dir']); out=root/'report';out.mkdir(parents=True,exist_ok=True)
    boot=cfg['statistics']['n_boot']; seed=cfg['statistics']['seed']; records=[]; cells={}; summaries={}
    for cell in matrix_cells(cfg)+baseline_cells(cfg):
        name=cell.get('configuration',cell['backend']); key=(cell['split'],name,cell['condition'])
        data=collect_cell(cell,frozen['datasets'][cell['split']],split_path=cfg['splits'][cell['split']]['path'],
                          frozen_sha256=sha(root/'frozen.json'));cells[key]=data
        rows,summaries['/'.join(key)]=cell_rows(key,data,boot,seed); records+=rows
    model_names=sorted({c['configuration'] for c in matrix_cells(cfg)})
    records+=comparison_rows(cells,model_names,cfg['splits'],boot,seed,
                             (('cosmos-reason2-8b','qwen3-vl-8b-instruct','posttraining_effect'),))
    probes={p.parent.name:read_json(p) for p in (root/'probes').glob('*/report.json')}
    write_report(out,sha(root/'frozen.json'),records,summaries,probes,boot,seed)
