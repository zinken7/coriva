"""Research-logic properties of the pipeline.

A malformed unit (`[null]`, a null node, a null actor) survives the
parser so the schema gate can fail it; every consumer after the scorer
(coherence, label diagnostics, chain diagnostics, report collection) must
treat it as a missing value and finish, without zeroing the other layer.
Every file the prompt builder reads is part of the frozen input
identity, so a byte that changes the prompt changes the fingerprint.
"""
import copy
import json
import os
import sys
from pathlib import Path

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from tests.test_evaluation_campaign import make_protocol_dataset  # noqa: E402


def _pipeline(tmp_path, name, mutate=None):
    from analysis.evaluation_campaign import dataset_inventory, score_cell, write_json, write_campaign_identity
    from analysis.evaluation_report import collect_cell
    from analysis.run_model import run_model
    base = make_protocol_dataset(tmp_path / name, n=1)
    rows = dataset_inventory(base, 1, 'nuscenes')
    out = tmp_path / name / 'run'; out.mkdir(); write_json(out / 'frozen.json', {})
    cell = dict(split='nuscenes', backend='oracle', condition='video', out_dir=str(out / 'cell'))
    cfg = {'campaign_dir': str(out), 'splits': {'nuscenes': {'path': str(base)}}}
    write_campaign_identity(cfg, cell)
    run_model(str(base), cell['out_dir'], backend='oracle')
    if mutate is not None:
        for layer in ('l3_pre.json', 'l4_pre.json'):
            path = Path(cell['out_dir']) / 'scenario_1' / layer
            doc = json.loads(path.read_text())
            mutate(layer, doc)
            path.write_text(json.dumps(doc))
    score_cell(cfg, cell, {'datasets': {'nuscenes': rows}})
    return collect_cell(cell, rows, split_path=base)


def _null_l3(layer, doc):
    if layer == 'l3_pre.json':
        doc['predicted_branches'].append(None)                       # null unit
        doc['predicted_branches'][0]['chain'].append(None)           # null node
        doc['predicted_branches'][0]['chain'][0]['actors_update'].append(None)  # null actor
        doc['predicted_branches'].append({'intervention': None, 'chain': 'x'})  # wrong types


def _null_l4(layer, doc):
    if layer == 'l4_pre.json':
        doc['predicted_counterfactuals'].append(None)
        doc['predicted_counterfactuals'].append({'intervention': {'intervention_type': 'npc_sudden_brake'},
                                                 'chain': [None]})


def test_malformed_units_reach_the_report_without_zeroing_the_other_layer(tmp_path):
    oracle = _pipeline(tmp_path, 'oracle')
    with_l3_nulls = _pipeline(tmp_path, 'null_l3', _null_l3)
    with_l4_nulls = _pipeline(tmp_path, 'null_l4', _null_l4)
    for data in (oracle, with_l3_nulls, with_l4_nulls):
        assert set(data['metrics']['CTMS']) == {'scenario_1'}
        assert 'l3_branch_completeness' in data['metrics']   # chain diagnostics ran
    # A null L3 unit is a compliance failure of that layer only: the branch
    # that carries the null node fails its gate, the other branches keep
    # their validity, and L4 is untouched.
    assert with_l3_nulls['metrics']['Counterfactual_Alignment_Score'] == oracle['metrics']['Counterfactual_Alignment_Score']
    assert with_l3_nulls['compliance']['L3']['scenario_1'][0] == oracle['compliance']['L3']['scenario_1'][0] - 1
    # A null L4 unit leaves the L3 scores and every coherence axis untouched.
    assert with_l4_nulls['metrics']['Causal_Foresight_F1'] == oracle['metrics']['Causal_Foresight_F1']
    for axis in ('monotonicity', 'responsiveness', 'action_direction', 'speed_response'):
        assert with_l4_nulls['metrics'][axis] == oracle['metrics'][axis]
        assert with_l3_nulls['metrics'][axis] == oracle['metrics'][axis]
    assert with_l4_nulls['metrics']['L4_first_node_label_match'] == oracle['metrics']['L4_first_node_label_match']


def test_coherence_pairs_skip_malformed_units_but_preserve_node_positions():
    from analysis.causal_coherence import pairs_from_prediction, coherence_from_pairs
    node = {'depth': 1, 'event_type': 'maintain_course',
            'actors_update': [None, {'id': 'ego', 'role_type': 'ego_agent', 'x': 5.0, 'y': 0.0, 'vx': 10.0, 'vy': 0.0, 'speed': 10.0}]}
    pred = {'predicted_branches': [
        None, 'text', {'intervention': None, 'chain': [node]},
        {'intervention': {'intervention_type': 'do_ego_no_action', 'target_actor_id': 'ego'}, 'chain': [None, node]},
        {'intervention': {'intervention_type': 'do_ego_comfortable_brake', 'target_actor_id': 'ego'}, 'chain': 'not a list'},
        {'intervention': {'intervention_type': 'do_ego_comfortable_brake', 'target_actor_id': 'ego'}, 'chain': [None]},
    ]}
    pairs = pairs_from_prediction(pred)
    assert [iv for iv, _ in pairs] == ['do_ego_no_action']
    assert pairs[0][1] == [None, node]
    root = {'actors': [{'id': 'ego', 'role_type': 'ego_agent', 'x': 0.0, 'y': 0.0, 'vx': 10.0, 'vy': 0.0, 'speed': 10.0}]}
    out = coherence_from_pairs(pairs, root)
    assert out['n_interventions'] == 1 and out['responsiveness'] is None


def test_prompt_source_config_is_part_of_the_frozen_input_identity(tmp_path):
    from analysis.smoke_admission import scenario_input_fingerprint, input_file_fingerprint
    from analysis.run_model import run_model
    base = make_protocol_dataset(tmp_path, split='carla', n=1)
    scen = base / 'scenario_1'
    (scen / 'scenario_config.json').unlink()
    without, files = scenario_input_fingerprint(str(scen))
    assert 'scenario_config.json' not in [f['path'] for f in files]
    hashes = {}
    for decel in (-5.0, -6.0):
        (scen / 'scenario_config.json').write_text(json.dumps({'env_behavior': {'npc_1': {'accel': decel, 'lat': 0.0}}}))
        fp, files = scenario_input_fingerprint(str(scen))
        assert [f['path'] for f in files][6] == 'scenario_config.json'
        assert [f['path'] for f in files][-8:] == [f'frames/{i:03d}.jpg' for i in range(8)]
        out = tmp_path / f'run_{int(abs(decel))}'
        run_model(str(base), str(out), backend='oracle')
        prompt_hash = json.loads((out / 'scenario_1' / 'run_manifest.json').read_text())['prompt_sha256']
        hashes[decel] = (fp, prompt_hash)
    assert hashes[-5.0][1] != hashes[-6.0][1], 'the config must change the prompt for this test to mean anything'
    assert hashes[-5.0][0] != hashes[-6.0][0]
    assert without not in (hashes[-5.0][0], hashes[-6.0][0])
    # The optional asset may only sit between the required assets and the frames.
    moved = files[:6] + files[7:] + [files[6]]
    with pytest.raises(ValueError, match='reordered'):
        input_file_fingerprint(moved)
    assert input_file_fingerprint(files) == hashes[-6.0][0]


# ── One speed-resolution rule for the primary axis and the gates ─────────

def _straight_branch(iv, x_end, vx, speed_scalar, depth=6):
    nodes = []
    for d in range(1, depth + 1):
        f = d / depth
        nodes.append({'depth': d, 'timestamp': 0.5 * d, 'event_type': 'maintain_course',
                      'actors_update': [{'id': 'ego', 'entity_type': 'ego_vehicle', 'role_type': 'ego_agent',
                                         'x': x_end * f, 'y': 0.0, 'vx': vx, 'vy': 0.0, 'speed': speed_scalar}]})
    return iv, nodes


def test_speed_response_reads_the_velocity_vector_like_the_scorer():
    """Same positions, same (vx, vy) = (15, 0) throughout; only the scalar
    differs. The scorer's canonical form replaces the scalar by |(vx, vy)|
    (within its tolerance), so the primary axis must give one verdict."""
    from analysis.causal_coherence import speed_response, _actor_speed
    root = {'actors': [{'id': 'ego', 'role_type': 'ego_agent', 'x': 0.0, 'y': 0.0, 'vx': 15.0, 'vy': 0.0, 'speed': 15.0}]}
    honest = [_straight_branch('do_ego_no_action', 45.0, 15.0, 15.0), _straight_branch('do_ego_comfortable_brake', 45.0, 15.0, 15.0)]
    declared = [_straight_branch('do_ego_no_action', 45.0, 15.0, 15.0), _straight_branch('do_ego_comfortable_brake', 45.0, 15.0, 13.8)]
    assert speed_response(honest, root) == pytest.approx(0.0)
    assert speed_response(declared, root) == pytest.approx(0.0)
    # A vector that brakes is read as braking whatever the scalar says.
    braked = [_straight_branch('do_ego_comfortable_brake', 30.0, 6.0, 15.0)]
    assert speed_response(braked, root) == pytest.approx(1.0)
    # The scalar remains the fallback when no vector is reported.
    assert _actor_speed({'speed': 4.0}) == 4.0 and _actor_speed({'vx': 3.0, 'vy': 4.0, 'speed': 9.0}) == 5.0
    assert _actor_speed({'vx': 'x', 'vy': 0.0, 'speed': 2.0}) == 2.0


# ── Direction is compared at matched forward progression ─────────────────

def _twin_curve_pairs(base_steer, actions=('do_ego_no_action', 'do_ego_comfortable_brake', 'do_ego_swerve_left')):
    from simulation.kinematic_twin import build_twin_branch
    root = {'depth': 0, 'timestamp': 0.0, 'event_type': 'observed_state',
            'actors': [{'id': 'ego', 'entity_type': 'ego_vehicle', 'role_type': 'ego_agent',
                        'x': 0.0, 'y': 0.0, 'vx': 15.0, 'vy': 0.0, 'speed': 15.0, 'length': 4.5, 'width': 2.0}]}
    branches = [build_twin_branch(str(i), i == 0, a, root, 6, 0.5, 0.5, model='bicycle', base_steer=base_steer)
                for i, a in enumerate(actions)]
    assert all(len(b['chain']) == 6 for b in branches)
    return root, [(b['intervention']['intervention_type'], b['chain']) for b in branches]


@pytest.mark.parametrize('base_steer', [-0.05, -0.03, 0.0, 0.03, 0.05])
def test_twin_swerve_on_a_curve_is_read_as_the_correct_direction(base_steer):
    """The twin's own three branches on a curve: the braking branch stops
    earlier along the curve, so an endpoint comparison would read the swerve
    as the wrong side of it; matched progression reads it correctly."""
    from analysis.causal_coherence import action_direction, coherence_from_pairs
    root, pairs = _twin_curve_pairs(base_steer)
    assert action_direction(pairs, root) == pytest.approx(1.0)
    out = coherence_from_pairs(pairs, root)
    assert out['monotonicity'] == 1.0 and out['speed_response'] == 1.0


def test_twin_wrong_way_swerve_on_a_curve_is_still_read_as_wrong():
    from analysis.causal_coherence import action_direction
    root, pairs = _twin_curve_pairs(-0.03)
    mislabelled = [('do_ego_swerve_right' if iv == 'do_ego_swerve_left' else iv, ch) for iv, ch in pairs]
    assert action_direction(mislabelled, root) == pytest.approx(0.0)


def test_direction_with_a_shorter_lateral_branch_interpolates_the_neutral_path():
    """brake_and_swerve ends before the no-action branch: the neutral path is
    interpolated at the lateral branch's endpoint."""
    from analysis.causal_coherence import action_direction
    root, pairs = _twin_curve_pairs(-0.03, ('do_ego_no_action', 'do_ego_brake_and_swerve_left'))
    assert action_direction(pairs, root) == pytest.approx(1.0)


def test_direction_falls_back_to_endpoints_when_progression_is_not_monotone():
    from analysis.causal_coherence import action_direction
    root = {'actors': [{'id': 'ego', 'role_type': 'ego_agent', 'x': 0.0, 'y': 0.0, 'vx': 5.0, 'vy': 0.0, 'speed': 5.0}]}
    reversing = _straight_branch('do_ego_no_action', 10.0, 5.0, 5.0)[1]
    reversing[-1]['actors_update'][0]['x'] = 2.0   # ego goes backwards at the last node
    swerve = _straight_branch('do_ego_swerve_left', 10.0, 5.0, 5.0)[1]
    for n in swerve:
        n['actors_update'][0]['y'] = -3.0
    assert action_direction([('do_ego_no_action', reversing), ('do_ego_swerve_left', swerve)], root) == pytest.approx(1.0)


def test_coherence_report_carries_its_version():
    from analysis.causal_coherence import COHERENCE_VERSION
    assert COHERENCE_VERSION == 'coriva-coherence-1.2'


# ── One output contract per split for the factual branch ─────────────────

def _recorded_scene():
    """nuScenes-style reference: the recorded ego brakes at -2 m/s^2 (band of
    comfortable_brake, published profile -3) and the recorded lead keeps its
    speed, while the generated branches script the lead at -4 m/s^2."""
    import copy
    from simulation.kinematic_twin import roll_branch, build_twin_gt
    from simulation.protocol import PROTOCOL_VERSION
    root = {'depth': 0, 'timestamp': 0.0, 'event_type': 'observed_state', 'actors': [
        dict(id='ego', entity_type='ego_vehicle', role_type='ego_agent', x=0.0, y=0.0, vx=15.0, vy=0.0, speed=15.0, length=4.5, width=2.0),
        dict(id='npc_1', entity_type='vehicle', role_type='lead_vehicle', x=40.0, y=0.0, vx=12.0, vy=0.0, speed=12.0, length=4.5, width=2.0)]}
    recorded = roll_branch(root['actors'], 'do_ego_no_action', 6, 0.5)
    for i, states in enumerate(recorded, 1):
        t = 0.5 * i
        states['ego'].update(x=15 * t - t * t, vx=15 - 2 * t, speed=15 - 2 * t)
    spec = dict(scenario_id='nuscenes_0001', n_steps=6, dt=0.5, sample_interval=0.5, factual_states=copy.deepcopy(recorded),
                env_behavior={'npc_1': {'accel': -4.0, 'lat': 0.0}},
                branches=[dict(branch_id='F', is_factual=True, intervention_type='do_ego_comfortable_brake'),
                          dict(branch_id='CF1', is_factual=False, intervention_type='do_ego_no_action'),
                          dict(branch_id='CF2', is_factual=False, intervention_type='do_ego_swerve_left')])
    gt = build_twin_gt(root, spec)
    gt.update(scenario_id='nuscenes_0001', protocol=dict(protocol_version=PROTOCOL_VERSION, track='primary',
                                                       future_steps=6, split='v1.0-trainval'))
    return root, recorded, spec, gt


def _l3(gt_branches, pred_branches, root):
    from ctms.ctms import load_ctms_environment, score_layer3
    env = load_ctms_environment('ctms/ontology.yaml', 'ctms/ctms_config.yaml', 'ctms/schema_branch.json', 'ctms/schema_counterfactual.json')
    return score_layer3(gt_branches, pred_branches, env['ontology'], env['config'], env['branch_schema'], root=root)


def test_factual_contract_has_one_compliant_answer_per_branch(tmp_path):
    """A program that follows the stated contract (recording for the factual
    branch, published profiles and scripted hazard elsewhere) reaches the
    reference ceiling; applying the profile to the factual branch too, which
    the old prompt demanded, scores below it although the recording is
    within the fidelity threshold."""
    import copy, json
    from simulation.kinematic_twin import build_twin_branch
    from simulation.nuscenes_adapter import infer_factual_action, factual_fidelity_error
    from simulation.observed_history import write_protocol_sidecars, factual_branch_contract
    from simulation.build_pilot import build_prompt
    root, recorded, spec, gt = _recorded_scene()
    assert infer_factual_action([15.0] + [s['ego']['speed'] for s in recorded], 0.5) == 'do_ego_comfortable_brake'
    assert factual_fidelity_error(root, recorded, 'do_ego_comfortable_brake', 0.5, model='bicycle') < 2.0
    factual = next(b for b in gt['branches'] if b['is_factual'])
    generated = [b for b in gt['branches'] if not b['is_factual']]
    lead = lambda b: next(a for a in b['chain'][-1]['actors_update'] if a['id'] == 'npc_1')['speed']
    assert lead(factual) == pytest.approx(12.0) and all(lead(b) < 12.0 for b in generated)  # hazard scripted only where generated

    ceiling = _l3(gt['branches'], copy.deepcopy(gt['branches']), root)['F1_cf']
    def rollout(action, is_factual):
        return build_twin_branch('p_' + action, is_factual, action, root, 6, 0.5, 0.5, env_behavior=spec['env_behavior'])
    contract = [copy.deepcopy(factual)] + [rollout(b['intervention']['intervention_type'], False) for b in generated]
    old_instruction = [rollout('do_ego_comfortable_brake', True)] + contract[1:]
    assert _l3(gt['branches'], contract, root)['F1_cf'] == pytest.approx(ceiling)
    assert _l3(gt['branches'], old_instruction, root)['F1_cf'] < ceiling - 0.1

    # The released query states the contract, the inventory checks it and the prompt prints it.
    contract_fields = factual_branch_contract(gt)
    assert contract_fields == {'factual_action': 'do_ego_comfortable_brake', 'factual_reference': 'recorded_future'}
    ontology = json.loads((tmp_path / 'o.json').write_text('{}') and '{}')
    query = write_protocol_sidecars(str(tmp_path), gt, ontology)
    assert {k: query[k] for k in contract_fields} == contract_fields
    text = build_prompt(root, 'nuscenes_0001', interventions=['npc_sudden_brake'], critical_actor_id='npc_1',
                        l3_actions=query['l3_ego_actions'], query=query, split='v1.0-trainval',
                        hazard_profile={'actor_id': 'npc_1', 'accel': -4.0, 'lat': 0.0})
    section = text.split('## Factual branch', 1)[1].split('## Task — Target-actor intervention', 1)[0]
    assert '`do_ego_comfortable_brake` is the FACTUAL branch' in section
    assert 'RECORDED continuation' in section and '-6.0 < mean acceleration <= -1.5' in section
    assert 'hazard actor\'s recorded motion need not follow the scripted manoeuvre' in section
    assert 'In every GENERATED (non-factual) ego-action branch' in text
    carla_gt = copy.deepcopy(gt); carla_gt['protocol']['split'] = 'carla'
    assert factual_branch_contract(carla_gt)['factual_reference'] == 'executed_action_profile'
    carla = build_prompt(root, 'carla_0001', interventions=['npc_sudden_brake'], critical_actor_id='npc_1',
                         l3_actions=query['l3_ego_actions'], query=dict(query, factual_reference='executed_action_profile'),
                         split='carla', hazard_profile={'actor_id': 'npc_1', 'accel': -6.0, 'lat': 0.0})
    assert 'executed it with the published profile' in carla and 'In EVERY ego-action branch' in carla
    assert 'RECORDED continuation' not in carla


def test_inventory_rejects_a_query_whose_factual_contract_differs_from_the_reference(tmp_path):
    import json
    from analysis.evaluation_campaign import dataset_inventory
    base = make_protocol_dataset(tmp_path, n=1)
    q = base / 'scenario_1' / 'query.json'
    doc = json.loads(q.read_text())
    gt = json.loads((base / 'scenario_1' / 'ground_truth.json').read_text())
    factual = next(b['intervention']['intervention_type'] for b in gt['branches'] if b.get('is_factual'))
    doc.update(factual_action=factual, factual_reference='recorded_future'); q.write_text(json.dumps(doc))
    dataset_inventory(base, 1, 'nuscenes')
    doc.update(factual_reference='executed_action_profile'); q.write_text(json.dumps(doc))
    with pytest.raises(ValueError, match='factual-branch contract'):
        dataset_inventory(base, 1, 'nuscenes')


# ── Valid-output agreement is conditional on a valid unit ────────────────

def test_zero_valid_scenarios_leave_k_and_both_deficits_undefined(tmp_path):
    def empty_l4(layer, doc):
        if layer == 'l4_pre.json':
            for c in doc['predicted_counterfactuals']:
                for key in _L4_CHAIN_KEYS:
                    if key in c:
                        c[key] = []
    oracle = _pipeline(tmp_path, 'oracle')
    none_valid = _pipeline(tmp_path, 'none_valid', empty_l4)
    m = none_valid['metrics']
    assert m['L4_zero_valid']['scenario_1'] == 1 and none_valid['compliance']['L4']['scenario_1'][0] == 0
    assert m['L4_valid_only']['scenario_1'] is None
    assert m['L4_valid_output_deficit']['scenario_1'] is None and m['L4_compliance_deficit']['scenario_1'] is None
    assert m['Counterfactual_Alignment_Score']['scenario_1'] == 0.0      # R stays measured
    o = oracle['metrics']
    assert o['L4_zero_valid']['scenario_1'] == 0 and o['L3_zero_valid']['scenario_1'] == 0
    # The reference as its own prediction does not reach K = 1: that is the scale's ceiling.
    assert 0 < o['L3_valid_output_deficit']['scenario_1'] < 0.1
    assert o['L3_compliance_deficit']['scenario_1'] == pytest.approx(0.0)


# ═════════════════════════════════════════════════════════════════════════
# Malformed-value robustness, self-agreement, export and version identity
# ═════════════════════════════════════════════════════════════════════════

_BAD_VALUES = (None, 'bad', 42, 1.5, [], {}, [None], ['x'], {'k': 1}, '42', True)
# Pair protocol: an L4 unit carries two chains (docs/l4_pair_protocol.md §2).
_L4_CHAIN_KEYS = ('treated_chain', 'control_chain')


def _unit_chains(layer, unit):
    """The chain lists of one predicted unit: `chain` for an L3 branch, the
    treated and control chains (present ones) for an L4 pair."""
    keys = ('chain',) if layer == 'l3_pre.json' else _L4_CHAIN_KEYS
    return [unit[k] for k in keys if isinstance(unit.get(k), list)]


def _fuzz_units(gt, l3, l4):
    """Every nested field of a valid output, replaced by every malformed
    value in turn (both chains of an L4 pair)."""
    import copy
    yield 'oracle', l3, l4
    for layer_name, key in (('l3', 'predicted_branches'), ('l4', 'predicted_counterfactuals')):
        base = l3 if layer_name == 'l3' else l4
        chain_keys = ('chain',) if layer_name == 'l3' else tuple(
            k for k in _L4_CHAIN_KEYS if isinstance(base[key][0].get(k), list))
        for bad in _BAD_VALUES:
            d = copy.deepcopy(base); d[key] = bad; yield f'{layer_name}.{key}={bad!r}', (d if layer_name == 'l3' else l3), (d if layer_name == 'l4' else l4)
            for field in ('intervention', 'is_factual', 'branch_id') + chain_keys:
                d = copy.deepcopy(base); d[key][0][field] = bad
                yield f'{layer_name}.unit.{field}={bad!r}', (d if layer_name == 'l3' else l3), (d if layer_name == 'l4' else l4)
            for field in ('intervention_type', 'target_actor_id', 'trigger_depth'):
                d = copy.deepcopy(base); d[key][0]['intervention'][field] = bad
                yield f'{layer_name}.intervention.{field}={bad!r}', (d if layer_name == 'l3' else l3), (d if layer_name == 'l4' else l4)
            for chain_key in chain_keys:
                for field in ('depth', 'timestamp', 'event_type', 'action_type', 'actors_update', 'terminal'):
                    d = copy.deepcopy(base); d[key][0][chain_key][0][field] = bad
                    yield f'{layer_name}.{chain_key}.node.{field}={bad!r}', (d if layer_name == 'l3' else l3), (d if layer_name == 'l4' else l4)
                for field in ('id', 'x', 'y', 'vx', 'vy', 'speed', 'role_type', 'entity_type'):
                    d = copy.deepcopy(base); d[key][0][chain_key][0]['actors_update'][0][field] = bad
                    yield f'{layer_name}.{chain_key}.actor.{field}={bad!r}', (d if layer_name == 'l3' else l3), (d if layer_name == 'l4' else l4)


def test_every_consumer_after_the_parser_survives_every_malformed_nested_field(tmp_path):
    """In-process sweep over every field x every malformed value through the
    scorer, the coherence pairs, the label diagnostic and the chain
    diagnostics. A malformed unit is gated or ignored, never an exception."""
    import json
    from analysis.run_model import run_model
    from ctms.ctms import load_ctms_environment
    from run_benchmark import score_sample
    from analysis.causal_coherence import pairs_from_prediction, coherence_from_pairs
    from analysis.chain_diagnostics import scenario_diagnostics
    from analysis.evaluation_report import node_label_diagnostics
    from scripts.assemble_dataset import merged_prediction
    from collections import defaultdict
    base = make_protocol_dataset(tmp_path, n=1)
    out = tmp_path / 'oracle'; run_model(str(base), str(out), backend='oracle')
    gt = json.loads((base / 'scenario_1' / 'ground_truth.json').read_text())
    l3 = json.loads((out / 'scenario_1' / 'l3_pre.json').read_text())
    l4 = json.loads((out / 'scenario_1' / 'l4_pre.json').read_text())
    env = load_ctms_environment('ctms/ontology.yaml', 'ctms/ctms_config.yaml', 'ctms/schema_branch.json', 'ctms/schema_counterfactual.json')
    idmap = {gt['scenario_id']: 'scenario_1'}
    count = 0
    for label, m3, m4 in _fuzz_units(gt, l3, l4):
        count += 1
        try:
            pred = merged_prediction(m3, m4)
            score_sample(gt, pred, env)
            coherence_from_pairs(pairs_from_prediction(m3), gt['observed_state'])
            scenario_diagnostics(gt, m3, m4)
            node_label_diagnostics({'scenarios': [{'scenario_id': gt['scenario_id'], 'samples': [{'ground_truth': gt}]}]},
                                   {'scenarios': [{'scenario_id': gt['scenario_id'], 'samples': [{'prediction': pred}]}]},
                                   idmap, defaultdict(dict))
        except Exception as exc:  # noqa: BLE001
            raise AssertionError(f'{label}: {type(exc).__name__}: {exc}') from exc
    assert count > 400


@pytest.mark.parametrize('label,mutate', [
    ('l4_intervention_string', lambda layer, doc: doc['predicted_counterfactuals'][0].__setitem__('intervention', 'bad') if layer == 'l4_pre.json' else None),
    ('l3_actor_container_number', lambda layer, doc: doc['predicted_branches'][0]['chain'][0].__setitem__('actors_update', 42) if layer == 'l3_pre.json' else None),
    ('l3_intervention_type_list', lambda layer, doc: doc['predicted_branches'][0]['intervention'].__setitem__('intervention_type', ['x']) if layer == 'l3_pre.json' else None),
    ('l4_chain_dict', lambda layer, doc: doc['predicted_counterfactuals'][0].__setitem__('chain', {'a': 1}) if layer == 'l4_pre.json' else None),
    ('l3_branches_string', lambda layer, doc: doc.__setitem__('predicted_branches', 'oops') if layer == 'l3_pre.json' else None),
    ('l3_branches_number', lambda layer, doc: doc.__setitem__('predicted_branches', 42) if layer == 'l3_pre.json' else None),
    ('l4_counterfactuals_number', lambda layer, doc: doc.__setitem__('predicted_counterfactuals', 42) if layer == 'l4_pre.json' else None),
    ('l3_first_node_null', lambda layer, doc: doc['predicted_branches'][0]['chain'].__setitem__(0, None) if layer == 'l3_pre.json' else None),
])
def test_malformed_nested_fields_reach_the_report(tmp_path, label, mutate):
    data = _pipeline(tmp_path, label, mutate)
    assert set(data['metrics']['CTMS']) == {'scenario_1'}
    assert 'l3_branch_completeness' in data['metrics']


# ── A root echo changes no diagnostic ────────────────────────────────────

def test_root_echo_changes_no_chain_diagnostic(tmp_path):
    import copy, json
    def echo(layer, doc):
        key = 'predicted_branches' if layer == 'l3_pre.json' else 'predicted_counterfactuals'
        root = json.loads((tmp_path / 'echo' / 'nuscenes' / 'scenario_1' / 'observed_state.json').read_text())
        for b in doc[key]:
            for chain in _unit_chains(layer, b):
                chain.insert(0, copy.deepcopy(root))
    oracle = _pipeline(tmp_path, 'oracle')
    echoed = _pipeline(tmp_path, 'echo', echo)
    from analysis.chain_diagnostics import METRICS
    for key in METRICS:
        assert echoed['metrics'][key] == oracle['metrics'][key], key
    assert echoed['metrics']['l3_ego_error_d1']['scenario_1'] == 0.0
    assert echoed['metrics']['CTMS'] == oracle['metrics']['CTMS']


# ── The reference agrees with itself on every scene ──────────────────────

def test_reference_scores_one_on_lateral_sign_agreement_even_when_it_crosses_or_curves():
    from analysis.chain_diagnostics import scenario_diagnostics
    def scene(ys):
        root = {'event_type': 'observed_state', 'actors': [
            {'id': 'ego', 'role_type': 'ego_agent', 'x': 0.0, 'y': 0.0, 'vx': 10.0, 'vy': 0.0, 'speed': 10.0},
            {'id': 'ped', 'entity_type': 'pedestrian', 'role_type': 'crossing_actor', 'x': 20.0, 'y': ys[0], 'vx': 0.0, 'vy': 0.0, 'speed': 0.0}]}
        chain = [{'depth': i, 'timestamp': 0.5 * i, 'event_type': 'maintain_course', 'action_type': 'ego_no_action',
                  'actors_update': [{'id': 'ego', 'role_type': 'ego_agent', 'x': 5.0 * i, 'y': 0.0, 'vx': 10.0, 'vy': 0.0, 'speed': 10.0},
                                    {'id': 'ped', 'entity_type': 'pedestrian', 'role_type': 'crossing_actor', 'x': 20.0, 'y': y, 'vx': 0.0, 'vy': 4.0, 'speed': 4.0}]}
                 for i, y in enumerate(ys[1:], 1)]
        iv = {'intervention_type': 'pedestrian_enters_road', 'target_actor_id': 'ped', 'trigger_depth': 1}
        gt = {'scenario_id': 's', 'observed_state': root, 'branches': [],
              'l4_queries': [{'intervention_type': 'pedestrian_enters_road', 'counterfactual_chain': {'intervention': iv, 'chain': chain}}]}
        import copy
        return gt, {'predicted_counterfactuals': [{'intervention': iv, 'chain': copy.deepcopy(chain)}]}
    for ys in ([-4, -2, 0, 2, 4, 6, 8], [-4.07, -3.95, -4.15, -4.71, -5.72, -7.78, -10.14], [3.5, 5.66, 12.4, 22.17]):
        gt, l4 = scene(ys)
        assert scenario_diagnostics(gt, None, l4)['l4_treated_lateral_sign_vs_reference'] == 1.0
    gt, l4 = scene([-4, -2, 0, 2, 4, 6, 8])
    for n in l4['predicted_counterfactuals'][0]['chain']:
        n['actors_update'][1]['y'] = -4.0 - 2.0 * n['depth']
    assert scenario_diagnostics(gt, None, l4)['l4_treated_lateral_sign_vs_reference'] == 0.0


# ── B_app scripts nothing in a recorded factual branch ───────────────────

def test_published_control_keeps_the_recorded_factual_branch_unscripted():
    import json
    from analysis.baselines import action_profile_published_text
    root, recorded, spec, gt = _recorded_scene()
    hazard = {'actor_id': 'npc_1', 'accel': -4.0, 'lat': 0.0}
    lead = lambda b: [next(a for a in n['actors_update'] if a['id'] == 'npc_1')['speed'] for n in b['chain']]
    pred = json.loads(action_profile_published_text(gt, hazard_profile=hazard))
    for b in pred['predicted_branches']:
        if b['is_factual']:
            assert lead(b) == pytest.approx([12.0] * len(b['chain']))
        else:
            assert lead(b)[0] == pytest.approx(10.0) and lead(b)[-1] < 12.0
    carla = json.loads(json.dumps(gt)); carla['protocol']['split'] = 'carla'
    pred = json.loads(action_profile_published_text(carla, hazard_profile=hazard))
    assert all(lead(b)[0] == pytest.approx(10.0) for b in pred['predicted_branches'])


# ── The CARLA prompt describes the road-frame offset ─────────────────────

def test_carla_prompt_states_the_road_frame_offset_and_nuscenes_header_states_the_factual_exception():
    from simulation.build_pilot import build_prompt
    root = {'actors': [{'id': 'ego', 'entity_type': 'ego_vehicle', 'role_type': 'ego_agent', 'x': 0, 'y': 0, 'vx': 10, 'vy': 0, 'speed': 10},
                       {'id': 'npc_1', 'entity_type': 'vehicle', 'role_type': 'lead_vehicle', 'x': 20, 'y': 0, 'vx': 8, 'vy': 0, 'speed': 8}]}
    actions = ['do_ego_no_action', 'do_ego_comfortable_brake', 'do_ego_swerve_left']
    q = {'l3_ego_actions': actions, 'l4_interventions': [{'intervention_type': 'npc_sudden_brake', 'target_actor_id': 'npc_1', 'trigger_depth': 1}],
         'l4_ego_policy': 'do_ego_no_action', 'factual_action': 'do_ego_no_action'}
    carla = build_prompt(root, 'carla_0001', interventions=['npc_sudden_brake'], l3_actions=actions, split='carla',
                         query=dict(q, factual_reference='executed_action_profile'))
    assert 'perpendicular to the road centreline' in carla and 'turns with the road' in carla
    assert 'Every declared EGO action is produced with the published parameters' in carla
    nus = build_prompt(root, 'nuscenes_0001', interventions=['npc_sudden_brake'], l3_actions=actions, split='v1.0-trainval',
                       query=dict(q, factual_reference='recorded_future'))
    assert 'Every declared EGO action except the factual branch' in nus
    assert 'perpendicular to the road centreline' not in nus


# ── Freeze verification sees a dependency that appears later ─────────────

def test_freeze_verification_detects_an_added_or_removed_prompt_source(tmp_path):
    import json
    from analysis.evaluation_campaign import dataset_inventory, frozen_inputs, write_json, code_inventory
    def campaign(with_config):
        data = make_protocol_dataset(tmp_path / ('c1' if with_config else 'c0'), split='carla', n=1)
        cfgfile = data / 'scenario_1' / 'scenario_config.json'
        # The optional config may be absent only when the public query itself
        # supplies CARLA's profile. Keep this a membership test, not a missing
        # intervention-contract test.
        queryfile = data / 'scenario_1' / 'query.json'
        query = json.loads(queryfile.read_text())
        query['l4_interventions'][0].update(resolved_accel_mps2=-8.0,
                                          resolved_lateral_velocity_mps=0.0)
        queryfile.write_text(json.dumps(query))
        if with_config:
            cfgfile.write_text(json.dumps({'env_behavior': {'npc_1': {'accel': -5.0, 'lat': 0.0}}}))
        else:
            cfgfile.unlink()
        rows = dataset_inventory(data, 1, 'carla'); camp = data.parent / 'campaign'; camp.mkdir()
        cfg = {'campaign_dir': str(camp), 'splits': {'carla': {'path': str(data)}}, 'smoke': {}, 'statistics': {}, 'probe': {}, 'human': {}, 'validation': {}}
        write_json(camp / 'frozen.json', {'settings': {k: cfg[k] for k in ('splits', 'smoke', 'statistics', 'probe', 'human', 'validation')},
                                          'code_files': code_inventory(), 'datasets': {'carla': rows}})
        frozen_inputs(cfg, verify_files=True)
        return cfg, cfgfile
    cfg, cfgfile = campaign(False)
    cfgfile.write_text(json.dumps({'env_behavior': {'npc_1': {'accel': -5.0, 'lat': 0.0}}}))
    with pytest.raises(ValueError, match='inventory changed.*added'):
        frozen_inputs(cfg, verify_files=True)
    cfg, cfgfile = campaign(True)
    cfgfile.write_text(json.dumps({'env_behavior': {'npc_1': {'accel': -6.0, 'lat': 0.0}}}))
    with pytest.raises(ValueError, match='Frozen dataset changed'):
        frozen_inputs(cfg, verify_files=True)
    cfgfile.unlink()
    with pytest.raises(ValueError, match='inventory changed.*removed'):
        frozen_inputs(cfg, verify_files=True)


# ── The floor identity survives CSV and LaTeX export ─────────────────────

def test_export_keeps_the_baseline_of_every_floor_row(tmp_path):
    import csv
    from analysis.evaluation_report import export_tables
    rows = [dict(split='carla', configuration='QV', condition='video', comparison=kind, metric='CTMS', mean=mean, lo=None, hi=None, n=1, baseline=floor)
            for kind in ('floor_relative', 'skill_vs_floor') for floor, mean in (('action_profile_kinematic', .4), ('action_profile_published', -.2))]
    export_tables(tmp_path, rows)
    got = {(r['comparison'], r['baseline']): float(r['mean']) for r in csv.DictReader((tmp_path / 'tables.csv').open())}
    assert got[('floor_relative', 'action_profile_kinematic')] == .4 and got[('floor_relative', 'action_profile_published')] == -.2
    tex = (tmp_path / 'floor_relative.tex').read_text()
    assert 'vs action\\_\\allowbreak{}profile\\_\\allowbreak{}published' in tex and 'vs action\\_\\allowbreak{}profile\\_\\allowbreak{}kinematic' in tex


# ── The coherence version is enforced when reports are read ──────────────

def test_mixed_coherence_versions_are_refused_and_campaign_identity_is_checked(tmp_path):
    import json
    from analysis.causal_coherence import coherence_condition_deltas, _load_coherence, COHERENCE_VERSION
    from analysis.evaluation_campaign import write_json, sha
    from analysis.evaluation_report import collect_cell
    from analysis.evaluation_campaign import dataset_inventory, score_cell
    from analysis.run_model import run_model
    row = {'scenario_id': 's1', 'monotonicity': 1.0, 'responsiveness': 1.0, 'locality': None, 'action_direction': 1.0, 'speed_response': 1.0, 'structure': 1.0}
    new = {'coherence_version': COHERENCE_VERSION, 'uses_reference_answers': False, 'per_scenario': [row]}
    old = dict(new, coherence_version='coriva-coherence-1.0')
    unversioned = {'uses_reference_answers': False, 'per_scenario': [row]}
    with pytest.raises(ValueError, match='coherence report version'):
        coherence_condition_deltas(old, new, new)
    with pytest.raises(ValueError, match='coherence report version'):
        _load_coherence([new, old])
    with pytest.raises(ValueError, match='coherence report version'):
        _load_coherence(unversioned)
    with pytest.raises(ValueError, match='coherence report version'):
        _load_coherence([new, unversioned])
    assert _load_coherence([new, new])['coherence_version'] == COHERENCE_VERSION
    # Campaign artifacts: a coherence.json of another version, or a
    # score_provenance of another campaign, is refused by the reader.
    base = make_protocol_dataset(tmp_path, n=1); rows = dataset_inventory(base, 1, 'nuscenes')
    out = tmp_path / 'run'; out.mkdir(); write_json(out / 'frozen.json', {'a': 1})
    cell = dict(split='nuscenes', backend='oracle', condition='video', out_dir=str(out / 'cell'))
    cfg = {'campaign_dir': str(out), 'splits': {'nuscenes': {'path': str(base)}}}
    from analysis.evaluation_campaign import write_campaign_identity
    write_campaign_identity(cfg, cell)
    run_model(str(base), cell['out_dir'], backend='oracle')
    score_cell(cfg, cell, {'datasets': {'nuscenes': rows}})
    collect_cell(cell, rows, frozen_sha256=sha(out / 'frozen.json'))
    with pytest.raises(ValueError, match='another campaign'):
        collect_cell(cell, rows, frozen_sha256='0' * 64)
    p = out / 'cell' / 'coherence.json'; doc = json.loads(p.read_text()); doc['coherence_version'] = 'coriva-coherence-1.0'; p.write_text(json.dumps(doc))
    proof = out / 'cell' / 'score_provenance.json'; pr = json.loads(proof.read_text()); pr['coherence_sha256'] = sha(p); proof.write_text(json.dumps(pr))
    with pytest.raises(ValueError, match='coherence.json version'):
        collect_cell(cell, rows)


# ── A shorter non-monotone path falls back to endpoints too ──────────────

def test_direction_fallback_applies_when_the_shorter_path_is_not_monotone():
    from analysis.causal_coherence import _matched_lateral_difference
    a = [(0.0, 0.0), (4.0, 2.5), (2.0, 2.5)]
    b = [(0.0, 0.0), (10.0, 5.0)]
    assert _matched_lateral_difference(a, b) == pytest.approx(-2.5)
    assert _matched_lateral_difference(b, a) == pytest.approx(2.5)
    assert _matched_lateral_difference([(0.0, 0.0), (10.0, -3.0)], [(0.0, 0.0), (10.0, 0.0)]) == pytest.approx(-3.0)  # equal endpoints


# ── Reference integrity at report time ───────────────────────────────────

def test_report_stage_refuses_a_reference_edited_after_scoring(tmp_path):
    """The chain diagnostics read the reference from the split directory; a
    reference that differs from the frozen inventory is refused, so a report
    can never mix two reference versions."""
    import json
    from analysis.evaluation_campaign import dataset_inventory, score_cell, write_json, write_campaign_identity
    from analysis.evaluation_report import collect_cell
    from analysis.run_model import run_model
    base = make_protocol_dataset(tmp_path, n=1); rows = dataset_inventory(base, 1, 'nuscenes')
    out = tmp_path / 'run'; out.mkdir(); write_json(out / 'frozen.json', {})
    cell = dict(split='nuscenes', backend='oracle', condition='video', out_dir=str(out / 'cell'))
    cfg = {'campaign_dir': str(out), 'splits': {'nuscenes': {'path': str(base)}}}
    write_campaign_identity(cfg, cell)
    run_model(str(base), cell['out_dir'], backend='oracle')
    score_cell(cfg, cell, {'datasets': {'nuscenes': rows}})
    before = collect_cell(cell, rows, split_path=base)['metrics']['l3_ego_error_d1']['scenario_1']
    assert before == 0.0
    p = base / 'scenario_1' / 'ground_truth.json'; gt = json.loads(p.read_text())
    for b in gt['branches']:
        for n in b['chain']:
            for a in n['actors_update']:
                if a['id'] == 'ego':
                    a['x'] += 100.0
    p.write_text(json.dumps(gt))
    with pytest.raises(ValueError, match='differs from the frozen inventory'):
        collect_cell(cell, rows, split_path=base)


def test_rescore_refuses_predictions_generated_under_another_campaign(tmp_path):
    from analysis.evaluation_campaign import dataset_inventory, score_cell, write_json, write_campaign_identity
    from analysis.run_model import run_model
    base = make_protocol_dataset(tmp_path, n=1); rows = dataset_inventory(base, 1, 'nuscenes')
    out = tmp_path / 'run'; out.mkdir(); write_json(out / 'frozen.json', {'v': 1})
    cell = dict(split='nuscenes', backend='oracle', condition='video', out_dir=str(out / 'cell'))
    cfg = {'campaign_dir': str(out), 'splits': {'nuscenes': {'path': str(base)}}}
    write_campaign_identity(cfg, cell)
    run_model(str(base), cell['out_dir'], backend='oracle')
    score_cell(cfg, cell, {'datasets': {'nuscenes': rows}})
    write_json(out / 'frozen.json', {'v': 2})       # a new campaign identity, old predictions
    with pytest.raises(ValueError, match='another campaign identity'):
        score_cell(cfg, cell, {'datasets': {'nuscenes': rows}})
    (out / 'cell' / 'campaign_identity.json').unlink()
    with pytest.raises(ValueError, match='no campaign_identity.json'):
        score_cell(cfg, cell, {'datasets': {'nuscenes': rows}})


def test_null_first_node_keeps_later_nodes_at_their_depth(tmp_path):
    def null_first(layer, doc):
        key = 'predicted_branches' if layer == 'l3_pre.json' else 'predicted_counterfactuals'
        for b in doc[key]:
            for chain in _unit_chains(layer, b):
                chain[0] = None
    oracle = _pipeline(tmp_path, 'oracle')
    nulled = _pipeline(tmp_path, 'null_first', null_first)
    for key in ('l3_ego_error_d3', 'l3_ego_error_end', 'l4_target_error_d3', 'l4_target_error_end'):
        assert nulled['metrics'][key] == oracle['metrics'][key], key
    assert nulled['metrics']['l3_ego_error_d1']['scenario_1'] is None   # nothing at depth 1 to compare


def test_carla_prompt_has_one_swerve_rule_and_nuscenes_the_other():
    from simulation.build_pilot import build_prompt
    root = {'actors': [{'id': 'ego', 'entity_type': 'ego_vehicle', 'role_type': 'ego_agent', 'x': 0, 'y': 0, 'vx': 10, 'vy': 0, 'speed': 10},
                       {'id': 'npc_1', 'entity_type': 'vehicle', 'role_type': 'lead_vehicle', 'x': 20, 'y': 0, 'vx': 8, 'vy': 0, 'speed': 8}]}
    actions = ['do_ego_no_action', 'do_ego_comfortable_brake', 'do_ego_swerve_left']
    q = {'l3_ego_actions': actions, 'l4_interventions': [{'intervention_type': 'npc_sudden_brake', 'target_actor_id': 'npc_1', 'trigger_depth': 1}],
         'l4_ego_policy': 'do_ego_no_action', 'factual_action': 'do_ego_no_action'}
    carla = build_prompt(root, 'carla_0001', interventions=['npc_sudden_brake'], l3_actions=actions, split='carla',
                         query=dict(q, factual_reference='executed_action_profile'))
    section = carla.split('## Published motion profiles', 1)[1].split('## Task — Ego-action branching', 1)[0]
    assert 'fixed T=0 frame' not in section and 'perpendicular to the road centreline' in section
    nus = build_prompt(root, 'nuscenes_0001', interventions=['npc_sudden_brake'], l3_actions=actions, split='v1.0-trainval',
                       query=dict(q, factual_reference='recorded_future'))
    section = nus.split('## Published motion profiles', 1)[1].split('## Task — Ego-action branching', 1)[0]
    assert 'fixed T=0 frame' in section and 'perpendicular to the road centreline' not in section


def test_locality_preserves_shared_depths_through_report(tmp_path):
    def mutate(missing):
        def change(layer, doc):
            if layer != 'l3_pre.json':
                return
            for branch in doc['predicted_branches']:
                for node in branch['chain']:
                    for actor in node['actors_update']:
                        if actor['id'] == 'npc_1':
                            # Identical positions across actions at each depth,
                            # different across depths: locality compares actions,
                            # not a static actor's temporal drift.
                            actor.update(role_type='static_barrier', entity_type='barrier',
                                         x=20 + 2 * node['depth'], y=2, vx=0, vy=0, speed=0)
            if missing:
                doc['predicted_branches'][0]['chain'][0] = None
        return change

    intact = _pipeline(tmp_path, 'intact', mutate(False))
    missing = _pipeline(tmp_path, 'missing', mutate(True))
    assert intact['metrics']['locality']['scenario_1'] == 1.0
    assert missing['metrics']['locality'] == intact['metrics']['locality']


def test_carla_swerve_prompt_projects_offset_into_both_t0_coordinates():
    from simulation.build_pilot import build_prompt
    root = {'actors': [{'id': 'ego', 'entity_type': 'ego_vehicle', 'role_type': 'ego_agent',
                        'x': 0, 'y': 0, 'vx': 10, 'vy': 0, 'speed': 10}]}
    prompt = build_prompt(root, 'carla_0001', interventions=[],
                          l3_actions=['do_ego_no_action', 'do_ego_swerve_right'], split='carla')
    section = prompt.split('## Published motion profiles', 1)[1].split('## Task — Ego-action branching', 1)[0]
    assert 'delta_x = -d * sin(theta)' in section
    assert 'delta_y = d * cos(theta)' in section
    assert 'road-following `y` plus that road-frame offset' not in section
    assert '60 degrees' in section and '1.75 m' in section
