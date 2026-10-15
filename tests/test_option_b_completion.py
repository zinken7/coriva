"""Option-B flow regressions, using temporary inputs and no inference service."""
import copy
import json

import pytest

from test_evaluation_campaign import make_protocol_dataset


def test_axis_export_retains_defined_and_total_counts(tmp_path):
    import csv
    from analysis.evaluation_report import mean_stat, paired_stat, export_tables
    result = mean_stat([1.0, None, 0.0], 20)
    assert (result['n'], result['n_total'], result['mean']) == (2, 3, .5)
    paired = paired_stat({'a': 1.0, 'b': None}, {'a': .5, 'b': 1.0}, 20)
    assert (paired['n'], paired['n_total']) == (1, 2)
    export_tables(tmp_path, [dict(metric='target_responsiveness', **result)])
    row = next(csv.DictReader((tmp_path / 'tables.csv').open()))
    assert row['n'] == '2' and row['n_total'] == '3'
    assert '2/3' in (tmp_path / 'scores.tex').read_text()


def test_pair_runner_uses_the_same_public_profile_as_prompt(tmp_path):
    from analysis.intervention_pair import run_l4_pair
    from analysis.run_model import run_model
    from simulation.build_pilot import public_intervention_profiles
    base = make_protocol_dataset(tmp_path, n=1)
    scene = base / "scenario_1"
    config = {"environment_event": {"event_type": "npc_sudden_brake", "params": {"intensity": 5.0}}}
    (scene / "scenario_config.json").write_text(json.dumps(config))
    out = tmp_path / "pred"
    run_model(str(base), str(out), backend="oracle")
    root = json.loads((scene / "observed_state.json").read_text())
    query = json.loads((scene / "query.json").read_text())
    profile = public_intervention_profiles(root, query, config)["npc_sudden_brake"]
    assert profile["accel"] == -5.0
    # Deliberately remove future answers: the metric must still work.
    (scene / "ground_truth.json").unlink()
    row = run_l4_pair(str(base), str(out))["per_scenario"][0]
    assert row["profile"]["accel"] == profile["accel"]


def test_missing_carla_profile_cannot_silently_use_twin_magnitudes(tmp_path):
    from analysis.evaluation_campaign import dataset_inventory
    base = make_protocol_dataset(tmp_path, split='carla', n=1)
    (base / 'scenario_1' / 'scenario_config.json').unlink()
    with pytest.raises(ValueError, match='CARLA L4 requires'):
        dataset_inventory(base, 1, 'carla')


def test_noise_study_perturbs_both_active_l4_references(monkeypatch):
    from analysis.evaluation_validation import full_noise_floor
    node = {"actors_update": [{"id": "ego", "x": 1.0, "y": 2.0}]}
    chain = {"intervention": {"intervention_type": "npc_sudden_brake"}, "chain": [node]}
    gt = {"observed_state": {"actors": []}, "branches": [],
          "l4_queries": [{"counterfactual_chain": chain, "control_chain": copy.deepcopy(chain)}]}
    original = copy.deepcopy(gt)
    seen = []

    def score(perturbed, prediction, env):
        seen.append(perturbed)
        return {"Causal_Foresight_F1": 1.0, "Counterfactual_Alignment_Score": 1.0, "CTMS": 1.0}

    monkeypatch.setattr("run_benchmark.score_sample", score)
    full_noise_floor([gt], [{}], [1.0], {}, n_trials=2)
    assert gt == original
    for perturbed in seen:
        assert perturbed["observed_state"] == gt["observed_state"]
        for key in ("control_chain", "counterfactual_chain"):
            assert perturbed["l4_queries"][0][key]["chain"] != gt["l4_queries"][0][key]["chain"]
            assert perturbed["l4_queries"][0][key]["intervention"] == gt["l4_queries"][0][key]["intervention"]


def test_reference_migration_marks_public_manifest_without_changing_query(tmp_path):
    from test_l4_control_audit import _synthetic_nuscenes_gt, _write_scene
    from scripts.audit_l4_control_references import audit_nuscenes_scene, write_control
    from simulation.l4_control import validate_pair_reference
    from pathlib import Path
    scene = Path(_write_scene(tmp_path, _synthetic_nuscenes_gt()))
    gt = json.loads((scene / 'ground_truth.json').read_text())
    manifest = copy.deepcopy(gt['protocol'])
    manifest['input_extra'] = 'preserve'
    (scene / 'input_manifest.json').write_text(json.dumps(manifest))
    query = {'l4_interventions': [gt['l4_queries'][0]['counterfactual_chain']['intervention']]}
    (scene / 'query.json').write_text(json.dumps(query))
    write_control(str(scene), audit_nuscenes_scene(str(scene)), 'nuscenes')
    migrated = json.loads((scene / 'ground_truth.json').read_text())
    public = json.loads((scene / 'input_manifest.json').read_text())
    validate_pair_reference(migrated, public)
    assert public['input_extra'] == 'preserve'
    assert json.loads((scene / 'query.json').read_text()) == query


def test_migration_cannot_write_only_a_limited_subset(tmp_path):
    import subprocess
    import sys
    result = subprocess.run([sys.executable, 'scripts/audit_l4_control_references.py',
                             '--split', 'nuscenes', '--root', str(tmp_path), '--limit', '1', '--write'],
                            capture_output=True, text=True)
    assert result.returncode == 2 and '--limit' in result.stderr


@pytest.mark.parametrize('steer', [.0, .1, .2, -.2])
def test_twin_lateral_direction_uses_its_declared_t0_frame(steer):
    from test_intervention_pair import _root, _pair, _query, CUT_IN_LEFT
    from analysis.intervention_pair import pair_axes
    root = _root('cutin')
    c, t = _pair(root, 'npc_cut_in_left', CUT_IN_LEFT, model='bicycle', base_steer=steer)
    axes = pair_axes(c, t, root, _query('npc_cut_in_left'),
                     {**CUT_IN_LEFT, 'direction_frame': 't0', 'lateral_realization': 'velocity'})
    assert axes['direction_consistency'] == 1.0


def test_carla_steering_is_not_a_linear_displacement_prediction():
    from test_intervention_pair import _arc_pair, _query
    from analysis.intervention_pair import pair_axes
    root, c, t = _arc_pair(-3.0, n=1)
    axes = pair_axes(c, t, root, _query('npc_cut_in_left'),
                     {'accel': 0.0, 'lat': -.2, 'direction_frame': 'control_target_normal',
                      'lateral_realization': 'steering'})
    assert axes['direction_consistency'] == 1.0
    assert axes['details']['direction_consistency']['expected_magnitude'] is None


def test_policy_is_declared_action_not_identical_state():
    from test_intervention_pair import _root, _pair, _query, BRAKE
    from analysis.intervention_pair import pair_axes
    root = _root('lead')
    c, t = _pair(root, 'npc_sudden_brake', BRAKE)
    for node in t:
        ego = next(a for a in node['actors_update'] if a['id'] == 'ego')
        ego['vx'] = ego['speed'] = 1.0
        ego['y'] += 3.0
    axes = pair_axes(c, t, root, _query('npc_sudden_brake'), BRAKE)
    assert axes['ego_policy_consistency'] == 1.0
    assert axes['details']['ego_kinematics']['speed_constancy_treated'] == 0
    t[0]['action_type'] = 'ego_emergency_brake'
    t[0]['event_type'] = 'ego_brake'
    assert pair_axes(c, t, root, _query('npc_sudden_brake'), BRAKE)['ego_policy_consistency'] == 0.0


def test_missing_policy_is_undefined_and_cannot_be_recovered_from_state():
    from test_intervention_pair import _root, _pair, _query, BRAKE
    from analysis.intervention_pair import pair_axes
    root = _root('lead')
    c, t = _pair(root, 'npc_sudden_brake', BRAKE)
    t[0].pop('action_type')
    assert pair_axes(c, t, root, _query('npc_sudden_brake'), BRAKE)['ego_policy_consistency'] is None


@pytest.mark.parametrize('case', ['oracle', 'missing_control', 'declared_brake', 'duplicate'])
def test_matched_pair_through_assembler_score_and_report(tmp_path, case):
    from test_research_review_20260917 import _pipeline

    def mutate(layer, doc):
        if layer != 'l4_pre.json':
            return
        item = doc['predicted_counterfactuals'][0]
        if case == 'missing_control':
            item.pop('control_chain')
        elif case == 'declared_brake':
            item['treated_chain'][0].update(event_type='ego_brake', action_type='ego_emergency_brake')
        elif case == 'duplicate':
            doc['predicted_counterfactuals'].append(copy.deepcopy(item))

    data = _pipeline(tmp_path, case, mutate)
    m = data['metrics']; sid = 'scenario_1'
    if case == 'oracle':
        assert m['L4_pair_valid'][sid] == 1
        assert m['target_responsiveness'][sid] == m['direction_consistency'][sid] == 1.0
        assert m['ego_policy_consistency'][sid] == 1.0
    elif case == 'missing_control':
        assert m['L4_treated_valid_only'][sid] == 1.0
        assert m['Counterfactual_Alignment_Score'][sid] == 0.0
        assert m['L4_valid_only'][sid] is None and m['L4_zero_valid'][sid] == 1
        assert m['ego_policy_consistency'][sid] is None
    elif case == 'declared_brake':
        assert m['L4_pair_valid'][sid] == 1
        assert m['ego_policy_consistency'][sid] == 0.0
    else:
        assert m['L4_pair_valid'][sid] == 0
        assert m['L4_control_similarity'][sid] == 0.0
        assert m['target_responsiveness'][sid] is None


def test_smoke_archives_pair_observations_without_a_score_cutoff(tmp_path):
    from test_smoke_admission import (_write_cell_output, _repository, SCENARIO, INPUT_FILES,
                                      INPUT_FINGERPRINT, PANEL_PATH)
    from analysis.model_panel import load_model_panel, resolve_configurations
    from analysis.smoke_admission import inspect_cell_output
    from analysis.intervention_pair import L4_PAIR_VERSION
    panel = load_model_panel(PANEL_PATH)
    config = resolve_configurations(panel, ['qwen3-vl-8b-instruct'])[0]
    _write_cell_output(tmp_path, panel, config, 'video')
    (tmp_path / 'l4_pair.json').write_text(json.dumps({
        'l4_pair_version': L4_PAIR_VERSION, 'coverage': {'pair_valid': 0},
        'axes': {'target_responsiveness': {'mean': None, 'n': 0, 'n_pairs': 1}}, 'per_scenario': []}))
    result = inspect_cell_output(panel, config, 'video', SCENARIO, str(tmp_path),
                                 INPUT_FINGERPRINT, _repository(), [], INPUT_FILES)
    assert result['status'] == 'passed'
    assert result['observations']['l4_pair']['coverage']['pair_valid'] == 0


def test_summary_decomposition_does_not_invent_agreement_for_zero_valid_pairs():
    from analysis.compliance_competence import decompose_report, to_latex
    row = decompose_report({'summary': {'Counterfactual_Alignment_Score_mean': 0,
                                       'Score_interv_valid_only_mean': 0,
                                       'compliance': {'counterfactuals': {'valid': 0, 'gate_pass_rate': 0}}}})
    assert row['counterfactuals']['competence'] is None
    assert row['counterfactuals']['compliance_deficit'] is None
    assert '--' in to_latex([{'model': 'test', **row}], axis='counterfactuals')


@pytest.mark.parametrize("fault", ["legacy", "unmarked", "wrong_target", "wrong_policy"])
def test_inventory_rejects_incompatible_l4_references(tmp_path, fault):
    from analysis.evaluation_campaign import dataset_inventory
    base = make_protocol_dataset(tmp_path, n=1)
    path = base / "scenario_1" / "ground_truth.json"
    gt = json.loads(path.read_text())
    if fault == "legacy":
        gt.pop("l4_queries", None)
    elif fault == "unmarked":
        gt["protocol"].pop("l4_pair_protocol", None)
    else:
        # This fixture must exercise actual Option-B references, never the
        # legacy counterfactuals compatibility path.
        assert gt.get("l4_queries")
        control = gt["l4_queries"][0]["control_chain"]
        if fault == "wrong_target":
            control["intervention"]["target_actor_id"] = "ego"
        else:
            control["semantics"]["ego_policy"] = "do_ego_emergency_brake"
    path.write_text(json.dumps(gt))
    with pytest.raises(ValueError, match="L4|pair"):
        dataset_inventory(base, 1, "nuscenes")


def test_missing_first_label_is_undefined_without_inventing_a_wrong_label():
    from collections import defaultdict
    from analysis.evaluation_report import node_label_diagnostics
    iv = {'intervention_type': 'npc_sudden_brake'}
    no_action = {'intervention_type': 'do_ego_no_action'}
    node = {'event_type': 'maintain_course'}
    reference = {'branches': [{'intervention': no_action, 'chain': [node]}],
                 'l4_queries': [{'counterfactual_chain': {'intervention': iv, 'chain': [node]}}]}
    prediction = {'branches': [{'intervention': no_action, 'chain': [None, node]}],
                  'predicted_counterfactuals': [{'intervention': iv, 'treated_chain': [None, node]}]}
    gt = {'scenarios': [{'scenario_id': 's', 'samples': [{'ground_truth': reference}]}]}
    pred = {'scenarios': [{'scenario_id': 's', 'samples': [{'prediction': prediction}]}]}
    metrics = defaultdict(dict)
    result = node_label_diagnostics(gt, pred, {'s': 's'}, metrics)
    assert metrics['L3_no_action_label_match']['s'] is None
    assert metrics['L4_first_node_label_match']['s'] is None
    assert result == {'l4_first_node_labels_used_instead': {}, 'l3_no_action_labels_used_instead': {}}
