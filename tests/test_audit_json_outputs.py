"""The audit must distinguish syntax, extraction and missing evidence."""
import json

from scripts.audit_json_outputs import (inspect_text, audit_run, build_report,
                                        delimiter_repair_probe, schema_repair_probe)


def repaired_objects(text, probe):
    objects = []
    for candidate in probe['candidates']:
        repaired = text
        for edit in reversed(candidate['insertions']):
            repaired = repaired[:edit['offset']] + edit['text'] + repaired[edit['offset']:]
        assert all(set(e['text']) <= set('{}[]') for e in candidate['insertions'])
        objects.append(json.loads(repaired))
    return objects


def test_schema_uses_field_ownership_to_keep_coordinates_inside_actor():
    text = '{"predicted_branches":[{"chain":[{"actors_update":[{"id":"ego","x":1,"y":2]}]}]}'
    probe = schema_repair_probe(text)
    assert probe['status'] == 'candidate_found'
    assert probe['search_complete']
    assert repaired_objects(text, probe)[0]['predicted_branches'][0]['chain'][0]['actors_update'] == [
        {'id': 'ego', 'x': 1, 'y': 2}]


def test_schema_closes_after_complete_scalar_at_eof_without_changing_digits():
    text = '{"predicted_branches":[{"chain":[{"actors_update":[{"x":1.25e2'
    probe = schema_repair_probe(text)
    assert probe['status'] == 'candidate_found'
    assert probe['eof_scalar_preserved']
    assert probe['candidates'][0]['eof_insertions'] == 7
    assert repaired_objects(text, probe)[0]['predicted_branches'][0]['chain'][0]['actors_update'][0]['x'] == 125


def test_schema_recovers_missing_openers_and_empty_array_without_values():
    for text in ['"predicted_branches":]}', '{"predicted_branches":{"chain":[]}]}']:
        probe = schema_repair_probe(text)
        assert probe['status'] == 'candidate_found'
        assert repaired_objects(text, probe)


def test_schema_handles_many_missing_braces_without_six_edit_ceiling():
    original = json.dumps({'predicted_branches': [{'chain': [{'actors_update': [
        {'id': f'actor_{i}', 'x': i, 'y': -i} for i in range(12)]}]}]})
    broken = original.replace('}', '')
    probe = schema_repair_probe(broken)
    assert probe['status'] == 'candidate_found'
    assert probe['minimal_insertions'] > 6
    assert repaired_objects(broken, probe) == [json.loads(original)]


def test_schema_preserves_semantic_errors_and_reports_missing_fields():
    text = '{"predicted_branches":[{"chain":[{"event_type":"wrong_label","actors_update":[{"id":"ego","x":-999,"speed":-2}]}]}]'
    probe = schema_repair_probe(text)
    assert probe['status'] == 'candidate_found'
    obj = repaired_objects(text, probe)[0]
    node = obj['predicted_branches'][0]['chain'][0]
    assert node['event_type'] == 'wrong_label'
    assert node['actors_update'][0]['speed'] == -2
    assert 'y' not in node['actors_update'][0]
    assert probe['candidates'][0]['missing_contract_fields']


def test_schema_keeps_real_structural_ambiguity_visible():
    # x may belong to ego or npc; both interpretations require two insertions.
    # Required-field completeness cannot be used to manufacture a preference.
    text = '{"predicted_branches":[{"chain":[{"actors_update":[{"id":"ego","x":1,"id":"npc","y":2}]}]}]}'
    probe = schema_repair_probe(text)
    assert probe['status'] == 'ambiguous'
    assert probe['candidate_count'] == 2
    assert all(len(o['predicted_branches'][0]['chain'][0]['actors_update']) >= 2
               for o in repaired_objects(text, probe))


def test_schema_never_supplies_missing_values_quotes_commas_or_arithmetic():
    for tail in ['"x":}', '"x":1+', '"x":1e', '"x":"unfinished',
                 '"x":1 "y":2}', '"x":1+2}']:
        probe = schema_repair_probe('{"predicted_branches":[{"chain":[{"actors_update":[{' + tail)
        assert probe['candidate_count'] == 0


def test_schema_limits_are_reported_and_reproducible():
    text = '{"predicted_branches":[{"chain":[]}]'
    probe = schema_repair_probe(text, max_states=1)
    assert probe['status'] == 'search_limit'
    assert not probe['search_complete']
    assert probe == schema_repair_probe(text, max_states=1)


def test_schema_does_not_overwrite_duplicate_keys():
    probe = schema_repair_probe('{"predicted_branches":[],"predicted_branches":[]}')
    assert probe['candidate_count'] == 0


def test_schema_probe_requires_recorded_contract_and_does_not_change_files(tmp_path):
    from scripts.audit_json_outputs import CONTRACT_MARKERS
    (tmp_path / 'run_manifest.json').write_text('{}')
    (tmp_path / 'final_response.txt').write_text('{"predicted_branches":[{"chain":[]}]')
    unverified = audit_run(tmp_path, 'model-a', 'carla', 'video', {}, schema_only=True)
    assert unverified['schema_repair_probe']['status'] == 'contract_unverified'
    (tmp_path / 'system_prompt.txt').write_text('\n'.join(CONTRACT_MARKERS))
    before = {p.name: p.read_bytes() for p in tmp_path.iterdir()}
    first = audit_run(tmp_path, 'model-a', 'carla', 'video', {}, schema_only=True)
    second = audit_run(tmp_path, 'model-b', 'carla', 'video', {}, schema_only=True)
    assert first['schema_repair_probe']['status'] == 'candidate_found'
    assert first['schema_repair_probe'] == second['schema_repair_probe']
    assert not first['parser_errors']  # the parser accepts this unique contract repair
    assert first['delimiter_repair_probe']['status'] == 'not_run'
    assert {p.name: p.read_bytes() for p in tmp_path.iterdir()} == before


def test_schema_limited_search_never_claims_uniqueness():
    text = '{"predicted_branches":[{"chain":[{"actors_update":[{"id":"ego","x":1,"id":"npc","y":2}]}]}]}'
    probe = schema_repair_probe(text, max_paths=1)
    assert not probe['search_complete']
    assert probe['status'] == 'search_limit'


def test_schema_handles_escaped_unicode_keys_and_bbox_shape():
    text = '{"predicted_branches":[{"chain":[{"actors_update":[{"\\u0069d":"é","x":1,"bbox":[1,2,3,4]}]}]}]'
    probe = schema_repair_probe(text)
    assert probe['status'] == 'candidate_found'
    actor = repaired_objects(text, probe)[0]['predicted_branches'][0]['chain'][0]['actors_update'][0]
    assert actor['id'] == 'é' and actor['bbox'] == [1, 2, 3, 4]


def test_probe_inserts_internal_closing_brace_and_keeps_values():
    text = '{"predicted_branches":[{"x":1],"note":"} ]"}'
    probe = delimiter_repair_probe(text)
    assert probe['status'] == 'candidate_found'
    assert probe['candidates'][0]['insertions'] == [{'offset': text.index('],'), 'text': '}'}]
    assert probe['candidates'][0]['internal_insertions'] == 1
    assert probe['candidates'][0]['eof_insertions'] == 0


def test_probe_inserts_missing_opening_object_and_array():
    for text, char in [('{"items":["x":1}]}', '{'), ('{"items":1,2]}', '['),
                       ('"items":[{"x":1}]}', '{')]:
        probe = delimiter_repair_probe(text)
        assert probe['status'] == 'candidate_found'
        assert any(e['text'] == char for e in probe['candidates'][0]['insertions'])


def test_probe_recovers_multiple_internal_and_eof_brackets():
    probe = delimiter_repair_probe('{"items":[{"x":1],"other":[{"y":2}]')
    assert probe['status'] == 'candidate_found'
    assert probe['candidates'][0]['internal_insertions'] == 1
    assert probe['candidates'][0]['eof_insertions'] == 1


def test_equivalent_closer_positions_prefer_eof_without_false_ambiguity():
    probe = delimiter_repair_probe('{"a":{"x":1}')
    assert probe['status'] == 'candidate_found'
    assert probe['candidate_count'] == 1
    assert probe['candidates'][0]['internal_insertions'] == 0
    assert probe['candidates'][0]['eof_insertions'] == 1


def test_missing_array_opener_before_a_long_container():
    text = '{"a": {"x": [1,2,3,4,5,6,7,8,9,10]}, {"x":2}]}'
    probe = delimiter_repair_probe(text)
    assert probe['status'] == 'candidate_found'
    assert probe['candidates'][0]['insertions'] == [{'offset': 6, 'text': '['}]


def test_reported_offsets_reconstruct_exact_values_with_unicode_and_escapes():
    text = '{"note":"é \\\" }", "items":[{"x":1.00e2],"b":[{"y":-2.50}]  '
    probe = delimiter_repair_probe(text)
    assert probe['candidate_count'] >= 1
    for candidate in probe['candidates']:
        repaired = text
        for edit in reversed(candidate['insertions']):
            repaired = repaired[:edit['offset']] + edit['text'] + repaired[edit['offset']:]
        assert json.loads(repaired)['items'] == [{'x': 100}]
        assert '1.00e2' in repaired and '-2.50' in repaired
        assert all(set(edit['text']) <= set('{}[]') for edit in candidate['insertions'])


def test_probe_reports_ambiguous_structure_instead_of_picking_one():
    # Either {"a": {"x": 1, "y": 2}} or {"a": {"x": 1}, "y": 2}.
    probe = delimiter_repair_probe('{"a":{"x":1,"y":2}')
    assert probe['status'] == 'ambiguous'
    assert probe['candidate_count'] >= 2


def test_probe_does_not_invent_values_or_evaluate_expressions():
    for text in ['{"x":}', '{"x":"unfinished', '{"x":12e',
                 '{"x":1+2}', '{"x":1 "y":2}', '{"x":1,}']:
        probe = delimiter_repair_probe(text)
        assert probe['candidate_count'] == 0, (text, probe)


def test_syntax_probe_preserves_complete_scalar_at_eof_too():
    text = '{"x":12'
    probe = delimiter_repair_probe(text)
    assert probe['status'] == 'candidate_found'
    assert repaired_objects(text, probe) == [{'x': 12}]


def test_probe_never_silently_overwrites_duplicate_keys():
    probe = delimiter_repair_probe('{"x":1,"x":2')
    assert probe['candidate_count'] == 0


def test_probe_reports_search_limit_and_is_deterministic():
    text = '{"a":[{"x":1,2]}'
    first = delimiter_repair_probe(text, max_states=1)
    assert first['status'] == 'search_limit'
    assert not first['search_complete']
    assert first == delimiter_repair_probe(text, max_states=1)


def test_inspection_exposes_new_probe_without_replacing_parser_errors():
    row = inspect_text('{"predicted_branches":[{"x":1]}')
    assert row['delimiter_repair_probe']['candidate_count'] >= 1
    assert row['syntax_status'] == 'invalid_json'
    assert row['parser_errors']


def test_wrapped_valid_json_ignores_delimiters_inside_strings():
    text = 'Answer:\n```json\n{"predicted_branches":[{"note":"} ] \\\""}]}\n```'
    row = inspect_text(text)
    assert row['syntax_status'] == 'valid_object'
    assert row['delimiter_scan']['unclosed_count'] == 0
    assert row['delimiter_scan']['mismatched_closer_count'] == 0
    assert not row['parser_errors']


def test_missing_outer_closer_is_only_a_syntax_probe():
    row = inspect_text('{"predicted_branches":[{"x":1}]')
    assert row['syntax_status'] == 'invalid_json'
    assert row['delimiter_scan']['unclosed_count'] == 1
    assert row['append_closers_probe_parses'] is True
    assert row['parser_errors']  # audit does not feed the probe to the parser


def test_unfinished_value_or_string_is_not_recovered():
    for text in ['{"predicted_branches":[{"x":', '{"x":"unfinished', '{"x":12']:
        assert inspect_text(text)['append_closers_probe_parses'] is False


def test_mismatched_bracket_is_separate_from_missing_eof_bracket():
    row = inspect_text('{"predicted_branches":[{"x":1}}')
    assert row['delimiter_scan']['mismatched_closer_count'] == 1
    assert row['append_closers_probe_parses'] is False


def test_arithmetic_is_detected_but_never_evaluated():
    row = inspect_text('{"predicted_branches":[{"x":54.207+24.0*0.5}]}')
    assert row['arithmetic_expression_outside_strings'] is True
    assert row['syntax_status'] == 'invalid_json'
    assert inspect_text('{"note":"54.207+24.0*0.5"}')['arithmetic_expression_outside_strings'] is False


def test_missing_prediction_key_is_not_a_json_syntax_error():
    row = inspect_text('{"intervention":{"target_actor_id":"ego"}}')
    assert row['syntax_status'] == 'valid_object'
    assert row['parser_errors'] == ['no predicted_branches found']


def test_valid_array_is_not_reported_as_valid_prediction_object():
    row = inspect_text('[{"predicted_branches":[{}]}]')
    assert row['syntax_status'] == 'valid_non_object'


def test_first_json_error_has_location_and_limited_excerpt():
    row = inspect_text('{"predicted_branches":[{"x":1 "y":2}]}')
    assert row['syntax_status'] == 'invalid_json'
    assert row['json_error']['line'] == 1
    assert row['json_error']['column'] > 1
    assert len(row['json_error']['excerpt']) <= 160


def test_length_finish_and_reasoning_alias_are_counted(tmp_path):
    scenario = tmp_path / 'qwen-test/cells/carla/qwen-test_video/scenario_1'
    scenario.mkdir(parents=True)
    manifest = {'max_output_tokens': 65536, 'temperature': 1.0,
                'generation_metadata': {'finish_reason': 'length', 'output_tokens': 65536,
                                        'reasoning_tokens': 65536}}
    (scenario / 'run_manifest.json').write_text(json.dumps(manifest))
    (scenario / 'final_response.txt').write_text('')
    row = audit_run(scenario, 'qwen-test', 'carla', 'video', {})
    assert row['output_cap_hit'] is True
    assert row['reasoning_tokens'] == 65536
    assert row['presence_penalty'] is None  # never invent an unrecorded default
    assert row['syntax_status'] == 'empty_response'


def test_missing_output_stays_separate_and_inputs_are_unchanged(tmp_path):
    root = tmp_path / 'panel'
    model = root / 'qwen-test'
    scenario = model / 'cells/carla/qwen-test_video/scenario_1'
    scenario.mkdir(parents=True)
    (model / 'frozen.json').write_text(json.dumps({'datasets': {'carla': [{'name': 'scenario_1'}, {'name': 'scenario_2'}]}}))
    (scenario / 'run_manifest.json').write_text('{}')
    before = {p: p.read_bytes() for p in root.rglob('*') if p.is_file()}
    report = build_report(root, ['qwen-test'])
    assert report['cells'][0]['expected_requests'] == 2
    assert report['cells'][0]['missing_scenarios'] == ['scenario_2']
    assert report['runs'][0]['syntax_status'] == 'missing_final_response'
    assert report['models']['qwen-test']['syntax_invalid'] == 0
    assert {p: p.read_bytes() for p in root.rglob('*') if p.is_file()} == before
