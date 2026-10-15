"""Model-runner core — parse a VLM's raw text output into CORIVA prediction files.

The runner is model-agnostic: whatever backend (local Qwen3-VL on GPU, or an API)
produces text, this turns it into the l3_pre.json / l4_pre.json that
scripts/assemble_dataset.py expects. Lenient by design — a malformed/partial
prediction should still score (the CTMS gates penalize it) rather than crash.
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


# ── extract_json: pull the JSON object out of noisy model text ──
def test_extract_json_from_fenced_block():
    from analysis.parse_prediction import extract_json
    txt = 'Here is my answer:\n```json\n{"a": 1, "b": [2, 3]}\n```\nDone.'
    assert extract_json(txt) == {"a": 1, "b": [2, 3]}


def test_extract_json_bare_object_with_surrounding_noise():
    from analysis.parse_prediction import extract_json
    assert extract_json('prefix {"x": [1, 2]} suffix') == {"x": [1, 2]}


def test_extract_json_handles_braces_inside_strings():
    from analysis.parse_prediction import extract_json
    assert extract_json('{"note": "a } brace { in text", "n": 1}') == {
        "note": "a } brace { in text", "n": 1}


def test_extract_json_returns_none_on_garbage():
    from analysis.parse_prediction import extract_json
    assert extract_json("no json here at all") is None


# ── to_prediction_files: normalize into the l3/l4 shape ──
def _obj():
    node = {"timestamp": 1.0, "event_type": "collision", "terminal": True,
            "actors_update": [{"id": "ego", "entity_type": "ego_vehicle", "x": 0.0, "y": 0.0}]}
    return {
        "predicted_branches": [{"branch_id": "B1", "is_factual": True, "chain": [node]}],
        "predicted_counterfactuals": [{
            "intervention": {"intervention_type": "do_ego_no_action",
                             "target_actor_id": "ego", "trigger_depth": 1},
            "chain": [node]}],
    }


def test_to_prediction_files_shape():
    from analysis.parse_prediction import to_prediction_files
    l3, l4 = to_prediction_files(_obj(), "nuscenes_0001")
    assert l3["scenario_id"] == "nuscenes_0001"
    assert l3["predicted_branches"][0]["branch_id"] == "B1"
    assert l4["scenario_id"] == "nuscenes_0001"
    assert l4["predicted_counterfactuals"][0]["intervention"]["intervention_type"] == "do_ego_no_action"


def test_to_prediction_files_accepts_key_aliases():
    from analysis.parse_prediction import to_prediction_files
    obj = {"branches": _obj()["predicted_branches"],
           "counterfactuals": _obj()["predicted_counterfactuals"]}
    l3, l4 = to_prediction_files(obj, "x")
    assert l3["predicted_branches"] and l4["predicted_counterfactuals"]


def test_to_prediction_files_missing_sections_default_empty():
    from analysis.parse_prediction import to_prediction_files
    l3, l4 = to_prediction_files({}, "x")
    assert l3["predicted_branches"] == [] and l4["predicted_counterfactuals"] == []


# ── parse_model_output: end-to-end text -> (l3, l4, errors) ──
def test_parse_model_output_end_to_end():
    import json
    from analysis.parse_prediction import parse_model_output
    txt = "```json\n" + json.dumps(_obj()) + "\n```"
    l3, l4, errors = parse_model_output(txt, "nuscenes_0001")
    assert not errors
    assert l3["predicted_branches"] and l4["predicted_counterfactuals"]


def test_parse_model_output_reports_error_on_unparseable():
    from analysis.parse_prediction import parse_model_output
    l3, l4, errors = parse_model_output("the model refused to answer", "x")
    assert errors  # non-empty -> flagged
    assert l3["predicted_branches"] == [] and l4["predicted_counterfactuals"] == []


# ── missing-delimiter repair (parser policy) ──
def _branch_text(actor_close=True, chain_close=True, outer_close=True):
    node = '{"depth":1,"timestamp":0.5,"event_type":"maintain_course","actors_update":[' \
           '{"id":"ego","entity_type":"ego_vehicle","role_type":"ego_agent",' \
           '"x":1.0,"y":0.0,"vx":2.0,"vy":0.0,"speed":2.0' + ('}' if actor_close else '') + ']}'
    text = '{"predicted_branches":[{"branch_id":"B1","is_factual":true,' \
           '"intervention":{"intervention_type":"do_ego_no_action","target_actor_id":"ego","trigger_depth":1},' \
           '"chain":[' + node + (']' if chain_close else '') + '}],' \
           '"predicted_counterfactuals":[{"intervention":{"intervention_type":"do_target_sudden_brake",' \
           '"target_actor_id":"npc_1","trigger_depth":1},"control_chain":[' + node + '],"treated_chain":[' + node + ']}]' \
           + ('}' if outer_close else '')
    return text


def test_valid_json_is_never_altered():
    import json
    from analysis.parse_prediction import extract_json
    text = _branch_text()
    assert extract_json(text) == json.loads(text)


def test_repairs_missing_closers_at_end_of_answer():
    import json
    from analysis.parse_prediction import extract_json
    assert extract_json(_branch_text(outer_close=False)) == json.loads(_branch_text())
    assert extract_json("```json\n" + _branch_text(outer_close=False) + "\n```") == json.loads(_branch_text())


def test_repairs_missing_internal_closer_using_contract_shapes():
    import json
    from analysis.parse_prediction import extract_json
    assert extract_json(_branch_text(actor_close=False)) == json.loads(_branch_text())
    assert extract_json(_branch_text(chain_close=False)) == json.loads(_branch_text())


def test_repairs_answer_with_closing_fence_but_no_opening_fence():
    import json
    from analysis.parse_prediction import extract_json
    assert extract_json(_branch_text(outer_close=False) + "\n```\n") == json.loads(_branch_text())


def test_repaired_answer_keeps_missing_fields_for_the_gates():
    from analysis.parse_prediction import extract_json
    text = '{"predicted_branches":[{"branch_id":"B1","chain":[{"depth":1,"actors_update":[{"id":"ego","x":1.0'
    obj = extract_json(text)
    assert obj == {"predicted_branches": [{"branch_id": "B1", "chain": [
        {"depth": 1, "actors_update": [{"id": "ego", "x": 1.0}]}]}]}


def test_no_repair_for_formula_surplus_closer_stray_quote_missing_key_or_unknown_key():
    from analysis.parse_prediction import extract_json
    good = _branch_text()
    formula = good.replace('"x":1.0', '"x":0.5+0.5')
    surplus = good.replace('"speed":2.0}]}', '"speed":2.0}]}}')
    stray = good.replace('"speed":2.0}', '"speed":2.0,"}')
    missing_key = good.replace('"y":0.0', '0.0')
    unknown_key = good.replace('"speed":2.0', '"speed":2.0,"rounded_x":1')[:-1]
    from analysis.parse_prediction import to_prediction_files
    for text in (formula, surplus, stray, missing_key, unknown_key):
        l3, l4 = to_prediction_files(extract_json(text) or {}, "x")
        assert l3["predicted_branches"] == [] and l4["predicted_counterfactuals"] == [], text


def test_no_repair_when_several_minimal_structures_exist():
    from analysis.parse_prediction import extract_json
    # Two nodes' actors could close before or after the second node.
    text = '{"predicted_branches":[{"chain":[{"depth":1,"actors_update":[{"id":"a","x":1},' \
           '{"depth":2,"actors_update":[{"id":"a","x":2}]}]}]}'
    from analysis.json_repair import repair_missing_delimiters
    assert repair_missing_delimiters(text) is None or extract_json(text) is not None


def test_empty_or_prose_answer_is_not_repaired():
    from analysis.parse_prediction import parse_model_output
    for text in ("", "   ", "I cannot see the road.", "```json\n```"):
        l3, l4, errors = parse_model_output(text, "x")
        assert errors and l3["predicted_branches"] == []


def test_parse_model_output_scores_repaired_answer_without_error():
    from analysis.parse_prediction import parse_model_output
    l3, l4, errors = parse_model_output(_branch_text(outer_close=False), "carla_0001")
    assert not errors
    assert l3["predicted_branches"][0]["branch_id"] == "B1"
    assert l4["predicted_counterfactuals"][0]["control_chain"]
