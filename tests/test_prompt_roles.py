"""Prompt roles and layout: fixed instructions as a system message, the scene
as the user message, no run metadata in either, one actor per JSON line."""
import hashlib
import json
import os
import re
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


def _scene(offset=0.0, target="npc_1"):
    root = {"actors": [
        {"id": "ego", "entity_type": "ego_vehicle", "role_type": "ego_agent",
         "x": 0, "y": 0, "vx": 10 + offset, "vy": 0, "speed": 10 + offset},
        {"id": target, "entity_type": "vehicle", "role_type": "lead_vehicle",
         "x": 20 + offset, "y": 0, "vx": 8, "vy": 0, "speed": 8},
        {"id": "park_1", "entity_type": "parked_vehicle", "role_type": "adjacent_vehicle",
         "x": 15, "y": 4, "vx": 0, "vy": 0, "speed": 0}]}
    return dict(root=root, interventions=["npc_sudden_brake"], critical_actor_id=target,
                l3_actions=["do_ego_no_action", "do_ego_emergency_brake"],
                factual_action="do_ego_no_action",
                observed_history={"records": [{"timestamp_sec": -0.5, "actors": root["actors"]}]})


def _parts(scenario_id, split="carla", condition="video_history", **scene):
    from simulation.build_pilot import build_prompt_parts
    kw = _scene(**scene)
    return build_prompt_parts(kw.pop("root"), scenario_id, split=split, condition=condition, **kw)


@pytest.mark.parametrize("split", ["carla", "nuscenes"])
@pytest.mark.parametrize("condition", ["video", "history_json", "video_history"])
def test_system_part_is_the_same_for_every_scene_of_a_split_and_condition(split, condition):
    s1, u1 = _parts("lead_car_brake_0000", split, condition)
    s2, u2 = _parts("cut_in_left_0007", split, condition, offset=3.0, target="npc_4")
    assert s1 == s2
    assert u1 != u2
    assert "npc_4" in u2 and "npc_4" not in s2


def test_combined_prompt_is_system_then_user():
    from simulation.build_pilot import build_prompt, join_prompt_parts
    kw = _scene()
    root = kw.pop("root")
    system, user = _parts("carla_0001")
    assert build_prompt(root, "carla_0001", split="carla", condition="video_history", **kw) \
        == join_prompt_parts(system, user) == f"{system}\n\n{user}"


def test_prompt_carries_no_scenario_id_or_prompt_version():
    from simulation.build_pilot import PROMPT_VERSION
    for part in _parts("lead_car_brake_0000"):
        assert "lead_car_brake" not in part
        assert PROMPT_VERSION not in part and "Prompt version" not in part


def test_every_prompt_json_parses_and_keeps_one_actor_per_line():
    for part in _parts("carla_0001"):
        blocks = re.findall(r"```json\n(.*?)\n```", part, re.S)
        assert blocks
        for block in blocks:
            json.loads(block)
            assert "},{" not in block and "}}" not in block
    system, user = _parts("carla_0001")
    state = re.search(r"## observed_state \(T=0\)\n```json\n(.*?)\n```", user, re.S).group(1)
    lines = state.splitlines()
    for actor in _scene()["root"]["actors"]:
        line = json.dumps(actor, separators=(",", ":"))
        assert sum(line in row for row in lines) == 1
    assert "one line" in system and "Never spread an actor's fields" in system


def test_run_model_sends_system_prompt_to_backends_that_take_one(tmp_path):
    from analysis.run_model import BACKENDS, SYSTEM_PROMPT_BACKENDS, run_model
    from analysis.evaluation_campaign import check_cell, dataset_inventory
    from simulation.build_pilot import join_prompt_parts
    from tests.test_evaluation_campaign import make_protocol_dataset
    seen = {}

    def spy(prompt, frame_paths, model_id, opts):
        seen[opts.get("name")] = (prompt, opts.get("system_prompt"))
        return json.dumps({"predicted_branches": [], "predicted_counterfactuals": []})

    config = dict(model_id="m", adapter="x", revision=None, inference_mode="direct",
                  runtime_ready=False, adapter_implemented=True, max_output_tokens=65536,
                  thinking_trace_storage="separate_artifact", scorer_input="final_json_only")
    base = make_protocol_dataset(tmp_path, n=1)
    rows = dataset_inventory(base, 1, "nuscenes")
    BACKENDS["spy_sys"] = BACKENDS["spy_one"] = spy
    SYSTEM_PROMPT_BACKENDS.add("spy_sys")
    try:
        for name in ("spy_sys", "spy_one"):
            run_model(str(base), str(tmp_path / name), backend=name, model_id="m",
                      model_config=dict(config, name=name, backend=name), provisional_smoke=True)
    finally:
        del BACKENDS["spy_sys"], BACKENDS["spy_one"]
        SYSTEM_PROMPT_BACKENDS.discard("spy_sys")

    user, system = seen["spy_sys"]
    assert system and user and "## Rules" in system and "## observed_state" in user
    assert seen["spy_one"] == (join_prompt_parts(system, user), None)

    out = tmp_path / "spy_sys" / "scenario_1"
    manifest = json.loads((out / "run_manifest.json").read_text())
    assert (out / "prompt.txt").read_text() == user
    assert (out / "system_prompt.txt").read_text() == system
    assert manifest["system_prompt_sha256"] == hashlib.sha256(system.encode()).hexdigest()
    assert manifest["prompt_sha256"] == hashlib.sha256(user.encode()).hexdigest()
    single = tmp_path / "spy_one" / "scenario_1"
    assert not (single / "system_prompt.txt").exists()
    assert json.loads((single / "run_manifest.json").read_text())["system_prompt_sha256"] is None

    kw = dict(condition="video", configuration="spy_sys", backend="spy_sys")
    assert check_cell(tmp_path / "spy_sys", rows, **kw)["complete"]
    (out / "system_prompt.txt").write_text(system + "tampered")
    assert any("system_prompt.txt" in e for e in check_cell(tmp_path / "spy_sys", rows, **kw)["errors"])
    (out / "system_prompt.txt").unlink()
    assert any("system_prompt.txt" in e for e in check_cell(tmp_path / "spy_sys", rows, **kw)["errors"])


def test_gemini_request_carries_the_system_instruction(monkeypatch):
    from analysis.backends import api
    captured = {}

    def fake_post(url, payload, headers, timeout):
        captured["payload"] = payload
        return {"candidates": [{"content": {"parts": [{"text": "{}"}]}}],
                "usageMetadata": {"promptTokenCount": 1, "candidatesTokenCount": 1}}

    monkeypatch.setattr(api, "_post_json", fake_post)
    api.gemini_generate_result("the scene", [], "gemini-2.5-pro", api_key="k",
                               system_prompt="the rules")
    assert captured["payload"]["systemInstruction"] == {"parts": [{"text": "the rules"}]}
    assert captured["payload"]["contents"][0]["parts"][0]["text"] == "the scene"
    api.gemini_generate_result("the scene", [], "gemini-2.5-pro", api_key="k")
    assert "systemInstruction" not in captured["payload"]


def test_openai_compatible_request_puts_the_rules_in_a_system_message(monkeypatch):
    from analysis.run_model import SYSTEM_PROMPT_BACKENDS
    from scripts import run_openrouter_cell as cell
    captured = {}

    def fake_stream(url, payload, headers, timeout, **limits):
        captured["payload"] = payload
        return {"choices": [{"message": {"content": "{}"}, "finish_reason": "stop"}]}

    monkeypatch.setattr(cell, "_stream_chat", fake_stream)
    monkeypatch.setattr(cell.api_backend, "get_api_key", lambda *a, **k: "k")
    assert "openai_compat" in SYSTEM_PROMPT_BACKENDS
    cell._openai_compat_backend("the scene", [], "m", {"system_prompt": "the rules"})
    messages = captured["payload"]["messages"]
    assert messages[0] == {"role": "system", "content": "the rules"}
    assert messages[1]["role"] == "user" and messages[1]["content"][0]["text"] == "the scene"
    cell._openai_compat_backend("the scene", [], "m", {})
    assert [m["role"] for m in captured["payload"]["messages"]] == ["user"]


# coriva-prompt-2026-09-23d was locked on 2026-09-23, after the pilot on
# Qwen3.5-9B, Mistral Small 3.2 and Gemini 2.5 Pro. These are the texts of the
# fixture scene; ontology, profiles and schema enums all feed into them.
LOCKED_PROMPT_VERSION = "coriva-prompt-2026-09-23d"
LOCKED_PROMPT_SHA256 = {
    ("carla", "video"): "b83c91a55207c505478c29f536aa96c9291bbf3a873570466aff5092fdd04939",
    ("carla", "history_json"): "8cbb22f6c834aea3f7ddd9722176065d3690b030dd104f772000a2d4522206d4",
    ("carla", "video_history"): "233dbcae048260f8da0ab42bcdec234311d31dbb227ca4580804cf59eafa9767",
    ("nuscenes", "video"): "7077243ddb519b9df83a74266c89ce8dcc2b6ee6d624745bcd37f7b5cc3b2bec",
    ("nuscenes", "history_json"): "d2c5df854e6894490c61aec47be4783ca279a8df37e39327baef7e087aabb537",
    ("nuscenes", "video_history"): "43c54317b3b205405f0b98e470d8e276809e3ad9b4235c4b89f6e88d6bc8fbe6",
}


@pytest.mark.parametrize("split,condition", sorted(LOCKED_PROMPT_SHA256))
def test_the_locked_prompt_text_does_not_change(split, condition):
    """A changed prompt is a new version and a new campaign root. Changing the
    text means bumping PROMPT_VERSION and re-pinning these hashes on purpose."""
    from simulation.build_pilot import PROMPT_VERSION
    system, user = _parts("carla_0001", split, condition)
    digest = hashlib.sha256(f"{system}\0{user}".encode()).hexdigest()
    assert PROMPT_VERSION == LOCKED_PROMPT_VERSION
    assert digest == LOCKED_PROMPT_SHA256[(split, condition)], (
        f"the {split}/{condition} prompt text changed under the locked version "
        f"{LOCKED_PROMPT_VERSION}: bump PROMPT_VERSION and start a new campaign root")
