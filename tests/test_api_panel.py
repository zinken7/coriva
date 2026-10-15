"""The hosted/self-hosted panel runner: configurations and resume guards."""
import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


def test_qwen_runs_on_our_vllm_server_with_the_pinned_revision():
    from analysis.model_panel import MAX_OUTPUT_TOKENS, load_model_panel
    from scripts import run_api_panel as P
    panel = load_model_panel(os.path.join(os.path.dirname(__file__), "..", "analysis", "model_panel.yaml"))
    pinned = next(c["revision"] for c in panel["checkpoints"] if c["key"] == "qwen3.5-9b")
    for name, thinking in (("qwen3.5-9b-direct", False), ("qwen3.5-9b-thinking", True),
                           ("qwen3.5-9b-direct-sampled", False)):
        cfg = P.model_config(name, P.PANEL[name])
        assert cfg["model_id"] == "Qwen/Qwen3.5-9B" and cfg["revision"] == pinned
        assert cfg["serving"] == "vllm" and cfg["api_base_url_env"] == "CORIVA_VLLM_BASE_URL"
        assert cfg["api_key_env"] is None and "provider" not in cfg and "reasoning" not in cfg
        assert cfg["chat_template_kwargs"] == {"enable_thinking": thinking}
        assert cfg["enable_thinking"] is thinking
        assert cfg["inference_mode"] == ("thinking" if thinking else "direct")
        assert cfg["max_output_tokens"] == MAX_OUTPUT_TOKENS == 65536


def test_mistral_stays_on_openrouter_parasail():
    from scripts import run_api_panel as P
    cfg = P.model_config("mistral-small-3.2", P.PANEL["mistral-small-3.2"])
    assert cfg["serving"] == "openrouter" and cfg["provider"] == "Parasail"
    assert cfg["quantizations"] == ["bf16"] and cfg["api_key_env"] == "OPENROUTER_API_KEY"
    assert cfg["max_output_tokens"] == 65536 and cfg["revision"] is None


def test_a_root_refuses_to_resume_on_another_vllm_server(tmp_path, monkeypatch):
    from scripts import run_api_panel as P
    cfg = P.model_config("qwen3.5-9b-direct", P.PANEL["qwen3.5-9b-direct"])
    monkeypatch.setattr(P, "choose", lambda split, n: [])
    monkeypatch.setattr(P, "dataset_inventory", lambda subset, n, split: [])
    monkeypatch.setenv("CORIVA_VLLM_BASE_URL", "http://localhost:8000/v1")
    server = {"vllm_version": "0.11.0", "served_models": ["Qwen/Qwen3.5-9B"], "max_model_len": 131072}
    monkeypatch.setattr(P, "vllm_server_info", lambda url: dict(server))
    ctx = P.prepare(tmp_path, "qwen3.5-9b-direct", cfg, 4)
    assert ctx["frozen"]["serving"] == server
    P.prepare(tmp_path, "qwen3.5-9b-direct", cfg, 4)          # same server: resumes
    server["vllm_version"] = "0.12.0"
    with pytest.raises(RuntimeError, match="vLLM"):
        P.prepare(tmp_path, "qwen3.5-9b-direct", cfg, 4)


def test_a_server_that_cannot_hold_the_cap_is_refused(tmp_path, monkeypatch):
    from scripts import run_api_panel as P
    cfg = P.model_config("qwen3.5-9b-thinking", P.PANEL["qwen3.5-9b-thinking"])
    monkeypatch.setattr(P, "choose", lambda split, n: [])
    monkeypatch.setattr(P, "dataset_inventory", lambda subset, n, split: [])
    monkeypatch.setenv("CORIVA_VLLM_BASE_URL", "http://localhost:8000/v1")
    monkeypatch.setattr(P, "vllm_server_info", lambda url: {
        "vllm_version": "0.11.0", "served_models": ["Qwen/Qwen3.5-9B"], "max_model_len": 65536})
    with pytest.raises(RuntimeError, match="max-model-len"):
        P.prepare(tmp_path, "qwen3.5-9b-thinking", cfg, 4)


QWEN_THINKING = {"temperature": 1.0, "top_p": 0.95, "top_k": 20, "min_p": 0.0,
                 "presence_penalty": 1.5, "repetition_penalty": 1.0}


def test_qwen_thinking_samples_as_its_model_card_advises_and_the_rest_stay_greedy():
    """Greedy decoding repeated Qwen3.5's thinking until the cap in every smoke
    answer, and the card's precise-task setting (temperature 0.6, no presence
    penalty) still did in five of eight; the thinking configuration takes the
    card's general thinking-mode setting. The direct configuration stays greedy
    and unchanged, so a root started before this change still resumes."""
    from scripts import run_api_panel as P
    cfg = P.model_config("qwen3.5-9b-thinking", P.PANEL["qwen3.5-9b-thinking"])
    assert cfg["sampling"] == QWEN_THINKING
    for name in ("qwen3.5-9b-direct", "mistral-small-3.2"):
        assert "sampling" not in P.model_config(name, P.PANEL[name])


def test_a_sampled_direct_control_isolates_thinking_from_decoding():
    """The decoding control: the same checkpoint with thinking off, sampled
    exactly as the thinking configuration (the card gives this setting for
    non-thinking reasoning tasks too). Direct greedy vs direct sampled isolates
    the decoding; direct sampled vs thinking isolates the thinking."""
    from scripts import run_api_panel as P
    direct = P.model_config("qwen3.5-9b-direct", P.PANEL["qwen3.5-9b-direct"])
    control = P.model_config("qwen3.5-9b-direct-sampled", P.PANEL["qwen3.5-9b-direct-sampled"])
    thinking = P.model_config("qwen3.5-9b-thinking", P.PANEL["qwen3.5-9b-thinking"])
    assert control["sampling"] == thinking["sampling"] == QWEN_THINKING
    assert control["chat_template_kwargs"] == direct["chat_template_kwargs"] == {"enable_thinking": False}
    same = ("model_id", "revision", "serving", "max_output_tokens", "inference_mode")
    assert {k: control[k] for k in same} == {k: direct[k] for k in same}


def test_a_cell_asks_for_the_decoding_its_configuration_declares(tmp_path, monkeypatch):
    from scripts import run_api_panel as P
    seen = {}

    def fake_run_model(*args, **kwargs):
        seen[kwargs["model_config"]["name"]] = (kwargs["temperature"], kwargs["seed"])

    monkeypatch.setattr(P, "run_model", fake_run_model)
    monkeypatch.setattr(P, "write_campaign_identity", lambda *a, **k: None)
    monkeypatch.setattr(P, "score_cell", lambda *a, **k: None)
    monkeypatch.setattr(P, "collect_cell", lambda *a, **k: {})
    monkeypatch.setattr(P, "sha", lambda path: "0" * 64)
    ctx = {"campaign": {"splits": {"carla": {"path": str(tmp_path)}}},
           "chosen": {"carla": [tmp_path / "scenario_1"]},
           "frozen": {"datasets": {"carla": []}}}
    for name in ("qwen3.5-9b-direct", "qwen3.5-9b-thinking", "qwen3.5-9b-direct-sampled"):
        P.run_cell(tmp_path, name, P.model_config(name, P.PANEL[name]), ctx, "carla", "video", 1)
    assert seen == {"qwen3.5-9b-direct": (0.0, 0), "qwen3.5-9b-thinking": (1.0, 0),
                    "qwen3.5-9b-direct-sampled": (1.0, 0)}


def test_a_panel_run_decodes_greedily_unless_its_configuration_declares_sampling():
    from analysis.run_model import _validate_model_config
    from scripts import run_api_panel as P
    greedy = P.model_config("qwen3.5-9b-direct", P.PANEL["qwen3.5-9b-direct"])
    sampled = P.model_config("qwen3.5-9b-thinking", P.PANEL["qwen3.5-9b-thinking"])
    _validate_model_config(greedy, "openai_compat", greedy["model_id"], 0.0,
                           allow_unadmitted_smoke=True)
    _validate_model_config(sampled, "openai_compat", sampled["model_id"], 1.0,
                           allow_unadmitted_smoke=True)
    with pytest.raises(ValueError, match="require temperature 0$"):
        _validate_model_config(greedy, "openai_compat", greedy["model_id"], 1.0,
                               allow_unadmitted_smoke=True)
    with pytest.raises(ValueError, match="require temperature 1$"):
        _validate_model_config(sampled, "openai_compat", sampled["model_id"], 0.0,
                               allow_unadmitted_smoke=True)


def test_a_cell_is_complete_only_under_the_decoding_its_configuration_declares(tmp_path):
    """A sampled answer completes a cell only when the configuration declares
    that sampling; greedy stays the protocol for every other cell."""
    import json
    from analysis.evaluation_campaign import check_cell, dataset_inventory
    from analysis.run_model import BACKENDS, run_model
    from tests.test_evaluation_campaign import make_protocol_dataset

    def spy(prompt, frame_paths, model_id, opts):
        return json.dumps({"predicted_branches": [], "predicted_counterfactuals": []})

    sampling = dict(QWEN_THINKING)
    config = dict(name="spy_sampled", backend="spy_sampled", model_id="m", adapter="x",
                  revision=None, inference_mode="thinking", runtime_ready=False,
                  adapter_implemented=True, max_output_tokens=65536,
                  thinking_trace_storage="separate_artifact", scorer_input="final_json_only")
    base = make_protocol_dataset(tmp_path, n=1)
    rows = dataset_inventory(base, 1, "nuscenes")
    BACKENDS["spy_sampled"] = spy
    try:
        run_model(str(base), str(tmp_path / "sampled"), backend="spy_sampled", model_id="m",
                  model_config=dict(config, sampling=sampling), temperature=1.0, seed=0,
                  provisional_smoke=True)
        run_model(str(base), str(tmp_path / "greedy"), backend="spy_sampled", model_id="m",
                  model_config=config, temperature=0.0, seed=0, provisional_smoke=True)
    finally:
        del BACKENDS["spy_sampled"]
    kw = dict(condition="video", configuration="spy_sampled", backend="spy_sampled")
    assert check_cell(tmp_path / "sampled", rows, sampling=sampling, **kw)["complete"]
    assert check_cell(tmp_path / "greedy", rows, **kw)["complete"]
    for out, declared in (("sampled", None), ("greedy", sampling),
                          ("sampled", dict(sampling, temperature=0.6))):
        errors = check_cell(tmp_path / out, rows, sampling=declared, **kw)["errors"]
        assert any("differs from final protocol" in e for e in errors), (out, declared)


def test_only_a_sampled_configuration_adds_its_decoding_to_the_cell(tmp_path, monkeypatch):
    from scripts import run_api_panel as P
    cells = {}
    monkeypatch.setattr(P, "run_model", lambda *a, **k: None)
    monkeypatch.setattr(P, "write_campaign_identity",
                        lambda campaign, cell: cells.setdefault(cell["configuration"], cell))
    monkeypatch.setattr(P, "score_cell", lambda *a, **k: None)
    monkeypatch.setattr(P, "collect_cell", lambda *a, **k: {})
    monkeypatch.setattr(P, "sha", lambda path: "0" * 64)
    ctx = {"campaign": {"splits": {"carla": {"path": str(tmp_path)}}},
           "chosen": {"carla": [tmp_path / "scenario_1"]},
           "frozen": {"datasets": {"carla": []}}}
    for name in ("qwen3.5-9b-direct", "qwen3.5-9b-thinking"):
        P.run_cell(tmp_path, name, P.model_config(name, P.PANEL[name]), ctx, "carla", "video", 1)
    assert "sampling" not in cells["qwen3.5-9b-direct"]
    assert cells["qwen3.5-9b-thinking"]["sampling"] == QWEN_THINKING


def test_the_recognition_probe_covers_the_four_models_not_the_decoding_control():
    from scripts import run_api_panel as P
    cfgs = P.probe_configs()
    assert sorted(cfgs) == ["cosmos-reason2-8b", "gemini-2.5-pro", "mistral-small-3.2",
                            "qwen3-vl-8b", "qwen3.5-9b-direct", "qwen3.5-9b-thinking"]
    for name in ("qwen3-vl-8b", "cosmos-reason2-8b"):
        assert cfgs[name]["serving"] == "vllm" and "sampling" not in cfgs[name]
    assert cfgs["qwen3.5-9b-thinking"]["sampling"] == QWEN_THINKING
    assert "sampling" not in cfgs["qwen3.5-9b-direct"] and "sampling" not in cfgs["mistral-small-3.2"]
    assert cfgs["gemini-2.5-pro"]["backend"] == "gemini"
    for name in ("qwen3.5-9b-direct", "qwen3.5-9b-thinking", "mistral-small-3.2"):
        assert cfgs[name]["backend"] == "openai_compat" and cfgs[name]["provider"] == "Parasail"


def test_a_sampled_model_is_probed_with_its_own_sampling():
    from scripts import run_api_panel as P
    sent = []

    def fake(prompt, frames, model_id, opts):
        sent.append((opts["temperature"], opts.get("seed")))
        return "{}"

    for name in ("qwen3.5-9b-thinking", "qwen3.5-9b-direct"):
        cfg = P.probe_configs()[name]
        P.probe_generator(cfg, fake)("p", [], cfg["model_id"], dict(cfg, seed=None, temperature=0.0))
    assert sent == [(1.0, 0), (0.0, None)]


def test_every_model_is_probed_on_the_same_twenty_nuscenes_scenes(tmp_path, monkeypatch):
    import json
    from types import SimpleNamespace
    from scripts import run_api_panel as P
    frozen = {"datasets": {"nuscenes": [{"name": f"scenario_{i}"} for i in range(1, 51)]}}
    (tmp_path / "qwen3.5-9b-direct").mkdir()
    (tmp_path / "qwen3.5-9b-direct" / "frozen.json").write_text(json.dumps(frozen))
    calls = []

    def fake_probe(cfg, name, fz, *, generator=None, resolved_model=None):
        calls.append((name, cfg["statistics"]["seed"], cfg["probe"]["limit"],
                      cfg["campaign_dir"], len(fz["datasets"]["nuscenes"])))
        return {"summary": {"n": 20}}

    monkeypatch.setattr(P, "run_panel_probe", fake_probe)
    P.cmd_probe(SimpleNamespace(out_root=tmp_path, config="all"))
    assert sorted(c[0] for c in calls) == sorted(P.probe_configs())
    assert {c[1:] for c in calls} == {(0, 20, str(tmp_path.resolve()), 50)}


def test_the_panel_report_ranks_four_models_and_uses_the_control_only_in_two_pairs(tmp_path, monkeypatch):
    """Two oracle-scored scenarios per split copied into the panel layout: the
    four models, the decoding control and the six baselines."""
    import json
    import shutil
    from pathlib import Path
    from types import SimpleNamespace
    import analysis.evaluation_campaign as c
    from analysis.run_model import run_model
    from scripts import run_api_panel as P
    from tests.test_evaluation_campaign import make_protocol_dataset
    scoring = tmp_path / "scoring"; scoring.mkdir()
    cfg = {"campaign_dir": str(scoring), "splits": {}}
    frozen = {"datasets": {}}
    for split in ("carla", "nuscenes"):
        base = make_protocol_dataset(tmp_path, split)
        cfg["splits"][split] = {"path": str(base)}
        frozen["datasets"][split] = c.dataset_inventory(base, 2, split)
    c.write_json(scoring / "frozen.json", frozen)
    for split in ("carla", "nuscenes"):
        cell = {"split": split, "condition": "video", "backend": "oracle", "out_dir": str(scoring / split)}
        c.write_campaign_identity(cfg, cell)
        run_model(cfg["splits"][split]["path"], cell["out_dir"], backend="oracle")
        c.score_cell(cfg, cell, frozen)
    monkeypatch.setattr(P, "SPLITS", {s: cfg["splits"][s]["path"] for s in ("carla", "nuscenes")})
    out_root = tmp_path / "api_panel"
    for key, cell, root in P.panel_cells(out_root):
        split, name, condition = key
        root.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(scoring / "frozen.json", root / "frozen.json")
        dst = Path(cell["out_dir"]); shutil.copytree(scoring / split, dst)
        for row in frozen["datasets"][split]:
            d = dst / row["name"]; m = c.read_json(d / "run_manifest.json")
            m.update(backend=cell["backend"], condition=condition, configuration_name=cell["configuration"])
            if condition == "history_json":
                m.update(frame_sha256=[], frame_count=0)
            if cell.get("sampling"):
                m.update(decoding="sampling", temperature=cell["sampling"]["temperature"], generation_seed=0)
            if cell["configuration"]:
                answer = '{"predicted_branches": [], "predicted_counterfactuals": []}'
                (d / "raw_response.txt").write_text(answer); (d / "final_response.txt").write_text(answer)
                c.write_json(d / "generation_metadata.json", {"output_tokens": 10})
                m.update(raw_response_sha256=c.sha(d / "raw_response.txt"),
                         final_response_sha256=c.sha(d / "final_response.txt"))
            c.write_json(d / "run_manifest.json", m)
        proof = c.read_json(dst / "score_provenance.json")
        proof["prediction_files"] = {f: c.sha(dst / f) for f in proof["prediction_files"]}
        c.write_json(dst / "score_provenance.json", proof)
    monkeypatch.setattr(P, "REPORT_BOOT", 50)
    P.cmd_report(SimpleNamespace(out_root=out_root))
    result = json.loads((out_root / "report" / "results.json").read_text())
    rows = result["rows"]
    comparisons = {r["comparison"] for r in rows}
    assert {"decoding_effect", "thinking_effect", "posttraining_effect", "floor_relative", "scores_common_subset"} <= comparisons
    pairs = {(r["comparison"], r["configuration"]) for r in rows if r["comparison"].endswith("_effect")}
    assert pairs == {("decoding_effect", "qwen3.5-9b-direct-sampled-qwen3.5-9b-direct"),
                     ("thinking_effect", "qwen3.5-9b-thinking-qwen3.5-9b-direct-sampled"),
                     ("posttraining_effect", "cosmos-reason2-8b-qwen3-vl-8b")}
    assert not set(P.SMOKE_ONLY) & set(P.REPORT_MODELS)
    assert result["prompt_variant"] == "full" and "information-matched" in result["controls"]
    assert result["cells_missing"] == []
    models = set(P.REPORT_MODELS)
    for comparison in ("scores_common_subset", "floor_relative"):
        assert {r["configuration"] for r in rows if r["comparison"] == comparison} == models
    scored = {r["configuration"] for r in rows if r["comparison"] == "scores"}
    assert "qwen3.5-9b-direct-sampled" not in scored and models <= scored
    assert (out_root / "report" / "tables.csv").is_file()


def test_a_root_holds_one_prompt_variant(tmp_path, monkeypatch):
    from scripts import run_api_panel as P
    from simulation.build_pilot import PROMPT_VERSION
    cfg = P.model_config("mistral-small-3.2", P.PANEL["mistral-small-3.2"])
    monkeypatch.setattr(P, "choose", lambda split, n: [])
    monkeypatch.setattr(P, "dataset_inventory", lambda subset, n, split: [])
    ctx = P.prepare(tmp_path, "mistral-small-3.2", cfg, 4, "recipe_withheld")
    assert ctx["prompt_variant"] == "recipe_withheld"
    assert ctx["frozen"]["prompt_version"] == PROMPT_VERSION + "+recipe_withheld"
    assert ctx["frozen"]["prompt_variant"] == "recipe_withheld"
    P.prepare(tmp_path, "mistral-small-3.2", cfg, 4, "recipe_withheld")   # same variant: resumes
    with pytest.raises(RuntimeError, match="prompt"):
        P.prepare(tmp_path, "mistral-small-3.2", cfg, 4)                   # arm A into an arm-B root


def test_qwen3_vl_and_cosmos_reason2_are_a_matched_pair_on_the_pod():
    """Cosmos-Reason2-8B is NVIDIA's physical-reasoning post-training of
    Qwen3-VL-8B-Instruct (model card), so the two run as one matched pair:
    the same size, the same serving, direct and greedy, pinned revisions from
    the panel manifest, and no thinking switch in either chat template."""
    from analysis.model_panel import MAX_OUTPUT_TOKENS, load_model_panel
    from scripts import run_api_panel as P
    panel = load_model_panel(os.path.join(os.path.dirname(__file__), "..", "analysis", "model_panel.yaml"))
    pins = {c["key"]: c["revision"] for c in panel["checkpoints"]}
    base = next(c for c in panel["checkpoints"] if c["key"] == "cosmos-reason2-8b")
    assert base["base_checkpoint"] == "qwen3-vl-8b-instruct"
    for name, model_id, key in (("qwen3-vl-8b", "Qwen/Qwen3-VL-8B-Instruct", "qwen3-vl-8b-instruct"),
                                ("cosmos-reason2-8b", "nvidia/Cosmos-Reason2-8B", "cosmos-reason2-8b")):
        cfg = P.model_config(name, P.PANEL[name])
        assert cfg["model_id"] == model_id and cfg["revision"] == pins[key]
        assert cfg["serving"] == "vllm" and cfg["api_base_url_env"] == "CORIVA_VLLM_BASE_URL"
        assert cfg["inference_mode"] == "direct" and cfg["enable_thinking"] is False
        assert cfg["chat_template_kwargs"] is None and "sampling" not in cfg
        assert cfg["max_output_tokens"] == MAX_OUTPUT_TOKENS


def test_the_pair_reasoning_configurations_append_the_cosmos_card_sentence(tmp_path):
    """The Cosmos-Reason2 card asks for its chain of thought by appending one
    sentence to the system prompt. The pair's reasoning configurations append
    exactly that sentence (Qwen3-VL gets the same as the control) and sample
    with the checkpoints' shipped generation config; the scene prompt and the
    rest of the system prompt are what the direct configurations send."""
    from analysis.evaluation_campaign import check_cell, dataset_inventory
    from analysis.run_model import BACKENDS, SYSTEM_PROMPT_BACKENDS, run_model
    from scripts import run_api_panel as P
    from tests.test_evaluation_campaign import make_protocol_dataset
    card = ("Answer the question in the following format: <think>\nyour reasoning\n</think>"
            "\n\n<answer>\nyour answer\n</answer>.")
    assert P.PAIR_REASONING_SUFFIX == "\n\n" + card
    assert P.PAIR_SAMPLING == dict(temperature=0.7, top_p=0.8, top_k=20, repetition_penalty=1.0)
    for model, direct in (("qwen3-vl-8b-reasoning", "qwen3-vl-8b"),
                          ("cosmos-reason2-8b-reasoning", "cosmos-reason2-8b")):
        cfg, base = P.model_config(model, P.PANEL[model]), P.model_config(direct, P.PANEL[direct])
        assert cfg["model_id"] == base["model_id"] and cfg["revision"] == base["revision"]
        assert cfg["inference_mode"] == "prompted_reasoning" and cfg["chat_template_kwargs"] is None
        assert cfg["system_prompt_suffix"] == P.PAIR_REASONING_SUFFIX
        assert cfg["sampling"] == P.PAIR_SAMPLING and "sampling" not in base
    seen = {}

    def spy(prompt, frame_paths, model_id, opts):
        seen[opts["name"]] = (opts["system_prompt"], prompt)
        return '{"predicted_branches": [], "predicted_counterfactuals": []}'

    base = make_protocol_dataset(tmp_path, n=1)
    BACKENDS["spy_system"] = spy
    SYSTEM_PROMPT_BACKENDS.add("spy_system")
    try:
        for name in ("qwen3-vl-8b", "qwen3-vl-8b-reasoning"):
            cfg = dict(P.model_config(name, P.PANEL[name]), backend="spy_system")
            run_model(str(base), str(tmp_path / name), backend="spy_system", model_id=cfg["model_id"],
                      model_config=cfg, temperature=(cfg.get("sampling") or {}).get("temperature", 0.0),
                      seed=0, provisional_smoke=True)
    finally:
        del BACKENDS["spy_system"]
        SYSTEM_PROMPT_BACKENDS.discard("spy_system")
    direct_system, direct_prompt = seen["qwen3-vl-8b"]
    system, prompt = seen["qwen3-vl-8b-reasoning"]
    assert system == direct_system + P.PAIR_REASONING_SUFFIX and prompt == direct_prompt
    saved = next((tmp_path / "qwen3-vl-8b-reasoning").glob("scenario_*"))
    assert (saved / "system_prompt.txt").read_text() == system
    rows = dataset_inventory(base, 1, "nuscenes")
    kw = dict(condition="video", configuration="qwen3-vl-8b-reasoning", backend="spy_system")
    assert check_cell(tmp_path / "qwen3-vl-8b-reasoning", rows, sampling=P.PAIR_SAMPLING, **kw)["complete"]


def test_a_reasoning_answer_is_scored_from_the_text_after_its_think_block(monkeypatch):
    """No reasoning parser runs on the pair's server: the runner stores the
    <think> block as the trace and parses the text after it, so a draft inside
    the reasoning never replaces the answer; an unclosed block is a schema
    failure, and an answer without a block is read whole."""
    from analysis.parse_prediction import parse_model_output
    from scripts import run_openrouter_cell as cell
    draft = '{"predicted_branches": [{"branch_id": "DRAFT"}]}'
    answer = '{"predicted_branches": [{"branch_id": "B1"}], "predicted_counterfactuals": []}'

    def call(content):
        monkeypatch.setattr(cell, "_stream_chat", lambda *a, **k: {
            "model": "m", "usage": {"completion_tokens": 9},
            "choices": [{"message": {"content": content}, "finish_reason": "stop"}]})
        return cell._openai_compat_backend("scene", [], "m", {"api_base_url": "http://pod/v1",
                                                              "api_key_env": None})

    result = call(f"<think>\nfirst try: {draft}\n</think>\n\n<answer>\n{answer}\n</answer>")
    assert result.thinking_trace.startswith("<think>") and "DRAFT" in result.thinking_trace
    l3, _, errors = parse_model_output(result.final_text, "s")
    assert not errors and [b["branch_id"] for b in l3["predicted_branches"]] == ["B1"]
    assert call(f"<think>\n{draft}").final_text == ""
    assert parse_model_output(call(answer).final_text, "s")[0]["predicted_branches"][0]["branch_id"] == "B1"


def test_a_report_of_a_video_only_arm_leaves_absent_cells_out_and_names_the_arm(tmp_path, monkeypatch):
    """Arm B runs the video condition only and may lack a model: the report
    uses the cells that exist, lists the absent ones, and describes the
    controls as information-advantaged."""
    import json
    import shutil
    from pathlib import Path
    from types import SimpleNamespace
    import analysis.evaluation_campaign as c
    from analysis.run_model import run_model
    from scripts import run_api_panel as P
    from tests.test_evaluation_campaign import make_protocol_dataset
    scoring = tmp_path / "scoring"; scoring.mkdir()
    cfg = {"campaign_dir": str(scoring), "splits": {}}
    frozen = {"datasets": {}, "prompt_variant": "recipe_withheld"}
    for split in ("carla", "nuscenes"):
        base = make_protocol_dataset(tmp_path, split)
        cfg["splits"][split] = {"path": str(base)}
        frozen["datasets"][split] = c.dataset_inventory(base, 2, split)
    c.write_json(scoring / "frozen.json", frozen)
    for split in ("carla", "nuscenes"):
        cell = {"split": split, "condition": "video", "backend": "oracle", "out_dir": str(scoring / split)}
        c.write_campaign_identity(cfg, cell)
        run_model(cfg["splits"][split]["path"], cell["out_dir"], backend="oracle")
        c.score_cell(cfg, cell, frozen)
    monkeypatch.setattr(P, "SPLITS", {s: cfg["splits"][s]["path"] for s in ("carla", "nuscenes")})
    out_root = tmp_path / "api_panel_b"
    kept = {"gemini-2.5-pro", "qwen3.5-9b-thinking", "qwen3.5-9b-direct", "qwen3.5-9b-direct-sampled"}
    for key, cell, root in P.panel_cells(out_root):
        split, name, condition = key
        if name in P.PANEL or name == P.GEMINI:
            if name not in kept or condition != "video":
                continue                                    # arm B: video only, no Mistral, no pair
        root.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(scoring / "frozen.json", root / "frozen.json")
        dst = Path(cell["out_dir"]); shutil.copytree(scoring / split, dst)
        for row in frozen["datasets"][split]:
            d = dst / row["name"]; m = c.read_json(d / "run_manifest.json")
            m.update(backend=cell["backend"], condition=condition, configuration_name=cell["configuration"],
                     prompt_variant="recipe_withheld")
            if cell.get("sampling"):
                m.update(decoding="sampling", temperature=cell["sampling"]["temperature"], generation_seed=0)
            if cell["configuration"]:
                answer = '{"predicted_branches": [], "predicted_counterfactuals": []}'
                (d / "raw_response.txt").write_text(answer); (d / "final_response.txt").write_text(answer)
                c.write_json(d / "generation_metadata.json", {"output_tokens": 10})
                m.update(raw_response_sha256=c.sha(d / "raw_response.txt"),
                         final_response_sha256=c.sha(d / "final_response.txt"))
            c.write_json(d / "run_manifest.json", m)
        proof = c.read_json(dst / "score_provenance.json")
        proof["prediction_files"] = {f: c.sha(dst / f) for f in proof["prediction_files"]}
        c.write_json(dst / "score_provenance.json", proof)
    monkeypatch.setattr(P, "REPORT_BOOT", 50)
    P.cmd_report(SimpleNamespace(out_root=out_root))
    result = json.loads((out_root / "report" / "results.json").read_text())
    assert result["prompt_variant"] == "recipe_withheld" and "information-ADVANTAGED" in result["controls"]
    missing = {tuple(k) for k in result["cells_missing"]}
    assert ("carla", "mistral-small-3.2", "video") in missing and ("carla", "gemini-2.5-pro", "history_json") in missing
    rows = result["rows"]
    scored = {r["configuration"] for r in rows if r["comparison"] == "scores"}
    assert scored & set(P.REPORT_MODELS) == {"gemini-2.5-pro", "qwen3.5-9b-thinking", "qwen3.5-9b-direct"}
    assert {r["condition"] for r in rows if r["comparison"] == "scores"} == {"video"}
    assert not [r for r in rows if r["comparison"] in ("images_added", "history_added", "representation_difference")]
    assert {r["comparison"] for r in rows if r["comparison"].endswith("_effect")} == {"decoding_effect", "thinking_effect"}


def test_a_root_refuses_to_resume_on_inputs_whose_files_changed(tmp_path, monkeypatch):
    """The same scenario names with another input fingerprint is another dataset."""
    from scripts import run_api_panel as P
    cfg = P.model_config("mistral-small-3.2", P.PANEL["mistral-small-3.2"])
    rows = {"carla": [{"name": "scenario_1", "fingerprint": "a" * 64}], "nuscenes": []}
    monkeypatch.setattr(P, "choose", lambda split, n: [])
    monkeypatch.setattr(P, "dataset_inventory", lambda subset, n, split: list(rows[split]))
    P.prepare(tmp_path, "mistral-small-3.2", cfg, 1)
    P.prepare(tmp_path, "mistral-small-3.2", cfg, 1)                    # unchanged: resumes
    rows["carla"][0]["fingerprint"] = "b" * 64
    with pytest.raises(RuntimeError, match="files have changed"):
        P.prepare(tmp_path, "mistral-small-3.2", cfg, 1)


def _scored_panel(tmp_path, monkeypatch, out_root, keep, variant, with_baselines):
    """A panel root with two oracle-scored scenarios per split copied into the
    cells of `keep` (video only) and, when asked, the six baselines."""
    import shutil
    from pathlib import Path
    import analysis.evaluation_campaign as c
    from analysis.run_model import run_model
    from scripts import run_api_panel as P
    from tests.test_evaluation_campaign import make_protocol_dataset
    scoring = tmp_path / f"scoring_{out_root.name}"; scoring.mkdir()
    cfg = {"campaign_dir": str(scoring), "splits": {}}
    frozen = {"datasets": {}, "prompt_variant": variant}
    for split in ("carla", "nuscenes"):
        base = tmp_path / "data" / split
        if not base.exists():
            make_protocol_dataset(tmp_path / "data", split)
        cfg["splits"][split] = {"path": str(base)}
        frozen["datasets"][split] = c.dataset_inventory(base, 2, split)
    c.write_json(scoring / "frozen.json", frozen)
    for split in ("carla", "nuscenes"):
        cell = {"split": split, "condition": "video", "backend": "oracle", "out_dir": str(scoring / split)}
        c.write_campaign_identity(cfg, cell)
        run_model(cfg["splits"][split]["path"], cell["out_dir"], backend="oracle")
        c.score_cell(cfg, cell, frozen)
    monkeypatch.setattr(P, "SPLITS", {s: cfg["splits"][s]["path"] for s in ("carla", "nuscenes")})
    for key, cell, root in P.panel_cells(out_root):
        split, name, condition = key
        if name in P.BASELINES:
            if not with_baselines:
                continue
        elif name not in keep or condition != "video":
            continue
        root.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(scoring / "frozen.json", root / "frozen.json")
        dst = Path(cell["out_dir"]); shutil.copytree(scoring / split, dst)
        for row in frozen["datasets"][split]:
            d = dst / row["name"]; m = c.read_json(d / "run_manifest.json")
            m.update(backend=cell["backend"], condition=condition, configuration_name=cell["configuration"],
                     prompt_variant=variant)
            if cell.get("sampling"):
                m.update(decoding="sampling", temperature=cell["sampling"]["temperature"], generation_seed=0)
            if cell["configuration"]:
                answer = '{"predicted_branches": [], "predicted_counterfactuals": []}'
                (d / "raw_response.txt").write_text(answer); (d / "final_response.txt").write_text(answer)
                c.write_json(d / "generation_metadata.json", {"output_tokens": 10})
                m.update(raw_response_sha256=c.sha(d / "raw_response.txt"),
                         final_response_sha256=c.sha(d / "final_response.txt"))
            c.write_json(d / "run_manifest.json", m)
        proof = c.read_json(dst / "score_provenance.json")
        proof["prediction_files"] = {f: c.sha(dst / f) for f in proof["prediction_files"]}
        c.write_json(dst / "score_provenance.json", proof)
    monkeypatch.setattr(P, "REPORT_BOOT", 50)


def test_an_arm_without_its_own_baselines_reports_what_it_can_or_reads_them_from_another_root(tmp_path, monkeypatch):
    import json
    from types import SimpleNamespace
    from scripts import run_api_panel as P
    arm_a = tmp_path / "api_panel"; arm_b = tmp_path / "api_panel_b"
    _scored_panel(tmp_path, monkeypatch, arm_a, {"gemini-2.5-pro"}, "full", with_baselines=True)
    _scored_panel(tmp_path, monkeypatch, arm_b, {"gemini-2.5-pro", "qwen3.5-9b-thinking"}, "recipe_withheld",
                  with_baselines=False)
    P.cmd_report(SimpleNamespace(out_root=arm_b, baselines_root=None))
    result = json.loads((arm_b / "report" / "results.json").read_text())
    comparisons = {r["comparison"] for r in result["rows"]}
    assert "scores" in comparisons and "scores_common_subset" in comparisons
    assert not comparisons & {"floor_relative", "skill_vs_floor", "valid_output_vs_ceiling", "contamination_excess"}
    assert ["carla", "oracle", "video"] in result["cells_missing"]
    assert result["baselines_root"] == str((arm_b / "baselines").resolve())
    P.cmd_report(SimpleNamespace(out_root=arm_b, baselines_root=arm_a / "baselines"))
    result = json.loads((arm_b / "report" / "results.json").read_text())
    comparisons = {r["comparison"] for r in result["rows"]}
    assert {"floor_relative", "skill_vs_floor", "valid_output_vs_ceiling"} <= comparisons
    assert ["carla", "oracle", "video"] not in result["cells_missing"]
    assert result["baselines_root"] == str((arm_a / "baselines").resolve())
    assert result["prompt_variant"] == "recipe_withheld"
    assert str(arm_a / "baselines") in result["frozen_sha256"]
