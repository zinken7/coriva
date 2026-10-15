"""Model runner — drive a backend over a GT dir and write l3_pre/l4_pre files.

Backend-agnostic: `mock` (constant, plumbing) and `oracle` (echoes GT -> should
score high through the real assemble+benchmark pipeline) are testable with no
GPU/key. The `local` backend (Qwen3-VL on GPU) is imported lazily so this module
loads without torch/transformers present.
"""

import json
import os
import subprocess
import sys

import pytest

ROOT = os.path.join(os.path.dirname(__file__), "..")
sys.path.insert(0, ROOT)


def _make_gt_dir(tmp_path, n=2):
    from analysis.rq3a import make_synthetic_gts
    gt_dir = tmp_path / "gt"
    for i, gt in enumerate(make_synthetic_gts(n), start=1):
        gt["scenario_id"] = f"nuscenes_{i:04d}"
        d = gt_dir / f"scenario_{i}"
        d.mkdir(parents=True)
        with open(d / "ground_truth.json", "w") as f:
            json.dump(gt, f)
    return gt_dir


def _ready_panel_config(name="qwen3-vl-8b-instruct"):
    from analysis.model_panel import load_model_panel, resolve_configurations
    panel = load_model_panel("analysis/model_panel.yaml")
    config = resolve_configurations(panel, [name])[0]
    config["runtime_ready"] = True
    config["runtime_blocker"] = None
    config["model_panel_version"] = panel["panel_version"]
    config["smoke_admission_version"] = "coriva-smoke-admission-1.0"
    config["smoke_admission_sha256"] = "d" * 64
    config["smoke_code_commit"] = "e" * 40
    config["smoke_completed_at"] = "2026-08-18T00:00:00+00:00"
    return config


def test_run_model_mock_writes_prediction_files(tmp_path):
    from analysis.run_model import run_model
    gt_dir = _make_gt_dir(tmp_path, 2)
    res = tmp_path / "res"
    counts = run_model(str(gt_dir), str(res), backend="mock", model_id="dev")
    assert counts["scenarios"] == 2
    l3 = json.load(open(res / "scenario_1" / "l3_pre.json"))
    l4 = json.load(open(res / "scenario_1" / "l4_pre.json"))
    assert "predicted_branches" in l3 and "predicted_counterfactuals" in l4
    assert l3["scenario_id"] == "nuscenes_0001"


def test_protocol_frame_contract_rejects_missing_frame_directory(tmp_path):
    from analysis.run_model import _resolve_frames

    with pytest.raises(ValueError, match="frame directory is missing"):
        _resolve_frames(str(tmp_path / "scenario_1"), 8, exact_count=8)


def test_protocol_frame_contract_rejects_wrong_count(tmp_path):
    from analysis.run_model import _resolve_frames

    frame_dir = tmp_path / "scenario_1" / "frames"
    frame_dir.mkdir(parents=True)
    for index in range(7):
        (frame_dir / f"{index:03d}.jpg").write_bytes(b"fake")
    with pytest.raises(ValueError, match="found 7, expected exactly 8"):
        _resolve_frames(str(tmp_path / "scenario_1"), 8, exact_count=8)


def test_run_model_rejects_track_mismatch_with_manifest(tmp_path):
    from analysis.run_model import run_model
    from simulation.protocol import protocol_manifest

    gt_dir = _make_gt_dir(tmp_path, 1)
    scenario_dir = gt_dir / "scenario_1"
    with open(scenario_dir / "input_manifest.json", "w") as f:
        json.dump(protocol_manifest(track="primary"), f)
    with pytest.raises(ValueError, match="does not match scenario manifest"):
        run_model(str(gt_dir), str(tmp_path / "res"), backend="mock",
                  track="challenge_6s")


def test_run_model_unknown_backend_raises(tmp_path):
    from analysis.run_model import run_model
    gt_dir = _make_gt_dir(tmp_path, 1)
    with pytest.raises(KeyError):
        run_model(str(gt_dir), str(tmp_path / "r"), backend="does_not_exist")


def test_run_model_oracle_scores_high_end_to_end(tmp_path):
    """Oracle echoes GT through the runner+parser; the real assemble+benchmark
    pipeline must then produce a high CTMS — proving the full loop reaches
    fidelity, not just that it runs."""
    from analysis.run_model import run_model
    gt_dir = _make_gt_dir(tmp_path, 3)
    res = tmp_path / "res"
    run_model(str(gt_dir), str(res), backend="oracle", model_id="oracle")

    gt_ds = tmp_path / "gt.json"
    pred_ds = tmp_path / "pred.json"
    report = tmp_path / "report.json"
    subprocess.run([sys.executable, "scripts/assemble_dataset.py",
                    "--gt-dir", str(gt_dir), "--pred-dir", str(res),
                    "--out-gt", str(gt_ds), "--out-pred", str(pred_ds)],
                   cwd=ROOT, check=True, capture_output=True)
    subprocess.run([sys.executable, "run_benchmark.py",
                    "--gt", str(gt_ds), "--pred", str(pred_ds), "--output", str(report)],
                   cwd=ROOT, check=True, capture_output=True)
    summary = json.load(open(report))["summary"]
    assert summary["CTMS_mean"] > 0.5, summary


def test_local_backend_module_imports_without_torch():
    # Importing the runner (and registering the local backend) must not require
    # torch/transformers — those load lazily inside generate().
    import importlib
    importlib.import_module("analysis.run_model")
    assert "local" in importlib.import_module("analysis.run_model").BACKENDS


def test_skip_existing_does_not_rerun_completed_scenarios(tmp_path):
    """Resume safety for paid GPU runs: with skip_existing=True, scenarios whose
    l3_pre.json and l4_pre.json already exist must not hit the backend again."""
    from analysis.run_model import run_model, BACKENDS
    gt_dir = _make_gt_dir(tmp_path, 2)
    res = tmp_path / "res"
    run_model(str(gt_dir), str(res), backend="mock")

    def exploding(prompt, frame_paths, model_id, opts):
        raise AssertionError("backend must not be called for completed scenarios")

    BACKENDS["exploding"] = exploding
    try:
        stats = run_model(str(gt_dir), str(res), backend="exploding",
                          skip_existing=True)
    finally:
        del BACKENDS["exploding"]
    assert stats["skipped"] == 2
    assert stats["scenarios"] == 0


def test_skip_existing_still_runs_missing_scenarios(tmp_path):
    from analysis.run_model import run_model
    gt_dir = _make_gt_dir(tmp_path, 2)
    res = tmp_path / "res"
    run_model(str(gt_dir), str(res), backend="mock")
    os.remove(res / "scenario_2" / "l3_pre.json")
    stats = run_model(str(gt_dir), str(res), backend="mock", skip_existing=True)
    assert stats["skipped"] == 1
    assert stats["scenarios"] == 1


def test_panel_run_archives_raw_final_trace_and_generation_metadata(tmp_path):
    from analysis.backends.result import GenerationResult
    from analysis.run_model import BACKENDS, run_model
    gt_dir = _make_gt_dir(tmp_path, 1)
    out_dir = tmp_path / "panel-result"
    config = _ready_panel_config("qwen3-vl-8b-instruct")
    seen = {}
    final = json.dumps({"predicted_branches": [],
                        "predicted_counterfactuals": []})

    def fake_local(prompt, frame_paths, model_id, opts):
        seen.update(opts)
        return GenerationResult(
            final_text=final,
            raw_text="private reasoning\n" + final,
            thinking_trace="private reasoning",
            metadata={"output_tokens": 42, "model_commit_hash": config["revision"]})

    original = BACKENDS["local"]
    BACKENDS["local"] = fake_local
    try:
        stats = run_model(
            str(gt_dir), str(out_dir), backend="local",
            model_id=config["model_id"], condition="video",
            model_config=config)
    finally:
        BACKENDS["local"] = original

    scenario = out_dir / "scenario_1"
    assert stats["scenarios"] == 1
    assert (scenario / "raw_response.txt").read_text().startswith("private reasoning")
    assert (scenario / "final_response.txt").read_text() == final
    assert (scenario / "thinking_trace.txt").read_text() == "private reasoning"
    metadata = json.load(open(scenario / "generation_metadata.json"))
    manifest = json.load(open(scenario / "run_manifest.json"))
    assert metadata["output_tokens"] == 42
    assert seen["adapter"] == "qwen3_vl"
    assert seen["revision"] == config["revision"]
    assert seen["max_output_tokens"] == 65536
    assert manifest["configuration_name"] == config["name"]
    assert manifest["model_revision"] == config["revision"]
    assert manifest["max_output_tokens"] == 65536
    assert manifest["thinking_trace_present"] is True
    from scripts.assemble_dataset import assemble_gt_dataset, assemble_pred_dataset
    pred_dataset = assemble_pred_dataset(
        str(out_dir), assemble_gt_dataset(str(gt_dir)))
    assert pred_dataset["model_panel_version"] == "coriva-model-panel-1.2"
    assert pred_dataset["configuration_name"] == config["name"]
    assert pred_dataset["model_revision"] == config["revision"]


def test_panel_run_rejects_configuration_before_backend_when_not_admitted(tmp_path):
    from analysis.run_model import run_model
    from analysis.model_panel import load_model_panel, resolve_configurations
    panel = load_model_panel("analysis/model_panel.yaml")
    config = resolve_configurations(panel, ["qwen3-vl-8b-instruct"])[0]
    with pytest.raises(RuntimeError, match="not runtime-admitted"):
        run_model(
            str(_make_gt_dir(tmp_path, 1)), str(tmp_path / "out"),
            backend=config["backend"], model_id=config["model_id"],
            model_config=config)


def test_panel_resume_rejects_results_from_different_revision(tmp_path):
    from analysis.backends.result import GenerationResult
    from analysis.run_model import BACKENDS, run_model
    gt_dir = _make_gt_dir(tmp_path, 1)
    out_dir = tmp_path / "resume"
    config = _ready_panel_config()
    final = json.dumps({"predicted_branches": [],
                        "predicted_counterfactuals": []})

    original = BACKENDS["local"]
    BACKENDS["local"] = lambda *args: GenerationResult(final, final)
    try:
        run_model(str(gt_dir), str(out_dir), backend="local",
                  model_id=config["model_id"], model_config=config)
        changed = dict(config, revision="f" * 40)
        with pytest.raises(ValueError, match="existing result metadata conflicts"):
            run_model(str(gt_dir), str(out_dir), backend="local",
                      model_id=changed["model_id"], model_config=changed,
                      skip_existing=True)
    finally:
        BACKENDS["local"] = original


def test_gemini_without_smoke_records_policy_and_resumes_same_answer(tmp_path, monkeypatch):
    from analysis.backends.result import GenerationResult
    from analysis.run_model import BACKENDS, run_model
    gt_dir = _make_gt_dir(tmp_path, 1)
    out_dir = tmp_path / 'gemini'
    config = dict(_ready_panel_config('gemini-2.5-pro'), smoke_status='not_run')
    final = json.dumps({'predicted_branches': [], 'predicted_counterfactuals': []})
    calls = []
    def fake_gemini(*args):
        calls.append(args)
        return GenerationResult(final, final, metadata={'model_version': 'provider-version'})
    monkeypatch.setitem(BACKENDS, 'gemini', fake_gemini)
    kwargs = dict(backend='gemini', model_id=config['model_id'], model_config=config)
    run_model(str(gt_dir), str(out_dir), **kwargs)
    manifest = json.loads((out_dir / 'scenario_1/run_manifest.json').read_text())
    assert manifest['smoke_status'] == 'not_run'
    assert manifest['generation_metadata']['model_version'] == 'provider-version'
    assert run_model(str(gt_dir), str(out_dir), skip_existing=True, **kwargs)['skipped'] == 1
    assert len(calls) == 1
    config['smoke_status'] = 'passed'
    with pytest.raises(ValueError, match='existing result metadata conflicts'):
        run_model(str(gt_dir), str(out_dir), skip_existing=True, **kwargs)


@pytest.mark.parametrize("mutation,match", [
    (lambda manifest: manifest.update(model_revision="f" * 40),
     "mixes model configuration metadata"),
    (lambda manifest: manifest["generation_metadata"].update(
        model_version="gemini-version-2"), "mixes provider model versions"),
    (lambda manifest: manifest["generation_metadata"].pop("model_version"),
     "missing provider model versions"),
])
def test_assembly_rejects_mixed_panel_or_provider_identity(
        tmp_path, mutation, match):
    from scripts.assemble_dataset import assemble_gt_dataset, assemble_pred_dataset

    gt_dir = _make_gt_dir(tmp_path, 2)
    pred_dir = tmp_path / "pred"
    base_manifest = {
        "protocol_version": "legacy",
        "scorer_input_track": "primary",
        "condition": "video",
        "scorer_version": "ctms-scorer-1.2",
        "parser_version": "prediction-parser-1.0",
        "model_panel_version": "coriva-model-panel-1.2",
        "configuration_name": "gemini-2.5-pro",
        "model_id": "gemini-2.5-pro",
        "model_revision": None,
        "adapter": "gemini_api",
        "inference_mode": "default",
        "max_output_tokens": 65536,
        "smoke_admission_version": "coriva-smoke-admission-1.0",
        "smoke_admission_sha256": "d" * 64,
        "smoke_code_commit": "e" * 40,
        "generation_metadata": {"model_version": "gemini-version-1"},
    }
    for index in (1, 2):
        scenario = pred_dir / f"scenario_{index}"
        scenario.mkdir(parents=True)
        (scenario / "l3_pre.json").write_text(
            json.dumps({"predicted_branches": []}))
        (scenario / "l4_pre.json").write_text(
            json.dumps({"predicted_counterfactuals": []}))
        manifest = dict(base_manifest)
        manifest["generation_metadata"] = dict(
            base_manifest["generation_metadata"])
        (scenario / "run_manifest.json").write_text(json.dumps(manifest))

    gt_dataset = assemble_gt_dataset(str(gt_dir))
    assembled = assemble_pred_dataset(str(pred_dir), gt_dataset)
    assert assembled["model_revision"] is None
    assert assembled["provider_model_version"] == "gemini-version-1"

    changed_path = pred_dir / "scenario_2" / "run_manifest.json"
    changed = json.loads(changed_path.read_text())
    mutation(changed)
    changed_path.write_text(json.dumps(changed))
    with pytest.raises(ValueError, match=match):
        assemble_pred_dataset(str(pred_dir), gt_dataset)


def test_run_model_records_the_prompt_variant(tmp_path):
    from analysis.run_model import run_model
    gt_dir = _make_gt_dir(tmp_path, 1)
    res = tmp_path / "res"
    run_model(str(gt_dir), str(res), backend="mock", model_id="dev", prompt_variant="recipe_withheld")
    manifest = json.loads(next(res.glob("scenario_*/run_manifest.json")).read_text())
    assert manifest["prompt_variant"] == "recipe_withheld"
    assert manifest["prompt_version"].endswith("+recipe_withheld")
    with pytest.raises(ValueError):
        run_model(str(gt_dir), str(tmp_path / "res2"), backend="mock", model_id="dev", prompt_variant="nope")


def test_a_panel_scenario_resumes_only_under_the_prompt_variant_it_was_run_with(tmp_path):
    from analysis.run_model import _existing_output_complete
    cfg = {"name": "cfg", "model_id": "m", "revision": None, "adapter": "a", "inference_mode": "direct",
           "max_output_tokens": 10, "cache_implementation": None}
    out = tmp_path / "scenario_1"; out.mkdir()
    for name in ("l3_pre.json", "l4_pre.json", "generation_metadata.json"):
        (out / name).write_text("{}")
    for name in ("raw_response.txt", "final_response.txt"):
        (out / name).write_text("x")
    manifest = {"configuration_name": "cfg", "model_id": "m", "model_revision": None, "adapter": "a",
                "inference_mode": "direct", "max_output_tokens": 10, "cache_implementation": None,
                "condition": "video"}
    (out / "run_manifest.json").write_text(json.dumps(manifest))          # pre-variant manifest = full
    assert _existing_output_complete(str(out), cfg, "video", "full")
    with pytest.raises(ValueError, match="prompt_variant"):
        _existing_output_complete(str(out), cfg, "video", "recipe_withheld")
    (out / "run_manifest.json").write_text(json.dumps(dict(manifest, prompt_variant="recipe_withheld")))
    assert _existing_output_complete(str(out), cfg, "video", "recipe_withheld")
    with pytest.raises(ValueError, match="prompt_variant"):
        _existing_output_complete(str(out), cfg, "video", "full")


def test_a_panel_scenario_is_not_resumed_under_another_prompt_text(tmp_path):
    from analysis.run_model import _existing_output_complete
    cfg = {"name": "cfg", "model_id": "m", "revision": None, "adapter": "a", "inference_mode": "direct",
           "max_output_tokens": 10, "cache_implementation": None}
    out = tmp_path / "scenario_1"; out.mkdir()
    for name in ("l3_pre.json", "l4_pre.json", "generation_metadata.json"):
        (out / name).write_text("{}")
    for name in ("raw_response.txt", "final_response.txt"):
        (out / name).write_text("x")
    manifest = {"configuration_name": "cfg", "model_id": "m", "model_revision": None, "adapter": "a",
                "inference_mode": "direct", "max_output_tokens": 10, "cache_implementation": None,
                "condition": "video", "prompt_variant": "full", "prompt_sha256": "p1", "system_prompt_sha256": "s1"}
    (out / "run_manifest.json").write_text(json.dumps(manifest))
    assert _existing_output_complete(str(out), cfg, "video", "full", "p1", "s1")
    with pytest.raises(ValueError, match="another prompt text"):
        _existing_output_complete(str(out), cfg, "video", "full", "p1", "s2")   # system prompt edited
    with pytest.raises(ValueError, match="another prompt text"):
        _existing_output_complete(str(out), cfg, "video", "full", "p2", "s1")   # scene prompt edited
