"""Step 4 smoke evidence, admission and runtime-gate regressions."""

import hashlib
import json
import os
import subprocess
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from analysis.model_panel import load_model_panel, resolve_configurations
from analysis.smoke_admission import (
    SMOKE_ADMISSION_VERSION, SMOKE_EVIDENCE_VERSION, admit_panel,
    build_admission, input_file_fingerprint, inspect_cell_output,
    scenario_input_fingerprint, validate_admission,
)
from simulation.protocol import INPUT_CONDITIONS


PANEL_PATH = "analysis/model_panel.yaml"
CODE_COMMIT = "c" * 40
SCENARIO = "scenario_smoke"
INPUT_FILES = [
    {"path": path, "sha256": hashlib.sha256(path.encode()).hexdigest()}
    for path in (
        "ground_truth.json", "input_manifest.json", "observed_state.json",
        "observed_history.json", "query.json", "ontology.json",
        *(f"frames/{index:03d}.jpg" for index in range(8)),
    )
]
INPUT_FINGERPRINT = input_file_fingerprint(INPUT_FILES)


def _repository():
    return {"code_commit": CODE_COMMIT, "worktree_clean": True}


@pytest.mark.parametrize("git_cwd", [".", "Coding"])
@pytest.mark.parametrize("state", ["clean", "modified", "untracked_runtime"])
def test_repository_state_tracks_runtime_paths_from_repo_or_coding(tmp_path, monkeypatch,
                                                                 git_cwd, state):
    from analysis import smoke_admission

    repo = tmp_path / "repo"
    repo.mkdir()

    def git(*args):
        return subprocess.run(["git", *args], cwd=repo, check=True,
                              capture_output=True, text=True)

    git("init")
    git("config", "user.name", "Test")
    git("config", "user.email", "test@example.invalid")
    git("config", "commit.gpgsign", "false")
    paths = [
        "analysis/smoke_admission.py", "analysis/model_panel.py",
        "analysis/model_panel.yaml", "analysis/run_model.py", "analysis/run_matrix.py",
        "analysis/backends/local_hf.py", "analysis/backends/api.py",
        "analysis/backends/result.py", "simulation/build_pilot.py",
        "simulation/protocol.py", "ctms/ctms.py",
    ]
    for name in paths:
        path = repo / "Coding" / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("# fixture\n")
    git("add", "Coding")
    git("commit", "-m", "Runtime fixture")
    target = "Coding/analysis/run_model.py"
    if state == "modified":
        (repo / target).write_text("# modified\n")
    elif state == "untracked_runtime":
        git("rm", "--cached", target)
        git("commit", "-m", "Leave runtime file untracked")

    monkeypatch.setattr(smoke_admission, "__file__",
                        str(repo / "Coding/analysis/smoke_admission.py"))
    actual = smoke_admission.repository_state(str(repo / git_cwd))
    assert actual["code_commit"] == git("rev-parse", "HEAD").stdout.strip()
    assert actual["worktree_clean"] is (state == "clean")
    assert actual["tracked_dirty_entry_count"] == (1 if state == "modified" else 0)
    assert actual["untracked_required_runtime_files"] == (
        [target] if state == "untracked_runtime" else [])
    if git_cwd == "Coding":
        assert smoke_admission.repository_state() == actual


def _valid_evidence(panel):
    configs = resolve_configurations(
        panel, [item["name"] for item in panel["configurations"]])
    records = []
    for config in configs:
        for condition in INPUT_CONDITIONS:
            metadata = {"model_version": "gemini-2.5-pro-provider-1"} \
                if config["backend"] == "gemini" else {}
            records.append({
                "evidence_version": SMOKE_EVIDENCE_VERSION,
                "panel_version": panel["panel_version"],
                "protocol_version": panel["protocol_version"],
                "scorer_version": panel["scorer_version"],
                "transformers_revision": panel["runtime_environment"][
                    "transformers_revision"],
                "weight_dtype": "bfloat16",
                "configuration_name": config["name"],
                "checkpoint": config["checkpoint"],
                "model_id": config["model_id"],
                "model_revision": config.get("revision"),
                "backend": config["backend"],
                "adapter": config["adapter"],
                "inference_mode": config["inference_mode"],
                "enable_thinking": config.get("enable_thinking"),
                "condition": condition,
                "max_output_tokens": config["max_output_tokens"],
                "scenario_name": SCENARIO,
                "input_fingerprint": INPUT_FINGERPRINT,
                "input_files": [dict(item) for item in INPUT_FILES],
                "code_commit": CODE_COMMIT,
                "worktree_clean": True,
                "status": "passed",
                "completed_at": "2026-08-18T01:00:00+00:00",
                "checks": {"mechanical_runtime_gate": True},
                "failed_checks": [],
                "observations": {
                    "parse_failure_count": 1,
                    "parse_failures": [{"errors": ["schema failure is measured"]}],
                    "generation_metadata": metadata,
                },
            })
    return records


def test_valid_15_cell_evidence_admits_all_configurations(tmp_path):
    panel = load_model_panel(PANEL_PATH)
    admission = build_admission(panel, _valid_evidence(panel))
    assert admission["status"] == "passed"
    assert admission["expected_cell_count"] == 15
    assert admission["observed_cell_count"] == 15
    assert admission["gemini_model_version"] == "gemini-2.5-pro-provider-1"
    # A schema failure is a benchmark compliance outcome, not model cherry-picking.
    assert all(cell["observations"]["parse_failure_count"] == 1
               for cell in admission["cells"])
    validate_admission(panel, admission)

    path = tmp_path / "admission.json"
    path.write_text(json.dumps(admission, sort_keys=True))
    admitted = admit_panel(panel, str(path), repository=_repository())
    configs = resolve_configurations(
        admitted, [item["name"] for item in admitted["configurations"]])
    assert all(config["runtime_ready"] is True for config in configs)
    assert all(config["runtime_blocker"] is None for config in configs)
    assert all(config["smoke_admission_version"] == SMOKE_ADMISSION_VERSION
               for config in configs)
    assert all(len(config["smoke_admission_sha256"]) == 64 for config in configs)


@pytest.mark.parametrize("mutation", [
    lambda records: records.pop(),
    lambda records: records[0]["checks"].update(mechanical_runtime_gate=False),
    lambda records: records[0].update(code_commit="d" * 40),
    lambda records: records[-1]["observations"]["generation_metadata"].update(
        model_version="different-provider-version"),
])
def test_incomplete_failed_or_mixed_evidence_never_admits(mutation):
    panel = load_model_panel(PANEL_PATH)
    records = _valid_evidence(panel)
    mutation(records)
    admission = build_admission(panel, records)
    assert admission["status"] == "failed"
    with pytest.raises(ValueError, match="does not pass"):
        validate_admission(panel, admission)


def test_admission_summary_tampering_is_detected():
    panel = load_model_panel(PANEL_PATH)
    admission = build_admission(panel, _valid_evidence(panel))
    admission["code_commit"] = "0" * 40
    with pytest.raises(ValueError, match="summary mismatch"):
        validate_admission(panel, admission)


def test_explicit_gemini_skip_admits_local_smoke_without_claiming_gemini_pass(tmp_path):
    panel = load_model_panel(PANEL_PATH)
    records = [r for r in _valid_evidence(panel) if r['backend'] == 'local']
    assert build_admission(panel, records)['status'] == 'failed'
    admission = build_admission(panel, records, skip_gemini_smoke=True)
    validate_admission(panel, admission)
    assert admission['expected_cell_count'] == admission['observed_cell_count'] == 12
    assert admission['gemini_smoke'] == {'status': 'not_run', 'reason': 'user_requested_skip'}
    assert admission['gemini_model_version'] is None
    assert all(r['backend'] == 'local' for r in admission['cells'])
    path = tmp_path / 'admission.json'
    path.write_text(json.dumps(admission))
    admitted = admit_panel(panel, str(path), repository=_repository())
    configs = resolve_configurations(admitted, [c['name'] for c in admitted['configurations']])
    assert all(c['runtime_ready'] for c in configs)
    assert {c['smoke_status'] for c in configs if c['backend'] == 'local'} == {'passed'}
    assert {c['smoke_status'] for c in configs if c['backend'] == 'gemini'} == {'not_run'}
    with pytest.raises(RuntimeError, match='code commit differs'):
        admit_panel(panel, str(path), repository={'code_commit': 'd' * 40, 'worktree_clean': True})
    admission['gemini_smoke']['status'] = 'passed'
    with pytest.raises(ValueError, match='invalid Gemini smoke exception'):
        validate_admission(panel, admission)


@pytest.mark.parametrize('mutation', [
    lambda records: records.pop(),
    lambda records: records[0]['checks'].update(mechanical_runtime_gate=False),
    lambda records: records[0].update(code_commit='d' * 40),
])
def test_gemini_skip_still_requires_complete_matching_local_smoke(mutation):
    panel = load_model_panel(PANEL_PATH)
    records = [r for r in _valid_evidence(panel) if r['backend'] == 'local']
    mutation(records)
    admission = build_admission(panel, records, skip_gemini_smoke=True)
    with pytest.raises(ValueError, match='does not pass'):
        validate_admission(panel, admission)


def test_campaign_admit_cli_records_explicit_gemini_skip(tmp_path, monkeypatch):
    import analysis.evaluation_campaign as campaign
    import analysis.smoke_admission as smoke
    panel = load_model_panel(PANEL_PATH)
    records = [r for r in _valid_evidence(panel) if r['backend'] == 'local']
    cfg = {'campaign_dir': str(tmp_path), 'admission': str(tmp_path / 'admission.json'),
           'hf_home': str(tmp_path / 'cache')}
    monkeypatch.setattr(campaign, 'load_config', lambda path: cfg)
    monkeypatch.setattr(campaign, 'frozen_inputs', lambda *a, **kw: {})
    monkeypatch.setattr(campaign, 'configure_model_cache', lambda cfg: None)
    monkeypatch.setenv('CORIVA_API_USAGE_LOG', str(tmp_path / 'usage.jsonl'))
    monkeypatch.setattr(smoke, '_load_evidence', lambda root: records)
    monkeypatch.setattr(sys, 'argv', ['evaluation_campaign', 'task', 'admit', '--skip-gemini-smoke'])
    assert campaign.main() == 0
    result = json.loads((tmp_path / 'admission.json').read_text())
    validate_admission(panel, result)
    assert result['gemini_smoke']['status'] == 'not_run'
    with pytest.raises(ValueError, match='only to task admit'):
        campaign.run_task(cfg, 'infer', skip_gemini_smoke=True)


def test_admission_cannot_unlock_a_different_code_commit(tmp_path):
    panel = load_model_panel(PANEL_PATH)
    admission = build_admission(panel, _valid_evidence(panel))
    path = tmp_path / "admission.json"
    path.write_text(json.dumps(admission))
    with pytest.raises(RuntimeError, match="code commit differs"):
        admit_panel(
            panel, str(path),
            repository={"code_commit": "a" * 40, "worktree_clean": True})


def _make_smoke_scenario(tmp_path):
    scenario = tmp_path / SCENARIO
    scenario.mkdir()
    for name in (
            "ground_truth.json", "input_manifest.json", "observed_state.json",
            "observed_history.json", "query.json", "ontology.json"):
        (scenario / name).write_text(json.dumps({"asset": name}))
    frames = scenario / "frames"
    frames.mkdir()
    for index in range(8):
        (frames / f"{index:03d}.jpg").write_bytes(f"frame-{index}".encode())
    return scenario


def test_smoke_input_fingerprint_covers_all_eight_ordered_frames(tmp_path):
    scenario = _make_smoke_scenario(tmp_path)
    first, records = scenario_input_fingerprint(str(scenario))
    assert len(records) == 14
    assert [item["path"] for item in records[-8:]] == [
        f"frames/{index:03d}.jpg" for index in range(8)]
    (scenario / "frames" / "003.jpg").write_bytes(b"changed")
    second, _ = scenario_input_fingerprint(str(scenario))
    assert first != second


def _write_cell_output(base, panel, config, condition):
    scenario_out = base / SCENARIO
    scenario_out.mkdir(parents=True)
    raw = '{"predicted_branches": [], "predicted_counterfactuals": []}'
    final = raw
    expected_frames = 8 if condition in {"video", "video_history"} else 0
    frame_inputs = INPUT_FILES[-8:] if expected_frames else []
    manifest = {
        "protocol_version": panel["protocol_version"],
        "scorer_version": panel["scorer_version"],
        "model_panel_version": panel["panel_version"],
        "scorer_input_track": "primary",
        "execution_purpose": "smoke_admission",
        "configuration_name": config["name"],
        "model_id": config["model_id"],
        "model_revision": config.get("revision"),
        "backend": config["backend"],
        "adapter": config["adapter"],
        "inference_mode": config["inference_mode"],
        "enable_thinking": config.get("enable_thinking"),
        "condition": condition,
        "max_output_tokens": 65536,
        "decoding": "greedy",
        "temperature": 0.0,
        "generation_seed": None,
        "seed": 0,
        "frame_count": expected_frames,
        "frame_files": [os.path.basename(item["path"]) for item in frame_inputs],
        "frame_sha256": [item["sha256"] for item in frame_inputs],
        "raw_response_sha256": hashlib.sha256(raw.encode()).hexdigest(),
        "final_response_sha256": hashlib.sha256(final.encode()).hexdigest(),
        "prompt_sha256": "a" * 64,
        "thinking_trace_present": False,
    }
    metadata = {
        "adapter": config["adapter"],
        "max_output_tokens": 65536,
        "ordered_image_count": expected_frames,
        "output_tokens": 10,
        "latency_seconds": 1.0,
    }
    if config["backend"] == "local":
        metadata.update({
            "model_commit_hash": config["revision"],
            "requested_revision": config["revision"],
            "transformers_commit_hash": panel["runtime_environment"][
                "transformers_revision"],
            "peak_cuda_memory_bytes": 1234,
            "cuda_device_name": "Synthetic GPU",
            "torch_version": "2.x",
            "transformers_version": "5.x",
            "model_dtype": "torch.bfloat16",
        })
    else:
        metadata.update({
            "model_version": "gemini-provider-version",
            "api_version": config["api_version"],
            "api_endpoint": config["api_endpoint"],
            "access_date": "2026-08-18",
            "response_id": "response-1",
        })
    manifest["generation_metadata"] = metadata
    (scenario_out / "raw_response.txt").write_text(raw)
    (scenario_out / "final_response.txt").write_text(final)
    (scenario_out / "generation_metadata.json").write_text(json.dumps(metadata))
    (scenario_out / "run_manifest.json").write_text(json.dumps(manifest))
    (scenario_out / "l3_pre.json").write_text("{}")
    (scenario_out / "l4_pre.json").write_text("{}")


@pytest.mark.parametrize("configuration,condition", [
    ("qwen3-vl-8b-instruct", "video"),
    ("qwen3-vl-8b-instruct", "history_json"),
    ("qwen3.5-9b-direct", "video_history"),
    ("gemini-2.5-pro", "video"),
])
def test_cell_inspection_is_mechanical_not_schema_based(
        tmp_path, configuration, condition):
    panel = load_model_panel(PANEL_PATH)
    config = resolve_configurations(panel, [configuration])[0]
    _write_cell_output(tmp_path, panel, config, condition)
    evidence = inspect_cell_output(
        panel, config, condition, SCENARIO, str(tmp_path),
        INPUT_FINGERPRINT, _repository(),
        [{"scenario_id": SCENARIO, "errors": ["invalid schema"]}],
        INPUT_FILES)
    assert evidence["status"] == "passed", evidence["failed_checks"]
    assert evidence["observations"]["parse_failure_count"] == 1


def test_cell_inspection_rejects_a_different_ordered_frame_hash(tmp_path):
    panel = load_model_panel(PANEL_PATH)
    config = resolve_configurations(panel, ["qwen3-vl-8b-instruct"])[0]
    _write_cell_output(tmp_path, panel, config, "video")
    manifest_path = tmp_path / SCENARIO / "run_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["frame_sha256"][0] = "0" * 64
    manifest_path.write_text(json.dumps(manifest))
    evidence = inspect_cell_output(
        panel, config, "video", SCENARIO, str(tmp_path),
        INPUT_FINGERPRINT, _repository(), [], INPUT_FILES)
    assert evidence["status"] == "failed"
    assert "ordered_frame_hashes_match_input" in evidence["failed_checks"]


def test_admission_rejects_tampered_embedded_input_inventory():
    panel = load_model_panel(PANEL_PATH)
    records = _valid_evidence(panel)
    records[0]["input_files"][-1]["sha256"] = "0" * 64
    admission = build_admission(panel, records)
    assert admission["status"] == "failed"
    assert any("embedded input fingerprint mismatch" in failure
               for failure in admission["failures"])


@pytest.mark.skip(reason="needs the legacy run matrix, which is not part of this release")
def test_run_manifest_with_admission_resolves_all_matrix_cells_ready(
        tmp_path, monkeypatch):
    from analysis import smoke_admission
    from analysis.run_matrix import (_dry_run_command, expand_matrix,
                                     load_run_manifest)

    panel = load_model_panel(PANEL_PATH)
    admission = build_admission(panel, _valid_evidence(panel))
    path = tmp_path / "admission.json"
    path.write_text(json.dumps(admission))
    monkeypatch.setattr(smoke_admission, "repository_state", lambda *args: _repository())
    manifest = load_run_manifest("analysis/run_matrix.yaml", str(path))
    cells = expand_matrix(manifest)
    assert len(cells) == 15
    assert all(cell["runtime_ready"] is True for cell in cells)
    assert all(cell["smoke_admission_version"] == SMOKE_ADMISSION_VERSION
               for cell in cells)
    command = _dry_run_command(cells[0], manifest)
    assert "--configuration qwen3-vl-8b-instruct" in command
    assert f"--model-panel {manifest['model_panel_path']}" in command
    assert f"--smoke-admission {path.resolve()}" in command
    assert "--backend" not in command
    assert "--model " not in command
