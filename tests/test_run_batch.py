"""
Batch orchestrator (`simulation/run_batch.py`).

The orchestrator turns a directory of scenario configs into per-scenario
output dirs (telemetry + ground_truth.json), applying the QC gate and
resuming across restarts. CARLA itself is injected as `scenario_fn`; these
tests drive the full local pipeline (config → telemetry → GT → QC → files)
with the repo's real 2-scenario telemetry fixture — the Phase-2 "dry-run
end-to-end with mocked CARLA" gate.
"""

import json
import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from simulation.run_batch import run_batch  # noqa: E402
from simulation.generate import ScenarioCandidateExhausted  # noqa: E402
from simulation.scenario_config import load_scenario_configs  # noqa: E402


@pytest.fixture(scope="module")
def configs():
    return load_scenario_configs("scenarios")


@pytest.fixture(scope="module")
def raw_by_id():
    with open("coriva_bench_examples.json") as f:
        raw = json.load(f)
    return {entry["scenario"]: entry for entry in raw}


def _fake_scenario_fn(raw_by_id):
    calls = []

    def fn(config):
        calls.append(config["scenario_id"])
        return raw_by_id[config["scenario_id"]]

    fn.calls = calls
    return fn


def test_batch_produces_gt_files_and_log(tmp_path, configs, raw_by_id):
    fn = _fake_scenario_fn(raw_by_id)
    summary = run_batch(configs, fn, str(tmp_path))
    assert summary["accepted"] == 2 and summary["failed"] == 0
    for sub in ("scenario_1", "scenario_2"):
        gt_path = tmp_path / sub / "ground_truth.json"
        assert gt_path.exists()
        gt = json.loads(gt_path.read_text())
        assert gt["observed_state"]["event_type"] == "observed_state"
        assert len(gt["branches"]) == 3
        assert (tmp_path / sub / "scenario_manifest.json").exists()
        assert (tmp_path / sub / "scenario_config.json").exists()
    assert (tmp_path / "dataset_manifest.json").exists()
    log = [json.loads(line) for line in (tmp_path / "batch_log.jsonl").read_text().splitlines()]
    assert {e["scenario_id"] for e in log} == {c["scenario_id"] for c in configs}
    assert all(e["status"] == "accepted" for e in log)


def test_batch_preserves_runtime_placement_provenance(
        tmp_path, configs, raw_by_id):
    config = configs[0]
    placement = {
        "map": config.get("map"),
        "simulation_seed": config.get("seed"),
        "candidate_number": 3,
        "base_transform": {"x": 1.0, "y": 2.0, "z": 0.0,
                           "pitch": 0.0, "yaw": 90.0, "roll": 0.0},
        "context_lateral_offsets_m": {"park_r1": 5.5},
    }

    def fn(_config):
        raw = json.loads(json.dumps(raw_by_id[config["scenario_id"]]))
        raw["_placement"] = placement
        return raw

    summary = run_batch([config], fn, str(tmp_path))
    assert summary["accepted"] == 1
    manifest = json.loads(
        (tmp_path / config["output_subdir"] / "scenario_manifest.json").read_text())
    assert manifest["runtime_placement"] == placement


def test_batch_resumes_skipping_existing(tmp_path, configs, raw_by_id):
    fn1 = _fake_scenario_fn(raw_by_id)
    run_batch(configs, fn1, str(tmp_path))
    fn2 = _fake_scenario_fn(raw_by_id)
    summary = run_batch(configs, fn2, str(tmp_path))
    assert fn2.calls == [], "already-built scenarios must not re-run the simulator"
    assert summary["skipped"] == 2


def test_batch_records_qc_rejection_and_continues(tmp_path, configs, raw_by_id):
    def broken_fn(config):
        raw = json.loads(json.dumps(raw_by_id[config["scenario_id"]]))
        if config["scenario_id"] == "high_speed_following":
            raw["_placement"] = {"candidate_number": 4}
            raw["_route_evidence"] = {
                "source": "carla_map_waypoint_route",
                "route_coverage_m": 120.0,
                "route_required_m": 120.0,
                "sample_spacing_m": 2.0,
            }
            # amputate every branch to one step → QC must reject
            for name in raw["branches"]:
                raw["branches"][name] = raw["branches"][name][:1]
        return raw

    summary = run_batch(configs, broken_fn, str(tmp_path))
    assert summary["failed"] >= 1
    assert summary["accepted"] >= 1, "one bad scenario must not stop the batch"
    assert not (tmp_path / "scenario_1" / "ground_truth.json").exists()
    diagnostic = tmp_path / ".rejected" / "scenario_1" / "seed_0"
    assert (diagnostic / "scenario_config.json").is_file()
    assert (diagnostic / "raw_telemetry.json.gz").is_file()
    assert (diagnostic / "ground_truth.json").is_file()
    reasons = json.loads((diagnostic / "qc_reasons.json").read_text())
    assert reasons["scenario_id"] == "high_speed_following"
    assert reasons["status"] == "rejected_qc"
    assert reasons["route_evidence"] == {
        "route_source": "carla_map_waypoint_route",
        "route_mode": None,
        "route_id": None,
        "coverage_status": "complete",
        "backward_coverage_status": "unavailable",
        "route_coverage_m": 120.0,
        "route_required_m": 120.0,
        "route_backward_coverage_m": None,
        "route_backward_required_m": None,
        "route_sample_spacing_m": 2.0,
        "candidate_number": 4,
    }
    assert not any(diagnostic.rglob("*.png"))
    log = [json.loads(line) for line in
           (tmp_path / "batch_log.jsonl").read_text().splitlines()]
    rejected = next(row for row in log
                    if row["scenario_id"] == "high_speed_following")
    assert rejected["diagnostic_path"] == ".rejected/scenario_1/seed_0"


def test_batch_scenario_fn_errors_are_caught(tmp_path, configs, raw_by_id):
    def exploding_fn(config):
        if config["scenario_id"] == "pedestrian_crossing":
            raise RuntimeError("CARLA died")
        return raw_by_id[config["scenario_id"]]

    summary = run_batch(configs, exploding_fn, str(tmp_path))
    assert summary["failed"] == 1 and summary["accepted"] == 1
    log = [json.loads(line) for line in (tmp_path / "batch_log.jsonl").read_text().splitlines()]
    err = next(e for e in log if e["status"] == "error")
    assert "CARLA died" in err["reasons"][0]


def test_batch_records_candidate_exhaustion_as_resampleable_reject(
        tmp_path, configs):
    def exhausted_fn(_config):
        raise ScenarioCandidateExhausted(
            "deterministic waypoint candidates exhausted after "
            "24/24 preflight rejection(s)")

    summary = run_batch([configs[0]], exhausted_fn, str(tmp_path))

    assert summary == {"accepted": 0, "failed": 1, "skipped": 0}
    log = [json.loads(line) for line in
           (tmp_path / "batch_log.jsonl").read_text().splitlines()]
    assert log[-1]["status"] == "rejected_placement"
    assert "24/24 preflight rejection" in log[-1]["reasons"][0]


def test_failed_retry_removes_attempt_frames_and_publishes_nothing(tmp_path, configs):
    cfg = configs[0]

    def fn(config):
        attempt = tmp_path / config["output_subdir"] / "Factual_AEB"
        attempt.mkdir(parents=True)
        (attempt / "001000.png").write_bytes(b"partial-current-attempt")
        raise RuntimeError("synthetic retry failure")

    summary = run_batch([cfg], fn, str(tmp_path))
    assert summary == {"accepted": 0, "failed": 1, "skipped": 0}
    assert not (tmp_path / cfg["output_subdir"]).exists()
    assert not any((tmp_path / ".attempts").rglob("*.png"))


def test_postprocessing_failure_is_logged_and_attempt_is_removed(
        tmp_path, configs, raw_by_id, monkeypatch):
    cfg = configs[0]
    factual = next(name for name, branch in cfg["branches"].items()
                   if branch["is_factual"])

    def fn(config):
        _write_branch_pngs(str(tmp_path), config["output_subdir"], factual, 80)
        raw = json.loads(json.dumps(raw_by_id[config["scenario_id"]]))
        no_action = next((name for name, branch in config["branches"].items()
                          if branch["intervention_type"] == "do_ego_no_action"), factual)
        raw["_l4_steps"] = raw["branches"][no_action]
        raw["_l4_control_steps"] = raw["branches"][no_action]
        return raw

    monkeypatch.setattr("simulation.run_batch._overlay_carla_frames",
                        lambda *_args, **_kwargs: (_ for _ in ()).throw(
                            RuntimeError("overlay failed")))
    summary = run_batch([cfg], fn, str(tmp_path), track="primary")
    assert summary == {"accepted": 0, "failed": 1, "skipped": 0}
    assert not (tmp_path / cfg["output_subdir"]).exists()
    log = [json.loads(line) for line in
           (tmp_path / "batch_log.jsonl").read_text().splitlines()]
    assert log[-1]["status"] == "error"
    assert "overlay failed" in log[-1]["reasons"][0]


def test_accepted_attempt_replaces_stale_partial_directory(tmp_path, configs, raw_by_id):
    stale = tmp_path / configs[0]["output_subdir"]
    stale.mkdir(parents=True)
    (stale / "stale.png").write_bytes(b"old-attempt")
    summary = run_batch([configs[0]], _fake_scenario_fn(raw_by_id), str(tmp_path))
    assert summary["accepted"] == 1
    assert not (stale / "stale.png").exists()
    assert (stale / "ground_truth.json").exists()


def test_resume_rebuilds_legacy_gt_without_current_release_contract(
        tmp_path, configs, raw_by_id):
    cfg = configs[0]
    stale = tmp_path / cfg["output_subdir"]
    stale.mkdir(parents=True)
    (stale / "ground_truth.json").write_text("{}")

    fn = _fake_scenario_fn(raw_by_id)
    summary = run_batch([cfg], fn, str(tmp_path))

    assert summary == {"accepted": 1, "failed": 0, "skipped": 0}
    assert fn.calls == [cfg["scenario_id"]]
    assert (stale / "scenario_config.json").is_file()


# ── Observed frames for the VLM (pre-intervention only, no answer leak) ──
# The recorder writes each branch's PNGs into <scen>/<branch>/ (6-digit CARLA
# frame numbers, chronological when sorted). run_model reads scenario-level
# <scen>/frames/. run_batch must bridge the two: copy the FACTUAL branch's
# frames STRICTLY BEFORE intervention_time_sec into <scen>/frames/ — frames
# from t >= intervention would show the model the outcome it must predict.

def _write_branch_pngs(out_root, subdir, branch, n, start_frame=1000):
    from PIL import Image
    d = os.path.join(out_root, subdir, branch)
    os.makedirs(d, exist_ok=True)
    for i in range(n):
        Image.new("RGB", (64, 48), color=(i % 250, 0, 0)).save(
            os.path.join(d, f"{start_frame + i:06d}.png"))


def test_observed_frames_extracted_pre_intervention_only(tmp_path, configs, raw_by_id):
    cfg = configs[0]
    factual = next(n for n, b in cfg["branches"].items() if b["is_factual"])
    other = next(n for n, b in cfg["branches"].items() if not b["is_factual"])

    def fn(config):
        # mimic the recorder side effect: 80 frames (8 s @ 10 fps) per branch
        _write_branch_pngs(str(tmp_path), config["output_subdir"], factual, 80)
        _write_branch_pngs(str(tmp_path), config["output_subdir"], other, 80,
                           start_frame=9000)
        return raw_by_id[config["scenario_id"]]

    run_batch([cfg], fn, str(tmp_path))
    frames_dir = tmp_path / cfg["output_subdir"] / "frames"
    assert frames_dir.is_dir()
    frames = sorted(os.listdir(frames_dir))
    expected = int(float(cfg["intervention_time_sec"]) * 10)  # 10 fps
    assert len(frames) == expected
    # renamed 000.png.. so run_model's glob+sort stays chronological
    assert frames[0] == "000.png" and frames[-1] == f"{expected - 1:03d}.png"
    # content must come from the FACTUAL branch's earliest frames (no 9000-range)
    first = (frames_dir / "000.png").read_bytes()
    src = tmp_path / cfg["output_subdir"] / factual / "001000.png"
    assert first == src.read_bytes()


def test_protocol_batch_writes_exact_eight_frames_and_manifest(tmp_path, configs, raw_by_id):
    cfg = configs[0]
    factual = next(n for n, b in cfg["branches"].items() if b["is_factual"])

    def fn(config):
        _write_branch_pngs(str(tmp_path), config["output_subdir"], factual, 80)
        raw = json.loads(json.dumps(raw_by_id[config["scenario_id"]]))
        no_action = next((name for name, branch in config["branches"].items()
                          if branch["intervention_type"] == "do_ego_no_action"), factual)
        raw["_l4_steps"] = raw["branches"][no_action]
        raw["_l4_control_steps"] = raw["branches"][no_action]
        return raw

    summary = run_batch([cfg], fn, str(tmp_path), track="primary")
    assert summary == {"accepted": 1, "failed": 0, "skipped": 0}
    scen_dir = tmp_path / cfg["output_subdir"]
    assert len(list((scen_dir / "frames").glob("*.png"))) == 8
    manifest = json.loads((scen_dir / "input_manifest.json").read_text())
    assert manifest["observed_frame_count"] == 8
    assert manifest["track"] == "primary"
    assert (scen_dir / "observed_state.json").is_file()
    assert (scen_dir / "observed_history.json").is_file()
    assert len(list((scen_dir / "raw_frames").glob("*.png"))) == 8
    gt = json.loads((scen_dir / "ground_truth.json").read_text())
    scenario_manifest = json.loads((scen_dir / "scenario_manifest.json").read_text())
    scenario_config = json.loads((scen_dir / "scenario_config.json").read_text())
    assert scenario_config == cfg
    assert scenario_manifest["protocol"] == gt["protocol"]
    assert scenario_manifest["config_sha256"]
    assert gt["protocol"]["l4_reference_source"] == "carla_fixed_policy_rollout"
    assert gt["protocol"]["primary_measurement"] == "ground_truth_free_l3_causal_coherence"
    assert gt["protocol"]["reference_evaluation_role"] == "secondary_declared_reference_agreement"
    assert gt["protocol"]["reference_semantics"] == "exact_deterministic_simulator_agreement"
    assert all((branch.get("intervention") or {}).get("target_actor_id") == "ego"
               for branch in gt["branches"])
    assert gt["l4_queries"][0]["counterfactual_chain"]["intervention"]["target_actor_id"] == "npc_1"


def test_no_branch_frames_means_no_frames_dir_and_no_crash(tmp_path, configs, raw_by_id):
    fn = _fake_scenario_fn(raw_by_id)
    summary = run_batch(configs, fn, str(tmp_path))
    assert summary["accepted"] == 2
    assert not (tmp_path / configs[0]["output_subdir"] / "frames").exists()


def test_batch_records_unbuildable_reference_as_reject(tmp_path, configs, raw_by_id):
    """A collision whose partner cannot be resolved is a reference defect,
    not a simulator error: logged as rejected_reference, batch continues."""
    def fn(config):
        raw = json.loads(json.dumps(raw_by_id[config["scenario_id"]]))
        if config["scenario_id"] == "high_speed_following":
            for steps in raw["branches"].values():
                for step in steps:
                    if step.get("collision"):
                        step["collision_partners"] = []
        return raw

    summary = run_batch(configs, fn, str(tmp_path))
    assert summary["failed"] == 1 and summary["accepted"] == 1
    log = [json.loads(line) for line in (tmp_path / "batch_log.jsonl").read_text().splitlines()]
    rejected = next(e for e in log if e["scenario_id"] == "high_speed_following")
    assert rejected["status"] == "rejected_reference"
    assert "partner" in rejected["reasons"][0]


def test_protocol_run_rejects_a_scene_whose_swerve_steers_into_the_hazard(
        tmp_path, raw_by_id, monkeypatch):
    """On a curved route the realised T0 offset can land on the side opposite
    the template offset that chose the lateral action. Such a scene is
    rejected and resampled, not published."""
    import simulation.run_batch as rb
    from simulation.scenario_config import load_scenario_configs
    config = load_scenario_configs("scenarios")[1]
    conflicting_gt = {
        "scenario_id": config["scenario_id"],
        "observed_state": {"actors": [{"id": "ego", "y": 0.0}, {"id": "npc_1", "y": 5.4}]},
        "branches": [
            {"branch_id": "F", "is_factual": True,
             "intervention": {"intervention_type": "do_ego_comfortable_brake"}, "chain": []},
            {"branch_id": "S", "is_factual": False,
             "intervention": {"intervention_type": "do_ego_swerve_right"}, "chain": []}],
        "l4_queries": [],
    }
    monkeypatch.setattr(rb, "build_scenario_gt", lambda *a, **k: conflicting_gt)
    summary = run_batch([config], _fake_scenario_fn(raw_by_id), str(tmp_path), track="primary")
    assert summary["accepted"] == 0
    log = [json.loads(line) for line in (tmp_path / "batch_log.jsonl").read_text().splitlines()]
    assert log[-1]["status"] == "rejected_reference"
    assert "steers toward critical actor npc_1" in str(log[-1]["reasons"])
