"""Chunk/restart supervisor tests; no CARLA installation is required."""

import io
import json
import signal
from types import SimpleNamespace

import pytest

from simulation import run_carla_supervised as supervisor
from scenarios.generate_configs import generate_configs


def _args(tmp_path, **overrides):
    values = {
        "scenarios_dir": "scenarios/generated_seed0",
        "out_root": str(tmp_path / "output"),
        "host": "localhost",
        "port": 2000,
        "track": "primary",
        "limit": 40,
        "chunk_size": 20,
        "max_server_restarts": 3,
        "max_batch_restarts": 2,
        "max_scenario_error_retries": 2,
        "max_attempt_seed": 9,
        "poll_interval": 0.0,
        "startup_timeout": 1.0,
        "shutdown_timeout": 0.01,
        "carla_root": "/workspace/carla_0.9.15",
        "carla_user": "carlauser",
        "vulkan_icd": "/etc/vulkan/icd.d/nvidia_icd.json",
        "python": ".venv/bin/python",
        "supervisor_state_dir": str(tmp_path / "runtime"),
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def test_chunk_targets_end_at_requested_limit():
    assert supervisor._chunk_targets(45, 20) == [20, 40, 45]


def test_server_command_matches_vast_offscreen_runtime(tmp_path):
    command = supervisor._server_command(_args(tmp_path))
    assert command == [
        "sudo", "-u", "carlauser", "env",
        "VK_ICD_FILENAMES=/etc/vulkan/icd.d/nvidia_icd.json",
        "/workspace/carla_0.9.15/CarlaUE4.sh",
        "-RenderOffScreen", "-nosound", "-carla-rpc-port=2000",
    ]


def test_server_command_binds_the_supervisor_port_for_shared_machines(tmp_path):
    """Four supervisors on one machine (one shard, out_root and port each)
    must each start their own server on their own port; recovery must then
    attribute a server to the supervisor that owns it."""
    for port in (2010, 2020, 2030):
        command = supervisor._server_command(_args(tmp_path, port=port))
        assert command[-1] == f"-carla-rpc-port={port}"
        assert supervisor._carla_process_port(command) == port
    assert supervisor._carla_process_port(
        supervisor._server_command(_args(tmp_path, port=2010))) != 2000


def test_carla_process_port_supports_default_and_explicit_rpc_port():
    assert supervisor._carla_process_port([
        "/workspace/carla/CarlaUE4-Linux-Shipping", "CarlaUE4",
        "-RenderOffScreen",
    ]) == 2000
    assert supervisor._carla_process_port([
        "CarlaUE4-Linux-Shipping", "-carla-rpc-port=2100",
    ]) == 2100
    assert supervisor._carla_process_port([
        "CarlaUE4-Linux-Shipping", "--carla-rpc-port", "2200",
    ]) == 2200


def test_command_port_reads_the_port_inside_a_shell_wrapper():
    """tmux runs the supervisor through `sh -c "<whole command>"`; the port
    must come from inside that string, not the 2000 default, or shard 0's
    recovery sees every other shard's wrapper as a client on its port."""
    wrapper = ["bash", "-c", ".venv/bin/python -u -m simulation.run_carla_supervised "
               "--scenarios-dir scenarios/shard_1 --out-root x --port 2010 --chunk-size 20"]
    assert supervisor._command_port(wrapper) == 2010
    assert supervisor._command_port(["sh", "-c", "python -m simulation.run_batch --port=2020 --limit 20"]) == 2020
    assert supervisor._command_port(["sh", "-c", "python -m simulation.run_batch --limit 20"]) == 2000
    assert supervisor._command_port(["python", "-m", "simulation.run_batch", "--port", "2030"]) == 2030


def test_ancestor_pids_contains_self_and_parent():
    ancestors = supervisor._ancestor_pids()
    import os
    assert os.getpid() in ancestors
    if os.path.isdir("/proc"):
        assert os.getppid() in ancestors


def test_completed_prefix_skips_already_published_chunks(tmp_path):
    configs = [
        {"output_subdir": f"scenario_{index}"} for index in range(1, 5)
    ]
    required = supervisor._PUBLISHED_PROTOCOL_FILES
    for index in (1, 2):
        directory = tmp_path / f"scenario_{index}"
        directory.mkdir()
        for name in required:
            (directory / name).write_text("{}")

    assert supervisor._completed_prefix(configs, str(tmp_path)) == 2


def test_missing_configs_preserves_published_slots_after_a_hole(tmp_path):
    configs = [
        {"slot_id": index, "output_subdir": f"scenario_{index}"}
        for index in range(1, 4)
    ]
    for index in (1, 3):
        directory = tmp_path / f"scenario_{index}"
        directory.mkdir()
        for name in supervisor._PUBLISHED_PROTOCOL_FILES:
            (directory / name).write_text("{}", encoding="utf-8")

    missing = supervisor._missing_configs(configs, str(tmp_path))

    assert [config["slot_id"] for config in missing] == [2]


def test_attempt_planner_resamples_deterministic_scenario_rejects():
    config = generate_configs(1, seed=0)[0]
    for status in ("rejected_admission", "rejected_placement", "rejected_qc",
                   "rejected_reference"):
        history = [{
            "slot_id": 1,
            "attempt_seed": 0,
            "status": status,
        }]
        assert supervisor._select_attempt_seed(config, history, 9, 2) == 1


def test_attempt_planner_migrates_legacy_candidate_exhaustion_errors():
    config = generate_configs(237, seed=0)[236]
    legacy_error = {
        "scenario_id": "pedestrian_cross_left_to_right_0236",
        "slot_id": 237,
        "attempt_seed": 0,
        "status": "error",
        "reasons": [
            "RuntimeError: deterministic waypoint candidates exhausted "
            "after 24/24 preflight rejection(s)"
        ],
    }

    assert supervisor._select_attempt_seed(
        config, [legacy_error, legacy_error, legacy_error], 9, 2) == 1


def test_attempt_planner_retries_runtime_error_without_changing_seed():
    config = generate_configs(1, seed=0)[0]
    error = {"slot_id": 1, "attempt_seed": 0, "status": "error"}
    assert supervisor._select_attempt_seed(config, [error], 9, 2) == 0
    assert supervisor._select_attempt_seed(config, [error, error], 9, 2) == 0
    with pytest.raises(supervisor._AttemptPlanningError, match="runtime errors"):
        supervisor._select_attempt_seed(config, [error, error, error], 9, 2)


def test_attempt_planner_fails_closed_for_protocol_reject():
    config = generate_configs(1, seed=0)[0]
    history = [{
        "slot_id": 1,
        "attempt_seed": 0,
        "status": "rejected_protocol",
    }]
    with pytest.raises(supervisor._AttemptPlanningError,
                       match="non-resampleable"):
        supervisor._select_attempt_seed(config, history, 9, 2)


def test_attempt_planner_stops_after_last_resample_seed():
    config = generate_configs(1, seed=0)[0]
    history = [{
        "slot_id": 1,
        "attempt_seed": 9,
        "status": "rejected_admission",
    }]
    with pytest.raises(supervisor._AttemptPlanningError,
                       match="exhausted attempt seeds through 9"):
        supervisor._select_attempt_seed(config, history, 9, 2)


def test_generated_retry_preserves_slot_invariants():
    config = generate_configs(1, seed=0)[0]
    retry = supervisor._generated_retry_config(config, 1, {}, 1)

    assert retry["attempt_seed"] == 1
    for field in (
            "scenario_index", "slot_id", "output_subdir", "template_id",
            "scenario_id", "map", "weather"):
        assert retry[field] == config[field]


def test_supervisor_retries_same_chunk_after_server_exit_139(
        tmp_path, monkeypatch):
    starts = []
    batch_targets = []
    monitor_results = iter([(None, 139), (0, None), (0, None)])

    class Process:
        def __init__(self, pid):
            self.pid = pid

        def poll(self):
            return None

    def start_server(_args, generation):
        starts.append(generation)
        return Process(100 + generation), None

    def start_batch(args, target):
        batch_targets.append(target)
        return Process(200 + len(batch_targets))

    monkeypatch.setattr(supervisor, "_port_is_open", lambda *_args: False)
    monkeypatch.setattr(supervisor, "_start_server", start_server)
    monkeypatch.setattr(supervisor, "_wait_for_ready", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(supervisor, "_start_batch", start_batch)
    monkeypatch.setattr(supervisor, "_monitor_batch",
                        lambda *_args, **_kwargs: next(monitor_results))
    monkeypatch.setattr(supervisor, "_stop_process", lambda *_args, **_kwargs: None)

    rc = supervisor.run_supervised(_args(tmp_path), total_configs=40)

    assert rc == 0
    assert batch_targets == [20, 20, 40]
    assert starts == [1, 2, 3]


def test_supervisor_stops_on_batch_failure_while_server_is_alive(
        tmp_path, monkeypatch):
    monkeypatch.setattr(supervisor, "_port_is_open", lambda *_args: False)
    monkeypatch.setattr(
        supervisor, "_start_server",
        lambda _args, generation: (SimpleNamespace(pid=generation), None))
    monkeypatch.setattr(supervisor, "_wait_for_ready", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(
        supervisor, "_start_batch", lambda *_args, **_kwargs: SimpleNamespace(pid=2))
    monkeypatch.setattr(
        supervisor, "_monitor_batch", lambda *_args, **_kwargs: (1, None))
    monkeypatch.setattr(supervisor, "_stop_process", lambda *_args, **_kwargs: None)

    assert supervisor.run_supervised(_args(tmp_path), total_configs=40) == 1


def test_supervisor_automatically_resamples_only_missing_slot(
        tmp_path, monkeypatch):
    args = _args(tmp_path, limit=1, chunk_size=1)
    base = generate_configs(1, seed=0)[0]
    real_loader = supervisor.load_scenario_configs
    attempted_seeds = []
    monitor_results = iter([(1, None), (0, None)])

    class Process:
        def __init__(self, pid):
            self.pid = pid

        def poll(self):
            return None

    def start_batch(_args, _limit, scenarios_dir=None):
        plan = real_loader(scenarios_dir)
        assert len(plan) == 1
        config = plan[0]
        seed = int(config["attempt_seed"])
        attempted_seeds.append(seed)
        out_root = tmp_path / "output"
        out_root.mkdir(exist_ok=True)
        with (out_root / "batch_log.jsonl").open("a", encoding="utf-8") as stream:
            stream.write(json.dumps({
                "scenario_id": config["scenario_id"],
                "slot_id": config["slot_id"],
                "attempt_seed": seed,
                "status": "rejected_admission" if seed == 0 else "accepted",
            }) + "\n")
        if seed == 1:
            scenario_dir = out_root / config["output_subdir"]
            scenario_dir.mkdir()
            for name in supervisor._PUBLISHED_PROTOCOL_FILES:
                (scenario_dir / name).write_text("{}", encoding="utf-8")
        return Process(200 + len(attempted_seeds))

    monkeypatch.setattr(supervisor, "load_scenario_configs", lambda _path: [base])
    monkeypatch.setattr(supervisor, "_port_is_open", lambda *_args: False)
    monkeypatch.setattr(
        supervisor, "_start_server",
        lambda _args, generation: (Process(100 + generation), None))
    monkeypatch.setattr(supervisor, "_wait_for_ready", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(supervisor, "_start_batch", start_batch)
    monkeypatch.setattr(
        supervisor, "_monitor_batch",
        lambda *_args, **_kwargs: next(monitor_results))
    monkeypatch.setattr(supervisor, "_stop_process", lambda *_args, **_kwargs: None)

    assert supervisor.run_supervised(args) == 0
    assert attempted_seeds == [0, 1]


def test_recover_stale_server_only_kills_verified_carla_listener(
        tmp_path, monkeypatch):
    args = _args(tmp_path)
    signals = []

    monkeypatch.setattr(supervisor, "_active_carla_client_pids", lambda _port: [])
    monkeypatch.setattr(supervisor, "_listener_pids", lambda _port: {321})
    monkeypatch.setattr(
        supervisor, "_process_matches_carla",
        lambda pid, _args: pid == 321)
    monkeypatch.setattr(
        supervisor, "_signal_processes",
        lambda pids, signum: signals.append((pids, signum)))
    monkeypatch.setattr(
        supervisor, "_wait_for_port_closed",
        lambda _host, _port, _timeout: True)

    assert supervisor._recover_stale_server(args) is True
    assert signals == [({321}, signal.SIGTERM)]


def test_recover_stale_server_falls_back_when_vast_hides_listener_fds(
        tmp_path, monkeypatch):
    args = _args(tmp_path)
    signals = []
    monkeypatch.setattr(supervisor, "_active_carla_client_pids", lambda _port: [])
    monkeypatch.setattr(supervisor, "_listener_pids", lambda _port: set())
    monkeypatch.setattr(
        supervisor, "_listening_socket_inodes",
        lambda _port: {"242971587"})
    monkeypatch.setattr(
        supervisor, "_matching_carla_process_pids", lambda _args: {20298})
    monkeypatch.setattr(
        supervisor, "_signal_processes",
        lambda pids, signum: signals.append((pids, signum)))
    monkeypatch.setattr(
        supervisor, "_wait_for_port_closed",
        lambda _host, _port, _timeout: True)

    assert supervisor._recover_stale_server(args) is True
    assert signals == [({20298}, signal.SIGTERM)]


def test_vast_fallback_refuses_ambiguous_carla_processes(tmp_path, monkeypatch):
    args = _args(tmp_path)
    monkeypatch.setattr(supervisor, "_active_carla_client_pids", lambda _port: [])
    monkeypatch.setattr(supervisor, "_listener_pids", lambda _port: set())
    monkeypatch.setattr(
        supervisor, "_listening_socket_inodes", lambda _port: {"123"})
    monkeypatch.setattr(
        supervisor, "_matching_carla_process_pids",
        lambda _args: {20298, 20301})
    monkeypatch.setattr(
        supervisor, "_signal_processes",
        lambda *_args: pytest.fail("ambiguous processes must not be killed"))

    assert supervisor._recover_stale_server(args) is False


def test_recover_stale_server_refuses_unknown_listener(tmp_path, monkeypatch):
    args = _args(tmp_path)
    monkeypatch.setattr(supervisor, "_active_carla_client_pids", lambda _port: [])
    monkeypatch.setattr(supervisor, "_listener_pids", lambda _port: {321})
    monkeypatch.setattr(
        supervisor, "_process_matches_carla", lambda _pid, _args: False)
    monkeypatch.setattr(
        supervisor, "_signal_processes",
        lambda *_args: pytest.fail("unknown listener must not be killed"))

    assert supervisor._recover_stale_server(args) is False


def test_recover_stale_server_refuses_when_legacy_batch_is_active(
        tmp_path, monkeypatch):
    args = _args(tmp_path)
    monkeypatch.setattr(
        supervisor, "_active_carla_client_pids", lambda _port: [111, 222])
    monkeypatch.setattr(
        supervisor, "_listener_pids",
        lambda _port: pytest.fail("must stop before inspecting listener"))

    assert supervisor._recover_stale_server(args) is False


def test_supervisor_recovers_stale_port_before_starting(tmp_path, monkeypatch):
    args = _args(tmp_path)
    recovered = []
    monkeypatch.setattr(supervisor, "_port_is_open", lambda *_args: True)
    monkeypatch.setattr(
        supervisor, "_recover_stale_server",
        lambda _args: recovered.append((_args.host, _args.port)) or True)

    assert supervisor.run_supervised(args, total_configs=0) == 0
    assert recovered == [("localhost", 2000)]


def test_stop_process_signals_group_after_launcher_already_exited(monkeypatch):
    signals = []
    process = SimpleNamespace(pid=777, poll=lambda: 139)
    monkeypatch.setattr(
        supervisor.os, "killpg",
        lambda pid, signum: signals.append((pid, signum)))

    supervisor._stop_process(process)

    assert signals == [(777, signal.SIGTERM)]


def test_second_supervisor_is_rejected_by_port_lease(tmp_path):
    args = _args(tmp_path)
    lease = supervisor._acquire_lease(args)
    try:
        assert supervisor.run_supervised(args, total_configs=0) == 2
    finally:
        supervisor._release_lease(lease)


def test_keyboard_interrupt_cleans_batch_server_state_and_port(
        tmp_path, monkeypatch):
    args = _args(tmp_path, limit=1, chunk_size=1)
    stopped = []
    stream = io.StringIO()

    class Process:
        def __init__(self, pid):
            self.pid = pid

        def poll(self):
            return None

    server = Process(101)
    batch = Process(202)
    monkeypatch.setattr(supervisor, "_port_is_open", lambda *_args: False)
    monkeypatch.setattr(
        supervisor, "_start_server", lambda *_args: (server, stream))
    monkeypatch.setattr(supervisor, "_wait_for_ready", lambda *_args: None)
    monkeypatch.setattr(supervisor, "_start_batch", lambda *_args: batch)
    monkeypatch.setattr(
        supervisor, "_monitor_batch",
        lambda *_args: (_ for _ in ()).throw(KeyboardInterrupt()))
    monkeypatch.setattr(
        supervisor, "_stop_process",
        lambda process, *_args, **_kwargs: stopped.append(process.pid)
        if process is not None else None)

    with pytest.raises(KeyboardInterrupt):
        supervisor.run_supervised(args, total_configs=1)

    assert stopped == [batch.pid, server.pid]
    assert stream.closed
    _, state_path = supervisor._runtime_paths(args)
    assert not supervisor.os.path.exists(state_path)


def test_attempt_plan_collects_unplannable_slots_instead_of_stopping():
    """One slot that exhausted its seeds must not stop the other slots from
    being planned; without a collector the planner still raises."""
    configs = generate_configs(2, seed=0)
    history = [{"slot_id": 1, "attempt_seed": 9, "status": "rejected_admission"}]
    skipped = []
    plan = supervisor._build_attempt_plan(configs, history, 9, 2, {}, 2, skipped=skipped)
    assert [config["slot_id"] for config in plan] == [2]
    assert [config["slot_id"] for config, _exc in skipped] == [1]
    assert "exhausted attempt seeds through 9" in str(skipped[0][1])
    with pytest.raises(supervisor._AttemptPlanningError):
        supervisor._build_attempt_plan(configs, history, 9, 2, {}, 2)


def test_supervisor_skips_exhausted_slot_builds_the_rest_and_exits_nonzero(
        tmp_path, monkeypatch):
    args = _args(tmp_path, limit=2, chunk_size=2)
    configs = generate_configs(2, seed=0)
    real_loader = supervisor.load_scenario_configs
    attempted = []
    out_root = tmp_path / "output"
    out_root.mkdir()
    with (out_root / "batch_log.jsonl").open("w", encoding="utf-8") as stream:
        stream.write(json.dumps({
            "scenario_id": configs[0]["scenario_id"], "slot_id": 1,
            "attempt_seed": 9, "status": "rejected_placement",
        }) + "\n")

    class Process:
        def __init__(self, pid):
            self.pid = pid

        def poll(self):
            return None

    def start_batch(_args, _limit, scenarios_dir=None):
        plan = real_loader(scenarios_dir)
        attempted.extend((config["slot_id"], int(config["attempt_seed"])) for config in plan)
        for config in plan:
            with (out_root / "batch_log.jsonl").open("a", encoding="utf-8") as stream:
                stream.write(json.dumps({
                    "scenario_id": config["scenario_id"], "slot_id": config["slot_id"],
                    "attempt_seed": config["attempt_seed"], "status": "accepted",
                }) + "\n")
            scenario_dir = out_root / config["output_subdir"]
            scenario_dir.mkdir()
            for name in supervisor._PUBLISHED_PROTOCOL_FILES:
                (scenario_dir / name).write_text("{}", encoding="utf-8")
        return Process(200)

    monkeypatch.setattr(supervisor, "load_scenario_configs", lambda _path: configs)
    monkeypatch.setattr(supervisor, "_port_is_open", lambda *_args: False)
    monkeypatch.setattr(
        supervisor, "_start_server",
        lambda _args, generation: (Process(100 + generation), None))
    monkeypatch.setattr(supervisor, "_wait_for_ready", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(supervisor, "_start_batch", start_batch)
    monkeypatch.setattr(supervisor, "_monitor_batch", lambda *_args, **_kwargs: (0, None))
    monkeypatch.setattr(supervisor, "_stop_process", lambda *_args, **_kwargs: None)

    assert supervisor.run_supervised(args) == 1
    assert attempted == [(2, 0)]  # slot 1 skipped, slot 2 built at its first seed
    assert (out_root / "scenario_2").is_dir() and not (out_root / "scenario_1").exists()
    events = [json.loads(line) for line in
              (out_root / ".carla_supervisor" / "supervisor_log.jsonl").read_text().splitlines()]
    exhausted = [event for event in events if event["event"] == "attempts_exhausted"]
    assert [event["slot_id"] for event in exhausted] == [1]
    summary = [event for event in events if event["event"] == "slots_exhausted"]
    assert len(summary) == 1 and summary[0]["slots"][0]["slot"] == "slot_id=1"


def test_more_attempts_come_from_raising_max_attempt_seed_not_from_another_config_seed():
    """After seeds 0-9 are spent, a config directory generated with another
    seed does not grant a retry (the next seed is chosen from history and
    capped); raising --max-attempt-seed continues at seed 10 with the same
    slot invariants."""
    config = generate_configs(1, seed=1)[0]
    history = [{"slot_id": 1, "attempt_seed": seed, "status": "rejected_placement"}
               for seed in range(10)]
    skipped = []
    assert supervisor._build_attempt_plan([config], history, 9, 2, {}, 1, skipped=skipped) == []
    assert len(skipped) == 1
    plan = supervisor._build_attempt_plan([config], history, 19, 2, {}, 1)
    assert [c["attempt_seed"] for c in plan] == [10]
    assert (plan[0]["slot_id"], plan[0]["map"], plan[0]["template_id"]) == (
        config["slot_id"], config["map"], config["template_id"])
