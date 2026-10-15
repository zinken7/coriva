"""Run resumable CARLA batches in bounded chunks with owned server restarts.

The supervisor starts CARLA, runs ``simulation.run_batch`` on the unpublished
slots in a cumulative chunk, then deliberately restarts UE4 before the next
work plan. If UE4 exits (including Linux exit 139), the batch child is stopped
and the same plan is retried. A per-port lease prevents concurrent supervisors.
At startup a stale listener is recovered only when its process identity matches
the requested CARLA root and user; unknown listeners and servers with active
clients are never killed. Placement/preflight, admission and QC rejects are
automatically regenerated at the next deterministic attempt seed, while runtime
errors retry the same seed within a bounded budget. Published scenarios are
never overwritten by retry planning.
"""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
import pwd
import re
import signal
import socket
import subprocess
import sys
import tempfile
import time
from collections import Counter
from typing import Any, Dict, IO, List, Optional, Set, Tuple

from simulation.scenario_config import load_scenario_configs

_PUBLISHED_PROTOCOL_FILES = (
    "ground_truth.json", "raw_telemetry.json", "scenario_manifest.json",
    "scenario_config.json", "observed_state.json", "observed_history.json",
    "query.json", "ontology.json", "input_manifest.json",
)
_RESAMPLE_STATUSES = frozenset({
    "rejected_admission", "rejected_placement", "rejected_qc",
    # The rollout ran but cannot be a valid reference (e.g. an in-horizon
    # collision with an actor outside the observed root): deterministic
    # for the sample, so resample with the next seed.
    "rejected_reference",
})
_ERROR_STATUS = "error"
_LEGACY_PLACEMENT_ERROR_FRAGMENTS = (
    "deterministic waypoint candidates exhausted after",
    "deterministic preflight candidates are exhausted",
    "unable to spawn complete cast at",
)


class _AttemptPlanningError(RuntimeError):
    """A missing slot cannot be retried safely within the configured budget."""


class _SupervisorLeaseBusy(RuntimeError):
    """The requested host/port already has a live supervisor lease."""


class _TerminationRequested(BaseException):
    def __init__(self, signum: int):
        super().__init__(signum)
        self.signum = signum


def _raise_termination(signum: int, _frame: Any) -> None:
    raise _TerminationRequested(signum)


def _runtime_paths(args: argparse.Namespace) -> Tuple[str, str]:
    state_dir = getattr(args, "supervisor_state_dir", tempfile.gettempdir())
    os.makedirs(state_dir, exist_ok=True)
    identity = f"{args.host}:{args.port}".encode("utf-8")
    key = hashlib.sha256(identity).hexdigest()[:12]
    stem = os.path.join(state_dir, f"coriva_carla_{key}")
    return f"{stem}.lock", f"{stem}.owner.json"


def _acquire_lease(args: argparse.Namespace) -> IO[str]:
    lock_path, _ = _runtime_paths(args)
    stream = open(lock_path, "a+", encoding="utf-8")
    try:
        fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError as exc:
        stream.close()
        raise _SupervisorLeaseBusy(
            f"another CARLA supervisor already owns {args.host}:{args.port}"
        ) from exc
    stream.seek(0)
    stream.truncate()
    stream.write(f"supervisor_pid={os.getpid()}\n")
    stream.flush()
    return stream


def _release_lease(stream: Optional[IO[str]]) -> None:
    if stream is None or stream.closed:
        return
    try:
        fcntl.flock(stream.fileno(), fcntl.LOCK_UN)
    finally:
        stream.close()


def _write_owner_state(
        args: argparse.Namespace, server: subprocess.Popen,
        generation: int, target: int) -> None:
    _, state_path = _runtime_paths(args)
    state = {
        "version": 1,
        "supervisor_pid": os.getpid(),
        "server_pid": server.pid,
        # start_new_session=True makes the launcher's PID its process-group ID.
        "server_pgid": server.pid,
        "host": args.host,
        "port": args.port,
        "carla_root": os.path.realpath(args.carla_root),
        "carla_user": args.carla_user,
        "out_root": os.path.realpath(args.out_root),
        "generation": generation,
        "target_limit": target,
        "timestamp": round(time.time(), 3),
    }
    temporary = f"{state_path}.{os.getpid()}.tmp"
    with open(temporary, "w", encoding="utf-8") as stream:
        json.dump(state, stream, sort_keys=True)
        stream.write("\n")
    os.replace(temporary, state_path)


def _clear_owner_state(args: argparse.Namespace) -> None:
    _, state_path = _runtime_paths(args)
    try:
        os.unlink(state_path)
    except FileNotFoundError:
        pass


def _chunk_targets(total: int, chunk_size: int) -> List[int]:
    if total <= 0:
        return []
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    targets = list(range(chunk_size, total, chunk_size))
    targets.append(total)
    return targets


def _server_command(args: argparse.Namespace) -> List[str]:
    executable = os.path.join(args.carla_root, "CarlaUE4.sh")
    command: List[str] = []
    if args.carla_user:
        command.extend(["sudo", "-u", args.carla_user])
    command.append("env")
    if args.vulkan_icd:
        command.append(f"VK_ICD_FILENAMES={args.vulkan_icd}")
    # Bind the server to this supervisor's RPC port so several supervisors can
    # share one machine (one shard, one out_root and one port each). CARLA
    # also takes port+1 (streaming) and port+2 (secondary server), so
    # neighbouring supervisors must be spaced at least three ports apart.
    # The port in the command line is also what stale-server recovery reads
    # (`_carla_process_port`) to tell this server from a neighbour's.
    command.extend([executable, "-RenderOffScreen", "-nosound",
                    f"-carla-rpc-port={int(args.port)}"])
    return command


def _completed_prefix(configs: List[dict], out_root: str) -> int:
    """Count the consecutive config prefix already fully published."""
    completed = 0
    for config in configs:
        scenario_dir = os.path.join(out_root, config["output_subdir"])
        if not all(os.path.isfile(os.path.join(scenario_dir, name))
                   for name in _PUBLISHED_PROTOCOL_FILES):
            break
        completed += 1
    return completed


def _scenario_is_published(config: Dict[str, Any], out_root: str) -> bool:
    scenario_dir = os.path.join(out_root, config["output_subdir"])
    return all(os.path.isfile(os.path.join(scenario_dir, name))
               for name in _PUBLISHED_PROTOCOL_FILES)


def _missing_configs(configs: List[Dict[str, Any]],
                     out_root: str) -> List[Dict[str, Any]]:
    return [config for config in configs
            if not _scenario_is_published(config, out_root)]


def _batch_history(out_root: str) -> List[Dict[str, Any]]:
    path = os.path.join(out_root, "batch_log.jsonl")
    try:
        with open(path, encoding="utf-8") as stream:
            lines = stream.readlines()
    except FileNotFoundError:
        return []
    records: List[Dict[str, Any]] = []
    for line in lines:
        try:
            record = json.loads(line)
        except (TypeError, json.JSONDecodeError):
            # A process killed while appending can leave one incomplete tail
            # record. The published-file checks remain the source of truth.
            continue
        if isinstance(record, dict):
            records.append(record)
    return records


def _slot_key(config: Dict[str, Any]) -> Tuple[str, Any]:
    slot_id = config.get("slot_id")
    if slot_id is not None:
        return "slot_id", slot_id
    return "output_subdir", config.get("output_subdir")


def _history_for_config(history: List[Dict[str, Any]],
                        config: Dict[str, Any]) -> List[Dict[str, Any]]:
    key, value = _slot_key(config)
    return [record for record in history if record.get(key) == value]


def _record_attempt_seed(record: Dict[str, Any], fallback: int) -> int:
    value = record.get("attempt_seed")
    try:
        return int(value) if value is not None else fallback
    except (TypeError, ValueError):
        return fallback


def _record_requests_resample(record: Dict[str, Any]) -> bool:
    status = str(record.get("status"))
    if status in _RESAMPLE_STATUSES:
        return True
    if status != _ERROR_STATUS:
        return False
    # Error records without a rejected_placement status are matched by message.
    # Match only deterministic candidate-exhaustion messages; generic runtime
    # errors must retain their bounded same-seed retry behavior.
    reasons = json.dumps(record.get("reasons", []), sort_keys=True).lower()
    return any(fragment in reasons
               for fragment in _LEGACY_PLACEMENT_ERROR_FRAGMENTS)


def _select_attempt_seed(
        config: Dict[str, Any], history: List[Dict[str, Any]],
        max_attempt_seed: int, max_scenario_error_retries: int) -> int:
    """Choose a deterministic seed without replaying known-bad attempts."""
    base_seed = int(config.get("attempt_seed", 0))
    records = [record for record in _history_for_config(history, config)
               if record.get("status") != "accepted"]
    if not records:
        selected = base_seed
    else:
        highest_seed = max(
            _record_attempt_seed(record, base_seed) for record in records)
        current = [record for record in records
                   if _record_attempt_seed(record, base_seed) == highest_seed]
        statuses = {str(record.get("status")) for record in current}
        if any(_record_requests_resample(record) for record in current):
            selected = highest_seed + 1
        elif statuses == {_ERROR_STATUS}:
            error_count = len(current)
            if error_count > max_scenario_error_retries:
                key, value = _slot_key(config)
                raise _AttemptPlanningError(
                    f"{key}={value} seed={highest_seed} produced "
                    f"{error_count} runtime errors (retry budget "
                    f"{max_scenario_error_retries} exhausted)")
            selected = highest_seed
        else:
            key, value = _slot_key(config)
            raise _AttemptPlanningError(
                f"{key}={value} has non-resampleable status(es) "
                f"{sorted(statuses)} at seed={highest_seed}")
    if selected > max_attempt_seed:
        key, value = _slot_key(config)
        raise _AttemptPlanningError(
            f"{key}={value} exhausted attempt seeds through "
            f"{max_attempt_seed}")
    return selected


def _generated_retry_config(
        base: Dict[str, Any], attempt_seed: int,
        generated_cache: Dict[int, List[Dict[str, Any]]],
        generation_size: int) -> Dict[str, Any]:
    base_seed = int(base.get("attempt_seed", 0))
    if attempt_seed == base_seed:
        return dict(base)
    try:
        scenario_index = int(base["scenario_index"])
    except (KeyError, TypeError, ValueError) as exc:
        raise _AttemptPlanningError(
            "automatic resampling requires integer scenario_index") from exc
    if attempt_seed not in generated_cache:
        from scenarios.generate_configs import generate_configs
        generated_cache[attempt_seed] = generate_configs(
            generation_size, seed=attempt_seed)
    try:
        retry = generated_cache[attempt_seed][scenario_index]
    except IndexError as exc:
        raise _AttemptPlanningError(
            f"scenario_index={scenario_index} is outside generated retry pool") from exc
    invariant_fields = (
        "scenario_index", "slot_id", "output_subdir", "template_id",
        "scenario_id", "map", "weather",
    )
    mismatched = [field for field in invariant_fields
                  if retry.get(field) != base.get(field)]
    if mismatched:
        raise _AttemptPlanningError(
            "generated retry changed slot invariants: " + ", ".join(mismatched))
    return retry


def _build_attempt_plan(
        missing: List[Dict[str, Any]], history: List[Dict[str, Any]],
        max_attempt_seed: int, max_scenario_error_retries: int,
        generated_cache: Dict[int, List[Dict[str, Any]]],
        generation_size: int,
        skipped: Optional[List[Tuple[Dict[str, Any], Exception]]] = None,
) -> List[Dict[str, Any]]:
    """Plan the next attempt of every missing slot. A slot that cannot be
    planned (attempt seeds exhausted, non-resampleable history) raises unless
    ``skipped`` is given, in which case it is appended there and the other
    slots are still planned: one slot the map cannot host must not stop the
    rest of the release."""
    plan = []
    for config in missing:
        try:
            attempt_seed = _select_attempt_seed(
                config, history, max_attempt_seed, max_scenario_error_retries)
            plan.append(_generated_retry_config(
                config, attempt_seed, generated_cache, generation_size))
        except _AttemptPlanningError as exc:
            if skipped is None:
                raise
            skipped.append((config, exc))
    return plan


def _write_attempt_plan(directory: str,
                        configs: List[Dict[str, Any]]) -> None:
    os.makedirs(directory, exist_ok=True)
    for index, config in enumerate(configs):
        name = f"{index:04d}_{config['output_subdir']}.yaml"
        # JSON is valid YAML and avoids a second serializer for ephemeral
        # supervisor-only work plans. Accepted configs are stored verbatim in
        # each published scenario directory by run_batch.
        with open(os.path.join(directory, name), "w", encoding="utf-8") as stream:
            json.dump(config, stream, indent=2)
            stream.write("\n")


def _port_is_open(host: str, port: int, timeout: float = 0.5) -> bool:
    try:
        with socket.create_connection((host, port), timeout=timeout):
            return True
    except OSError:
        return False


def _wait_for_port_closed(host: str, port: int, timeout: float) -> bool:
    deadline = time.monotonic() + timeout
    while _port_is_open(host, port):
        if time.monotonic() >= deadline:
            return False
        time.sleep(0.25)
    return True


def _listening_socket_inodes(port: int) -> Set[str]:
    """Return Linux socket inodes listening on *port*."""
    inodes: Set[str] = set()
    for table in ("/proc/net/tcp", "/proc/net/tcp6"):
        try:
            with open(table, encoding="ascii") as stream:
                rows = stream.readlines()[1:]
        except OSError:
            continue
        for row in rows:
            fields = row.split()
            if len(fields) < 10 or fields[3] != "0A":
                continue
            try:
                local_port = int(fields[1].rsplit(":", 1)[1], 16)
            except (IndexError, ValueError):
                continue
            if local_port == port:
                inodes.add(fields[9])
    return inodes


def _listener_pids(port: int) -> Set[int]:
    inodes = _listening_socket_inodes(port)
    if not inodes:
        return set()
    sockets = {f"socket:[{inode}]" for inode in inodes}
    pids: Set[int] = set()
    try:
        proc_entries = os.listdir("/proc")
    except OSError:
        return set()
    for entry in proc_entries:
        if not entry.isdigit():
            continue
        fd_dir = os.path.join("/proc", entry, "fd")
        try:
            descriptors = os.listdir(fd_dir)
        except OSError:
            continue
        for descriptor in descriptors:
            try:
                target = os.readlink(os.path.join(fd_dir, descriptor))
            except OSError:
                continue
            if target in sockets:
                pids.add(int(entry))
                break
    return pids


def _process_command(pid: int) -> List[str]:
    try:
        with open(f"/proc/{pid}/cmdline", "rb") as stream:
            return [part.decode("utf-8", errors="replace")
                    for part in stream.read().split(b"\0") if part]
    except OSError:
        return []


def _process_matches_carla(pid: int, args: argparse.Namespace) -> bool:
    try:
        expected_uid = (pwd.getpwnam(args.carla_user).pw_uid
                        if args.carla_user else os.getuid())
        actual_uid = os.stat(f"/proc/{pid}").st_uid
    except (KeyError, OSError):
        return False
    if actual_uid != expected_uid:
        return False
    command = " ".join(_process_command(pid))
    try:
        executable = os.path.realpath(os.readlink(f"/proc/{pid}/exe"))
    except OSError:
        executable = ""
    evidence = f"{command} {executable}"
    carla_root = os.path.realpath(args.carla_root)
    return "CarlaUE4" in evidence and carla_root in evidence


def _carla_process_port(command: List[str], default: int = 2000) -> Optional[int]:
    option_names = ("-carla-rpc-port", "--carla-rpc-port")
    for index, argument in enumerate(command):
        for option in option_names:
            if argument.startswith(f"{option}="):
                try:
                    return int(argument.split("=", 1)[1])
                except ValueError:
                    return None
            if argument == option:
                try:
                    return int(command[index + 1])
                except (IndexError, ValueError):
                    return None
    return default


def _matching_carla_process_pids(args: argparse.Namespace) -> Set[int]:
    """Find exact-root/user CARLA servers, even when /proc fd access is denied."""
    matches: Set[int] = set()
    try:
        entries = os.listdir("/proc")
    except OSError:
        return matches
    for entry in entries:
        if not entry.isdigit():
            continue
        pid = int(entry)
        if not _process_matches_carla(pid, args):
            continue
        if _carla_process_port(_process_command(pid)) == args.port:
            matches.add(pid)
    return matches


def _command_port(command: List[str], default: int = 2000) -> Optional[int]:
    for argument in command:
        if argument.startswith("--port="):
            try:
                return int(argument.split("=", 1)[1])
            except ValueError:
                return None
    try:
        index = command.index("--port")
    except ValueError:
        # A tmux/`sh -c` wrapper carries the whole supervisor command line as
        # ONE argument. Read the port out of that string; otherwise every
        # wrapper on the machine looks like port 2000 and blocks the recovery
        # of every other shard's server.
        match = re.search(r"--port[= ](\d+)", " ".join(command))
        return int(match.group(1)) if match else default
    try:
        return int(command[index + 1])
    except (IndexError, ValueError):
        return None


def _ancestor_pids(pid: Optional[int] = None) -> Set[int]:
    """This process and its ancestors (the tmux `sh -c` wrapper included)."""
    current = os.getpid() if pid is None else pid
    ancestors: Set[int] = set()
    while current > 1 and current not in ancestors:
        ancestors.add(current)
        try:
            with open(f"/proc/{current}/stat", encoding="utf-8") as stream:
                # "pid (comm) state ppid ..." — comm may contain spaces.
                current = int(stream.read().rsplit(")", 1)[1].split()[1])
        except (OSError, IndexError, ValueError):
            break
    return ancestors


def _active_carla_client_pids(port: int) -> List[int]:
    """Find OTHER supervisors/batches on `port` so recovery never kills their
    server. This supervisor's own wrapper shell is not another supervisor."""
    matches: List[int] = []
    own = _ancestor_pids()
    try:
        entries = os.listdir("/proc")
    except OSError:
        return matches
    for entry in entries:
        if not entry.isdigit() or int(entry) in own:
            continue
        command = _process_command(int(entry))
        joined = " ".join(command)
        if not any(module in joined for module in (
                "simulation.run_carla_supervised", "simulation.run_batch")):
            continue
        if _command_port(command) == port:
            matches.append(int(entry))
    return sorted(matches)


def _signal_processes(pids: Set[int], signum: int) -> None:
    for pid in sorted(pids):
        try:
            os.kill(pid, signum)
        except ProcessLookupError:
            continue


def _recover_stale_server(args: argparse.Namespace) -> bool:
    """Stop a listener only when it is verifiably this CARLA installation."""
    client_pids = _active_carla_client_pids(args.port)
    if client_pids:
        print(
            f"ERROR: an existing CARLA supervisor/batch is still active "
            f"(pid(s) {client_pids}); resume that run instead of starting a "
            "second one.", file=sys.stderr)
        return False

    listener_pids = _listener_pids(args.port)
    if listener_pids:
        matching = {pid for pid in listener_pids
                    if _process_matches_carla(pid, args)}
        if matching != listener_pids:
            return False
    else:
        # Vast containers can expose /proc/net/tcp while denying cross-UID
        # access to /proc/<pid>/fd. In that case socket-to-PID attribution is
        # unavailable even to container root. Recover only when the port has a
        # real listening inode and exactly one process matches root, user and
        # CARLA's configured/default RPC port.
        if not _listening_socket_inodes(args.port):
            return False
        matching = _matching_carla_process_pids(args)
        if len(matching) != 1:
            return False

    print(
        f"[supervisor] recovering stale CARLA listener on "
        f"{args.host}:{args.port} (pid(s) {sorted(matching)})")
    # Signal only the verified listener PIDs. A manually launched CARLA may
    # share a process group with an unrelated shell or job.
    _signal_processes(matching, signal.SIGTERM)
    shutdown_timeout = getattr(args, "shutdown_timeout", 30.0)
    if _wait_for_port_closed(args.host, args.port, shutdown_timeout):
        _clear_owner_state(args)
        return True
    _signal_processes(matching, signal.SIGKILL)
    recovered = _wait_for_port_closed(args.host, args.port, 5.0)
    if recovered:
        _clear_owner_state(args)
    return recovered


def _append_event(out_root: str, event: str, **fields: Any) -> None:
    log_dir = os.path.join(out_root, ".carla_supervisor")
    os.makedirs(log_dir, exist_ok=True)
    record = {"timestamp": round(time.time(), 3), "event": event, **fields}
    with open(os.path.join(log_dir, "supervisor_log.jsonl"), "a",
              encoding="utf-8") as stream:
        stream.write(json.dumps(record, sort_keys=True) + "\n")


def _start_server(
        args: argparse.Namespace, generation: int
        ) -> Tuple[subprocess.Popen, IO[str]]:
    log_dir = os.path.join(args.out_root, ".carla_supervisor")
    os.makedirs(log_dir, exist_ok=True)
    log_path = os.path.join(log_dir, f"carla_{generation:03d}.log")
    stream = open(log_path, "w", encoding="utf-8")
    try:
        process = subprocess.Popen(
            _server_command(args), cwd=args.carla_root,
            stdout=stream, stderr=subprocess.STDOUT,
            start_new_session=True,
        )
    except OSError:
        stream.close()
        raise
    return process, stream


def _carla_api_ready(host: str, port: int) -> bool:
    try:
        import carla
    except ImportError:
        return _port_is_open(host, port)
    try:
        client = carla.Client(host, port)
        client.set_timeout(2.0)
        world = client.get_world()
        world.get_snapshot()
        return True
    except Exception:
        return False


def _wait_for_ready(process: subprocess.Popen, host: str, port: int,
                    timeout: float) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        returncode = process.poll()
        if returncode is not None:
            raise RuntimeError(f"CARLA exited before readiness (exit={returncode})")
        if _carla_api_ready(host, port):
            return
        time.sleep(1.0)
    raise TimeoutError(f"CARLA did not become ready on {host}:{port} in {timeout:g}s")


def _start_batch(args: argparse.Namespace, target: int,
                 scenarios_dir: Optional[str] = None) -> subprocess.Popen:
    command = [
        args.python, "-u", "-m", "simulation.run_batch",
        "--scenarios-dir", scenarios_dir or args.scenarios_dir,
        "--out-root", args.out_root,
        "--host", args.host,
        "--port", str(args.port),
        "--limit", str(target),
        "--track", args.track,
    ]
    return subprocess.Popen(command, start_new_session=True)


def _stop_process(process: Optional[subprocess.Popen], timeout: float = 15.0) -> None:
    if process is None:
        return
    running = process.poll() is None
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        return
    except PermissionError:
        if running:
            process.terminate()
    if not running:
        return
    if running:
        try:
            process.wait(timeout=timeout)
            return
        except subprocess.TimeoutExpired:
            pass
    try:
        os.killpg(process.pid, signal.SIGKILL)
    except ProcessLookupError:
        return
    except PermissionError:
        if process.poll() is None:
            process.kill()
    if process.poll() is None:
        process.wait(timeout=5.0)


def _monitor_batch(batch: subprocess.Popen, server: subprocess.Popen,
                   poll_interval: float) -> Tuple[Optional[int], Optional[int]]:
    """Return (batch_rc, server_rc), stopping neither process."""
    while True:
        server_rc = server.poll()
        batch_rc = batch.poll()
        if server_rc is not None:
            return batch_rc, server_rc
        if batch_rc is not None:
            return batch_rc, None
        time.sleep(poll_interval)


def _close_stream(stream: Optional[IO[str]]) -> None:
    if stream is not None and not stream.closed:
        stream.close()


def _cleanup_server(
        args: argparse.Namespace, server: Optional[subprocess.Popen],
        stream: Optional[IO[str]]) -> bool:
    try:
        _stop_process(server)
    finally:
        _close_stream(stream)
    if server is None:
        return True
    shutdown_timeout = getattr(args, "shutdown_timeout", 30.0)
    if not _wait_for_port_closed(args.host, args.port, shutdown_timeout):
        # The sudo/shell launcher may have exited before a UE4 descendant.
        if not _recover_stale_server(args):
            _append_event(
                args.out_root, "server_cleanup_failed", server_pid=server.pid,
                host=args.host, port=args.port)
            return False
    _clear_owner_state(args)
    return True


def _run_plan_with_restarts(
        args: argparse.Namespace, generation: int, target: int,
        scenarios_dir: str, batch_limit: int
        ) -> Tuple[int, Optional[int], Optional[int]]:
    """Run one work plan, retrying only CARLA lifecycle failures.

    Returns ``(generation, batch_rc, fatal_rc)``. A non-zero ``batch_rc`` is
    interpreted by the caller using batch_log records; ``fatal_rc`` means the
    server could not be started, restarted, or cleaned up safely.
    """
    restarts = 0
    while True:
        generation += 1
        server: Optional[subprocess.Popen] = None
        server_stream: Optional[IO[str]] = None
        batch: Optional[subprocess.Popen] = None
        start_error: Optional[BaseException] = None
        batch_rc: Optional[int] = None
        server_rc: Optional[int] = None
        cleanup_ok = True
        try:
            try:
                server, server_stream = _start_server(args, generation)
                _write_owner_state(args, server, generation, target)
                _append_event(
                    args.out_root, "server_started",
                    generation=generation, pid=server.pid,
                    target_limit=target, batch_config_count=batch_limit)
                print(
                    f"[supervisor] CARLA generation {generation}; "
                    f"target prefix {target}, work configs {batch_limit}")
                _wait_for_ready(
                    server, args.host, args.port, args.startup_timeout)
                _append_event(
                    args.out_root, "server_ready",
                    generation=generation, target_limit=target)
            except (OSError, RuntimeError, TimeoutError) as exc:
                start_error = exc
                _append_event(
                    args.out_root, "server_start_failed",
                    generation=generation, target_limit=target,
                    error=str(exc))

            if start_error is None:
                if scenarios_dir == args.scenarios_dir:
                    batch = _start_batch(args, batch_limit)
                else:
                    batch = _start_batch(args, batch_limit, scenarios_dir)
                batch_rc, server_rc = _monitor_batch(
                    batch, server, args.poll_interval)
                if server_rc is not None:
                    _append_event(
                        args.out_root, "server_exited",
                        generation=generation, target_limit=target,
                        exit_code=server_rc, batch_exit_code=batch_rc)
        finally:
            # Covers normal completion, Python exceptions, Ctrl+C, SIGHUP and
            # SIGTERM. The latter two are translated by main into exceptions.
            _stop_process(batch)
            cleanup_ok = _cleanup_server(args, server, server_stream)

        if not cleanup_ok:
            print(
                f"ERROR: CARLA did not release {args.host}:{args.port} "
                "after supervised cleanup.", file=sys.stderr)
            return generation, None, 1
        if start_error is not None:
            restarts += 1
            if restarts > args.max_server_restarts:
                print(f"ERROR: {start_error}", file=sys.stderr)
                return generation, None, 1
            continue
        if server_rc is not None:
            if batch_rc == 0:
                # The plan was published before UE4 exited.
                return generation, batch_rc, None
            restarts += 1
            print(
                f"[supervisor] CARLA exited {server_rc}; retrying target "
                f"prefix {target} ({restarts}/{args.max_server_restarts})")
            if restarts > args.max_server_restarts:
                print(
                    "ERROR: CARLA exceeded restart budget at target prefix "
                    f"{target}", file=sys.stderr)
                return generation, None, 1
            continue
        return generation, batch_rc, None


def run_supervised(args: argparse.Namespace,
                   total_configs: Optional[int] = None) -> int:
    lease: Optional[IO[str]] = None
    try:
        try:
            lease = _acquire_lease(args)
        except _SupervisorLeaseBusy as exc:
            print(f"ERROR: {exc}", file=sys.stderr)
            return 2

        if _port_is_open(args.host, args.port):
            if not _recover_stale_server(args):
                print(
                    f"ERROR: {args.host}:{args.port} is already in use and is "
                    "not a recoverable stale CARLA process from the requested "
                    "--carla-root/--carla-user.", file=sys.stderr)
                return 2
        else:
            # A hard-killed supervisor can leave metadata after UE4 has exited.
            _clear_owner_state(args)

        configs = None
        if total_configs is None:
            configs = load_scenario_configs(args.scenarios_dir)
            total_configs = len(configs)
        total = min(args.limit, total_configs) if args.limit else total_configs
        targets = _chunk_targets(total, args.chunk_size)
        if configs is not None:
            completed = min(
                _completed_prefix(configs[:total], args.out_root), total)
            targets = [target for target in targets if target > completed]
            if completed:
                print(f"[supervisor] verified published prefix: {completed}/{total}")
        if not targets:
            print(
                "No scenario configs to process; requested prefix is already "
                "complete.")
            return 0

        generation = 0
        generated_cache: Dict[int, List[Dict[str, Any]]] = {}
        exhausted: Dict[Tuple[str, Any], str] = {}
        generation_size = (
            max(int(config.get("scenario_index", index))
                for index, config in enumerate(configs)) + 1
            if configs else 0)
        for target in targets:
            # total_configs is a unit-test hook. Real CLI runs always
            # have loaded configs and therefore use per-slot retry planning.
            if configs is None:
                generation, batch_rc, fatal_rc = _run_plan_with_restarts(
                    args, generation, target, args.scenarios_dir, target)
                if fatal_rc is not None:
                    return fatal_rc
                if batch_rc != 0:
                    _append_event(
                        args.out_root, "batch_failed",
                        generation=generation, target_limit=target,
                        batch_exit_code=batch_rc)
                    print(
                        f"ERROR: run_batch exited {batch_rc} while CARLA "
                        f"remained alive at --limit {target}; inspect "
                        "batch_log.jsonl.", file=sys.stderr)
                    return int(batch_rc or 1)
            else:
                prefix = configs[:target]
                no_result_failures = 0
                plan_round = 0
                with tempfile.TemporaryDirectory(
                        prefix=f"coriva_carla_target_{target}_") as work_root:
                    while True:
                        missing = [
                            config for config in _missing_configs(prefix, args.out_root)
                            if _slot_key(config) not in exhausted]
                        if not missing:
                            break
                        history = _batch_history(args.out_root)
                        skipped: List[Tuple[Dict[str, Any], Exception]] = []
                        plan = _build_attempt_plan(
                            missing, history, args.max_attempt_seed,
                            args.max_scenario_error_retries,
                            generated_cache, generation_size, skipped=skipped)
                        for config, exc in skipped:
                            # Record and skip: the remaining slots are still
                            # built, and the run ends with a non-zero exit
                            # that names every slot left unpublished.
                            exhausted[_slot_key(config)] = str(exc)
                            _append_event(
                                args.out_root, "attempts_exhausted",
                                target_limit=target, error=str(exc),
                                slot_id=config.get("slot_id"),
                                scenario_id=config.get("scenario_id"))
                            print(f"ERROR: {exc}; skipping this slot and "
                                  "continuing with the others", file=sys.stderr)
                        if not plan:
                            break

                        plan_round += 1
                        plan_dir = os.path.join(
                            work_root, f"round_{plan_round:03d}")
                        _write_attempt_plan(plan_dir, plan)
                        seed_counts = Counter(
                            int(config.get("attempt_seed", 0))
                            for config in plan)
                        seed_summary = ", ".join(
                            f"seed {seed}: {count}"
                            for seed, count in sorted(seed_counts.items()))
                        print(
                            f"[supervisor] target prefix {target}: "
                            f"{len(plan)} missing slot(s); {seed_summary}")
                        history_count = len(history)
                        missing_count = len(missing)
                        generation, batch_rc, fatal_rc = _run_plan_with_restarts(
                            args, generation, target, plan_dir, len(plan))
                        if fatal_rc is not None:
                            return fatal_rc

                        remaining = [
                            config for config in _missing_configs(prefix, args.out_root)
                            if _slot_key(config) not in exhausted]
                        if batch_rc == 0:
                            if remaining:
                                print(
                                    "ERROR: run_batch returned success but "
                                    f"{len(remaining)} target slot(s) remain "
                                    "unpublished.", file=sys.stderr)
                                return 1
                            break

                        new_history = _batch_history(args.out_root)
                        plan_keys = {_slot_key(config) for config in plan}
                        outcomes = [
                            record for record in new_history[history_count:]
                            if any(record.get(key) == value
                                   for key, value in plan_keys)
                        ]
                        if not outcomes and len(remaining) >= missing_count:
                            no_result_failures += 1
                            if no_result_failures > args.max_batch_restarts:
                                print(
                                    "ERROR: run_batch repeatedly exited "
                                    f"{batch_rc} without publishing or logging "
                                    "an outcome.", file=sys.stderr)
                                return int(batch_rc or 1)
                            print(
                                f"[supervisor] run_batch exited {batch_rc} "
                                "without a result; retrying the same work plan "
                                f"({no_result_failures}/"
                                f"{args.max_batch_restarts})")
                            continue

                        no_result_failures = 0
                        status_counts = Counter(
                            str(record.get("status")) for record in outcomes)
                        _append_event(
                            args.out_root, "batch_outcomes_retrying",
                            generation=generation, target_limit=target,
                            batch_exit_code=batch_rc,
                            remaining_slots=len(remaining),
                            statuses=dict(status_counts))
                        status_summary = ", ".join(
                            f"{status}={count}" for status, count
                            in sorted(status_counts.items()))
                        print(
                            f"[supervisor] batch outcomes: {status_summary}; "
                            f"automatically retrying {len(remaining)} "
                            "missing slot(s)")

            _append_event(
                args.out_root, "chunk_complete",
                generation=generation, target_limit=target,
                exhausted_slots=len(exhausted))
            print(f"[supervisor] completed cumulative --limit {target}"
                  + (f" with {len(exhausted)} slot(s) left unpublished"
                     if exhausted else ""))
        if exhausted:
            _append_event(
                args.out_root, "slots_exhausted",
                slots=[{"slot": f"{key}={value}", "error": error}
                       for (key, value), error in sorted(exhausted.items(), key=str)])
            print("ERROR: the release is incomplete; slots left unpublished "
                  "after every attempt seed: "
                  + ", ".join(f"{key}={value}" for (key, value) in sorted(exhausted, key=str)),
                  file=sys.stderr)
            return 1
        return 0
    finally:
        _release_lease(lease)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--scenarios-dir", default="scenarios/generated_seed0")
    parser.add_argument("--out-root", default="output_carla400_protocol10")
    parser.add_argument("--host", default="localhost")
    parser.add_argument("--port", type=int, default=2000)
    parser.add_argument("--track", choices=["primary", "challenge_6s"],
                        default="primary")
    parser.add_argument("--limit", type=int, default=0,
                        help="total configs to complete (0 = all)")
    parser.add_argument("--chunk-size", type=int, default=20)
    parser.add_argument("--max-server-restarts", type=int, default=3)
    parser.add_argument(
        "--max-batch-restarts", type=int, default=2,
        help="retries when run_batch exits without recording any result")
    parser.add_argument(
        "--max-scenario-error-retries", type=int, default=2,
        help="same-seed retries after a logged runtime error")
    parser.add_argument(
        "--max-attempt-seed", type=int, default=9,
        help="highest deterministic seed allowed for admission/QC resampling")
    parser.add_argument("--startup-timeout", type=float, default=180.0)
    parser.add_argument("--shutdown-timeout", type=float, default=30.0)
    parser.add_argument("--poll-interval", type=float, default=1.0)
    parser.add_argument("--carla-root", default="/workspace/carla_0.9.15")
    parser.add_argument("--carla-user", default="carlauser")
    parser.add_argument(
        "--vulkan-icd", default="/etc/vulkan/icd.d/nvidia_icd.json")
    parser.add_argument(
        "--supervisor-state-dir", default=tempfile.gettempdir(),
        help=argparse.SUPPRESS)
    parser.add_argument("--python", default=sys.executable)
    return parser


def main() -> int:
    args = _parser().parse_args()
    if args.limit < 0:
        raise SystemExit("--limit must be >= 0")
    if args.chunk_size <= 0:
        raise SystemExit("--chunk-size must be > 0")
    if args.max_server_restarts < 0:
        raise SystemExit("--max-server-restarts must be >= 0")
    if args.max_batch_restarts < 0:
        raise SystemExit("--max-batch-restarts must be >= 0")
    if args.max_scenario_error_retries < 0:
        raise SystemExit("--max-scenario-error-retries must be >= 0")
    if args.max_attempt_seed < 0:
        raise SystemExit("--max-attempt-seed must be >= 0")
    if args.shutdown_timeout < 0:
        raise SystemExit("--shutdown-timeout must be >= 0")
    handled_signals = [signal.SIGTERM]
    if hasattr(signal, "SIGHUP"):
        handled_signals.append(signal.SIGHUP)
    previous_handlers: Dict[int, Any] = {}
    for signum in handled_signals:
        previous_handlers[signum] = signal.getsignal(signum)
        signal.signal(signum, _raise_termination)
    try:
        return run_supervised(args)
    except KeyboardInterrupt:
        print(
            "\n[supervisor] interrupted; CARLA and run_batch were stopped.",
            file=sys.stderr)
        return 130
    except _TerminationRequested as exc:
        print(
            f"\n[supervisor] received signal {exc.signum}; CARLA and "
            "run_batch were stopped.", file=sys.stderr)
        return 128 + exc.signum
    finally:
        for signum, previous in previous_handlers.items():
            signal.signal(signum, previous)


if __name__ == "__main__":
    raise SystemExit(main())
