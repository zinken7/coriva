"""Reference terminal rules shared by the kinematic twin (kinematic-twin-5)
and the CARLA builder:

* the collision node is the last sampled state at or before first contact,
  contact being located between samples (twin: interpolation at 0.1 s;
  CARLA: the recorded 0.1 s hit tick);
* `safe_stop` requires the ego to stay stopped from the stop through the end
  of the horizon; a stop followed by motion is not a terminal event.
"""
import copy
import importlib.util
import json
import os
import sys
from pathlib import Path

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from simulation.kinematic_twin import (  # noqa: E402
    TWIN_VERSION, TwinReferenceError, _detect_terminal_partner, build_twin_branch)


def _load_test_module(name):
    path = Path(__file__).with_name(name)
    spec = importlib.util.spec_from_file_location(name[:-3], path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _ego(x, speed, y=0.0):
    return {"id": "ego", "entity_type": "ego_vehicle", "role_type": "ego_agent",
            "x": x, "y": y, "vx": speed, "vy": 0.0, "speed": speed, "length": 4.0, "width": 1.8}


def _car(aid, x, speed=0.0, y=0.0, role="lead_vehicle"):
    return {"id": aid, "entity_type": "vehicle", "role_type": role,
            "x": x, "y": y, "vx": speed, "vy": 0.0, "speed": speed, "length": 4.5, "width": 1.9}


# ── twin: collision node = last sampled state at or before contact ───────

def test_twin_version_is_five():
    assert TWIN_VERSION == "kinematic-twin-5"


def test_contact_between_samples_is_labelled_at_the_previous_sample():
    # ego 10 m/s towards a stationary car 12 m ahead: half-lengths sum to
    # 4.25 m, so contact at (12 - 4.25) / 10 = 0.775 s, between samples 1 and 2.
    root = {"depth": 0, "timestamp": 0.0, "event_type": "observed_state",
            "actors": [_ego(0.0, 10.0), _car("npc_1", 12.0)]}
    branch = build_twin_branch("b", False, "do_ego_no_action", root, 6, 0.5, 0.5)
    chain = branch["chain"]
    assert [n["event_type"] for n in chain] == ["collision"]
    assert chain[-1]["depth"] == 1 and chain[-1]["terminal"] is True
    ego = next(a for a in chain[-1]["actors_update"] if a["id"] == "ego")
    npc = next(a for a in chain[-1]["actors_update"] if a["id"] == "npc_1")
    assert npc["role_type"] == "impacted_actor"
    assert abs(npc["x"] - ego["x"]) > 4.25  # the node state is pre-contact


def test_contact_exactly_on_a_sample_is_labelled_at_that_sample():
    # contact at (13.25 - 4.25) / 9 = 1.0 s exactly -> node depth 2 (t = 1.0 s)
    root = {"depth": 0, "timestamp": 0.0, "event_type": "observed_state",
            "actors": [_ego(0.0, 9.0), _car("npc_1", 13.25)]}
    branch = build_twin_branch("b", False, "do_ego_no_action", root, 6, 0.5, 0.5)
    chain = branch["chain"]
    assert [n["event_type"] for n in chain] == ["maintain_course", "collision"]
    assert chain[-1]["depth"] == 2


def test_pass_through_contact_between_samples_is_detected():
    # A crossing actor cuts through the ego's path between two samples and
    # is clear again at both: the sampled states never overlap, the
    # interpolated ones do.
    root_actors = [_ego(0.0, 20.0), _car("ped", 10.0, y=-6.0, role="crossing_actor")]
    for a in root_actors[1:]:
        a.update(entity_type="pedestrian", length=0.6, width=0.6, vx=0.0, vy=12.0, speed=12.0)
    # The ego moves x 10 -> 20 while the pedestrian stands at x = 15 and
    # crosses y -3 -> 3: clear at both samples (|dx| = 5 > 2.3), overlapping
    # at 40 % of the interval (|dx| = 1, |dy| = 0.6).
    states = [{"ego": _ego(10.0, 20.0), "ped": dict(root_actors[1], x=15.0, y=-3.0)},
              {"ego": _ego(20.0, 20.0), "ped": dict(root_actors[1], x=15.0, y=3.0)}]
    ev, idx, partner = _detect_terminal_partner(states, root_actors, 4.0, 2.0)
    assert (ev, idx, partner) == ("collision", 0, ["ped"])


def test_contact_inside_the_first_interval_is_not_representable():
    root = {"depth": 0, "timestamp": 0.0, "event_type": "observed_state",
            "actors": [_ego(0.0, 10.0), _car("npc_1", 6.0)]}
    with pytest.raises(TwinReferenceError, match="first sample interval"):
        build_twin_branch("b", False, "do_ego_no_action", root, 6, 0.5, 0.5)


# ── twin: safe_stop requires holding the stop to the horizon ─────────────

def _worlds(speeds):
    out = []
    x = 0.0
    for s in speeds:
        x += s * 0.5
        out.append({"ego": _ego(x, s), "npc_1": _car("npc_1", 200.0)})
    return out


def test_stop_then_resume_is_not_a_terminal_event():
    root = {"depth": 0, "timestamp": 0.0, "event_type": "observed_state",
            "actors": [_ego(0.0, 5.0), _car("npc_1", 200.0)]}
    branch = build_twin_branch("factual", True, "do_ego_no_action", root, 6, 0.5, 0.5,
                               override_states=_worlds([0.0, 2.0, 2.0, 2.0, 2.0, 2.0]))
    chain = branch["chain"]
    assert len(chain) == 6 and "safe_stop" not in [n["event_type"] for n in chain]
    assert not chain[-1].get("terminal")


def test_stop_held_to_the_horizon_is_safe_stop_at_the_first_stopped_sample():
    root = {"depth": 0, "timestamp": 0.0, "event_type": "observed_state",
            "actors": [_ego(0.0, 5.0), _car("npc_1", 200.0)]}
    branch = build_twin_branch("factual", True, "do_ego_no_action", root, 6, 0.5, 0.5,
                               override_states=_worlds([3.0, 0.2, 0.0, 0.0, 0.0, 0.0]))
    chain = branch["chain"]
    assert [n["event_type"] for n in chain] == ["maintain_course", "safe_stop"]
    assert chain[-1]["depth"] == 2 and chain[-1]["terminal"] is True


def test_transient_stop_before_a_later_stop_is_not_terminal():
    root = {"depth": 0, "timestamp": 0.0, "event_type": "observed_state",
            "actors": [_ego(0.0, 5.0), _car("npc_1", 200.0)]}
    branch = build_twin_branch("factual", True, "do_ego_no_action", root, 6, 0.5, 0.5,
                               override_states=_worlds([0.0, 3.0, 3.0, 0.0, 0.0, 0.0]))
    assert [n["event_type"] for n in branch["chain"]] == ["maintain_course"] * 3 + ["safe_stop"]
    assert branch["chain"][-1]["depth"] == 4


# ── CARLA builder: the same stop rule on raw telemetry ───────────────────

def _carla_steps(helper, until, stop_from, resume_at=None):
    steps = []
    n = int(round(until / helper.STEP))
    x = 0.0
    for i in range(n + 1):
        t = round(i * helper.STEP, 2)
        stopped = t >= stop_from and (resume_at is None or t < resume_at)
        speed = 0.0 if stopped else 10.0
        x += speed * helper.STEP if i else 0.0
        steps.append({"time_sec": t, "ego": helper._state(x, 0.0, speed),
                      "npc_1": helper._state(100.0, 0.0, 10.0),
                      "park_l1": helper._state(200.0, -3.5, 0.0),
                      "npc_3": helper._state(-300.0, 0.0, 10.0)})
    return steps


def test_carla_stop_then_resume_is_not_safe_stop():
    helper = _load_test_module("test_carla_reference_collision.py")
    steps = _carla_steps(helper, 8.0, stop_from=4.5, resume_at=5.5)
    chain = helper._chain(steps)
    assert len(chain) == 6 and "safe_stop" not in [n["event_type"] for n in chain]
    assert not chain[-1].get("terminal")


def test_carla_stop_held_through_horizon_is_safe_stop():
    helper = _load_test_module("test_carla_reference_collision.py")
    steps = _carla_steps(helper, 8.0, stop_from=4.5)
    chain = helper._chain(steps)
    assert chain[-1]["event_type"] == "safe_stop" and chain[-1]["depth"] == 2


def test_carla_stop_resumed_after_the_horizon_is_still_safe_stop():
    helper = _load_test_module("test_carla_reference_collision.py")
    steps = _carla_steps(helper, 8.0, stop_from=4.5, resume_at=7.0)  # horizon ends at 6.5 s
    chain = helper._chain(steps)
    assert chain[-1]["event_type"] == "safe_stop" and chain[-1]["depth"] == 2


# ── verifier re-derives the stop rule from raw ───────────────────────────

def _stop_raw_ego(raw, stream, stop_from, resume_at=None):
    for step in raw["branches"][stream]:
        t = step["time_sec"]
        if t >= stop_from and (resume_at is None or t < resume_at):
            step["ego"]["speed_kmh"] = 0.0
            step["ego"]["vx"] = 0.0


def test_verifier_rejects_safe_stop_when_raw_ego_moves_again(tmp_path):
    vt = _load_test_module("test_verify_carla_dataset.py")
    from scenarios.verify_carla_dataset import verify
    scenario = vt._fixture(tmp_path)
    raw, gt = vt._load(scenario, "raw_telemetry.json"), vt._load(scenario, "ground_truth.json")
    _stop_raw_ego(raw, "Factual_Brake", stop_from=4.5, resume_at=5.5)
    gt["branches"][0]["chain"] = gt["branches"][0]["chain"][:2]
    gt["branches"][0]["chain"][-1].update(event_type="safe_stop", terminal=True)
    gt["branches"][0]["chain"][-1].pop("action_type", None)
    vt._write_json(scenario / "raw_telemetry.json", raw)
    vt._write_json(scenario / "ground_truth.json", gt)
    report = verify(str(tmp_path), expected=1)
    assert not report["passed"]
    assert any("Factual_Brake" in e and "does not stay stopped" in e for e in report["errors"]), report["errors"]


def test_verifier_rejects_missing_safe_stop_when_raw_ego_holds(tmp_path):
    vt = _load_test_module("test_verify_carla_dataset.py")
    from scenarios.verify_carla_dataset import verify
    scenario = vt._fixture(tmp_path)
    raw = vt._load(scenario, "raw_telemetry.json")
    _stop_raw_ego(raw, "Factual_Brake", stop_from=4.5)
    vt._write_json(scenario / "raw_telemetry.json", raw)
    report = verify(str(tmp_path), expected=1)
    assert not report["passed"]
    assert any("Factual_Brake" in e and "has no safe_stop node" in e for e in report["errors"]), report["errors"]


def test_verifier_accepts_safe_stop_matching_raw_hold(tmp_path):
    vt = _load_test_module("test_verify_carla_dataset.py")
    from scenarios.verify_carla_dataset import verify
    scenario = vt._fixture(tmp_path)
    raw, gt = vt._load(scenario, "raw_telemetry.json"), vt._load(scenario, "ground_truth.json")
    _stop_raw_ego(raw, "Factual_Brake", stop_from=4.5)
    gt["branches"][0]["chain"] = gt["branches"][0]["chain"][:2]
    gt["branches"][0]["chain"][-1].update(event_type="safe_stop", terminal=True)
    gt["branches"][0]["chain"][-1].pop("action_type", None)
    vt._write_json(scenario / "raw_telemetry.json", raw)
    vt._write_json(scenario / "ground_truth.json", gt)
    report = verify(str(tmp_path), expected=1)
    assert report["passed"], report["errors"]


# ── offline rebuild script: change detection and write ───────────────────

def test_rebuild_script_reports_and_writes_changes(tmp_path, monkeypatch):
    spec = importlib.util.spec_from_file_location(
        "rebuild_carla_gt_from_raw",
        Path(__file__).resolve().parents[1] / "scripts" / "rebuild_carla_gt_from_raw.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    vt = _load_test_module("test_verify_carla_dataset.py")
    scenario = vt._fixture(tmp_path)
    old = vt._load(scenario, "ground_truth.json")

    def fake_build(raw, cfg, track="primary", protocol=True):
        new = copy.deepcopy(old)
        new["branches"][0]["chain"] = new["branches"][0]["chain"][:2]
        new["branches"][0]["chain"][-1].update(event_type="safe_stop", terminal=True)
        new["_observed_history"] = {"dropped": True}
        return new
    monkeypatch.setattr(mod, "build_scenario_gt", fake_build)
    r = mod.rebuild_one(str(scenario))
    assert not r["identical"] and r["changes"][0]["stream"] == "Factual_Brake"
    assert r["changes"][0]["before"] == (6, "maintain_course", False)
    assert r["changes"][0]["after"] == (2, "safe_stop", True)
    assert "_observed_history" not in r["new"]
    monkeypatch.setattr(sys, "argv", ["rebuild", "--root", str(tmp_path), "--write",
                                      "--report", str(tmp_path / "report.json")])
    assert mod.main() == 0
    written = vt._load(scenario, "ground_truth.json")
    assert written["branches"][0]["chain"][-1]["event_type"] == "safe_stop"
    report = json.loads((tmp_path / "report.json").read_text())
    assert report and report[0]["changes"][0]["stream"] == "Factual_Brake"
    # identical rebuild -> nothing to report
    monkeypatch.setattr(mod, "build_scenario_gt", lambda raw, cfg, track="primary", protocol=True: copy.deepcopy(written))
    assert mod.rebuild_one(str(scenario))["identical"]


# ── partners, first-interval policy, coverage, exit code ─────────────────

def test_simultaneous_contacts_label_every_partner_regardless_of_root_order():
    def build(order):
        actors = [_ego(0.0, 10.0), _car("npc_1", 12.0, y=0.0), _car("npc_2", 12.0, y=0.0, role="adjacent_vehicle")]
        if order == "swapped":
            actors = [actors[0], actors[2], actors[1]]
        root = {"depth": 0, "timestamp": 0.0, "event_type": "observed_state", "actors": actors}
        chain = build_twin_branch("b", False, "do_ego_no_action", root, 6, 0.5, 0.5)["chain"]
        return sorted(a["id"] for a in chain[-1]["actors_update"] if a["role_type"] == "impacted_actor")
    assert build("normal") == ["npc_1", "npc_2"]
    assert build("swapped") == ["npc_1", "npc_2"]


def test_first_interval_contact_is_labelled_at_the_first_sample_for_predictions():
    root = {"depth": 0, "timestamp": 0.0, "event_type": "observed_state",
            "actors": [_ego(0.0, 10.0), _car("npc_1", 6.0)]}
    with pytest.raises(TwinReferenceError):
        build_twin_branch("b", False, "do_ego_no_action", root, 6, 0.5, 0.5)
    chain = build_twin_branch("b", False, "do_ego_no_action", root, 6, 0.5, 0.5,
                              first_interval="label_first")["chain"]
    assert [n["event_type"] for n in chain] == ["collision"] and chain[0]["depth"] == 1


def test_action_profile_control_never_raises_on_early_contact():
    import json
    from analysis.baselines import action_profile_kinematic_text
    from simulation.kinematic_twin import build_twin_gt
    root = {"depth": 0, "timestamp": 0.0, "event_type": "observed_state",
            "actors": [_ego(0.0, 10.0), _car("npc_1", 20.0)]}
    gt = build_twin_gt(root, dict(scenario_id="early", n_steps=6, dt=.5, sample_interval=.5, branches=[
        dict(branch_id="brake", is_factual=True, intervention_type="do_ego_emergency_brake"),
        dict(branch_id="go", is_factual=False, intervention_type="do_ego_accelerate_pass")]))
    # Make the control's own rollout hit inside the first interval: move the
    # car close in the root the control reads (the reference above is valid).
    gt["observed_state"]["actors"][1]["x"] = 6.0
    pred = json.loads(action_profile_kinematic_text(gt))
    assert all(len(b["chain"]) >= 1 for b in pred["predicted_branches"])


def test_carla_stop_without_raw_coverage_of_the_horizon_is_not_safe_stop():
    helper = _load_test_module("test_carla_reference_collision.py")
    steps = _carla_steps(helper, 4.5, stop_from=4.5)  # raw ends at 4.5 s, horizon 6.5 s
    chain = helper._chain(steps)
    assert "safe_stop" not in [n["event_type"] for n in chain]


def test_verifier_flags_raw_that_ends_before_the_horizon(tmp_path):
    vt = _load_test_module("test_verify_carla_dataset.py")
    from scenarios.verify_carla_dataset import verify
    scenario = vt._fixture(tmp_path)
    raw, gt = vt._load(scenario, "raw_telemetry.json"), vt._load(scenario, "ground_truth.json")
    _stop_raw_ego(raw, "Factual_Brake", stop_from=4.5)
    raw["branches"]["Factual_Brake"] = [s for s in raw["branches"]["Factual_Brake"] if s["time_sec"] <= 4.5]
    gt["branches"][0]["chain"] = gt["branches"][0]["chain"][:2]
    gt["branches"][0]["chain"][-1].update(event_type="safe_stop", terminal=True)
    gt["branches"][0]["chain"][-1].pop("action_type", None)
    vt._write_json(scenario / "raw_telemetry.json", raw)
    vt._write_json(scenario / "ground_truth.json", gt)
    report = verify(str(tmp_path), expected=1)
    assert not report["passed"]
    assert any("Factual_Brake" in e and "ends at" in e for e in report["errors"]), report["errors"]
    assert any("Factual_Brake" in e and "does not stay stopped" in e for e in report["errors"]), report["errors"]


def test_rebuild_script_refuses_to_write_when_any_scenario_fails(tmp_path, monkeypatch):
    spec = importlib.util.spec_from_file_location(
        "rebuild_carla_gt_from_raw2",
        Path(__file__).resolve().parents[1] / "scripts" / "rebuild_carla_gt_from_raw.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    vt = _load_test_module("test_verify_carla_dataset.py")
    good = vt._fixture(tmp_path)
    bad = tmp_path / "scenario_2"
    bad.mkdir()
    for name in ("raw_telemetry.json", "scenario_config.json", "ground_truth.json"):
        (bad / name).write_text((good / name).read_text())
    old = vt._load(good, "ground_truth.json")

    def fake_build(raw, cfg, track="primary", protocol=True):
        if cfg.get("_fail"):
            raise KeyError("actor_defs")
        new = copy.deepcopy(old)
        new["branches"][0]["chain"] = new["branches"][0]["chain"][:2]
        new["branches"][0]["chain"][-1].update(event_type="safe_stop", terminal=True)
        return new
    cfg_bad = json.loads((bad / "scenario_config.json").read_text())
    cfg_bad["_fail"] = True
    (bad / "scenario_config.json").write_text(json.dumps(cfg_bad))
    monkeypatch.setattr(mod, "build_scenario_gt", fake_build)
    monkeypatch.setattr(sys, "argv", ["rebuild", "--root", str(tmp_path), "--write"])
    assert mod.main() == 1
    assert vt._load(good, "ground_truth.json") == old  # nothing written
    assert not list(tmp_path.glob("*/ground_truth.json.rebuild.tmp"))


# ── review round 3: shared stop threshold, prompt wording, missing files ──

def test_stop_threshold_is_shared_by_twin_builder_verifier_and_prompt():
    from simulation.protocol import STOP_SPEED_MPS
    import simulation.kinematic_twin as twin
    import simulation.build_ground_truth as bgt
    import scenarios.verify_carla_dataset as ver
    from simulation.build_pilot import build_prompt
    assert twin.STOP_SPEED_MPS == STOP_SPEED_MPS == 0.5
    defs = {"ego": {"canonical_id": "ego", "entity_type": "ego_vehicle"}}
    assert bgt._ego_stopped({"ego": {"speed_kmh": 1.8}}, defs)
    assert not bgt._ego_stopped({"ego": {"speed_kmh": 1.9}}, defs)
    assert ver._expected_stop_depth([{"time_sec": 3.5 + 0.1 * i, "ego": {"speed_kmh": 1.8}} for i in range(31)],
                                    3.5, 6.5, defs) == 1
    root = {"depth": 0, "timestamp": 0.0, "event_type": "observed_state",
            "actors": [_ego(0.0, 5.0), _car("npc_1", 30.0)]}
    text = build_prompt(root, "s", ["do_ego_no_action"], condition="video", track="primary")
    assert "at or below 0.5 m/s" in text and "AT OR BEFORE" in text and "inside the horizon" in text
    assert "2–" not in text  # chains may end at depth 1


def test_rebuild_script_treats_missing_inputs_as_errors(tmp_path, monkeypatch):
    spec = importlib.util.spec_from_file_location(
        "rebuild_carla_gt_from_raw3",
        Path(__file__).resolve().parents[1] / "scripts" / "rebuild_carla_gt_from_raw.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    vt = _load_test_module("test_verify_carla_dataset.py")
    scenario = vt._fixture(tmp_path)
    (scenario / "scenario_config.json").unlink()
    (tmp_path / ".rejected").mkdir()
    monkeypatch.setattr(sys, "argv", ["rebuild", "--root", str(tmp_path)])
    assert mod.main() == 1
    monkeypatch.setattr(sys, "argv", ["rebuild", "--root", str(tmp_path / "empty")])
    (tmp_path / "empty").mkdir()
    assert mod.main() == 1


def test_rebuild_script_qc_blocks_writing_a_reference_the_gates_reject(tmp_path, monkeypatch):
    spec = importlib.util.spec_from_file_location(
        "rebuild_carla_gt_from_raw4",
        Path(__file__).resolve().parents[1] / "scripts" / "rebuild_carla_gt_from_raw.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    vt = _load_test_module("test_verify_carla_dataset.py")
    scenario = vt._fixture(tmp_path)
    old = vt._load(scenario, "ground_truth.json")

    def fake_build(raw, cfg, track="primary", protocol=True):
        new = copy.deepcopy(old)
        new["branches"][0]["chain"][2]["timestamp"] = 1.7  # off the protocol grid
        return new
    monkeypatch.setattr(mod, "build_scenario_gt", fake_build)
    monkeypatch.setattr(sys, "argv", ["rebuild", "--root", str(tmp_path), "--qc", "--write",
                                      "--report", str(tmp_path / "r.json")])
    assert mod.main() == 1
    assert vt._load(scenario, "ground_truth.json") == old
    report = json.loads((tmp_path / "r.json").read_text())
    assert report[0]["error"] == "qc_rejected"


def test_ego_held_exactly_at_the_stop_threshold_is_safe_stop_in_all_three_places():
    """Boundary of the shared convention: speed == STOP_SPEED_MPS counts as
    stopped for the twin, the CARLA builder and the verifier alike."""
    from simulation.protocol import STOP_SPEED_MPS
    import scenarios.verify_carla_dataset as ver
    # twin: recorded factual holding exactly 0.5 m/s for the whole horizon
    root = {"depth": 0, "timestamp": 0.0, "event_type": "observed_state",
            "actors": [_ego(0.0, STOP_SPEED_MPS), _car("npc_1", 200.0)]}
    branch = build_twin_branch("factual", True, "do_ego_no_action", root, 6, 0.5, 0.5,
                               override_states=_worlds([STOP_SPEED_MPS] * 6))
    assert [n["event_type"] for n in branch["chain"]] == ["safe_stop"]
    assert branch["chain"][0]["depth"] == 1
    # just above the threshold is still moving
    branch = build_twin_branch("factual", True, "do_ego_no_action", root, 6, 0.5, 0.5,
                               override_states=_worlds([STOP_SPEED_MPS + 0.01] * 6))
    assert "safe_stop" not in [n["event_type"] for n in branch["chain"]]
    # CARLA builder on raw holding exactly 1.8 km/h
    helper = _load_test_module("test_carla_reference_collision.py")
    steps = []
    for i in range(int(round(8.0 / helper.STEP)) + 1):
        t = round(i * helper.STEP, 2)
        steps.append({"time_sec": t, "ego": helper._state(STOP_SPEED_MPS * t, 0.0, STOP_SPEED_MPS),
                      "npc_1": helper._state(100.0, 0.0, 10.0),
                      "park_l1": helper._state(200.0, -3.5, 0.0),
                      "npc_3": helper._state(-300.0, 0.0, 10.0)})
    chain = helper._chain(steps)
    assert chain[-1]["event_type"] == "safe_stop" and chain[-1]["depth"] == 1
    # verifier re-derivation on the same raw
    defs = helper._actor_defs()
    assert ver._expected_stop_depth(steps, helper.T0, helper.T0 + 3.0, defs) == 1


def test_builder_direct_script_import_branch_also_carries_the_stop_threshold():
    """`build_ground_truth.py` can run as a script from Coding/simulation/;
    that fallback import must expose the same STOP_SPEED_MPS as the package
    import, or `_ego_stopped` would raise NameError there."""
    import subprocess
    code = ("import sys; sys.path.insert(0, 'simulation'); sys.modules.pop('simulation', None); "
            "import runpy; g = runpy.run_path('simulation/build_ground_truth.py', run_name='not_main'); "
            "print(g['STOP_SPEED_MPS'], g['_ego_stopped']({'ego': {'speed_kmh': 1.8}}, "
            "{'ego': {'canonical_id': 'ego', 'entity_type': 'ego_vehicle'}}))")
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True,
                         cwd=str(Path(__file__).resolve().parents[1]))
    assert out.returncode == 0, out.stderr
    assert out.stdout.strip() == "0.5 True"
