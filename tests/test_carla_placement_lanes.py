"""Lane-aware CARLA placement and the NPC pass-through gate.

Fixed lateral slots resolved without asking the map would put cut-in cars on
the pavement and drive NPCs through parked cars, so placement asks the map:
a moving vehicle's slot must be a same-direction driving lane, a parked
vehicle's fallback slot must stay clear of every moving actor's lane, and any
rollout in which a moving actor still passes through a static actor is
rejected before it can become a reference.
"""
import copy
import importlib.util
import json
import os
import sys
from pathlib import Path

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from simulation.generate import (  # noqa: E402
    PlacementRejected, ScenarioCandidateExhausted, _lane_class, _lane_path_class,
    _parked_lateral_candidates, spawn_cast)
from simulation.build_ground_truth import ReferenceBuildError, static_pass_through  # noqa: E402


def _runner():
    spec = importlib.util.spec_from_file_location("test_carla_runner", Path(__file__).with_name("test_carla_runner.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


class _Loc:
    def __init__(self, x, y, z=0.0):
        self.x, self.y, self.z = x, y, z


class _Rot:
    def __init__(self, yaw):
        self.yaw = yaw

    def get_right_vector(self):
        return _Loc(0.0, 1.0, 0.0)


class _Tf:
    def __init__(self, x, y, yaw):
        self.location = _Loc(x, y)
        self.rotation = _Rot(yaw)


class _Wp:
    def __init__(self, x, y, yaw=0.0, lane_width=3.5, name="wp"):
        self.transform = _Tf(x, y, yaw)
        self.lane_width = lane_width
        self.name = name

    def next(self, dist):
        return [self]

    def previous(self, dist):
        return [self]


class _MapWorld:
    """Fake world whose map answers get_waypoint with the nearest driving-lane
    waypoint: lanes at y = 0 (ego, yaw 0) and y = 3.5 (right lane, yaw 0)
    only when `right_lane` is set; y = -3.5 is an opposite-direction lane."""

    def __init__(self, right_lane=True):
        self.right_lane = right_lane
        self.cleanups = 0
        self.ticks = 0

    def get_waypoint(self, x, y, z=0.0):
        lanes = [(0.0, 0.0), (-3.5, 180.0)] + ([(3.5, 0.0)] if self.right_lane else [])
        ly, yaw = min(lanes, key=lambda item: abs(item[0] - y))
        return _Wp(x, ly, yaw)

    def tick(self):
        self.ticks += 1

    def cleanup(self):
        self.cleanups += 1


# ── _lane_class ───────────────────────────────────────────────────────────

def test_lane_class_distinguishes_same_opposite_and_off_driving_ground():
    world = _MapWorld(right_lane=False)
    ref = _Tf(0.0, 0.0, 0.0)
    assert _lane_class(world, _Loc(10.0, 0.0), ref) == "driving_same"
    assert _lane_class(world, _Loc(10.0, -3.5), ref) == "driving_opposite"
    assert _lane_class(world, _Loc(10.0, 3.5), ref) == "off_driving"  # nearest lane 3.5 m away
    assert _lane_class(_runner().FakeWorld(), _Loc(10.0, 3.5), ref) is None  # fake without lane data: unknown


# ── moving vehicle slots ──────────────────────────────────────────────────

def _cut_in_config(lateral=3.5):
    return {
        "scenario_id": "cut_in_test", "seed": 0,
        "intervention_time_sec": 3.5, "total_duration_sec": 8.0,
        "environment_event": {"event_type": "npc_cut_in_left", "trigger_time_sec": 3.5},
        "actor_defs": {
            "ego": {"canonical_id": "ego", "entity_type": "ego_vehicle", "default_role": "ego_agent",
                    "model": "vehicle.tesla.model3", "initial_speed": 15.0, "distance_ahead": 0.0, "lateral_offset": 0.0},
            "npc_1": {"canonical_id": "npc_1", "entity_type": "vehicle", "default_role": "adjacent_vehicle",
                      "model": "vehicle.audi.tt", "initial_speed": 14.0, "distance_ahead": 8.0, "lateral_offset": lateral},
        },
        "env_behavior": {"npc_1": {"accel": 0.0, "lat": -1.8}},
        "branches": {"b": {"is_factual": True, "intervention_type": "do_ego_no_action",
                           "target_actor_id": "ego", "trigger_depth": 1}},
    }


def test_cut_in_slot_on_the_pavement_moves_to_the_next_candidate():
    r = _runner()

    class CandidateWorld(_MapWorld):
        def __init__(self):
            super().__init__(right_lane=False)
            self.calls = 0

        def get_scenario_waypoints(self, *args, **kwargs):
            return [_Wp(0.0, 0.0, name="no_right_lane"), _Wp(500.0, 0.0, name="with_right_lane")]

        def get_waypoint(self, x, y, z=0.0):
            self.right_lane = x >= 250.0  # the second candidate has a right lane
            return super().get_waypoint(x, y, z)

    world = CandidateWorld()
    placement = {}
    cast = spawn_cast(_cut_in_config(3.5), world, r.FAKE_API, placement=placement)
    actors = cast[0]
    assert placement["candidate_number"] == 2
    assert actors["npc_1"].kwargs["waypoint"].name == "with_right_lane"
    assert placement["moving_actor_lane_checks"] == {"npc_1": {"class": "driving_same", "span_m": [8.0, 8.0], "travel_m": 0.0}}
    assert world.cleanups == 1  # the partial cast at candidate 1 was destroyed


def test_cut_in_slot_against_traffic_is_rejected_everywhere():
    r = _runner()

    class OneCandidate(_MapWorld):
        def get_scenario_waypoints(self, *args, **kwargs):
            return [_Wp(0.0, 0.0, name="only")]

    with pytest.raises(ScenarioCandidateExhausted, match="driving_opposite"):
        spawn_cast(_cut_in_config(-3.5), OneCandidate(), r.FAKE_API, placement={})


def test_lane_checks_are_unknown_for_legacy_fake_worlds():
    r = _runner()
    placement = {}
    spawn_cast(_cut_in_config(3.5), r.FakeWorld(), r.FAKE_API, placement=placement)
    assert placement["moving_actor_lane_checks"] == {"npc_1": {"class": "unknown", "span_m": [8.0, 8.0], "travel_m": 0.0}}


# ── parked slots ──────────────────────────────────────────────────────────

def _parked_config():
    cfg = _cut_in_config(3.5)
    cfg["actor_defs"]["npc_2"] = {"canonical_id": "npc_2", "entity_type": "vehicle", "default_role": "adjacent_vehicle",
                                  "model": "vehicle.audi.a2", "initial_speed": 12.0, "distance_ahead": 25.0, "lateral_offset": 3.5}
    cfg["actor_defs"]["park_r1"] = {"canonical_id": "park_r1", "entity_type": "parked_vehicle", "default_role": "adjacent_vehicle",
                                    "model": "vehicle.audi.a2", "initial_speed": 0.0, "distance_ahead": 22.0, "lateral_offset": 8.5}
    cfg["actor_defs"]["park_l1"] = {"canonical_id": "park_l1", "entity_type": "parked_vehicle", "default_role": "adjacent_vehicle",
                                    "model": "vehicle.audi.a2", "initial_speed": 0.0, "distance_ahead": 15.0, "lateral_offset": -8.5}
    return cfg


def test_parked_fallbacks_skip_every_slot_in_a_moving_actors_lane():
    cfg = _parked_config()
    # right side: npc_1 and npc_2 drive at +3.5 -> 3.5, 4.25, 5.5 are inside 2.5 m of that lane
    assert _parked_lateral_candidates(cfg, "park_r1", 8.5) == [8.5, 7.0]
    # left side: no moving actor -> the full same-side ladder down to the lane next to the ego
    assert _parked_lateral_candidates(cfg, "park_l1", -8.5) == [-8.5, -7.0, -5.5, -4.25, -3.5]
    # the ego lane itself (lateral 0) also keeps parked cars at least 2.5 m away
    cfg["actor_defs"]["park_r1"]["lateral_offset"] = 2.0
    assert 2.0 not in _parked_lateral_candidates(cfg, "park_r1", 2.0)


def test_parked_car_never_falls_back_into_the_npc_lane(monkeypatch):
    r = _runner()

    class ParkedNPC(r.FakeNPC):
        def __init__(self, world, initial_speed=None, lane_offset=None, **kwargs):
            if initial_speed == 0.0:
                world.parked_attempts.append(lane_offset)
                if abs(lane_offset) > 4.0:  # 8.5 and 7.0 are blocked by buildings
                    raise RuntimeError("Spawn failed because of collision at spawn position")
            super().__init__(world, initial_speed=initial_speed, lane_offset=lane_offset, **kwargs)

    class OneCandidate(_MapWorld):
        def __init__(self):
            super().__init__(right_lane=True)
            self.parked_attempts = []

        def get_scenario_waypoints(self, *args, **kwargs):
            return [_Wp(0.0, 0.0, name="only")]

    world = OneCandidate()
    with pytest.raises(ScenarioCandidateExhausted):
        spawn_cast(_parked_config(), world, dict(r.FAKE_API, NPC=ParkedNPC), placement={})
    # The ladder must stop before 5.5 -> 4.25 -> 3.5, which would park the car in npc_2's lane.
    assert world.parked_attempts == [8.5, 7.0]


# ── pass-through gate on raw telemetry ───────────────────────────────────

def _steps(park_x, npc_path):
    steps = []
    for i, npc_x in enumerate(npc_path):
        t = round(i * 0.5, 2)
        steps.append({"time_sec": t,
                      "ego": {"x": 5.0 * t, "y": 0.0, "vx": 5.0, "vy": 0.0, "speed_kmh": 18.0},
                      "npc_2": {"x": npc_x, "y": 3.5, "vx": 10.0, "vy": 0.0, "speed_kmh": 36.0},
                      "park_r1": {"x": park_x, "y": 3.5, "vx": 0.0, "vy": 0.0, "speed_kmh": 0.0}})
    return steps


def _defs():
    return {"ego": {"canonical_id": "ego", "entity_type": "ego_vehicle"},
            "npc_2": {"canonical_id": "npc_2", "entity_type": "vehicle"},
            "park_r1": {"canonical_id": "park_r1", "entity_type": "parked_vehicle"}}


def test_static_pass_through_reports_every_overlapping_step():
    raw = {"branches": {"b": _steps(22.0, [10.0, 15.0, 20.5, 26.0])},
           "_l4_steps": _steps(22.0, [40.0, 45.0])}
    hits = static_pass_through(raw, _defs())
    # bodies 4.5 m long: the first overlap is between the 0.5 s and 1.0 s samples
    # (mid-tick interpolation), centres 4.25 m apart; later samples overlap too.
    assert (hits[0]["branch"], hits[0]["time_sec"], hits[0]["mover"], hits[0]["static"]) == ("b", 0.75, "npc_2", "park_r1")
    assert hits[0]["distance_m"] == 4.25
    assert {(h["branch"], h["mover"], h["static"]) for h in hits} == {("b", "npc_2", "park_r1")}
    assert static_pass_through({"branches": {"b": _steps(60.0, [10.0, 15.0])}}, _defs()) == []


def test_run_batch_rejects_a_rollout_with_a_pass_through(tmp_path):
    from simulation.run_batch import run_batch
    cfg = {"scenario_id": "pass_through_case", "output_subdir": "scenario_x", "scenario_type": "rear_end_collision_risk",
           "intervention_time_sec": 0.0, "sample_interval_sec": 0.5, "d_max": 6, "total_duration_sec": 3.0,
           "actor_defs": _defs(), "branches": {"b": {"is_factual": True, "intervention_type": "do_ego_no_action",
                                                     "target_actor_id": "ego", "trigger_depth": 1}}}
    for spec in cfg["actor_defs"].values():
        spec.update(default_role="ego_agent" if spec["canonical_id"] == "ego" else "adjacent_vehicle",
                    model="x", initial_speed=0.0, distance_ahead=0.0, lateral_offset=0.0)

    def fn(config):
        return {"scenario": config["scenario_id"], "branches": {"b": _steps(22.0, [10.0, 15.0, 20.5, 26.0, 31.0, 36.0, 41.0])}}
    summary = run_batch([cfg], fn, str(tmp_path))
    assert summary["accepted"] == 0 and summary["failed"] == 1
    log = [json.loads(l) for l in (tmp_path / "batch_log.jsonl").read_text().splitlines()]
    assert log[-1]["status"] == "rejected_reference"
    assert "npc_2 passes through static park_r1" in log[-1]["reasons"][0]


def test_verifier_flags_a_pass_through_in_the_raw(tmp_path):
    spec = importlib.util.spec_from_file_location("test_verify_carla_dataset", Path(__file__).with_name("test_verify_carla_dataset.py"))
    vt = importlib.util.module_from_spec(spec); spec.loader.exec_module(vt)
    from scenarios.verify_carla_dataset import verify
    scenario = vt._fixture(tmp_path)
    raw = vt._load(scenario, "raw_telemetry.json")
    cfg = vt._load(scenario, "scenario_config.json")
    movers = [k for k, s in cfg["actor_defs"].items() if s.get("entity_type") == "vehicle" and k != "ego"]
    mover, parked = movers[0], movers[-1]
    cfg["actor_defs"][parked]["entity_type"] = "parked_vehicle"  # turn the last NPC into a parked car
    for step in raw["branches"]["Factual_Brake"][20:25]:
        step[mover] = dict(step[parked])  # the NPC sits on the parked car for half a second
    vt._write_json(scenario / "raw_telemetry.json", raw)
    vt._write_json(scenario / "scenario_config.json", cfg)
    report = verify(str(tmp_path), expected=1)
    assert not report["passed"]
    assert any("passes through static" in e for e in report["errors"]), report["errors"][:3]


# ── path check, map errors, oriented footprints ──────────────────────────

class _RoutePlan:
    """Straight route along +x; lateral offsets are plain y shifts."""
    route_id = "straight"
    route_mode = "single_deterministic_intended_route"
    max_s = 200.0

    def pose_at(self, s):
        return _Wp(s, 0.0, 0.0, name=f"route-{s:g}")

    def offset_location_at(self, s, lateral):
        return _Loc(float(s), float(lateral))

    def evidence(self):
        return {"coordinate_frame": "world", "source": "carla_map_intended_route",
                "route_id": self.route_id, "route_mode": self.route_mode,
                "route_centerline": [[0.0, 0.0], [200.0, 0.0]], "route_half_width_m": 8.0,
                "route_coverage_m": 200.0, "route_required_m": 100.0,
                "route_backward_coverage_m": 6.0, "route_backward_required_m": 5.0,
                "sample_spacing_m": 2.0, "route_coverage_complete": True,
                "route_backward_coverage_complete": True}


class _LaneEndsWorld(_MapWorld):
    """The right lane exists only up to x = lane_end."""

    def __init__(self, lane_end):
        super().__init__(right_lane=True)
        self.lane_end = lane_end

    def get_waypoint(self, x, y, z=0.0):
        self.right_lane = x <= self.lane_end
        return super().get_waypoint(x, y, z)

    def get_scenario_waypoints(self, *args, **kwargs):
        return [_Wp(0.0, 0.0, name="only")]

    def build_intended_route(self, wp, forward_distance, backward_distance):
        return _RoutePlan()


def test_lane_path_class_reports_the_worst_ground_along_the_span():
    world = _LaneEndsWorld(lane_end=10.0)
    cls, span, first = _lane_path_class(world, _RoutePlan(), 8.0, 30.0, 3.5)
    assert cls == "off_driving" and span == [8.0, 30.0]
    assert first is not None and 8.0 < first <= 12.0  # the first sample past the lane end
    cls, span, first = _lane_path_class(world, _RoutePlan(), 0.0, 10.0, 3.5)
    assert cls == "driving_same" and span == [0.0, 10.0] and first is None
    cls, _, first = _lane_path_class(world, _RoutePlan(), 0.0, 10.0, -3.5)
    assert cls == "driving_opposite" and first == 0.0


def test_cut_in_slot_whose_lane_ends_later_is_rejected():
    r = _runner()
    world = _LaneEndsWorld(lane_end=10.0)  # spawn at 8 m is fine, the lane ends at 10 m
    with pytest.raises(ScenarioCandidateExhausted,
                       match=r"off_driving ground from 1[02] m along the route \(checked 8 to"):
        spawn_cast(_cut_in_config(3.5), world, r.FAKE_API, placement={})
    # the budget is the whole rollout (8 s at 14 m/s + 5 m = 117 m) even for a
    # cut-in, because the L4 control rollout keeps the car in its slot
    world = _LaneEndsWorld(lane_end=8.0 + 117.0)
    placement = {}
    spawn_cast(_cut_in_config(3.5), world, r.FAKE_API, placement=placement)
    entry = placement["moving_actor_lane_checks"]["npc_1"]
    assert entry == {"class": "driving_same", "span_m": [8.0, 125.0], "travel_m": 117.0}
    with pytest.raises(ScenarioCandidateExhausted, match="off_driving"):
        spawn_cast(_cut_in_config(3.5), _LaneEndsWorld(lane_end=100.0), r.FAKE_API, placement={})


def test_lane_keeping_npc_is_checked_over_the_whole_rollout():
    r = _runner()
    cfg = _cut_in_config(3.5)
    cfg["env_behavior"]["npc_1"] = {"accel": 0.0, "lat": 0.0}
    cfg["total_duration_sec"] = 8.0
    world = _LaneEndsWorld(lane_end=8.0 + 14.0 * 3.5 + 5.0)  # enough for a cut-in, not for 8 s
    with pytest.raises(ScenarioCandidateExhausted, match="off_driving"):
        spawn_cast(cfg, world, r.FAKE_API, placement={})


def test_map_failure_is_an_error_not_unknown():
    r = _runner()

    class BrokenMap(_MapWorld):
        def get_scenario_waypoints(self, *args, **kwargs):
            return [_Wp(0.0, 0.0, name="only")]

        def get_waypoint(self, x, y, z=0.0):
            raise RuntimeError("map server offline")

    with pytest.raises(RuntimeError, match="map query failed"):
        spawn_cast(_cut_in_config(3.5), BrokenMap(), r.FAKE_API, placement={})


def test_verifier_requires_driving_same_lane_evidence(tmp_path):
    spec = importlib.util.spec_from_file_location("test_verify_carla_dataset", Path(__file__).with_name("test_verify_carla_dataset.py"))
    vt = importlib.util.module_from_spec(spec); spec.loader.exec_module(vt)
    from scenarios.verify_carla_dataset import verify
    scenario = vt._fixture(tmp_path)
    cfg = vt._load(scenario, "scenario_config.json")
    raw = vt._load(scenario, "raw_telemetry.json")
    mover = next(k for k, s in cfg["actor_defs"].items() if s.get("entity_type") == "vehicle" and k != "ego")
    cfg["actor_defs"][mover]["lateral_offset"] = 3.5
    vt._write_json(scenario / "scenario_config.json", cfg)
    # the fixture actor drives 10 m/s from route s = 8 m: record its route coordinate
    # in every stream (L3 branches, L4 treated, L4 control)
    def _streams(r):
        return list(r["branches"].values()) + [r[k] for k in ("_l4_steps", "_l4_control_steps") if r.get(k)]
    for steps in _streams(raw):
        for step in steps:
            if mover in step:
                step[mover]["route_s"] = 8.0 + 10.0 * float(step["time_sec"])
    for evidence in (None, {"class": "unknown", "span_m": [0, 0]}, {"class": "off_driving", "span_m": [0, 40]},
                     "driving_opposite"):
        raw["_placement"] = dict(raw.get("_placement") or {}, moving_actor_lane_checks=({mover: evidence} if evidence else {}))
        vt._write_json(scenario / "raw_telemetry.json", raw)
        report = verify(str(tmp_path), expected=1)
        assert any(f"moving actor {mover}" in e and "lane evidence" in e for e in report["errors"]), (evidence, report["errors"][:3])
    # structured but without a travel budget, or with a budget below what the
    # actor actually drove (10 m/s for 6.5 s = 65 m in the fixture): rejected
    for evidence in ({"class": "driving_same", "span_m": [0, 40]},
                     {"class": "driving_same", "span_m": [0, 40], "travel_m": 20.0},
                     {"class": "driving_same", "span_m": [40, 0], "travel_m": 80.0}):
        raw["_placement"]["moving_actor_lane_checks"] = {mover: evidence}
        vt._write_json(scenario / "raw_telemetry.json", raw)
        report = verify(str(tmp_path), expected=1)
        assert any(f"moving actor {mover}" in e and ("travel" in e or "span_m" in e) for e in report["errors"]), (evidence, report["errors"][:3])
    # a budget large enough but a checked span that does not enclose the route
    # coordinates the actor occupied (8-73 m): rejected
    for span in ([8, 8], [1000, 1001], [float("nan"), float("nan")], [20, 80]):
        raw["_placement"]["moving_actor_lane_checks"] = {mover: {"class": "driving_same", "span_m": span, "travel_m": 80.0}}
        vt._write_json(scenario / "raw_telemetry.json", raw)
        report = verify(str(tmp_path), expected=1)
        assert any(f"moving actor {mover}" in e and ("checked only" in e or "not a finite" in e) for e in report["errors"]), (span, report["errors"][:3])
    raw["_placement"]["moving_actor_lane_checks"] = {mover: {"class": "driving_same", "span_m": [8, 80], "travel_m": float("nan")}}
    vt._write_json(scenario / "raw_telemetry.json", raw)
    assert any("finite travel_m" in e for e in verify(str(tmp_path), expected=1)["errors"])
    raw["_placement"]["moving_actor_lane_checks"] = {mover: {"class": "driving_same", "span_m": [8, 80], "travel_m": 80.0}}
    vt._write_json(scenario / "raw_telemetry.json", raw)
    report = verify(str(tmp_path), expected=1)
    assert not any("lane evidence" in e or "travelled" in e or "checked only" in e for e in report["errors"]), report["errors"][:3]
    # without route_s telemetry the evidence cannot be tied to the road: rejected
    for steps in _streams(raw):
        for step in steps:
            if mover in step:
                step[mover].pop("route_s", None)
    vt._write_json(scenario / "raw_telemetry.json", raw)
    assert any("without a finite route_s" in e for e in verify(str(tmp_path), expected=1)["errors"])


def _oriented_raw(mover_states, parked):
    steps = []
    for i, st in enumerate(mover_states):
        steps.append({"time_sec": round(i * 0.1, 2),
                      "ego": {"x": -20.0, "y": 0.0, "vx": 10.0, "vy": 0.0, "heading_deg": 0.0, "speed_kmh": 36.0},
                      "npc_1": dict(st), "park_r2": dict(parked)})
    return {"branches": {"b": steps}}


def _defs2():
    return {"ego": {"canonical_id": "ego", "entity_type": "ego_vehicle"},
            "npc_1": {"canonical_id": "npc_1", "entity_type": "vehicle"},
            "park_r2": {"canonical_id": "park_r2", "entity_type": "parked_vehicle"}}


def test_oriented_footprints_catch_a_nose_to_tail_pass_through():
    # Recorded CARLA states, _l4_control_steps at 5.4 s: centres 3.5 m apart
    # along the lane (> 2 m centre test) but the 4.5 m bodies overlap by 1 m.
    mover = {"x": 92.433, "y": -10.601, "vx": -18.598, "vy": -0.278, "heading_deg": -179.144, "speed_kmh": 67.0}
    parked = {"x": 88.93, "y": -10.755, "vx": 0.0, "vy": 0.0, "heading_deg": 188.594, "speed_kmh": 0.0}
    hits = static_pass_through(_oriented_raw([mover], parked), _defs2())
    assert hits and hits[0]["mover"] == "npc_1" and hits[0]["static"] == "park_r2"
    assert hits[0]["distance_m"] == pytest.approx(3.51, abs=0.01)
    # side by side in adjacent lanes (2.5 m apart laterally): no overlap
    beside = dict(parked, y=parked["y"] + 2.5, vx=-10.0, heading_deg=-179.144)
    assert static_pass_through(_oriented_raw([beside], parked), _defs2()) == []


def test_mid_tick_interpolation_catches_a_jump_over_a_parked_car():
    parked = {"x": 0.0, "y": 0.0, "vx": 0.0, "vy": 0.0, "heading_deg": 0.0, "speed_kmh": 0.0}
    before = {"x": -6.0, "y": 0.0, "vx": 120.0, "vy": 0.0, "heading_deg": 0.0, "speed_kmh": 432.0}
    after = dict(before, x=6.0)
    assert static_pass_through(_oriented_raw([before], parked), _defs2()) == []
    hits = static_pass_through(_oriented_raw([before, after], parked), _defs2())
    assert hits and hits[0]["time_sec"] == 0.05 and hits[0]["distance_m"] == 0.0


def test_props_without_heading_use_the_ego_heading():
    defs = _defs2(); defs["park_r2"] = {"canonical_id": "sign_1", "entity_type": "traffic_sign"}
    mover = {"x": 0.0, "y": 0.0, "vx": 10.0, "vy": 0.0, "heading_deg": 0.0, "speed_kmh": 36.0}
    sign = {"x": 2.0, "y": 0.0, "vx": 0.0, "vy": 0.0, "heading_deg": None, "speed_kmh": 0.0}
    assert static_pass_through(_oriented_raw([mover], sign), defs)


# ── review round 3: curves, cached placement, prop heading ────────────────

class _CurveRoutePlan(_RoutePlan):
    """Outside of a curve: 1 m along the offset polyline advances the route
    coordinate by only 0.8 m, exactly what advance_along_offset returns."""

    def advance_along_offset(self, s, travel, lateral):
        return float(s) + float(travel) * 0.8


def test_lane_span_uses_the_actors_own_advance_along_offset():
    from simulation.generate import _lane_check_span
    cfg = _cut_in_config(3.5)
    start, end, travel = _lane_check_span(cfg, "npc_1", 8.0, 14.0, 3.5, _CurveRoutePlan())
    assert (start, travel) == (8.0, 117.0) and end == pytest.approx(8.0 + 117.0 * 0.8)
    start, end, travel = _lane_check_span(cfg, "npc_1", 8.0, 14.0, 3.5, _RoutePlan())  # no advance API
    assert end == pytest.approx(125.0)
    cfg["env_behavior"]["npc_1"]["accel"] = 1.0  # positive acceleration extends the budget
    assert _lane_check_span(cfg, "npc_1", 8.0, 14.0, 3.5, None)[2] == pytest.approx(117.0 + 0.5 * 64.0)


def test_curved_route_span_is_checked_where_the_actor_really_drives():
    r = _runner()

    class CurveWorld(_LaneEndsWorld):
        def build_intended_route(self, wp, forward_distance, backward_distance):
            return _CurveRoutePlan()

    # lane ends at route s = 110: the naive 8 + 117 = 125 m span would reject,
    # the actor really reaches s = 8 + 117 * 0.8 = 101.6 -> accepted
    placement = {}
    spawn_cast(_cut_in_config(3.5), CurveWorld(lane_end=110.0), r.FAKE_API, placement=placement)
    entry = placement["moving_actor_lane_checks"]["npc_1"]
    assert entry["span_m"][1] == pytest.approx(101.6) and entry["travel_m"] == 117.0
    with pytest.raises(ScenarioCandidateExhausted, match="off_driving"):
        spawn_cast(_cut_in_config(3.5), CurveWorld(lane_end=100.0), r.FAKE_API, placement={})


def test_cached_placement_that_fails_geometry_resamples_instead_of_erroring():
    r = _runner()
    world = _LaneEndsWorld(lane_end=200.0)
    placement = {}
    spawn_cast(_cut_in_config(3.5), world, r.FAKE_API, placement=placement)
    assert placement.get("waypoint") is not None
    world.lane_end = 20.0  # a later rollout re-checks the cached placement and finds the lane gone
    with pytest.raises(ScenarioCandidateExhausted, match="cached placement rejected"):
        spawn_cast(_cut_in_config(3.5), world, r.FAKE_API, placement=placement)


def test_static_prop_reports_its_spawn_heading_and_gate_ignores_the_ego_heading(monkeypatch):
    import types, sys as _sys
    fake_carla = types.SimpleNamespace(
        Location=lambda x=0, y=0, z=0: types.SimpleNamespace(x=x, y=y, z=z),
        Rotation=lambda pitch=0, yaw=0, roll=0: types.SimpleNamespace(pitch=pitch, yaw=yaw, roll=roll),
        Transform=lambda loc, rot: (loc, rot))
    monkeypatch.setitem(_sys.modules, "carla", fake_carla)
    import importlib
    import simulation.carla.static_prop as sp
    importlib.reload(sp)

    class W:
        def get_blueprint(self, m):
            return m

        def spawn_actor(self, bp, tf):
            return types.SimpleNamespace()
    prop = sp.StaticProp(W(), prop_type="street_barrier", model="x",
                         position=fake_carla.Location(2.0, 0.0, 0.0), rotation=fake_carla.Rotation(yaw=90.0))
    assert prop.get_ground_truth()["heading_deg"] == 90.0
    # gate: a 1.5 x 0.5 barrier at yaw 90 beside a car at x=0 -> overlap depends on the
    # barrier's own heading, not on the ego's heading at that tick
    defs = _defs2(); defs["park_r2"] = {"canonical_id": "wall_r", "entity_type": "guardrail"}
    car = {"x": 0.0, "y": 0.0, "vx": 10.0, "vy": 0.0, "heading_deg": 0.0, "speed_kmh": 36.0}
    barrier = {"x": 2.6, "y": 0.0, "vx": 0.0, "vy": 0.0, "heading_deg": 90.0, "speed_kmh": 0.0}
    for ego_yaw in (0.0, 90.0):
        raw = _oriented_raw([car], barrier)
        raw["branches"]["b"][0]["ego"]["heading_deg"] = ego_yaw
        assert static_pass_through(raw, defs) == []  # barrier across the lane, 0.25 m half-length: 2.6 > 2.25 + 0.25
    barrier_along = dict(barrier, heading_deg=0.0)  # along the lane: half-length 0.75 -> 2.6 < 2.25 + 0.75
    for ego_yaw in (0.0, 90.0):
        raw = _oriented_raw([car], barrier_along)
        raw["branches"]["b"][0]["ego"]["heading_deg"] = ego_yaw
        assert static_pass_through(raw, defs)


# ── review round 4: short offset polylines ────────────────────────────────

class _ShortOffsetRoutePlan(_RoutePlan):
    """Inside of a curve: the offset polyline holds only 70 % of the route
    length in physical metres; advancing beyond that raises like IntendedRoute."""

    def __init__(self, coverage):
        self.coverage = float(coverage)
        self.max_s = self.coverage

    def advance_along_offset(self, s, travel, lateral):
        if float(s) + float(travel) > self.coverage * 0.7:
            raise ValueError(f"intended route has insufficient offset coverage after {self.coverage * 0.7:.2f}m")
        return float(s) + float(travel)

    def evidence(self):
        e = super().evidence(); e["route_coverage_m"] = self.coverage; return e


def test_short_offset_polyline_extends_the_route_instead_of_recording_an_unchecked_budget():
    r = _runner()

    class ExtendingWorld(_LaneEndsWorld):
        def __init__(self):
            super().__init__(lane_end=10_000.0)
            self.built = []

        def build_intended_route(self, wp, forward_distance, backward_distance):
            self.built.append(round(forward_distance, 1))
            return _ShortOffsetRoutePlan(forward_distance)

    world = ExtendingWorld()
    placement = {}
    spawn_cast(_cut_in_config(3.5), world, r.FAKE_API, placement=placement)
    # required route ~ 8 + 14*8 + 20 = 140 m -> offset 98 m < 8 + 117 -> extended once to 210 m (147 m offset)
    assert len(world.built) == 2 and world.built[1] == pytest.approx(world.built[0] * 1.5)
    entry = placement["moving_actor_lane_checks"]["npc_1"]
    assert entry["class"] == "driving_same" and entry["span_m"] == [8.0, 125.0] and entry["travel_m"] == 117.0
    assert placement["route_evidence"]["route_coverage_m"] == pytest.approx(world.built[1])


def test_offset_coverage_that_cannot_be_extended_rejects_the_candidate():
    r = _runner()

    class StubbornWorld(_LaneEndsWorld):
        def __init__(self):
            super().__init__(lane_end=10_000.0)

        def build_intended_route(self, wp, forward_distance, backward_distance):
            return _ShortOffsetRoutePlan(30.0)  # never long enough

    with pytest.raises(ScenarioCandidateExhausted, match="offset coverage insufficient"):
        spawn_cast(_cut_in_config(3.5), StubbornWorld(), r.FAKE_API, placement={})


# ── review round 5: extension failures, lateral motion, partial route_s ───

def test_failed_route_extension_rejects_the_candidate_and_tries_the_next():
    r = _runner()

    class TwoCandidates(_LaneEndsWorld):
        def __init__(self):
            super().__init__(lane_end=10_000.0)
            self.builds = []

        def get_scenario_waypoints(self, *args, **kwargs):
            return [_Wp(0.0, 0.0, name="short_road"), _Wp(500.0, 0.0, name="long_road")]

        def build_intended_route(self, wp, forward_distance, backward_distance):
            self.builds.append((wp.name, round(forward_distance, 1)))
            if wp.name == "short_road" and forward_distance > 160.0:
                raise RuntimeError("road ends after 160 m")  # the real builder's failure
            if wp.name == "short_road":
                return _ShortOffsetRoutePlan(forward_distance)  # needs extending
            return _RoutePlan()

    world = TwoCandidates()
    placement = {}
    spawn_cast(_cut_in_config(3.5), world, r.FAKE_API, placement=placement)
    assert placement["candidate_number"] == 2
    assert [b[0] for b in world.builds] == ["short_road", "short_road", "long_road"]

    class OnlyShort(TwoCandidates):
        def get_scenario_waypoints(self, *args, **kwargs):
            return [_Wp(0.0, 0.0, name="short_road")]

    # every candidate fails deterministically -> ScenarioCandidateExhausted, which
    # run_batch records as rejected_placement (resample), never as error
    with pytest.raises(ScenarioCandidateExhausted, match="cannot be extended"):
        spawn_cast(_cut_in_config(3.5), OnlyShort(), r.FAKE_API, placement={})


class _LateralDependentRoutePlan(_RoutePlan):
    """Inside of a left curve: the -3.5 m polyline advances the route
    coordinate 0.95 m per metre, the ego lane 1.0 m per metre."""

    def advance_along_offset(self, s, travel, lateral):
        rate = 0.95 if float(lateral) < 0 else 1.0
        return float(s) + float(travel) * rate


def test_span_covers_every_lateral_the_actor_may_occupy():
    from simulation.generate import _lane_check_laterals, _lane_check_span
    cfg = _cut_in_config(-3.5)
    cfg["env_behavior"]["npc_1"] = {"accel": 0.0, "lat": 1.8}  # cuts in towards the ego lane
    assert _lane_check_laterals(cfg, "npc_1", -3.5) == [-3.5, 0.0]
    start, end, travel = _lane_check_span(cfg, "npc_1", 8.0, 14.0, -3.5, _LateralDependentRoutePlan())
    assert end == pytest.approx(8.0 + 117.0)  # the ego-lane reach, not the slot's 8 + 111.15
    keep = _cut_in_config(-3.5)
    keep["env_behavior"]["npc_1"] = {"accel": 0.0, "lat": 0.0}
    assert _lane_check_laterals(keep, "npc_1", -3.5) == [-3.5]
    assert _lane_check_span(keep, "npc_1", 8.0, 14.0, -3.5, _LateralDependentRoutePlan())[1] == pytest.approx(8.0 + 111.15)
    # a lateral motion that would cross the ego lane is clamped at it
    cross = _cut_in_config(-3.5)
    cross["env_behavior"]["npc_1"] = {"accel": 0.0, "lat": 5.0}
    assert _lane_check_laterals(cross, "npc_1", -3.5) == [-3.5, 0.0]


def _evidence_raw(route_s_values, span, travel=80.0):
    steps = []
    for i, s_val in enumerate(route_s_values):
        state = {"x": 8.0 + 10.0 * i, "y": 3.5, "vx": 10.0, "vy": 0.0, "heading_deg": 0.0, "speed_kmh": 36.0}
        if s_val is not None:
            state["route_s"] = s_val
        steps.append({"time_sec": round(i * 1.0, 1), "ego": {"x": 10.0 * i, "y": 0.0, "vx": 10.0, "vy": 0.0,
                                                             "heading_deg": 0.0, "speed_kmh": 36.0}, "npc_1": state})
    return {"branches": {"b": steps},
            "_placement": {"moving_actor_lane_checks": {"npc_1": {"class": "driving_same", "span_m": span, "travel_m": travel}}}}


def _lane_defs():
    return {"ego": {"canonical_id": "ego", "entity_type": "ego_vehicle", "lateral_offset": 0.0},
            "npc_1": {"canonical_id": "npc_1", "entity_type": "vehicle", "lateral_offset": 3.5}}


def test_partial_route_s_is_a_coverage_gap():
    from simulation.build_ground_truth import lane_evidence_gaps
    assert lane_evidence_gaps(_evidence_raw([8.0, 18.0, 28.0], [8, 30]), _lane_defs()) == []
    gaps = lane_evidence_gaps(_evidence_raw([8.0, None, float("nan")], [8, 8]), _lane_defs())
    assert any("2 tick(s) in b without a finite route_s" in g for g in gaps), gaps
    gaps = lane_evidence_gaps(_evidence_raw([8.0, 18.0, 28.0], [8, 20]), _lane_defs())
    assert any("checked only 8.0-20.0" in g for g in gaps), gaps


def test_run_batch_rejects_a_protocol_rollout_whose_lane_evidence_does_not_cover_it(tmp_path):
    from simulation.run_batch import run_batch
    cfg = {"scenario_id": "lane_cov_case", "output_subdir": "scenario_x", "scenario_type": "cut_in_collision_risk",
           "intervention_time_sec": 0.0, "sample_interval_sec": 0.5, "d_max": 6, "total_duration_sec": 3.0,
           "actor_defs": _lane_defs(), "branches": {"b": {"is_factual": True, "intervention_type": "do_ego_no_action",
                                                         "target_actor_id": "ego", "trigger_depth": 1}}}
    for spec in cfg["actor_defs"].values():
        spec.update(default_role="ego_agent" if spec["canonical_id"] == "ego" else "adjacent_vehicle",
                    model="x", initial_speed=10.0, distance_ahead=8.0 if spec["canonical_id"] != "ego" else 0.0)
    raw = _evidence_raw([8.0 + 5.0 * i for i in range(7)], [8, 20])
    raw["scenario"] = "lane_cov_case"
    for i, step in enumerate(raw["branches"]["b"]):
        step["time_sec"] = round(i * 0.5, 2)
    summary = run_batch([cfg], lambda config: raw, str(tmp_path), track="primary")
    assert summary["accepted"] == 0 and summary["failed"] == 1
    log = [json.loads(l) for l in (tmp_path / "batch_log.jsonl").read_text().splitlines()]
    assert log[-1]["status"] == "rejected_reference"
    assert "lane evidence does not cover" in log[-1]["reasons"][0]


# ── final check: road-entry cyclists start off the road by design ─────────

def test_crossing_cyclist_on_the_pavement_is_not_lane_checked():
    r = _runner()
    from simulation.build_ground_truth import lane_evidence_gaps
    cfg = _cut_in_config(3.5)
    cfg["actor_defs"]["npc_1"].update(entity_type="bicycle", default_role="crossing_actor",
                                      initial_speed=0.0, lateral_offset=-5.0)
    cfg["env_behavior"]["npc_1"] = {"accel": 0.0, "lat": 2.5}
    world = _LaneEndsWorld(lane_end=10_000.0)  # -5 m is off_driving everywhere
    placement = {}
    spawn_cast(cfg, world, r.FAKE_API, placement=placement)  # must not be rejected
    assert "npc_1" not in placement["moving_actor_lane_checks"]
    defs = {"ego": {"canonical_id": "ego", "entity_type": "ego_vehicle", "lateral_offset": 0.0},
            "npc_1": {"canonical_id": "npc_1", "entity_type": "bicycle", "default_role": "crossing_actor", "lateral_offset": -5.0}}
    assert lane_evidence_gaps({"branches": {"b": []}, "_placement": {}}, defs) == []
