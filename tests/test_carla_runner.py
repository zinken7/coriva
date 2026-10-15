"""
Config-driven CARLA runner (`simulation/generate.py`).

`run_scenario(config, world, api)` must consume the unified scenario config
(single source of truth) and drive the CARLA wrappers generically. CARLA
itself cannot run here, so the runner takes an injected
`api` of wrapper classes and these tests drive it with fakes — verifying the
config→spawn dispatch, the environment-event and intervention scheduling, and
the telemetry contract that build_ground_truth.py consumes.
"""

import copy
import os
import sys
import types

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from simulation.generate import (  # noqa: E402
    INTERVENTION_TO_EGO_ACTION, _default_api, _required_cast_route_distance,
    _required_route_distance, _validate_protocol_preflight,
    ScenarioCandidateExhausted, run_scenario, spawn_cast)


def test_default_api_loads_coriva_wrapper_namespace(monkeypatch):
    module = types.ModuleType("simulation.carla")
    names = (
        "World", "Ego", "EgoActions", "NPC", "NPCActions", "Pedestrian",
        "PedestrianActions", "StaticProp", "TrajectoryRecorder",
        "CollisionDetector",
    )
    wrappers = {name: type(name, (), {}) for name in names}
    for name, value in wrappers.items():
        setattr(module, name, value)
    monkeypatch.setitem(sys.modules, "simulation.carla", module)

    assert _default_api() == wrappers


# ── Fakes ────────────────────────────────────────────────────────────────

class FakeWaypoint:
    def next(self, dist):
        return [self]

    @property
    def transform(self):
        class T:
            class location:
                x, y, z = 0.0, 0.0, 0.0

            class rotation:
                @staticmethod
                def get_right_vector():
                    class V:
                        x, y, z = 0.0, 1.0, 0.0
                    return V()
        return T()


class NamedWaypoint(FakeWaypoint):
    def __init__(self, name):
        self.name = name


class FakeWorld:
    def __init__(self):
        self.ticks = 0
        self.cleanups = 0

    def get_waypoint(self, x, y, z=0.0):
        return FakeWaypoint()

    def tick(self):
        self.ticks += 1

    def cleanup(self):
        self.cleanups += 1


class FakeActor:
    def __init__(self, world, **kwargs):
        self.kwargs = kwargs
        self.updates = 0
        self.carla_actor = object()

    def update(self, dt):
        self.updates += 1

    def get_ground_truth(self):
        return {"x": 0.0, "y": 0.0}


class FakeEgo(FakeActor):
    pass


class FakeNPC(FakeActor):
    pass


class FakePedestrian(FakeActor):
    pass


class FakeStaticProp(FakeActor):
    def update(self, dt):  # static: never stepped
        raise AssertionError("static props must not be updated")


class FakeActions:
    def __init__(self, actor):
        self.actor = actor
        self.calls = []

    def __getattr__(self, name):
        def record(*a, **k):
            self.calls.append((name, k or a))
        return record


class FakeRecorder:
    def __init__(self, world, attach_to=None, output_dir=None, fov=100, **kwargs):
        self.steps = []
        self.collisions = []
        self.camera_kwargs = {"fov": fov, **kwargs}

    def start(self):
        pass

    def wait_for_frame(self, timeout=2.0):
        return 1

    def record_step(self, t, branch, **actors):
        self.steps.append((t, branch, sorted(actors.keys())))

    def record_collision(self, t, branch, extra_frames=5, partners=None):
        self.collisions.append((t, branch, list(partners or [])))

    def get_trajectory(self):
        steps = [{"t": t, "branch": b} for (t, b, _a) in self.steps]
        for t, _branch, partners in self.collisions:
            hit = next(step for step in steps if step["t"] == t)
            hit.update(collision=True, collision_time=t, collision_partners=partners)
        return steps


class FakeDetector:
    def __init__(self, a, b, **kwargs):
        pass

    def check(self):
        return False


FAKE_API = {
    "Ego": FakeEgo, "NPC": FakeNPC, "Pedestrian": FakePedestrian,
    "StaticProp": FakeStaticProp, "EgoActions": FakeActions,
    "NPCActions": FakeActions, "PedestrianActions": FakeActions,
    "TrajectoryRecorder": FakeRecorder, "CollisionDetector": FakeDetector,
}


def _config():
    return {
        "scenario_id": "rear_end_0001",
        "output_subdir": "scenario_1",
        "scenario_type": "rear_end_collision_risk",
        "map": "Town04",
        "seed": 7,
        "actor_defs": {
            "ego": {"canonical_id": "ego", "entity_type": "ego_vehicle",
                    "default_role": "ego_agent", "model": "vehicle.tesla.model3",
                    "initial_speed": 20.0, "distance_ahead": 0.0, "lateral_offset": 0.0},
            "npc_1": {"canonical_id": "npc_1", "entity_type": "vehicle",
                      "default_role": "lead_vehicle", "collision_role": "impacted_actor",
                      "model": "vehicle.lincoln.mkz_2020", "initial_speed": 18.0,
                      "distance_ahead": 40.0, "lateral_offset": 0.0},
            "ped_1": {"canonical_id": "ped_1", "entity_type": "pedestrian",
                      "default_role": "crossing_pedestrian", "model": "walker.pedestrian.0001",
                      "initial_speed": 1.6, "distance_ahead": 60.0, "lateral_offset": -6.0},
            "wall_r": {"canonical_id": "wall_r", "entity_type": "guardrail",
                       "default_role": "road_boundary", "model": "static.prop.streetbarrier",
                       "initial_speed": 0.0, "distance_ahead": 20.0, "lateral_offset": 7.0},
        },
        "env_behavior": {
            "npc_1": {"accel": -6.0, "lat": 0.0},
            "ped_1": {"accel": 0.0, "lat": 1.6},
            "wall_r": {"accel": 0.0, "lat": 0.0},
        },
        "environment_event": {"event_type": "npc_sudden_brake",
                              "trigger_time_sec": 1.0, "params": {"intensity": 6.0}},
        "intervention_time_sec": 1.5,
        "sample_interval_sec": 0.45,
        "d_max": 10,
        "total_duration_sec": 3.0,
        "branches": {
            "Factual_AEB": {"is_factual": True, "intervention_type": "do_ego_emergency_brake",
                            "target_actor_id": "ego", "trigger_depth": 1},
            "CF_NoAction": {"is_factual": False, "intervention_type": "do_ego_no_action",
                            "target_actor_id": "ego", "trigger_depth": 1},
        },
    }


# ── Spawn dispatch ───────────────────────────────────────────────────────

def test_spawn_dispatches_by_entity_type(tmp_path):
    world = FakeWorld()
    actors, controllers, dynamic = spawn_cast(_config(), world, FAKE_API)
    assert isinstance(actors["ego"], FakeEgo)
    assert isinstance(actors["npc_1"], FakeNPC)
    assert isinstance(actors["ped_1"], FakePedestrian)
    assert isinstance(actors["wall_r"], FakeStaticProp)
    # static actors are not in the dynamic (update/collision) set
    assert "wall_r" not in dynamic and "ego" in dynamic and "npc_1" in dynamic


def test_spawn_passes_config_parameters(tmp_path):
    actors, _c, _d = spawn_cast(_config(), FakeWorld(), FAKE_API)
    assert actors["npc_1"].kwargs["initial_speed"] == 18.0
    assert actors["npc_1"].kwargs["distance_ahead"] == 40.0
    assert actors["npc_1"].kwargs["model"] == "vehicle.lincoln.mkz_2020"
    # pedestrian side/offset derived from the signed lateral offset
    assert actors["ped_1"].kwargs["side"] == "left"
    assert actors["ped_1"].kwargs["offset"] == 6.0


def test_spawn_collision_cleans_partial_cast_and_retries_next_waypoint():
    class RetryWorld(FakeWorld):
        def __init__(self):
            super().__init__()
            self.waypoints = [NamedWaypoint("blocked"), NamedWaypoint("clear")]
            self.npc_attempts = []

        def get_scenario_waypoints(self, *args, **kwargs):
            return self.waypoints

    class CollisionThenNPC(FakeNPC):
        def __init__(self, world, waypoint=None, **kwargs):
            world.npc_attempts.append(waypoint.name)
            if waypoint.name == "blocked":
                raise RuntimeError(
                    "Spawn failed because of collision at spawn position")
            super().__init__(world, waypoint=waypoint, **kwargs)

    world = RetryWorld()
    api = dict(FAKE_API, NPC=CollisionThenNPC)
    placement = {}

    actors, _controllers, _dynamic = spawn_cast(
        _config(), world, api, placement=placement)

    assert actors["npc_1"].kwargs["waypoint"].name == "clear"
    assert world.npc_attempts[:2] == ["blocked", "clear"]
    assert world.cleanups == 1
    assert placement == {
        "waypoint": world.waypoints[1],
        "candidate_number": 2,
        "context_lateral_offsets": {"wall_r": 7.0},
        "moving_actor_lane_checks": {},
    }

    # Other rollouts of the scenario reuse the selected base directly.
    world.npc_attempts.clear()
    spawn_cast(_config(), world, api, placement=placement)
    assert world.npc_attempts == ["clear"]
    assert placement["candidate_number"] == 2


def test_spawn_retries_candidate_with_incomplete_map_route():
    class RouteWorld(FakeWorld):
        def __init__(self):
            super().__init__()
            self.waypoints = [NamedWaypoint("truncated"), NamedWaypoint("covered")]
            self.route_calls = []

        def get_scenario_waypoints(self, *args, **kwargs):
            return self.waypoints

        def sample_route_evidence(self, waypoint, forward_distance):
            self.route_calls.append((waypoint.name, forward_distance))
            if waypoint.name == "truncated":
                raise RuntimeError("route evidence incomplete: synthetic dead end")
            return {
                "coordinate_frame": "world",
                "source": "carla_map_waypoint_route",
                "route_centerline": [[0.0, 0.0], [forward_distance, 0.0]],
                "route_lane_width_m": 3.5,
                "route_coverage_m": forward_distance,
                "route_required_m": forward_distance,
            }

    world = RouteWorld()
    placement = {}

    actors, _controllers, _dynamic = spawn_cast(
        _config(), world, FAKE_API, placement=placement)

    assert actors["ego"].kwargs["waypoint"].name == "covered"
    assert [name for name, _distance in world.route_calls] == [
        "truncated", "covered"]
    assert placement["candidate_number"] == 2
    assert placement["route_evidence"]["source"] == "carla_map_waypoint_route"


def test_spawn_can_resume_from_the_next_deterministic_candidate():
    class CandidateWorld(FakeWorld):
        def __init__(self):
            super().__init__()
            self.waypoints = [NamedWaypoint(name) for name in ("one", "two", "three")]

        def get_scenario_waypoints(self, *args, **kwargs):
            return self.waypoints

    world = CandidateWorld()
    placement = {"candidate_offset": 2}
    actors, _controllers, _dynamic = spawn_cast(
        _config(), world, FAKE_API, placement=placement)

    assert actors["ego"].kwargs["waypoint"].name == "three"
    assert placement["candidate_number"] == 3


def test_spawn_reports_exhausted_preflight_candidates_as_sample_reject():
    class CandidateWorld(FakeWorld):
        def get_scenario_waypoints(self, *args, **kwargs):
            return [NamedWaypoint("one"), NamedWaypoint("two")]

    with pytest.raises(ScenarioCandidateExhausted,
                       match="2/2 preflight rejection"):
        spawn_cast(
            _config(), CandidateWorld(), FAKE_API,
            placement={"candidate_offset": 2})


def test_context_vehicle_uses_and_caches_same_side_lateral_fallback():
    class ContextWorld(FakeWorld):
        def __init__(self):
            super().__init__()
            self.parked_attempts = []

    class ContextNPC(FakeNPC):
        def __init__(self, world, initial_speed=None, lane_offset=None, **kwargs):
            if initial_speed == 0.0:
                world.parked_attempts.append(lane_offset)
                if abs(lane_offset) > 4.25:
                    raise RuntimeError(
                        "Spawn failed because of collision at spawn position")
            super().__init__(
                world, initial_speed=initial_speed, lane_offset=lane_offset,
                **kwargs)

    config = _config()
    config["actor_defs"]["parked"] = {
        "canonical_id": "parked", "entity_type": "parked_vehicle",
        "default_role": "adjacent_vehicle", "model": "vehicle.audi.a2",
        "initial_speed": 0.0, "distance_ahead": 55.0,
        "lateral_offset": -8.5,
    }
    world = ContextWorld()
    placement = {}
    api = dict(FAKE_API, NPC=ContextNPC)

    # ped_1 walks at lateral -6.0 m: the parked fallbacks
    # skip every slot within (widths/2 + 0.5 m) of a moving actor's lane on
    # the same side, so -7.0/-5.5/-4.25 are never tried and the first clear
    # fallback after the blocked -8.5 is -3.5 (3.5 m from the ego lane).
    spawn_cast(config, world, api, placement=placement)
    assert world.parked_attempts == [-8.5, -3.5]
    assert placement["context_lateral_offsets"]["parked"] == -3.5

    world.parked_attempts.clear()
    spawn_cast(config, world, api, placement=placement)
    assert world.parked_attempts == [-3.5]


def test_parked_vehicle_uses_route_pose_without_runtime_route_binding():
    class RoutePlan:
        route_id = "shared-route"
        route_mode = "single_deterministic_intended_route"

        def pose_at(self, distance):
            return NamedWaypoint(f"route-{distance:g}")

        def evidence(self):
            return {
                "coordinate_frame": "world",
                "source": "carla_map_intended_route",
                "route_id": self.route_id,
                "route_mode": self.route_mode,
                "route_centerline": [[0.0, 0.0], [300.0, 0.0]],
                "route_half_width_m": 8.0,
                "route_coverage_m": 300.0,
                "route_required_m": 300.0,
            }

    class RouteWorld(FakeWorld):
        def __init__(self):
            super().__init__()
            self.plan = RoutePlan()

        def build_intended_route(self, _waypoint, **_kwargs):
            return self.plan

    config = _config()
    config["actor_defs"]["parked"] = {
        "canonical_id": "parked", "entity_type": "parked_vehicle",
        "default_role": "adjacent_vehicle", "model": "vehicle.audi.a2",
        "initial_speed": 0.0, "distance_ahead": 55.0,
        "lateral_offset": -8.5,
    }
    world = RouteWorld()
    actors, _controllers, dynamic = spawn_cast(
        config, world, FAKE_API, placement={})

    parked = actors["parked"]
    assert parked.kwargs["waypoint"].name == "route-55"
    assert parked.kwargs["distance_ahead"] == 0.0
    assert "route_plan" not in parked.kwargs
    assert "parked" not in dynamic
    assert actors["npc_1"].kwargs["route_plan"] is world.plan


def test_context_static_prop_uses_lateral_fallback(monkeypatch):
    class ContextWorld(FakeWorld):
        def __init__(self):
            super().__init__()
            self.prop_attempts = []

    class ContextProp(FakeStaticProp):
        def __init__(self, world, position=None, **kwargs):
            world.prop_attempts.append(position)
            if abs(position) > 5.5:
                raise RuntimeError(
                    "Spawn failed because of collision at spawn position")
            super().__init__(world, position=position, **kwargs)

    monkeypatch.setattr(
        "simulation.generate._offset_location", lambda _waypoint, lateral: lateral)
    world = ContextWorld()
    placement = {}
    spawn_cast(
        _config(), world, dict(FAKE_API, StaticProp=ContextProp),
        placement=placement)

    assert world.prop_attempts == [7.0, 5.5]
    assert placement["context_lateral_offsets"]["wall_r"] == 5.5


def test_spawn_does_not_retry_or_mask_non_collision_error():
    class RetryWorld(FakeWorld):
        def __init__(self):
            super().__init__()
            self.waypoints = [NamedWaypoint("one"), NamedWaypoint("two")]
            self.npc_attempts = []

        def get_scenario_waypoints(self, *args, **kwargs):
            return self.waypoints

    class BrokenNPC(FakeNPC):
        def __init__(self, world, waypoint=None, **kwargs):
            world.npc_attempts.append(waypoint.name)
            raise RuntimeError("blueprint missing")

    world = RetryWorld()
    with pytest.raises(RuntimeError, match="blueprint missing"):
        spawn_cast(_config(), world, dict(FAKE_API, NPC=BrokenNPC))
    assert world.npc_attempts == ["one"]
    assert world.cleanups == 1


# ── Scheduling: environment event + per-branch intervention ─────────────

def _run(config, branch):
    world = FakeWorld()
    return run_scenario(config, world, FAKE_API, output_dir="/tmp/coriva_test_out",
                        only_branches=[branch]), world


def test_env_behavior_applied_at_trigger():
    result, _w = _run(_config(), "CF_NoAction")
    calls = result["_controllers"]["CF_NoAction"]["npc_1"].calls
    assert ("brake", {"deceleration": 6.0}) in calls
    ped_calls = result["_controllers"]["CF_NoAction"]["ped_1"].calls
    assert any(name == "start_crossing" for name, _ in ped_calls)


def test_vehicle_lateral_behavior_uses_declared_direction():
    config = _config()
    config["env_behavior"]["npc_1"] = {"accel": 0.0, "lat": -1.75}
    result, _w = _run(config, "CF_NoAction")
    calls = result["_controllers"]["CF_NoAction"]["npc_1"].calls
    assert ("steer_left", {"intensity": pytest.approx(0.5)}) in calls


def test_parked_vehicle_is_a_non_updating_npc():
    config = _config()
    config["actor_defs"]["parked"] = {
        "canonical_id": "parked", "entity_type": "parked_vehicle",
        "default_role": "adjacent_vehicle", "model": "vehicle.audi.a2",
        "initial_speed": 0.0, "distance_ahead": 25.0, "lateral_offset": 5.0,
    }
    actors, _controllers, dynamic = spawn_cast(config, FakeWorld(), FAKE_API)
    assert isinstance(actors["parked"], FakeNPC)
    assert "parked" not in dynamic


def test_branch_intervention_applied_to_ego():
    result, _w = _run(_config(), "Factual_AEB")
    ego_calls = result["_controllers"]["Factual_AEB"]["ego"].calls
    assert any(name == "brake" for name, _ in ego_calls), ego_calls


def test_no_action_branch_never_brakes_ego():
    result, _w = _run(_config(), "CF_NoAction")
    ego_calls = result["_controllers"]["CF_NoAction"]["ego"].calls
    assert not any(name == "brake" for name, _ in ego_calls), ego_calls


def test_l4_paired_control_removes_only_named_actor_intervention(monkeypatch):
    """Admission control must preserve the rest of the traffic scene."""
    monkeypatch.setattr("simulation.generate.time.sleep", lambda _seconds: None)
    config = _config()
    config["l4_query"] = {
        "intervention_type": "npc_sudden_brake",
        "target_actor_id": "npc_1",
        "trigger_depth": 1,
        "ego_policy": "do_ego_no_action",
    }
    result = run_scenario(config, FakeWorld(), FAKE_API,
                          output_dir="/tmp/coriva_test_out")

    treated = result["_controllers"]["__L4_FIXED_EGO__"]
    control = result["_controllers"]["__L4_NO_INTERVENTION__"]
    assert any(name == "brake" for name, _ in treated["npc_1"].calls)
    assert not any(name == "brake" for name, _ in control["npc_1"].calls)
    assert any(name == "start_crossing" for name, _ in treated["ped_1"].calls)
    assert any(name == "start_crossing" for name, _ in control["ped_1"].calls)


# ── Telemetry contract for build_ground_truth ────────────────────────────

def test_scenario_raw_shape_covers_all_branches():
    world = FakeWorld()
    raw = run_scenario(_config(), world, FAKE_API, output_dir="/tmp/coriva_test_out")
    assert raw["scenario"] == "rear_end_0001"
    assert set(raw["branches"].keys()) == {"Factual_AEB", "CF_NoAction"}
    for steps in raw["branches"].values():
        assert len(steps) > 0
    assert world.cleanups == len(raw["branches"])  # cleanup after every branch
    assert raw["_placement"]["candidate_number"] == 1
    assert raw["_placement"]["context_lateral_offsets_m"] == {"wall_r": 7.0}


def test_scenario_raw_carries_one_branch_independent_map_route():
    class RouteWorld(FakeWorld):
        def __init__(self):
            super().__init__()
            self.route_calls = 0

        def sample_route_evidence(self, _waypoint, forward_distance):
            self.route_calls += 1
            return {
                "coordinate_frame": "world",
                "source": "carla_map_waypoint_route",
                "route_centerline": [[0.0, 0.0], [forward_distance, 0.0]],
                "route_lane_width_m": 3.5,
                "route_coverage_m": forward_distance,
                "route_required_m": forward_distance,
            }

    config = _config()
    config["branches"]["CF_Swerve"] = {
        "is_factual": False,
        "intervention_type": "do_ego_swerve_left",
        "target_actor_id": "ego",
        "trigger_depth": 1,
    }
    config["l4_query"] = {
        "intervention_type": "npc_sudden_brake",
        "target_actor_id": "npc_1",
        "trigger_depth": 1,
        "ego_policy": "do_ego_no_action",
    }
    world = RouteWorld()
    raw = run_scenario(config, world, FAKE_API,
                       output_dir="/tmp/coriva_test_out")

    assert world.route_calls == 1
    assert world.cleanups == 5  # 3 L3 + L4 treated + paired L4 control
    assert set(raw["branches"]) == {"Factual_AEB", "CF_NoAction", "CF_Swerve"}
    assert raw["_l4_steps"]
    assert raw["_l4_control_steps"]
    assert raw["_route_evidence"]["source"] == "carla_map_waypoint_route"
    assert raw["_route_evidence"]["route_centerline"][0] == [0.0, 0.0]


def test_one_intended_route_object_is_shared_by_all_vehicle_rollouts():
    class RoutePlan:
        route_id = "shared-route"
        route_mode = "single_deterministic_intended_route"

        def pose_at(self, _distance):
            return FakeWaypoint()

        def evidence(self):
            return {
                "coordinate_frame": "world",
                "source": "carla_map_intended_route",
                "route_id": self.route_id,
                "route_mode": self.route_mode,
                "route_centerline": [[0.0, 0.0], [300.0, 0.0]],
                "route_lane_width_m": 3.5,
                "route_half_width_m": 8.0,
                "route_coverage_m": 300.0,
                "route_required_m": 300.0,
            }

    class RouteWorld(FakeWorld):
        def __init__(self):
            super().__init__()
            self.plan = RoutePlan()
            self.route_requests = []

        def build_intended_route(self, _waypoint, **_kwargs):
            self.route_requests.append(_kwargs)
            return self.plan

    plans = []

    class RecordingEgo(FakeEgo):
        def __init__(self, world, route_plan=None, **kwargs):
            plans.append(route_plan)
            super().__init__(world, route_plan=route_plan, **kwargs)

    class RecordingNPC(FakeNPC):
        def __init__(self, world, route_plan=None, **kwargs):
            plans.append(route_plan)
            super().__init__(world, route_plan=route_plan, **kwargs)

    config = _config()
    config["branches"]["CF_Swerve"] = {
        "is_factual": False,
        "intervention_type": "do_ego_swerve_left",
        "target_actor_id": "ego",
        "trigger_depth": 1,
    }
    config["l4_query"] = {
        "intervention_type": "npc_sudden_brake",
        "target_actor_id": "npc_1",
        "trigger_depth": 1,
        "ego_policy": "do_ego_no_action",
    }
    world = RouteWorld()
    run_scenario(
        config, world, dict(FAKE_API, Ego=RecordingEgo, NPC=RecordingNPC),
        output_dir="/tmp/coriva_test_out")

    assert len(world.route_requests) == 1
    assert world.route_requests[0]["forward_distance"] == pytest.approx(
        _required_cast_route_distance(config))
    assert world.route_requests[0]["backward_distance"] == pytest.approx(5.0)
    assert len(plans) == 10  # ego + one road NPC in each of five rollouts
    assert all(plan is world.plan for plan in plans)


def test_required_route_distance_includes_l4_ego_acceleration():
    config = _config()
    config["actor_defs"]["ego"]["initial_speed"] = 10.0
    config["total_duration_sec"] = 8.0
    config["intervention_time_sec"] = 3.5
    config["l4_query"] = {
        "ego_policy": "do_ego_accelerate_escape",
        "target_actor_id": "npc_1",
    }

    # 80 m constant-speed travel + acceleration over the 4.5 s post-T0
    # interval + the documented 20 m route-coverage margin.
    expected = 80.0 + 0.5 * 3.5 * 4.5 ** 2 + 20.0
    assert _required_route_distance(config) == pytest.approx(expected)


def test_cast_route_distance_covers_furthest_accelerating_vehicle():
    config = _config()
    config["total_duration_sec"] = 8.0
    config["environment_event"]["trigger_time_sec"] = 2.0
    config["actor_defs"]["npc_1"]["distance_ahead"] = 80.0
    config["actor_defs"]["npc_1"]["initial_speed"] = 12.0
    config["env_behavior"]["npc_1"]["accel"] = 2.0

    expected = 80.0 + 12.0 * 8.0 + 0.5 * 2.0 * 6.0 ** 2 + 20.0
    assert _required_cast_route_distance(config) == pytest.approx(expected)


def test_run_branch_cleans_world_when_spawn_cast_fails(monkeypatch):
    world = FakeWorld()

    def fail_spawn(*_args, **_kwargs):
        raise RuntimeError("spawn setup failed")

    monkeypatch.setattr("simulation.generate.spawn_cast", fail_spawn)
    with pytest.raises(RuntimeError, match="spawn setup failed"):
        run_scenario(_config(), world, FAKE_API,
                     output_dir="/tmp/coriva_test_out",
                     only_branches=["CF_NoAction"])
    assert world.cleanups == 1


def test_all_l3_branches_record_t0_before_future_rollout():
    world = FakeWorld()
    config = _config()
    raw = run_scenario(config, world, FAKE_API, output_dir="/tmp/coriva_test_out")
    t0 = config["intervention_time_sec"]
    for steps in raw["branches"].values():
        assert any(s["t"] == t0 for s in steps)


def test_trajectory_timestamps_start_before_any_update_and_include_duration(monkeypatch):
    monkeypatch.setattr("simulation.generate.time.sleep", lambda _seconds: None)
    config = _config()
    raw = run_scenario(config, FakeWorld(), FAKE_API,
                       output_dir="/tmp/coriva_test_out",
                       only_branches=["CF_NoAction"])
    steps = raw["branches"]["CF_NoAction"]
    assert steps[0]["t"] == 0.0
    assert steps[-1]["t"] == config["total_duration_sec"]


def test_intervention_map_covers_all_generator_interventions():
    """Every intervention_type the config generator can emit must have an
    ego-action mapping, and the magnitudes must match the twin's EGO_PROFILES
    (cross-split consistency)."""
    from simulation.kinematic_twin import EGO_PROFILES
    for itype, prof in EGO_PROFILES.items():
        assert itype in INTERVENTION_TO_EGO_ACTION, itype
        action, params = INTERVENTION_TO_EGO_ACTION[itype]
        if prof["accel"] < 0:
            assert params.get("deceleration") == pytest.approx(-prof["accel"]), itype
        elif prof["accel"] > 0:
            assert params.get("acceleration") == pytest.approx(prof["accel"]), itype
        if prof["lat"]:
            assert action in {
                "steer_left", "steer_right",
                "brake_and_steer_left", "brake_and_steer_right",
            }
            assert 3.5 * params["intensity"] == pytest.approx(abs(prof["lat"])), itype


def _protocol_trajectory(*, invisible_at=None, collision_time=None, omit_time=None):
    steps = []
    for index in range(36):
        t = round(index * 0.1, 2)
        if omit_time is not None and t == pytest.approx(omit_time):
            continue
        bbox = ([0.0, 0.0, 20.0, 20.0]
                if invisible_at is None or t != pytest.approx(invisible_at)
                else [0.0, 0.0, 5.0, 5.0])
        step = {
            "time_sec": t,
            "ego": {"x": t, "y": 0.0},
            "npc_1": {"x": t + 10.0, "y": 0.0, "bbox_2d": bbox},
        }
        if collision_time is not None and t == pytest.approx(collision_time):
            step["collision"] = True
            step["collision_time"] = collision_time
        steps.append(step)
    return steps


def _protocol_config():
    config = copy.deepcopy(_config())
    config.update({
        "template_id": "lead_car_brake",
        "critical_actor_id": "npc_1",
        "intervention_time_sec": 3.5,
    })
    return config


def test_protocol_preflight_accepts_exact_visible_collision_free_history():
    ok, reasons = _validate_protocol_preflight(
        _protocol_trajectory(), _protocol_config())
    assert ok is True, reasons


def test_protocol_preflight_rejects_history_outside_intended_route():
    route = {
        "coordinate_frame": "world",
        "route_centerline": [[50.0, 0.0], [100.0, 0.0]],
        "route_lane_width_m": 3.5,
    }
    ok, reasons = _validate_protocol_preflight(
        _protocol_trajectory(), _protocol_config(), route_evidence=route)

    assert ok is False
    assert any("intended route does not cover ego" in reason
               for reason in reasons)


def test_route_provenance_is_not_exported_in_model_visible_history():
    from simulation.build_ground_truth import build_observed_inputs

    trajectory = _protocol_trajectory()
    for step in trajectory:
        for actor_id in ("ego", "npc_1"):
            step[actor_id].update({
                "route_s": step["time_sec"],
                "route_id": "0123456789abcdef",
                "route_mode": "single_deterministic_intended_route",
            })
    raw = {"branches": {"Factual_AEB": trajectory}}
    root, history, _reference = build_observed_inputs(
        raw, _protocol_config())

    forbidden = {"route_s", "route_id", "route_mode"}
    assert all(not forbidden.intersection(actor) for actor in root["actors"])
    assert all(
        not forbidden.intersection(actor)
        for record in history["records"]
        for actor in record["actors"]
    )


@pytest.mark.parametrize(
    ("trajectory", "message"),
    [
        (_protocol_trajectory(omit_time=3.0), "missing T=-0.5s"),
        (_protocol_trajectory(invisible_at=2.5), "not visible at T=-1.0s"),
        (_protocol_trajectory(collision_time=3.0), "collision at or before T0"),
    ],
)
def test_protocol_preflight_rejects_bad_history_before_other_rollouts(
        trajectory, message):
    ok, reasons = _validate_protocol_preflight(trajectory, _protocol_config())
    assert ok is False
    assert any(message in reason for reason in reasons)


def test_protocol_run_retries_waypoint_before_counterfactual_rollouts(
        tmp_path, monkeypatch):
    config = _protocol_config()
    config["output_subdir"] = "scenario_preflight"
    calls = []

    def fake_run_branch(config_arg, branch_name, _world, _api, output_dir,
                        placement=None):
        if placement.get("waypoint") is None:
            candidate = int(placement.get("candidate_offset", 0)) + 1
            placement.update({
                "waypoint": NamedWaypoint(f"candidate-{candidate}"),
                "candidate_number": candidate,
                "context_lateral_offsets": {},
            })
        candidate = placement["candidate_number"]
        calls.append((branch_name, candidate))
        trajectory = _protocol_trajectory(
            invisible_at=2.5 if branch_name == "Factual_AEB" and candidate == 1
            else None)
        branch_dir = tmp_path / config_arg["output_subdir"] / branch_name
        branch_dir.mkdir(parents=True, exist_ok=True)
        for index, step in enumerate(trajectory):
            step["camera_frame"] = 1000 + index
            (branch_dir / f"{1000 + index:06d}.png").write_bytes(b"png")
        return trajectory, {}

    monkeypatch.setattr("simulation.generate.run_branch", fake_run_branch)
    result = run_scenario(
        config, FakeWorld(), FAKE_API, output_dir=str(tmp_path))

    assert calls == [
        ("Factual_AEB", 1),
        ("Factual_AEB", 2),
        ("CF_NoAction", 2),
    ]
    assert result["_placement"]["candidate_number"] == 2


# ── Collision partner recording ────────────────────────────────────────

def test_runner_records_every_collision_partner_at_the_hit(monkeypatch):
    """The recorder must store which actors overlapped the ego, not just
    that some detector fired; the reference builder needs the partner."""
    monkeypatch.setattr("simulation.generate.time.sleep", lambda _seconds: None)

    class HitDetector:
        def __init__(self, a, b, **kwargs):
            self.a, self.b = a, b

        def check(self):
            # The lead vehicle (and only it) overlaps the ego from tick 5 on.
            return self.b.kwargs.get("distance_ahead") == 40.0 and self.a.world.ticks >= 5

    class WorldEgo(FakeEgo):
        def __init__(self, world, **kwargs):
            super().__init__(world, **kwargs)
            self.world = world

    api = dict(FAKE_API, CollisionDetector=HitDetector, Ego=WorldEgo)
    world = FakeWorld()
    raw = run_scenario(_config(), world, api, output_dir="/tmp/coriva_test_out",
                       only_branches=["Factual_AEB"])
    steps = raw["branches"]["Factual_AEB"]
    hits = [step for step in steps if step.get("collision")]
    assert len(hits) == 1 and hits[-1] is steps[-1], "rollout stops at the hit"
    assert hits[0]["collision_partners"] == ["npc_1"]
