"""CARLA-wrapper geometry, weather, and camera tests without a CARLA install."""

from __future__ import annotations

import importlib.util
import math
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest


ROOT = Path(__file__).resolve().parents[1]


class Vector:
    def __init__(self, x=0.0, y=0.0, z=0.0):
        self.x, self.y, self.z = x, y, z


class Location(Vector):
    pass


class Rotation:
    def __init__(self, pitch=0.0, yaw=0.0, roll=0.0):
        self.pitch, self.yaw, self.roll = pitch, yaw, roll

    def get_right_vector(self):
        radians = math.radians(self.yaw)
        return Vector(-math.sin(radians), math.cos(radians), 0.0)

    def get_forward_vector(self):
        radians = math.radians(self.yaw)
        return Vector(math.cos(radians), math.sin(radians), 0.0)


class Transform:
    def __init__(self, location=None, rotation=None):
        self.location = location or Location()
        self.rotation = rotation or Rotation()


class VehicleLightState(int):
    pass


VehicleLightState.Position = 1
VehicleLightState.Brake = 2
VehicleLightState.NONE = 0


def _fake_carla():
    return SimpleNamespace(
        Location=Location,
        Rotation=Rotation,
        Transform=Transform,
        Vector3D=Vector,
        VehicleLightState=VehicleLightState,
        WeatherParameters=SimpleNamespace(ClearNoon=object(), HardRainNoon=object()),
    )


def _load(monkeypatch, relative_path, module_name):
    monkeypatch.setitem(sys.modules, "carla", _fake_carla())
    spec = importlib.util.spec_from_file_location(module_name, ROOT / relative_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


class SpawnActor:
    def __init__(self):
        self.physics = None
        self.transform = None
        self.light_state = None

    def set_simulate_physics(self, value):
        self.physics = value

    def set_transform(self, transform):
        self.transform = transform

    def set_light_state(self, state):
        self.light_state = state


class SpawnWorld:
    def __init__(self):
        self.transform = None

    def get_blueprint(self, name):
        return name

    def spawn_actor(self, _blueprint, transform):
        self.transform = transform
        return SpawnActor()


class Waypoint:
    def __init__(self, x=0.0, y=0.0, yaw=0.0, lane_width=3.5):
        self.transform = Transform(Location(x, y, 0.0), Rotation(yaw=yaw))
        self.lane_id = 1
        self.lane_width = lane_width
        self.next_calls = []
        self.previous_calls = []

    def next(self, distance):
        self.next_calls.append(distance)
        return [Waypoint(self.transform.location.x + distance,
                         self.transform.location.y,
                         self.transform.rotation.yaw)]

    def previous(self, distance):
        self.previous_calls.append(distance)
        return [Waypoint(self.transform.location.x - distance,
                         self.transform.location.y,
                         self.transform.rotation.yaw)]


def test_negative_longitudinal_offset_uses_previous(monkeypatch):
    ego = _load(monkeypatch, "simulation/carla/ego.py", "_coriva_test_ego_previous")
    world, waypoint = SpawnWorld(), Waypoint()
    vehicle = ego._KinematicVehicle(
        world, "vehicle.test", waypoint, initial_speed=0.0,
        distance_ahead=-12.0, lane_offset=0.0, z_offset=0.0)
    assert waypoint.previous_calls == [12.0]
    assert waypoint.next_calls == []
    assert vehicle.x == pytest.approx(-12.0)


def test_positive_lateral_offset_moves_spawn_to_road_right(monkeypatch):
    ego = _load(monkeypatch, "simulation/carla/ego.py", "_coriva_test_ego_lateral")
    world, waypoint = SpawnWorld(), Waypoint(x=10.0, y=20.0, yaw=0.0)
    vehicle = ego._KinematicVehicle(
        world, "vehicle.test", waypoint, initial_speed=0.0,
        distance_ahead=0.0, lane_offset=3.5, z_offset=0.0)
    assert vehicle.x == pytest.approx(10.0)
    assert vehicle.y == pytest.approx(23.5)


def test_vehicle_telemetry_velocity_uses_carla_world_axes(monkeypatch):
    ego = _load(monkeypatch, "simulation/carla/ego.py", "_coriva_test_ego_velocity")
    vehicle = ego._KinematicVehicle(
        SpawnWorld(), "vehicle.test", Waypoint(yaw=90.0), initial_speed=10.0,
        distance_ahead=0.0, lane_offset=0.0, z_offset=0.0)
    state = vehicle.get_ground_truth()
    assert state["vx"] == pytest.approx(0.0, abs=1e-6)
    assert state["vy"] == pytest.approx(10.0)


def test_vehicle_spawn_uses_transient_vertical_lift_then_canonical_pose(monkeypatch):
    ego = _load(monkeypatch, "simulation/carla/ego.py", "_coriva_test_ego_lift")

    class LiftWorld(SpawnWorld):
        def __init__(self):
            super().__init__()
            self.attempted_z = []

        def spawn_actor(self, _blueprint, transform):
            self.attempted_z.append(transform.location.z)
            if transform.location.z < 0.5:
                raise RuntimeError(
                    "Spawn failed because of collision at spawn position")
            self.transform = transform
            return SpawnActor()

    world = LiftWorld()
    vehicle = ego._KinematicVehicle(
        world, "vehicle.bmw.grandtourer", Waypoint(), initial_speed=0.0,
        lane_offset=0.0, z_offset=0.05)

    assert world.attempted_z == pytest.approx([0.05, 0.30, 0.55])
    assert vehicle.carla_actor.physics is False
    assert vehicle.carla_actor.transform.location.z == pytest.approx(0.05)

    second = ego._KinematicVehicle(
        world, "vehicle.bmw.grandtourer", Waypoint(), initial_speed=0.0,
        lane_offset=0.0, z_offset=0.05)
    assert world.attempted_z[-1:] == pytest.approx([0.55])
    assert second.carla_actor.transform.location.z == pytest.approx(0.05)


def test_far_lateral_spawn_does_not_overwrite_vehicle_lift_cache(monkeypatch):
    ego = _load(monkeypatch, "simulation/carla/ego.py", "_coriva_test_ego_lift_cache")

    class CacheWorld(SpawnWorld):
        def __init__(self):
            super().__init__()
            self.attempted_z = []

        def spawn_actor(self, _blueprint, transform):
            self.attempted_z.append(transform.location.z)
            if transform.location.z < 0.5 and len(self.attempted_z) <= 2:
                raise RuntimeError(
                    "Spawn failed because of collision at spawn position")
            self.transform = transform
            return SpawnActor()

    world = CacheWorld()
    ego._KinematicVehicle(
        world, "vehicle.test", Waypoint(), initial_speed=0.0,
        lane_offset=0.0, z_offset=0.05)
    assert world._vehicle_spawn_lift_cache["vehicle.test"] == 0.5

    ego._KinematicVehicle(
        world, "vehicle.test", Waypoint(), initial_speed=0.0,
        lane_offset=8.5, z_offset=0.05)
    assert world._vehicle_spawn_lift_cache["vehicle.test"] == 0.5

    world.attempted_z.clear()
    ego._KinematicVehicle(
        world, "vehicle.test", Waypoint(), initial_speed=0.0,
        lane_offset=0.0, z_offset=0.05)
    assert world.attempted_z == pytest.approx([0.55])


def test_vehicle_update_rejects_opposite_direction_nearest_lane(monkeypatch):
    ego = _load(
        monkeypatch, "simulation/carla/ego.py", "_coriva_test_ego_direction")

    opposite_waypoint = Waypoint(yaw=180.0)
    fake_map = SimpleNamespace(get_waypoint=lambda _location: opposite_waypoint)

    class MotionWorld(SpawnWorld):
        carla_world = SimpleNamespace(get_map=lambda: fake_map)

    vehicle = ego._KinematicVehicle(
        MotionWorld(), "vehicle.test", Waypoint(yaw=0.0), initial_speed=5.0,
        lane_offset=0.0, z_offset=0.0)
    vehicle.update(0.1)
    vehicle.update(0.1)

    assert vehicle.x == pytest.approx(1.0)
    assert vehicle.road_fwd.x == pytest.approx(1.0)
    assert vehicle.get_ground_truth()["vx"] == pytest.approx(5.0)


def test_vehicle_update_rejects_implausible_route_heading_jump(monkeypatch):
    ego = _load(
        monkeypatch, "simulation/carla/ego.py", "_coriva_test_ego_continuity")

    fake_map = SimpleNamespace(
        get_waypoint=lambda _location: Waypoint(x=0.5, yaw=60.0, lane_width=4.0))

    class MotionWorld(SpawnWorld):
        carla_world = SimpleNamespace(get_map=lambda: fake_map)

    vehicle = ego._KinematicVehicle(
        MotionWorld(), "vehicle.test", Waypoint(yaw=0.0), initial_speed=5.0,
        lane_offset=0.0, z_offset=0.0)
    vehicle.update(0.1)

    assert vehicle.road_fwd.x == pytest.approx(1.0)
    assert vehicle.get_ground_truth()["road_center_x"] == pytest.approx(0.0)


def test_vehicle_update_accepts_continuous_curve_and_records_route_evidence(monkeypatch):
    ego = _load(
        monkeypatch, "simulation/carla/ego.py", "_coriva_test_ego_curve")

    fake_map = SimpleNamespace(
        get_waypoint=lambda _location: Waypoint(x=0.5, y=0.05, yaw=5.0, lane_width=4.0))

    class MotionWorld(SpawnWorld):
        carla_world = SimpleNamespace(get_map=lambda: fake_map)

    vehicle = ego._KinematicVehicle(
        MotionWorld(), "vehicle.test", Waypoint(yaw=0.0), initial_speed=5.0,
        lane_offset=0.0, z_offset=0.0)
    vehicle.update(0.1)
    state = vehicle.get_ground_truth()

    assert vehicle.road_fwd.x == pytest.approx(math.cos(math.radians(5.0)))
    assert state["road_center_x"] == pytest.approx(0.5)
    assert state["road_center_y"] == pytest.approx(0.05)
    assert state["road_lane_width"] == pytest.approx(4.0)


def test_named_weather_preset_is_applied_and_unknown_fails(monkeypatch):
    world_module = _load(monkeypatch, "simulation/carla/world.py", "_coriva_test_world")
    wrapper = world_module.World()
    applied = []
    wrapper._world = SimpleNamespace(set_weather=applied.append)
    wrapper.apply_weather("HardRainNoon")
    assert applied == [sys.modules["carla"].WeatherParameters.HardRainNoon]
    with pytest.raises(ValueError, match="unknown CARLA weather"):
        wrapper.apply_weather("NotAWeather")


def test_scenario_waypoints_are_route_filtered_and_seed_ordered(monkeypatch):
    world_module = _load(
        monkeypatch, "simulation/carla/world.py", "_coriva_test_world_waypoints")

    class RouteWaypoint:
        def __init__(self, name, forward=True, backward=True):
            self.name = name
            self.forward = forward
            self.backward = backward

        def next(self, _distance):
            return [self] if self.forward else []

        def previous(self, _distance):
            return [self] if self.backward else []

    locations = [Location(x=float(i)) for i in range(4)]
    transforms = [Transform(location=location) for location in locations]
    waypoints = {
        0.0: RouteWaypoint("zero"),
        1.0: RouteWaypoint("one", forward=False),
        2.0: RouteWaypoint("two"),
        3.0: RouteWaypoint("three"),
    }
    fake_map = SimpleNamespace(
        get_spawn_points=lambda: transforms,
        get_waypoint=lambda location: waypoints[location.x],
    )
    wrapper = world_module.World(map_name="TownTest")
    wrapper._world = SimpleNamespace(get_map=lambda: fake_map)

    candidates = wrapper.get_scenario_waypoints(
        seed=1, forward_required=20.0, backward_required=10.0, limit=2)
    assert [waypoint.name for waypoint in candidates] == ["two", "three"]
    assert wrapper.get_scenario_waypoint(
        seed=1, forward_required=20.0, backward_required=10.0).name == "two"


def test_map_route_sampling_covers_requested_distance(monkeypatch):
    world_module = _load(
        monkeypatch, "simulation/carla/world.py", "_coriva_test_world_route")
    wrapper = world_module.World(map_name="TownTest")

    evidence = wrapper.sample_route_evidence(
        Waypoint(x=0.0, y=0.0, yaw=0.0, lane_width=3.5),
        forward_distance=5.0,
        sample_spacing=2.0,
    )

    assert evidence["source"] == "carla_map_intended_route"
    assert evidence["route_mode"] == "single_deterministic_intended_route"
    assert len(evidence["route_id"]) == 16
    assert evidence["route_centerline"] == [
        [0.0, 0.0], [2.0, 0.0], [4.0, 0.0], [6.0, 0.0],
    ]
    assert evidence["route_coverage_m"] == pytest.approx(6.0)
    assert evidence["route_required_m"] == pytest.approx(5.0)
    assert evidence["route_lane_width_m"] == pytest.approx(3.5)
    assert evidence["route_half_width_m"] == pytest.approx(8.0)


def test_intended_route_supports_signed_progress_and_interpolation(monkeypatch):
    world_module = _load(
        monkeypatch, "simulation/carla/world.py", "_coriva_test_route_plan")
    wrapper = world_module.World(map_name="TownTest")

    route = wrapper.build_intended_route(
        Waypoint(x=0.0, y=0.0, yaw=0.0, lane_width=3.5),
        forward_distance=5.0,
        backward_distance=3.0,
        sample_spacing=2.0,
    )

    assert route.min_s == pytest.approx(-4.0)
    assert route.max_s == pytest.approx(6.0)
    assert route.pose_at(-1.0).transform.location.x == pytest.approx(-1.0)
    assert route.pose_at(4.5).transform.location.x == pytest.approx(4.5)
    evidence = route.evidence()
    assert evidence["route_backward_coverage_m"] == pytest.approx(4.0)
    assert evidence["route_backward_required_m"] == pytest.approx(3.0)
    with pytest.raises(ValueError, match="outside"):
        route.pose_at(6.1)


def test_route_sampling_oversamples_fractional_requirement_without_lowering_it(
        monkeypatch):
    world_module = _load(
        monkeypatch, "simulation/carla/world.py", "_coriva_test_route_fraction")
    wrapper = world_module.World(map_name="TownTest")

    route = wrapper.build_intended_route(
        Waypoint(), forward_distance=24.013, backward_distance=24.013,
        sample_spacing=2.0)
    evidence = route.evidence()

    assert route.max_s == pytest.approx(26.0)
    assert route.min_s == pytest.approx(-26.0)
    assert evidence["route_required_m"] == pytest.approx(24.013)
    assert evidence["route_coverage_m"] == pytest.approx(26.0)
    assert evidence["route_backward_required_m"] == pytest.approx(24.013)
    assert evidence["route_backward_coverage_m"] == pytest.approx(26.0)


def test_route_sampling_parameterizes_waypoints_by_physical_progress(monkeypatch):
    world_module = _load(
        monkeypatch, "simulation/carla/world.py", "_coriva_test_route_metric")
    wrapper = world_module.World(map_name="TownTest")

    class LongStepWaypoint(Waypoint):
        def next(self, distance):
            self.next_calls.append(distance)
            return [LongStepWaypoint(
                self.transform.location.x + 1.2 * distance,
                self.transform.location.y,
                self.transform.rotation.yaw,
            )]

    route = wrapper.build_intended_route(
        LongStepWaypoint(), forward_distance=5.0, sample_spacing=2.0)
    evidence = route.evidence()

    # CARLA was queried in full 2 m increments, but its returned waypoints are
    # 2.4 physical metres apart.  Route s/coverage must describe that physical
    # geometry rather than the query argument.
    assert route.max_s == pytest.approx(7.2)
    assert evidence["route_coverage_m"] == pytest.approx(7.2)
    assert evidence["route_required_m"] == pytest.approx(5.0)
    assert route.pose_at(3.6).transform.location.x == pytest.approx(3.6)


def test_route_sampling_still_rejects_true_full_step_no_progress(monkeypatch):
    world_module = _load(
        monkeypatch, "simulation/carla/world.py", "_coriva_test_route_no_progress")
    wrapper = world_module.World(map_name="TownTest")
    calls = []
    start = Waypoint()

    def stationary(distance):
        calls.append(distance)
        return [Waypoint()]

    start.next = stationary
    with pytest.raises(RuntimeError, match="made no spatial progress"):
        wrapper.build_intended_route(
            start, forward_distance=0.013, sample_spacing=2.0)
    assert calls == [2.0]


@pytest.mark.parametrize(
    ("lane_offset", "acceleration"),
    [(0.0, 0.0), (3.5, 0.0), (-3.5, 3.0), (0.0, -8.0)],
)
def test_curved_route_motion_uses_physical_offset_path_metres(
        monkeypatch, lane_offset, acceleration):
    world_module = _load(
        monkeypatch, "simulation/carla/world.py", "_coriva_test_offset_metric")
    ego = _load(
        monkeypatch, "simulation/carla/ego.py", "_coriva_test_offset_vehicle")

    radius = 7.0
    segment_length = 2.0
    angle_step = 2.0 * math.asin(segment_length / (2.0 * radius))
    samples = []
    for index in range(40):
        angle = -math.pi / 2.0 + index * angle_step
        samples.append({
            "s": index * segment_length,
            "x": radius * math.cos(angle),
            "y": radius * math.sin(angle),
            "z": 0.0,
            "pitch": 0.0,
            "yaw": math.degrees(angle + math.pi / 2.0),
            "roll": 0.0,
            "lane_id": 1,
            "lane_width": 3.5,
        })
    route = world_module.IntendedRoute(
        samples,
        forward_required=70.0,
        forward_coverage=78.0,
        backward_required=0.0,
        backward_coverage=0.0,
        sample_spacing=2.0,
    )
    vehicle = ego._KinematicVehicle(
        SpawnWorld(), "vehicle.test", Waypoint(), initial_speed=10.0,
        lane_offset=lane_offset, z_offset=0.0, route_plan=route)
    vehicle.ax = acceleration
    initial = vehicle.get_ground_truth()
    previous_xy = (initial["x"], initial["y"])
    step_speeds = []
    for _ in range(5):
        vehicle.update(0.1)
        step = vehicle.get_ground_truth()
        step_speeds.append(math.hypot(
            step["x"] - previous_xy[0], step["y"] - previous_xy[1]) / 0.1)
        previous_xy = (step["x"], step["y"])
    current = vehicle.get_ground_truth()

    displacement_speed = math.hypot(
        current["x"] - initial["x"],
        current["y"] - initial["y"],
    ) / 0.5
    velocity_speed = math.hypot(current["vx"], current["vy"])
    # The shipped kinematic gate allows 4 m/s for endpoint-vs-interval
    # acceleration.  Centreline and both adjacent offsets must remain inside
    # that unchanged gate even on this tight synthetic curve.
    assert abs(velocity_speed - displacement_speed) < 4.0
    if acceleration == 0.0:
        assert step_speeds == pytest.approx([10.0] * 5, abs=0.02)


def test_speed_or_acceleration_cannot_change_intended_junction_route(monkeypatch):
    ego = _load(
        monkeypatch, "simulation/carla/ego.py", "_coriva_test_bound_route")

    class BoundRoute:
        route_id = "route-fixed"
        route_mode = "single_deterministic_intended_route"

        def pose_at(self, distance):
            # The declared continuation turns left after x=2. A nearest-map
            # query would instead return the synthetic right branch below.
            if distance <= 2.0:
                waypoint = Waypoint(x=distance, y=0.0, yaw=0.0)
            else:
                waypoint = Waypoint(x=2.0, y=distance - 2.0, yaw=90.0)
            return waypoint

    class ForbiddenMap:
        def get_waypoint(self, _location):
            raise AssertionError("route-bound motion must not query nearest lane")

    class MotionWorld(SpawnWorld):
        carla_world = SimpleNamespace(get_map=lambda: ForbiddenMap())

    route = BoundRoute()
    constant_speed = ego._KinematicVehicle(
        MotionWorld(), "vehicle.test", Waypoint(), initial_speed=2.0,
        lane_offset=0.0, z_offset=0.0, route_plan=route)
    accelerated = ego._KinematicVehicle(
        MotionWorld(), "vehicle.test", Waypoint(), initial_speed=1.0,
        lane_offset=0.0, z_offset=0.0, route_plan=route)
    accelerated.ax = 1.0
    for _ in range(4):
        constant_speed.update(0.5)
    for _ in range(3):
        accelerated.update(0.5)

    constant_state = constant_speed.get_ground_truth()
    accelerated_state = accelerated.get_ground_truth()
    assert (constant_speed.x, constant_speed.y) == pytest.approx((2.0, 2.0))
    assert (accelerated.x, accelerated.y) == pytest.approx((2.0, 1.0))
    assert constant_state["route_s"] == pytest.approx(4.0)
    assert accelerated_state["route_s"] == pytest.approx(3.0)
    assert constant_state["route_id"] == accelerated_state["route_id"]
    assert constant_state["route_id"] == "route-fixed"
    assert constant_state["heading_deg"] == pytest.approx(90.0)
    assert accelerated_state["heading_deg"] == pytest.approx(90.0)


def test_route_progress_does_not_jump_at_self_intersection(monkeypatch):
    ego = _load(
        monkeypatch, "simulation/carla/ego.py", "_coriva_test_self_intersection")

    class FigureEightRoute:
        route_id = "town04-figure-eight"
        route_mode = "single_deterministic_intended_route"

        def pose_at(self, distance):
            poses = {
                0.0: Waypoint(x=0.0, y=0.0, yaw=0.0),
                1.0: Waypoint(x=1.0, y=0.0, yaw=0.0),
                2.0: Waypoint(x=2.0, y=0.0, yaw=90.0),
                # Same geometric point as s=1, but later on the route and in
                # the opposite direction: a Town04-style close crossing.
                3.0: Waypoint(x=1.0, y=0.0, yaw=180.0),
                4.0: Waypoint(x=0.0, y=0.0, yaw=180.0),
            }
            return poses[round(float(distance), 6)]

    class ForbiddenMap:
        def get_waypoint(self, _location):
            raise AssertionError("self-intersection must not trigger map snapping")

    class MotionWorld(SpawnWorld):
        carla_world = SimpleNamespace(get_map=lambda: ForbiddenMap())

    vehicle = ego._KinematicVehicle(
        MotionWorld(), "vehicle.test", Waypoint(), initial_speed=1.0,
        lane_offset=0.0, z_offset=0.0, route_plan=FigureEightRoute())
    vehicle.update(1.0)
    first_crossing = vehicle.get_ground_truth()
    vehicle.update(1.0)
    vehicle.update(1.0)
    second_crossing = vehicle.get_ground_truth()

    assert (first_crossing["x"], first_crossing["y"]) == pytest.approx((1.0, 0.0))
    assert (second_crossing["x"], second_crossing["y"]) == pytest.approx((1.0, 0.0))
    assert first_crossing["route_s"] == pytest.approx(1.0)
    assert second_crossing["route_s"] == pytest.approx(3.0)
    assert first_crossing["heading_deg"] == pytest.approx(0.0)
    assert second_crossing["heading_deg"] == pytest.approx(180.0)


def test_map_route_sampling_chooses_straightest_deterministic_successor(
        monkeypatch):
    world_module = _load(
        monkeypatch, "simulation/carla/world.py", "_coriva_test_world_junction")
    wrapper = world_module.World(map_name="TownTest")

    finish = Waypoint(x=4.0, y=0.0, yaw=0.0)
    straight = Waypoint(x=2.0, y=0.0, yaw=0.0)
    turning = Waypoint(x=0.0, y=2.0, yaw=90.0)
    start = Waypoint(x=0.0, y=0.0, yaw=0.0)
    start.next = lambda _distance: [turning, straight]
    straight.next = lambda _distance: [finish]

    evidence = wrapper.sample_route_evidence(
        start, forward_distance=4.0, sample_spacing=2.0)

    assert evidence["route_centerline"] == [
        [0.0, 0.0], [2.0, 0.0], [4.0, 0.0],
    ]


def test_map_route_sampling_rejects_truncated_waypoint_route(monkeypatch):
    world_module = _load(
        monkeypatch, "simulation/carla/world.py", "_coriva_test_world_dead_end")
    wrapper = world_module.World(map_name="TownTest")
    start = Waypoint(x=0.0, y=0.0, yaw=0.0)
    start.next = lambda _distance: []

    with pytest.raises(RuntimeError, match="route evidence incomplete"):
        wrapper.sample_route_evidence(
            start, forward_distance=10.0, sample_spacing=2.0)


class CameraBlueprint:
    def __init__(self):
        self.attributes = {}

    def set_attribute(self, name, value):
        self.attributes[name] = value


class Camera:
    def __init__(self):
        self.callback = None

    def listen(self, callback):
        self.callback = callback

    def get_transform(self):
        identity = [[1.0 if row == col else 0.0 for col in range(4)]
                    for row in range(4)]
        return SimpleNamespace(get_inverse_matrix=lambda: identity)


class CameraWorld:
    def __init__(self):
        self.blueprint = CameraBlueprint()
        self.camera = Camera()
        self.spawn = None

    def get_blueprint(self, name):
        assert name == "sensor.camera.rgb"
        return self.blueprint

    def spawn_attached(self, blueprint, transform, parent):
        self.spawn = (blueprint, transform, parent)
        return self.camera


def test_windshield_camera_calibration_and_real_bbox_projection(monkeypatch, tmp_path):
    recorder_module = _load(
        monkeypatch, "simulation/carla/recorder.py", "_coriva_test_recorder")
    world = CameraWorld()
    recorder = recorder_module.TrajectoryRecorder(
        world, attach_to=object(), output_dir=str(tmp_path))
    recorder.start()

    assert world.blueprint.attributes == {
        "image_size_x": "1024", "image_size_y": "768", "fov": "100"
    }
    transform = world.spawn[1]
    assert (transform.location.x, transform.location.y, transform.location.z) == (
        1.5, 0.0, 1.7)
    assert transform.rotation.pitch == 0.0

    vertices = [Location(10.0, y, z) for y in (-1.0, 1.0)
                for z in (-1.0, 1.0)] * 2
    box = SimpleNamespace(get_world_vertices=lambda _transform: vertices)
    carla_actor = SimpleNamespace(
        bounding_box=box, get_transform=lambda: Transform())
    actor = SimpleNamespace(carla_actor=carla_actor)
    projected = recorder._project_actor_bbox(actor)
    assert projected is not None
    x1, y1, x2, y2 = projected
    assert x1 < 512 < x2 and y1 < 384 < y2
    assert (x2 - x1) * (y2 - y1) > 100.0


# --- Published swerve cap ----------------------------------------------------
#
# The prompt and the kinematic twin state that a swerve adds 2.5 m/s laterally
# until the ego is EGO_SWERVE_LANE_WIDTH_M = 3.5 m off its road-following path.
# The CARLA ego caps at that published width, not the REAL lane width of the
# map.  These tests pin the published cap on a wide lane while the real width
# is still recorded as raw evidence.

CARLA_SCENARIO_1_Y = [-1.25, -2.5, -3.5, -3.5, -3.5, -3.5]  # 0.5 s samples, left swerve


def _straight_route(world_module, lane_width):
    samples = [{
        "s": index * 2.0, "x": index * 2.0, "y": 0.0, "z": 0.0,
        "pitch": 0.0, "yaw": 0.0, "roll": 0.0, "lane_id": 1,
        "lane_width": lane_width,
    } for index in range(60)]
    return world_module.IntendedRoute(
        samples, forward_required=70.0, forward_coverage=118.0,
        backward_required=0.0, backward_coverage=0.0, sample_spacing=2.0)


def _swerve_left_samples(ego_module, vehicle, seconds=3.0, dt=0.1):
    """Run the released ego swerve (generate._SWERVE_INTENSITY) and return
    the CARLA y at every 0.5 s plus the lateral speed while moving."""
    from simulation.protocol import EGO_SWERVE_SPEED_MPS
    actions = ego_module.EgoActions(vehicle)
    actions.steer_left(intensity=EGO_SWERVE_SPEED_MPS / actions.MAX_LATERAL)
    moving_vy = vehicle.vy
    ys = []
    steps = int(round(seconds / dt))
    for step in range(1, steps + 1):
        vehicle.update(dt)
        if step % 5 == 0:
            ys.append(vehicle.get_ground_truth()["y"])
    return ys, moving_vy


def test_ego_swerve_constants_mirror_the_published_protocol(monkeypatch):
    from simulation import generate, protocol
    ego = _load(monkeypatch, "simulation/carla/ego.py", "_coriva_test_ego_consts")

    assert ego.EGO_SWERVE_LANE_WIDTH_M == protocol.EGO_SWERVE_LANE_WIDTH_M == 3.5
    assert ego.EgoActions.STEER_LATERAL_LIMIT_M == protocol.EGO_SWERVE_LANE_WIDTH_M
    # The released swerve intensity is normalised by the controller's own
    # MAX_LATERAL, so the realised lateral speed is exactly the published one.
    assert generate._CARLA_MAX_LATERAL_SPEED_MPS == ego._VehicleActions.MAX_LATERAL
    assert (ego._VehicleActions.MAX_LATERAL * generate._SWERVE_INTENSITY
            == pytest.approx(protocol.EGO_SWERVE_SPEED_MPS))


def test_ego_swerve_on_wide_waypoint_lane_caps_at_published_width(monkeypatch):
    """Nearest-waypoint path (no route plan): waypoint lane 4.0 m, cap still 3.5 m."""
    ego = _load(monkeypatch, "simulation/carla/ego.py", "_coriva_test_ego_wide_wp")

    fake_map = SimpleNamespace(
        get_waypoint=lambda location: Waypoint(x=location.x, y=0.0, yaw=0.0, lane_width=4.0))

    class MotionWorld(SpawnWorld):
        carla_world = SimpleNamespace(get_map=lambda: fake_map)

    vehicle = ego.Ego(MotionWorld(), "vehicle.test", waypoint=Waypoint(lane_width=4.0),
                      initial_speed=12.2)
    actions = ego.EgoActions(vehicle)
    assert actions._get_real_lane_width() == pytest.approx(4.0)

    ys, moving_vy = _swerve_left_samples(ego, vehicle)

    assert vehicle._lateral_limit == pytest.approx(3.5)
    assert moving_vy == pytest.approx(2.5)
    assert ys[:2] == pytest.approx(CARLA_SCENARIO_1_Y[:2])
    # The nearest-waypoint path integrates the position AFTER the clamp, so its recorded
    # y lags the accumulated offset by one 0.1 s step at the cap (3.25 here).
    # Protocol runs never use this path (they are route-bound, tested below);
    # the cap itself is the published 3.5 m, not the 4.0 m lane.
    assert max(abs(y) for y in ys) <= 3.5 + 1e-9
    assert vehicle.get_lateral_displacement() == pytest.approx(3.5)
    assert vehicle.vy == 0.0
    assert vehicle.get_ground_truth()["road_lane_width"] == pytest.approx(4.0)


def test_ego_swerve_on_wide_route_lane_caps_at_published_width(monkeypatch):
    """Protocol route-bound path: route lane 4.0 m, cap still 3.5 m and the
    4.0 m stays in the raw evidence."""
    world_module = _load(monkeypatch, "simulation/carla/world.py", "_coriva_test_world_wide_route")
    ego = _load(monkeypatch, "simulation/carla/ego.py", "_coriva_test_ego_wide_route")

    vehicle = ego.Ego(SpawnWorld(), "vehicle.test", waypoint=Waypoint(),
                      initial_speed=12.2, route_plan=_straight_route(world_module, 4.0))
    actions = ego.EgoActions(vehicle)
    assert actions._get_real_lane_width() == pytest.approx(4.0)

    ys, moving_vy = _swerve_left_samples(ego, vehicle)
    state = vehicle.get_ground_truth()

    assert vehicle._lateral_limit == pytest.approx(3.5)
    assert moving_vy == pytest.approx(2.5)
    assert ys == pytest.approx(CARLA_SCENARIO_1_Y)
    assert vehicle.get_lateral_displacement() == pytest.approx(3.5)
    assert state["road_lane_width"] == pytest.approx(4.0)
    assert state["x"] == pytest.approx(12.2 * 3.0)


def test_ego_swerve_right_on_wide_lane_caps_at_published_width(monkeypatch):
    world_module = _load(monkeypatch, "simulation/carla/world.py", "_coriva_test_world_wide_right")
    ego = _load(monkeypatch, "simulation/carla/ego.py", "_coriva_test_ego_wide_right")

    vehicle = ego.Ego(SpawnWorld(), "vehicle.test", waypoint=Waypoint(),
                      initial_speed=12.2, route_plan=_straight_route(world_module, 4.0))
    ego.EgoActions(vehicle).steer_right(intensity=2.5 / 3.5)
    for _ in range(30):
        vehicle.update(0.1)

    assert vehicle.get_lateral_displacement() == pytest.approx(-3.5)
    assert vehicle.get_ground_truth()["y"] == pytest.approx(3.5)
    assert vehicle.get_ground_truth()["road_lane_width"] == pytest.approx(4.0)


def test_base_vehicle_steer_keeps_real_lane_width_cap(monkeypatch):
    """NPC controllers (NPCActions inherits the base) are unchanged: their
    lateral auto-stop still uses the map's real lane width."""
    world_module = _load(monkeypatch, "simulation/carla/world.py", "_coriva_test_world_npc_cap")
    ego = _load(monkeypatch, "simulation/carla/ego.py", "_coriva_test_ego_npc_cap")

    vehicle = ego._KinematicVehicle(
        SpawnWorld(), "vehicle.test", Waypoint(), initial_speed=10.0,
        lane_offset=0.0, z_offset=0.0, route_plan=_straight_route(world_module, 4.0))
    assert ego._VehicleActions.STEER_LATERAL_LIMIT_M is None
    ego._VehicleActions(vehicle).steer_left(intensity=1.0)
    assert vehicle._lateral_limit == pytest.approx(4.0)
