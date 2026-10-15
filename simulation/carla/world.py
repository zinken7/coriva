import carla
from bisect import bisect_right
import hashlib
import json
import math
import os
import time

from simulation.protocol import CARLA_ROUTE_HALF_WIDTH_M


# All available maps in CARLA (0.9.15+)
# Each map has an _Opt (layered) version that allows toggling layers on/off
CARLA_MAPS = {
    "Town01": {
        "description": "Small town with river and bridges",
        "road_type": "urban",
        "lanes": "1-2",
        "junctions": "T-junctions, bridges",
        "features": "River, bridges, commercial + residential buildings",
        "additional_package": False,
    },
    "Town02": {
        "description": "Small residential area with commercial streets",
        "road_type": "urban",
        "lanes": "1-2",
        "junctions": "T-junctions",
        "features": "Gas station, church, tree-lined streets, apartments",
        "additional_package": False,
    },
    "Town03": {
        "description": "Large urban area with central roundabout and metro",
        "road_type": "urban",
        "lanes": "2-4",
        "junctions": "Roundabout, 4-way, T-junctions, underpass",
        "features": "Roundabout, elevated metro, gas station, park",
        "additional_package": False,
    },
    "Town04": {
        "description": "Figure-8 road around mountains, small highway",
        "road_type": "highway + urban",
        "lanes": "2-4",
        "junctions": "Overpass/underpass, slip roads",
        "features": "Mountains, gas station, shopping area, overpass/underpass roads",
        "additional_package": False,
    },
    "Town05": {
        "description": "Urban area with divided roads and elevated highway",
        "road_type": "urban + highway",
        "lanes": "2-4",
        "junctions": "Large intersections, highway ramps",
        "features": "Commercial buildings, construction sites, parking lots",
        "additional_package": False,
    },
    "Town06": {
        "description": "Large multi-lane roads, Michigan Left",
        "road_type": "highway",
        "lanes": "4-6",
        "junctions": "Michigan left, slip roads, culdesacs",
        "features": "5-lane roads, dedicated turn lanes, many slip roads",
        "additional_package": True,
    },
    "Town07": {
        "description": "Rural countryside with farms",
        "road_type": "rural",
        "lanes": "1-2",
        "junctions": "Simple junctions, unmarked roads",
        "features": "Barns, windmills, grain silos, cornfields, ponds",
        "additional_package": True,
    },
    "Town10": {
        "description": "Coastal urban area with skyscrapers",
        "road_type": "urban",
        "lanes": "2-4",
        "junctions": "Grid layout, yellow-box, turning lanes",
        "features": "Tree-lined avenues, museum, coastal promenade, skyscrapers",
        "additional_package": False,
    },
    "Town11": {
        "description": "Undecorated urban area (roads and ground only)",
        "road_type": "urban",
        "lanes": "2-4",
        "junctions": "Various",
        "features": "Empty map — suitable for custom decoration",
        "additional_package": False,
    },
    "Town12": {
        "description": "Largest map — urban + suburban + rural",
        "road_type": "mixed (urban + suburban + rural + highway)",
        "lanes": "1-6",
        "junctions": "All types",
        "features": "10x larger than other maps, diverse environments",
        "additional_package": False,
    },
}


class _RoutePose:
    """Waypoint-shaped interpolated pose on one deterministic route."""

    def __init__(self, transform, lane_id, lane_width):
        self.transform = transform
        self.lane_id = lane_id
        self.lane_width = lane_width


class IntendedRoute:
    """Immutable runtime route shared by every rollout of one placement."""

    def __init__(self, samples, forward_required, forward_coverage,
                 backward_required, backward_coverage, sample_spacing):
        if len(samples) < 2:
            raise ValueError("intended route requires at least two samples")
        self._samples = tuple(samples)
        self._distances = tuple(float(item["s"]) for item in samples)
        self.forward_required = float(forward_required)
        self.forward_coverage = float(forward_coverage)
        self.backward_required = float(backward_required)
        self.backward_coverage = float(backward_coverage)
        self.sample_spacing = float(sample_spacing)
        identity_payload = [
            [round(float(item["s"]), 3), round(float(item["x"]), 3),
             round(float(item["y"]), 3), round(float(item["yaw"]), 3)]
            for item in self._samples
        ]
        self.route_id = hashlib.sha256(json.dumps(
            identity_payload, separators=(",", ":")).encode("utf-8")
        ).hexdigest()[:16]
        self.route_mode = "single_deterministic_intended_route"

    @staticmethod
    def _interpolate_angle(start, end, fraction):
        delta = (float(end) - float(start) + 180.0) % 360.0 - 180.0
        return float(start) + fraction * delta

    @staticmethod
    def _transform(sample):
        return carla.Transform(
            carla.Location(
                x=float(sample["x"]), y=float(sample["y"]),
                z=float(sample["z"])),
            carla.Rotation(
                pitch=float(sample["pitch"]), yaw=float(sample["yaw"]),
                roll=float(sample["roll"])),
        )

    @property
    def min_s(self):
        return self._distances[0]

    @property
    def max_s(self):
        return self._distances[-1]

    def _bracket(self, distance):
        """Return the two samples and interpolation fraction for route ``s``."""
        value = float(distance)
        if value < self.min_s - 1e-6 or value > self.max_s + 1e-6:
            raise ValueError(
                f"intended route coordinate {value:.2f}m outside "
                f"[{self.min_s:.2f}, {self.max_s:.2f}]m")
        value = max(self.min_s, min(self.max_s, value))
        right = bisect_right(self._distances, value)
        if right == 0:
            left = right = 0
        elif right >= len(self._samples):
            left = right = len(self._samples) - 1
        else:
            left = right - 1
        first = self._samples[left]
        second = self._samples[right]
        span = float(second["s"]) - float(first["s"])
        fraction = 0.0 if span <= 1e-9 else (
            value - float(first["s"])) / span
        return value, left, right, first, second, fraction

    @staticmethod
    def _offset_xy(sample, lateral_right):
        """Shift one centreline sample along its CARLA road-right vector."""
        yaw = math.radians(float(sample["yaw"]))
        right_x, right_y = -math.sin(yaw), math.cos(yaw)
        return (
            float(sample["x"]) + right_x * float(lateral_right),
            float(sample["y"]) + right_y * float(lateral_right),
        )

    def offset_location_at(self, distance, lateral_right):
        """Return a point on the polyline parallel to the intended route.

        Shifting the sampled endpoints before interpolation gives every fixed
        lane offset its own continuous polyline.  That path can then be
        parameterized in physical metres instead of inheriting the centreline
        metric on curves.
        """
        _value, _left, _right, first, second, fraction = self._bracket(
            distance)
        first_x, first_y = self._offset_xy(first, lateral_right)
        second_x, second_y = self._offset_xy(second, lateral_right)
        return carla.Location(
            x=first_x + fraction * (second_x - first_x),
            y=first_y + fraction * (second_y - first_y),
            z=(float(first["z"])
               + fraction * (float(second["z"]) - float(first["z"]))),
        )

    def advance_along_offset(self, distance, travel_distance, lateral_right):
        """Advance monotonically by physical metres on one offset polyline."""
        remaining = float(travel_distance)
        if remaining < 0.0:
            raise ValueError("route travel distance must be non-negative")
        value = self._bracket(distance)[0]
        if remaining <= 1e-9:
            return value

        while remaining > 1e-9:
            (_value, _left, right, first, second,
             fraction) = self._bracket(value)
            if right >= len(self._samples) - 1 and first is second:
                raise ValueError(
                    f"intended route has insufficient offset coverage after "
                    f"{value:.2f}m")

            first_x, first_y = self._offset_xy(first, lateral_right)
            second_x, second_y = self._offset_xy(second, lateral_right)
            current_x = first_x + fraction * (second_x - first_x)
            current_y = first_y + fraction * (second_y - first_y)
            available = math.hypot(
                second_x - current_x, second_y - current_y)
            segment_end_s = float(second["s"])

            if available <= 1e-9:
                value = segment_end_s
                continue
            if remaining <= available + 1e-9:
                route_remaining = segment_end_s - value
                value += route_remaining * min(1.0, remaining / available)
                return min(value, self.max_s)
            remaining -= available
            value = segment_end_s

        return value

    def pose_at(self, distance):
        """Return an interpolated waypoint-like pose at route coordinate ``s``."""
        (_value, _left, _right, first, second,
         fraction) = self._bracket(distance)
        sample = {
            "x": float(first["x"]) + fraction * (float(second["x"]) - float(first["x"])),
            "y": float(first["y"]) + fraction * (float(second["y"]) - float(first["y"])),
            "z": float(first["z"]) + fraction * (float(second["z"]) - float(first["z"])),
            "pitch": self._interpolate_angle(
                first["pitch"], second["pitch"], fraction),
            "yaw": self._interpolate_angle(first["yaw"], second["yaw"], fraction),
            "roll": self._interpolate_angle(first["roll"], second["roll"], fraction),
        }
        metadata = first if fraction < 0.5 else second
        return _RoutePose(
            self._transform(sample), metadata.get("lane_id"),
            float(metadata.get("lane_width", 3.5)),
        )

    def evidence(self):
        points = [[float(item["x"]), float(item["y"])]
                  for item in self._samples]
        widths = [float(item.get("lane_width", 0.0))
                  for item in self._samples
                  if float(item.get("lane_width", 0.0)) > 0.0]
        return {
            "schema_version": 2,
            "coordinate_frame": "world",
            "source": "carla_map_intended_route",
            "route_mode": self.route_mode,
            "route_id": self.route_id,
            "route_centerline": points,
            "route_lane_width_m": max(widths, default=3.5),
            "route_half_width_m": CARLA_ROUTE_HALF_WIDTH_M,
            "route_coverage_m": round(self.forward_coverage, 3),
            "route_required_m": round(self.forward_required, 3),
            "route_backward_coverage_m": round(self.backward_coverage, 3),
            "route_backward_required_m": round(self.backward_required, 3),
            "sample_spacing_m": round(self.sample_spacing, 3),
        }


class World:
    """Manages CARLA world: connection, sync mode, blueprint, waypoint, cleanup."""

    def __init__(self, host='localhost', port=2000, timeout=30.0,
                 map_name='Town04', fps=10, step_length=0.1):
        self.host = host
        self.port = port
        self.timeout = timeout
        self.map_name = map_name
        self.fps = fps
        self.step_length = step_length

        self._client = None
        self._world = None
        self._blueprint_library = None
        self._tracked_actors = []

    def setup(self):
        """Connect to CARLA server, load map, enable synchronous mode."""
        self._client = carla.Client(self.host, self.port)
        self._client.set_timeout(self.timeout)
        self._world = self._client.load_world(self.map_name)

        settings = self._world.get_settings()
        settings.synchronous_mode = True
        settings.fixed_delta_seconds = self.step_length
        self._world.apply_settings(settings)

        self._blueprint_library = self._world.get_blueprint_library()
        self._tracked_actors = []
        return self

    @property
    def client(self):
        return self._client

    @property
    def carla_world(self):
        return self._world

    def tick(self):
        """Advance 1 simulation step."""
        self._world.tick()

    def get_blueprint(self, name):
        """Find blueprint by name. Example: 'vehicle.tesla.model3'."""
        return self._blueprint_library.find(name)

    def get_waypoint(self, x, y, z=0.0):
        """Get the nearest waypoint at coordinates (x, y, z)."""
        return self._world.get_map().get_waypoint(
            carla.Location(x=x, y=y, z=z)
        )

    def get_scenario_waypoints(self, seed=0, forward_required=0.0,
                               backward_required=0.0, limit=None):
        """Return route-valid waypoints in a deterministic, seed-rotated order."""
        spawn_points = self._world.get_map().get_spawn_points()
        if not spawn_points:
            raise RuntimeError(f"map {self.map_name} has no spawn points")
        start = int(seed) % len(spawn_points)
        candidates = []
        for step in range(len(spawn_points)):
            transform = spawn_points[(start + step) % len(spawn_points)]
            waypoint = self._world.get_map().get_waypoint(transform.location)
            if waypoint is None:
                continue
            if forward_required > 0 and not waypoint.next(float(forward_required)):
                continue
            if backward_required > 0 and not waypoint.previous(float(backward_required)):
                continue
            candidates.append(waypoint)
            if limit is not None and len(candidates) >= int(limit):
                break
        if not candidates:
            raise RuntimeError(
                f"map {self.map_name} has no spawn point with "
                f"{forward_required:.1f} m forward/"
                f"{backward_required:.1f} m backward route")
        return candidates

    def get_scenario_waypoint(self, seed=0, forward_required=0.0,
                              backward_required=0.0):
        """Choose the first deterministic waypoint with enough road."""
        return self.get_scenario_waypoints(
            seed=seed,
            forward_required=forward_required,
            backward_required=backward_required,
            limit=1,
        )[0]

    @staticmethod
    def _route_successor(current, candidates):
        """Choose a deterministic, direction-continuous waypoint successor."""
        current_yaw = float(current.transform.rotation.yaw)

        def key(candidate):
            candidate_yaw = float(candidate.transform.rotation.yaw)
            yaw_delta = abs(
                (candidate_yaw - current_yaw + 180.0) % 360.0 - 180.0)
            same_lane = (
                getattr(candidate, "lane_id", None)
                == getattr(current, "lane_id", None)
            )
            location = candidate.transform.location
            return (
                round(yaw_delta, 6),
                0 if same_lane else 1,
                int(getattr(candidate, "road_id", 0)),
                int(getattr(candidate, "section_id", 0)),
                int(getattr(candidate, "lane_id", 0)),
                round(float(location.x), 6),
                round(float(location.y), 6),
            )

        return min(candidates, key=key)

    @staticmethod
    def _route_sample(distance, waypoint):
        transform = waypoint.transform
        location = transform.location
        rotation = transform.rotation
        return {
            "s": float(distance),
            "x": float(location.x),
            "y": float(location.y),
            "z": float(location.z),
            "pitch": float(getattr(rotation, "pitch", 0.0)),
            "yaw": float(getattr(rotation, "yaw", 0.0)),
            "roll": float(getattr(rotation, "roll", 0.0)),
            "lane_id": getattr(waypoint, "lane_id", None),
            "lane_width": float(getattr(waypoint, "lane_width", 3.5)),
        }

    def _sample_route_direction(self, waypoint, required, spacing, method):
        current = waypoint
        location = current.transform.location
        samples = []
        covered = 0.0
        max_steps = max(2, int(math.ceil(required / spacing)) * 4 + 8)
        for _ in range(max_steps):
            if covered >= required:
                break
            # CARLA waypoint traversal is discretized at ``spacing``. Always
            # take a full final step so achieved coverage is >= the exact
            # requirement; a fractional tail below the 5 cm progress guard
            # would otherwise be rejected as false no-progress.
            requested_step = spacing
            candidates = list(getattr(current, method)(requested_step) or [])
            if not candidates:
                raise RuntimeError(
                    "route evidence incomplete: CARLA waypoint has no "
                    f"{method} waypoint after {covered:.1f}/{required:.1f} m")
            successor = self._route_successor(current, candidates)
            next_location = successor.transform.location
            progress = math.hypot(
                float(next_location.x) - float(location.x),
                float(next_location.y) - float(location.y),
            )
            if progress < 0.05:
                raise RuntimeError(
                    "route evidence incomplete: CARLA waypoint traversal made "
                    f"no spatial progress after {covered:.1f}/{required:.1f} m")
            # ``Waypoint.next(spacing)`` is a topology query, not a metric
            # guarantee.  CARLA can return samples more than 20% farther apart
            # on real maps.  Route ``s`` and coverage must therefore use the
            # physical planar progress that drives x/y displacement checks.
            covered += progress
            signed_distance = covered if method == "next" else -covered
            samples.append(self._route_sample(signed_distance, successor))
            current = successor
            location = next_location
        else:
            raise RuntimeError(
                "route evidence incomplete: waypoint traversal exceeded its "
                f"step budget after {covered:.1f}/{required:.1f} m")
        if covered < required:
            raise RuntimeError(
                f"route evidence incomplete: sampled only "
                f"{covered:.1f}/{required:.1f} m")
        return samples, covered

    def build_intended_route(self, waypoint, forward_distance,
                             backward_distance=0.0, sample_spacing=2.0):
        """Build the one route plan shared by all scenario rollouts.

        Junction choice is resolved once here. Vehicle speed and interventions
        can change progress or lateral offset, but cannot select a different
        road continuation later in the simulator.
        """
        forward_required = float(forward_distance)
        backward_required = float(backward_distance)
        spacing = float(sample_spacing)
        if forward_required <= 0.0:
            raise ValueError("forward_distance must be positive")
        if backward_required < 0.0:
            raise ValueError("backward_distance must be non-negative")
        if spacing <= 0.0:
            raise ValueError("sample_spacing must be positive")
        forward, forward_covered = self._sample_route_direction(
            waypoint, forward_required, spacing, "next")
        backward = []
        backward_covered = 0.0
        if backward_required > 0.0:
            backward, backward_covered = self._sample_route_direction(
                waypoint, backward_required, spacing, "previous")
        samples = list(reversed(backward))
        samples.append(self._route_sample(0.0, waypoint))
        samples.extend(forward)
        return IntendedRoute(
            samples, forward_required, forward_covered,
            backward_required, backward_covered, spacing)

    def sample_route_evidence(self, waypoint, forward_distance,
                              sample_spacing=2.0):
        """Compatibility API returning JSON evidence for an intended route."""
        return self.build_intended_route(
            waypoint, forward_distance, sample_spacing=sample_spacing,
        ).evidence()

    def apply_weather(self, preset):
        """Apply a named CARLA WeatherParameters preset fail-closed."""
        try:
            weather = getattr(carla.WeatherParameters, str(preset))
        except AttributeError as exc:
            raise ValueError(f"unknown CARLA weather preset: {preset!r}") from exc
        self._world.set_weather(weather)

    def spawn_actor(self, blueprint, transform):
        """Spawn actor and track for automatic cleanup."""
        actor = self._world.spawn_actor(blueprint, transform)
        self._tracked_actors.append(actor)
        return actor

    def spawn_attached(self, blueprint, transform, parent):
        """Spawn actor attached to parent (e.g.: camera attached to vehicle)."""
        actor = self._world.spawn_actor(blueprint, transform, attach_to=parent)
        self._tracked_actors.append(actor)
        return actor

    def cleanup(self):
        """Stop sensors and destroy all tracked actors."""
        for actor in reversed(self._tracked_actors):
            if actor and actor.is_alive:
                if 'sensor' in actor.type_id:
                    actor.stop()
                actor.destroy()
        self._tracked_actors.clear()
        if self._world:
            self._world.tick()

    def change_map(self, new_map_name):
        """Switch to a new map, keep sync mode."""
        self.cleanup()
        self.map_name = new_map_name
        self._world = self._client.load_world(new_map_name)

        settings = self._world.get_settings()
        settings.synchronous_mode = True
        settings.fixed_delta_seconds = self.step_length
        self._world.apply_settings(settings)

        self._blueprint_library = self._world.get_blueprint_library()
        time.sleep(3)
        return self
