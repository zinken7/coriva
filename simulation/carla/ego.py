import carla

from simulation.protocol import EGO_SWERVE_LANE_WIDTH_M


_SPAWN_LIFT_CANDIDATES_M = (0.0, 0.25, 0.5)
_MAX_ROUTE_YAW_RATE_DEG_PER_SEC = 90.0
_MIN_ROUTE_YAW_TOLERANCE_DEG = 5.0


class _VehicleActions:
    """Base class for EgoActions and NPCActions.
    
    Controls kinematic state of vehicle via intensity [0.0, 1.0]
    or direct values.
    """

    MAX_DECEL = 9.0    # m/s² (emergency braking)
    MIN_DECEL = 2.0    # m/s² (light braking)
    MAX_ACCEL = 4.0    # m/s² (maximum acceleration)
    # The published ego swerve realises EGO_SWERVE_SPEED_MPS = 2.5 m/s
    # (simulation/protocol.py): generate._SWERVE_INTENSITY is
    # EGO_SWERVE_SPEED_MPS / _CARLA_MAX_LATERAL_SPEED_MPS = 2.5 / 3.5, and
    # steer_left/steer_right set vy = MAX_LATERAL * intensity = 2.5 m/s.
    MAX_LATERAL = 3.5  # m/s (maximum lane change speed)
    LANE_WIDTH = 3.5   # m   (standard lane width; fallback when no waypoint)
    # Lateral auto-stop cap for steer_left/steer_right. None = the real lane
    # width read from the CARLA waypoint (NPC cut-ins keep this). EgoActions
    # overrides it with the published EGO_SWERVE_LANE_WIDTH_M so the ego
    # swerve ends one PUBLISHED lane width (3.5 m) off its route on every map,
    # exactly as the prompt and the kinematic twin state. The real lane
    # width is still recorded as raw evidence in `road_lane_width` /
    # `route_lane_width_m`.
    STEER_LATERAL_LIMIT_M = None

    def __init__(self, vehicle):
        """
        Args:
            vehicle: Ego or NPC instance.
        """
        self.vehicle = vehicle

    def brake(self, intensity=1.0, deceleration=None):
        """Brake with intensity [0.0, 1.0] or a direct deceleration value.

        Args:
            intensity: 0.0 = lightest braking (-2.0 m/s²), 
                       1.0 = emergency braking (-9.0 m/s²).
            deceleration: If provided, use this value directly (positive, 
                          e.g.: 7.0 → ax = -7.0 m/s²). Overrides intensity.
        """
        if deceleration is not None:
            self.vehicle.ax = -abs(deceleration)
        else:
            intensity = max(0.0, min(1.0, intensity))
            self.vehicle.ax = -(self.MIN_DECEL + intensity * (self.MAX_DECEL - self.MIN_DECEL))
        self.vehicle.vy = 0.0

    def maintain_speed(self):
        """Maintain current speed, no steering."""
        self.vehicle.ax = 0.0
        self.vehicle.vy = 0.0

    def steer_left(self, intensity=0.5, auto_stop_at_lane=True):
        """Steer left with intensity [0.0, 1.0].

        Args:
            intensity: 0.0 = no steering, 1.0 = maximum lateral speed.
            auto_stop_at_lane: If True, auto-stop when actual lane width displacement is reached.
        """
        intensity = max(0.0, min(1.0, intensity))
        self.vehicle.vy = self.MAX_LATERAL * intensity  # Positive = left (along road_left)
        self.vehicle._auto_stop_lateral = auto_stop_at_lane
        self.vehicle._lateral_limit = self._steer_lateral_limit()

    def steer_right(self, intensity=0.5, auto_stop_at_lane=True):
        """Steer right with intensity [0.0, 1.0]."""
        intensity = max(0.0, min(1.0, intensity))
        self.vehicle.vy = -self.MAX_LATERAL * intensity  # Negative = right
        self.vehicle._auto_stop_lateral = auto_stop_at_lane
        self.vehicle._lateral_limit = self._steer_lateral_limit()

    def _steer_lateral_limit(self):
        """Lateral offset (m) at which an auto-stopping steer holds.

        The published constant when STEER_LATERAL_LIMIT_M is set (ego),
        otherwise the actual lane width from the CARLA waypoint (NPC).
        """
        if self.STEER_LATERAL_LIMIT_M is not None:
            return float(self.STEER_LATERAL_LIMIT_M)
        return self._get_real_lane_width()

    def _get_real_lane_width(self):
        """Get actual lane width from CARLA waypoint. Fallback = LANE_WIDTH."""
        if getattr(self.vehicle, "_route_plan", None) is not None:
            return float(self.vehicle._road_lane_width)
        try:
            carla_map = self.vehicle._world.carla_world.get_map()
            wp = carla_map.get_waypoint(
                carla.Location(x=self.vehicle.x, y=self.vehicle.y, z=self.vehicle.z)
            )
            if wp and hasattr(wp, 'lane_width'):
                return wp.lane_width
        except Exception:
            pass
        return self.LANE_WIDTH

    def accelerate(self, intensity=0.5, acceleration=None):
        """Accelerate with intensity [0.0, 1.0].

        Args:
            intensity: 0.0 = no acceleration, 1.0 = maximum acceleration.
            acceleration: Direct value (m/s²). Overrides intensity.
        """
        if acceleration is not None:
            self.vehicle.ax = abs(acceleration)
        else:
            intensity = max(0.0, min(1.0, intensity))
            self.vehicle.ax = self.MAX_ACCEL * intensity


class _KinematicVehicle:
    """Base class for Ego and NPC — kinematic state + teleport.

    Protocol runs bind every road vehicle to one precomputed intended route.
    Callers without a route plan get nearest-waypoint following.
    """

    def __init__(self, world, model, waypoint, initial_speed,
                 distance_ahead=0.0, lane_offset=0.0, z_offset=0.05,
                 route_plan=None):
        """
        Args:
            world: World instance.
            model: Blueprint name (e.g.: 'vehicle.tesla.model3').
            waypoint: CARLA waypoint for spawning.
            initial_speed: Initial speed (m/s).
            distance_ahead: Distance ahead of waypoint to spawn (m).
            lane_offset: Lateral offset from lane center (m).
            route_plan: Immutable intended route shared by all rollouts.
        """
        self._world = world
        self._route_plan = route_plan
        self._route_s = float(distance_ahead)
        self._route_lane_offset = float(lane_offset)

        # Resolve the signed longitudinal offset.  The old implementation
        # silently mapped every negative distance to the ego waypoint, which
        # made rear vehicles overlap the ego at spawn.
        if route_plan is not None:
            spawn_wp = route_plan.pose_at(self._route_s)
        elif distance_ahead > 0:
            candidates = waypoint.next(distance_ahead)
            if not candidates:
                raise ValueError(f"no waypoint {distance_ahead:.1f} m ahead")
            spawn_wp = candidates[0]
        elif distance_ahead < 0:
            candidates = waypoint.previous(abs(distance_ahead))
            if not candidates:
                raise ValueError(f"no waypoint {abs(distance_ahead):.1f} m behind")
            spawn_wp = candidates[0]
        else:
            spawn_wp = waypoint

        base_tf = spawn_wp.transform
        right = base_tf.rotation.get_right_vector()
        # Config coordinates use positive lateral offset to the right.
        if (route_plan is not None
                and hasattr(route_plan, "offset_location_at")):
            spawn_location = route_plan.offset_location_at(
                self._route_s, lane_offset)
        else:
            spawn_location = carla.Location(
                x=base_tf.location.x + right.x * lane_offset,
                y=base_tf.location.y + right.y * lane_offset,
                z=base_tf.location.z,
            )
        self._spawn_transform = carla.Transform(
            spawn_location,
            carla.Rotation(pitch=base_tf.rotation.pitch,
                           yaw=base_tf.rotation.yaw,
                           roll=base_tf.rotation.roll),
        )
        # Since we teleport with physics disabled, z_offset determines whether wheels "touch the ground".
        # A small offset helps avoid clipping without making the vehicle "float".
        self._z_offset = float(z_offset)
        self._spawn_transform.location.z += self._z_offset

        # Road vectors (updated each step)
        self.road_fwd = self._spawn_transform.rotation.get_forward_vector()
        self.road_left = self._spawn_transform.rotation.get_right_vector()
        self.road_left.x *= -1
        self.road_left.y *= -1

        # Current rotation (updated each step to follow road direction)
        self._current_rotation = self._spawn_transform.rotation
        self._road_center_x = float(base_tf.location.x)
        self._road_center_y = float(base_tf.location.y)
        self._road_lane_width = float(getattr(spawn_wp, "lane_width", 3.5))

        # CARLA waypoints sit on the road surface, while some vehicle
        # blueprints (notably bmw.grandtourer and mercedes.coupe) require a
        # higher transform for the initial collision-checked spawn. Try a
        # small vertical lift only on/next to a driving lane, disable physics,
        # then teleport to the canonical road pose used by telemetry.
        bp = world.get_blueprint(model)
        base_lifts = (_SPAWN_LIFT_CANDIDATES_M
                      if abs(float(lane_offset)) <= 3.5 + 1e-6 else (0.0,))
        lift_cache = getattr(world, "_vehicle_spawn_lift_cache", None)
        if lift_cache is None:
            lift_cache = {}
            setattr(world, "_vehicle_spawn_lift_cache", lift_cache)
        cache_enabled = len(base_lifts) > 1
        cached_lift = lift_cache.get(model) if cache_enabled else None
        lifts = base_lifts
        if cached_lift in base_lifts:
            lifts = (cached_lift,) + tuple(
                value for value in base_lifts if value != cached_lift)
        self._actor = None
        last_collision = None
        selected_lift = 0.0
        for lift in lifts:
            candidate_transform = carla.Transform(
                carla.Location(
                    x=self._spawn_transform.location.x,
                    y=self._spawn_transform.location.y,
                    z=self._spawn_transform.location.z + lift,
                ),
                carla.Rotation(
                    pitch=self._spawn_transform.rotation.pitch,
                    yaw=self._spawn_transform.rotation.yaw,
                    roll=self._spawn_transform.rotation.roll,
                ),
            )
            try:
                self._actor = world.spawn_actor(bp, candidate_transform)
            except RuntimeError as exc:
                if "collision at spawn position" not in str(exc).lower():
                    raise
                last_collision = exc
                continue
            selected_lift = lift
            break
        if self._actor is None:
            raise RuntimeError(
                f"vehicle model {model} collided at spawn lifts {list(lifts)}; "
                f"last error: {last_collision}") from last_collision
        self._actor.set_simulate_physics(False)
        # Far-lateral scenery actors intentionally do not use vertical lift.
        # Their zero-lift success must not overwrite the model cache selected
        # for actors that spawn on or next to a driving lane.
        if cache_enabled:
            lift_cache[model] = selected_lift
        if selected_lift:
            self._actor.set_transform(self._spawn_transform)
            if selected_lift != cached_lift:
                print(f"[spawn lift] {model}: +{selected_lift:g} m transient lift")

        # Kinematic state
        self.x = self._spawn_transform.location.x
        self.y = self._spawn_transform.location.y
        self.z = self._spawn_transform.location.z
        self.vx = initial_speed     # Longitudinal speed (m/s) along road_fwd
        self.vy = 0.0               # Lateral speed (m/s) along road_left
        self.ax = 0.0               # Longitudinal acceleration (m/s²)

        # Lane change tracking (accumulated instead of calculated from spawn origin)
        self._accumulated_lateral = 0.0
        self._auto_stop_lateral = False
        self._lateral_limit = 3.5

        # Metadata
        self.model = model
        self.entity_type = 'vehicle'  # Override in subclass (ego/npc)
        self.lane_id = spawn_wp.lane_id if hasattr(spawn_wp, 'lane_id') else 0
        self.bounding_length = 4.5
        self.bounding_width = 2.0

    @property
    def speed_kmh(self):
        """Current speed in km/h."""
        return self.vx * 3.6

    @property
    def carla_actor(self):
        """Original CARLA actor."""
        return self._actor

    def get_lateral_displacement(self):
        """Accumulated lateral displacement (positive = left)."""
        return self._accumulated_lateral

    def update(self, dt):
        """Update kinematic state, follow road curves, teleport.

        A bound route advances by monotonic arc-length progress. Longitudinal
        speed can therefore never change junction continuation. Lateral actions
        remain offsets around the same route centreline.
        """
        previous_lateral_right = (
            self._route_lane_offset - self._accumulated_lateral)

        # 1. Update velocity
        self.vx += self.ax * dt
        if self.vx < 0:
            self.vx = 0.0

        # 2. Accumulate lateral + clamp if exceeding limit
        self._accumulated_lateral += self.vy * dt
        if self._auto_stop_lateral and abs(self._accumulated_lateral) >= self._lateral_limit:
            sign = 1.0 if self._accumulated_lateral > 0 else -1.0
            self._accumulated_lateral = sign * self._lateral_limit
            self.vy = 0.0
            self._auto_stop_lateral = False

        if self._route_plan is not None:
            # 3a. Route-bound protocol motion. Position is reconstructed from
            # route progress plus the declared/accumulated lateral offset.
            # Each parallel lane has a different physical arc length on a
            # curve, so advance on the actor's midpoint-offset path instead of
            # treating centreline metres as actor-path metres.
            lateral_right = (
                self._route_lane_offset - self._accumulated_lateral)
            midpoint_lateral_right = 0.5 * (
                previous_lateral_right + lateral_right)
            if hasattr(self._route_plan, "advance_along_offset"):
                self._route_s = self._route_plan.advance_along_offset(
                    self._route_s, self.vx * dt, midpoint_lateral_right)
            else:
                # Route-plan fakes without advance_along_offset.
                self._route_s += self.vx * dt
            route_pose = self._route_plan.pose_at(self._route_s)
            route_transform = route_pose.transform
            route_right = route_transform.rotation.get_right_vector()
            if hasattr(self._route_plan, "offset_location_at"):
                route_location = self._route_plan.offset_location_at(
                    self._route_s, lateral_right)
                self.x = route_location.x
                self.y = route_location.y
                self.z = route_location.z + self._z_offset
            else:
                self.x = (route_transform.location.x
                          + route_right.x * lateral_right)
                self.y = (route_transform.location.y
                          + route_right.y * lateral_right)
                self.z = route_transform.location.z + self._z_offset
            self.road_fwd = route_transform.rotation.get_forward_vector()
            self.road_left = route_right
            self.road_left.x *= -1
            self.road_left.y *= -1
            self._current_rotation = route_transform.rotation
            self.lane_id = route_pose.lane_id
            self._road_center_x = float(route_transform.location.x)
            self._road_center_y = float(route_transform.location.y)
            self._road_lane_width = float(route_pose.lane_width)
        else:
            # 3b. Motion for callers without a RoutePlan.
            self.x += self.road_fwd.x * self.vx * dt
            self.y += self.road_fwd.y * self.vx * dt
            self.x += self.road_left.x * self.vy * dt
            self.y += self.road_left.y * self.vy * dt
            carla_map = self._world.carla_world.get_map()
            current_wp = carla_map.get_waypoint(
                carla.Location(x=self.x, y=self.y, z=self.z)
            )
            if current_wp:
                candidate_fwd = current_wp.transform.rotation.get_forward_vector()
                current_yaw = float(self._current_rotation.yaw)
                candidate_yaw = float(current_wp.transform.rotation.yaw)
                yaw_delta = (candidate_yaw - current_yaw + 180.0) % 360.0 - 180.0
                max_yaw_delta = max(
                    _MIN_ROUTE_YAW_TOLERANCE_DEG,
                    _MAX_ROUTE_YAW_RATE_DEG_PER_SEC * float(dt),
                )
                if abs(yaw_delta) <= max_yaw_delta + 1e-6:
                    self.road_fwd = candidate_fwd
                    self.road_left = current_wp.transform.rotation.get_right_vector()
                    self.road_left.x *= -1
                    self.road_left.y *= -1
                    self._current_rotation = current_wp.transform.rotation
                    self.lane_id = (current_wp.lane_id
                                    if hasattr(current_wp, 'lane_id') else self.lane_id)
                    self._road_center_x = float(current_wp.transform.location.x)
                    self._road_center_y = float(current_wp.transform.location.y)
                    self._road_lane_width = float(
                        getattr(current_wp, "lane_width", self._road_lane_width))
                self.z = current_wp.transform.location.z + self._z_offset

        # 4. Teleport CARLA actor (rotation follows intended road direction)
        new_transform = carla.Transform(
            carla.Location(x=self.x, y=self.y, z=self.z),
            self._current_rotation
        )
        self._actor.set_transform(new_transform)

        # 6. Display brake lights.
        # Only activate when braking to preserve existing behavior.
        base = carla.VehicleLightState.Position
        if self.ax < 0:
            light_mask = carla.VehicleLightState(
                int(base) | int(carla.VehicleLightState.Brake)
            )
            self._actor.set_light_state(light_mask)
        else:
            self._actor.set_light_state(carla.VehicleLightState.NONE)

    def get_ground_truth(self):
        """Return normalized ground truth dict for JSON output.

        Unified schema for all actor types:
        x, y, vx, vy, ax, speed_kmh, lane_id, entity_type
        """
        # Positions are CARLA world coordinates, so velocity components must
        # be world-coordinate components too.  Returning longitudinal/lateral
        # scalars here made the later T0-ego transform rotate them a second
        # time on curved or non-axis-aligned roads.
        world_vx = self.road_fwd.x * self.vx + self.road_left.x * self.vy
        world_vy = self.road_fwd.y * self.vx + self.road_left.y * self.vy
        state = {
            "x": round(self.x, 3),
            "y": round(self.y, 3),
            "vx": round(world_vx, 3),
            "vy": round(world_vy, 3),
            "ax": round(self.ax, 3),
            "speed_kmh": round(self.speed_kmh, 1),
            "lane_id": self.lane_id,
            "entity_type": self.entity_type,
            "heading_deg": round(float(self._current_rotation.yaw), 3),
            # Raw-only route evidence. build_ground_truth turns these trusted
            # CARLA lane-centre samples into a curved scene corridor; actor
            # schema projection intentionally drops them from scored nodes.
            "road_center_x": round(self._road_center_x, 3),
            "road_center_y": round(self._road_center_y, 3),
            "road_lane_width": round(self._road_lane_width, 3),
        }
        if self._route_plan is not None:
            # Raw-only audit fields. They prove that all branches advanced on
            # the same immutable route, while actor-schema projection drops
            # them from the published scored nodes.
            state["route_s"] = round(self._route_s, 3)
            state["route_id"] = getattr(self._route_plan, "route_id", None)
            state["route_mode"] = getattr(self._route_plan, "route_mode", None)
        return state

    def destroy(self):
        """Destroy CARLA actor."""
        if self._actor and self._actor.is_alive:
            self._actor.destroy()


class Ego(_KinematicVehicle):
    """Ego vehicle — detailed kinematic state."""

    def __init__(self, world, model='vehicle.tesla.model3',
                 waypoint=None, initial_speed=20.0, lane_offset=0.0,
                 route_plan=None):
        super().__init__(world, model, waypoint, initial_speed,
                         distance_ahead=0.0, lane_offset=lane_offset,
                         z_offset=0.05, route_plan=route_plan)
        self.entity_type = 'ego'


class EgoActions(_VehicleActions):
    """Ego controls — inherits _VehicleActions.

    The ego swerve auto-stops at the PUBLISHED lane width
    (EGO_SWERVE_LANE_WIDTH_M = 3.5 m, simulation/protocol.py), not at the map's
    real lane width, so the CARLA reference realises the same lateral profile
    the prompt declares and the kinematic twin executes.
    """

    STEER_LATERAL_LIMIT_M = EGO_SWERVE_LANE_WIDTH_M

    def __init__(self, ego: Ego):
        super().__init__(ego)
