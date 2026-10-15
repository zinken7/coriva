import carla


class Pedestrian:
    """Pedestrian — kinematic state + animation.

    Spawns on the sidewalk (left/right) and walks across the road.
    Refer to WALKER_TYPES to select a model by group (adult, child, police).
    """

    WALKER_TYPES = {
        "adult": {
            "models": [
                "walker.pedestrian.0001",
                "walker.pedestrian.0002",
                "walker.pedestrian.0003",
                "walker.pedestrian.0004",
                "walker.pedestrian.0005",
                "walker.pedestrian.0006",
                "walker.pedestrian.0007",
                "walker.pedestrian.0008",
                "walker.pedestrian.0015",
                "walker.pedestrian.0016",
                "walker.pedestrian.0017",
                "walker.pedestrian.0018",
                "walker.pedestrian.0019",
                "walker.pedestrian.0020",
                "walker.pedestrian.0021",
                "walker.pedestrian.0022",
                "walker.pedestrian.0023",
                "walker.pedestrian.0024",
                "walker.pedestrian.0025",
                "walker.pedestrian.0026",
                "walker.pedestrian.0027",
                "walker.pedestrian.0028",
                "walker.pedestrian.0029",
                "walker.pedestrian.0031",
                "walker.pedestrian.0033",
                "walker.pedestrian.0034",
                "walker.pedestrian.0035",
                "walker.pedestrian.0036",
                "walker.pedestrian.0037",
                "walker.pedestrian.0038",
                "walker.pedestrian.0039",
                "walker.pedestrian.0040",
                "walker.pedestrian.0041",
                "walker.pedestrian.0042",
                "walker.pedestrian.0043",
                "walker.pedestrian.0044",
                "walker.pedestrian.0045",
                "walker.pedestrian.0046",
                "walker.pedestrian.0047",
            ],
        },
        "child": {
            "models": [
                "walker.pedestrian.0009",
                "walker.pedestrian.0010",
                "walker.pedestrian.0011",
                "walker.pedestrian.0012",
                "walker.pedestrian.0013",
                "walker.pedestrian.0014",
                "walker.pedestrian.0048",
                "walker.pedestrian.0049",
            ],
        },
        "police": {
            "models": [
                "walker.pedestrian.0030",
                "walker.pedestrian.0032",
            ],
        },
    }

    def __init__(self, world, model='walker.pedestrian.0001',
                 road_waypoint=None, distance_ahead=100.0,
                 side='left', offset=6.0, facing='cross_right'):
        """
        Args:
            world: World instance.
            model: Blueprint name.
            road_waypoint: Base waypoint on the road (Ego position).
            distance_ahead: Distance ahead of the base waypoint (m).
            side: 'left' or 'right' — which sidewalk.
            offset: Distance from road center to sidewalk (m).
            facing: 'cross_right' or 'cross_left' — crossing direction.
        """
        self._world = world

        # Calculate position on the road ahead
        if distance_ahead > 0:
            candidates = road_waypoint.next(distance_ahead)
            if not candidates:
                raise ValueError(f"no waypoint {distance_ahead:.1f} m ahead")
            ped_waypoint = candidates[0]
        elif distance_ahead < 0:
            candidates = road_waypoint.previous(abs(distance_ahead))
            if not candidates:
                raise ValueError(f"no waypoint {abs(distance_ahead):.1f} m behind")
            ped_waypoint = candidates[0]
        else:
            ped_waypoint = road_waypoint

        # Calculate road vectors at pedestrian position
        road_right = ped_waypoint.transform.rotation.get_right_vector()
        road_left = carla.Vector3D(-road_right.x, -road_right.y, 0)

        # Cross direction (road crossing direction)
        if facing == 'cross_right':
            self.cross_dir = carla.Vector3D(road_right.x, road_right.y, 0)
        else:
            self.cross_dir = carla.Vector3D(road_left.x, road_left.y, 0)

        # Spawn position on the sidewalk
        if side == 'left':
            side_vec = road_left
        else:
            side_vec = road_right

        ped_loc = ped_waypoint.transform.location + carla.Location(
            x=side_vec.x * offset,
            y=side_vec.y * offset,
            z=1.0
        )

        # Rotation: walker faces cross_dir
        ped_rotation = ped_waypoint.transform.rotation
        if facing == 'cross_right':
            ped_rotation.yaw += 90.0
        else:
            ped_rotation.yaw -= 90.0

        self._spawn_transform = carla.Transform(ped_loc, ped_rotation)

        # Spawn CARLA walker
        bp = world.get_blueprint(model)
        self._actor = world.spawn_actor(bp, self._spawn_transform)

        # Kinematic state
        self.x = ped_loc.x
        self.y = ped_loc.y
        self.z = ped_loc.z
        self.speed = 0.0
        self.direction = self.cross_dir
        self.is_moving = False

        # Metadata
        self.model = model
        self.side = side
        self.bounding_length = 0.6
        self.bounding_width = 0.6

    @property
    def carla_actor(self):
        return self._actor

    def update(self, dt):
        """Update kinematic position + teleport + walking animation."""
        if not self.is_moving or self.speed <= 0:
            return

        # Kinematic update
        self.x += self.direction.x * self.speed * dt
        self.y += self.direction.y * self.speed * dt

        # Teleport
        walker_transform = carla.Transform(
            carla.Location(x=self.x, y=self.y, z=self.z),
            self._spawn_transform.rotation
        )
        self._actor.set_transform(walker_transform)

        # Walking animation
        self._actor.apply_control(carla.WalkerControl(
            direction=carla.Vector3D(self.direction.x, self.direction.y, 0),
            speed=self.speed
        ))

    def get_ground_truth(self):
        """Return normalized ground truth dict (same schema as vehicle).

        vx, vy are resolved from speed + direction.
        ax = 0 (walker has no kinematic acceleration).
        """
        vx = self.speed * self.direction.x if self.is_moving else 0.0
        vy = self.speed * self.direction.y if self.is_moving else 0.0
        return {
            "x": round(self.x, 3),
            "y": round(self.y, 3),
            "vx": round(vx, 3),
            "vy": round(vy, 3),
            "ax": 0.0,
            "speed_kmh": round(self.speed * 3.6, 1),
            "lane_id": None,
            "entity_type": "walker",
        }

    def destroy(self):
        if self._actor and self._actor.is_alive:
            self._actor.destroy()


class PedestrianActions:
    """Pedestrian controls."""

    def __init__(self, pedestrian: Pedestrian):
        self.ped = pedestrian

    def maintain_direction(self):
        """Maintain current direction + speed."""
        pass  # No changes — state is preserved

    def stop(self):
        """Stop moving."""
        self.ped.speed = 0.0
        self.ped.is_moving = False

    def change_direction(self, direction_vector, speed=None):
        """Change movement direction.

        Args:
            direction_vector: carla.Vector3D — new direction.
            speed: New speed (m/s). If None, keep current speed.
        """
        self.ped.direction = direction_vector
        if speed is not None:
            self.ped.speed = speed
        self.ped.is_moving = True

    def start_crossing(self, speed=2.0):
        """Start crossing the road along the cross_dir set at spawn time.

        Args:
            speed: Walking speed (m/s). Default 2.0 m/s.
        """
        self.ped.direction = self.ped.cross_dir
        self.ped.speed = speed
        self.ped.is_moving = True
