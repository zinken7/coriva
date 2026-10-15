import carla


class StaticProp:
    """Static object on the road — traffic cone, barrier, debris, etc.

    Has no kinematic state, no need for update() each frame.
    Only spawns at a fixed position and provides ground truth for the recorder.
    """

    PROP_TYPES = {
        # --- Traffic / Construction ---
        "traffic_cone": {
            "models": [
                "static.prop.constructioncone",
                "static.prop.trafficcone01",
                "static.prop.trafficcone02",
            ],
            "length": 0.4,
            "width": 0.4,
            "height": 0.7,
        },
        "street_barrier": {
            "models": [
                "static.prop.streetbarrier",
            ],
            "length": 1.5,
            "width": 0.5,
            "height": 1.0,
        },
        "warning_sign": {
            "models": [
                "static.prop.trafficwarning",
                "static.prop.warningaccident",
                "static.prop.warningconstruction",
            ],
            "length": 0.6,
            "width": 0.6,
            "height": 1.2,
        },
        "iron_plank": {
            "models": [
                "static.prop.ironplank",
            ],
            "length": 3.0,
            "width": 1.5,
            "height": 0.05,
        },
        # --- Debris / Obstacles ---
        "debris": {
            "models": [
                "static.prop.dirtdebris01",
                "static.prop.dirtdebris02",
                "static.prop.dirtdebris03",
            ],
            "length": 1.0,
            "width": 1.0,
            "height": 0.2,
        },
        "broken_tile": {
            "models": [
                "static.prop.brokentile01",
                "static.prop.brokentile02",
                "static.prop.brokentile03",
                "static.prop.brokentile04",
            ],
            "length": 0.8,
            "width": 0.8,
            "height": 0.1,
        },
        # --- Containers / Large obstacles ---
        "box": {
            "models": [
                "static.prop.box01",
                "static.prop.box02",
                "static.prop.box03",
            ],
            "length": 0.6,
            "width": 0.6,
            "height": 0.6,
        },
        "barrel": {
            "models": [
                "static.prop.barrel",
            ],
            "length": 0.6,
            "width": 0.6,
            "height": 0.9,
        },
        "garbage_container": {
            "models": [
                "static.prop.container",
            ],
            "length": 1.8,
            "width": 1.2,
            "height": 1.3,
        },
        "trash_can": {
            "models": [
                "static.prop.trashcan01",
                "static.prop.trashcan02",
                "static.prop.trashcan03",
                "static.prop.trashcan04",
                "static.prop.trashcan05",
            ],
            "length": 0.5,
            "width": 0.5,
            "height": 1.0,
        },
    }

    def __init__(self, world, prop_type='traffic_cone', model=None,
                 position=None, rotation=None):
        """
        Args:
            world: World instance.
            prop_type: Prop type (key in PROP_TYPES).
            model: Specific blueprint name. If None, uses the first model
                   from the prop_type's models list.
            position: carla.Location — spawn position.
            rotation: carla.Rotation — orientation. Default (0, 0, 0).
        """
        self._world = world

        # Look up prop type info
        type_info = self.PROP_TYPES.get(prop_type, self.PROP_TYPES["traffic_cone"])
        self.prop_type = prop_type
        self.entity_type = prop_type

        # Bounding dimensions
        self.bounding_length = type_info["length"]
        self.bounding_width = type_info["width"]
        self.bounding_height = type_info["height"]

        # Select model
        if model is None:
            model = type_info["models"][0]
        self.model = model

        # Fixed position
        if position is None:
            position = carla.Location(x=0, y=0, z=0)
        if rotation is None:
            rotation = carla.Rotation(pitch=0, yaw=0, roll=0)

        self.x = position.x
        self.y = position.y
        self.z = position.z
        # The spawn yaw is the prop's heading for the whole run (it never
        # moves); telemetry reports it so footprint checks can orient the
        # declared length/width instead of guessing.
        self.yaw = float(getattr(rotation, "yaw", 0.0))

        # Spawn CARLA actor
        spawn_transform = carla.Transform(position, rotation)
        bp = world.get_blueprint(model)
        self._actor = world.spawn_actor(bp, spawn_transform)
        if hasattr(self._actor, "set_simulate_physics"):
            self._actor.set_simulate_physics(False)

    @property
    def carla_actor(self):
        """Original CARLA actor."""
        return self._actor

    def get_ground_truth(self):
        """Return normalized ground truth dict.

        Schema compatible with vehicle/walker — kinematic fields = 0
        since the prop is stationary.
        """
        return {
            "x": round(self.x, 3),
            "y": round(self.y, 3),
            "vx": 0.0,
            "vy": 0.0,
            "ax": 0.0,
            "speed_kmh": 0.0,
            "heading_deg": round(self.yaw, 3),
            "lane_id": None,
            "entity_type": self.entity_type,
        }

    def destroy(self):
        """Destroy CARLA actor."""
        if self._actor and self._actor.is_alive:
            self._actor.destroy()
