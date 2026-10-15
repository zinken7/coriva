import carla
from .ego import _KinematicVehicle, _VehicleActions


class NPC(_KinematicVehicle):
    """NPC vehicle — car, truck, van, bus, motorcycle, bicycle.

    Inherits _KinematicVehicle, adds vehicle type info and bounding dimensions.
    """

    VEHICLE_TYPES = {
        "car": {
            "models": [
                # --- Generation 2 ---
                "vehicle.dodge.charger_2020",
                "vehicle.dodge.charger_police_2020",
                "vehicle.ford.crown",
                "vehicle.lincoln.mkz_2020",
                "vehicle.mercedes.coupe_2020",
                "vehicle.mini.cooper_s_2021",
                "vehicle.nissan.patrol_2021",
                # --- Generation 1 ---
                "vehicle.audi.a2",
                "vehicle.audi.etron",
                "vehicle.audi.tt",
                "vehicle.bmw.grandtourer",
                "vehicle.chevrolet.impala",
                "vehicle.citroen.c3",
                "vehicle.dodge.charger_police",
                "vehicle.ford.mustang",
                "vehicle.jeep.wrangler_rubicon",
                "vehicle.lincoln.mkz_2017",
                "vehicle.mercedes.coupe",
                "vehicle.micro.microlino",
                "vehicle.mini.cooper_s",
                "vehicle.nissan.micra",
                "vehicle.nissan.patrol",
                "vehicle.seat.leon",
                "vehicle.tesla.model3",
                "vehicle.toyota.prius",
            ],
            "length": 4.5,
            "width": 2.0,
        },
        "truck": {
            "models": [
                "vehicle.carlamotors.european_hgv",
                "vehicle.carlamotors.firetruck",
                "vehicle.tesla.cybertruck",
                "vehicle.carlamotors.carlacola",
            ],
            "length": 7.0,
            "width": 2.5,
        },
        "motorcycle": {
            "models": [
                "vehicle.harley-davidson.low_rider",
                "vehicle.yamaha.yzf",
                "vehicle.kawasaki.ninja",
                "vehicle.vespa.zx125",
            ],
            "length": 2.2,
            "width": 0.8,
        },
        "bicycle": {
            "models": [
                "vehicle.bh.crossbike",
                "vehicle.diamondback.century",
                "vehicle.gazelle.omafiets",
            ],
            "length": 1.8,
            "width": 0.6,
        },
        "van": {
            "models": [
                "vehicle.ford.ambulance",
                "vehicle.mercedes.sprinter",
                "vehicle.volkswagen.t2",
                "vehicle.volkswagen.t2_2021",
            ],
            "length": 6.0,
            "width": 2.3,
        },
        "bus": {
            "models": [
                "vehicle.mitsubishi.fusorosa",
            ],
            "length": 12.0,
            "width": 2.5,
        },
    }

    def __init__(self, world, vehicle_type='car',
                 model='vehicle.audi.tt', waypoint=None,
                 initial_speed=20.0, distance_ahead=0.0, lane_offset=0.0,
                 z_offset=0.05, route_plan=None):
        """
        Args:
            world: World instance.
            vehicle_type: Vehicle type ('car', 'truck', 'van', 'bus', 'motorcycle', 'bicycle').
            model: Specific blueprint name.
            waypoint: Base CARLA waypoint (Ego position).
            initial_speed: Initial speed (m/s).
            distance_ahead: Distance ahead of waypoint to spawn (m).
            lane_offset: Lateral offset from lane center (m, + = right) —
                adjacent-lane / parked-vehicle casts in scenario configs.
            z_offset: Z-axis offset from road surface when teleporting (physics disabled).
            route_plan: Immutable intended route shared by all rollouts.
        """
        super().__init__(world, model, waypoint, initial_speed,
                         distance_ahead=distance_ahead, lane_offset=lane_offset,
                         z_offset=z_offset, route_plan=route_plan)

        self.vehicle_type = vehicle_type
        self.entity_type = vehicle_type  # Override base class 'vehicle' → 'car'/'truck'/etc.

        # Bounding dimensions from VEHICLE_TYPES
        type_info = self.VEHICLE_TYPES.get(vehicle_type, self.VEHICLE_TYPES["car"])
        self.bounding_length = type_info["length"]
        self.bounding_width = type_info["width"]


class NPCActions(_VehicleActions):
    """NPC controls — inherits _VehicleActions."""

    def __init__(self, npc: NPC):
        super().__init__(npc)
