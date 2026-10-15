"""Geometry-based role assignment in the nuScenes adapter (pure)."""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


def test_vehicle_ahead_in_lane_is_lead():
    from simulation.nuscenes_adapter import assign_role
    assert assign_role("vehicle", x=30.0, y=0.5, lane_halfwidth=2.0) == "lead_vehicle"


def test_vehicle_ahead_off_lane_is_adjacent():
    from simulation.nuscenes_adapter import assign_role
    assert assign_role("vehicle", x=30.0, y=4.0, lane_halfwidth=2.0) == "adjacent_vehicle"


def test_vehicle_behind_is_rear():
    from simulation.nuscenes_adapter import assign_role
    assert assign_role("vehicle", x=-15.0, y=0.0, lane_halfwidth=2.0) == "rear_vehicle"


def test_pedestrian_and_bicycle_are_crossing():
    from simulation.nuscenes_adapter import assign_role
    assert assign_role("pedestrian", x=10.0, y=8.0, lane_halfwidth=2.0) == "crossing_actor"
    assert assign_role("bicycle", x=-5.0, y=3.0, lane_halfwidth=2.0) == "crossing_actor"


def test_static_entities_get_static_roles():
    from simulation.nuscenes_adapter import assign_role
    # barrier/cone in the ego path -> obstacle; off to the side -> static barrier
    assert assign_role("barrier", x=20.0, y=0.5, lane_halfwidth=2.0) == "obstacle_actor"
    assert assign_role("traffic_cone", x=20.0, y=5.0, lane_halfwidth=2.0) == "static_barrier"
    assert assign_role("guardrail", x=10.0, y=7.0, lane_halfwidth=2.0) == "road_boundary"
    assert assign_role("traffic_sign", x=10.0, y=6.0, lane_halfwidth=2.0) == "static_barrier"


def test_category_to_entity_maps_static_nuscenes_objects():
    from simulation.nuscenes_adapter import category_to_entity
    assert category_to_entity("movable_object.barrier") == "barrier"
    assert category_to_entity("movable_object.trafficcone") == "traffic_cone"
    assert category_to_entity("vehicle.car") == "vehicle"
    assert category_to_entity("static_object.bicycle_rack") is None  # unmodeled -> skipped
