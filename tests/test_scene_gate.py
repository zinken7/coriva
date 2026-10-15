"""Regression coverage for the implemented Layer-3 scene geometry gate."""

import os
import sys

import yaml

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


def _actor(x, y):
    return {"id": "ego", "entity_type": "ego_vehicle", "role_type": "ego_agent",
            "x": x, "y": y, "vx": 2.0, "vy": 0.0, "speed": 2.0}


def _branch(points):
    return {"chain": [
        {"depth": i, "timestamp": i * 0.5, "event_type": "maintain_course",
         "action_type": "ego_no_action", "actors_update": [_actor(x, y)]}
        for i, (x, y) in enumerate(points, start=1)
    ]}


def _root(x=0.0, y=0.0):
    return {"event_type": "observed_state", "timestamp": 0.0,
            "actors": [_actor(x, y)]}


def _config(**scene_overrides):
    scene = {
        "check_offroad_outside_drivable_area": False,
        "check_static_boundary_crossing": False,
        "check_impossible_teleportation": False,
        "check_kinematic_violation": False,
        "local_drivable_half_width_m": 4.0,
    }
    scene.update(scene_overrides)
    return {"scene_gate": scene}


def test_shipped_config_enables_both_geometry_checks():
    with open(os.path.join("ctms", "ctms_config.yaml"), encoding="utf-8") as handle:
        scene = yaml.safe_load(handle)["scene_gate"]
    assert scene["check_offroad_outside_drivable_area"] is True
    assert scene["check_static_boundary_crossing"] is True


def test_offroad_gate_rejects_path_outside_local_corridor():
    from ctms.ctms import check_scene_validity

    ok, errors = check_scene_validity(
        _branch([(2.0, 0.0), (4.0, 5.0)]),
        _config(check_offroad_outside_drivable_area=True), root=_root())
    assert ok is False
    assert any("outside local drivable corridor" in error for error in errors)


def test_offroad_gate_uses_drivable_polygons_when_provided():
    from ctms.ctms import check_scene_validity

    map_data = {"drivable_polygons": [
        [[-1.0, -1.0], [5.0, -1.0], [5.0, 1.0], [-1.0, 1.0]]
    ]}
    ok, errors = check_scene_validity(
        _branch([(2.0, 0.0), (4.0, 2.0)]),
        _config(check_offroad_outside_drivable_area=True),
        map_data=map_data, root=_root())
    assert ok is False
    assert any("outside drivable area" in error for error in errors)


def test_offroad_gate_follows_curved_route_evidence():
    from ctms.ctms import check_scene_validity

    map_data = {
        "route_centerline": [[0.0, 0.0], [4.0, 0.0], [6.0, 2.0], [6.0, 6.0]],
        "route_half_width_m": 1.0,
    }
    ok, errors = check_scene_validity(
        _branch([(3.0, 0.0), (5.5, 1.5), (6.0, 4.0)]),
        _config(check_offroad_outside_drivable_area=True),
        map_data=map_data, root=_root())
    assert ok is True, errors


def test_curved_route_gate_still_rejects_genuinely_offroad_path():
    from ctms.ctms import check_scene_validity

    map_data = {
        "route_centerline": [[0.0, 0.0], [4.0, 0.0], [6.0, 2.0], [6.0, 6.0]],
        "route_half_width_m": 1.0,
    }
    ok, errors = check_scene_validity(
        _branch([(3.0, 0.0), (3.0, 3.0)]),
        _config(check_offroad_outside_drivable_area=True),
        map_data=map_data, root=_root())
    assert ok is False
    assert any("outside route drivable corridor" in error for error in errors)


def test_fully_covered_route_rejects_nine_point_five_metre_lateral_departure():
    from ctms.ctms import check_scene_validity

    map_data = {
        "route_centerline": [[0.0, 0.0], [20.0, 0.0]],
        "route_half_width_m": 8.0,
    }
    ok, errors = check_scene_validity(
        _branch([(5.0, 0.0), (10.0, 9.5)]),
        _config(check_offroad_outside_drivable_area=True),
        map_data=map_data, root=_root())
    assert ok is False
    assert any("outside route drivable corridor" in error for error in errors)


def test_route_endpoint_overrun_is_incomplete_not_lateral_offroad():
    from ctms.ctms import check_scene_validity

    map_data = {
        "route_centerline": [[0.0, 0.0], [10.0, 0.0]],
        "route_half_width_m": 8.0,
    }
    ok, errors = check_scene_validity(
        _branch([(5.0, 0.0), (20.0, 0.0)]),
        _config(check_offroad_outside_drivable_area=True),
        map_data=map_data, root=_root())
    assert ok is False
    assert any("route evidence incomplete" in error for error in errors)
    assert not any("outside route drivable corridor" in error for error in errors)


def test_curved_route_truncation_is_reported_as_incomplete():
    from ctms.ctms import check_scene_validity

    map_data = {
        "route_centerline": [[0.0, 0.0], [4.0, 0.0], [6.0, 2.0]],
        "route_half_width_m": 8.0,
    }
    ok, errors = check_scene_validity(
        _branch([(3.0, 0.0), (5.0, 1.0), (9.0, 5.0)]),
        _config(check_offroad_outside_drivable_area=True),
        map_data=map_data, root=_root())
    assert ok is False
    assert any("unsupported_after_end" in error for error in errors)
    assert not any("outside route drivable corridor" in error for error in errors)


def test_polyline_corner_vertex_counts_as_supported_route_coverage():
    from ctms.ctms import check_scene_validity

    map_data = {
        "route_centerline": [[0.0, 0.0], [5.0, 0.0], [5.0, 5.0]],
        "route_half_width_m": 1.0,
    }
    ok, errors = check_scene_validity(
        _branch([(4.0, 0.0), (5.5, -0.5)]),
        _config(check_offroad_outside_drivable_area=True),
        map_data=map_data, root=_root())
    assert ok is True, errors


def test_declared_incomplete_route_horizon_fails_closed():
    from ctms.ctms import check_scene_validity

    map_data = {
        "route_centerline": [[0.0, 0.0], [20.0, 0.0]],
        "route_half_width_m": 8.0,
        "route_coverage_m": 20.0,
        "route_required_m": 40.0,
        "route_coverage_complete": False,
    }
    ok, errors = check_scene_validity(
        _branch([(5.0, 0.0), (10.0, 0.0)]),
        _config(check_offroad_outside_drivable_area=True),
        map_data=map_data, root=_root())
    assert ok is False
    assert errors == [
        "scene: route evidence incomplete for required horizon "
        "(sampled=20.0m, required=40.0m)"
    ]


def test_curved_route_prevents_false_root_boundary_crossing():
    from ctms.ctms import check_scene_validity

    root = _root()
    root["actors"].append({
        "id": "wall_l", "entity_type": "guardrail", "role_type": "road_boundary",
        "x": 2.0, "y": 3.0, "vx": 0.0, "vy": 0.0, "speed": 0.0,
    })
    map_data = {
        "route_centerline": [[0.0, 0.0], [4.0, 0.0], [6.0, 2.0], [6.0, 6.0]],
        "route_half_width_m": 4.0,
    }
    ok, errors = check_scene_validity(
        _branch([(3.0, 0.0), (5.5, 1.5), (6.0, 4.0)]),
        _config(check_static_boundary_crossing=True),
        map_data=map_data, root=root)
    assert ok is True, errors


def test_ground_truth_builds_t0_aligned_route_geometry_from_carla_waypoints():
    from simulation.build_ground_truth import _build_route_scene_geometry

    config = {"actor_defs": {
        "ego": {"canonical_id": "ego", "entity_type": "ego_vehicle"},
    }}
    raw = {"branches": {"factual": [
        {"ego": {"road_center_x": 10.0, "road_center_y": 20.0,
                 "road_lane_width": 4.0}},
        {"ego": {"road_center_x": 10.0, "road_center_y": 22.0,
                 "road_lane_width": 4.0}},
        {"ego": {"road_center_x": 9.0, "road_center_y": 24.0,
                 "road_lane_width": 4.0}},
    ]}}
    geometry = _build_route_scene_geometry(
        raw, config, reference=(10.0, 20.0, 3.141592653589793 / 2.0))

    assert geometry["source"] == "carla_waypoint_route"
    assert geometry["route_half_width_m"] == 8.0
    assert geometry["route_centerlines"][0] == [
        [0.0, 0.0], [2.0, 0.0], [4.0, 1.0],
    ]


def test_ground_truth_prefers_branch_independent_carla_map_route():
    from simulation.build_ground_truth import _build_route_scene_geometry

    config = {"actor_defs": {
        "ego": {"canonical_id": "ego", "entity_type": "ego_vehicle"},
    }}
    raw = {
        "_route_evidence": {
            "coordinate_frame": "world",
            "source": "carla_map_intended_route",
            "route_mode": "single_deterministic_intended_route",
            "route_id": "0123456789abcdef",
            "route_centerline": [[10.0, 20.0], [10.0, 24.0], [9.0, 28.0]],
            "route_lane_width_m": 3.5,
            "route_coverage_m": 8.123,
            "route_required_m": 8.0,
            "route_backward_coverage_m": 6.0,
            "route_backward_required_m": 5.25,
            "sample_spacing_m": 2.0,
        },
        # Deliberately bad/frozen branch telemetry must not influence the route.
        "branches": {"factual": [
            {"ego": {"road_center_x": 999.0, "road_center_y": 999.0,
                     "road_lane_width": 20.0}},
        ]},
    }
    geometry = _build_route_scene_geometry(
        raw, config, reference=(10.0, 20.0, 3.141592653589793 / 2.0))

    assert geometry["source"] == "carla_map_intended_route"
    assert geometry["route_mode"] == "single_deterministic_intended_route"
    assert geometry["route_id"] == "0123456789abcdef"
    assert geometry["route_half_width_m"] == 8.0
    assert geometry["route_coverage_complete"] is True
    assert geometry["route_backward_coverage_complete"] is True
    assert geometry["route_backward_required_m"] == 5.25
    assert geometry["route_centerlines"] == [[
        [0.0, 0.0], [4.0, 0.0], [8.0, 1.0],
    ]]


def test_static_boundary_gate_rejects_swept_crossing():
    from ctms.ctms import check_scene_validity

    map_data = {"static_boundary_segments": [
        [[2.0, -2.0], [2.0, 2.0]]
    ]}
    ok, errors = check_scene_validity(
        _branch([(1.0, 0.0), (3.0, 0.0)]),
        _config(check_static_boundary_crossing=True),
        map_data=map_data, root=_root())
    assert ok is False
    assert any("crosses static boundary segment" in error for error in errors)


def test_static_boundary_gate_uses_root_road_boundary_actor():
    from ctms.ctms import check_scene_validity

    root = _root()
    root["actors"].append({
        "id": "wall_left", "entity_type": "guardrail", "role_type": "road_boundary",
        "x": 5.0, "y": 2.0, "vx": 0.0, "vy": 0.0, "speed": 0.0,
    })
    ok, errors = check_scene_validity(
        _branch([(1.0, 0.0), (2.0, 3.0)]),
        _config(check_static_boundary_crossing=True), root=root)
    assert ok is False
    assert any("root static boundary actor=wall_left" in error for error in errors)


def test_scene_gate_accepts_path_inside_polygon_without_crossing():
    from ctms.ctms import check_scene_validity

    map_data = {
        "drivable_polygons": [[[-1.0, -2.0], [6.0, -2.0], [6.0, 2.0], [-1.0, 2.0]]],
        "static_boundary_segments": [[[0.0, 3.0], [6.0, 3.0]]],
    }
    ok, errors = check_scene_validity(
        _branch([(1.0, 0.0), (3.0, 0.0), (5.0, 0.0)]),
        _config(check_offroad_outside_drivable_area=True,
                check_static_boundary_crossing=True),
        map_data=map_data, root=_root())
    assert ok is True, errors
