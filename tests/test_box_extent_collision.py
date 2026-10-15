"""Per-actor box footprint in the kinematic twin.

The twin keeps actors as point centers but carries a rectangular body extent
(`length` along x, `width` along y, axis-aligned in the ego frame). Collision is
then footprint-aware: it fires when the centers are within the SUM of half-extents
on each axis, instead of a single global collision_gap/lane_halfwidth. Actors
without dims fall back to the old fixed gaps (backward compatible).
"""

import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


def _stat(role, x, y, **extra):
    a = {"id": "ego" if role == "ego_agent" else "npc_1", "entity_type": "vehicle",
         "role_type": role, "x": x, "y": y, "vx": 10.0, "vy": 0.0, "speed": 10.0}
    a.update(extra)
    return a


# ── footprint-aware collision ─────────────────────────────────────────
# Contact is located between samples and the collision node is the last
# sampled state at or before contact. Each fixture therefore has a
# far root, a first sample still clear of the other actor and a second sample
# in the tested geometry, so the tested geometry decides whether index 0 is a
# collision node.

def _two_step(ego_far, other_far, ego_near, other_near):
    root = [dict(ego_far), dict(other_far)]
    states = [{"ego": dict(ego_far), "npc_1": dict(other_far)},
              {"ego": dict(ego_near), "npc_1": dict(other_near)}]
    return states, root


def test_collision_uses_per_actor_length_extent():
    from simulation.kinematic_twin import _detect_terminal
    # ego (len 4) and a long truck (len 12), centers 7 m apart in x at the
    # second sample. Fixed gap 4.0 would NOT fire (7 > 4); summed
    # half-lengths (4+12)/2 = 8 do.
    ego_far = _stat("ego_agent", 25.0, 0.0, length=4.0, width=1.8)   # 15 m clear
    truck = _stat("lead_vehicle", 40.0, 0.0, length=12.0, width=2.5)
    ego_near = _stat("ego_agent", 36.0, 0.0, length=4.0, width=1.8)  # 4 m: inside 8, contact between samples
    states, root = _two_step(ego_far, truck, ego_near, truck)
    ev, idx = _detect_terminal(states, root, collision_gap=4.0, lane_halfwidth=2.0)
    assert ev == "collision" and idx == 0


def test_collision_falls_back_to_fixed_gap_without_extents():
    from simulation.kinematic_twin import _detect_terminal
    # Same geometry but NO dims -> fixed gap: 4 is not < collision_gap 4 -> no collision.
    ego_far = _stat("ego_agent", 25.0, 0.0)
    truck = _stat("lead_vehicle", 40.0, 0.0)
    ego_near = _stat("ego_agent", 36.0, 0.0)  # 4 m is not < collision_gap 4.0
    states, root = _two_step(ego_far, truck, ego_near, truck)
    ev, _ = _detect_terminal(states, root, collision_gap=4.0, lane_halfwidth=2.0)
    assert ev != "collision"


def test_lateral_extent_respects_width():
    from simulation.kinematic_twin import _detect_terminal
    # Centers 1.8 m apart laterally, aligned in x at the second sample. Two
    # narrow cars width 1.6 -> summed half-widths (1.6+1.6)/2 = 1.6 < 1.8 ->
    # NO collision despite |dy| < 2.0.
    ego_far = _stat("ego_agent", 0.0, 0.0, length=4.0, width=1.6)
    car = _stat("adjacent_vehicle", 40.0, 1.8, length=4.0, width=1.6)
    ego_near = _stat("ego_agent", 40.0, 0.0, length=4.0, width=1.6)
    states, root = _two_step(ego_far, car, ego_near, car)
    ev, _ = _detect_terminal(states, root, collision_gap=4.0, lane_halfwidth=2.0)
    assert ev != "collision"


# ── step functions carry the body dims forward ───────────────────────
def test_step_actor_preserves_box_dims():
    from simulation.kinematic_twin import step_actor
    a = _stat("lead_vehicle", 0.0, 0.0, length=12.0, width=2.5)
    nxt = step_actor(a, accel=0.0, lat_v=0.0, dt=0.5)
    assert nxt["length"] == pytest.approx(12.0) and nxt["width"] == pytest.approx(2.5)


def test_step_actor_bicycle_preserves_box_dims():
    from simulation.kinematic_twin import step_actor_bicycle
    a = _stat("ego_agent", 0.0, 0.0, length=4.0, width=1.8)
    nxt = step_actor_bicycle(a, accel=0.0, steer=0.0, dt=0.5)
    assert nxt["length"] == pytest.approx(4.0) and nxt["width"] == pytest.approx(1.8)


def test_step_actor_omits_dims_when_absent():
    from simulation.kinematic_twin import step_actor
    nxt = step_actor(_stat("lead_vehicle", 0.0, 0.0), accel=0.0, lat_v=0.0, dt=0.5)
    assert "length" not in nxt and "width" not in nxt
