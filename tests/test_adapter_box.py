"""nuScenes box dimensions -> per-actor axis-aligned body extent.

nuScenes annotations carry a 3D box `size` ([w, l, h]) and a global `rotation`.
The adapter projects that oriented box onto the ego-aligned axes and stores the
resulting `length` (x) and `width` (y) on each actor, so the twin's collision
footprint matches the real vehicle (a 12 m truck, not a generic 4 m gap).
"""

import math
import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


def test_box_extent_aligned_with_ego():
    from simulation.nuscenes_adapter import box_extent_in_ego_frame
    # size = [w=2, l=4, h=1.5]; box heads the same way as ego.
    length, width = box_extent_in_ego_frame([2.0, 4.0, 1.5], box_yaw=0.0, ego_yaw=0.0)
    assert length == pytest.approx(4.0) and width == pytest.approx(2.0)


def test_box_extent_perpendicular_to_ego():
    from simulation.nuscenes_adapter import box_extent_in_ego_frame
    # Box rotated 90deg vs ego -> length and width of the AABB swap.
    length, width = box_extent_in_ego_frame([2.0, 4.0, 1.5], box_yaw=math.pi / 2, ego_yaw=0.0)
    assert length == pytest.approx(2.0) and width == pytest.approx(4.0)


def test_box_extent_uses_relative_yaw():
    from simulation.nuscenes_adapter import box_extent_in_ego_frame
    # Equal absolute yaws cancel: a box aligned with a rotated ego is axis-aligned.
    length, width = box_extent_in_ego_frame([2.0, 4.0, 1.5], box_yaw=0.7, ego_yaw=0.7)
    assert length == pytest.approx(4.0) and width == pytest.approx(2.0)


def test_box_extent_45_degrees():
    from simulation.nuscenes_adapter import box_extent_in_ego_frame
    length, width = box_extent_in_ego_frame([2.0, 4.0, 1.5], box_yaw=math.pi / 4, ego_yaw=0.0)
    expected = (2.0 + 4.0) / 2 * math.sqrt(2)  # |hl*c|+|hw*s| doubled = (l+w)/2*√2
    assert length == pytest.approx(expected, abs=1e-3) and width == pytest.approx(expected, abs=1e-3)
