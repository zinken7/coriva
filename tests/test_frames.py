"""Model-runner core — deterministic uniform frame sampling.

Video-LMs frame-sample internally anyway; for a benchmark we pre-select a fixed,
deterministic set so every model sees the SAME frames (fair + reproducible) and
the vision-token/VRAM budget is bounded. Pure list logic — no image/video deps.
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


def test_select_frames_uniform_span():
    from analysis.frames import select_frames
    f = [f"f{i}" for i in range(10)]
    assert select_frames(f, 4) == ["f0", "f3", "f6", "f9"]


def test_select_frames_includes_endpoints_and_exact_count():
    from analysis.frames import select_frames
    r = select_frames([f"f{i}" for i in range(10)], 8)
    assert len(r) == 8 and r[0] == "f0" and r[-1] == "f9"
    # monotonic (order preserved)
    idx = [int(x[1:]) for x in r]
    assert idx == sorted(idx)


def test_select_frames_fewer_than_requested_returns_all():
    from analysis.frames import select_frames
    assert select_frames(["a", "b", "c"], 8) == ["a", "b", "c"]


def test_select_frames_single_returns_middle():
    from analysis.frames import select_frames
    assert select_frames([f"f{i}" for i in range(5)], 1) == ["f2"]


def test_select_frames_deterministic():
    from analysis.frames import select_frames
    f = [f"f{i}" for i in range(37)]
    assert select_frames(f, 8) == select_frames(f, 8)


def test_select_frames_empty():
    from analysis.frames import select_frames
    assert select_frames([], 8) == []
