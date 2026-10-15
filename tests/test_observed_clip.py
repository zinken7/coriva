"""
Extract an observed review video clip (real nuScenes CAM_FRONT
frames from scene start up to T=0). Skips if v1.0-mini is absent.
"""

import os
import shutil
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

DATAROOT = "data/nuscenes"
MINI = os.path.join(DATAROOT, "v1.0-nuscenes" if False else "v1.0-mini")
pytestmark = pytest.mark.skipif(
    not os.path.isdir(MINI), reason="nuScenes v1.0-mini not downloaded")


def _sample_a_few_steps_in(tables, steps=3):
    scene = next(iter(tables["scene"].values()))
    tok = scene["first_sample_token"]
    for _ in range(steps):
        nxt = tables["sample"][tok]["next"]
        if not nxt:
            break
        tok = nxt
    return tok


def test_collect_cam_frames_ordered_and_exist():
    from simulation.nuscenes_adapter import load_tables
    from simulation.observed_clip import collect_cam_frames
    tables = load_tables(DATAROOT, "v1.0-mini")
    tok = _sample_a_few_steps_in(tables, 3)
    frames = collect_cam_frames(tables, tok)
    assert len(frames) >= 2
    for fn in frames:
        assert os.path.exists(os.path.join(DATAROOT, fn)), fn


def test_build_observed_clip_creates_mp4(tmp_path):
    if shutil.which("ffmpeg") is None:
        pytest.skip("ffmpeg not installed")
    from simulation.nuscenes_adapter import load_tables
    from simulation.observed_clip import build_observed_clip
    tables = load_tables(DATAROOT, "v1.0-mini")
    tok = _sample_a_few_steps_in(tables, 3)
    out = str(tmp_path / "observed.mp4")
    build_observed_clip(DATAROOT, tables, tok, out, fps=2)
    assert os.path.exists(out) and os.path.getsize(out) > 0
