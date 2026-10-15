"""Frame prep — copy consecutive real CAM_FRONT keyframes a local VLM reads
into <scenario_dir>/frames/ (co-located with ground_truth.json)."""

import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

DATAROOT = "data/nuscenes"
MINI = os.path.join(DATAROOT, "v1.0-mini")
pytestmark = pytest.mark.skipif(not os.path.isdir(MINI), reason="nuScenes v1.0-mini not downloaded")


def _first_sample_with_next(tables):
    for scene in tables["scene"].values():
        tok = scene["first_sample_token"]
        # Advance to the eighth keyframe so the fixed 8-frame history exists.
        for _ in range(7):
            nxt = tables["sample"][tok]["next"]
            if nxt:
                tok = nxt
        if tables["sample"][tok]["next"]:
            return tok
    raise AssertionError("no sample")


def test_write_scenario_frames_copies_selected(tmp_path):
    from simulation.nuscenes_adapter import load_tables
    from simulation.observed_clip import write_scenario_frames
    tables = load_tables(DATAROOT, "v1.0-mini")
    tok = _first_sample_with_next(tables)
    paths = write_scenario_frames(tables, tok, DATAROOT, str(tmp_path), num_frames=8)
    assert len(paths) == 8
    for p in paths:
        assert os.path.exists(p) and p.endswith(".jpg")
    # frames live under <scenario_dir>/frames/
    assert all(os.path.basename(os.path.dirname(p)) == "frames" for p in paths)


def test_write_scenario_frames_rejects_short_history(tmp_path):
    from simulation.nuscenes_adapter import load_tables
    from simulation.observed_clip import write_scenario_frames
    tables = load_tables(DATAROOT, "v1.0-mini")
    first = next(iter(tables["scene"].values()))["first_sample_token"]
    with pytest.raises(ValueError, match="fewer than 8"):
        write_scenario_frames(tables, first, DATAROOT, str(tmp_path), num_frames=8)
