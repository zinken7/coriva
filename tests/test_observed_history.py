"""Observed-history schema and exact-T0 protocol invariants."""

import copy
import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from simulation.observed_history import validate_observed_history
from simulation.protocol import (COORDINATE_CONVENTION, COORDINATE_FRAME,
                                 HISTORY_TIMESTAMPS, INPUT_CONDITIONS,
                                 PROTOCOL_VERSION, protocol_manifest,
                                 validate_protocol_manifest)


def _root():
    return {"depth": 0, "timestamp": 0.0, "event_type": "observed_state",
            "actors": [
                {"id": "ego", "entity_type": "ego_vehicle", "role_type": "ego_agent",
                 "x": 0.0, "y": 0.0, "vx": 8.0, "vy": 0.0, "speed": 8.0,
                 "length": 4.5, "width": 2.0},
                {"id": "npc_1", "entity_type": "pedestrian", "role_type": "crossing_actor",
                 "x": 12.0, "y": 3.0, "vx": 0.0, "vy": -1.0, "speed": 1.0,
                 "length": 0.6, "width": 0.6},
            ]}


def _history(root):
    records = []
    for timestamp in HISTORY_TIMESTAMPS:
        actors = []
        for actor in root["actors"]:
            item = copy.deepcopy(actor)
            item["visibility"] = actor["id"] != "ego"
            actors.append(item)
        records.append({"timestamp_sec": timestamp, "actors": actors})
    return {"schema_version": 1, "coordinate_frame": "t0_ego", "records": records}


def test_final_protocol_has_three_declared_conditions():
    assert INPUT_CONDITIONS == ("video", "history_json", "video_history")


def test_hybrid_protocol_declares_primary_and_secondary_measurement_roles():
    manifest = protocol_manifest(split="v1.0-trainval")
    assert PROTOCOL_VERSION == "coriva-protocol-1.0"
    assert manifest["primary_measurement"] == "ground_truth_free_l3_causal_coherence"
    assert manifest["reference_evaluation_role"] == "secondary_declared_reference_agreement"
    assert manifest["reference_semantics"] == "observed_factual_and_canonical_twin_agreement"
    assert manifest["coordinate_frame"] == COORDINATE_FRAME == "t0_ego"
    assert manifest["coordinate_convention"] == COORDINATE_CONVENTION
    assert manifest["coordinate_convention"]["x_axis"] == "positive_forward_at_t0"
    assert manifest["coordinate_convention"]["y_axis"] == "positive_right_at_t0"

    carla = protocol_manifest(split="carla")
    assert carla["reference_semantics"] == "exact_deterministic_simulator_agreement"


@pytest.mark.parametrize("split", ["carla", "v1.0-trainval"])
def test_protocol_10_manifest_validates_for_both_released_splits(split):
    manifest = protocol_manifest(split=split)
    assert validate_protocol_manifest(manifest) is manifest


def test_history_has_exact_eight_timestamps_and_t0_copy():
    root = _root()
    history = validate_observed_history(_history(root), root)
    assert [r["timestamp_sec"] for r in history["records"]] == list(HISTORY_TIMESTAMPS)


def test_history_rejects_rounded_or_recomputed_t0():
    root = _root()
    history = _history(root)
    history["records"][-1]["actors"][1]["x"] += 0.001
    with pytest.raises(ValueError, match="exactly equal"):
        validate_observed_history(history, root)


def test_history_rejects_actor_universe_change():
    root = _root()
    history = _history(root)
    history["records"][3]["actors"].pop()
    with pytest.raises(ValueError, match="universe"):
        validate_observed_history(history, root)


def test_manifest_rejects_ambiguous_or_reversed_lateral_axis():
    manifest = protocol_manifest(split="v1.0-trainval")
    manifest["coordinate_convention"]["y_axis"] = "positive_left_at_t0"
    with pytest.raises(ValueError, match="coordinate_convention"):
        validate_protocol_manifest(manifest)
