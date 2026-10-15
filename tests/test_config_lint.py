"""
Config-set linter (`scenarios/lint_configs.py`).

400 configs will drive paid GPU hours; a malformed config must be caught at
generation time, not after a CARLA run. Per-config checks: exactly one
factual branch, known intervention types (twin EGO_PROFILES = the canonical
vocabulary), no two actors spawned into the same spot. Set-level checks:
unique scenario ids/subdirs/seeds, template stratification, map diversity.
"""

import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from scenarios.generate_configs import generate_configs  # noqa: E402
from scenarios.lint_configs import lint_config, lint_config_set  # noqa: E402


@pytest.fixture(scope="module")
def config():
    return generate_configs(1, seed=0)[0]


def test_valid_generated_config_passes(config):
    assert lint_config(config) == []


def test_two_factual_branches_flagged(config):
    import copy
    bad = copy.deepcopy(config)
    for b in bad["branches"].values():
        b["is_factual"] = True
    assert any("factual" in e for e in lint_config(bad))


def test_unknown_intervention_flagged(config):
    import copy
    bad = copy.deepcopy(config)
    next(iter(bad["branches"].values()))["intervention_type"] = "do_ego_teleport"
    assert any("intervention" in e for e in lint_config(bad))


def test_spawn_collision_flagged(config):
    import copy
    bad = copy.deepcopy(config)
    ids = [a for a in bad["actor_defs"] if a != "ego"][:2]
    for aid in ids:
        bad["actor_defs"][aid]["distance_ahead"] = 30.0
        bad["actor_defs"][aid]["lateral_offset"] = 0.0
    assert any("spawn" in e for e in lint_config(bad))


def test_large_vehicle_body_overlap_is_flagged_even_when_centres_are_apart():
    import copy
    configs = generate_configs(16, seed=0)
    bus = next(c for c in configs if c["actor_defs"]["npc_1"]["entity_type"] == "bus")
    bad = copy.deepcopy(bus)
    bad["actor_defs"]["npc_1"]["distance_ahead"] = 7.0
    assert any("spawn body overlap" in error for error in lint_config(bad))


def test_full_400_set_lints_clean_and_stratified():
    configs = generate_configs(400, seed=0)
    report = lint_config_set(configs)
    assert report["errors"] == []
    assert report["n"] == 400
    # stratification: every template appears, no template dominates
    counts = report["by_scenario_type"]
    assert len(counts) >= 4
    assert max(counts.values()) - min(counts.values()) <= 1
    assert len(report["by_map"]) >= 2
    assert set(report["by_template_id"].values()) == {50}
    assert min(report["by_weather"].values()) == 66
    assert max(report["by_weather"].values()) == 67


def test_protocol_camera_drift_is_rejected(config):
    import copy
    bad = copy.deepcopy(config)
    bad["camera"]["fov_deg"] = 90.0
    assert any("camera calibration" in error for error in lint_config(bad))


def test_crossing_actor_outside_t0_camera_margin_is_rejected():
    import copy
    crossing = next(c for c in generate_configs(8, seed=0)
                    if c["actor_defs"]["npc_1"]["entity_type"] == "pedestrian")
    bad = copy.deepcopy(crossing)
    bad["expected_t0_critical_longitudinal_m"] = 3.0
    assert any("T0 camera angle" in error for error in lint_config(bad))


def test_duplicate_ids_flagged_at_set_level():
    configs = generate_configs(3, seed=0)
    configs[2]["scenario_id"] = configs[0]["scenario_id"]
    report = lint_config_set(configs)
    assert any("duplicate" in e for e in report["errors"])
