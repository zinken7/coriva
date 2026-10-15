"""
Scenario config generator.

Generates N diverse, structurally-valid scenario configs from a small set of
templates with seeded parameter sampling. Deterministic and CARLA-free
(produces configs only; CARLA runs them later).
"""

import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


def _ontology_interventions():
    import yaml
    with open(os.path.join("ctms", "ontology.yaml")) as f:
        onto = yaml.safe_load(f)
    labels = set()
    for group in onto["intervention_type"].values():
        labels.update(group)
    return labels


def test_generates_requested_count():
    from scenarios.generate_configs import generate_configs
    assert len(generate_configs(10, seed=0)) == 10


def test_deterministic_for_same_seed():
    from scenarios.generate_configs import generate_configs
    assert generate_configs(12, seed=7) == generate_configs(12, seed=7)


def test_distinct_indices_and_subdirs():
    from scenarios.generate_configs import generate_configs
    cfgs = generate_configs(15, seed=1)
    assert [c["scenario_index"] for c in cfgs] == list(range(15))
    assert len({c["output_subdir"] for c in cfgs}) == 15
    assert len({c["scenario_id"] for c in cfgs}) == 15


def test_each_scenario_has_exactly_one_factual_branch():
    from scenarios.generate_configs import generate_configs
    for c in generate_configs(20, seed=2):
        factual = [b for b in c["branches"].values() if b["is_factual"]]
        assert len(factual) == 1, c["scenario_id"]


def test_templates_are_stratified():
    from scenarios.generate_configs import generate_configs, TEMPLATES
    types = {c["scenario_type"] for c in generate_configs(len(TEMPLATES), seed=3)}
    assert types == {t["scenario_type"] for t in TEMPLATES}


def test_full_release_has_eight_balanced_variants_and_actor_quotas():
    from collections import Counter
    from scenarios.generate_configs import generate_configs
    configs = generate_configs(400, seed=0)
    assert set(Counter(c["template_id"] for c in configs).values()) == {50}
    assert Counter(c["actor_defs"]["npc_1"]["entity_type"] for c in configs) == {
        "vehicle": 100, "truck": 25, "bus": 25, "motorcycle": 50,
        "pedestrian": 100, "bicycle": 100,
    }
    assert all(len(c["actor_defs"]) == 10 for c in configs)


def test_crossing_roots_keep_critical_actor_near_enough_for_history_visibility():
    """Static roadside targets cannot start 100+ m away and pass all-8 visibility."""
    from scenarios.generate_configs import generate_configs
    crossing = [c for c in generate_configs(400, seed=0)
                if c["actor_defs"]["npc_1"]["entity_type"]
                in {"pedestrian", "bicycle"}]
    assert crossing
    assert max(c["actor_defs"]["npc_1"]["distance_ahead"] for c in crossing) <= 43.0
    assert max(c["actor_defs"]["ego"]["initial_speed"] for c in crossing) <= 7.0


def test_crossing_roots_are_inside_camera_fov_at_t0():
    import math
    from scenarios.generate_configs import (
        CARLA_CAMERA, CROSSING_VISIBILITY_HALF_FOV_DEG, generate_configs)

    crossing = [c for c in generate_configs(400, seed=0)
                if c["actor_defs"]["npc_1"]["entity_type"]
                in {"pedestrian", "bicycle"}]
    for config in crossing:
        critical = config["actor_defs"]["npc_1"]
        longitudinal = config["expected_t0_critical_longitudinal_m"]
        camera_depth = longitudinal - CARLA_CAMERA["offset_m"][0]
        angle = math.degrees(math.atan2(
            abs(critical["lateral_offset"]), camera_depth))
        assert angle <= CROSSING_VISIBILITY_HALF_FOV_DEG + 0.05, (
            config["scenario_id"], angle)
        assert config["target_ttc_sec"] == pytest.approx(
            longitudinal / config["actor_defs"]["ego"]["initial_speed"],
            abs=0.01)


def test_camera_calibration_is_fixed_for_every_release_slot():
    from scenarios.generate_configs import CARLA_CAMERA, generate_configs
    assert all(c["camera"] == CARLA_CAMERA for c in generate_configs(400, seed=0))


def test_large_lead_vehicles_never_overlap_ego_at_spawn():
    from scenarios.generate_configs import generate_configs
    configs = generate_configs(400, seed=0)
    half_lengths = {"truck": (4.5 + 7.0) / 2.0,
                    "bus": (4.5 + 12.0) / 2.0}
    for config in configs:
        critical = config["actor_defs"]["npc_1"]
        entity = critical["entity_type"]
        if entity in half_lengths:
            assert critical["distance_ahead"] > half_lengths[entity]


def test_all_intervention_labels_are_in_ontology():
    from scenarios.generate_configs import generate_configs
    allowed = _ontology_interventions()
    env_allowed = allowed  # environment events live in the same intervention_type block
    for c in generate_configs(20, seed=4):
        assert c["environment_event"]["event_type"] in env_allowed, c["scenario_id"]
        for b in c["branches"].values():
            assert b["intervention_type"] in allowed, (c["scenario_id"], b["intervention_type"])


def test_config_has_builder_required_keys():
    from scenarios.generate_configs import generate_configs
    required = {"scenario_id", "scenario_index", "output_subdir", "scenario_type",
                "actor_defs", "environment_event", "intervention_time_sec",
                "sample_interval_sec", "d_max", "total_duration_sec", "branches"}
    for c in generate_configs(5, seed=5):
        assert required.issubset(c.keys()), required - c.keys()
        assert "ego" in c["actor_defs"]


# ── C#5.2 — multi-actor realism ───────────────────────────────────────
def _ontology():
    import yaml
    return yaml.safe_load(open(os.path.join("ctms", "ontology.yaml")))


def test_configs_are_multi_actor():
    from scenarios.generate_configs import generate_configs
    for c in generate_configs(15, seed=1):
        non_ego = [k for k in c["actor_defs"] if k != "ego"]
        assert len(non_ego) >= 2, (c["scenario_type"], non_ego)


def test_actor_entities_and_roles_valid():
    from scenarios.generate_configs import generate_configs
    onto = _ontology()
    entities = set(onto["entity_type"]["dynamic"]) | set(onto["entity_type"]["static"])
    roles = set(onto["role_type"])
    for c in generate_configs(15, seed=2):
        for key, a in c["actor_defs"].items():
            assert a["entity_type"] in entities or a["entity_type"] == "ego_vehicle", a
            assert a["default_role"] in roles, a


def test_env_behavior_keyed_and_crossing_moves_laterally():
    from scenarios.generate_configs import generate_configs
    for c in generate_configs(15, seed=3):
        eb = c["env_behavior"]
        non_ego = [k for k in c["actor_defs"] if k != "ego"]
        assert set(eb.keys()) == set(non_ego), (eb.keys(), non_ego)
        for key, a in c["actor_defs"].items():
            if key == "ego":
                continue
            beh = eb[key]
            assert "accel" in beh and "lat" in beh
            if a["default_role"] == "crossing_actor":
                assert beh["lat"] != 0.0, (c["scenario_type"], key)  # pedestrians/cyclists cross


def test_scenarios_have_street_furniture_both_sides():
    from scenarios.generate_configs import generate_configs
    for c in generate_configs(10, seed=4):
        ad = c["actor_defs"]
        parked = [a for a in ad.values() if a["entity_type"] == "parked_vehicle"]
        lats = [a["lateral_offset"] for a in parked]
        assert any(l > 0 for l in lats) and any(l < 0 for l in lats), (c["scenario_type"], lats)
        guardrails = [a for a in ad.values() if a["entity_type"] == "guardrail"]
        assert guardrails
        assert all(a["default_role"] == "static_barrier" for a in guardrails)
        assert any(a["entity_type"] in {"traffic_sign", "traffic_cone"}
                   for a in ad.values()), c["scenario_type"]


def test_no_actor_is_spawned_behind_the_ego():
    """The model sees CAM_FRONT only, so an actor behind the ego can never be
    in the observed root and must not exist: every outcome a reference
    records has to be caused by something the model could observe."""
    from scenarios.generate_configs import generate_configs
    for config in generate_configs(400, seed=0):
        behind = [aid for aid, spec in config["actor_defs"].items()
                  if float(spec.get("distance_ahead", 0.0)) < 0.0]
        assert behind == [], (config["scenario_id"], behind)
        assert "npc_3" not in config["actor_defs"]


def test_release_map_plan_is_balanced_and_lawful_per_template():
    """Templates whose cast keeps a vehicle in an adjacent same-direction lane
    for the whole rollout cannot be placed on a single-carriageway map
    (Town10HD). The plan keeps them off those maps, gives the crossing
    templates that share instead, and still seats exactly 100 slots per map
    with every template spread evenly over its allowed maps."""
    from collections import Counter
    from scenarios.generate_configs import (
        MAPS, PLANNED_DATASET_SIZE, SINGLE_CARRIAGEWAY_MAPS, TEMPLATES,
        generate_configs, plan_maps, slot_map, template_allowed_maps)
    configs = generate_configs(PLANNED_DATASET_SIZE, seed=0)
    assert Counter(c["map"] for c in configs) == {name: 100 for name in MAPS}
    for c in configs:
        template = TEMPLATES[c["scenario_index"] % len(TEMPLATES)]
        assert c["map"] in template_allowed_maps(template), c["scenario_id"]
        if template["critical_role"] != "crossing_actor":
            assert c["map"] not in SINGLE_CARRIAGEWAY_MAPS
    for template in TEMPLATES:
        cells = Counter(c["map"] for c in configs if c["template_id"] == template["template_id"])
        dual = [cells[name] for name in MAPS if name not in SINGLE_CARRIAGEWAY_MAPS]
        assert max(dual) - min(dual) <= 1, (template["template_id"], dict(cells))
        if template["critical_role"] == "crossing_actor":
            assert cells["Town10HD"] == 25, dict(cells)
    # map is not tied to severity or to the ego-speed stratum
    for key in (lambda c: c["severity_stratum"], lambda c: (c["scenario_index"] // 8) % 3):
        by_map = {}
        for c in configs:
            by_map.setdefault(c["map"], Counter())[key(c)] += 1
        for name, counts in by_map.items():
            assert max(counts.values()) - min(counts.values()) <= 6, (name, dict(counts))


def test_slot_map_is_stable_across_run_sizes_and_seeds():
    from scenarios.generate_configs import generate_configs, plan_maps, slot_map
    full = plan_maps(400)
    assert [slot_map(i) for i in range(400)] == full
    assert [c["map"] for c in generate_configs(20, seed=3)] == full[:20]
    assert [c["map"] for c in generate_configs(400, seed=7)] == full


def test_lateral_escape_steers_away_from_the_critical_actor():
    """A crossing actor entering from the ego's left must not make the only
    lateral alternative steer into it, and the scorer's action-direction axis
    needs both signs to exist in the release."""
    from collections import Counter
    from scenarios.generate_configs import TEMPLATES, generate_configs
    from simulation.protocol import LATERAL_NEAR_SIDE_M
    for t in TEMPLATES:
        lateral = [a for _n, _f, a in t["branches"] if "swerve" in a]
        if not lateral:
            continue
        expected = ("do_ego_swerve_right"
                    if float(t["critical_lateral"]) < -LATERAL_NEAR_SIDE_M
                    else "do_ego_swerve_left")
        assert lateral == [expected], (t["template_id"], lateral)
    configs = generate_configs(400, seed=0)
    actions = Counter(b["intervention_type"] for c in configs for b in c["branches"].values())
    assert actions["do_ego_swerve_left"] == 200 and actions["do_ego_swerve_right"] == 100
    for c in configs:
        lat = float(c["actor_defs"]["npc_1"]["lateral_offset"])
        for b in c["branches"].values():
            if b["intervention_type"] == "do_ego_swerve_left":
                assert lat >= -LATERAL_NEAR_SIDE_M, c["scenario_id"]
            if b["intervention_type"] == "do_ego_swerve_right":
                assert lat < -LATERAL_NEAR_SIDE_M, c["scenario_id"]
