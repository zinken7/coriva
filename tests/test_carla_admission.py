"""Protocol-1.0 CARLA admission and clean-attempt regression tests."""

import copy

from simulation.carla_admission import evaluate_carla_admission


def _state(x, y=0.0, speed=10.0, bbox=None):
    state = {"x": x, "y": y, "vx": speed, "vy": 0.0,
             "speed_kmh": speed * 3.6, "heading_deg": 0.0}
    if bbox is not None:
        state["bbox_2d"] = bbox
    return state


def _fixture():
    config = {
        "scenario_id": "lead_car_brake_0000", "template_id": "lead_car_brake",
        "critical_actor_id": "npc_1", "intervention_time_sec": 3.5,
        "actor_defs": {
            "ego": {"canonical_id": "ego", "entity_type": "ego_vehicle"},
            "npc_1": {"canonical_id": "npc_1", "entity_type": "vehicle"},
        },
        "branches": {
            "brake": {"is_factual": True},
            "no_action": {"is_factual": False},
            "swerve": {"is_factual": False},
        },
    }
    factual = []
    for index in range(36):
        t = round(index * 0.1, 1)
        ego_x = 10.0 * t
        factual.append({"time_sec": t, "ego": _state(ego_x),
                        "npc_1": _state(ego_x + 12.0, speed=9.0,
                                        bbox=[400, 250, 430, 300])})

    branches = {}
    for name in config["branches"]:
        steps = []
        for depth in range(1, 7):
            t = 3.5 + depth * 0.5
            if name == "brake":
                ego = _state(35.0 + depth * 3.0, speed=6.0)
            elif name == "no_action":
                ego = _state(35.0 + depth * 5.0, speed=10.0)
            else:
                ego = _state(35.0 + depth * 5.0, y=depth * 1.0, speed=10.0)
            steps.append({"time_sec": t, "ego": ego,
                          "npc_1": _state(47.0 + depth * 2.0, speed=4.0)})
        branches[name] = factual + steps if name == "brake" else steps

    treated = []
    control = []
    for depth in range(1, 7):
        t = 3.5 + depth * 0.5
        ego = _state(35.0 + depth * 5.0, speed=10.0)
        control.append({"time_sec": t, "ego": ego,
                        "npc_1": _state(47.0 + depth * 4.5, speed=9.0)})
        treated.append({"time_sec": t, "ego": ego,
                        "npc_1": _state(47.0 + depth * 2.0, speed=4.0)})
    raw = {"branches": branches, "_l4_steps": treated,
           "_l4_control_steps": control}
    root_actors = [{"id": actor_id} for actor_id in
                   ("ego", "npc_1", "npc_2", "park_l1", "wall_l")]
    gt = {"observed_state": {"actors": root_actors}}
    return config, raw, gt


def test_admission_passes_visible_effective_interaction():
    config, raw, gt = _fixture()
    ok, metrics = evaluate_carla_admission(raw, config, gt)
    assert ok, metrics
    assert metrics["max_l4_target_position_effect_m"] >= 1.0
    assert metrics["min_ego_critical_edge_gap_m"] <= 5.0


def test_admission_rejects_no_effect_intervention():
    config, raw, gt = _fixture()
    raw["_l4_steps"] = copy.deepcopy(raw["_l4_control_steps"])
    ok, metrics = evaluate_carla_admission(raw, config, gt)
    assert not ok
    assert "l4_intervention_has_no_material_target_effect" in metrics["reasons"]


def test_admission_rejects_missing_projected_bbox():
    config, raw, gt = _fixture()
    for step in raw["branches"]["brake"]:
        step["npc_1"].pop("bbox_2d", None)
    ok, metrics = evaluate_carla_admission(raw, config, gt)
    assert not ok
    assert "critical_actor_projected_bbox_not_visible_all_8" in metrics["reasons"]


def _far_target_with_collision(collision_time, partners):
    """Critical actor never within 5 m; one raw collision flagged in the L4 rollout."""
    config, raw, gt = _fixture()
    config["actor_defs"]["npc_2"] = {"canonical_id": "npc_2", "entity_type": "vehicle"}
    for step in raw["_l4_steps"]:
        step["npc_1"] = _state(step["ego"]["x"] + 60.0, speed=10.0)
        step["npc_2"] = _state(step["ego"]["x"] - 80.0, speed=10.0)
    step = min(raw["_l4_steps"], key=lambda s: abs(s["time_sec"] - collision_time))
    step["collision"] = True
    step["collision_time"] = collision_time
    step["collision_partners"] = partners
    return config, raw, gt


def test_admission_counts_collision_with_critical_actor_inside_horizon():
    config, raw, gt = _far_target_with_collision(5.5, ["npc_1"])
    ok, metrics = evaluate_carla_admission(raw, config, gt)
    assert ok, metrics
    assert metrics["collision"] is True
    assert metrics["collision_time_sec"] == 5.5
    assert metrics["collision_partners"] == ["npc_1"]


def test_admission_ignores_collision_with_another_actor():
    config, raw, gt = _far_target_with_collision(5.5, ["npc_2"])
    ok, metrics = evaluate_carla_admission(raw, config, gt)
    assert not ok
    assert metrics["collision"] is False
    assert metrics["collision_partners"] == ["npc_2"]
    assert "critical_actor_does_not_interact_with_ego_within_3s" in metrics["reasons"]


def test_admission_ignores_collision_after_horizon():
    config, raw, gt = _far_target_with_collision(6.5, ["npc_1"])
    raw["_l4_steps"][-1]["collision_time"] = 6.8
    ok, metrics = evaluate_carla_admission(raw, config, gt)
    assert not ok
    assert metrics["collision"] is False and metrics["collision_time_sec"] is None
    assert "critical_actor_does_not_interact_with_ego_within_3s" in metrics["reasons"]
