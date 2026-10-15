"""The consequence self-consistency measure (scripts/explore/consequence.py):
exact first contact over every actor, contract-consistent node labelling,
complete-evidence verdicts, separate extrapolation, stop-aware accelerations."""
import importlib.util
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
_spec = importlib.util.spec_from_file_location(
    "consequence", os.path.join(os.path.dirname(__file__), "..", "scripts", "explore", "consequence.py"))
cq = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(cq)


def _car(aid, x, vx=0.0, role="lead_vehicle", y=0.0):
    return {"id": aid, "entity_type": "vehicle", "role_type": role, "x": x, "y": y,
            "vx": vx, "vy": 0.0, "speed": abs(vx), "length": 4.5, "width": 2.0}


def _root(others):
    return {"ego": _car("ego", 0.0, 10.0, "ego_agent"), **{o["id"]: o for o in others}}


def _chain(others_by_t, ego_v=10.0, n=6, ego_x=None):
    return [{"depth": d, "timestamp": 0.5 * d, "event_type": "maintain_course",
             "actors_update": [_car("ego", (ego_x(0.5 * d) if ego_x else ego_v * 0.5 * d), ego_v, "ego_agent")] + others_by_t(0.5 * d)}
            for d in range(1, n + 1)]


def _scan(chain, root, extrapolate=False):
    return cq.contact_scan(cq.states_from_chain(chain, root), root, "ego", extrapolate)


def test_contact_inside_an_interval_labels_the_sample_before_it():
    root = _root([_car("lead", 14.0)])                      # ego front meets the lead's rear at 0.95 s
    res = _scan(_chain(lambda t: [_car("lead", 14.0)]), root)
    assert (res["node"], res["partners"], res["evidence"]) == (0, ["lead"], "complete")


def test_contact_exactly_on_a_sample_labels_that_sample():
    root = _root([_car("lead", 14.5)])                      # touching at exactly 1.0 s
    res = _scan(_chain(lambda t: [_car("lead", 14.5)]), root)
    assert (res["node"], res["partners"]) == (1, ["lead"])


def test_a_contact_just_before_a_sample_labels_the_previous_sample():
    root = _root([_car("lead", 14.45)])                     # contact at exactly 0.995 s
    res = _scan(_chain(lambda t: [_car("lead", 14.45)]), root)
    assert (res["node"], res["partners"]) == (0, ["lead"])


def test_a_brief_pass_through_between_samples_is_still_found():
    # a crossing pedestrian (0.5 x 0.5 m) whose path crosses the ego's between two samples:
    # the overlap lasts far less than 0.01 s yet the exact solver finds it
    ped = lambda t: [dict(_car("ped", 1.0, 0.0, "crossing_actor", y=-30.0 + 30.0 * t), length=0.5, width=0.5, vy=30.0, speed=30.0)]
    root = _root([dict(_car("ped", 1.0, 0.0, "crossing_actor", y=-30.0), length=0.5, width=0.5, vy=30.0, speed=30.0)])
    chain = _chain(ped, ego_v=0.0, ego_x=lambda t: 0.0)
    res = _scan(chain, root)
    # the bodies overlap only for t in [0.958, 1.042]: first contact at 0.958 s, inside interval 2, labels node 0
    assert res["node"] == 0 and res["partners"] == ["ped"]


def test_the_earliest_contact_wins_whatever_the_actor_order():
    near, far = _car("near", 14.0), _car("far", 24.0)
    for order in ((near, far), (far, near)):
        root = _root(order)
        res = _scan(_chain(lambda t, o=order: [dict(x) for x in o]), root)
        assert (res["node"], res["partners"]) == (0, ["near"]), order


def test_a_contact_before_the_first_sample_has_no_valid_node():
    root = _root([_car("lead", 6.0)])                       # contact at 0.15 s
    res = _scan(_chain(lambda t: [_car("lead", 6.0)]), root)
    assert res["first_interval"] is True and res["node"] == 0


def test_extrapolation_distinguishes_constant_velocity_from_the_braking_the_model_reported():
    # ego brakes hard: 10 -> 5 -> 0 m/s over two nodes; a lead sits 2 m beyond the stop
    nodes = [{"depth": 1, "timestamp": 0.5, "event_type": "ego_brake", "actors_update": [
                  dict(_car("ego", 3.75, 5.0, "ego_agent")), _car("lead", 12.0)]},
             {"depth": 2, "timestamp": 1.0, "event_type": "collision", "terminal": True, "actors_update": [
                  dict(_car("ego", 5.0, 0.5, "ego_agent")), _car("lead", 12.0)]}]
    root = _root([_car("lead", 12.0)])
    res = _scan(nodes, root, extrapolate=True)
    assert res["node"] is None
    assert res["extrapolated"]["velocity"] is False and res["extrapolated"]["accel"] is False
    fast = [dict(nodes[0]), dict(nodes[1])]
    fast[1] = dict(nodes[1], actors_update=[dict(_car("ego", 5.0, 9.0, "ego_agent")), _car("lead", 12.0)])
    res = _scan(fast, root, extrapolate=True)
    assert res["extrapolated"]["velocity"] is True            # 9 m/s for 0.5 s reaches the lead's rear at 7.5 m


def test_partial_and_empty_chains_are_undefined_not_no_collision(tmp_path):
    inputs = tmp_path / "inputs"; cell = tmp_path / "cell"
    (inputs / "scenario_1").mkdir(parents=True); (cell / "scenario_1").mkdir(parents=True)
    root = _root([_car("lead", 14.0)])
    (inputs / "scenario_1" / "observed_state.json").write_text(json.dumps({"actors": list(root.values())}))
    partial = [{"depth": d, "timestamp": 0.5 * d, "event_type": "maintain_course",
                "actors_update": [_car("ego", 5.0 * d, 10.0, "ego_agent")] + ([_car("lead", 14.0)] if d == 1 else [])}
               for d in range(1, 7)]
    labelled = [dict(n) for n in partial]; labelled[-1] = dict(partial[-1], event_type="collision", terminal=True)
    broken = [dict(partial[0], actors_update=7), None, "x"]
    (cell / "scenario_1" / "l3_pre.json").write_text(json.dumps({"predicted_branches": [
        {"branch_id": "B1", "intervention": {"intervention_type": "do_ego_no_action"}, "chain": partial},
        {"branch_id": "B2", "intervention": {"intervention_type": "do_ego_comfortable_brake"}, "chain": labelled},
        {"branch_id": "B3", "intervention": {"intervention_type": "do_ego_swerve_left"}, "chain": broken},
        {"branch_id": "B4", "intervention": {"intervention_type": "do_ego_panic_brake"}, "chain": []}]}))
    (cell / "scenario_1" / "l4_pre.json").write_text(json.dumps({"predicted_counterfactuals": []}))
    rows = cq.analyse_cell(cell, inputs, "m", "video", "carla", "A")
    assert [r["category"] for r in rows] == ["insufficient_data", "insufficient_data", "insufficient_data", "empty_chain"]
    assert all(r["impossible"] is None for r in rows)


def test_stop_aware_acceleration_and_example_copy():
    root = {"ego": _car("ego", 0.0, 1.0, "ego_agent")}
    # 1 m/s braking at -2.4 stops after 0.417 s, x = 0.2083 m: the first-step speed change would read -2.0
    states = [{"ego": dict(_car("ego", 1.0 * 1.0 / 2.4 / 2, 0.0, "ego_agent"), speed=0.0)}]
    assert abs(cq.accel_estimate(states, root, "ego") + 2.4) < 1e-6
    moving = [{"ego": dict(_car("ego", 3.0, 4.0, "ego_agent"), speed=4.0)}]
    root2 = {"ego": _car("ego", 0.0, 5.2, "ego_agent")}
    assert abs(cq.accel_estimate(moving, root2, "ego") + 2.4) < 1e-6


def test_own_effect_direction_is_magnitude_free():
    root = {"ego": _car("ego", 0.0, 10.0, "ego_agent"), "npc_1": _car("npc_1", 12.0, 10.0)}
    ctrl = [{"npc_1": dict(_car("npc_1", 17.0, 10.0), speed=10.0)}]
    slow = [{"npc_1": dict(_car("npc_1", 16.0, 7.0), speed=7.0)}]
    tiny = [{"npc_1": dict(_car("npc_1", 16.9, 9.6), speed=9.6)}]
    fast = [{"npc_1": dict(_car("npc_1", 18.0, 12.0), speed=12.0)}]
    assert cq.own_effect_direction(ctrl, slow, "npc_1", "npc_sudden_brake", root) == "sign_ok"
    assert cq.own_effect_direction(ctrl, tiny, "npc_1", "npc_sudden_brake", root) == "below_threshold"
    assert cq.own_effect_direction(ctrl, fast, "npc_1", "npc_sudden_brake", root) == "sign_wrong"


def _cell(tmp_path, root, branches, counterfactuals=(), split="carla"):
    inputs = tmp_path / "inputs"; cell = tmp_path / "cell"
    (inputs / "scenario_1").mkdir(parents=True, exist_ok=True); (cell / "scenario_1").mkdir(parents=True, exist_ok=True)
    (inputs / "scenario_1" / "observed_state.json").write_text(json.dumps({"actors": list(root.values())}))
    (cell / "scenario_1" / "l3_pre.json").write_text(json.dumps({"predicted_branches": [
        {"branch_id": f"B{i}", "intervention": {"intervention_type": a}, "chain": ch} for i, (a, ch) in enumerate(branches)]}))
    (cell / "scenario_1" / "l4_pre.json").write_text(json.dumps({"predicted_counterfactuals": list(counterfactuals)}))
    return [r["category"] for r in cq.analyse_cell(cell, inputs, "m", "video", split, "A")]


def test_a_collision_label_with_no_contact_in_a_complete_six_node_chain_is_a_contradiction(tmp_path):
    root = _root([_car("lead", 60.0, 10.0)])                  # same speed, 60 m ahead: never meets
    far = _chain(lambda t: [_car("lead", 60.0 + 10.0 * t, 10.0)])
    labelled = [dict(n) for n in far]; labelled[-1] = dict(far[-1], event_type="collision", terminal=True)
    assert _cell(tmp_path, root, [("do_ego_no_action", labelled)]) == ["label_without_contact"]
    # the verdict on a complete chain does not depend on whether the after-horizon extrapolation can be formed
    no_velocity = [dict(n) for n in labelled]
    no_velocity[-1] = dict(labelled[-1], actors_update=[dict(a, vx=None, vy=None) for a in labelled[-1]["actors_update"]])
    assert _cell(tmp_path, root, [("do_ego_no_action", no_velocity)]) == ["label_without_contact"]


def test_a_last_node_label_on_a_short_chain_is_unsupported_not_contradicted(tmp_path):
    root = _root([_car("lead", 60.0, 10.0)])
    far = _chain(lambda t: [_car("lead", 60.0 + 10.0 * t, 10.0)])
    short = [dict(n) for n in far[:3]]; short[-1] = dict(far[2], event_type="collision", terminal=True)
    assert _cell(tmp_path, root, [("do_ego_no_action", short)]) == ["label_unsupported"]
    # six nodes, the lead 1 m beyond the ego's horizon reach: contact only after 3.0 s under extrapolation
    root2 = _root([_car("lead", 34.0 + 1.0, 0.0)])            # ego front at 3.0 s = 30 + 2.25; lead rear = 35 - 2.25
    six = _chain(lambda t: [_car("lead", 35.0, 0.0)]); six[-1] = dict(six[-1], event_type="collision", terminal=True)
    assert _cell(tmp_path, root2, [("do_ego_no_action", six)]) == ["label_beyond_horizon"]


def test_a_one_node_chain_extrapolates_its_acceleration_from_the_root():
    # root 10 m/s, node 1 at 3.75 m and 5 m/s: constant deceleration (-10 m/s^2, read against the root)
    # stops the ego at 5.0 m (front 7.25 m) short of a lead whose rear is at 7.75 m; constant velocity
    # (front 8.5 m) reaches it
    root = _root([_car("lead", 10.0, 0.0)])
    nodes = [{"depth": 1, "timestamp": 0.5, "event_type": "collision", "terminal": True,
              "actors_update": [dict(_car("ego", 3.75, 5.0, "ego_agent")), _car("lead", 10.0, 0.0)]}]
    res = _scan(nodes, root, extrapolate=True)
    assert res["extrapolated"] == {"velocity": True, "accel": False}
    missing = [dict(nodes[0], actors_update=[dict(_car("ego", 3.75, 5.0, "ego_agent"), vx=None), _car("lead", 10.0, 0.0)])]
    res = _scan(missing, root, extrapolate=True)
    assert res["extrapolated"] == {"velocity": None, "accel": None}


def test_the_stop_label_threshold_is_not_a_zero_for_the_acceleration_estimate():
    root = {"ego": _car("ego", 0.0, 1.0, "ego_agent")}
    slow = [{"ego": dict(_car("ego", 0.35, 0.4, "ego_agent"), speed=0.4)}]   # 1 -> 0.4 m/s over 0.5 s, 0.35 m
    assert abs(cq.accel_estimate(slow, root, "ego") + 1.2) < 1e-9
    stopped = [{"ego": dict(_car("ego", 0.35, 0.0, "ego_agent"), speed=0.0)}]
    assert abs(cq.accel_estimate(stopped, root, "ego") + 1.0 / 0.7) < 1e-9


def test_a_lateral_effect_is_read_along_the_control_targets_heading_on_carla():
    root = {"ego": _car("ego", 0.0, 10.0, "ego_agent"), "npc_1": _car("npc_1", 12.0, 10.0, y=3.5)}
    heading = dict(_car("npc_1", 17.0, 7.07, y=8.5), vy=7.07, speed=10.0)          # control target travels at 45 deg
    ctrl = [{"npc_1": heading}]
    trt = [{"npc_1": dict(heading, x=17.0 + 1.5, y=8.5 - 0.5)}]                 # 1.41 m to the target's left, i.e. towards the ego lane
    assert cq.own_effect_direction(ctrl, trt, "npc_1", "npc_cut_in_left", root, "carla") == "sign_ok"
    assert cq.own_effect_direction(ctrl, trt, "npc_1", "npc_cut_in_left", root, "nuscenes") == "below_threshold"
    away = [{"npc_1": dict(heading, x=17.0 - 1.5, y=8.5 + 0.5)}]
    assert cq.own_effect_direction(ctrl, away, "npc_1", "npc_cut_in_left", root, "carla") == "sign_wrong"
