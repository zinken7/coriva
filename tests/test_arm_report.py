"""scripts/arm_report.py: the arm A / arm B measures on a
synthetic two-arm layout with the published control as the recipe."""
import csv
import importlib.util
import json
import os
import shutil
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
_spec = importlib.util.spec_from_file_location(
    "arm_report", os.path.join(os.path.dirname(__file__), "..", "scripts", "arm_report.py"))
ar = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(ar)


def _car(aid, x, vx=0.0, role="lead_vehicle", y=0.0, vy=0.0):
    return {"id": aid, "entity_type": "vehicle", "role_type": role, "x": x, "y": y,
            "vx": vx, "vy": vy, "speed": (vx * vx + vy * vy) ** 0.5, "length": 4.5, "width": 2.0}


def _chain(ego_xy, n=6, lead=lambda t: _car("lead", 60.0 + 10.0 * t, 10.0), event="maintain_course", last=None):
    nodes = []
    for d in range(1, n + 1):
        t = 0.5 * d; x, y = ego_xy(t)
        nodes.append({"depth": d, "timestamp": t, "event_type": event,
                      "actors_update": [dict(_car("ego", x, 10.0, "ego_agent"), y=y), lead(t)]})
    if last:
        nodes[-1].update(event_type=last, terminal=True)
    return nodes


ROOT = {"actors": [_car("ego", 0.0, 10.0, "ego_agent"), _car("lead", 60.0, 10.0)]}
STRAIGHT = lambda t: (10.0 * t, 0.0)
SWERVE_LEFT = lambda t: (10.0 * t, -min(2.5 * t, 3.5))         # -y = left, the twin's rule
SWERVE_WEAK = lambda t: (10.0 * t, -0.4)
GT = {"branches": [{"branch_id": "F", "intervention": {"intervention_type": "do_ego_no_action"}, "chain": _chain(STRAIGHT, n=2, last="collision")},
                   {"branch_id": "S", "intervention": {"intervention_type": "do_ego_swerve_left"}, "chain": _chain(SWERVE_LEFT)},
                   {"branch_id": "B", "intervention": {"intervention_type": "do_ego_comfortable_brake"}, "chain": _chain(STRAIGHT, n=3, last="safe_stop")}],
      "l4_queries": [{"intervention_type": "npc_sudden_brake", "counterfactual_chain": {
          "intervention": {"intervention_type": "npc_sudden_brake", "target_actor_id": "lead", "trigger_depth": 1},
          "chain": _chain(STRAIGHT, n=2, last="collision")}}]}


def _write_cell(root, split, cell_name, scenarios, variant=None, with_collected=True):
    cell = root / "cells" / split / cell_name
    for scenario, (l3, l4, metrics) in scenarios.items():
        d = cell / scenario; d.mkdir(parents=True, exist_ok=True)
        (d / "l3_pre.json").write_text(json.dumps(l3)); (d / "l4_pre.json").write_text(json.dumps(l4))
        (d / "run_manifest.json").write_text(json.dumps({"prompt_variant": variant} if variant else {}))
        inp = root / "inputs" / split / scenario; inp.mkdir(parents=True, exist_ok=True)
        (inp / "observed_state.json").write_text(json.dumps(ROOT)); (inp / "ground_truth.json").write_text(json.dumps(GT))
    if with_collected:
        (cell / "collected.json").write_text(json.dumps({"metrics": {
            "monotonicity": {s: m.get("monotonicity") for s, (_, _, m) in scenarios.items()},
            "CTMS": {s: m.get("CTMS") for s, (_, _, m) in scenarios.items()}}}))
    return cell


def _l3(no_action, swerve, brake=None):
    branches = [{"branch_id": "N", "intervention": {"intervention_type": "do_ego_no_action"}, "chain": no_action},
                {"branch_id": "S", "intervention": {"intervention_type": "do_ego_swerve_left"}, "chain": swerve}]
    if brake is not None:
        branches.append({"branch_id": "B", "intervention": {"intervention_type": "do_ego_comfortable_brake"}, "chain": brake})
    return {"predicted_branches": branches}


def _l4(treated_last="collision"):
    return {"predicted_counterfactuals": [{"intervention": {"intervention_type": "npc_sudden_brake", "target_actor_id": "lead"},
                                          "control_chain": _chain(STRAIGHT), "treated_chain": _chain(STRAIGHT, n=2, last=treated_last)}]}


@pytest.fixture
def layout(tmp_path):
    control = _l3(_chain(STRAIGHT, n=2, last="collision"), _chain(SWERVE_LEFT), _chain(STRAIGHT, n=3, last="safe_stop"))
    baselines = tmp_path / "baselines"
    _write_cell(baselines, "carla", "action_profile_published", {"scenario_1": (control, _l4(), {"monotonicity": 1.0, "CTMS": 1.0}),
                                                                 "scenario_2": (control, _l4(), {"monotonicity": 1.0, "CTMS": 1.0})})
    # arm A copies the recipe exactly; arm B drifts 1 m at every node, swerves weakly and misses the collision
    copies = _l3(_chain(STRAIGHT, n=2, last="collision"), _chain(SWERVE_LEFT), _chain(STRAIGHT, n=3, last="safe_stop"))
    drift = _l3(_chain(lambda t: (10.0 * t + 1.0, 0.0)), _chain(SWERVE_WEAK), _chain(lambda t: (10.0 * t + 1.0, 0.0), n=3, last="safe_stop"))
    a = tmp_path / "arm_a"; b = tmp_path / "arm_b"
    _write_cell(a / "m", "carla", "m_video", {"scenario_1": (copies, _l4(), {"monotonicity": 1.0, "CTMS": 0.9}),
                                               "scenario_2": (copies, _l4(), {"monotonicity": 1.0, "CTMS": 0.8})}, "full")
    _write_cell(b / "m", "carla", "m_video", {"scenario_1": (drift, _l4("maintain_course"), {"monotonicity": 1.0, "CTMS": 0.4}),
                                               "scenario_2": (drift, _l4("maintain_course"), {"monotonicity": 0.5, "CTMS": 0.3})}, "recipe_withheld")
    return a, b, baselines


def test_recipe_reproduction_counts_same_depth_slots_of_the_shorter_chain():
    control = _chain(STRAIGHT, n=4)
    assert ar.recipe_reproduction(_chain(STRAIGHT, n=6), control) == 1.0
    assert ar.recipe_reproduction(_chain(lambda t: (10.0 * t + (0.0 if t <= 1.0 else 1.0), 0.0), n=6), control) == 0.5
    assert ar.recipe_reproduction([], control) is None and ar.recipe_reproduction(None, control) is None
    # a slot the model leaves without the ego, or malformed, counts against it; the denominator is the shorter chain
    two = _chain(STRAIGHT, n=2)
    assert ar.recipe_reproduction([two[0], dict(two[1], actors_update=[])], control) == 0.5
    assert ar.recipe_reproduction([two[0], dict(two[1], actors_update=7)], control) == 0.5
    assert ar.recipe_reproduction([two[0], None], control) == 0.5
    echoed = [{"depth": 0, "event_type": "observed_state", "actors_update": [_car("ego", 0.0, 10.0, "ego_agent")]}] + _chain(STRAIGHT, n=2)
    assert ar.recipe_reproduction(ar.future_nodes(echoed), control) == 1.0


def test_swerve_offsets_are_signed_towards_the_branches_own_side():
    root = json.loads(json.dumps(ROOT))
    offsets = dict(ar.swerve_offsets({"do_ego_no_action": _chain(STRAIGHT), "do_ego_swerve_left": _chain(SWERVE_LEFT),
                                      "do_ego_swerve_right": _chain(lambda t: (10.0 * t, 1.5))}, root))
    assert offsets["do_ego_swerve_left"] == pytest.approx(3.5) and offsets["do_ego_swerve_right"] == pytest.approx(1.5)
    assert ar.swerve_offsets({"do_ego_swerve_left": _chain(SWERVE_LEFT)}, root) == []        # no neutral path


def test_collision_events_count_missing_branches_as_misses():
    l3 = _l3(_chain(STRAIGHT, n=2, last="collision"), _chain(SWERVE_LEFT))          # the brake branch is missing
    out, depth = ar.collision_events(GT, l3, _l4("maintain_course"))
    assert out["collision/L3_recall"] == (1, 1) and out["collision/L3_precision"] == (1, 1)
    assert out["collision/L3_recall/do_ego_no_action"] == (1, 1) and out["collision/L3_precision/do_ego_no_action"] == (1, 1)
    assert out["collision/safe_stop_recall"] == (0, 1)
    assert out["collision/L4_treated_recall"] == (0, 1) and out["collision/L4_treated_recall/npc_sudden_brake"] == (0, 1)
    assert "collision/L4_treated_precision" not in out
    assert depth == {"collision/L3_depth_error_signed": [0], "collision/L3_depth_error_abs": [0]}
    # a collision labelled one step late on the ego branch and on time on the treated chain
    late = _l3(_chain(STRAIGHT, n=3, last="collision"), _chain(SWERVE_LEFT, n=2, last="collision"))
    out, depth = ar.collision_events(GT, late, _l4())
    assert out["collision/L3_precision"] == (1, 2) and out["collision/L3_precision/do_ego_swerve_left"] == (0, 1)
    assert out["collision/L4_treated_precision/npc_sudden_brake"] == (1, 1)
    assert depth["collision/L3_depth_error_signed"] == [1] and depth["collision/L4_treated_depth_error_abs"] == [0]


def test_ratio_and_paired_statistics_resample_the_covered_scenarios():
    a = {"s1": (1, 2), "s2": (0, 2), "s3": (2, 2)}
    st = ar.ratio_ci(a, ["s1", "s2", "s3", "s4"], 200, 0)                              # s4 covered, no event
    assert st["estimate"] == pytest.approx(0.5) and st["n"] == 4 and st["n_events"] == 3 and st["lo"] <= 0.5 <= st["hi"]
    assert ar.ratio_ci({"s1": (0, 0)}, ["s1"], 200, 0)["estimate"] is None
    b = {"s1": (0, 2), "s2": (0, 2), "s3": (0, 2), "s4": (5, 5)}
    d = ar.paired_ratio_ci(a, b, ["s1", "s2", "s3"], 200, 0)
    assert d["estimate"] == pytest.approx(0.5) and d["n"] == 3                        # s4 is not covered by both
    # a scenario with events in one arm only stays in the pair as (0, 0) in the other:
    # A = 1/2 (s1 hit, s2 miss), B = 1/1 (s2 has no reference event in B's answer set) -> A - B = -0.5, not 0
    a2 = {"s1": (1, 1), "s2": (0, 1)}; b2 = {"s1": (1, 1)}
    d = ar.paired_ratio_ci(a2, b2, ["s1", "s2"], 200, 0)
    assert d["estimate"] == pytest.approx(-0.5) and d["n"] == 2
    assert ar.paired_ratio_ci(a2, {}, ["s1", "s2"], 200, 0)["estimate"] is None
    m = ar.paired_mean_ci({"s1": 1.0, "s2": 0.5}, {"s1": 0.5, "s2": 0.5, "s9": 0.0}, 200, 0)
    assert m["estimate"] == pytest.approx(0.25) and m["n"] == 2


def test_the_report_pairs_the_arms_and_reads_the_control_as_the_recipe(layout, tmp_path):
    a, b, baselines = layout
    out = tmp_path / "report"
    result = ar.build({"A": a, "B": b}, baselines, out, boot=200, seed=0)
    rows = {(r["arm"], r["measure"]): r for r in result["rows"]}
    assert rows[("A", "recipe/branch_share")]["estimate"] == pytest.approx(1.0)
    assert rows[("B", "recipe/branch_share")]["estimate"] == pytest.approx(0.0)
    assert rows[("control", "recipe/branch_share")]["estimate"] == pytest.approx(1.0)
    assert rows[("A", "collision/L3_recall")]["estimate"] == 1.0 and rows[("B", "collision/L3_recall")]["estimate"] == 0.0
    assert rows[("A", "collision/L4_treated_recall")]["estimate"] == 1.0 and rows[("B", "collision/L4_treated_recall")]["estimate"] == 0.0
    assert rows[("A", "swerve/below_1m")]["estimate"] == 0.0 and rows[("B", "swerve/below_1m")]["estimate"] == 1.0
    assert rows[("A", "swerve/sign_ok_given_over_1m")] ["estimate"] == 1.0 and rows[("B", "swerve/sign_ok_given_over_1m")]["estimate"] is None
    assert rows[("A", "consequence/agree")]["den"] > 0 and rows[("A", "monotonicity")]["n"] == 2
    paired = {(r["comparison"], r["measure"]): r for r in result["paired"]}
    assert paired[("A_minus_B", "recipe/branch_share")]["estimate"] == pytest.approx(1.0)
    assert paired[("A_minus_B", "CTMS")]["estimate"] == pytest.approx(0.5) and paired[("A_minus_B", "CTMS")]["n"] == 2
    assert paired[("A_minus_B", "collision/L3_recall")]["estimate"] == pytest.approx(1.0)
    assert paired[("A_minus_B", "collision/L3_precision")]["n"] == 2                      # B has no collision label: (0, 0), kept
    assert rows[("A", "collision/L3_depth_error_abs")]["estimate"] == 0.0 and rows[("A", "collision/L4_treated_recall/npc_sudden_brake")]["estimate"] == 1.0
    assert rows[("B", "collision/L3_recall")]["n"] == 2 and rows[("B", "consequence/impossible")]["n"] == 0
    assert {p.name for p in out.iterdir()} == {"rows.csv", "paired.csv", "per_scenario.csv", "swerve_offsets.csv", "summary.md", "results.json"}
    offsets = list(csv.DictReader((out / "swerve_offsets.csv").open()))
    assert {o["arm"] for o in offsets} == {"A", "B", "control"} and len(offsets) == 6
    text = (out / "summary.md").read_text()
    assert "A minus B" in text and "branch_share" in text


def test_a_cell_run_under_the_other_prompt_is_refused(layout, tmp_path):
    a, b, baselines = layout
    with pytest.raises(ValueError, match="run under the recipe_withheld prompt"):
        ar.build({"A": b}, baselines, tmp_path / "r", boot=50, seed=0)


def test_the_condition_factor_is_read_within_an_arm_and_as_an_interaction(layout, tmp_path):
    """With video and history_json in an arm, the report gives the paired
    video - history_json difference, the representation identity between the
    two cells (descriptive) and, when all four cells exist, the interaction."""
    a, b, baselines = layout
    copies = _l3(_chain(STRAIGHT, n=2, last="collision"), _chain(SWERVE_LEFT), _chain(STRAIGHT, n=3, last="safe_stop"))
    shifted = _l3(_chain(lambda t: (10.0 * t + 1.5, 0.0)), _chain(lambda t: (10.0 * t, -0.6)), _chain(lambda t: (10.0 * t + 1.5, 0.0), n=3, last="safe_stop"))
    # arm A: history_json identical to video; arm B: every track differs from video by 0.2-0.5 m and the pair is answered as in video
    _write_cell(a / "m", "carla", "m_history_json", {"scenario_1": (copies, _l4(), {"monotonicity": 1.0, "CTMS": 0.9}),
                                                      "scenario_2": (copies, _l4(), {"monotonicity": 1.0, "CTMS": 0.8})}, "full")
    _write_cell(b / "m", "carla", "m_history_json", {"scenario_1": (shifted, _l4("maintain_course"), {"monotonicity": 1.0, "CTMS": 0.6}),
                                                      "scenario_2": (shifted, _l4("maintain_course"), {"monotonicity": 0.5, "CTMS": 0.1})}, "recipe_withheld")
    result = ar.build({"A": a, "B": b}, baselines, tmp_path / "report", boot=200, seed=0)
    ident = {(r["arm"], r["measure"]): r for r in result["rows"] if r["condition"] == "video~history_json"}
    assert ident[("A", "identity/branch_identical_0.1m")]["estimate"] == 1.0
    assert ident[("B", "identity/branch_identical_0.1m")]["estimate"] == 0.0 and ident[("B", "identity/branch_within_1m")]["estimate"] == 1.0
    assert ident[("A", "identity/target_identical_0.1m")]["estimate"] == 1.0 and ident[("A", "identity/branch_same_length")]["estimate"] == 1.0
    paired = {(r["comparison"], r["arm"], r["measure"]): r for r in result["paired"]}
    assert paired[("video_minus_history_json", "A", "CTMS")]["estimate"] == pytest.approx(0.0)
    assert paired[("video_minus_history_json", "B", "CTMS")]["estimate"] == pytest.approx((0.4 + 0.3) / 2 - (0.6 + 0.1) / 2)
    assert paired[("video_minus_history_json", "B", "recipe/branch_share")]["n"] == 2
    inter = paired[("interaction_recipe_x_condition", "A-B", "CTMS")]
    assert inter["estimate"] == pytest.approx(((0.9 - 0.4) + (0.8 - 0.3)) / 2 - ((0.9 - 0.6) + (0.8 - 0.1)) / 2) and inter["n"] == 2
    assert paired[("interaction_recipe_x_condition", "A-B", "collision/L3_recall")]["estimate"] == pytest.approx(0.0)
    text = (tmp_path / "report" / "summary.md").read_text()
    assert "video minus history_json" in text and "Interaction" in text and "identity" in text
    assert ar.track_identity([(0.0, 0.0), None], [(0.05, 0.0), (9.0, 9.0)]) == pytest.approx(0.05)
    assert ar.track_identity([None], [(0.0, 0.0)]) is None


def test_the_thinking_and_decoding_effects_are_paired_within_an_arm(layout, tmp_path):
    """QT - QS isolates thinking and QS - QD isolates the sampling; both are
    read within an arm, paired on shared scenarios, like the Cosmos pair."""
    a, b, baselines = layout
    copies = _l3(_chain(STRAIGHT, n=2, last="collision"), _chain(SWERVE_LEFT), _chain(STRAIGHT, n=3, last="safe_stop"))
    drift = _l3(_chain(lambda t: (10.0 * t + 1.0, 0.0)), _chain(SWERVE_WEAK), _chain(lambda t: (10.0 * t + 1.0, 0.0), n=3, last="safe_stop"))
    for name, l3, ctms in (("qwen3.5-9b-thinking", copies, 0.9), ("qwen3.5-9b-direct-sampled", drift, 0.4), ("qwen3.5-9b-direct", drift, 0.2)):
        _write_cell(a / name, "carla", f"{name}_video", {"scenario_1": (l3, _l4(), {"monotonicity": 1.0, "CTMS": ctms}),
                                                         "scenario_2": (l3, _l4(), {"monotonicity": 1.0, "CTMS": ctms})}, "full")
    result = ar.build({"A": a, "B": b}, baselines, tmp_path / "report", boot=200, seed=0)
    paired = {(r["comparison"], r["arm"], r["measure"]): r for r in result["paired"]}
    think = paired[("thinking_minus_sampled", "A", "CTMS")]
    assert think["estimate"] == pytest.approx(0.5) and think["n"] == 2
    assert think["configuration"] == "qwen3.5-9b-thinking-qwen3.5-9b-direct-sampled"
    assert paired[("thinking_minus_sampled", "A", "recipe/branch_share")]["estimate"] == pytest.approx(1.0)
    assert paired[("sampled_minus_greedy", "A", "CTMS")]["estimate"] == pytest.approx(0.2)
    assert paired[("sampled_minus_greedy", "A", "recipe/branch_share")]["estimate"] == pytest.approx(0.0)
    assert ("thinking_minus_sampled", "B", "CTMS") not in paired                        # arm B has no such cells here
    text = (tmp_path / "report" / "summary.md").read_text()
    assert "thinking_minus_sampled" in text and "sampled_minus_greedy" in text



def test_a_third_arm_is_contrasted_with_both_others(layout, tmp_path):
    """Arm C (`ego_profiles_withheld`) is paired with A and with B on the
    shared scenarios; A - B is unchanged; a C cell listed as another arm is
    refused like any other mismatch."""
    a, b, baselines = layout
    c = tmp_path / "arm_c"
    copies = _l3(_chain(STRAIGHT, n=2, last="collision"), _chain(SWERVE_LEFT), _chain(STRAIGHT, n=3, last="safe_stop"))
    _write_cell(c / "m", "carla", "m_video", {"scenario_1": (copies, _l4(), {"monotonicity": 1.0, "CTMS": 0.7}),
                                               "scenario_2": (copies, _l4(), {"monotonicity": 1.0, "CTMS": 0.6})},
                "ego_profiles_withheld")
    result = ar.build({"A": a, "B": b, "C": c}, baselines, tmp_path / "r3", boot=50, seed=0)
    paired = {(r["comparison"], r["measure"]): r for r in result["paired"]}
    assert paired[("A_minus_B", "CTMS")]["estimate"] == pytest.approx(0.5)
    assert paired[("A_minus_C", "CTMS")]["estimate"] == pytest.approx((0.9 + 0.8) / 2 - (0.7 + 0.6) / 2)
    assert paired[("C_minus_B", "CTMS")]["estimate"] == pytest.approx((0.7 + 0.6) / 2 - (0.4 + 0.3) / 2)
    assert paired[("A_minus_C", "recipe/branch_share")]["estimate"] == pytest.approx(0.0)
    assert {r["arm"] for r in result["rows"]} == {"A", "B", "C", "control"}
    text = (tmp_path / "r3" / "summary.md").read_text()
    assert "A minus B" in text and "A minus C" in text and "C minus B" in text and "C = ego_profiles_withheld" in text
    with pytest.raises(ValueError, match="run under the ego_profiles_withheld prompt"):
        ar.build({"A": c}, baselines, tmp_path / "r4", boot=50, seed=0)


def test_a_cell_pulled_without_inputs_is_relinked_not_skipped(layout, tmp_path, monkeypatch):
    """A root pulled from the pod with --exclude '/inputs/' keeps its cells in
    the report: discovery recreates inputs/ through reparse_cells.split_inputs
    (fingerprints checked against frozen.json) instead of skipping the cell."""
    import scripts.reparse_cells as rc
    a, b, baselines = layout
    saved = tmp_path / "saved_inputs"
    shutil.copytree(a / "m" / "inputs" / "carla", saved)
    shutil.rmtree(a / "m" / "inputs")
    frozen = {"datasets": {"carla": [{"name": "scenario_1"}, {"name": "scenario_2"}]}}
    (a / "m" / "frozen.json").write_text(json.dumps(frozen))
    calls = []

    def relink(base, split, frozen_arg):
        calls.append((base, split, frozen_arg))
        shutil.copytree(saved, base / "inputs" / split)
        return base / "inputs" / split

    monkeypatch.setattr(rc, "split_inputs", relink)
    result = ar.build({"A": a, "B": b}, baselines, tmp_path / "r", boot=50, seed=0)
    assert calls == [(a / "m", "carla", frozen)]
    assert {r["arm"] for r in result["rows"]} == {"A", "B", "control"}


def test_a_cell_with_neither_inputs_nor_frozen_inputs_stops_the_report(layout, tmp_path):
    a, b, baselines = layout
    shutil.rmtree(a / "m" / "inputs")
    with pytest.raises(FileNotFoundError, match="neither inputs/carla nor frozen.json"):
        ar.build({"A": a, "B": b}, baselines, tmp_path / "r", boot=50, seed=0)
