"""Two reference-free L3 coherence axes on action semantics.

The three structural axes (ordering, endpoint spread, static locality) are
necessary conditions that never check the *semantics* of the requested action.
A swerve-left branch that moves right, or a braking branch that keeps the root
speed, could pass all three. These axes close that gap using only the model's
own branches, the supplied root state and the published action profiles:

  action_direction — a lateral action must displace the ego laterally in the
                     profile's direction relative to the lateral-neutral
                     branches (robust to road curvature shared by all branches);
  speed_response   — a braking action must end slower than the root, an
                     accelerating action faster, by more than the 1 m/s floor.
"""

import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from analysis import causal_coherence as cc  # noqa: E402


def _root(speed=8.0):
    return {"event_type": "observed_state",
            "actors": [{"id": "ego", "entity_type": "ego_vehicle", "role_type": "ego_agent",
                        "x": 0.0, "y": 0.0, "vx": speed, "vy": 0.0, "speed": speed}]}


def _chain(x, y=0.0, speed=8.0, depth=3):
    nodes = []
    for d in range(1, depth + 1):
        f = d / depth
        nodes.append({"depth": d, "timestamp": 0.5 * d, "event_type": "lane_follow",
                      "actors_update": [{"id": "ego", "entity_type": "ego_vehicle",
                                         "role_type": "ego_agent", "x": x * f, "y": y * f,
                                         "vx": speed, "vy": 0.0, "speed": speed}]})
    return nodes


# ── action_direction ──────────────────────────────────────────────────────

def test_action_direction_correct_when_swerve_left_moves_to_negative_y():
    pairs = [("do_ego_no_action", _chain(24.0, 0.0)),
             ("do_ego_swerve_left", _chain(24.0, -3.0))]
    assert cc.action_direction(pairs, _root()) == pytest.approx(1.0)


def test_action_direction_zero_when_swerve_left_moves_right():
    pairs = [("do_ego_no_action", _chain(24.0, 0.0)),
             ("do_ego_swerve_left", _chain(24.0, +3.0))]
    assert cc.action_direction(pairs, _root()) == pytest.approx(0.0)


def test_action_direction_zero_when_swerve_does_not_move_laterally():
    pairs = [("do_ego_no_action", _chain(24.0, 0.0)),
             ("do_ego_swerve_left", _chain(24.0, -0.2))]
    assert cc.action_direction(pairs, _root()) == pytest.approx(0.0)


def test_action_direction_is_relative_to_neutral_branches_on_curved_road():
    # Every branch drifts right (+y) with the road; swerve-left still ends left
    # of the lateral-neutral branches, which is the tested relation.
    pairs = [("do_ego_no_action", _chain(24.0, +5.0)),
             ("do_ego_comfortable_brake", _chain(12.0, +4.0)),
             ("do_ego_swerve_left", _chain(24.0, +1.0))]
    assert cc.action_direction(pairs, _root()) == pytest.approx(1.0)


def test_action_direction_compares_opposite_swerves_without_neutral_branch():
    pairs = [("do_ego_swerve_left", _chain(24.0, -3.0)),
             ("do_ego_swerve_right", _chain(24.0, +3.0))]
    assert cc.action_direction(pairs, _root()) == pytest.approx(1.0)
    swapped = [("do_ego_swerve_left", _chain(24.0, +3.0)),
               ("do_ego_swerve_right", _chain(24.0, -3.0))]
    assert cc.action_direction(swapped, _root()) == pytest.approx(0.0)


def test_action_direction_none_without_lateral_actions():
    pairs = [("do_ego_no_action", _chain(24.0)),
             ("do_ego_comfortable_brake", _chain(12.0))]
    assert cc.action_direction(pairs, _root()) is None


# ── speed_response ────────────────────────────────────────────────────────

def test_speed_response_correct_when_brake_ends_slower_than_root():
    pairs = [("do_ego_no_action", _chain(24.0, speed=8.0)),
             ("do_ego_comfortable_brake", _chain(12.0, speed=3.0))]
    assert cc.speed_response(pairs, _root(8.0)) == pytest.approx(1.0)


def test_speed_response_zero_when_brake_keeps_root_speed():
    pairs = [("do_ego_no_action", _chain(24.0, speed=8.0)),
             ("do_ego_comfortable_brake", _chain(12.0, speed=8.0))]
    assert cc.speed_response(pairs, _root(8.0)) == pytest.approx(0.0)


def test_speed_response_accelerating_action_must_end_faster():
    pairs = [("do_ego_accelerate_escape", _chain(30.0, speed=12.0))]
    assert cc.speed_response(pairs, _root(8.0)) == pytest.approx(1.0)
    pairs = [("do_ego_accelerate_escape", _chain(30.0, speed=8.5))]
    assert cc.speed_response(pairs, _root(8.0)) == pytest.approx(0.0)


def test_speed_response_reads_speed_from_velocity_when_speed_missing():
    chain = _chain(12.0, speed=8.0)
    for node in chain:
        node["actors_update"][0].pop("speed")
        node["actors_update"][0]["vx"] = 2.0
    assert cc.speed_response([("do_ego_panic_brake", chain)], _root(8.0)) == pytest.approx(1.0)


def test_speed_response_none_without_speed_changing_actions():
    pairs = [("do_ego_no_action", _chain(24.0)), ("do_ego_swerve_left", _chain(24.0, -3.0))]
    assert cc.speed_response(pairs, _root()) is None


def test_speed_response_excludes_braking_from_a_stationary_root():
    pairs = [("do_ego_comfortable_brake", _chain(0.0, speed=0.0))]
    assert cc.speed_response(pairs, _root(0.0)) is None


# ── composite and dataset-level plumbing ──────────────────────────────────

def test_coherence_from_pairs_reports_five_axes():
    pairs = [("do_ego_no_action", _chain(24.0, speed=8.0)),
             ("do_ego_comfortable_brake", _chain(12.0, speed=3.0)),
             ("do_ego_swerve_left", _chain(24.0, -3.0, speed=8.0))]
    out = cc.coherence_from_pairs(pairs, _root())
    assert set(cc.COHERENCE_AXES) == {"monotonicity", "responsiveness", "locality",
                                      "action_direction", "speed_response"}
    assert out["action_direction"] == pytest.approx(1.0)
    assert out["speed_response"] == pytest.approx(1.0)


def test_synthetic_twin_gt_sits_at_ceiling_on_new_axes():
    from analysis.rq3a import make_synthetic_gts
    for gt in make_synthetic_gts(6):
        out = cc.coherence_from_pairs(cc.pairs_from_gt(gt), gt["observed_state"])
        assert out["action_direction"] == pytest.approx(1.0)
        assert out["speed_response"] == pytest.approx(1.0)


def test_summary_and_deltas_include_new_axes():
    rows = [{"scenario_id": "s1", "monotonicity": 1.0, "responsiveness": 0.5, "locality": None,
             "action_direction": 1.0, "speed_response": 0.0},
            {"scenario_id": "s2", "monotonicity": 0.0, "responsiveness": 0.5, "locality": 1.0,
             "action_direction": 0.0, "speed_response": 1.0}]
    report = {"coherence_version": cc.COHERENCE_VERSION, "uses_reference_answers": False, "per_scenario": rows}
    summary = cc.summarize_coherence_reports(report, n_boot=50)
    assert summary["axes"]["action_direction"]["mean"] == pytest.approx(0.5)
    assert summary["axes"]["speed_response"]["n"] == 2
    deltas = cc.coherence_condition_deltas(report, report, report, n_boot=50)
    assert deltas["axes"]["speed_response"]["video_history_minus_video"]["n"] == 2


# ── Robustness: non-numeric fields are missing values, never exceptions ────
# Coherence reads predictions before any gate (by design), so a malformed
# field must make the affected branch unusable, not abort the whole cell.

def _bad_chain(field, value, speed=8.0):
    chain = _chain(12.0, speed=speed)
    for node in chain:
        node["actors_update"][0][field] = value
    return chain


def test_non_numeric_speed_falls_back_to_velocity_or_undefined_not_an_error():
    # speed "unknown" but vx/vy numeric: speed is recovered from the velocity
    chain = _bad_chain("speed", "unknown", speed=8.0)  # vx stays 8 -> no slowdown
    assert cc.speed_response([("do_ego_comfortable_brake", chain)], _root()) == pytest.approx(0.0)
    # neither speed nor velocity usable: the branch supplies no comparison
    for node in chain:
        node["actors_update"][0]["vx"] = None
    assert cc.speed_response([("do_ego_comfortable_brake", chain)], _root()) is None
    assert cc.speed_response([("do_ego_comfortable_brake", chain),
                              ("do_ego_panic_brake", _chain(6.0, speed=2.0))], _root()) == pytest.approx(1.0)


def test_non_numeric_position_is_skipped_by_every_axis():
    bad = _bad_chain("x", "n/a")
    good = _chain(24.0, 0.0)
    assert cc.braking_monotonicity([("do_ego_comfortable_brake", bad), ("do_ego_no_action", good)], _root()) is None
    assert cc.responsiveness([("do_ego_comfortable_brake", bad), ("do_ego_no_action", good)], _root()) is None
    lateral = _bad_chain("y", None)
    assert cc.action_direction([("do_ego_swerve_left", lateral), ("do_ego_no_action", good)], _root()) is None
    static = _chain(24.0)
    static[0]["actors_update"].append({"id": "bar", "entity_type": "barrier", "role_type": "static_barrier",
                                       "x": "?", "y": -6.0})
    assert cc.intervention_locality([("do_ego_no_action", static), ("do_ego_comfortable_brake", static)], _root()) is None


def test_run_coherence_survives_a_malformed_prediction(tmp_path):
    import json
    from analysis.rq3a import make_synthetic_gts
    gt = make_synthetic_gts(1)[0]
    d = tmp_path / "gt" / "scenario_1"; d.mkdir(parents=True)
    (d / "ground_truth.json").write_text(json.dumps(gt))
    pred = {"predicted_branches": []}
    for b in gt["branches"]:
        chain = json.loads(json.dumps(b["chain"]))
        for node in chain:
            for a in node["actors_update"]:
                a["speed"] = "unknown"; a["vx"] = None; a["vy"] = "n/a"
        pred["predicted_branches"].append({"branch_id": b["branch_id"], "is_factual": b["is_factual"],
                                           "intervention": b["intervention"], "chain": chain})
    pd = tmp_path / "pred" / "scenario_1"; pd.mkdir(parents=True)
    (pd / "l3_pre.json").write_text(json.dumps(pred))
    report = cc.run_coherence(str(tmp_path / "gt"), pred_dir=str(tmp_path / "pred"))
    assert report["n_scenarios"] == 1
    row = report["per_scenario"][0]
    assert row["speed_response"] is None and row["monotonicity"] is not None


# ── coherence 1.2: direction against the road path of the farthest neutral branch ──

def test_action_direction_is_not_defeated_by_a_neutral_branch_that_stops_early():
    # The panic brake stops after 1.5 m at depth 1. At that forward progress
    # the swerve has moved only 0.19 m sideways, so a pairwise comparison at
    # matched progress would score this 0.5 although the swerve is fully
    # developed. The lateral branch is compared with the road-following path
    # of the neutral branch that progresses farthest instead.
    pairs = [("do_ego_no_action", _chain(24.0, 0.0)),
             ("do_ego_panic_brake", _chain(1.5, 0.0, depth=1)),
             ("do_ego_swerve_left", _chain(24.0, -3.0))]
    assert cc.action_direction(pairs, _root()) == pytest.approx(1.0)
    wrong = [("do_ego_no_action", _chain(24.0, 0.0)),
             ("do_ego_panic_brake", _chain(1.5, 0.0, depth=1)),
             ("do_ego_swerve_left", _chain(24.0, +3.0))]
    assert cc.action_direction(wrong, _root()) == pytest.approx(0.0)


def test_action_direction_reads_the_road_from_the_farthest_neutral_branch_on_a_curve():
    # Both neutral branches drift right with the road; the brake stops early.
    # The swerve-left is judged against the no-action path at the swerve's
    # own forward progress, where the road offset is the same for both.
    pairs = [("do_ego_no_action", _chain(24.0, +5.0)),
             ("do_ego_panic_brake", _chain(3.0, +0.2, depth=1)),
             ("do_ego_swerve_left", _chain(24.0, +1.0))]
    assert cc.action_direction(pairs, _root()) == pytest.approx(1.0)
