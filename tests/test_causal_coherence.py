"""
Causal-coherence metrics: ground-truth-free scoring of whether a
model's branch set obeys causal axioms, checked against the model's OWN output
(no twin, no ground truth). Three axioms: braking monotonicity, intervention
non-collapse, intervention locality.

Validation strategy encoded here + run end-to-end: GT is coherent by
construction (scores ~1), a collapsed set scores ~0 on non-collapse, an
order-violating set scores low on monotonicity.
"""

import os
import sys
import json

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from analysis import causal_coherence as cc  # noqa: E402


def _root():
    return {"event_type": "observed_state",
            "actors": [{"id": "ego", "entity_type": "ego_vehicle", "role_type": "ego_agent",
                        "x": 0.0, "y": 0.0, "vx": 8.0, "vy": 0.0, "speed": 8.0},
                       {"id": "npc_1", "entity_type": "vehicle", "role_type": "lead_vehicle",
                        "x": 20.0, "y": 0.0, "vx": 4.0, "vy": 0.0, "speed": 4.0}]}


def _chain(ego_terminal_x, ego_terminal_y=0.0, barrier_x=15.0, depth=4):
    """A chain ending with ego at x and a STATIC barrier at barrier_x."""
    nodes = []
    for d in range(1, depth + 1):
        f = d / depth
        nodes.append({"depth": d, "timestamp": 0.5 * d, "event_type": "lane_follow",
                      "actors_update": [
                          {"id": "ego", "entity_type": "ego_vehicle", "role_type": "ego_agent",
                           "x": ego_terminal_x * f, "y": ego_terminal_y * f},
                          {"id": "bar_1", "entity_type": "barrier", "role_type": "static_barrier",
                           "x": barrier_x, "y": -6.0}]})
    return nodes


def _pairs(mapping):
    """mapping: {intervention_type: ego_terminal_x} -> list of (iv, chain)."""
    return [(iv, _chain(x)) for iv, x in mapping.items()]


@pytest.mark.parametrize('missing_index', [0, 1, 2, 3])
@pytest.mark.parametrize('missing_node', [None, 'bad', 42])
def test_locality_ignores_missing_slots_without_shifting_other_depths(missing_index, missing_node):
    import copy
    chain = _chain(24.0)
    for i, node in enumerate(chain):
        node['actors_update'][1]['x'] = 15.0 + 2 * i
    incomplete = copy.deepcopy(chain)
    incomplete[missing_index] = missing_node
    pred = {'predicted_branches': [
        {'intervention': {'intervention_type': 'do_ego_no_action'}, 'chain': chain},
        {'intervention': {'intervention_type': 'do_ego_comfortable_brake'}, 'chain': incomplete},
    ]}
    pairs = cc.pairs_from_prediction(pred)
    assert cc.intervention_locality(pairs, _root()) == 1.0


# ── Braking monotonicity ─────────────────────────────────────────────────

def test_monotonicity_perfect_when_order_respected():
    # harder brake -> shorter forward displacement (correct causal order)
    pairs = _pairs({"do_ego_panic_brake": 4.0, "do_ego_comfortable_brake": 10.0,
                    "do_ego_no_action": 24.0, "do_ego_accelerate_pass": 30.0})
    assert cc.braking_monotonicity(pairs, _root()) == pytest.approx(1.0)


def test_monotonicity_zero_when_order_inverted():
    # harder brake -> LONGER displacement (causally wrong)
    pairs = _pairs({"do_ego_panic_brake": 30.0, "do_ego_comfortable_brake": 20.0,
                    "do_ego_no_action": 10.0, "do_ego_accelerate_pass": 4.0})
    assert cc.braking_monotonicity(pairs, _root()) == pytest.approx(0.0)


def test_monotonicity_none_when_no_displacement_spread():
    # every intervention gives the SAME displacement -> no comparable pair to
    # order -> None (a collapsed set cannot earn a misleading monotonicity 1.0)
    pairs = _pairs({"do_ego_panic_brake": 20.0, "do_ego_no_action": 20.0,
                    "do_ego_accelerate_pass": 20.0})
    assert cc.braking_monotonicity(pairs, _root()) is None


# ── Responsiveness (interventions must produce distinguishable futures) ───

def test_responsiveness_high_when_spread_is_large():
    pairs = _pairs({"do_ego_panic_brake": 4.0, "do_ego_no_action": 24.0,
                    "do_ego_accelerate_pass": 30.0})
    assert cc.responsiveness(pairs, _root()) == pytest.approx(1.0)


def test_responsiveness_zero_when_all_identical():
    # the "single continuation dressed as branches" failure
    pairs = _pairs({"do_ego_panic_brake": 20.0, "do_ego_no_action": 20.0,
                    "do_ego_accelerate_pass": 20.0})
    assert cc.responsiveness(pairs, _root()) == pytest.approx(0.0)


def test_responsiveness_none_with_single_intervention():
    assert cc.responsiveness(_pairs({"do_ego_no_action": 20.0}), _root()) is None


# ── Intervention locality (STATIC actors only) ───────────────────────────

def test_locality_high_when_static_actor_stable():
    # a barrier stays put regardless of ego action -> locality respected
    pairs = _pairs({"do_ego_panic_brake": 4.0, "do_ego_no_action": 24.0})
    assert cc.intervention_locality(pairs, _root()) == pytest.approx(1.0)


def test_locality_low_when_static_actor_swings_with_ego_action():
    pairs = [("do_ego_panic_brake", _chain(4.0, barrier_x=15.0)),
             ("do_ego_no_action", _chain(24.0, barrier_x=40.0))]  # barrier "moved"
    assert cc.intervention_locality(pairs, _root()) < 1.0


def test_locality_none_when_no_static_actors():
    # chains with only dynamic actors -> locality does not apply (folds out)
    dyn = [("do_ego_panic_brake", [{"depth": 1, "actors_update": [
                {"id": "ego", "role_type": "ego_agent", "x": 4.0, "y": 0.0},
                {"id": "npc_1", "entity_type": "vehicle", "role_type": "lead_vehicle",
                 "x": 20.0, "y": 0.0}]}]),
           ("do_ego_no_action", [{"depth": 1, "actors_update": [
                {"id": "ego", "role_type": "ego_agent", "x": 24.0, "y": 0.0},
                {"id": "npc_1", "entity_type": "vehicle", "role_type": "lead_vehicle",
                 "x": 25.0, "y": 0.0}]}])]
    assert cc.intervention_locality(dyn, _root()) is None


# ── Composite + adapters ─────────────────────────────────────────────────

def test_coherence_reports_axes_separately_no_naive_composite():
    pairs = _pairs({"do_ego_panic_brake": 4.0, "do_ego_comfortable_brake": 10.0,
                    "do_ego_no_action": 24.0})
    out = cc.coherence_from_pairs(pairs, _root())
    # three axioms reported separately; NO single averaged 'composite' (audit #1)
    assert set(out) >= {"monotonicity", "responsiveness", "locality", "structure"}
    assert "composite" not in out
    # 'structure' is the ceiling-axioms-only summary (mono + locality), NOT
    # averaged with the no-target responsiveness
    ceiling = [out[k] for k in ("monotonicity", "locality") if out[k] is not None]
    assert out["structure"] == pytest.approx(sum(ceiling) / len(ceiling), abs=1e-9)


def test_responsiveness_distinguishes_lateral_interventions():
    """swerve-left vs swerve-right have equal forward displacement but opposite
    lateral: a longitudinal-only measure would call them collapsed (audit #4b);
    the 2D measure must not."""
    left = [{"depth": 1, "actors_update": [
        {"id": "ego", "role_type": "ego_agent", "x": 20.0, "y": 3.6}]}]
    right = [{"depth": 1, "actors_update": [
        {"id": "ego", "role_type": "ego_agent", "x": 20.0, "y": -3.6}]}]
    r = cc.responsiveness([("do_ego_swerve_left", left), ("do_ego_swerve_right", right)], _root())
    assert r > 0.0


def test_collapsed_set_scores_below_coherent_on_responsiveness():
    coherent = _pairs({"do_ego_panic_brake": 4.0, "do_ego_no_action": 24.0})
    collapsed = _pairs({"do_ego_panic_brake": 20.0, "do_ego_no_action": 20.0})
    assert cc.responsiveness(coherent, _root()) > cc.responsiveness(collapsed, _root())


def test_pairs_from_gt_uses_branch_interventions():
    gt = {"observed_state": _root(),
          "branches": [
              {"is_factual": True, "intervention": {"intervention_type": "do_ego_no_action"},
               "chain": _chain(24.0)},
              {"is_factual": False, "intervention": {"intervention_type": "do_ego_panic_brake"},
               "chain": _chain(4.0)}]}
    pairs = cc.pairs_from_gt(gt)
    assert {iv for iv, _ in pairs} == {"do_ego_no_action", "do_ego_panic_brake"}


def test_pairs_from_prediction_uses_l3_only_and_ignores_l4():
    pred = {"predicted_branches": [{"is_factual": True,
                                     "intervention": {"intervention_type": "do_ego_no_action",
                                                      "target_actor_id": "ego"},
                                     "chain": _chain(24.0)}],
            "predicted_counterfactuals": [
                {"intervention": {"intervention_type": "do_ego_panic_brake"}, "chain": _chain(4.0)}]}
    pairs = cc.pairs_from_prediction(pred, factual_action="do_ego_no_action")
    ivs = {iv for iv, _ in pairs}
    assert ivs == {"do_ego_no_action"}


def test_protocol_coherence_does_not_recover_missing_action_from_gt(tmp_path):
    scenario = tmp_path / "scenarios" / "scenario_1"
    prediction = tmp_path / "pred" / "scenario_1"
    scenario.mkdir(parents=True)
    prediction.mkdir(parents=True)
    (scenario / "observed_state.json").write_text(json.dumps(_root()))
    (scenario / "ground_truth.json").write_text(json.dumps({
        "scenario_id": "secret-id",
        "observed_state": _root(),
        "branches": [{
            "is_factual": True,
            "intervention": {"intervention_type": "do_ego_no_action"},
            "chain": _chain(24.0),
        }],
    }))
    (prediction / "l3_pre.json").write_text(json.dumps({
        "predicted_branches": [{"is_factual": True, "chain": _chain(24.0)}],
    }))

    out = cc.run_coherence(str(tmp_path / "scenarios"), str(tmp_path / "pred"))
    assert out["measurement_role"] == "primary"
    assert out["uses_reference_answers"] is False
    assert out["used_legacy_reference_metadata"] is False
    assert out["per_scenario"][0]["n_interventions"] == 0
    assert out["per_scenario"][0]["scenario_id"] == "scenario_1"

    legacy = cc.run_coherence(
        str(tmp_path / "scenarios"), str(tmp_path / "pred"), factual_from_gt=True)
    assert legacy["used_legacy_reference_metadata"] is True
    assert legacy["per_scenario"][0]["n_interventions"] == 1


def test_protocol_coherence_requires_standalone_observed_state(tmp_path):
    scenario = tmp_path / "scenarios" / "scenario_1"
    prediction = tmp_path / "pred" / "scenario_1"
    scenario.mkdir(parents=True)
    prediction.mkdir(parents=True)
    (scenario / "input_manifest.json").write_text(json.dumps({
        "protocol_version": "coriva-protocol-1.0",
    }))
    (scenario / "ground_truth.json").write_text(json.dumps({
        "observed_state": _root(),
        "branches": [],
    }))
    (prediction / "l3_pre.json").write_text(json.dumps({
        "predicted_branches": [],
    }))

    with pytest.raises(ValueError, match="requires observed_state.json"):
        cc.run_coherence(str(tmp_path / "scenarios"), str(tmp_path / "pred"))


def test_primary_coherence_condition_deltas_are_paired_per_axis():
    def report(values):
        return {
            "coherence_version": cc.COHERENCE_VERSION,
            "uses_reference_answers": False,
            "per_scenario": [
                {"scenario_id": sid, "monotonicity": value,
                 "responsiveness": value, "locality": value}
                for sid, value in values.items()
            ],
        }

    video = report({"s1": 0.4, "s2": 0.5})
    history = report({"s1": 0.3, "s2": 0.4})
    both = report({"s1": 0.7, "s2": 0.8})
    out = cc.coherence_condition_deltas(video, history, both, n_boot=100, seed=0)
    assert out["measurement_role"] == "primary"
    for axis in ("monotonicity", "responsiveness", "locality"):
        deltas = out["axes"][axis]
        assert deltas["video_history_minus_video"]["mean_diff"] == pytest.approx(0.3)
        assert deltas["video_history_minus_history_json"]["mean_diff"] == pytest.approx(0.4)
        assert deltas["video_minus_history_json"]["mean_diff"] == pytest.approx(0.1)


def test_primary_coherence_condition_deltas_average_seeds_first():
    def report(value):
        return {
            "coherence_version": cc.COHERENCE_VERSION,
            "uses_reference_answers": False,
            "per_scenario": [{"scenario_id": "s1",
                              "monotonicity": value,
                              "responsiveness": value,
                              "locality": value}],
        }

    video = [report(0.2), report(0.4)]
    history = [report(0.1), report(0.3)]
    both = [report(0.6), report(0.8)]
    out = cc.coherence_condition_deltas(video, history, both, n_boot=10, seed=0)
    for axis in ("monotonicity", "responsiveness", "locality"):
        deltas = out["axes"][axis]
        assert deltas["video_history_minus_video"]["mean_diff"] == pytest.approx(0.4)
        assert deltas["video_history_minus_history_json"]["mean_diff"] == pytest.approx(0.5)
        assert deltas["video_minus_history_json"]["mean_diff"] == pytest.approx(0.1)


def test_primary_coherence_summary_averages_seeds_before_bootstrap():
    def report(a, b):
        return {
            "coherence_version": cc.COHERENCE_VERSION,
            "uses_reference_answers": False,
            "per_scenario": [
                {"scenario_id": "s1", "monotonicity": a,
                 "responsiveness": a, "locality": a},
                {"scenario_id": "s2", "monotonicity": b,
                 "responsiveness": b, "locality": b},
            ],
        }

    out = cc.summarize_coherence_reports(
        [report(0.2, 0.4), report(0.6, 0.8)], n_boot=100, seed=0)
    assert out["measurement_role"] == "primary"
    assert out["n_seeds"] == 2 and out["n_scenarios"] == 2
    for axis in ("monotonicity", "responsiveness", "locality"):
        assert out["axes"][axis]["mean"] == pytest.approx(0.5)
        assert out["axes"][axis]["n"] == 2
        assert out["axes"][axis]["lo"] <= 0.5 <= out["axes"][axis]["hi"]


def test_gt_is_ceiling_on_order_and_locality_axioms():
    """The validity anchor: the ground truth is coherent by construction, so it
    must sit at the CEILING on the two GT-referenced axioms — braking
    monotonicity and static-actor locality (both ~1.0). Responsiveness is a
    collapse-detector, not a GT-matched target (it reflects how divergent a
    scene's interventions genuinely are), so it is not asserted here."""
    import glob
    import json
    paths = sorted(glob.glob("pilot30/scenario_*/ground_truth.json"))
    if not paths:
        pytest.skip("pilot30 not built")
    mono, loc = [], []
    for p in paths:
        gt = json.load(open(p))
        out = cc.coherence_from_pairs(cc.pairs_from_gt(gt), gt["observed_state"])
        if out["monotonicity"] is not None:
            mono.append(out["monotonicity"])
        if out["locality"] is not None:
            loc.append(out["locality"])
    assert mono, "no scenario yielded a monotonicity score"
    assert sum(mono) / len(mono) >= 0.95, f"GT monotonicity should be ~1, got {sum(mono)/len(mono)}"
    if loc:
        assert sum(loc) / len(loc) >= 0.95, f"GT locality should be ~1, got {sum(loc)/len(loc)}"


def test_collapsed_prediction_scores_below_gt_on_responsiveness():
    """copy_factual emits one chain for every intervention -> responsiveness 0,
    well below the GT's -- the collapse the metric exists to catch. Compared on
    responsiveness (the collapse axiom) rather than a naive composite (audit #1)."""
    import glob
    import json
    from analysis.baselines import copy_factual_text
    from analysis.parse_prediction import parse_model_output
    paths = sorted(glob.glob("pilot30/scenario_*/ground_truth.json"))
    if not paths:
        pytest.skip("pilot30 not built")
    gt_r, cf_r = [], []
    for p in paths:
        gt = json.load(open(p))
        root = gt["observed_state"]
        gr = cc.responsiveness(cc.pairs_from_gt(gt), root)
        if gr is not None:
            gt_r.append(gr)
        l3, _l4, _e = parse_model_output(copy_factual_text(gt), gt.get("scenario_id", "s"))
        fa = next((b["intervention"]["intervention_type"] for b in gt["branches"]
                   if b.get("is_factual")), None)
        pairs = cc.pairs_from_prediction(l3, factual_action=fa)
        cr = cc.responsiveness(pairs, root)
        if cr is not None:
            cf_r.append(cr)
    assert gt_r and cf_r
    assert sum(cf_r) / len(cf_r) < sum(gt_r) / len(gt_r)


# ── coherence 1.2: forward travel is compared at the same depth ──────────

def test_monotonicity_compares_forward_travel_at_the_same_depth():
    # The accelerating branch ends in a collision at depth 2, so by its LAST
    # node it has travelled less (12 m) than the brake that runs to depth 4
    # (16 m). At depth 2 it has travelled more (12 m vs 8 m), which is the
    # ordering the actions imply. Reading endpoints would score this 0.
    brake = _chain(16.0, depth=4)
    accel = _chain(24.0, depth=4)[:2]
    accel[-1]["event_type"] = "collision"
    pairs = [("do_ego_comfortable_brake", brake), ("do_ego_accelerate_pass", accel)]
    assert cc.braking_monotonicity(pairs, _root()) == pytest.approx(1.0)
    inverted = [("do_ego_comfortable_brake", accel), ("do_ego_accelerate_pass", brake)]
    assert cc.braking_monotonicity(inverted, _root()) == pytest.approx(0.0)


def test_monotonicity_uses_the_deepest_slot_both_branches_report():
    # A missing node keeps its slot; the comparison moves to the deepest depth
    # at which both branches report the ego (depth 3 here), never to a shifted index.
    brake = _chain(8.0, depth=4)
    accel = _chain(24.0, depth=4)
    accel[3] = None
    pairs = [("do_ego_comfortable_brake", brake), ("do_ego_accelerate_pass", accel)]
    assert cc.braking_monotonicity(pairs, _root()) == pytest.approx(1.0)
