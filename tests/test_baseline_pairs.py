"""Every non-learning control and the oracle answer an L4 query with a
matched control/treated pair (docs/l4_pair_protocol.md §6), and the pair
passes the scorer's gates on a pair-protocol reference."""
import json
import math
import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from analysis.baselines import (  # noqa: E402
    action_profile_kinematic_text, action_profile_typical_text, action_profile_published_text, copy_factual_text,
    kinematic_extrapolation_text, random_valid_text)
from analysis.run_model import _oracle_text  # noqa: E402
from ctms.ctms import apply_reference_gate_policy, evaluate_ctms, load_ctms_environment  # noqa: E402
from simulation.build_nuscenes_dataset import _append_reviewed_l4, _prepare_candidate_spec  # noqa: E402
from simulation.kinematic_twin import TWIN_VERSION, build_twin_gt  # noqa: E402
from simulation.nuscenes_admission import canonical_environment_query, resolve_nuscenes_intervention  # noqa: E402
from simulation.protocol import protocol_manifest  # noqa: E402


def _root():
    return {"depth": 0, "timestamp": 0.0, "event_type": "observed_state", "actors": [
        {"id": "ego", "entity_type": "ego_vehicle", "role_type": "ego_agent", "x": 0.0, "y": 0.0,
         "vx": 10.0, "vy": 0.0, "speed": 10.0, "length": 4.0, "width": 1.8},
        {"id": "npc_1", "entity_type": "vehicle", "role_type": "lead_vehicle", "x": 25.0, "y": 0.0,
         "vx": 10.0, "vy": 0.0, "speed": 10.0, "length": 4.5, "width": 2.0},
        {"id": "cone_1", "entity_type": "traffic_cone", "role_type": "obstacle_actor", "x": 30.0, "y": -5.0,
         "vx": 0.0, "vy": 0.0, "speed": 0.0, "length": 0.4, "width": 0.4}]}


@pytest.fixture(scope="module")
def pair_gt():
    root = _root()
    spec = _prepare_candidate_spec("nuscenes_0001", root, None, "do_ego_no_action", 6, 0.5, "point_mass", 0.0,
                                   t0_visible_ids=["ego", "npc_1", "cone_1"], critical_actor_id="npc_1")
    applicability = resolve_nuscenes_intervention(root["actors"][1], "npc_sudden_brake")
    _append_reviewed_l4(spec, root, canonical_environment_query("npc_1", applicability))
    gt = build_twin_gt(root, spec)
    gt["scenario_template"] = spec["scenario_template"]
    gt["protocol"] = protocol_manifest(track="primary", split="v1.0-trainval")
    gt["protocol"].update(twin_version=TWIN_VERSION, l4_ego_policy="do_ego_no_action",
                          l4_pair_protocol="matched_control_treated")
    gt["sample_interval_sec"] = 0.5
    assert "control_chain" in gt["l4_queries"][0]
    return gt


@pytest.fixture(scope="module")
def env():
    e = load_ctms_environment("ctms/ontology.yaml", "ctms/ctms_config.yaml",
                              "ctms/schema_branch.json", "ctms/schema_counterfactual.json")
    return e


BASELINES = {
    "oracle": lambda gt: _oracle_text(gt),
    "copy_factual": copy_factual_text,
    "kinematic_extrapolation": kinematic_extrapolation_text,
    "action_profile_kinematic": action_profile_kinematic_text,
    "action_profile_typical": action_profile_typical_text,
    "action_profile_published": lambda gt: action_profile_published_text(
        gt, intervention_profiles={"npc_sudden_brake": {"accel": -6.0, "lat": 0.0, "source": "test"}}),
    "random_valid": lambda gt: random_valid_text(gt, 0),
}


def _actor(node, aid):
    return next(a for a in node["actors_update"] if a["id"] == aid)


@pytest.mark.parametrize("name", sorted(BASELINES))
def test_every_baseline_emits_a_complete_pair_per_query(pair_gt, name):
    out = json.loads(BASELINES[name](pair_gt))
    cfs = out["predicted_counterfactuals"]
    assert len(cfs) == len(pair_gt["l4_queries"])
    for unit in cfs:
        assert unit["intervention"] == pair_gt["l4_queries"][0]["counterfactual_chain"]["intervention"]
        assert isinstance(unit["control_chain"], list) and unit["control_chain"]
        assert isinstance(unit["treated_chain"], list) and unit["treated_chain"]
        assert "chain" not in unit


# `random_valid` is the chance floor under the gates only for a slow root: its
# random walk from a 10 m/s root fails the root-seeded kinematic gate in BOTH
# chains (a property of that control, unchanged by the pair protocol).
@pytest.mark.parametrize("name", sorted(n for n in BASELINES if n != "random_valid"))
def test_every_baseline_pair_is_pair_valid_under_the_scorer(pair_gt, env, name):
    out = json.loads(BASELINES[name](pair_gt))
    pred = {"branches": out["predicted_branches"], "predicted_counterfactuals": out["predicted_counterfactuals"]}
    cfg = apply_reference_gate_policy(env["config"], pair_gt)
    r = evaluate_ctms(pair_gt, pred, env["ontology"], cfg, env["branch_schema"], env["counterfactual_schema"])
    unit = r["layer4"]["results"][0]
    assert unit["pair"]["pair_valid"] is True, (name, unit["diagnostics"])
    if name == "oracle":
        assert unit["Score_interv"] == pytest.approx(1.0) and unit["control_similarity"] == pytest.approx(1.0)


@pytest.mark.parametrize("text", [action_profile_kinematic_text,
                                  lambda gt: action_profile_published_text(
                                      gt, intervention_profiles={"npc_sudden_brake": {"accel": -6.0, "lat": 0.0}})])
def test_action_profile_controls_keep_the_target_on_its_baseline_in_the_control_chain(pair_gt, text):
    unit = json.loads(text(pair_gt))["predicted_counterfactuals"][0]
    v0 = _actor({"actors_update": pair_gt["observed_state"]["actors"]}, "npc_1")["speed"]
    for node in unit["control_chain"]:
        assert _actor(node, "npc_1")["speed"] == pytest.approx(v0, abs=1e-6)
        assert _actor(node, "ego")["speed"] == pytest.approx(10.0, abs=1e-6)
    assert _actor(unit["treated_chain"][0], "npc_1")["speed"] < v0 - 1.0
    assert _actor(unit["treated_chain"][0], "ego")["speed"] == pytest.approx(10.0, abs=1e-6)


def test_the_two_floors_differ_only_in_the_treated_magnitude(pair_gt):
    apk = json.loads(action_profile_kinematic_text(pair_gt))["predicted_counterfactuals"][0]
    app = json.loads(action_profile_published_text(
        pair_gt, intervention_profiles={"npc_sudden_brake": {"accel": -6.0, "lat": 0.0}}))["predicted_counterfactuals"][0]
    assert apk["control_chain"] == app["control_chain"]
    assert _actor(apk["treated_chain"][0], "npc_1")["speed"] == pytest.approx(10.0 - 4.0, abs=1e-6)  # ENV_PROFILES -8
    assert _actor(app["treated_chain"][0], "npc_1")["speed"] == pytest.approx(10.0 - 3.0, abs=1e-6)  # printed -6


def test_action_profile_pair_satisfies_the_reference_free_axes(pair_gt):
    from analysis.intervention_pair import L4_PAIR_AXES, pair_axes
    unit = json.loads(action_profile_kinematic_text(pair_gt))["predicted_counterfactuals"][0]
    axes = pair_axes(unit["control_chain"], unit["treated_chain"], pair_gt["observed_state"],
                     unit["intervention"], {"accel": -6.0, "lat": 0.0})
    assert [axes[a] for a in L4_PAIR_AXES] == [1.0, 1.0, 1.0], axes["details"]


def test_published_control_carries_no_hazard_script(pair_gt):
    """The hazard profile scripts the L3 branches; the L4 control keeps every
    actor at its root velocity, exactly as the kinematic control does."""
    hazard = {"actor_id": "npc_1", "accel": -4.0, "lat": 0.0}
    app = json.loads(action_profile_published_text(pair_gt, hazard_profile=hazard))["predicted_counterfactuals"][0]
    apk = json.loads(action_profile_kinematic_text(pair_gt))["predicted_counterfactuals"][0]
    assert app["control_chain"] == apk["control_chain"]
    assert all(_actor(n, "npc_1")["speed"] == pytest.approx(10.0, abs=1e-6) for n in app["control_chain"])


def test_oracle_on_a_legacy_reference_emits_only_the_treated_chain(pair_gt):
    import copy
    legacy = copy.deepcopy(pair_gt)
    legacy["l4_queries"][0].pop("control_chain")
    unit = json.loads(_oracle_text(legacy))["predicted_counterfactuals"][0]
    assert "control_chain" not in unit and unit["treated_chain"]
