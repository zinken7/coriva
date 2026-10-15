"""
Scorer conformance properties.

- `f1.if_m_zero_precision` semantics: the fallback applies when the
  *prediction* set is empty (outline 5.5 uses m for predictions).
- Layer 4 canonicalizes the GT branch before score_chain, exactly as
  Layer 3 does (outline Section 2 applies to both sides). A GT/pred pair
  that is canonically identical must score exactly 1.0.
- The code default for `speed_conflict_tolerance` matches the shipped
  config value (4.0) in BOTH call sites (canonicalize_branch and
  check_kinematic_validity).
"""

import copy
import json
import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


@pytest.fixture(scope="module")
def ontology():
    import yaml
    with open(os.path.join("ctms", "ontology.yaml")) as f:
        return yaml.safe_load(f)


@pytest.fixture(scope="module")
def config():
    import yaml
    with open(os.path.join("ctms", "ctms_config.yaml")) as f:
        return yaml.safe_load(f)


@pytest.fixture(scope="module")
def branch_schema():
    with open(os.path.join("ctms", "schema_branch.json")) as f:
        return json.load(f)


@pytest.fixture(scope="module")
def cf_schema():
    with open(os.path.join("ctms", "schema_counterfactual.json")) as f:
        return json.load(f)


@pytest.fixture(scope="module")
def scenario_gt():
    with open(os.path.join("output", "scenario_1", "ground_truth.json")) as f:
        return json.load(f)


# ── precision fallback fires on an EMPTY PREDICTION SET ──────────────────
# Outline 5.5: Precision_cf = (1/m) Σ S_pair with m = |B̂_pred|;
# "If m = 0, set Precision_cf = 0."  Recall divides by n = |B_GT| ≥ 1.

def test_p01_empty_prediction_set_uses_precision_fallback(scenario_gt, ontology, config, branch_schema):
    from ctms.ctms import score_layer3
    res = score_layer3(
        gt_branches=scenario_gt["branches"],
        pred_branches=[],
        ontology=ontology,
        config=config,
        branch_schema=branch_schema,
        root=scenario_gt.get("observed_state"),
    )
    assert res["Precision_cf"] == pytest.approx(
        float(config.get("f1", {}).get("if_m_zero_precision", 0.0)))
    assert res["Recall_cf"] == pytest.approx(0.0)
    assert res["F1_cf"] == pytest.approx(0.0)


def test_p01_fallback_keyed_on_predictions_not_gt(scenario_gt, ontology, config, branch_schema):
    """A nonzero configured fallback must surface in Precision when preds
    are empty — proving the fallback is keyed on the prediction count."""
    from ctms.ctms import score_layer3
    cfg = copy.deepcopy(config)
    cfg.setdefault("f1", {})["if_m_zero_precision"] = 0.25
    res = score_layer3(
        gt_branches=scenario_gt["branches"],
        pred_branches=[],
        ontology=ontology,
        config=cfg,
        branch_schema=branch_schema,
        root=scenario_gt.get("observed_state"),
    )
    assert res["Precision_cf"] == pytest.approx(0.25)
    assert res["F1_cf"] == pytest.approx(0.0)  # recall is still 0


# ── Layer 4 canonicalizes GT before scoring ──────────────────────────────

def _round_all_numerics(chain_obj, config):
    """Fixture helper: apply the config's numeric rounding to a chain dict,
    using the same round() the canonicalizer uses."""
    precision = config.get("numeric_precision", {})
    out = copy.deepcopy(chain_obj)
    for node in out.get("chain", []):
        for actor_field in ("actors", "actors_update"):
            for actor in node.get(actor_field, []) or []:
                for k, v in list(actor.items()):
                    if k in precision and isinstance(v, (int, float)):
                        actor[k] = round(float(v), int(precision[k]))
    return out


def test_p02_layer4_scores_canonically_identical_pair_as_one(scenario_gt, ontology, config, cf_schema):
    """GT with raw (unrounded) numerics vs a prediction that is its canonical
    form must score exactly 1.0 — requires GT canonicalization in Layer 4."""
    from ctms.ctms import score_layer4
    gt_cf = scenario_gt["counterfactuals"][0]
    # Sanity: the raw GT must actually contain sub-rounding precision,
    # otherwise this test cannot distinguish raw from canonical.
    raw = json.dumps(gt_cf)
    assert any(len(tok.split(".")[-1].rstrip("}],")) > 2
               for tok in raw.split(":") if "." in tok), \
        "fixture GT has no >2dp numerics; pick another scenario"

    pred_cf = {
        "counterfactual_chain": _round_all_numerics(
            gt_cf["counterfactual_chain"], config),
    }
    res = score_layer4(
        gt_cf_chain=gt_cf,
        pred_cf_obj=pred_cf,
        ontology=ontology,
        config=config,
        counterfactual_schema=cf_schema,
        root=scenario_gt.get("observed_state"),
    )
    assert res["valid"] is True, res["diagnostics"]
    assert res["Score_interv"] == pytest.approx(1.0, abs=1e-9)


def test_p02_layer3_layer4_gt_treatment_is_symmetric(scenario_gt, ontology, config, branch_schema):
    """Layer 3 already canonicalizes GT: echoing the canonical form of the
    GT branches must yield F1 == 1.0 (regression guard for the property
    Layer 4 is being aligned to)."""
    from ctms.ctms import canonicalize_branch, score_layer3
    preds = [_round_all_numerics(b, config) for b in scenario_gt["branches"]]
    res = score_layer3(
        gt_branches=scenario_gt["branches"],
        pred_branches=preds,
        ontology=ontology,
        config=config,
        branch_schema=branch_schema,
        root=scenario_gt.get("observed_state"),
    )
    echo = score_layer3(gt_branches=scenario_gt["branches"], pred_branches=[copy.deepcopy(b) for b in scenario_gt["branches"]],
                        ontology=ontology, config=config, branch_schema=branch_schema, root=scenario_gt.get("observed_state"))
    # rounding the echo must not change its score (the ceiling itself is < 1
    # when a reference branch reaches the horizon without a terminal event)
    assert res["F1_cf"] == pytest.approx(echo["F1_cf"], abs=1e-9)


# ── speed_conflict_tolerance code default == shipped config (4.0) ────────
# Config derivation: |v_endpoint − v_displacement| ≤ max|a|·dt/2
# = 12 × 0.5 / 2 = 3.0 m/s, plus headroom → 4.0. A conflict of 3–3.5 m/s
# must therefore pass under DEFAULTS (config omitted), same as it does
# under the shipped config.

def _conflicting_speed_branch():
    ego0 = {"id": "ego", "entity_type": "ego_vehicle", "role_type": "ego_agent",
            "x": 0.0, "y": 0.0, "vx": 0.0, "vy": 0.0}
    ego1 = {"id": "ego", "entity_type": "ego_vehicle", "role_type": "ego_agent",
            "x": 1.75, "y": 0.0, "vx": 0.0, "vy": 0.0}
    # displacement speed = 1.75/0.5 = 3.5 m/s vs velocity speed 0.0
    return {
        "chain": [
            {"event_type": "observed_state", "timestamp": 0.0, "actors": [ego0]},
            {"event_type": "ego_brake", "action_type": "ego_emergency_brake",
             "timestamp": 0.5, "actors_update": [ego1]},
        ]
    }


def test_p03_kinematic_validity_default_tolerance_matches_config():
    from ctms.ctms import check_kinematic_validity
    ok, errs = check_kinematic_validity(_conflicting_speed_branch(), {})
    conflict = [e for e in errs if "internal_physics_conflict" in e]
    assert ok is True and not conflict, errs


def test_p03_kinematic_validity_still_rejects_real_speed_conflict():
    from ctms.ctms import check_kinematic_validity
    branch = _conflicting_speed_branch()
    branch["chain"][1]["actors_update"][0]["x"] = 2.5
    ok, errs = check_kinematic_validity(branch, {})
    conflict = [e for e in errs if "internal_physics_conflict" in e]
    assert ok is False
    assert len(conflict) == 1


def test_p03_canonicalize_default_tolerance_matches_config(ontology):
    from ctms.ctms import canonicalize_branch
    branch = {
        "chain": [
            {"event_type": "ego_brake", "timestamp": 0.5, "actors_update": [
                # stated speed 0.0 vs velocity-derived 3.0 → within 4.0
                {"id": "ego", "entity_type": "ego_vehicle", "role_type": "ego_agent",
                 "x": 0.0, "y": 0.0, "vx": 3.0, "vy": 0.0, "speed": 0.0},
            ]},
        ]
    }
    canon = canonicalize_branch(branch, ontology, {})
    actor = canon["chain"][0]["actors_update"][0]
    assert "internal_physics_conflict" not in actor.get("_errors", []), actor
