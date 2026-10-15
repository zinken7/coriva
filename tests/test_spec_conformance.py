"""
Spec-conformance tests for CTMS (outline Section V).

Encodes the exact formulas from Research/outline.md and asserts ctms.py
matches them. Targets two audited divergences:

  H1 — S_split must divide by D_max (Section 5.3), not D_max+1.
  H2 — S_term = 1 only if BOTH branches terminate AND terminal types match
       (Section 5.3); two non-terminal branches sharing a last event_type
       must score 0.
"""

import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


# ── H1: split similarity denominator ──────────────────────────────────
# Spec 5.3:  S_split = max(0, 1 - |T_split(b_i) - T_split(b̂_j)| / D_max)

_CFG = {"bounds": {"D_max": 10}}


def test_h1_split_similarity_unit_difference():
    """|diff|=1, D_max=10 → 1 - 1/10 = 0.9 (not 1 - 1/11 = 0.909...)."""
    from ctms.ctms import _split_similarity
    assert _split_similarity(1, 2, _CFG) == pytest.approx(0.9)


def test_h1_split_similarity_max_difference_is_zero():
    """|diff|=D_max=10 → 1 - 10/10 = 0.0 (the /11 bug yields 0.0909)."""
    from ctms.ctms import _split_similarity
    assert _split_similarity(1, 11, _CFG) == pytest.approx(0.0)


def test_h1_split_similarity_identical_is_one():
    """Identical split depths → 1.0 (unchanged by the fix)."""
    from ctms.ctms import _split_similarity
    assert _split_similarity(3, 3, _CFG) == pytest.approx(1.0)


# ── H2: terminal similarity requires both to terminate ─────────────────
# Spec 5.3:  S_term = 1 iff both branches terminate AND terminal types match

_TERMINALS = {"safe_stop", "collision", "out_of_bounds"}


def test_h2_terminal_match_both_terminal_scores_one():
    """Both end in the same terminal type → 1.0."""
    from ctms.ctms import _terminal_similarity
    assert _terminal_similarity("safe_stop", "safe_stop", _TERMINALS) == 1.0


def test_h2_terminal_nonterminal_match_scores_zero():
    """Two non-terminal branches sharing a last event_type → 0.0 (the bug)."""
    from ctms.ctms import _terminal_similarity
    assert _terminal_similarity("ego_steer", "ego_steer", _TERMINALS) == 0.0


def test_h2_terminal_mismatch_scores_zero():
    """Different terminal types → 0.0 (unchanged by the fix)."""
    from ctms.ctms import _terminal_similarity
    assert _terminal_similarity("safe_stop", "collision", _TERMINALS) == 0.0


def test_h2_terminal_empty_branches_score_zero():
    """Empty/absent terminal ('none') on both → 0.0 (not terminal)."""
    from ctms.ctms import _terminal_similarity
    assert _terminal_similarity("none", "none", _TERMINALS) == 0.0


# ── M3: Layer 4 must not over-reject via role-completeness ─────────────
# Spec Section 6: Layer 4 applies only G_schema·G_kin (plus basic structural
# validity); it must NOT reject collision/off-road outcomes of a forced
# intervention just because a required role (e.g. impacted_actor) is absent.
# check_semantic_validity gains a check_role_completeness flag; Layer 3 keeps
# the default (True), Layer 4 turns it off.

import os as _os


@pytest.fixture(scope="module")
def ontology():
    import yaml
    with open(_os.path.join("ctms", "ontology.yaml")) as f:
        return yaml.safe_load(f)


def _collision_branch_missing_impacted_actor():
    """Otherwise-valid chain ending in collision, but with no impacted_actor role."""
    ego = {"id": "ego", "entity_type": "ego_vehicle", "role_type": "ego_agent"}
    return {
        "chain": [
            {"event_type": "observed_state", "timestamp": 0.0, "actors": [dict(ego)]},
            {"event_type": "collision", "timestamp": 0.5, "terminal": True,
             "actors_update": [dict(ego)]},
        ]
    }


def test_m3_role_completeness_on_rejects_missing_impacted_actor(ontology):
    """With role completeness ON (Layer 3 / default), missing impacted_actor → invalid."""
    from ctms.ctms import check_semantic_validity
    ok, errs = check_semantic_validity(
        _collision_branch_missing_impacted_actor(), ontology, check_role_completeness=True
    )
    assert ok is False
    assert any("impacted_actor" in e for e in errs), errs


def test_m3_role_completeness_off_allows_missing_impacted_actor(ontology):
    """With role completeness OFF (Layer 4), the collision outcome is NOT rejected for roles."""
    from ctms.ctms import check_semantic_validity
    ok, errs = check_semantic_validity(
        _collision_branch_missing_impacted_actor(), ontology, check_role_completeness=False
    )
    assert ok is True, errs


def test_m3_default_preserves_role_completeness(ontology):
    """Default (no flag) keeps role completeness ON, so Layer 3 behavior is unchanged."""
    from ctms.ctms import check_semantic_validity
    ok, _errs = check_semantic_validity(
        _collision_branch_missing_impacted_actor(), ontology
    )
    assert ok is False


def test_m3_structural_checks_still_apply_when_roles_off(ontology):
    """Role completeness OFF must NOT disable hallucination detection."""
    from ctms.ctms import check_semantic_validity
    branch = _collision_branch_missing_impacted_actor()
    # Inject a hallucinated actor not present in the root universe.
    branch["chain"][1]["actors_update"].append(
        {"id": "ghost", "entity_type": "vehicle", "role_type": "impacted_actor"}
    )
    ok, errs = check_semantic_validity(branch, ontology, check_role_completeness=False)
    assert ok is False
    assert any("hallucinated" in e for e in errs), errs


# ── M1: P_role compares role-holder actor IDs ────────────────────────────
# Spec 3.1: a required role assignment is correct iff the role is held by the
# same actor IDs as in the reference and every holder has an allowed entity
# type, so naming the wrong collision partner does not keep full categorical
# credit.

def _collision_node(impacted_id, impacted_etype="vehicle"):
    return {
        "event_type": "collision",
        "terminal": True,
        "actors_update": [
            {"id": "ego", "entity_type": "ego_vehicle", "role_type": "ego_agent"},
            {"id": impacted_id, "entity_type": impacted_etype, "role_type": "impacted_actor"},
        ],
    }


def test_m1_p_role_requires_same_impacted_actor_id(ontology):
    """Same roles + allowed entity types but a different impacted-actor ID → the
    impacted_actor role earns nothing; ego_agent still counts → P_role = 0.5."""
    from ctms.ctms import _sim_cat
    gt = _collision_node("npc_1")
    assert _sim_cat(gt, _collision_node("npc_1"), ontology) == pytest.approx(1.0)
    assert _sim_cat(gt, _collision_node("npc_2"), ontology) == pytest.approx(0.5)


def test_m1_p_role_still_penalizes_wrong_entity_type(ontology):
    """A required role filled by a disallowed entity_type still loses credit."""
    from ctms.ctms import _sim_cat
    gt = _collision_node("npc_1")
    pred = _collision_node("npc_1", impacted_etype="traffic_cone")  # not allowed for collision
    assert _sim_cat(gt, pred, ontology) == pytest.approx(0.5)


def test_m1_p_role_multi_partner_uses_id_set_overlap(ontology):
    """Two reference partners, one predicted → Jaccard 1/2 on that role."""
    from ctms.ctms import _sim_cat
    gt = _collision_node("npc_1")
    gt["actors_update"].append({"id": "npc_2", "entity_type": "vehicle", "role_type": "impacted_actor"})
    pred = _collision_node("npc_1")
    assert _sim_cat(gt, pred, ontology) == pytest.approx((1.0 + 0.5) / 2)


# ── M2: Sim_cont spatial contract (IoU → distance fallback, default τ) ─
# Spec 3.2 + audit M2: bbox present → IoU; absent → max(0, 1 - dist/τ).
# The code default for τ must match the configured value (2.0 m).

def _pos_node(x, y, bbox=None):
    a = {"id": "a1", "entity_type": "vehicle", "role_type": "lead_vehicle", "x": x, "y": y}
    if bbox is not None:
        a["bbox"] = bbox
    return {"event_type": "lane_follow", "actors_update": [a]}


def test_m2_sim_cont_uses_iou_when_bbox_present(ontology):
    from ctms.ctms import _sim_cont
    cfg = {"bbox_format": "cx_cy_w_h"}
    gt = _pos_node(0, 0, bbox=[0, 0, 2, 2])
    pred_same = _pos_node(0, 0, bbox=[0, 0, 2, 2])
    pred_far = _pos_node(0, 0, bbox=[10, 10, 2, 2])
    assert _sim_cont(gt, pred_same, cfg, ontology) == pytest.approx(1.0)
    assert _sim_cont(gt, pred_far, cfg, ontology) == pytest.approx(0.0)


def test_m2_sim_cont_distance_fallback_uses_configured_threshold(ontology):
    from ctms.ctms import _sim_cont
    cfg = {"node_similarity": {"thresholds": {"position_distance_m": 2.0}}}
    gt = _pos_node(0, 0)
    pred = _pos_node(1, 0)  # dist=1, τ=2 → 1 - 1/2 = 0.5
    assert _sim_cont(gt, pred, cfg, ontology) == pytest.approx(0.5)


def test_m2_sim_cont_default_threshold_matches_config(ontology):
    """With no threshold in config, the default τ must be 2.0 m (not 5.0)."""
    from ctms.ctms import _sim_cont
    cfg = {}
    gt = _pos_node(0, 0)
    pred = _pos_node(2, 0)  # dist=2 → with τ=2 → 0.0 (τ=5 would give 0.6)
    assert _sim_cont(gt, pred, cfg, ontology) == pytest.approx(0.0)


# ── L1W: Layer 1 categorical/continuous weighting is a free parameter ──
# Dissertation 4.3: Sim = w·Sim_cat + (1−w)·Sim_cont with w = 0.5 by default.
# The parameter-perturbation study (Section 5.4) varies w, so it must be
# configurable as layer1.cat_weight while the default preserves old scores.

def _l1w_nodes():
    """Same categorical content (lane_follow constraints satisfied);
    continuous score forced to 0 (position ≥ τ, speed fully off), so Sim
    isolates the weighting."""
    def node(x, speed):
        return {"event_type": "lane_follow", "action_type": "ego_no_action",
                "actors_update": [
                    {"id": "ego", "entity_type": "ego_vehicle",
                     "role_type": "ego_agent", "x": x, "y": 0.0,
                     "speed": speed}]}
    return node(0.0, 10.0), node(50.0, 0.0)


def test_l1w_default_weighting_preserves_equal_split(ontology):
    from ctms.ctms import _node_similarity, _sim_cat
    gt, pred = _l1w_nodes()
    cat = _sim_cat(gt, pred, ontology)
    assert cat > 0
    assert _node_similarity(gt, pred, ontology, {}) == pytest.approx(0.5 * cat)


def test_l1w_cat_weight_is_configurable(ontology):
    from ctms.ctms import _node_similarity, _sim_cat
    gt, pred = _l1w_nodes()
    cat = _sim_cat(gt, pred, ontology)
    cfg = {"layer1": {"cat_weight": 0.8}}
    assert _node_similarity(gt, pred, ontology, cfg) == pytest.approx(0.8 * cat)
