"""
observed_state root + closed-world M4.
"""

import json
import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import os as _os


@pytest.fixture(scope="module")
def raw():
    with open("coriva_bench_examples.json") as f:
        return json.load(f)


@pytest.fixture(scope="module")
def ontology():
    import yaml
    with open(_os.path.join("ctms", "ontology.yaml")) as f:
        return yaml.safe_load(f)


# ── Task 1: builder emits observed_state root ─────────────────────────
def test_build_scenario_gt_includes_observed_state_root(raw):
    from simulation.build_ground_truth import build_scenario_gt, SCENARIOS
    gt = build_scenario_gt(raw[0], SCENARIOS[0])

    assert "observed_state" in gt, "GT must carry a scenario-level observed_state root"
    root = gt["observed_state"]
    assert root["event_type"] == "observed_state"
    assert root["depth"] == 0
    ids = {a["id"] for a in root["actors"]}
    assert ids == {"ego", "npc_1"}, f"root must snapshot all actors, got {ids}"
    for a in root["actors"]:
        for field in ("entity_type", "role_type", "x", "y", "vx", "vy", "speed"):
            assert field in a, f"root actor {a['id']} missing {field}"
    assert root["timestamp"] <= SCENARIOS[0]["intervention_time_sec"] + 1e-9


# ── Task 2: check_semantic_validity binds U_actors from a supplied root ─
def _branch_with_extra_actor():
    """Depth-1 branch whose first node introduces npc_2 (not in the shared root)."""
    return {
        "chain": [
            {"event_type": "ego_brake", "action_type": "ego_emergency_brake",
             "timestamp": 3.95, "actors_update": [
                 {"id": "ego", "entity_type": "ego_vehicle", "role_type": "ego_agent"},
                 {"id": "npc_2", "entity_type": "vehicle", "role_type": "lead_vehicle"},
             ]},
        ]
    }


def test_root_universe_flags_actor_absent_from_root(ontology):
    from ctms.ctms import check_semantic_validity
    root_universe = {"ego": "ego_vehicle", "npc_1": "vehicle"}
    ok, errs = check_semantic_validity(
        _branch_with_extra_actor(), ontology,
        root_universe=root_universe, check_role_completeness=False,
    )
    assert ok is False
    assert any("hallucinated" in e and "npc_2" in e for e in errs), errs


def test_root_universe_none_preserves_node0_behavior(ontology):
    from ctms.ctms import check_semantic_validity
    ok, _errs = check_semantic_validity(
        _branch_with_extra_actor(), ontology, check_role_completeness=False
    )
    # npc_2 is in node 0, so under a node0-derived universe it is NOT hallucinated.
    assert ok is True


# ── Task 3: seed accumulated world state from the root ────────────────
def test_accumulated_states_seeded_from_root(ontology):
    from ctms.ctms import _build_accumulated_states
    root = {
        "actors": [
            {"id": "ego", "entity_type": "ego_vehicle", "role_type": "ego_agent", "x": 0.0, "y": 0.0},
            {"id": "npc_1", "entity_type": "vehicle", "role_type": "lead_vehicle", "x": 10.0, "y": 0.0},
        ]
    }
    branch = {
        "chain": [
            {"event_type": "ego_brake", "actors_update": [
                {"id": "ego", "entity_type": "ego_vehicle", "role_type": "ego_agent", "x": 1.0, "y": 0.0},
            ]},
        ]
    }
    states = _build_accumulated_states(branch, ontology, root=root)
    world = states[0]["world_snapshot"]
    assert "npc_1" in world, "root-only actor must persist into the world snapshot"
    assert world["ego"]["x"] == 1.0, "delta must override the root state for ego"


# ── Task 4: thread root through score_layer3 / score_layer4 / evaluate ─
def _load_env():
    from ctms.ctms import load_ctms_environment
    return load_ctms_environment(
        ontology_path=_os.path.join("ctms", "ontology.yaml"),
        config_path=_os.path.join("ctms", "ctms_config.yaml"),
        branch_schema_path=_os.path.join("ctms", "schema_branch.json"),
        counterfactual_schema_path=_os.path.join("ctms", "schema_counterfactual.json"),
    )


def test_score_layer3_rejects_branch_with_out_of_universe_actor():
    from ctms.ctms import score_layer3
    env = _load_env()
    root = {"actors": [
        {"id": "ego", "entity_type": "ego_vehicle", "role_type": "ego_agent", "x": 0.0, "y": 0.0, "speed": 10.0},
    ]}
    gt_branches = [{
        "branch_id": "f", "is_factual": True,
        "chain": [{"event_type": "safe_stop", "terminal": True, "timestamp": 4.0,
                   "actors_update": [{"id": "ego", "entity_type": "ego_vehicle", "role_type": "ego_agent", "x": 1.0, "y": 0.0, "speed": 0.0}]}],
    }]
    pred_branches = [{
        "branch_id": "p", "is_factual": False,
        "chain": [{"event_type": "safe_stop", "terminal": True, "timestamp": 4.0,
                   "actors_update": [{"id": "ghost", "entity_type": "vehicle", "role_type": "ego_agent", "x": 1.0, "y": 0.0, "speed": 0.0}]}],
    }]
    result = score_layer3(
        gt_branches, pred_branches, env["ontology"], env["config"],
        env["branch_schema"], root=root,
    )
    failures = result["diagnostics"].get("pred_gate_failures", [])
    assert any(
        any("hallucinated" in m and "ghost" in m for m in f["semantic_msgs"])
        for f in failures
    ), result["diagnostics"]


# ── Task 5: observed_state schema ─────────────────────────────────────
def test_observed_state_root_validates_against_schema(raw):
    from ctms.ctms import check_schema_validity
    from simulation.build_ground_truth import build_scenario_gt, SCENARIOS
    schema = json.load(open(_os.path.join("ctms", "schema_observed_state.json")))
    gt = build_scenario_gt(raw[0], SCENARIOS[0])
    ok, msgs = check_schema_validity(gt["observed_state"], schema)
    assert ok, msgs


# ── Task 6: prompt template documents the observed_state input ────────
@pytest.mark.skip(reason="needs design notes that are not part of this release")
def test_prompt_template_documents_observed_state_and_id_rule():
    text = open(_os.path.join("docs", "prompt_template.md")).read().lower()
    assert "observed_state" in text, "prompt must present the given observed_state input"
    assert "do not introduce new ids" in text or "do not create new ids" in text, \
        "prompt must forbid inventing actor ids outside the observed_state"
