"""
Preservation property tests for the CORIVA scoring pipeline.

These tests observe CORRECT behavior on UNFIXED code and assert it is preserved.
Tests 1–4 PASS on unfixed code (baseline behavior to preserve).
Test 5 FAILS on unfixed code (Bug 2 exists) — marked xfail(strict=True).

Requirements: 3.1, 3.2, 3.3, 3.4
"""

import copy
import json
import os
import sys

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

# ── Path setup ────────────────────────────────────────────────────────
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

GT_PATH = os.path.join("output", "scenario_1", "ground_truth.json")


# ── Fixtures ──────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def gt():
    with open(GT_PATH) as f:
        return json.load(f)


@pytest.fixture(scope="module")
def ontology():
    import yaml
    with open(os.path.join("ctms", "ontology.yaml")) as f:
        return yaml.safe_load(f)


# ══════════════════════════════════════════════════════════════════════
# Test 1 — Preservation: Counterfactual_Distracted collision terminal
# Requirement 3.1
#
# Observes that the collision terminal is already correct on unfixed GT.
# PASSES on unfixed code. Must continue to PASS after fixes.
# ══════════════════════════════════════════════════════════════════════

def test_preservation_counterfactual_distracted_collision_terminal(gt):
    """
    **Validates: Requirements 3.1**

    Counterfactual_Distracted chain[-1] has:
      - event_type: collision
      - terminal: True
      - no action_type key

    This is already correct on unfixed GT and must not regress.
    """
    distracted = next(
        (b for b in gt["branches"] if b["branch_id"] == "Counterfactual_Distracted"),
        None,
    )
    assert distracted is not None, "Counterfactual_Distracted branch not found"

    last = distracted["chain"][-1]
    assert last["event_type"] == "collision", (
        f"Expected collision terminal, got: {last['event_type']}"
    )
    assert last.get("terminal") is True, (
        f"Expected terminal=True on collision node, got: {last.get('terminal')!r}"
    )
    assert "action_type" not in last, (
        f"collision node must not have action_type, got: {last.get('action_type')!r}"
    )


# ══════════════════════════════════════════════════════════════════════
# Test 2 — Preservation: Factual_AEB safe_stop terminal
# Requirement 3.2
#
# Observes that the factual branch terminal and action_type are correct.
# PASSES on unfixed code. Must continue to PASS after fixes.
# ══════════════════════════════════════════════════════════════════════

def test_preservation_factual_aeb_safe_stop_terminal(gt):
    """
    **Validates: Requirements 3.2**

    Factual_AEB chain[-1] has event_type: safe_stop, terminal: True.
    All preceding nodes have action_type: ego_emergency_brake.
    """
    factual = next(
        (b for b in gt["branches"] if b["branch_id"] == "Factual_AEB"),
        None,
    )
    assert factual is not None, "Factual_AEB branch not found"

    chain = factual["chain"]
    last = chain[-1]
    assert last["event_type"] == "safe_stop", (
        f"Expected safe_stop terminal, got: {last['event_type']}"
    )
    assert last.get("terminal") is True, (
        f"Expected terminal=True on safe_stop node, got: {last.get('terminal')!r}"
    )

    # All preceding nodes must have action_type: ego_emergency_brake
    for node in chain[:-1]:
        assert node.get("action_type") == "ego_emergency_brake", (
            f"depth={node['depth']}: expected action_type=ego_emergency_brake, "
            f"got: {node.get('action_type')!r}"
        )


# ══════════════════════════════════════════════════════════════════════
# Test 3 — Preservation: _int_from_digits matching
# Requirement 3.4
#
# Verifies scenario ID numeric matching works correctly on unfixed code.
# PASSES on unfixed code. Must continue to PASS after fixes.
# ══════════════════════════════════════════════════════════════════════

def test_preservation_int_from_digits_matching():
    """
    **Validates: Requirements 3.4**

    _int_from_digits correctly extracts the numeric index from scenario IDs,
    enabling GT/pred directory alignment (scenario_001 ↔ scenario_1).
    """
    from scripts.assemble_dataset import _int_from_digits

    assert _int_from_digits("scenario_001") == 1, (
        "_int_from_digits('scenario_001') should return 1"
    )
    assert _int_from_digits("scenario_1") == 1, (
        "_int_from_digits('scenario_1') should return 1"
    )
    # Both map to the same integer — confirming the alignment logic works
    assert _int_from_digits("scenario_001") == _int_from_digits("scenario_1")


# ══════════════════════════════════════════════════════════════════════
# Test 4 — Property: _pre_normalize_branch idempotency
# Requirement 3.3
#
# For branch nodes that already use actors_update, _pre_normalize_branch
# returns them unchanged (idempotent). PASSES on unfixed code.
# ══════════════════════════════════════════════════════════════════════

# Strategy: generate a minimal branch node that already uses actors_update
_actor_strategy = st.fixed_dictionaries({
    "id": st.text(min_size=1, max_size=10),
    "entity_type": st.sampled_from(["ego_vehicle", "vehicle", "pedestrian"]),
    "role_type": st.sampled_from(["ego_agent", "lead_vehicle", "crossing_actor"]),
    "x": st.floats(min_value=-500.0, max_value=500.0, allow_nan=False, allow_infinity=False),
    "y": st.floats(min_value=-500.0, max_value=500.0, allow_nan=False, allow_infinity=False),
    "vx": st.floats(min_value=-30.0, max_value=30.0, allow_nan=False, allow_infinity=False),
    "vy": st.floats(min_value=-30.0, max_value=30.0, allow_nan=False, allow_infinity=False),
    "speed": st.floats(min_value=0.0, max_value=50.0, allow_nan=False, allow_infinity=False),
})

_non_observed_event_types = st.sampled_from([
    "ego_brake", "ego_steer", "ego_combined_maneuver", "ego_accelerate",
    "maintain_course", "safe_stop", "collision",
])

_node_with_actors_update = st.fixed_dictionaries({
    "depth": st.integers(min_value=1, max_value=20),
    "timestamp": st.floats(min_value=0.1, max_value=20.0, allow_nan=False, allow_infinity=False),
    "event_type": _non_observed_event_types,
    "actors_update": st.lists(_actor_strategy, min_size=1, max_size=3),
})

_branch_with_actors_update = st.fixed_dictionaries({
    "branch_id": st.text(min_size=1, max_size=20),
    "is_factual": st.booleans(),
    "chain": st.lists(_node_with_actors_update, min_size=1, max_size=5),
})


@given(branch=_branch_with_actors_update)
@settings(max_examples=100, deadline=None)
def test_preservation_pre_normalize_branch_idempotent(branch, ontology):
    """
    **Validates: Requirements 3.3**

    For any branch whose non-observed_state nodes already use actors_update,
    _pre_normalize_branch returns the branch unchanged (idempotent).

    This confirms that after the GT fix (actors → actors_update), the
    normalization adapter becomes a no-op for GT data.
    """
    from ctms.ctms import _pre_normalize_branch

    original = copy.deepcopy(branch)
    normalized = _pre_normalize_branch(branch, ontology)

    # The chain nodes must be unchanged — actors_update key preserved, no rename
    for orig_node, norm_node in zip(original["chain"], normalized["chain"]):
        assert "actors_update" in norm_node, (
            f"actors_update key was removed by _pre_normalize_branch at depth={orig_node['depth']}"
        )
        assert "actors" not in norm_node, (
            f"actors key was unexpectedly added by _pre_normalize_branch at depth={orig_node['depth']}"
        )
        assert norm_node["actors_update"] == orig_node["actors_update"], (
            f"actors_update content was mutated at depth={orig_node['depth']}"
        )


# ══════════════════════════════════════════════════════════════════════
# Test 5 — Property: finite horizon is not a terminal event
#
# A chain may end because the scored horizon is exhausted. It must only claim
# safe_stop if the emitted ego state is actually stopped.
# ══════════════════════════════════════════════════════════════════════

@given(d_max=st.integers(min_value=1, max_value=20))
@settings(max_examples=50, deadline=None)
def test_property_build_chain_never_labels_moving_ego_safe_stop(d_max):
    """
    For any requested horizon, a moving final ego state must not be rewritten
    as a terminal safe_stop merely because no further node is emitted.
    """
    import json as _json
    from simulation.build_ground_truth import build_chain, SCENARIOS

    raw = _json.load(open("coriva_bench_examples.json"))
    scenario_raw = raw[0]
    branch_steps = scenario_raw["branches"]["Counterfactual_Swerve"]

    # Build a scenario config with the given d_max
    sc = {k: v for k, v in SCENARIOS[0].items()}
    sc = dict(SCENARIOS[0])
    sc["d_max"] = d_max

    branch_meta = SCENARIOS[0]["branches"]["Counterfactual_Swerve"]
    chain = build_chain(branch_steps, sc, branch_meta)

    assert len(chain) > 0, "build_chain returned empty chain"

    last = chain[-1]
    ego = next(actor for actor in last["actors_update"] if actor["id"] == "ego")
    if ego["speed"] >= (1.0 / 3.6):
        assert last["event_type"] != "safe_stop", (
            f"d_max={d_max}: moving ego was mislabeled safe_stop")
        assert last.get("terminal") is not True
