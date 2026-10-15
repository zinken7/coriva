"""
Scoring-pipeline properties on the shipped fixture.

Each test asserts a current rule of the scorer or the reference builder.
They are written generically (no hard-coded prediction branch_ids) so they
survive data regeneration.
"""

import json
import os

import pytest

GT_PATH = os.path.join("output", "scenario_1", "ground_truth.json")

TERMINAL_EVENT_TYPES = {"safe_stop", "collision", "out_of_bounds"}


@pytest.fixture(scope="module")
def gt():
    with open(GT_PATH) as f:
        return json.load(f)


def _nodes(branch):
    return branch.get("chain", [])


def _actors(node):
    return node.get("actors") or node.get("actors_update") or []


# ── GT uses actors_update on non-observed_state nodes ────────────────────
def test_gt_non_observed_nodes_use_actors_update(gt):
    """Every non-observed_state node uses 'actors_update', never 'actors'."""
    offenders = [
        (b["branch_id"], n.get("depth"))
        for b in gt["branches"]
        for n in _nodes(b)
        if n.get("event_type") != "observed_state" and "actors" in n
    ]
    assert offenders == [], f"non-observed_state nodes still using 'actors' key: {offenders}"

    # And confirm they positively carry actors_update
    missing = [
        (b["branch_id"], n.get("depth"))
        for b in gt["branches"]
        for n in _nodes(b)
        if n.get("event_type") != "observed_state" and "actors_update" not in n
    ]
    assert missing == [], f"non-observed_state nodes missing 'actors_update': {missing}"


# ── A stopping branch terminates with safe_stop ──────────────────────────
def test_stopping_branch_terminates_with_safe_stop(gt):
    """A branch whose ego stops and holds resolves to a safe_stop terminal
    node (the braking branch; the swerve branch of this fixture reaches the
    horizon without a terminal event)."""
    stopping = next((b for b in gt["branches"] if _nodes(b)[-1]["event_type"] == "safe_stop"), None)
    assert stopping is not None, "no branch of the fixture ends with safe_stop"

    last = _nodes(stopping)[-1]
    assert last["event_type"] == "safe_stop", (
        f"stopping branch last node should be safe_stop, got {last['event_type']!r}"
    )
    assert last.get("terminal") is True, (
        f"stopping branch last node should have terminal=True, got {last.get('terminal')!r}"
    )
