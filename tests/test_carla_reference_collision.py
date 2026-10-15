"""Raw -> GT collision semantics for CARLA references.

Synthetic raw telemetry only; no CARLA. Each case pins one rule: the correct
collision partner, `safe_stop` never cutting a branch before an in-horizon
collision, a post-horizon collision never pulled into depth 6, and a
collision partner outside the observed root.
"""

import math

import pytest

from simulation.build_ground_truth import (
    ReferenceBuildError, build_chain, collision_partner_keys)

T0 = 3.5
STEP = 0.1


def _actor_defs():
    return {
        "ego": {"canonical_id": "ego", "entity_type": "ego_vehicle",
                "default_role": "ego_agent"},
        "npc_1": {"canonical_id": "npc_1", "entity_type": "vehicle",
                  "default_role": "lead_vehicle", "collision_role": "impacted_actor"},
        "park_l1": {"canonical_id": "park_l1", "entity_type": "parked_vehicle",
                    "default_role": "adjacent_vehicle"},
        "npc_3": {"canonical_id": "npc_3", "entity_type": "vehicle",
                  "default_role": "rear_vehicle"},
    }


def _config():
    return {"actor_defs": _actor_defs(), "intervention_time_sec": T0,
            "sample_interval_sec": 0.5, "d_max": 6,
            "branches": {"b": {"is_factual": True,
                               "intervention_type": "do_ego_emergency_brake",
                               "target_actor_id": "ego", "trigger_depth": 1}}}


def _state(x, y=0.0, speed=10.0):
    return {"x": x, "y": y, "vx": speed, "vy": 0.0, "speed_kmh": speed * 3.6,
            "heading_deg": 0.0}


def _steps(until, ego_stop_at=None, npc1=lambda t: _state(100.0, 0.0, 10.0),
           park=lambda t: _state(200.0, -3.5, 0.0),
           npc3=lambda t: _state(-300.0, 0.0, 10.0),
           collision_at=None, partners=None):
    steps = []
    n = int(round(until / STEP))
    for i in range(n + 1):
        t = round(i * STEP, 2)
        speed = 10.0
        if ego_stop_at is not None and t >= ego_stop_at:
            speed = 0.0
        step = {"time_sec": t, "ego": _state(10.0 * min(t, ego_stop_at or t), 0.0, speed),
                "npc_1": npc1(t), "park_l1": park(t), "npc_3": npc3(t)}
        if collision_at is not None and abs(t - collision_at) < 1e-6:
            step["collision"] = True
            step["collision_time"] = collision_at
            if partners is not None:
                step["collision_partners"] = list(partners)
        steps.append(step)
    return steps


def _chain(steps, root=("ego", "npc_1", "park_l1")):
    cfg = _config()
    return build_chain(steps, cfg, cfg["branches"]["b"], reference=None,
                       root_actor_ids=set(root), protocol=True)


def _impacted(node):
    return sorted(a["id"] for a in node["actors_update"]
                  if a["role_type"] == "impacted_actor")


def test_recorded_partner_is_the_impacted_actor_not_the_primary():
    # Ego swerves into the parked car at 5.0 s; npc_1 is far ahead.
    steps = _steps(5.0, collision_at=5.0, partners=["park_l1"])
    chain = _chain(steps)
    last = chain[-1]
    assert last["event_type"] == "collision" and last["terminal"] is True
    assert _impacted(last) == ["park_l1"]


def test_partner_reconstructed_from_overlap_when_raw_has_no_record():
    # Raw without collision_partners; park_l1 overlaps the ego footprint.
    def park(t):
        return _state(10.0 * 5.0 + 3.0, 1.5, 0.0) if t >= 5.0 else _state(200.0, -3.5, 0.0)
    steps = _steps(5.0, park=park, collision_at=5.0)
    assert collision_partner_keys(steps[-1], _actor_defs()) == ["park_l1"]
    assert _impacted(_chain(steps)[-1]) == ["park_l1"]


def test_unresolvable_partner_fails_closed():
    steps = _steps(5.0, collision_at=5.0)  # nothing overlaps, nothing recorded
    with pytest.raises(ReferenceBuildError, match="partner"):
        _chain(steps)


def test_safe_stop_does_not_hide_a_later_in_horizon_collision():
    # Ego stops at 4.5 s; the rear car npc_3 (here in root) hits it at 5.5 s.
    steps = _steps(5.5, ego_stop_at=4.5, collision_at=5.5, partners=["npc_3"])
    chain = _chain(steps, root=("ego", "npc_1", "park_l1", "npc_3"))
    events = [n["event_type"] for n in chain]
    assert "safe_stop" not in events
    assert events[-1] == "collision" and chain[-1]["depth"] == 4
    assert all("terminal" not in n for n in chain[:-1])
    assert _impacted(chain[-1]) == ["npc_3"]


def test_ego_stop_without_later_collision_is_still_safe_stop():
    steps = _steps(8.0, ego_stop_at=4.5)
    chain = _chain(steps)
    assert chain[-1]["event_type"] == "safe_stop" and chain[-1]["depth"] == 2


def test_post_horizon_collision_is_not_pulled_into_depth_6():
    # Collision at 6.8 s is after T0 + 3.0 = 6.5 s: the chain reaches the
    # horizon with six non-terminal nodes.
    steps = _steps(6.8, collision_at=6.8, partners=["npc_1"])
    chain = _chain(steps)
    assert len(chain) == 6
    assert all(n["event_type"] != "collision" for n in chain)
    assert "terminal" not in chain[-1]


def test_collision_exactly_at_horizon_end_is_kept():
    steps = _steps(6.5, collision_at=6.5, partners=["npc_1"])
    chain = _chain(steps)
    assert chain[-1]["event_type"] == "collision" and chain[-1]["depth"] == 6


def test_collision_between_samples_is_emitted_at_last_sample_before_it():
    steps = _steps(4.6, collision_at=4.6, partners=["npc_1"])
    chain = _chain(steps)
    assert chain[-1]["event_type"] == "collision" and chain[-1]["depth"] == 2


def test_collision_partner_outside_root_is_rejected():
    steps = _steps(5.5, ego_stop_at=4.5, collision_at=5.5, partners=["npc_3"])
    with pytest.raises(ReferenceBuildError, match="npc_3"):
        _chain(steps, root=("ego", "npc_1", "park_l1"))


def test_post_horizon_collision_with_hidden_partner_does_not_reject():
    steps = _steps(6.8, ego_stop_at=4.5, collision_at=6.8, partners=["npc_3"])
    chain = _chain(steps, root=("ego", "npc_1", "park_l1"))
    assert chain[-1]["event_type"] == "safe_stop"


def test_reconstruction_uses_runtime_wrapper_extents_and_heading():
    # Traffic sign is 1.5 x 0.5 in the runtime wrapper (street_barrier), and
    # the overlap test rotates into the ego heading frame.
    defs = _actor_defs()
    defs["sign_1"] = {"canonical_id": "sign_1", "entity_type": "traffic_sign",
                      "default_role": "static_barrier"}
    yaw = 90.0
    ego = {"x": 0.0, "y": 0.0, "heading_deg": yaw}
    # 2.9 m ahead along +y (heading 90): forward < (4.5+1.5)/2 = 3.0
    step = {"ego": ego, "sign_1": {"x": 0.0, "y": 2.9},
            "npc_1": {"x": 50.0, "y": 0.0}, "park_l1": {"x": 60.0, "y": 0.0},
            "npc_3": {"x": -60.0, "y": 0.0}}
    assert collision_partner_keys(step, defs) == ["sign_1"]
    step["sign_1"] = {"x": 0.0, "y": 3.1}
    assert collision_partner_keys(step, defs) == []
