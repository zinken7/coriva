"""kinematic-twin-5: both references realise one swerve.

CARLA-400 scenario_1 (straight road, ego 12.2 m/s, do_ego_swerve_left) records
y = -1.25, -2.5, -3.5 then holds; vy = -2.5, -2.5, 0; speed 12.45 then 12.2.
The twin (bicycle, base_steer 0) and the action-profile control (point-mass)
must emit the same ego states, so "both splits share the ego action profiles"
is a tested statement about the realised kinematics, not only the table.

What is proven, precisely:

* On a STRAIGHT route the CARLA ego (simulation/carla/ego.py, EgoActions) and
  the twin realise identical ego states for any map lane width: 2.5 m/s
  laterally (generate._SWERVE_INTENSITY = 2.5 / MAX_LATERAL 3.5) until the
  offset reaches the published EGO_SWERVE_LANE_WIDTH_M = 3.5 m, then held.
  This file pins the twin/control side against the CARLA-400 scenario_1
  record; tests/test_carla_primitives.py (`test_ego_swerve_on_wide_*`) runs
  the CARLA controller itself under the fake CARLA API on a 4.0 m lane and
  obtains the same y = -1.25, -2.5, -3.5, hold, with the real 4.0 m still in
  the raw `road_lane_width` evidence (the cap is the published lane width,
  not the map's, so a 4.0 m lane does not end 4.0 m off).
* On a CURVED route the two are NOT identical and this is a documented
  difference, not a tested equivalence: CARLA applies the offset
  perpendicular to the route centreline at the ego's current progress (road
  frame), whereas the twin applies the 2.5 m/s in the fixed T0 frame on top
  of its road-following (base_steer) path. On a straight route the two
  frames coincide, which is why the equality above holds there.
"""
import math
import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

CARLA_SCENARIO_1 = dict(y=[-1.25, -2.5, -3.5, -3.5, -3.5, -3.5],
                        vy=[-2.5, -2.5, 0.0, 0.0, 0.0, 0.0],
                        speed=[12.45, 12.45, 12.2, 12.2, 12.2, 12.2])


def _ego(vx=12.2):
    return {"id": "ego", "entity_type": "ego_vehicle", "role_type": "ego_agent",
            "x": 0.0, "y": 0.0, "vx": vx, "vy": 0.0, "speed": vx, "length": 4.5, "width": 2.0}


@pytest.mark.parametrize("model", ["bicycle", "point_mass"])
def test_twin_swerve_matches_the_carla_reference(model):
    from simulation.kinematic_twin import roll_branch
    states = roll_branch([_ego()], "do_ego_swerve_left", n_steps=6, dt=0.5, model=model)
    egos = [s["ego"] for s in states]
    assert [e["y"] for e in egos] == pytest.approx(CARLA_SCENARIO_1["y"])
    assert [e["vy"] for e in egos] == pytest.approx(CARLA_SCENARIO_1["vy"], abs=1e-6)
    assert [e["speed"] for e in egos] == pytest.approx(CARLA_SCENARIO_1["speed"], abs=1e-2)
    assert [e["x"] for e in egos] == pytest.approx([6.1 * k for k in range(1, 7)], abs=1e-6)


def test_action_profile_control_uses_the_same_swerve():
    """The control builds its branches with the point-mass twin, so it
    inherits the lane cap: no more 7.5 m after 3 s."""
    from analysis.baselines import action_profile_kinematic_text
    import json
    root = {"depth": 0, "timestamp": 0.0, "event_type": "observed_state", "actors": [_ego(), {
        "id": "npc_1", "entity_type": "vehicle", "role_type": "lead_vehicle",
        "x": 30.0, "y": 0.0, "vx": 12.0, "vy": 0.0, "speed": 12.0, "length": 4.5, "width": 2.0}]}
    gt = {"scenario_id": "s", "observed_state": root, "protocol": {"future_steps": 6},
          "branches": [{"intervention": {"intervention_type": a, "target_actor_id": "ego", "trigger_depth": 1},
                        "is_factual": a == "do_ego_no_action", "chain": []}
                       for a in ("do_ego_no_action", "do_ego_swerve_left")],
          "l4_queries": []}
    pred = json.loads(action_profile_kinematic_text(gt))
    swerve = next(b for b in pred["predicted_branches"] if b["intervention"]["intervention_type"] == "do_ego_swerve_left")
    ys = [next(a for a in n["actors_update"] if a["id"] == "ego")["y"] for n in swerve["chain"]]
    assert ys == pytest.approx(CARLA_SCENARIO_1["y"])


def test_brake_and_swerve_combines_both_profiles():
    from simulation.kinematic_twin import roll_branch, EGO_PROFILES
    states = roll_branch([_ego(vx=20.0)], "do_ego_brake_and_swerve_right", n_steps=4, dt=0.5, model="point_mass")
    egos = [s["ego"] for s in states]
    assert [e["y"] for e in egos] == pytest.approx([1.25, 2.5, 3.5, 3.5])
    assert egos[0]["vx"] == pytest.approx(20.0 + EGO_PROFILES["do_ego_brake_and_swerve_right"]["accel"] * 0.5)
    assert egos[0]["speed"] == pytest.approx(math.hypot(egos[0]["vx"], 2.5), abs=1e-2)
