"""
SP2 scenario config generator (multi-actor, realistic casts).

Generates N diverse CORIVA scenario configs from a few base templates. Each
scenario has a realistic cast — ego plus several other road users and static
objects (lead/adjacent/rear vehicles, crossing pedestrians/cyclists, parked
vehicles, barriers/cones) — with seeded parameter sampling. Output is the
unified config shape consumed by build_ground_truth.py and (on a server) the
CARLA runner. Each config also carries `env_behavior` ({id: {accel, lat}}) so
the kinematic twin / CARLA controllers can drive every non-ego actor.

Deterministic: the same (n, seed) yields identical configs.
CLI: python3 scenarios/generate_configs.py --n 400 --seed 0 --out-dir scenarios/generated
"""

import argparse
import math
from collections import Counter
import os
import random
import sys
from typing import Any, Dict, List, Optional

import yaml

if __package__ in (None, "") and __name__ == "__main__":
    # `python scenarios/generate_configs.py` puts scenarios/ on sys.path, not
    # the repository root, so the simulation package would not import. The
    # runbook (steps_full.md C4) uses that direct form.
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from simulation.protocol import LATERAL_NEAR_SIDE_M


# ── Branch sets (ego intervention set per scene type) ─────────────────
_REAR_END_BRANCHES = [
    ("Factual_AEB", True, "do_ego_emergency_brake"),
    ("CF_NoAction", False, "do_ego_no_action"),
    ("CF_Swerve", False, "do_ego_swerve_left"),
]
_YIELD_BRANCHES = [
    ("Factual_Yield", True, "do_ego_comfortable_brake"),
    ("CF_Panic", False, "do_ego_panic_brake"),
    ("CF_Swerve", False, "do_ego_swerve_left"),
]
_CUTIN_BRANCHES = [
    ("Factual_Brake", True, "do_ego_comfortable_brake"),
    ("CF_NoAction", False, "do_ego_no_action"),
    ("CF_Escape", False, "do_ego_accelerate_escape"),
]


# A cast actor: id, entity_type, role, spawn (distance_ahead, lateral_offset),
# whether it moves, and its environment-event behavior (longitudinal accel,
# lateral velocity). The first entry is the primary (collision) target.
def _actor(aid, entity, role, dist, lat, moving, accel=0.0, lat_v=0.0, primary=False):
    return {"id": aid, "entity_type": entity, "role": role, "dist": dist, "lat": lat,
            "moving": moving, "behavior": {"accel": accel, "lat": lat_v}, "primary": primary}


# Shared static street furniture is positioned from the desired T0 scene, then
# transformed back to the beginning of the 3.5 s observed window.  This keeps
# it in front of the camera instead of letting a fast ego pass it before T0.
def _street_furniture(ego_speed: float, occurrence: int) -> List[Dict[str, Any]]:
    travel = ego_speed * 3.5
    prop_entity = "traffic_cone" if occurrence % 2 else "traffic_sign"
    return [
        _actor("park_l1", "parked_vehicle", "adjacent_vehicle", travel + 15.0, -8.5, False),
        _actor("park_l2", "parked_vehicle", "adjacent_vehicle", travel + 38.0, -8.5, False),
        _actor("park_r1", "parked_vehicle", "adjacent_vehicle", travel + 22.0, 8.5, False),
        _actor("park_r2", "parked_vehicle", "adjacent_vehicle", travel + 48.0, 8.5, False),
        # These are isolated 1.5 m street barriers, not continuous map/curb
        # boundaries. Marking a point prop as road_boundary makes the scene
        # gate extend it to an infinite line and false-reject curved roads.
        _actor("wall_l", "guardrail", "static_barrier", travel + 32.0, -12.0, False),
        _actor("wall_r", "guardrail", "static_barrier", travel + 32.0, 12.0, False),
        _actor("sign_1", prop_entity, "static_barrier", travel + 42.0, 10.5, False),
    ]


# Eight balanced causal variants.  They deliberately cover semantic actor and
# interaction classes rather than every CARLA cosmetic blueprint.
TEMPLATES: List[Dict[str, Any]] = [
    {
        "template_id": "lead_car_brake",
        "scenario_type": "rear_end_collision_risk",
        "env_event_type": "npc_sudden_brake",
        "branches": _REAR_END_BRANCHES,
        "critical_entities": ("vehicle",), "critical_role": "lead_vehicle",
        "critical_lateral": 0.0, "critical_accel": -6.0, "critical_lat": 0.0,
    },
    {
        "template_id": "lead_large_vehicle_brake",
        "scenario_type": "rear_end_collision_risk",
        "env_event_type": "npc_sudden_brake",
        "branches": _REAR_END_BRANCHES,
        "critical_entities": ("truck", "bus"), "critical_role": "lead_vehicle",
        "critical_lateral": 0.0, "critical_accel": -5.0, "critical_lat": 0.0,
    },
    {
        "template_id": "car_cut_in_left",
        "scenario_type": "cut_in_collision_risk",
        "env_event_type": "npc_cut_in_left",
        "branches": _CUTIN_BRANCHES,
        "critical_entities": ("vehicle",), "critical_role": "adjacent_vehicle",
        "critical_lateral": 3.5, "critical_accel": 0.0, "critical_lat": -1.8,
    },
    {
        "template_id": "motorcycle_cut_in_right",
        "scenario_type": "cut_in_collision_risk",
        "env_event_type": "npc_cut_in_right",
        "branches": _CUTIN_BRANCHES,
        "critical_entities": ("motorcycle",), "critical_role": "adjacent_vehicle",
        "critical_lateral": -3.5, "critical_accel": 0.0, "critical_lat": 1.8,
    },
    {
        "template_id": "pedestrian_cross_left_to_right",
        "scenario_type": "pedestrian_collision_risk",
        "env_event_type": "pedestrian_enters_road",
        "branches": _YIELD_BRANCHES,
        "critical_entities": ("pedestrian",), "critical_role": "crossing_actor",
        "critical_lateral": -6.0, "critical_accel": 0.0, "critical_lat": 2.0,
    },
    {
        "template_id": "pedestrian_cross_right_to_left",
        "scenario_type": "pedestrian_collision_risk",
        "env_event_type": "pedestrian_enters_road",
        "branches": _YIELD_BRANCHES,
        "critical_entities": ("pedestrian",), "critical_role": "crossing_actor",
        "critical_lateral": 6.0, "critical_accel": 0.0, "critical_lat": -2.0,
    },
    {
        "template_id": "cyclist_cross_left_to_right",
        "scenario_type": "cyclist_collision_risk",
        "env_event_type": "cyclist_enters_road",
        "branches": _YIELD_BRANCHES,
        "critical_entities": ("bicycle",), "critical_role": "crossing_actor",
        "critical_lateral": -5.0, "critical_accel": 0.0, "critical_lat": 2.5,
    },
    {
        "template_id": "cyclist_cross_right_to_left",
        "scenario_type": "cyclist_collision_risk",
        "env_event_type": "cyclist_enters_road",
        "branches": _YIELD_BRANCHES,
        "critical_entities": ("bicycle",), "critical_role": "crossing_actor",
        "critical_lateral": 5.0, "critical_accel": 0.0, "critical_lat": -2.5,
    },
]

# The lateral escape steers AWAY from the critical actor (protocol
# `scenario_actions`): a crossing actor entering from the ego's left turns
# CF_Swerve into do_ego_swerve_right. Templates whose critical actor is ahead
# or on the right keep do_ego_swerve_left. The branch name is unchanged so
# directory layout and quotas stay stable.
for _t in TEMPLATES:
    _left_side = float(_t["critical_lateral"]) < -LATERAL_NEAR_SIDE_M
    _t["branches"] = [
        (name, is_factual,
         "do_ego_swerve_right" if (_left_side and action == "do_ego_swerve_left") else action)
        for name, is_factual, action in _t["branches"]]

MAPS = ["Town03", "Town04", "Town05", "Town10HD"]
# The released dataset size. Slot maps are planned for this size so that a
# slot's map never depends on how many slots a run builds (a --limit prefix
# or a unit-test sample sees the same map as the full release).
PLANNED_DATASET_SIZE = 400
# Maps that are predominantly one driving lane per direction. A template
# whose cast keeps a vehicle in an adjacent same-direction lane for the whole
# rollout (the lead-vehicle templates' adjacent car, the cut-in car before it
# moves) cannot be placed lawfully there: in the 2026-09-14 regeneration slot
# 18 (lead_large_vehicle_brake on Town10HD) failed all 231 candidate probes of
# ten seeds, 163 of them because no same-direction lane existed at +3.5 m for
# the 145-200 m the adjacent car drives. Crossing templates need no adjacent
# lane and take those maps' share instead; every map still receives exactly
# PLANNED_DATASET_SIZE / len(MAPS) slots.
SINGLE_CARRIAGEWAY_MAPS = ("Town10HD",)


def template_allowed_maps(template: Dict[str, Any]) -> List[str]:
    """Maps on which every actor of ``template`` can be placed on a lawful
    lane: crossing templates on all maps, templates with an adjacent-lane
    vehicle only on maps with dual carriageways."""
    if template["critical_role"] == "crossing_actor":
        return list(MAPS)
    return [name for name in MAPS if name not in SINGLE_CARRIAGEWAY_MAPS]


def _split_quota(total: int, names: List[str]) -> Dict[str, int]:
    """Largest-remainder split of ``total`` over ``names`` (first names get
    the remainder), so the parts are integers that sum to ``total``."""
    base, extra = divmod(total, len(names))
    return {name: base + (1 if i < extra else 0) for i, name in enumerate(names)}


# Per-template rotation offsets. Chosen (exhaustive search over 3**8 tables,
# 2026-09-14) so that with the formulas in ``slot_map`` every map receives
# exactly PLANNED_DATASET_SIZE / len(MAPS) slots and every template's
# occurrences are spread 16/17 (dual-carriageway templates) or 8/9/25
# (crossing templates) over its maps; ``_check_map_plan`` enforces both.
_MAP_ROTATION_OFFSETS = (0, 0, 1, 2, 0, 1, 1, 2)


def _latin_rotation(names: List[str], k: int, offset: int) -> str:
    """The k-th element of a rotation whose phase advances by one every
    len(names) steps: over len(names)**2 consecutive k every (name, k mod
    len(names)) pair occurs exactly once, so the ego-speed stratum (period 3)
    is not tied to a map, and the odd period leaves severity (period 2)
    balanced as well."""
    return names[(k + k // len(names) + offset) % len(names)]


def slot_map(index: int) -> str:
    """The planned map of slot ``index`` (0-based), a pure function of the
    template and its occurrence, so it never depends on how many slots a run
    builds. Dual-carriageway templates rotate over the maps that can host
    them. Crossing templates give the single-carriageway maps the seats the
    other templates cannot use: half of their occurrences (pattern
    single, dual, dual, single, so both the severity and the speed stratum
    see every map) and rotate the rest over the dual-carriageway maps."""
    template_index = index % len(TEMPLATES)
    occurrence = index // len(TEMPLATES)
    offset = _MAP_ROTATION_OFFSETS[template_index]
    allowed = template_allowed_maps(TEMPLATES[template_index])
    single = [name for name in allowed if name in SINGLE_CARRIAGEWAY_MAPS]
    dual = [name for name in allowed if name not in SINGLE_CARRIAGEWAY_MAPS]
    if not single:
        return _latin_rotation(dual, occurrence, offset)
    if occurrence % 4 in (0, 3):
        return single[(occurrence // 4) % len(single)]
    k = 2 * (occurrence // 4) + occurrence % 4 - 1
    return _latin_rotation(dual, k, offset)


def plan_maps(n: int = PLANNED_DATASET_SIZE) -> List[str]:
    """Planned map per slot index for ``n`` slots (see ``slot_map``)."""
    return [slot_map(index) for index in range(n)]


def _check_map_plan() -> None:
    """The release plan must give every map an equal share and spread every
    template over its allowed maps; fail at import if the templates, maps or
    offsets ever drift apart."""
    plan = plan_maps(PLANNED_DATASET_SIZE)
    counts = Counter(plan)
    if counts != Counter(_split_quota(PLANNED_DATASET_SIZE, MAPS)):
        raise ValueError(f"map plan is unbalanced: {dict(counts)}")
    n_single = len([name for name in MAPS if name in SINGLE_CARRIAGEWAY_MAPS])
    crossing = [t for t in TEMPLATES if len(template_allowed_maps(t)) == len(MAPS)]
    if n_single and 2 * n_single * len(TEMPLATES) != len(MAPS) * len(crossing):
        raise ValueError("slot_map assumes crossing templates give half their "
                         "occurrences to the single-carriageway maps")
    for template_index, template in enumerate(TEMPLATES):
        cells = Counter(plan[index] for index in range(template_index, len(plan), len(TEMPLATES)))
        allowed = template_allowed_maps(template)
        if set(cells) != set(allowed):
            raise ValueError(f"{template['template_id']} is planned on {sorted(cells)}, "
                             f"allowed {allowed}")
        dual = [cells[name] for name in allowed if name not in SINGLE_CARRIAGEWAY_MAPS]
        if max(dual) - min(dual) > 1:
            raise ValueError(f"{template['template_id']} is unevenly spread: {dict(cells)}")


_check_map_plan()


WEATHERS = ["ClearNoon", "CloudySunset", "WetCloudyNoon", "ClearSunset", "MidRainyNoon", "HardRainNoon"]
CARLA_CAMERA = {
    "name": "CAM_FRONT", "resolution": [1024, 768], "fov_deg": 100.0,
    "offset_m": [1.5, 0.0, 1.7], "pitch_deg": 0.0,
}
# Keep a crossing actor's centre at least five degrees inside the horizontal
# camera boundary at T0. Earlier history is farther away and therefore easier
# to see. This analytic floor prevents paying for CARLA rollouts that the
# all-eight-frame visibility gate must reject.
CROSSING_VISIBILITY_MARGIN_DEG = 5.0
CROSSING_VISIBILITY_HALF_FOV_DEG = (
    CARLA_CAMERA["fov_deg"] / 2.0 - CROSSING_VISIBILITY_MARGIN_DEG)
MODELS_BY_ENTITY = {
    "vehicle": ["vehicle.lincoln.mkz_2020", "vehicle.audi.a2", "vehicle.tesla.model3", "vehicle.nissan.patrol", "vehicle.toyota.prius", "vehicle.ford.mustang"],
    "truck": ["vehicle.carlamotors.carlacola", "vehicle.carlamotors.european_hgv", "vehicle.tesla.cybertruck"],
    "bus": ["vehicle.mitsubishi.fusorosa"],
    "motorcycle": ["vehicle.harley-davidson.low_rider", "vehicle.yamaha.yzf", "vehicle.kawasaki.ninja", "vehicle.vespa.zx125"],
    "parked_vehicle": ["vehicle.lincoln.mkz_2020", "vehicle.audi.a2", "vehicle.mercedes.coupe"],
    "pedestrian": ["walker.pedestrian.0001", "walker.pedestrian.0002", "walker.pedestrian.0003", "walker.pedestrian.0004", "walker.pedestrian.0009", "walker.pedestrian.0010", "walker.pedestrian.0030"],
    "bicycle": ["vehicle.bh.crossbike", "vehicle.diamondback.century", "vehicle.gazelle.omafiets"],
    "barrier": ["static.prop.streetbarrier"],
    "traffic_cone": ["static.prop.constructioncone", "static.prop.trafficcone01"],
    "guardrail": ["static.prop.streetbarrier"],
    "traffic_sign": ["static.prop.trafficwarning", "static.prop.warningaccident", "static.prop.warningconstruction"],
}
EGO_MODELS = ["vehicle.tesla.model3", "vehicle.audi.tt", "vehicle.bmw.grandtourer"]
_CROSSING_SPEED = {"pedestrian": 1.6, "bicycle": 4.0}
_BODY_EXTENTS = {
    "ego_vehicle": (4.5, 2.0), "vehicle": (4.5, 2.0),
    "truck": (7.0, 2.5), "bus": (12.0, 2.5),
    "motorcycle": (2.2, 0.8), "bicycle": (1.8, 0.6),
    "pedestrian": (0.6, 0.6),
}


def _initial_speed(entity: str, moving: bool, ego_speed: float, rng: random.Random) -> float:
    if not moving:
        return 0.0
    if entity in _CROSSING_SPEED:
        return round(_CROSSING_SPEED[entity] * rng.uniform(0.8, 1.2), 1)
    return round(ego_speed * rng.uniform(0.7, 1.05), 1)  # other vehicles near ego speed


def _model(entity: str, occurrence: int, attempt_seed: int) -> str:
    models = MODELS_BY_ENTITY[entity]
    return models[(occurrence + attempt_seed) % len(models)]


def _build_config(index: int, template_index: int, template: Dict[str, Any],
                  rng: random.Random, attempt_seed: int,
                  map_name: Optional[str] = None) -> Dict[str, Any]:
    if map_name is None:
        map_name = slot_map(index)
    stype = template["scenario_type"]
    occurrence = index // len(TEMPLATES)
    # A stationary pedestrian/cyclist waiting at the roadside must remain
    # large enough in all eight images.  Urban crossing roots therefore use
    # a lower ego-speed range than vehicle-following/cut-in roots; otherwise
    # the ego can cover 80+ m during history and the target cannot be visible
    # both at -3.5 s and T0.
    if any(entity in {"pedestrian", "bicycle"}
           for entity in template["critical_entities"]):
        speed_ranges = ((4.5, 5.5), (5.0, 6.0), (6.0, 7.0))
    else:
        speed_ranges = ((8.0, 13.0), (13.0, 19.0), (19.0, 25.0))
    speed_low, speed_high = speed_ranges[occurrence % len(speed_ranges)]
    ego_speed = round(rng.uniform(speed_low, speed_high), 1)
    scenario_seed = rng.randint(0, 2 ** 31 - 1)
    severity = "high" if occurrence % 2 == 0 else "moderate"
    ttc_range = (1.0, 1.75) if severity == "high" else (1.75, 2.75)
    target_ttc = round(rng.uniform(*ttc_range), 2)
    critical_entities = template["critical_entities"]
    critical_entity = critical_entities[occurrence % len(critical_entities)]

    crossing = critical_entity in {"pedestrian", "bicycle"}
    target_length, target_width = _BODY_EXTENTS[critical_entity]
    ego_length, ego_width = _BODY_EXTENTS["ego_vehicle"]
    half_length = (ego_length + target_length) / 2.0
    if crossing:
        critical_speed = 0.0
        relative_speed = ego_speed
        camera_forward = float(CARLA_CAMERA["offset_m"][0])
        min_camera_depth = abs(float(template["critical_lateral"])) / math.tan(
            math.radians(CROSSING_VISIBILITY_HALF_FOV_DEG))
        min_t0_longitudinal = camera_forward + min_camera_depth
        # Round TTC upward so the serialized value still guarantees the
        # visibility floor when t0_longitudinal is reconstructed from it.
        target_ttc = math.ceil(max(
            target_ttc, min_t0_longitudinal / ego_speed) * 100.0) / 100.0
        t0_longitudinal = ego_speed * target_ttc
    else:
        critical_speed = round(ego_speed * rng.uniform(0.88, 1.0), 1)
        relative_speed = max(0.0, ego_speed - critical_speed)
        if template["critical_role"] == "lead_vehicle":
            # Place the lead by body-edge gap, not centre distance.  The gap
            # closes near target_ttc after the declared sudden deceleration.
            desired_edge = (relative_speed * target_ttc
                            + 0.5 * abs(float(template["critical_accel"]))
                            * target_ttc ** 2)
        else:
            # A cut-in must become spatially interactive after it reaches the
            # ego lane even when the two actors have nearly equal speed.
            desired_edge = 2.0 + relative_speed * target_ttc
        t0_longitudinal = half_length + desired_edge
    critical_dist = t0_longitudinal + relative_speed * 3.5
    gap_x = max(0.0, t0_longitudinal - half_length)
    gap_y = max(0.0, abs(float(template["critical_lateral"]))
                - (ego_width + target_width) / 2.0)
    t0_edge_gap = (gap_x ** 2 + gap_y ** 2) ** 0.5

    actor_defs: Dict[str, Any] = {
        "ego": {
            "canonical_id": "ego", "entity_type": "ego_vehicle", "default_role": "ego_agent",
            "model": rng.choice(EGO_MODELS), "initial_speed": ego_speed,
            "distance_ahead": 0.0, "lateral_offset": 0.0,
        }
    }
    if template["critical_role"] == "lead_vehicle":
        context_actor = _actor("npc_2", "vehicle", "adjacent_vehicle",
                               25.0, 3.5, True)
    else:
        context_actor = _actor(
            "npc_2", "vehicle", "lead_vehicle",
            35.0 + (ego_speed - ego_speed * 0.95) * 3.5,
            0.0, True, accel=-1.0)
    # No actor is spawned behind the ego: the model observes CAM_FRONT only,
    # so anything behind can never enter the observed root, and a reference
    # outcome caused by it would be unobservable. The former rear vehicle
    # npc_3 rear-ended a braking ego inside the horizon in 219/400 scenes of
    # the first CARLA-400 build (repair audit 2026-09-12).
    dynamic_cast = [
        _actor("npc_1", critical_entity, template["critical_role"],
               critical_dist, template["critical_lateral"], True,
               accel=template["critical_accel"], lat_v=template["critical_lat"],
               primary=True),
        context_actor,
    ]
    env_behavior: Dict[str, Dict[str, float]] = {}
    for a in dynamic_cast + _street_furniture(ego_speed, occurrence):
        aid = a["id"]
        initial_speed = (critical_speed if aid == "npc_1"
                         else _initial_speed(a["entity_type"], a["moving"], ego_speed, rng))
        actor_defs[aid] = {
            "canonical_id": aid,
            "entity_type": a["entity_type"],
            "default_role": a["role"],
            "model": _model(a["entity_type"], occurrence + template_index, attempt_seed),
            "initial_speed": initial_speed,
            "distance_ahead": a["dist"],
            "lateral_offset": a["lat"],
        }
        if a.get("primary"):
            actor_defs[aid]["collision_role"] = "impacted_actor"
        env_behavior[aid] = {"accel": a["behavior"]["accel"], "lat": a["behavior"]["lat"]}

    branches = {
        name: {"is_factual": is_f, "intervention_type": it, "target_actor_id": "ego", "trigger_depth": 1}
        for (name, is_f, it) in template["branches"]
    }

    return {
        "scenario_id": f"{template['template_id']}_{index:04d}",
        "scenario_index": index,
        "slot_id": index + 1,
        "template_id": template["template_id"],
        "attempt_seed": attempt_seed,
        "output_subdir": f"scenario_{index + 1}",
        "scenario_type": stype,
        "map": map_name,
        "weather": {"preset": WEATHERS[(occurrence + 2 * template_index) % len(WEATHERS)]},
        "seed": scenario_seed,
        "actor_defs": actor_defs,
        "env_behavior": env_behavior,
        "environment_event": {
            "event_type": template["env_event_type"],
            # Applied immediately after the T0 state is recorded; its first
            # scored consequence is the depth-1 state at T=0.5 s.
            "trigger_time_sec": 3.5,
            "params": {"intensity": abs(float(template["critical_accel"] or template["critical_lat"]))},
        },
        "critical_actor_id": "npc_1",
        "l4_query": {
            "intervention_type": template["env_event_type"],
            "target_actor_id": "npc_1",
            "trigger_depth": 1,
            "ego_policy": "do_ego_no_action",
        },
        "intervention_time_sec": 3.5,
        "sample_interval_sec": 0.5,
        "d_max": 6,
        "total_duration_sec": 8.0,
        "lateral_escape": {
            # Declared before the rollout from the template's route offset; the
            # published root decides the final side and a scene that disagrees
            # is rejected (protocol.lateral_action_conflict).
            "action": next((a for _n, _f, a in template["branches"]
                            if a in ("do_ego_swerve_left", "do_ego_swerve_right")), None),
            "basis": "template_lateral_offset",
            "critical_actor_id": "npc_1",
            "critical_lateral_m": float(template["critical_lateral"]),
            "near_side_threshold_m": LATERAL_NEAR_SIDE_M,
        },
        "severity_stratum": severity,
        "target_ttc_sec": target_ttc,
        "expected_t0_critical_gap_m": round(t0_edge_gap, 2),
        "expected_t0_critical_longitudinal_m": round(t0_longitudinal, 2),
        "camera": {key: (list(value) if isinstance(value, list) else value)
                   for key, value in CARLA_CAMERA.items()},
        "branches": branches,
    }


def generate_configs(n: int, seed: int = 0) -> List[Dict[str, Any]]:
    """Generate n scenario configs, stratified round-robin over TEMPLATES, seeded."""
    rng = random.Random(seed)
    maps = plan_maps(max(PLANNED_DATASET_SIZE, n))
    return [_build_config(i, i % len(TEMPLATES), TEMPLATES[i % len(TEMPLATES)],
                          rng, seed, map_name=maps[i]) for i in range(n)]


def main():
    parser = argparse.ArgumentParser(description="Generate CORIVA scenario configs")
    parser.add_argument("--n", type=int, required=True, help="number of configs")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out-dir", default="scenarios/generated")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    configs = generate_configs(args.n, seed=args.seed)
    for c in configs:
        path = os.path.join(args.out_dir, c["output_subdir"] + ".yaml")
        with open(path, "w", encoding="utf-8") as f:
            yaml.safe_dump(c, f, sort_keys=False, default_flow_style=False, allow_unicode=True)
    print(f"Wrote {len(configs)} configs to {args.out_dir} "
          f"(avg {sum(len(c['actor_defs']) for c in configs)/len(configs):.1f} actors/scenario)")


if __name__ == "__main__":
    main()
