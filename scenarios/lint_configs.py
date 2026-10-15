"""
Config-set linter (TASK.md P2.3): catch malformed scenario configs at
generation time, before they burn paid CARLA GPU-hours.

Per config: exactly one factual branch; every intervention_type in the twin's
EGO_PROFILES (the canonical dynamics vocabulary shared by both splits); no two
actors spawned into the same spot. Per set: unique scenario_id / output_subdir
/ seed, template stratification, map diversity.

CLI:
    python3 -m scenarios.lint_configs --scenarios-dir scenarios/generated
"""

from __future__ import annotations

import argparse
import math
import sys
from collections import Counter
from typing import Any, Dict, List

import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from simulation.kinematic_twin import EGO_PROFILES  # noqa: E402
from scenarios.generate_configs import (  # noqa: E402
    CARLA_CAMERA, CROSSING_VISIBILITY_HALF_FOV_DEG, MAPS, TEMPLATES, WEATHERS)

# Two actors closer than this at spawn is a collision at T=0, not a scenario.
MIN_SPAWN_GAP_M = 2.0
_SPAWN_EXTENTS = {
    "ego_vehicle": (4.5, 2.0), "vehicle": (4.5, 2.0),
    "parked_vehicle": (4.5, 2.0), "truck": (7.0, 2.5),
    "bus": (12.0, 2.5), "motorcycle": (2.2, 0.8),
    "bicycle": (1.8, 0.6), "pedestrian": (0.6, 0.6),
    "guardrail": (1.5, 0.5), "traffic_sign": (0.6, 0.6),
    "traffic_cone": (0.4, 0.4),
}


def _spawn_extent(spec: Dict[str, Any]) -> tuple[float, float]:
    length, width = _SPAWN_EXTENTS.get(str(spec.get("entity_type")), (1.5, 0.8))
    return float(spec.get("length", length)), float(spec.get("width", width))


def lint_config(config: Dict[str, Any]) -> List[str]:
    """Return a list of human-readable errors; [] = clean."""
    errors: List[str] = []
    sid = config.get("scenario_id", "<missing id>")

    branches = config.get("branches", {})
    n_factual = sum(1 for b in branches.values() if b.get("is_factual"))
    if n_factual != 1:
        errors.append(f"{sid}: expected exactly 1 factual branch, got {n_factual}")

    for name, b in branches.items():
        itype = b.get("intervention_type")
        if itype not in EGO_PROFILES:
            errors.append(f"{sid}: branch {name} has unknown intervention_type {itype!r}")

    actor_defs = config.get("actor_defs", {})
    if "ego" not in actor_defs:
        errors.append(f"{sid}: no ego in actor_defs")
    positions = {aid: (float(s.get("distance_ahead", 0.0)), float(s.get("lateral_offset", 0.0)))
                 for aid, s in actor_defs.items()}
    aids = sorted(positions)
    for i, a in enumerate(aids):
        for b in aids[i + 1:]:
            if math.dist(positions[a], positions[b]) < MIN_SPAWN_GAP_M:
                errors.append(f"{sid}: spawn collision between {a} and {b} "
                              f"at {positions[a]} / {positions[b]}")
                continue
            length_a, width_a = _spawn_extent(actor_defs[a])
            length_b, width_b = _spawn_extent(actor_defs[b])
            longitudinal_gap = (abs(positions[a][0] - positions[b][0])
                                - (length_a + length_b) / 2.0)
            lateral_gap = (abs(positions[a][1] - positions[b][1])
                           - (width_a + width_b) / 2.0)
            if longitudinal_gap < 0.0 and lateral_gap < 0.0:
                errors.append(
                    f"{sid}: spawn body overlap between {a} and {b}; "
                    f"edge gaps=({longitudinal_gap:.2f},{lateral_gap:.2f})")

    for key in ("map", "seed", "intervention_time_sec", "total_duration_sec",
                "environment_event", "output_subdir"):
        if key not in config:
            errors.append(f"{sid}: missing required key {key!r}")

    # Protocol-1.0 generated slots have a stricter release contract than the
    # two legacy hand-written fixtures that share this linter.
    if config.get("template_id"):
        template_ids = {str(template["template_id"]) for template in TEMPLATES}
        if str(config.get("template_id")) not in template_ids:
            errors.append(f"{sid}: unknown template_id {config.get('template_id')!r}")
        if len(actor_defs) != 10:
            errors.append(f"{sid}: expected exactly 10 simulated actors")
        behind = [aid for aid, spec in actor_defs.items()
                  if float(spec.get("distance_ahead", 0.0)) < 0.0]
        if behind:
            errors.append(f"{sid}: actors spawned behind the ego (outside CAM_FRONT): {behind}")
        if len(branches) != 3:
            errors.append(f"{sid}: expected exactly 3 Layer-3 branches")
        if config.get("camera") != CARLA_CAMERA:
            errors.append(f"{sid}: camera calibration does not match protocol CAM_FRONT")
        if float(config.get("sample_interval_sec", -1.0)) != 0.5:
            errors.append(f"{sid}: sample_interval_sec must be 0.5")
        if int(config.get("d_max", -1)) != 6:
            errors.append(f"{sid}: d_max must be 6 for the primary track")
        critical = str(config.get("critical_actor_id", ""))
        if critical not in actor_defs:
            errors.append(f"{sid}: critical_actor_id is missing from actor_defs")
        query = config.get("l4_query") or {}
        if str(query.get("target_actor_id", "")) != critical:
            errors.append(f"{sid}: L4 target does not match critical_actor_id")
        if query.get("intervention_type") != (config.get("environment_event") or {}).get("event_type"):
            errors.append(f"{sid}: L4 query and environment event disagree")
        critical_spec = actor_defs.get(critical, {})
        if critical_spec.get("entity_type") in {"pedestrian", "bicycle"}:
            longitudinal = float(config.get("expected_t0_critical_longitudinal_m", 0.0))
            camera_depth = longitudinal - float(CARLA_CAMERA["offset_m"][0])
            lateral = abs(float(critical_spec.get("lateral_offset", 0.0)))
            angle = (90.0 if camera_depth <= 0.0
                     else math.degrees(math.atan2(lateral, camera_depth)))
            if angle > CROSSING_VISIBILITY_HALF_FOV_DEG + 1e-9:
                errors.append(
                    f"{sid}: critical crossing actor T0 camera angle "
                    f"{angle:.1f} exceeds {CROSSING_VISIBILITY_HALF_FOV_DEG:.1f} deg")
    return errors


def lint_config_set(configs: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Lint every config plus set-level uniqueness and stratification."""
    errors: List[str] = []
    for c in configs:
        errors.extend(lint_config(c))

    for key in ("scenario_id", "output_subdir", "seed"):
        dupes = [v for v, n in Counter(c.get(key) for c in configs).items() if n > 1]
        if dupes:
            errors.append(f"duplicate {key}: {dupes[:5]}")

    by_template = Counter(c.get("template_id") for c in configs)
    by_map = Counter(c.get("map") for c in configs)
    by_weather = Counter((c.get("weather") or {}).get("preset") for c in configs)
    if len(configs) == 400 and all(c.get("template_id") for c in configs):
        expected_templates = {str(template["template_id"]): 50 for template in TEMPLATES}
        if dict(by_template) != expected_templates:
            errors.append(f"template quota mismatch: {dict(by_template)}")
        if by_map != Counter({name: 100 for name in MAPS}):
            errors.append(f"map quota mismatch: {dict(by_map)}")
        if (set(by_weather) != set(WEATHERS) or min(by_weather.values()) < 66
                or max(by_weather.values()) > 67):
            errors.append(f"weather quota mismatch: {dict(by_weather)}")

    return {
        "n": len(configs),
        "errors": errors,
        "by_scenario_type": dict(Counter(c.get("scenario_type") for c in configs)),
        "by_template_id": dict(by_template),
        "by_map": dict(by_map),
        "by_weather": dict(by_weather),
    }


def main() -> int:
    from simulation.scenario_config import load_scenario_configs
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--scenarios-dir", default="scenarios/generated")
    args = parser.parse_args()
    report = lint_config_set(load_scenario_configs(args.scenarios_dir))
    print(f"n={report['n']} errors={len(report['errors'])}")
    print("by_scenario_type:", report["by_scenario_type"])
    print("by_template_id:", report["by_template_id"])
    print("by_map:", report["by_map"])
    print("by_weather:", report["by_weather"])
    for e in report["errors"][:20]:
        print("ERROR:", e)
    return 0 if not report["errors"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
