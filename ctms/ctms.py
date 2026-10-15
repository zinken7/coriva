from __future__ import annotations

import copy
import json
import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

SCORER_VERSION = "ctms-scorer-1.3"
RESCORABLE_SCORER_VERSIONS = frozenset({"ctms-scorer-1.2", SCORER_VERSION})
# Option B (docs/l4_pair_protocol.md): an L4 query is answered by a matched
# control/treated pair. The synthetic intervention label of the control chain
# and the reference marker of a migrated dataset.
CONTROL_INTERVENTION_TYPE = "target_baseline"
L4_PAIR_PROTOCOL = "matched_control_treated"
L4_GATE_KEYS = ("schema", "semantic", "scene", "kinematic", "metadata", "intervention")
# References declaring this input contract are scored with the protocol gates
# (metadata gate, timeline grid). Must equal simulation.protocol.PROTOCOL_VERSION;
# tests/test_scorer_1_0.py pins the two together.
GATED_PROTOCOL_VERSION = "coriva-protocol-1.0"
import yaml
from jsonschema import Draft202012Validator
from scipy.optimize import linear_sum_assignment


def reference_protocol_version(reference: Any) -> Optional[str]:
    """Protocol version a reference declares. A scenario ground truth carries it
    in its ``protocol`` block (both builders); an assembled dataset or manifest
    carries it at top level. None when the reference declares nothing."""
    if not isinstance(reference, dict):
        return None
    block = reference.get("protocol")
    if isinstance(block, dict) and block.get("protocol_version"):
        return str(block["protocol_version"])
    version = reference.get("protocol_version")
    return str(version) if version else None


def reference_future_steps(reference: Any) -> Optional[int]:
    """Number of scored future nodes the reference declares for its track:
    ``protocol.future_steps`` (written by protocol_manifest), else derived from
    ``protocol.track``; None when undeclared."""
    if not isinstance(reference, dict):
        return None
    block = reference.get("protocol")
    if not isinstance(block, dict):
        return None
    steps = block.get("future_steps")
    if isinstance(steps, int) and steps > 0:
        return steps
    track = block.get("track")
    if track:
        try:
            from simulation.protocol import track_steps
            return int(track_steps(track))
        except Exception:
            return None
    return None


def apply_reference_gate_policy(config: Dict[str, Any], reference: Any,
                                horizon: Optional[int] = None) -> Dict[str, Any]:
    """Return a copy of ``config`` with the reference-dependent settings:

    * ``layer3.use_metadata_gate`` and ``timing.require_protocol_grid`` are on
      iff the reference declares GATED_PROTOCOL_VERSION, off for a reference
      that declares another version (variable dt), untouched when the
      reference declares nothing;
    * ``bounds.D_max`` follows the reference's declared track (6 primary,
      12 challenge_6s) so a challenge reference is not cut at node 7; an
      explicit ``horizon`` wins (prefix scoring truncates both sides to a
      depth and scores at that D_max).

    Used by run_benchmark (CLI and score_sample), dataset QC and any direct
    caller, so every scoring path agrees."""
    version = reference_protocol_version(reference)
    cfg = copy.deepcopy(config)
    if version is not None:
        gated = version == GATED_PROTOCOL_VERSION
        cfg.setdefault("layer3", {})["use_metadata_gate"] = gated
        cfg.setdefault("timing", {})["require_protocol_grid"] = gated
    steps = reference_future_steps(reference)
    if horizon is not None:
        cfg.setdefault("bounds", {})["D_max"] = int(horizon)
    elif steps is not None:
        cfg.setdefault("bounds", {})["D_max"] = steps
    return cfg


FORBIDDEN_ONTOLOGY_RUNTIME_KEYS = {
    "bounds",
    "timing",
    "layer3_validity",
    "layer4_validity",
    "overall",
    "matching",
    "kinematic_bounds",
}


def load_ctms_environment(
    ontology_path: str | Path = "ontology.yaml",
    config_path: str | Path = "ctms_config.yaml",
    branch_schema_path: str | Path = "schema_branch.json",
    counterfactual_schema_path: str | Path = "schema_counterfactual.json",
) -> Dict[str, Any]:
    ontology_path = Path(ontology_path)
    config_path = Path(config_path)
    branch_schema_path = Path(branch_schema_path)
    counterfactual_schema_path = Path(counterfactual_schema_path)

    with ontology_path.open("r", encoding="utf-8") as f:
        ontology = yaml.safe_load(f)
    with config_path.open("r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    with branch_schema_path.open("r", encoding="utf-8") as f:
        branch_schema = json.load(f)
    with counterfactual_schema_path.open("r", encoding="utf-8") as f:
        counterfactual_schema = json.load(f)

    forbidden = sorted(set(ontology.keys()) & FORBIDDEN_ONTOLOGY_RUNTIME_KEYS)
    if forbidden:
        raise ValueError(
            f"ontology.yaml must not contain runtime config keys: {forbidden}. Move them into ctms_config.yaml."
        )

    return {
        "ontology": ontology,
        "config": config,
        "branch_schema": branch_schema,
        "counterfactual_schema": counterfactual_schema,
    }


def check_schema_validity(obj: Dict[str, Any], json_schema: Dict[str, Any]) -> Tuple[bool, List[str]]:
    """Validate obj against a JSON Schema.

    CONTRACT: In the scoring pipeline (score_layer3, score_layer4), this
    function MUST NOT be called on raw input branches. All raw branches
    must pass through _pre_normalize_branch(branch, ontology) first, which
    normalizes alias labels to canonical forms and converts full-snapshot
    actors to actors_update for non-observed_state nodes. The schema enums
    enforce a strict canonical-only vocabulary; un-normalized aliases will
    be rejected.
    """
    validator = Draft202012Validator(json_schema)
    errors = sorted(validator.iter_errors(obj), key=lambda e: list(e.path))
    messages = []
    for err in errors:
        loc = "/".join(str(x) for x in err.path)
        messages.append(f"{loc}: {err.message}" if loc else err.message)
    return len(messages) == 0, messages


def _safe_float_or_none(value: Any) -> Optional[float]:
    """Convert value to float, returning None on failure. Never raises."""
    if value is None:
        return None
    try:
        return float(value)
    except (ValueError, TypeError):
        return None


_NUMERIC_NODE_FIELDS = ("depth", "timestamp")
_NUMERIC_ACTOR_FIELDS = ("x", "y", "vx", "vy", "speed")


def malformed_unit_errors(unit: Any) -> List[str]:
    """Shape errors of ONE raw predicted unit (an L3 branch or the
    `counterfactual_chain` of an L4 answer) that the gates cannot evaluate.
    Such a unit (a `null` chain, a `null` node, a `null` actor, a textual
    timestamp) is a schema failure of that unit alone: the other units of
    the sample, and the other task, are scored normally, and nothing raises
    out of canonicalisation."""
    errors: List[str] = []
    if not isinstance(unit, dict):
        return [f"schema: unit is {type(unit).__name__}, expected an object"]
    chain = unit.get("chain")
    if chain is None:
        chain = []
    if not isinstance(chain, list):
        return [f"schema: chain is {type(chain).__name__}, expected an array of nodes"]
    if "intervention" in unit and unit["intervention"] is not None and \
            not isinstance(unit["intervention"], dict):
        errors.append("schema: intervention is not an object")
    for idx, node in enumerate(chain):
        if not isinstance(node, dict):
            errors.append(f"schema: node[{idx}] is {type(node).__name__}, expected an object")
            continue
        for field in _NUMERIC_NODE_FIELDS:
            value = node.get(field)
            if value is not None and (isinstance(value, bool) or not isinstance(value, (int, float))):
                errors.append(f"schema: node[{idx}].{field} is not a number")
        for actor_field in ("actors", "actors_update"):
            actors = node.get(actor_field)
            if actors is None:
                continue
            if not isinstance(actors, list):
                errors.append(f"schema: node[{idx}].{actor_field} is not an array")
                continue
            for a_idx, actor in enumerate(actors):
                if not isinstance(actor, dict):
                    errors.append(f"schema: node[{idx}].{actor_field}[{a_idx}] is not an object")
                    continue
                for field in _NUMERIC_ACTOR_FIELDS:
                    value = actor.get(field)
                    if value is not None and (isinstance(value, bool) or not isinstance(value, (int, float))):
                        errors.append(f"schema: node[{idx}].{actor_field}[{a_idx}].{field} is not a number")
    return errors


def _pre_normalize_branch(branch: Dict[str, Any], ontology: Dict[str, Any]) -> Dict[str, Any]:
    """Pre-schema normalization adapter for raw input branches.

    Two responsibilities, applied BEFORE schema validation:

    1. **Label normalization** â€” Maps alias entity_type labels (car, ego,
       walker, person, bike, railing) to their canonical forms using
       ontology.synonyms.entity_type. This allows the schema to enforce
       a strict closed-set enum of canonical labels only.

    2. **Structure normalization** â€” Renames 'actors' â†’ 'actors_update'
       for any node whose event_type â‰  'observed_state', enforcing the
       strict schema contract:
           observed_state  â†’ must use 'actors'   (root world snapshot)
           all other nodes â†’ must use 'actors_update' (delta-patch rollout)

    Returns a deep copy; never mutates the input.
    """
    b = copy.deepcopy(branch)
    synonyms = ontology.get("synonyms", {}).get("entity_type", {})

    # Strip any leading observed_state node(s) BEFORE schema validation.
    # The root is model INPUT (never scored); a prediction that includes it
    # must be treated identically to one that does not, and the root node's
    # shape (depth 0, 'actors') is not part of the prediction schema.
    ev_syn = ontology.get("synonyms", {}).get("event_type", {})
    chain = b.get("chain") or []
    if not isinstance(chain, list):
        chain = []
    while chain and isinstance(chain[0], dict) and \
            ev_syn.get(chain[0].get("event_type"), chain[0].get("event_type")) == "observed_state":
        chain = chain[1:]
    b["chain"] = chain

    for node in b.get("chain", []):
        if not isinstance(node, dict):
            continue
        # --- Label normalization ---
        for actor_field in ("actors", "actors_update"):
            actors = node.get(actor_field, []) or []
            for actor in actors:
                raw_et = actor.get("entity_type")
                if raw_et and raw_et in synonyms:
                    actor["entity_type"] = synonyms[raw_et]

        # --- Structure normalization ---
        event_type = node.get("event_type")
        if event_type != "observed_state" and "actors" in node and "actors_update" not in node:
            node["actors_update"] = node.pop("actors")

    return b


# Fallback for kinematic_resolution.speed_conflict_tolerance when absent from
# config. MUST equal the shipped ctms_config.yaml value. Derivation: endpoint
# vs displacement-average speed differ by up to max|a|·dt/2 = 10 × 0.5 / 2
# = 2.5 m/s under kinematic_bounds; the existing 4.0 m/s tolerance retains
# headroom for rounding.
DEFAULT_SPEED_CONFLICT_TOLERANCE = 4.0
# Numerical slack only: decimal speeds at the inclusive acceleration bounds
# can yield values such as -10.000000000000002 after floating-point subtraction.
ACCELERATION_BOUND_EPSILON = 1e-9


def canonicalize_branch(branch: Dict[str, Any], ontology: Dict[str, Any], config: Dict[str, Any]) -> Dict[str, Any]:
    b = copy.deepcopy(branch)
    canon_cfg = config.get("canonicalization", {})
    kin_cfg = config.get("kinematic_resolution", {})
    precision = config.get("numeric_precision", {})
    synonyms = ontology.get("synonyms", {})
    bbox_format = config.get("bbox_format", "cx_cy_w_h")

    # Strip metadata fields before canonical comparison
    strip_fields = canon_cfg.get("strip_metadata_fields", [])
    for field in strip_fields:
        b.pop(field, None)

    def normalize_enum(value: Any, mapping: Dict[str, str]) -> Any:
        if isinstance(value, str):
            return mapping.get(value, value)
        return value

    def round_if_needed(field: str, value: Any) -> Any:
        if isinstance(value, (int, float)) and field in precision:
            return round(float(value), int(precision[field]))
        return value

    chain = b.get("chain", [])
    # The observed_state root is model INPUT and is never scored.
    # Strip any leading observed_state node(s) so gates and scoring are
    # format-neutral (a chain with or without the given root scores the same)
    # and echoing the input earns nothing (outline §V.2).
    ev_syn = synonyms.get("event_type", {})
    while chain and isinstance(chain[0], dict) and \
            normalize_enum(chain[0].get("event_type"), ev_syn) == "observed_state":
        chain = chain[1:]
    b["chain"] = chain
    for node_idx, node in enumerate(chain, start=1):
        # Strip metadata from nodes too
        for field in strip_fields:
            node.pop(field, None)

        node["event_type"] = normalize_enum(node.get("event_type"), synonyms.get("event_type", {}))
        if "action_type" in node:
            node["action_type"] = normalize_enum(node.get("action_type"), synonyms.get("action_type", {}))
        if canon_cfg.get("derive_depth_from_chain_order", True):
            node["depth"] = node_idx

        for actor_field in ("actors", "actors_update"):
            actors = node.get(actor_field, []) or []
            for actor in actors:
                actor["entity_type"] = normalize_enum(actor.get("entity_type"), synonyms.get("entity_type", {}))

                # Crash-safe numeric canonicalization: never raise on malformed values
                for k in list(actor.keys()):
                    if k in precision:
                        parsed = _safe_float_or_none(actor[k])
                        if parsed is not None:
                            actor[k] = round(parsed, int(precision[k]))
                        elif actor[k] is not None:
                            actor.setdefault("_errors", []).append(f"invalid_numeric_type:{k}")
                            actor[k] = None

                bbox = actor.get("bbox")
                if isinstance(bbox, list) and len(bbox) == 4:
                    parsed_bbox = [_safe_float_or_none(x) for x in bbox]
                    if all(v is not None for v in parsed_bbox):
                        actor["bbox"] = [round(v, int(precision.get("bbox", 2))) for v in parsed_bbox]
                        if bbox_format == "cx_cy_w_h" and config.get("bbox_validation", {}).get("require_non_negative_size", True):
                            if actor["bbox"][2] < 0 or actor["bbox"][3] < 0:
                                actor.setdefault("_errors", []).append("invalid_bbox_size")
                    else:
                        actor.setdefault("_errors", []).append("invalid_numeric_type:bbox")
                        actor["bbox"] = None

                vx = _safe_float_or_none(actor.get("vx"))
                vy = _safe_float_or_none(actor.get("vy"))
                raw_speed = _safe_float_or_none(actor.get("speed"))
                # Propagate parse failures back to actor dict
                if actor.get("vx") is not None and vx is None:
                    actor.setdefault("_errors", []).append("invalid_numeric_type:vx")
                    actor["vx"] = None
                if actor.get("vy") is not None and vy is None:
                    actor.setdefault("_errors", []).append("invalid_numeric_type:vy")
                    actor["vy"] = None
                if actor.get("speed") is not None and raw_speed is None:
                    actor.setdefault("_errors", []).append("invalid_numeric_type:speed")
                    actor["speed"] = None

                if kin_cfg.get("derive_speed_from_velocity", True) and vx is not None and vy is not None:
                    derived_speed = math.sqrt(vx ** 2 + vy ** 2)
                    if raw_speed is not None and kin_cfg.get("reject_on_internal_physics_conflict", True):
                        tol = float(kin_cfg.get("speed_conflict_tolerance", DEFAULT_SPEED_CONFLICT_TOLERANCE))
                        if abs(raw_speed - derived_speed) > tol:
                            actor.setdefault("_errors", []).append("internal_physics_conflict")
                    actor["speed"] = round_if_needed("speed", derived_speed)

            # Sort actor arrays by deep canonical key for order-invariant dedup
            if node.get(actor_field):
                node[actor_field] = sorted(
                    node[actor_field],
                    key=lambda a: json.dumps(a, sort_keys=True, separators=(",", ":"), ensure_ascii=False),
                )
    return b


def check_semantic_validity(branch: Dict[str, Any], ontology: Dict[str, Any], root_universe: Optional[Dict[str, str]] = None, check_role_completeness: bool = True) -> Tuple[bool, List[str]]:
    """Semantic ontology gate.

    `check_role_completeness` toggles the required_roles completeness check
    (e.g. collision nodes must carry an impacted_actor). Layer 3 keeps it ON
    (default). Layer 4 turns it OFF so a forced-intervention outcome
    (collision / off-road) is not rejected merely for an absent role — per
    outline Section 6, which restricts Layer 4 to schema + kinematic gates.
    Structural validity (closed-world hallucination, entity_type mutation,
    duplicate IDs, terminal constraints, timestamp monotonicity) always applies.
    """
    errors: List[str] = []
    chain = branch.get("chain", [])
    event_constraints = ontology.get("event_constraints", {})
    terminal_constraints = ontology.get("terminal_constraints", {})
    required_roles = ontology.get("required_roles", {})

    prev_ts: Optional[float] = None
    # Immutable Root Binding: ID -> entity_type.
    # If a shared root universe is supplied, bind to it and validate EVERY
    # node (including node 0) against it; otherwise derive it from node 0.
    external_universe = root_universe is not None
    universe: Dict[str, str] = dict(root_universe) if external_universe else {}

    for idx, node in enumerate(chain):
        event_type = node.get("event_type")
        ec = event_constraints.get(event_type, {})
        if ec.get("requires_action_type"):
            action_type = node.get("action_type")
            allowed = set(ec.get("allowed_action_types", []))
            if action_type is None:
                errors.append(f"node[{idx}]: missing action_type for event_type={event_type}")
            elif action_type not in allowed:
                errors.append(f"node[{idx}]: invalid action_type={action_type} for event_type={event_type}")

        tc = terminal_constraints.get(event_type)
        if tc:
            field = tc.get("required_field", "terminal")
            expected = tc.get("enforced_value", True)
            if node.get(field) != expected:
                errors.append(f"node[{idx}]: terminal constraint violated for event_type={event_type}")
            if tc.get("must_be_last_node", False) and idx != len(chain) - 1:
                errors.append(f"node[{idx}]: terminal event_type={event_type} is not the final node")

        if check_role_completeness:
            role_spec = required_roles.get(event_type, {})
            actors = node.get("actors", []) or node.get("actors_update", []) or []
            role_map = {a.get("role_type"): a.get("entity_type") for a in actors if a.get("role_type") is not None}
            for role_name, role_cfg in role_spec.items():
                allowed_types = set(role_cfg.get("allowed_entity_types", []))
                if role_name not in role_map:
                    errors.append(f"node[{idx}]: missing required role={role_name}")
                elif role_map[role_name] not in allowed_types:
                    errors.append(f"node[{idx}]: role={role_name} has invalid entity_type={role_map[role_name]}")

        # Root universe binding + duplicate ID + hallucination + mutation checks
        for actor_field in ("actors", "actors_update"):
            actor_list = node.get(actor_field, []) or []
            seen_ids_in_node: set = set()
            for actor_idx, actor in enumerate(actor_list):
                aid = actor.get("id")
                etype = actor.get("entity_type")

                if aid is not None:
                    aid_str = str(aid)

                    if not external_universe and idx == 0:
                        # No shared root universe: node 0 initializes it
                        universe[aid_str] = etype
                    else:
                        # Reject hallucinated new actors not in U_actors
                        if aid_str not in universe:
                            errors.append(f"node[{idx}].{actor_field}: hallucinated new actor id={aid_str!r} not in root U_actors")
                        # Reject entity_type mutation
                        elif universe[aid_str] != etype:
                            errors.append(
                                f"node[{idx}].{actor_field}: mutated entity_type for id={aid_str!r} "
                                f"(expected {universe[aid_str]!r}, got {etype!r})"
                            )

                    if aid_str in seen_ids_in_node:
                        errors.append(f"node[{idx}].{actor_field}: duplicate actor id={aid_str}")
                    seen_ids_in_node.add(aid_str)

                for e in actor.get("_errors", []):
                    errors.append(f"node[{idx}].{actor_field}[{actor_idx}]: {e}")

        # Timestamp monotonicity â€” time must always advance forward
        curr_ts = node.get("timestamp")
        if curr_ts is not None:
            if prev_ts is not None and float(curr_ts) <= float(prev_ts):
                errors.append(
                    f"node[{idx}]: timestamp={curr_ts} not strictly greater than previous={prev_ts} (causality violation)"
                )
            prev_ts = float(curr_ts)

    return len(errors) == 0, errors


def _actor_xy(actor: Optional[Dict[str, Any]]) -> Optional[Tuple[float, float]]:
    if not actor:
        return None
    x = _safe_float_or_none(actor.get("x"))
    y = _safe_float_or_none(actor.get("y"))
    return (x, y) if x is not None and y is not None else None


def _point_on_segment(point: Tuple[float, float], start: Tuple[float, float],
                      end: Tuple[float, float], eps: float = 1e-9) -> bool:
    px, py = point
    ax, ay = start
    bx, by = end
    cross = (px - ax) * (by - ay) - (py - ay) * (bx - ax)
    if abs(cross) > eps:
        return False
    return (min(ax, bx) - eps <= px <= max(ax, bx) + eps
            and min(ay, by) - eps <= py <= max(ay, by) + eps)


def _point_in_polygon(point: Tuple[float, float], polygon: List[Any]) -> bool:
    """Boundary-inclusive ray casting; no geometry dependency is required."""
    try:
        vertices = [(float(p[0]), float(p[1])) for p in polygon]
    except (TypeError, ValueError, IndexError):
        return False
    if len(vertices) < 3:
        return False
    inside = False
    px, py = point
    for i, start in enumerate(vertices):
        end = vertices[(i + 1) % len(vertices)]
        if _point_on_segment(point, start, end):
            return True
        ax, ay = start
        bx, by = end
        if (ay > py) != (by > py):
            x_cross = (bx - ax) * (py - ay) / (by - ay) + ax
            if px < x_cross:
                inside = not inside
    return inside


def _orientation(a: Tuple[float, float], b: Tuple[float, float],
                 c: Tuple[float, float], eps: float = 1e-9) -> int:
    value = (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])
    if abs(value) <= eps:
        return 0
    return 1 if value > 0 else -1


def _segments_intersect(a: Tuple[float, float], b: Tuple[float, float],
                        c: Tuple[float, float], d: Tuple[float, float]) -> bool:
    o1, o2 = _orientation(a, b, c), _orientation(a, b, d)
    o3, o4 = _orientation(c, d, a), _orientation(c, d, b)
    if o1 != o2 and o3 != o4:
        return True
    return ((o1 == 0 and _point_on_segment(c, a, b))
            or (o2 == 0 and _point_on_segment(d, a, b))
            or (o3 == 0 and _point_on_segment(a, c, d))
            or (o4 == 0 and _point_on_segment(b, c, d)))


def _route_centerlines(map_data: Optional[Dict[str, Any]]) -> List[List[Tuple[float, float]]]:
    """Return well-formed scenario-local route polylines."""
    raw_lines: List[Any] = []
    if map_data:
        if map_data.get("route_centerlines") is not None:
            raw_lines.extend(map_data.get("route_centerlines") or [])
        elif map_data.get("route_centerline") is not None:
            raw_lines.append(map_data.get("route_centerline"))
    lines: List[List[Tuple[float, float]]] = []
    for raw_line in raw_lines:
        try:
            line = [(float(point[0]), float(point[1])) for point in raw_line]
        except (TypeError, ValueError, IndexError):
            continue
        if len(line) >= 2:
            lines.append(line)
    return lines


def _project_to_polyline(
        point: Tuple[float, float],
        line: List[Tuple[float, float]]) -> Dict[str, Any]:
    """Project a point onto a finite route without relabelling overrun.

    Interior vertices belong to the supported polyline. Only a closest-point
    projection beyond the global first or final endpoint is classified as
    incomplete route coverage. This prevents longitudinal endpoint distance
    from being reported as lateral off-road distance.
    """
    px, py = point
    best: Optional[Dict[str, Any]] = None
    along_before = 0.0
    last_segment_index = len(line) - 2
    for segment_index, (start, end) in enumerate(zip(line, line[1:])):
        ax, ay = start
        bx, by = end
        dx, dy = bx - ax, by - ay
        length_sq = dx * dx + dy * dy
        if length_sq <= 1e-12:
            continue
        segment_length = math.sqrt(length_sq)
        raw_fraction = ((px - ax) * dx + (py - ay) * dy) / length_sq
        fraction = max(0.0, min(1.0, raw_fraction))
        qx, qy = ax + fraction * dx, ay + fraction * dy
        distance = math.hypot(px - qx, py - qy)
        cross = dx * (py - qy) - dy * (px - qx)
        status = "supported"
        if segment_index == 0 and raw_fraction < 0.0:
            status = "unsupported_before_start"
        elif segment_index == last_segment_index and raw_fraction > 1.0:
            status = "unsupported_after_end"
        candidate = {
            "distance": distance,
            "signed_distance": distance if cross >= 0.0 else -distance,
            "along_track": along_before + fraction * segment_length,
            "status": status,
            "segment_index": segment_index,
            "fraction": raw_fraction,
        }
        if (best is None
                or distance < float(best["distance"]) - 1e-9
                or (abs(distance - float(best["distance"])) <= 1e-9
                    and status == "supported"
                    and best["status"] != "supported")):
            best = candidate
        along_before += segment_length
    if best is None:
        return {
            "distance": float("inf"), "signed_distance": 0.0,
            "along_track": 0.0, "status": "invalid_route",
            "segment_index": None, "fraction": None,
        }
    return best


def _nearest_route_projection(
        point: Tuple[float, float],
        lines: List[List[Tuple[float, float]]]) -> Dict[str, Any]:
    candidates = [_project_to_polyline(point, line) for line in lines]
    supported = [item for item in candidates if item["status"] == "supported"]
    pool = supported or candidates
    return min(pool, key=lambda item: float(item["distance"]))


def _ego_scene_path(branch: Dict[str, Any], root: Optional[Dict[str, Any]],
                    max_steps: Optional[int] = None) -> List[Tuple[float, float]]:
    path: List[Tuple[float, float]] = []
    if root:
        ego = next((a for a in (root.get("actors") or [])
                    if a.get("role_type") == "ego_agent"), None)
        xy = _actor_xy(ego)
        if xy is not None:
            path.append(xy)
    states = _build_accumulated_states(branch, root=root)
    if max_steps is not None:
        states = states[:max_steps]
    for state in states:
        ego = next((a for a in state["world_snapshot"].values()
                    if a.get("role_type") == "ego_agent"), None)
        xy = _actor_xy(ego)
        if xy is not None and (not path or xy != path[-1]):
            path.append(xy)
    return path


def _path_heading(path: List[Tuple[float, float]], root: Optional[Dict[str, Any]]) -> Optional[Tuple[float, float]]:
    # Prefer observed displacement: vx/vy are not guaranteed to be global
    # vector components. Fall back to root velocity for a stationary/one-node
    # path.
    if path:
        ox, oy = path[0]
        for x, y in path[1:]:
            dx, dy = x - ox, y - oy
            norm = math.hypot(dx, dy)
            if norm > 1e-6:
                return dx / norm, dy / norm
    if root:
        ego = next((a for a in (root.get("actors") or [])
                    if a.get("role_type") == "ego_agent"), None)
        if ego:
            vx = _safe_float_or_none(ego.get("vx"))
            vy = _safe_float_or_none(ego.get("vy"))
            if vx is not None and vy is not None and math.hypot(vx, vy) > 1e-6:
                norm = math.hypot(vx, vy)
                return vx / norm, vy / norm
    return None


def check_scene_validity(branch: Dict[str, Any], config: Dict[str, Any],
                         map_data: Optional[Dict[str, Any]] = None,
                         root: Optional[Dict[str, Any]] = None) -> Tuple[bool, List[str]]:
    errors: List[str] = []
    scene_cfg = config.get("scene_gate", {})
    d_max = int(config.get("bounds", {}).get("D_max", len(branch.get("chain", []))))
    path = _ego_scene_path(branch, root, max_steps=d_max)
    heading = _path_heading(path, root)

    if scene_cfg.get("check_offroad_outside_drivable_area", False) and path:
        polygons = (map_data or {}).get("drivable_polygons", [])
        route_lines = _route_centerlines(map_data)
        if polygons:
            for idx, point in enumerate(path):
                if not any(_point_in_polygon(point, polygon) for polygon in polygons):
                    errors.append(
                        f"scene: ego outside drivable area at path_index={idx}")
                    break
        elif route_lines:
            halfwidth = float((map_data or {}).get(
                "route_half_width_m",
                scene_cfg.get("local_drivable_half_width_m", 8.0)))
            coverage_incomplete = (
                (map_data or {}).get("route_coverage_complete") is False)
            if coverage_incomplete:
                errors.append(
                    "scene: route evidence incomplete for required horizon "
                    f"(sampled={(map_data or {}).get('route_coverage_m')}m, "
                    f"required={(map_data or {}).get('route_required_m')}m)")
            else:
                for idx, point in enumerate(path):
                    projection = _nearest_route_projection(point, route_lines)
                    if projection["status"] != "supported":
                        errors.append(
                            f"scene: route evidence incomplete at path_index={idx} "
                            f"(coverage={projection['status']})")
                        break
                    distance = float(projection["distance"])
                    if distance > halfwidth:
                        errors.append(
                            f"scene: ego outside route drivable corridor at "
                            f"path_index={idx} (distance={distance:.2f}m, "
                            f"halfwidth={halfwidth:.2f}m)")
                        break
        elif heading is not None:
            # Dataset builders use a scenario-local road corridor when an HD-map
            # polygon is unavailable. Coordinates may be global (CARLA) or T0
            # ego-aligned (nuScenes), so measure signed lateral displacement from
            # the first observed ego position rather than absolute x/y.
            halfwidth = float(scene_cfg.get("local_drivable_half_width_m", 8.0))
            ox, oy = path[0]
            hx, hy = heading
            for idx, (x, y) in enumerate(path):
                lateral = -hy * (x - ox) + hx * (y - oy)
                if abs(lateral) > halfwidth:
                    errors.append(
                        f"scene: ego outside local drivable corridor at path_index={idx} "
                        f"(lateral={lateral:.2f}m, halfwidth={halfwidth:.2f}m)")
                    break

    if scene_cfg.get("check_static_boundary_crossing", False) and len(path) >= 2:
        boundary_segments = (map_data or {}).get("static_boundary_segments", [])
        crossed = False
        for raw_segment in boundary_segments:
            try:
                c = (float(raw_segment[0][0]), float(raw_segment[0][1]))
                d = (float(raw_segment[1][0]), float(raw_segment[1][1]))
            except (TypeError, ValueError, IndexError):
                errors.append("scene: malformed static boundary segment")
                crossed = True
                break
            if any(_segments_intersect(a, b, c, d) for a, b in zip(path, path[1:])):
                errors.append("scene: ego path crosses static boundary segment")
                crossed = True
                break

        # CARLA roots carry guardrail/curb actors rather than a separate map
        # artifact. With route evidence, compare actor/path lateral positions
        # in the curved route frame; inputs without route evidence use the
        # initial-heading construction below.
        route_lines = _route_centerlines(map_data)
        if not crossed and root and (route_lines or heading is not None):
            ox, oy = path[0]
            if route_lines:
                path_projections = [
                    _nearest_route_projection(point, route_lines)
                    for point in path]
                if any(item["status"] != "supported"
                       for item in path_projections):
                    lateral_path = []
                else:
                    lateral_path = [float(item["signed_distance"])
                                    for item in path_projections]
            else:
                hx, hy = heading  # type: ignore[misc]
                lateral_path = [-hy * (x - ox) + hx * (y - oy) for x, y in path]
            if lateral_path:
                for actor in (root.get("actors") or []):
                    if actor.get("role_type") != "road_boundary":
                        continue
                    xy = _actor_xy(actor)
                    if xy is None:
                        continue
                    if route_lines:
                        boundary_projection = _nearest_route_projection(xy, route_lines)
                        if boundary_projection["status"] != "supported":
                            continue
                        boundary_lat = float(boundary_projection["signed_distance"])
                    else:
                        boundary_lat = -hy * (xy[0] - ox) + hx * (xy[1] - oy)
                    for prev_lat, curr_lat in zip(lateral_path, lateral_path[1:]):
                        if ((prev_lat - boundary_lat) * (curr_lat - boundary_lat) <= 0
                                and abs(curr_lat - prev_lat) > 1e-9):
                            errors.append(
                                f"scene: ego path crosses root static boundary actor={actor.get('id')}")
                            crossed = True
                            break
                    if crossed:
                        break

    return len(errors) == 0, errors


def check_kinematic_validity(branch: Dict[str, Any], config: Dict[str, Any], root: Optional[Dict[str, Any]] = None) -> Tuple[bool, List[str]]:
    """Kinematic gate. `root` (the observed_state) seeds the per-actor history
    so the T0 -> first-node transition is checked too; the root
    itself is model input and is never scored."""
    bounds = config.get("kinematic_bounds", {})
    return _check_displacement_and_speed(branch, bounds, config, prefix="kin", root=root)


def compute_split_depth(branch: Dict[str, Any], factual_branch: Dict[str, Any], config: Dict[str, Any], ontology: Optional[Dict[str, Any]] = None, root: Optional[Dict[str, Any]] = None) -> int:
    """Section 5.2: Split depth via accumulated world-state comparison.

    Uses state propagation: actors from observed_state are carried forward,
    then actors_update patches are applied incrementally. The full world
    snapshot is hashed at each depth, not just the local update.
    """
    d_max = int(config.get("bounds", {}).get("D_max", 6))
    b_states = _build_accumulated_states(branch, ontology, root=root)
    f_states = _build_accumulated_states(factual_branch, ontology, root=root)
    horizon = min(len(b_states), len(f_states), d_max)
    for i in range(horizon):
        if not _accumulated_states_match(b_states[i], f_states[i], config):
            return i + 1
    if len(b_states) != len(f_states) and horizon < d_max:
        return horizon + 1
    return d_max + 1


def score_chain(gt_chain: Dict[str, Any] | List[Dict[str, Any]], pred_chain: Dict[str, Any] | List[Dict[str, Any]], ontology: Dict[str, Any], config: Dict[str, Any]) -> float:
    gamma = float(config.get("layer2", {}).get("gamma", 0.9))
    gt_nodes = gt_chain["chain"] if isinstance(gt_chain, dict) else gt_chain
    pred_nodes = pred_chain["chain"] if isinstance(pred_chain, dict) else pred_chain
    D = max(len(gt_nodes), len(pred_nodes))
    if D == 0:
        return 0.0

    sims: List[float] = []
    dummy = float(config.get("layer2", {}).get("dummy_node_similarity", 0.0))
    for i in range(D):
        if i < len(gt_nodes) and i < len(pred_nodes):
            sims.append(_node_similarity(gt_nodes[i], pred_nodes[i], ontology, config))
        else:
            sims.append(dummy)

    weights = [gamma ** i for i in range(D)]
    prefix_mins = []
    running = 1.0
    for s in sims:
        running = min(running, s)
        prefix_mins.append(running)

    Z = sum(weights)
    return sum(w * s for w, s in zip(weights, prefix_mins)) / Z if Z > 0 else 0.0


def _root_universe_from(root: Optional[Dict[str, Any]]) -> Optional[Dict[str, str]]:
    if root is None:
        return None
    return {
        str(a.get("id")): a.get("entity_type")
        for a in (root.get("actors") or []) if a.get("id") is not None
    }


def score_layer3(gt_branches: List[Dict[str, Any]], pred_branches: List[Dict[str, Any]], ontology: Dict[str, Any], config: Dict[str, Any], branch_schema: Dict[str, Any], map_data: Optional[Dict[str, Any]] = None, root: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    factual_gt = [b for b in gt_branches if b.get("is_factual") is True]
    if len(factual_gt) != 1:
        return {
            "F1_cf": 0.0,
            "Precision_cf": 0.0,
            "Recall_cf": 0.0,
            "matches": [],
            "diagnostics": {"gt_factual_count": len(factual_gt), "reason": "invalid_gt_factual_branch_count"},
        }
    factual_branch = canonicalize_branch(factual_gt[0], ontology, config)
    root_universe = _root_universe_from(root)

    gt_canon = [canonicalize_branch(b, ontology, config) for b in gt_branches]
    declared_actions = {
        (b.get("intervention") or {}).get("intervention_type")
        for b in gt_branches
        if (b.get("intervention") or {}).get("target_actor_id", "ego") == "ego"
    }
    declared_actions.discard(None)
    # A branch container that is not an array (null, a string, a number)
    # carries no evaluable unit: it scores as an empty prediction rather than
    # raising out of the sample.
    if not isinstance(pred_branches, list):
        pred_branches = []
    predicted_action_counts: Dict[str, int] = {}
    for branch in pred_branches:
        if not isinstance(branch, dict) or not isinstance(branch.get("intervention"), dict):
            continue
        action = branch["intervention"].get("intervention_type")
        if action:
            predicted_action_counts[str(action)] = predicted_action_counts.get(str(action), 0) + 1

    k_max = int(config.get("bounds", {}).get("K_max", 8))

    # â”€â”€ Anti-gaming pipeline â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    # Order: pre-normalize ALL â†’ canonicalize ALL â†’ dedup ALL â†’ THEN cap.
    # This prevents an attacker from controlling which branches reach the
    # scorer by manipulating submission order.
    pred_eval_all = []
    seen = set()
    for pred_index, pred in enumerate(pred_branches):
        # A unit whose shape cannot be evaluated fails ITS OWN
        # schema gate; it never raises out of the sample. Every other gate of
        # that unit is "not evaluable" (False) and it is matched to nothing.
        malformed_msgs = malformed_unit_errors(pred)
        if not malformed_msgs:
            try:
                # Pre-normalize: alias labels â†’ canonical, actors â†’ actors_update
                pred_normalized = _pre_normalize_branch(pred, ontology)
                schema_ok, schema_msgs = check_schema_validity(pred_normalized, branch_schema)
                canon = canonicalize_branch(pred_normalized, ontology, config)
                semantic_ok, semantic_msgs = check_semantic_validity(canon, ontology, root_universe=root_universe)
                scene_ok, scene_msgs = check_scene_validity(
                    canon, config, map_data=map_data, root=root)
                kinematic_ok, kinematic_msgs = check_kinematic_validity(canon, config, root=root)
            except Exception as exc:  # shape the pre-check did not anticipate
                malformed_msgs = [f"schema: unit not evaluable ({type(exc).__name__}: {exc})"]
        if malformed_msgs:
            schema_ok, schema_msgs = False, malformed_msgs
            semantic_ok = scene_ok = kinematic_ok = False
            semantic_msgs = scene_msgs = kinematic_msgs = ["not evaluable: malformed unit"]
            canon = {"chain": [], "_malformed_unit_index": pred_index}
        intervention = pred.get("intervention") if isinstance(pred, dict) else None
        intervention = intervention if isinstance(intervention, dict) else {}
        action = intervention.get("intervention_type")
        metadata_msgs = []
        if not isinstance(pred, dict) or not isinstance(pred.get("intervention"), dict):
            metadata_msgs.append("metadata: missing Layer-3 intervention object")
        if not isinstance(action, str) or action not in declared_actions:
            metadata_msgs.append(f"metadata: undeclared ego action {action!r}")
        if intervention.get("target_actor_id") != "ego":
            metadata_msgs.append("metadata: Layer-3 target_actor_id must be ego")
        if intervention.get("trigger_depth") != 1:
            metadata_msgs.append("metadata: Layer-3 trigger_depth must be 1")
        if action and predicted_action_counts.get(str(action), 0) != 1:
            metadata_msgs.append(f"metadata: ego action {action!r} must appear exactly once")
        metadata_ok = not metadata_msgs
        canonical_string = json.dumps(canon, sort_keys=True, separators=(",", ":"))
        is_dup = canonical_string in seen
        if not is_dup:
            seen.add(canonical_string)
        pred_eval_all.append({
            "raw": pred,
            "canon": canon,
            "schema_ok": schema_ok,
            "semantic_ok": semantic_ok,
            "scene_ok": scene_ok,
            "kinematic_ok": kinematic_ok,
            "metadata_ok": metadata_ok,
            "schema_msgs": schema_msgs,
            "semantic_msgs": semantic_msgs,
            "scene_msgs": scene_msgs,
            "kinematic_msgs": kinematic_msgs,
            "metadata_msgs": metadata_msgs,
            "duplicate": is_dup,
            "malformed": bool(malformed_msgs),
        })

    # Dedup first, THEN enforce K_max on unique set
    pred_unique = [p for p in pred_eval_all if not p["duplicate"]]

    gt_valid = [{"canon": g, "valid": True} for g in gt_canon]
    # Naming: outline 5.5 uses m = |B̂_pred| (predictions) and n = |B_GT|;
    # explicit names keep the two apart.
    n_gt = len(gt_valid)

    # â”€â”€ K_max overflow: proportional penalty â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    # If unique predictions exceed K_max, ALL of them are still scored (no
    # subset selection â†’ order cannot influence which branches count) and
    # the final F1 is scaled by K_max / n_unique. Spam is strictly worse
    # than restraint (precision already divides by n_pred; the factor adds
    # a structural penalty), but one branch over the cap does not zero
    # an otherwise-good submission.
    k_max_penalty = 1.0
    if len(pred_unique) > k_max:
        k_max_penalty = k_max / len(pred_unique)

    pred_active = pred_unique
    n_pred = len(pred_active)  # outline 5.5's m
    size = max(n_gt, n_pred)
    cost = [[1.0 for _ in range(size)] for _ in range(size)]
    score_matrix = [[0.0 for _ in range(size)] for _ in range(size)]

    pw = config.get("pairwise_weights", {})
    lambda_chain = float(pw.get("lambda_chain", 0.7))
    mu_split = float(pw.get("mu_split", 0.2))
    nu_term = float(pw.get("nu_term", 0.1))
    terminal_types = set(ontology.get("terminal_event_types", []))

    # Action-labelled matching: a (reference, prediction) pair
    # whose ego-action labels are both present and differ scores zero, so a
    # submission cannot earn credit by attaching the right chain to the wrong
    # action. Pairs where either side carries no label stay unconstrained;
    # protocol runs also enforce labels via the metadata gate.
    match_by_action = bool(config.get("layer3", {}).get("match_by_action", False))
    gt_actions = [(b.get("intervention") or {}).get("intervention_type") for b in gt_branches]
    label_mismatch_zeroed = 0

    for i, gt_item in enumerate(gt_valid):
        gt_branch = gt_item["canon"]
        gt_split = compute_split_depth(gt_branch, factual_branch, config, ontology, root=root)
        gt_term = _branch_terminal_type(gt_branch)
        for j, pred_item in enumerate(pred_active):
            pred_branch = pred_item["canon"]
            if pred_item.get("malformed"):
                cost[i][j] = 1.0
                score_matrix[i][j] = 0.0
                continue
            raw_intervention = pred_item["raw"].get("intervention") if isinstance(pred_item["raw"], dict) else None
            pred_action = (raw_intervention or {}).get("intervention_type")
            if (match_by_action and gt_actions[i] is not None and pred_action is not None
                    and str(gt_actions[i]) != str(pred_action)):
                label_mismatch_zeroed += 1
                cost[i][j] = 1.0
                score_matrix[i][j] = 0.0
                continue
            # Gate reads config toggles â€” not hardcoded
            l3_cfg = config.get("layer3", {})
            gate_pass = True
            if l3_cfg.get("use_schema_gate", True) and not pred_item["schema_ok"]:
                gate_pass = False
            if l3_cfg.get("use_semantic_gate", True) and not pred_item["semantic_ok"]:
                gate_pass = False
            if l3_cfg.get("use_scene_gate", True) and not pred_item["scene_ok"]:
                gate_pass = False
            if l3_cfg.get("use_kinematic_gate", True) and not pred_item["kinematic_ok"]:
                gate_pass = False
            if l3_cfg.get("use_metadata_gate", True) and not pred_item["metadata_ok"]:
                gate_pass = False
            gate = 1.0 if gate_pass else 0.0
            s_chain = score_chain(gt_branch, pred_branch, ontology, config)
            pred_split = compute_split_depth(pred_branch, factual_branch, config, ontology, root=root)
            s_split = _split_similarity(gt_split, pred_split, config)
            pred_term = _branch_terminal_type(pred_branch)
            s_term = _terminal_similarity(gt_term, pred_term, terminal_types)
            s_pair = gate * (lambda_chain * s_chain + mu_split * s_split + nu_term * s_term)
            cost[i][j] = 1.0 - s_pair
            score_matrix[i][j] = s_pair

    assignment = hungarian_min_cost(cost)
    match_scores = []
    real_matches = []
    matched_gt = 0
    matched_pred = 0

    for i, j in assignment:
        if i < n_gt and j < n_pred:
            s_pair = score_matrix[i][j]
            match_scores.append(s_pair)
            if s_pair > 0:
                matched_gt += 1
                matched_pred += 1
                real_matches.append({
                    "gt_index": i,
                    "pred_index": j,
                    "S_pair": s_pair,
                    "depth_scores": [
                        _node_similarity(gt_valid[i]["canon"]["chain"][d],
                                         pred_active[j]["canon"]["chain"][d],
                                         ontology, config)
                        for d in range(min(
                            len(gt_valid[i]["canon"].get("chain", [])),
                            len(pred_active[j]["canon"].get("chain", [])),
                            int(config.get("bounds", {}).get("D_max", 6))))
                    ],
                })

    # Outline 5.5: Precision divides by the prediction count (its m); the
    # if_m_zero_precision fallback therefore fires on an EMPTY PREDICTION set.
    precision = sum(match_scores) / n_pred if n_pred > 0 else float(config.get("f1", {}).get("if_m_zero_precision", 0.0))
    recall = sum(match_scores) / n_gt if n_gt > 0 else 0.0
    denom = precision + recall
    f1 = 0.0 if denom == 0 else (2 * precision * recall) / denom
    f1 *= k_max_penalty

    # â”€â”€ Compliance vs reasoning separation â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    # gate_failure_counts: how many unique branches each gate rejected
    # (compliance axis). *_valid_only: the same F1 with precision conditioned
    # on gate-passing branches (reasoning axis). The K_max penalty is a
    # compliance failure, so it is NOT applied to the valid-only score.
    gate_failure_counts = {
        "schema": sum(1 for p in pred_active if not p["schema_ok"]),
        "semantic": sum(1 for p in pred_active if not p["semantic_ok"]),
        "scene": sum(1 for p in pred_active if not p["scene_ok"]),
        "kinematic": sum(1 for p in pred_active if not p["kinematic_ok"]),
        "metadata": sum(1 for p in pred_active if not p["metadata_ok"])
        if config.get("layer3", {}).get("use_metadata_gate", False) else 0,
    }
    metadata_required = config.get("layer3", {}).get("use_metadata_gate", False)
    first_failure_counts = {name: 0 for name in
                            ("schema", "semantic", "scene", "kinematic", "metadata")}
    for item in pred_active:
        checks = [("schema", item["schema_ok"]), ("semantic", item["semantic_ok"]),
                  ("scene", item["scene_ok"]), ("kinematic", item["kinematic_ok"]),
                  ("metadata", item["metadata_ok"] or not metadata_required)]
        for name, passed in checks:
            if not passed:
                first_failure_counts[name] += 1
                break
    n_valid = sum(1 for p in pred_active if p["schema_ok"] and p["semantic_ok"]
                  and p["scene_ok"] and p["kinematic_ok"]
                  and (p["metadata_ok"] or not metadata_required))
    precision_valid = sum(match_scores) / n_valid if n_valid > 0 else 0.0
    denom_valid = precision_valid + recall
    f1_valid = 0.0 if denom_valid == 0 else (2 * precision_valid * recall) / denom_valid

    return {
        "F1_cf": f1,
        "Precision_cf": precision,
        "Recall_cf": recall,
        "F1_cf_valid_only": f1_valid,
        "Precision_cf_valid_only": precision_valid,
        "matches": real_matches,
        "diagnostics": {
            "gt_count": n_gt,
            "pred_total_submitted": len(pred_branches),
            "pred_duplicates_collapsed": sum(1 for p in pred_eval_all if p["duplicate"]),
            "pred_unique_count": len(pred_unique),
            "pred_scored_count": n_pred,
            "pred_valid_count": n_valid,
            "k_max": k_max,
            "k_max_penalty": k_max_penalty,
            "match_by_action": match_by_action,
            "label_mismatch_pairs_zeroed": label_mismatch_zeroed,
            "malformed_units": sum(1 for p in pred_active if p.get("malformed")),
            "gate_failure_counts": gate_failure_counts,
            "first_failure_counts": first_failure_counts,
            "pred_gate_failures": [
                {
                    "index": idx,
                    "schema_msgs": p["schema_msgs"],
                    "semantic_msgs": p["semantic_msgs"],
                    "scene_msgs": p["scene_msgs"],
                    "kinematic_msgs": p["kinematic_msgs"],
                    "metadata_msgs": p["metadata_msgs"],
                }
                for idx, p in enumerate(pred_active)
                if not (p["schema_ok"] and p["semantic_ok"] and p["scene_ok"]
                        and p["kinematic_ok"]
                        and (p["metadata_ok"] or not metadata_required))
            ],
        },
    }


def _invalid_layer4(reason: str, msgs: List[str]) -> Dict[str, Any]:
    return {
        "Score_interv": 0.0, "valid": False,
        "gates": {"schema": False, "semantic": False, "scene": False,
                  "kinematic": False, "metadata": False, "intervention": False},
        "diagnostics": {"reason": reason, "schema_msgs": msgs},
    }


def score_layer4(gt_cf_chain: Dict[str, Any], pred_cf_obj: Dict[str, Any], ontology: Dict[str, Any], config: Dict[str, Any], counterfactual_schema: Dict[str, Any], root: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Score ONE predicted L4 chain wrapper against ONE reference chain.
    Validity comes from `layer4_chain_gates` (reference-free); the similarity
    `score_chain` is added only when the chain is valid."""
    gt_branch = gt_cf_chain.get("counterfactual_chain", gt_cf_chain) if isinstance(gt_cf_chain, dict) else gt_cf_chain
    gt_intervention = (gt_branch.get("intervention") if isinstance(gt_branch, dict) else None) or {}
    gated = layer4_chain_gates(gt_intervention, pred_cf_obj, ontology, config,
                               counterfactual_schema, root)
    pred_canon = gated.pop("pred_canon", None)
    score = 0.0
    if gated["valid"] and pred_canon is not None:
        try:
            # Canonicalize BOTH sides (outline Section 2), mirroring Layer 3 — scoring
            # a raw GT against a canonicalized prediction penalizes rounding artifacts.
            gt_canon = canonicalize_branch(gt_branch, ontology, config)
            score = score_chain(gt_canon, pred_canon, ontology, config)
        except Exception as exc:  # a reference shape the gates cannot anticipate
            return _invalid_layer4("malformed_counterfactual_chain",
                                   [f"schema: answer not evaluable ({type(exc).__name__}: {exc})"])
    gated["Score_interv"] = score
    return gated


def layer4_chain_gates(gt_intervention: Dict[str, Any], pred_cf_obj: Dict[str, Any], ontology: Dict[str, Any], config: Dict[str, Any], counterfactual_schema: Dict[str, Any], root: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Reference-free validity of ONE predicted L4 chain wrapper
    (`{"counterfactual_chain": {"intervention", "chain"}}`): schema, semantic,
    scene and kinematic gates, the intervention-metadata match against the
    PUBLIC query tuple `gt_intervention` and the target-present check. It
    reads no reference chain, so `analysis/intervention_pair.py` applies the
    very gates the scorer applies before computing the reference-free pair
    axes. Returns {valid, gates, diagnostics, pred_canon}; `pred_canon` is
    None when the wrapper holds no evaluable chain. A malformed answer fails
    its own schema gate; it never raises out of the sample and never touches
    the other queries or L3."""
    if not isinstance(pred_cf_obj, dict):
        out = _invalid_layer4("malformed_counterfactual", ["schema: answer is not an object"])
        out["pred_canon"] = None
        return out
    raw_chain = pred_cf_obj.get("counterfactual_chain")
    if raw_chain is not None:
        malformed_msgs = malformed_unit_errors(raw_chain)
        if malformed_msgs:
            out = _invalid_layer4("malformed_counterfactual_chain", malformed_msgs)
            out["pred_canon"] = None
            return out
    try:
        return _layer4_chain_gates_evaluable(gt_intervention, pred_cf_obj, ontology, config,
                                             counterfactual_schema, root)
    except Exception as exc:  # shape the pre-check did not anticipate
        out = _invalid_layer4("malformed_counterfactual_chain",
                              [f"schema: answer not evaluable ({type(exc).__name__}: {exc})"])
        out["pred_canon"] = None
        return out


def _layer4_chain_gates_evaluable(gt_intervention: Dict[str, Any], pred_cf_obj: Dict[str, Any], ontology: Dict[str, Any], config: Dict[str, Any], counterfactual_schema: Dict[str, Any], root: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    # Pre-normalize: alias labels → canonical, actors → actors_update
    pred_cf_normalized = copy.deepcopy(pred_cf_obj)
    if isinstance(pred_cf_normalized.get("counterfactual_chain"), dict):
        pred_cf_normalized["counterfactual_chain"] = _pre_normalize_branch(
            pred_cf_normalized["counterfactual_chain"], ontology
        )
    schema_ok, schema_msgs = check_schema_validity(pred_cf_normalized, counterfactual_schema)
    pred_branch = pred_cf_normalized.get("counterfactual_chain")
    if not isinstance(pred_branch, dict):
        return {
            "valid": False,
            "gates": {"schema": bool(schema_ok), "semantic": False, "scene": False,
                      "kinematic": False, "metadata": False, "intervention": False},
            "diagnostics": {"reason": "missing_counterfactual_chain", "schema_msgs": schema_msgs},
            "pred_canon": None,
        }

    pred_canon = canonicalize_branch(pred_branch, ontology, config)
    # Layer 4: drop required_roles completeness by default (outline Section 6) so
    # collision/off-road outcomes of a forced intervention are not rejected for
    # an absent role. Structural validity still applies.
    check_roles_l4 = config.get("layer4", {}).get("use_role_completeness_gate", False)
    semantic_ok, semantic_msgs = check_semantic_validity(pred_canon, ontology, root_universe=_root_universe_from(root), check_role_completeness=check_roles_l4)
    scene_ok, scene_msgs = check_scene_validity(pred_canon, config, root=root)
    kin_ok, kin_msgs = check_kinematic_validity(pred_canon, config, root=root)

    # Intervention metadata exact-match gate against the public query tuple
    interv_ok, interv_msgs = _intervention_matches_gt(
        {"counterfactual_chain": {"intervention": gt_intervention or {}}}, pred_cf_normalized)
    # Validate target_actor_id exists in pred chain
    target_ok, target_msgs = _validate_intervention_target_exists(pred_canon, pred_cf_normalized)

    # Gate reads config toggles — not hardcoded
    l4_cfg = config.get("layer4", {})
    valid = True
    if l4_cfg.get("use_schema_gate", True) and not schema_ok:
        valid = False
    if l4_cfg.get("use_semantic_gate", True) and not semantic_ok:
        valid = False
    if l4_cfg.get("use_kinematic_gate", True) and not kin_ok:
        valid = False
    if l4_cfg.get("use_scene_gate", False) and not scene_ok:
        valid = False
    if not interv_ok:
        valid = False
    if not target_ok:
        valid = False

    return {
        "valid": valid,
        # Compliance axis: raw per-gate outcomes, independent of the
        # layer4.use_*_gate toggles that decide validity.
        "gates": {
            "schema": bool(schema_ok),
            "semantic": bool(semantic_ok),
            "scene": bool(scene_ok),
            "kinematic": bool(kin_ok),
            "metadata": bool(interv_ok and target_ok),
            "intervention": bool(interv_ok and target_ok),
        },
        "diagnostics": {
            "schema_msgs": schema_msgs,
            "semantic_msgs": semantic_msgs,
            "scene_msgs": scene_msgs,
            "kin_msgs": kin_msgs,
            "interv_msgs": interv_msgs + target_msgs,
        },
        "pred_canon": pred_canon,
    }


def control_intervention_for(intervention: Dict[str, Any]) -> Dict[str, Any]:
    """The intervention block of the control chain paired with `intervention`:
    the same target and trigger, the synthetic `target_baseline` label."""
    interv = intervention if isinstance(intervention, dict) else {}
    return {"intervention_type": CONTROL_INTERVENTION_TYPE,
            "target_actor_id": interv.get("target_actor_id"),
            "trigger_depth": interv.get("trigger_depth", 1)}


def split_l4_prediction_item(item: Any) -> Dict[str, Any]:
    """Read one raw `predicted_counterfactuals[]` item into its parts.
    `treated_chain` is the treated chain; the pre-pair key `chain` is
    accepted as its alias. `control_chain` is the control chain. Presence is
    reported separately from shape (a present but malformed chain still
    counts as present and then fails its own gates)."""
    if not isinstance(item, dict):
        return {"intervention": None, "treated_present": False, "control_present": False,
                "treated_chain": None, "control_chain": None}
    treated_present = "treated_chain" in item or "chain" in item
    treated_chain = item.get("treated_chain") if "treated_chain" in item else item.get("chain")
    return {
        "intervention": item.get("intervention"),
        "treated_present": treated_present,
        "control_present": "control_chain" in item,
        "treated_chain": treated_chain,
        "control_chain": item.get("control_chain"),
    }


def score_layer4_pair(gt_query: Dict[str, Any], pred_item: Any, ontology: Dict[str, Any], config: Dict[str, Any], counterfactual_schema: Dict[str, Any], root: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Score one L4 query answered by a matched control/treated pair
    (docs/l4_pair_protocol.md §3, §5).

    The PAIR is the unit: `valid` is `pair_valid` (both chains pass their own
    gates) and the reported `Score_interv` is the treated-chain agreement of a
    pair-valid answer, else 0.0, so the RQ2 identity 1−R = (1−K) + (K−R)
    holds on pairs. Each chain's own gate result and own-validity agreement
    stay under `treated` / `control`; `control_similarity` is the control
    chain's agreement whenever the control chain itself is valid (reported
    separately, never folded into CTMS). A reference without `control_chain`
    (a dataset not migrated to the pair protocol) is scored on the treated
    chain alone and says so in `pair.protocol`; campaigns refuse such a
    dataset (`evaluation_campaign.dataset_inventory`)."""
    gt = gt_query if isinstance(gt_query, dict) else {}
    gt_control = gt.get("control_chain") if isinstance(gt.get("control_chain"), dict) else None
    parts = split_l4_prediction_item(pred_item)
    interv = parts["intervention"] if isinstance(parts["intervention"], dict) else {}
    treated_wrapper = {"counterfactual_chain": {"intervention": interv, "chain": parts["treated_chain"]}} \
        if parts["treated_present"] else ({"counterfactual_chain": {"intervention": interv}} if parts["intervention"] is not None else {})
    treated = score_layer4(gt, treated_wrapper, ontology, config, counterfactual_schema, root)
    if not parts["treated_present"] and treated.get("diagnostics", {}).get("reason") is None:
        treated["diagnostics"]["reason"] = "missing_treated_chain"

    if gt_control is None:
        pair = {"protocol": "single_treated_legacy", "control_reference_present": False,
                "treated_present": parts["treated_present"], "control_present": parts["control_present"],
                "treated_valid": bool(treated["valid"]), "control_valid": None,
                "pair_complete": None, "pair_valid": None}
        out = dict(treated)
        out.update({"treated": treated, "control": None, "control_similarity": None, "pair": pair})
        return out

    if parts["control_present"]:
        control_wrapper = {"counterfactual_chain": {"intervention": control_intervention_for(interv),
                                                    "chain": parts["control_chain"]}}
        control = score_layer4(gt_control, control_wrapper, ontology, config, counterfactual_schema, root)
    else:
        control = _invalid_layer4("missing_control_chain", [])
    pair_complete = parts["treated_present"] and parts["control_present"]
    pair_valid = bool(treated["valid"] and control["valid"])
    if pair_valid:
        reason = None
    elif not pair_complete:
        reason = ("missing_control_chain" if parts["treated_present"] else
                  "missing_treated_chain" if parts["control_present"] else "missing_pair")
    elif not treated["valid"] and not control["valid"]:
        reason = "both_chains_invalid"
    else:
        reason = "control_chain_invalid" if not control["valid"] else "treated_chain_invalid"
    gates = {g: bool(treated["gates"].get(g) and control["gates"].get(g)) for g in L4_GATE_KEYS}
    return {
        "Score_interv": float(treated["Score_interv"]) if pair_valid else 0.0,
        "valid": pair_valid,
        "gates": gates,
        "diagnostics": {"reason": reason, "treated": treated.get("diagnostics"),
                        "control": control.get("diagnostics")},
        "treated": treated,
        "control": control,
        "control_similarity": float(control["Score_interv"]) if control["valid"] else 0.0,
        "pair": {"protocol": L4_PAIR_PROTOCOL, "control_reference_present": True,
                 "treated_present": parts["treated_present"], "control_present": parts["control_present"],
                 "treated_valid": bool(treated["valid"]), "control_valid": bool(control["valid"]),
                 "pair_complete": pair_complete, "pair_valid": pair_valid},
    }


def _intervention_matches_gt(gt_cf_obj: Dict[str, Any], pred_cf_obj: Dict[str, Any]) -> Tuple[bool, List[str]]:
    """Exact-match gate for intervention metadata between GT and prediction."""
    errors: List[str] = []
    gt_chain = gt_cf_obj if isinstance(gt_cf_obj, dict) else {}
    pred_chain = pred_cf_obj.get("counterfactual_chain", {}) if isinstance(pred_cf_obj, dict) else {}

    gt_interv = gt_chain.get("intervention", {})
    if not gt_interv:
        inner = gt_chain.get("counterfactual_chain")
        if isinstance(inner, dict):
            gt_interv = inner.get("intervention", {})
    pred_interv = pred_chain.get("intervention", {})

    if not gt_interv:
        # GT has no intervention block to validate against
        return True, []

    if not pred_interv:
        errors.append("intervention: missing intervention block in prediction")
        return False, errors

    for field in ("intervention_type", "target_actor_id", "trigger_depth"):
        gt_val = gt_interv.get(field)
        if gt_val is None:
            continue  # GT doesn't require this field
        pred_val = pred_interv.get(field)
        if pred_val != gt_val:
            errors.append(f"intervention: {field} mismatch: pred={pred_val!r} != gt={gt_val!r}")

    return len(errors) == 0, errors


def _validate_intervention_target_exists(branch_canon: Dict[str, Any], pred_cf_obj: Dict[str, Any]) -> Tuple[bool, List[str]]:
    """Validate that intervention.target_actor_id exists in the root U_actors universe (node[0])."""
    errors: List[str] = []
    pred_chain = pred_cf_obj.get("counterfactual_chain", {}) if isinstance(pred_cf_obj, dict) else {}
    interv = pred_chain.get("intervention", {})
    target_id = interv.get("target_actor_id")
    if target_id is None:
        return True, []  # No target to validate

    chain = branch_canon.get("chain", [])
    if not chain:
        errors.append("intervention: cannot validate target against empty chain")
        return False, errors

    # Collect U_actors from root node (node[0]) only â€” handles both actors and actors_update
    root_node = chain[0]
    root_actors = root_node.get("actors", []) or root_node.get("actors_update", []) or []
    root_universe_ids = {str(a.get("id")) for a in root_actors if a.get("id") is not None}

    if str(target_id) not in root_universe_ids:
        errors.append(f"intervention: target_actor_id={target_id!r} not found in root U_actors universe")

    return len(errors) == 0, errors


def evaluate_ctms(sample_gt: Dict[str, Any], sample_pred: Dict[str, Any], ontology: Dict[str, Any], config: Dict[str, Any], branch_schema: Dict[str, Any], counterfactual_schema: Dict[str, Any], map_data: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    root = sample_gt.get("observed_state")
    layer3 = score_layer3(
        gt_branches=sample_gt.get("branches", []),
        pred_branches=sample_pred.get("branches", []),
        ontology=ontology,
        config=config,
        branch_schema=branch_schema,
        map_data=map_data,
        root=root,
    )

    # ── Layer 4: per-intervention matching ───────────────────────────
    # The reference L4 query set is `l4_queries`; datasets that carry it as a
    # `counterfactuals` list or a single `counterfactual` object are also accepted.
    gt_cf_list = sample_gt.get("l4_queries") or sample_gt.get("counterfactuals")
    if not gt_cf_list:
        legacy = sample_gt.get("counterfactual")
        gt_cf_list = [legacy] if legacy else []

    # Build pred lookup: intervention_type → wrapped pred object.
    # Pred may use predicted_counterfactuals[] or a single counterfactual
    # object. Each item is wrapped to match score_layer4
    # contract: {"counterfactual_chain": {"intervention": ..., "chain": ...}}.
    pred_cf_map: Dict[str, Dict[str, Any]] = {}
    # Spec §V.6: exactly ONE predicted chain per intervention. More than one
    # chain for the same intervention_type invalidates that intervention.
    duplicate_cf_types: set = set()
    pred_cf_items = sample_pred.get("predicted_counterfactuals")
    if isinstance(pred_cf_items, list):
        for item in pred_cf_items:
            if not isinstance(item, dict):
                continue
            interv = item.get("intervention")
            if not isinstance(interv, dict):
                # No pairable query: the answer is dropped here and the query
                # it may have meant stays unmatched (scored 0, valid False).
                continue
            itype = interv.get("intervention_type")
            # A non-string type names no query: the answer stays unpaired and
            # the query it may have meant is scored 0 / valid False, like a
            # missing answer.
            if itype and isinstance(itype, str):
                if itype in pred_cf_map:
                    duplicate_cf_types.add(itype)
                # The raw item is kept: `score_layer4_pair` reads its treated
                # chain (`treated_chain`, alias `chain`) and `control_chain`.
                pred_cf_map[itype] = item
    else:
        # Single counterfactual object
        legacy_pred = sample_pred.get("counterfactual")
        if isinstance(legacy_pred, dict):
            cc = legacy_pred.get("counterfactual_chain", {})
            itype = cc.get("intervention", {}).get("intervention_type") if isinstance(cc, dict) else None
            if itype and isinstance(itype, str):
                pred_cf_map[itype] = {"intervention": cc.get("intervention"),
                                      "chain": cc.get("chain", [])}

    # Score each GT query against the matched predicted pair. Spec V.6 still
    # holds per query: one PAIR per intervention_type; a second item of the
    # same type invalidates the query.
    layer4_results = []
    for gt_cf in gt_cf_list:
        if not isinstance(gt_cf, dict):
            continue
        itype = gt_cf.get("intervention_type")
        if not itype:
            cc = gt_cf.get("counterfactual_chain")
            if isinstance(cc, dict):
                itype = (cc.get("intervention") or {}).get("intervention_type")
        pred_cf_obj = pred_cf_map.get(itype, {}) if itype else {}
        if itype and itype in duplicate_cf_types:
            result = _invalid_layer4("multiple_chains_for_intervention", [])
            result.update({"treated": None, "control": None,
                           "control_similarity": 0.0 if isinstance(gt_cf.get("control_chain"), dict) else None,
                           "pair": {"protocol": (L4_PAIR_PROTOCOL if isinstance(gt_cf.get("control_chain"), dict)
                                                 else "single_treated_legacy"),
                                    "control_reference_present": isinstance(gt_cf.get("control_chain"), dict),
                                    "treated_present": None, "control_present": None,
                                    "treated_valid": False, "control_valid": False,
                                    "pair_complete": None, "pair_valid": False}})
        else:
            result = score_layer4_pair(
                gt_query=gt_cf,
                pred_item=pred_cf_obj,
                ontology=ontology,
                config=config,
                counterfactual_schema=counterfactual_schema,
                root=root,
            )
        result["intervention_type"] = itype
        gt_intervention = ((gt_cf.get("counterfactual_chain") or {}).get("intervention")
                           if isinstance(gt_cf, dict) else {}) or {}
        target_id = gt_intervention.get("target_actor_id")
        target_actor = next((a for a in (root or {}).get("actors", [])
                             if str(a.get("id")) == str(target_id)), None)
        result["target_actor_id"] = target_id
        result["target_class"] = (target_actor or {}).get("entity_type", "unknown")
        result["matched"] = bool(pred_cf_obj)
        layer4_results.append(result)

    # Score_interv = mean over all GT CFs (unmatched → 0.0)
    if layer4_results:
        score_interv = sum(r["Score_interv"] for r in layer4_results) / len(layer4_results)
    else:
        score_interv = 0.0

    f1_cf = float(layer3.get("F1_cf", 0.0))
    ctms = math.sqrt(max(0.0, f1_cf * score_interv))
    return {
        "Causal_Foresight_F1": f1_cf,
        "Counterfactual_Alignment_Score": score_interv,
        "CTMS": ctms,
        "layer3": layer3,
        "layer4": {
            "Score_interv": score_interv,
            "results": layer4_results,
        },
    }


def hungarian_min_cost(cost_matrix: List[List[float]]) -> List[Tuple[int, int]]:
    """O(nÂ³) Hungarian algorithm via scipy, replacing O(nÂ·2^n) bitmask DP."""
    n = len(cost_matrix)
    if n == 0:
        return []
    cost_arr = np.array(cost_matrix, dtype=np.float64)
    row_ind, col_ind = linear_sum_assignment(cost_arr)
    return list(zip(row_ind.tolist(), col_ind.tolist()))


def _check_displacement_and_speed(branch: Dict[str, Any], bounds: Dict[str, Any], config: Dict[str, Any], prefix: str, root: Optional[Dict[str, Any]] = None) -> Tuple[bool, List[str]]:
    """Kinematic plausibility gate for ALL actors.

    Uses per-actor timestamp tracking to avoid sparse time-warp.
    Cross-checks displacement-derived speed against vx/vy and raw speed
    to prevent kinematic decoupling fraud.

    Two further rules:
    * when `root` is given, its actors seed the per-actor history at the root
      timestamp, so the first predicted state is checked against the supplied
      T0 state instead of being exempt;
    * when `timing.require_protocol_grid` is on, node k must carry
      timestamp T0 + k * default_dt (within `protocol_grid_tolerance_seconds`);
      a chain shifted in time or sampled off the grid is a gate failure
      instead of being silently matched by index.
    """
    errors: List[str] = []
    chain = branch.get("chain", [])
    kin_cfg = config.get("kinematic_resolution", {})
    timing_cfg = config.get("timing", {})
    default_dt = float(timing_cfg.get("default_dt_seconds", 0.5))
    ts_field = timing_cfg.get("timestamp_field", "timestamp")
    max_dt = float(bounds.get("max_dt_seconds", 2.0))
    min_dt = float(bounds.get("min_dt_seconds", 0.01))
    speed_tol = float(kin_cfg.get("speed_conflict_tolerance", DEFAULT_SPEED_CONFLICT_TOLERANCE))

    max_speed = float(bounds.get("max_speed_mps", float("inf")))
    max_accel = float(bounds.get("max_accel_mps2", float("inf")))
    min_decel = float(bounds.get("min_decel_mps2", float("-inf")))
    max_disp_per_dt = float(bounds.get("max_displacement_per_0_5s", float("inf")))

    # Per-actor tracking: {actor_id: {"actor": dict, "ts": float}}
    prev_records: Dict[str, Dict[str, Any]] = {}
    prev_node_ts: Optional[float] = None
    root_ts: Optional[float] = None
    if root is not None:
        root_ts = _safe_float_or_none(root.get(ts_field))
        if root_ts is None:
            root_ts = 0.0
        for actor in root.get("actors") or []:
            if actor.get("id") is None:
                continue
            prev_records[str(actor["id"])] = {"actor": actor, "ts": root_ts}
        prev_node_ts = root_ts

    require_grid = bool(timing_cfg.get("require_protocol_grid", False))
    grid_tol = float(timing_cfg.get("protocol_grid_tolerance_seconds", 1e-3))
    grid_origin = root_ts if root_ts is not None else 0.0
    d_max = int(config.get("bounds", {}).get("D_max", 6))

    for idx, node in enumerate(chain):
        curr_ts_raw = node.get(ts_field)
        curr_ts = _safe_float_or_none(curr_ts_raw)
        if curr_ts is not None and not math.isfinite(curr_ts):
            # NaN/inf survive json.load and compare as False against every
            # bound; treat them as an unusable timestamp.
            errors.append(f"{prefix}: node[{idx}] timestamp={curr_ts_raw} is not finite")
            curr_ts = None

        if require_grid:
            if idx + 1 > d_max:
                errors.append(
                    f"{prefix}: node[{idx}] beyond the {d_max}-node horizon of the track")
            expected_ts = grid_origin + (idx + 1) * default_dt
            if curr_ts is None or abs(curr_ts - expected_ts) > grid_tol:
                errors.append(
                    f"{prefix}: node[{idx}] timestamp={curr_ts_raw} off the protocol grid "
                    f"(expected {expected_ts:.2f}s = T0 + {idx + 1}x{default_dt}s)")

        # Chain-level timestamp monotonicity (kept as global check)
        if curr_ts is not None and prev_node_ts is not None:
            chain_dt = curr_ts - prev_node_ts
            if chain_dt > max_dt:
                errors.append(f"{prefix}: node[{idx}] dt={chain_dt:.2f}s exceeds max_dt={max_dt}s (timestamp manipulation)")

        # Collect all actors in this node
        actors = node.get("actors", []) or node.get("actors_update", []) or []

        for actor in actors:
            aid = actor.get("id")
            if aid is None:
                continue
            aid = str(aid)

            prev_rec = prev_records.get(aid)
            prev_actor = prev_rec["actor"] if prev_rec else None

            for field in ("x", "y", "vx", "vy", "speed"):
                val = _safe_float_or_none(actor.get(field))
                if val is not None and not math.isfinite(val):
                    errors.append(f"{prefix}: node[{idx}] actor={aid} {field}={actor.get(field)} is not finite")

            # Compute per-actor dt from the last time THIS actor was seen
            if curr_ts is not None and prev_rec is not None and prev_rec["ts"] is not None:
                dt_actor = curr_ts - prev_rec["ts"]
                if dt_actor < min_dt:
                    dt_actor = default_dt
            else:
                dt_actor = default_dt

            curr_speed = _recover_actor_speed(actor=actor, prev_actor=prev_actor, dt=dt_actor, kin_cfg=kin_cfg)
            # Only enforce unrecoverable_speed when prev_actor exists.
            if curr_speed is None and prev_actor is not None and kin_cfg.get("fail_if_speed_unrecoverable", True):
                errors.append(f"{prefix}: node[{idx}] actor={aid} unrecoverable_speed")
                prev_records[aid] = {"actor": actor, "ts": curr_ts}
                continue

            if curr_speed is not None and curr_speed > max_speed:
                errors.append(f"{prefix}: node[{idx}] actor={aid} speed>{max_speed}")

            if prev_actor is not None and dt_actor > 0:
                # --- Displacement check ---
                ax_val = _safe_float_or_none(actor.get("x"))
                ay_val = _safe_float_or_none(actor.get("y"))
                px_val = _safe_float_or_none(prev_actor.get("x"))
                py_val = _safe_float_or_none(prev_actor.get("y"))

                if ax_val is not None and ay_val is not None and px_val is not None and py_val is not None:
                    dx = ax_val - px_val
                    dy = ay_val - py_val
                    disp = math.sqrt(dx * dx + dy * dy)
                    scaled_max_disp = max_disp_per_dt * (dt_actor / 0.5) if dt_actor > 0 else max_disp_per_dt
                    if disp > scaled_max_disp:
                        errors.append(f"{prefix}: node[{idx}] actor={aid} displacement>{scaled_max_disp:.1f} (dt={dt_actor:.2f}s)")

                    # --- Bug 2 fix: Cross-check speed sources ---
                    speed_from_disp = disp / dt_actor
                    vx = _safe_float_or_none(actor.get("vx"))
                    vy = _safe_float_or_none(actor.get("vy"))
                    speed_from_vel = math.sqrt(vx ** 2 + vy ** 2) if vx is not None and vy is not None else None
                    speed_from_raw = _safe_float_or_none(actor.get("speed"))

                    if speed_from_vel is not None and abs(speed_from_vel - speed_from_disp) > speed_tol:
                        errors.append(
                            f"{prefix}: node[{idx}] actor={aid} internal_physics_conflict: "
                            f"velocity_speed={speed_from_vel:.2f} vs displacement_speed={speed_from_disp:.2f} (dt={dt_actor:.2f}s)"
                        )
                    if speed_from_raw is not None and abs(speed_from_raw - speed_from_disp) > speed_tol:
                        errors.append(
                            f"{prefix}: node[{idx}] actor={aid} internal_physics_conflict: "
                            f"raw_speed={speed_from_raw:.2f} vs displacement_speed={speed_from_disp:.2f} (dt={dt_actor:.2f}s)"
                        )

                # --- Acceleration check ---
                prev_speed = _recover_actor_speed(actor=prev_actor, prev_actor=None, dt=dt_actor, kin_cfg=kin_cfg)
                if curr_speed is not None and prev_speed is not None and dt_actor > 0:
                    accel = (curr_speed - prev_speed) / dt_actor
                    if (accel > max_accel + ACCELERATION_BOUND_EPSILON
                            or accel < min_decel - ACCELERATION_BOUND_EPSILON):
                        errors.append(f"{prefix}: node[{idx}] actor={aid} accel out of bounds (dt={dt_actor:.2f}s)")

            # Update per-actor record
            prev_records[aid] = {"actor": actor, "ts": curr_ts}

        prev_node_ts = curr_ts if curr_ts is not None else prev_node_ts

    return len(errors) == 0, errors


def _recover_actor_speed(actor: Optional[Dict[str, Any]], prev_actor: Optional[Dict[str, Any]], dt: float, kin_cfg: Dict[str, Any]) -> Optional[float]:
    if actor is None:
        return None
    raw_speed = actor.get("speed")
    vx = actor.get("vx")
    vy = actor.get("vy")

    if kin_cfg.get("derive_speed_from_velocity", True) and vx is not None and vy is not None:
        return math.sqrt(float(vx) ** 2 + float(vy) ** 2)
    if raw_speed is not None:
        return float(raw_speed)
    if (
        kin_cfg.get("derive_speed_from_position_delta", True)
        and prev_actor is not None
        and dt > 0
        and actor.get("x") is not None and actor.get("y") is not None
        and prev_actor.get("x") is not None and prev_actor.get("y") is not None
    ):
        dx = float(actor["x"]) - float(prev_actor["x"])
        dy = float(actor["y"]) - float(prev_actor["y"])
        return math.sqrt(dx * dx + dy * dy) / dt
    return None


def _safe_str(value: Any) -> str:
    return "" if value is None else str(value)


def _safe_float(value: Any, default: float = float("inf")) -> float:
    if value is None:
        return default
    return float(value)


def _build_accumulated_states(branch: Dict[str, Any], ontology: Optional[Dict[str, Any]] = None, root: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
    """Build full world-state snapshot at each depth via state propagation.

    Node 0 (observed_state) initializes the world with 'actors'.
    All subsequent nodes patch the world via 'actors_update'.
    Actors with lifecycle state removes_from_world=true are removed.
    The full snapshot is stored per depth, not just the delta.
    """
    chain = branch.get("chain", [])
    accumulated: List[Dict[str, Any]] = []
    world: Dict[str, Dict[str, Any]] = {}  # actor_id -> latest actor state
    # Seed the world from the shared root so delta-only chains inherit
    # root actors not mentioned in their actors_update.
    if root is not None:
        for a in (root.get("actors") or []):
            world[str(a.get("id", ""))] = a

    # Derive despawn states from ontology instead of hardcoding
    despawn_states: set = set()
    if ontology:
        lifecycle = ontology.get("actor_lifecycle_states", {})
        for state_name, state_cfg in lifecycle.items():
            if isinstance(state_cfg, dict) and state_cfg.get("removes_from_world", False):
                despawn_states.add(state_name)
    if not despawn_states:
        # Fallback if ontology unavailable
        despawn_states = {"despawned", "exited", "out_of_scope"}

    for node in chain:
        # observed_state uses 'actors', everything else uses 'actors_update'
        for actor_field in ("actors", "actors_update"):
            for a in (node.get(actor_field) or []):
                aid = str(a.get("id", ""))
                if a.get("state") in despawn_states:
                    world.pop(aid, None)  # Actor leaves the scene
                else:
                    world[aid] = a

        accumulated.append({
            "event_type": node.get("event_type"),
            "action_type": node.get("action_type"),
            "intervention_type": node.get("intervention_type"),
            "world_snapshot": dict(world),  # shallow copy of full state
        })

    return accumulated


def _accumulated_states_match(state_a: Dict[str, Any], state_b: Dict[str, Any], config: Dict[str, Any]) -> bool:
    """Tolerance-based world-state comparison for split-depth.

    Two accumulated states count as "not yet diverged" when every categorical
    field agrees exactly and every actor's continuous state sits within the
    metric's own tolerances: position within τ (node_similarity threshold,
    default 2.0 m) and speed within the 1.0 m/s speed-similarity floor.
    Exact equality on 2-dp-rounded values would make every real prediction
    "split" at depth 1 against the factual baseline, collapsing S_split into
    a constant. Heading is deliberately
    excluded: in a 2D rollout a heading change with no positional or speed
    consequence within τ is below the instrument's resolution.
    """
    for field in ("event_type", "action_type", "intervention_type"):
        if _safe_str(state_a.get(field)) != _safe_str(state_b.get(field)):
            return False
    world_a = state_a.get("world_snapshot", {})
    world_b = state_b.get("world_snapshot", {})
    if set(world_a.keys()) != set(world_b.keys()):
        return False

    tol_cfg = config.get("layer3", {}).get("split_tolerance", {})
    tol_pos = tol_cfg.get("position_m")
    if tol_pos is None:
        tol_pos = config.get("node_similarity", {}).get("thresholds", {}).get("position_distance_m", 2.0)
    tol_pos = float(tol_pos)
    tol_speed = float(tol_cfg.get("speed_mps", 1.0))

    for aid, actor_a in world_a.items():
        actor_b = world_b[aid]
        for field in ("entity_type", "role_type", "state"):
            if _safe_str(actor_a.get(field)) != _safe_str(actor_b.get(field)):
                return False
        ax, ay = _safe_float_or_none(actor_a.get("x")), _safe_float_or_none(actor_a.get("y"))
        bx, by = _safe_float_or_none(actor_b.get("x")), _safe_float_or_none(actor_b.get("y"))
        if (ax is None) != (bx is None) or (ay is None) != (by is None):
            return False
        if ax is not None and ay is not None and bx is not None and by is not None:
            if math.hypot(ax - bx, ay - by) > tol_pos:
                return False
        a_speed = _safe_float_or_none(actor_a.get("speed"))
        b_speed = _safe_float_or_none(actor_b.get("speed"))
        if (a_speed is None) != (b_speed is None):
            return False
        if a_speed is not None and b_speed is not None and abs(a_speed - b_speed) > tol_speed:
            return False
    return True


def _branch_terminal_type(branch: Dict[str, Any]) -> str:
    chain = branch.get("chain", [])
    if not chain:
        return "none"
    return _safe_str(chain[-1].get("event_type"))


def _split_similarity(gt_split: int, pred_split: int, config: Dict[str, Any]) -> float:
    # Section 5.3: S_split = max(0, 1 - |T_split(b_i) - T_split(b̂_j)| / D_max)
    d_max = int(config.get("bounds", {}).get("D_max", 6))
    if d_max <= 0:
        return 1.0 if gt_split == pred_split else 0.0
    return max(0.0, 1.0 - abs(gt_split - pred_split) / d_max)


def _terminal_similarity(gt_term: str, pred_term: str, terminal_types: set) -> float:
    # Section 5.3: S_term = 1 iff BOTH branches terminate AND terminal types match.
    # Two non-terminal branches sharing a last event_type score 0.
    if gt_term in terminal_types and pred_term in terminal_types and gt_term == pred_term:
        return 1.0
    return 0.0


def _node_similarity(gt_node: Dict[str, Any], pred_node: Dict[str, Any], ontology: Dict[str, Any], config: Dict[str, Any]) -> float:
    """Section 3: Sim(Ni, NÌ‚i) = w·Sim_cat + (1âˆ'w)·Sim_cont, w = layer1.cat_weight.

    Default w = 0.5 is an equal split; the weight exists so
    the parameter-perturbation study (Section VI.3) can vary it via config.
    If a node violates schema or ontology constraints â†’ Sim = 0.
    """
    # Section 3.3: If event_type mismatch â†’ ontology constraint violation â†’ Sim = 0
    if gt_node.get("event_type") != pred_node.get("event_type"):
        return 0.0

    sim_cat = _sim_cat(gt_node, pred_node, ontology)
    sim_cont = _sim_cont(gt_node, pred_node, config, ontology)

    w = float(config.get("layer1", {}).get("cat_weight", 0.5))
    return w * sim_cat + (1.0 - w) * sim_cont


def _sim_cat(gt_node: Dict[str, Any], pred_node: Dict[str, Any], ontology: Dict[str, Any]) -> float:
    """Section 3.1: Sim_cat = P_cov Ã— P_role

    P_cov = #matched required slots / #required slots (or 1 if 0 required)
    P_role = #correct required role assignments / #required roles (or 1 if 0 required)
    """
    event_type = gt_node.get("event_type")
    event_constraints = ontology.get("event_constraints", {})
    required_roles_spec = ontology.get("required_roles", {})
    terminal_events = set(ontology.get("terminal_event_types", []))

    # Determine required slots for this event_type
    required_slots: List[str] = ["event_type"]
    ec = event_constraints.get(event_type, {})
    if ec.get("requires_action_type"):
        required_slots.append("action_type")
    if event_type in terminal_events:
        required_slots.append("terminal")

    # P_cov: required slots must be present AND match GT value
    if not required_slots:
        p_cov = 1.0
    else:
        covered = sum(
            1 for s in required_slots
            if pred_node.get(s) is not None and pred_node.get(s) == gt_node.get(s)
        )
        p_cov = covered / len(required_slots)

    # P_role: a required role assignment is correct iff the role
    # is held by the same actor IDs as in the reference and every holder has
    # an allowed entity_type. Holders are compared as ID sets (a collision may
    # have several impacted actors); partial overlap earns the Jaccard ratio.
    # Checking only the entity_type would let the wrong collision partner
    # earn full categorical credit.
    role_spec = required_roles_spec.get(event_type, {})
    if not role_spec:
        p_role = 1.0
    else:
        gt_actors = gt_node.get("actors", []) or gt_node.get("actors_update", []) or []
        pred_actors = pred_node.get("actors", []) or pred_node.get("actors_update", []) or []
        credit = 0.0
        for role_name, role_cfg in role_spec.items():
            allowed = set(role_cfg.get("allowed_entity_types", []))
            gt_ids = {str(a.get("id", "")) for a in gt_actors if a.get("role_type") == role_name}
            pred_holders = [a for a in pred_actors if a.get("role_type") == role_name]
            pred_ids = {str(a.get("id", "")) for a in pred_holders}
            if not gt_ids or not pred_ids:
                continue
            if any(a.get("entity_type") not in allowed for a in pred_holders):
                continue
            credit += len(gt_ids & pred_ids) / len(gt_ids | pred_ids)
        p_role = credit / len(role_spec)

    return p_cov * p_role


def _sim_cont(gt_node: Dict[str, Any], pred_node: Dict[str, Any], config: Dict[str, Any], ontology: Optional[Dict[str, Any]] = None) -> float:
    """Section 3.2: Sim_cont = (1/|F_valid|) Î£ Sim_f

    Supported fields: Sim_IoU (spatial), Sim_speed.
    Despawned actors: skip physics scoring, only check state agreement.
    Hallucinated pred actors: penalized with 0.0 per defined field.
    If |F_valid| = 0 â†’ Sim_cont = 1.
    """
    gt_actors = gt_node.get("actors", []) or gt_node.get("actors_update", []) or []
    pred_actors = pred_node.get("actors", []) or pred_node.get("actors_update", []) or []
    pred_map: Dict[str, Dict[str, Any]] = {}
    for a in pred_actors:
        aid = a.get("id")
        if aid is not None:
            pred_map[str(aid)] = a

    # Derive despawn states from ontology
    despawn_states: set = set()
    if ontology:
        lifecycle = ontology.get("actor_lifecycle_states", {})
        for state_name, state_cfg in lifecycle.items():
            if isinstance(state_cfg, dict) and state_cfg.get("removes_from_world", False):
                despawn_states.add(state_name)
    if not despawn_states:
        despawn_states = {"despawned", "exited", "out_of_scope"}

    gt_ids = {str(a.get("id", "")) for a in gt_actors}
    field_scores: List[float] = []
    bbox_fmt = config.get("bbox_format", "cx_cy_w_h")
    # Default τ matches ctms_config.yaml (node_similarity.thresholds.position_distance_m).
    dist_thr = float(config.get("node_similarity", {}).get("thresholds", {}).get("position_distance_m", 2.0))

    # ---- Score GT actors (missing pred â†’ 0, matched â†’ compute) ----
    for gt_actor in gt_actors:
        gt_id = str(gt_actor.get("id", ""))
        pred_actor = pred_map.get(gt_id)
        gt_state = gt_actor.get("state")

        # If GT actor is despawned, skip physics â€” only check state agreement
        if gt_state in despawn_states:
            if pred_actor is not None and pred_actor.get("state") == gt_state:
                field_scores.append(1.0)  # Correct despawn prediction
            else:
                field_scores.append(0.0)  # Missed or wrong state
            continue

        # Spatial overlap: Sim_IoU
        gt_has_pos = gt_actor.get("x") is not None and gt_actor.get("y") is not None
        if gt_has_pos:
            if pred_actor is not None and pred_actor.get("x") is not None and pred_actor.get("y") is not None:
                if gt_actor.get("bbox") and pred_actor.get("bbox"):
                    field_scores.append(_bbox_iou(gt_actor["bbox"], pred_actor["bbox"], bbox_fmt))
                else:
                    dx = float(gt_actor["x"]) - float(pred_actor["x"])
                    dy = float(gt_actor["y"]) - float(pred_actor["y"])
                    dist = math.sqrt(dx * dx + dy * dy)
                    field_scores.append(max(0.0, 1.0 - dist / dist_thr))
            else:
                field_scores.append(0.0)

        # Speed: Sim_speed = max(0, 1 - |v_pred - v_GT| / max(v_GT, 1.0))
        g_speed = _recover_actor_speed(
            gt_actor, None, 1.0,
            {"derive_speed_from_velocity": True, "derive_speed_from_position_delta": False},
        )
        if g_speed is not None:
            if pred_actor is not None:
                p_speed = _recover_actor_speed(
                    pred_actor, None, 1.0,
                    {"derive_speed_from_velocity": True, "derive_speed_from_position_delta": False},
                )
                if p_speed is not None:
                    field_scores.append(max(0.0, 1.0 - abs(p_speed - g_speed) / max(g_speed, 1.0)))
                else:
                    field_scores.append(0.0)
            else:
                field_scores.append(0.0)

    # ---- Penalize hallucinated pred actors (not in GT) ----
    for pred_actor in pred_actors:
        pred_id = str(pred_actor.get("id", ""))
        if pred_id in gt_ids:
            continue  # Already scored above
        # Each hallucinated actor injects 0.0 for its defined fields
        if pred_actor.get("x") is not None and pred_actor.get("y") is not None:
            field_scores.append(0.0)  # Spurious position
        p_speed = _recover_actor_speed(
            pred_actor, None, 1.0,
            {"derive_speed_from_velocity": True, "derive_speed_from_position_delta": False},
        )
        if p_speed is not None:
            field_scores.append(0.0)  # Spurious speed

    # Section 3.2: |F_valid| = 0 â†’ Sim_cont = 1
    if not field_scores:
        return 1.0
    return sum(field_scores) / len(field_scores)


def _bbox_iou(box_a: List[float], box_b: List[float], bbox_format: str) -> float:
    ax1, ay1, ax2, ay2 = _bbox_to_xyxy(box_a, bbox_format)
    bx1, by1, bx2, by2 = _bbox_to_xyxy(box_b, bbox_format)
    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)
    inter_w = max(0.0, inter_x2 - inter_x1)
    inter_h = max(0.0, inter_y2 - inter_y1)
    inter_area = inter_w * inter_h
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter_area
    if union <= 0:
        return 0.0
    return inter_area / union


def _bbox_to_xyxy(box: List[float], bbox_format: str) -> Tuple[float, float, float, float]:
    if bbox_format == "cx_cy_w_h":
        cx, cy, w, h = [float(x) for x in box]
        return cx - w / 2.0, cy - h / 2.0, cx + w / 2.0, cy + h / 2.0
    if bbox_format == "xmin_ymin_xmax_ymax":
        x1, y1, x2, y2 = [float(x) for x in box]
        return x1, y1, x2, y2
    raise ValueError(f"Unsupported bbox_format={bbox_format}")
