"""Shared validation and serialization for CORIVA observed inputs.

The release contract deliberately keeps the present state and the numeric
history as separate files.  Visual-only runs receive the former but never the
latter; the two history ablations receive both.  Builders use this module so
CARLA and nuScenes cannot silently emit different wire formats.
"""

from __future__ import annotations

import copy
import json
import math
import os
from typing import Any, Dict, Iterable, List, Sequence

from simulation.protocol import HISTORY_TIMESTAMPS, OBSERVED_FRAME_COUNT


STATE_FIELDS = ("id", "entity_type", "role_type", "x", "y", "vx", "vy",
                "speed", "length", "width")


def _finite_number(value: Any, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{field} must be numeric")
    value = float(value)
    if not math.isfinite(value):
        raise ValueError(f"{field} must be finite")
    return value


def validate_observed_state(root: Dict[str, Any]) -> Dict[str, Any]:
    if root.get("event_type") != "observed_state" or root.get("depth") != 0:
        raise ValueError("observed_state must be the depth-0 observed_state node")
    if float(root.get("timestamp", 0.0)) != 0.0:
        raise ValueError("observed_state timestamp must be exactly 0.0")
    actors = root.get("actors")
    if not isinstance(actors, list) or not actors:
        raise ValueError("observed_state actors must be a non-empty array")
    ids = []
    for actor in actors:
        for field in STATE_FIELDS:
            if field not in actor:
                raise ValueError(f"observed_state actor missing {field}")
        ids.append(str(actor["id"]))
        for field in ("x", "y", "vx", "vy", "speed", "length", "width"):
            _finite_number(actor[field], field)
    if len(ids) != len(set(ids)):
        raise ValueError("observed_state actor ids must be unique")
    return root


def history_record(timestamp_sec: float, actors: Iterable[Dict[str, Any]]) -> Dict[str, Any]:
    return {"timestamp_sec": round(float(timestamp_sec), 3),
            "actors": [copy.deepcopy(a) for a in actors]}


def _shared_actor(actor: Dict[str, Any]) -> Dict[str, Any]:
    return {field: actor.get(field) for field in STATE_FIELDS}


def validate_observed_history(history: Dict[str, Any], root: Dict[str, Any]) -> Dict[str, Any]:
    validate_observed_state(root)
    records = history.get("records")
    if not isinstance(records, list) or len(records) != OBSERVED_FRAME_COUNT:
        raise ValueError("observed_history must contain exactly eight records")
    got_ts = tuple(round(_finite_number(r.get("timestamp_sec"), "timestamp_sec"), 3)
                   for r in records)
    if got_ts != HISTORY_TIMESTAMPS:
        raise ValueError("observed_history timestamps must be -3.5 through 0.0")
    root_ids = [str(a["id"]) for a in root["actors"]]
    for index, record in enumerate(records):
        actors = record.get("actors")
        if not isinstance(actors, list):
            raise ValueError(f"history record {index} actors must be an array")
        ids = [str(a.get("id")) for a in actors]
        if ids != root_ids:
            raise ValueError(f"history record {index} actor order/universe differs from T0")
        for actor in actors:
            for field in STATE_FIELDS:
                if field not in actor:
                    raise ValueError(f"history actor missing {field}")
            if not isinstance(actor.get("visibility"), bool):
                raise ValueError("history actor visibility must be boolean")
            for field in ("x", "y", "vx", "vy", "speed", "length", "width"):
                _finite_number(actor[field], field)
    t0_by_id = {str(a["id"]): _shared_actor(a) for a in records[-1]["actors"]}
    root_by_id = {str(a["id"]): _shared_actor(a) for a in root["actors"]}
    if t0_by_id != root_by_id:
        raise ValueError("T0 history state is not exactly equal to observed_state")
    return history


def write_observed_artifacts(scenario_dir: str, root: Dict[str, Any],
                             history: Dict[str, Any]) -> None:
    validate_observed_history(history, root)
    with open(os.path.join(scenario_dir, "observed_state.json"), "w", encoding="utf-8") as f:
        json.dump(root, f, indent=2)
    with open(os.path.join(scenario_dir, "observed_history.json"), "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)


FACTUAL_REFERENCES = ("recorded_future", "executed_action_profile")


def factual_branch_contract(gt: Dict[str, Any]) -> Optional[Dict[str, str]]:
    """{factual_action, factual_reference} for the ground truth's factual
    branch, or None when the reference declares no factual branch.
    `factual_reference` is `recorded_future` for a nuScenes reference (the
    factual branch is the recording, `simulation.kinematic_twin.build_twin_gt`
    `factual_states`) and `executed_action_profile` for CARLA."""
    factual = next((b for b in gt.get("branches", []) if b.get("is_factual")), None)
    if factual is None:
        return None
    action = (factual.get("intervention") or {}).get("intervention_type")
    if not action:
        return None
    split = str((gt.get("protocol") or {}).get("split") or "").lower()
    reference = "executed_action_profile" if split == "carla" else "recorded_future"
    return {"factual_action": action, "factual_reference": reference}


def write_protocol_sidecars(scenario_dir: str, gt: Dict[str, Any],
                            ontology: Dict[str, Any]) -> Dict[str, Any]:
    """Write the model-visible ontology and explicit L3/L4 query metadata."""
    l3_actions = [(branch.get("intervention") or {}).get("intervention_type")
                  for branch in gt.get("branches", [])
                  if (branch.get("intervention") or {}).get("target_actor_id", "ego") == "ego"]
    l4 = []
    for item in gt.get("l4_queries", []):
        intervention = (item.get("counterfactual_chain") or {}).get("intervention") or {}
        l4.append(copy.deepcopy(intervention))
    query = {"l3_ego_actions": [action for action in l3_actions if action],
             "l4_interventions": l4,
             "l4_ego_policy": (gt.get("protocol") or {}).get(
                 "l4_ego_policy", "do_ego_no_action")}
    # The factual-branch contract: which
    # declared action is the factual branch and what its reference is. On
    # nuScenes the factual branch is the RECORDED continuation (every actor),
    # labelled by the band of the recorded ego acceleration, so the published
    # profile does not apply to it; on CARLA the simulator executed the
    # factual action with the published profile like every other branch.
    factual = factual_branch_contract(gt)
    if factual is not None:
        query.update(factual)
    with open(os.path.join(scenario_dir, "query.json"), "w", encoding="utf-8") as f:
        json.dump(query, f, indent=2)
    with open(os.path.join(scenario_dir, "ontology.json"), "w", encoding="utf-8") as f:
        json.dump(ontology, f, indent=2)
    return query


def load_observed_artifacts(scenario_dir: str) -> tuple[Dict[str, Any], Dict[str, Any]]:
    with open(os.path.join(scenario_dir, "observed_state.json"), encoding="utf-8") as f:
        root = json.load(f)
    with open(os.path.join(scenario_dir, "observed_history.json"), encoding="utf-8") as f:
        history = json.load(f)
    return root, validate_observed_history(history, root)
