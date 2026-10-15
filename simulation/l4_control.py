"""The L4 control reference of the matched-pair protocol (Option B).

One place defines how a builder, the migration audit and the verifier
represent the control chain next to the treated chain in
``ground_truth.json -> l4_queries[i]`` (docs/l4_pair_protocol.md §2):

    counterfactual_chain: {intervention: {ι, τ, trigger_depth}, chain}      # treated (unchanged)
    control_chain:        {intervention: {intervention_type: "target_baseline", target_actor_id: τ,
                                          trigger_depth: 1, paired_with: ι},
                           chain, semantics: {ego_policy, target_baseline, target_baseline_rule, source}}

The baseline is what both builders already ran for admission: the named
target retains its baseline motion (CARLA: ``env_behavior[target] = {accel: 0,
lat: 0}`` in the retained ``_l4_control_steps``; nuScenes:
``observed_velocity_baseline(root)`` for every non-ego actor) while the ego
follows the same fixed policy. ``protocol.l4_pair_protocol`` marks a
reference that carries the control.
"""
from typing import Any, Dict, List

CONTROL_INTERVENTION_TYPE = "target_baseline"
L4_PAIR_PROTOCOL = "matched_control_treated"

CONTROL_SEMANTICS: Dict[str, Dict[str, str]] = {
    "carla": {
        "ego_policy": "do_ego_no_action",
        "target_baseline": "constant_speed_along_route",
        "target_baseline_rule": "target env_behavior {accel: 0, lat: 0}; every other actor as in treated",
        "source": "carla_paired_control_rollout",
    },
    "nuscenes": {
        "ego_policy": "do_ego_no_action",
        "target_baseline": "constant_t0_velocity",
        "target_baseline_rule": "observed_velocity_baseline(root) for every non-ego actor; no named intervention",
        "source": "deterministic_canonical_twin",
    },
}


def control_intervention(intervention: Dict[str, Any]) -> Dict[str, Any]:
    """The control chain's intervention block paired with `intervention`."""
    return {"intervention_type": CONTROL_INTERVENTION_TYPE,
            "target_actor_id": intervention.get("target_actor_id"),
            "trigger_depth": int(intervention.get("trigger_depth", 1)),
            "paired_with": intervention.get("intervention_type")}


def control_query_entry(intervention: Dict[str, Any], chain: List[Dict[str, Any]], split: str,
                        **extra_semantics: Any) -> Dict[str, Any]:
    """`l4_queries[i].control_chain` for the treated `intervention`."""
    semantics = dict(CONTROL_SEMANTICS[split])
    semantics.update({k: v for k, v in extra_semantics.items() if v is not None})
    return {"intervention": control_intervention(intervention), "chain": chain, "semantics": semantics}


def mark_pair_protocol(protocol: Dict[str, Any], split: str) -> Dict[str, Any]:
    protocol["l4_pair_protocol"] = L4_PAIR_PROTOCOL
    protocol["l4_control_reference_source"] = CONTROL_SEMANTICS[split]["source"]
    return protocol


def has_control_reference(query: Any) -> bool:
    return isinstance(query, dict) and isinstance(query.get("control_chain"), dict) \
        and isinstance(query["control_chain"].get("chain"), list) and bool(query["control_chain"]["chain"])


def validate_pair_reference(gt: Dict[str, Any], manifest: Dict[str, Any]) -> None:
    """Reject a legacy or mismatched pair before freezing a campaign/smoke.

    This checks representation and shared assumptions, not reference fidelity.
    The migration audit separately proves that the retained rollouts reproduce
    the existing treated references without selecting new scenarios.
    """
    for name, protocol in (("GT", gt.get("protocol") or {}), ("input manifest", manifest)):
        if protocol.get("l4_pair_protocol") != L4_PAIR_PROTOCOL:
            raise ValueError(f"L4 {name} lacks the matched-pair protocol marker; migrate the split first")
        if protocol.get("l4_ego_policy", "do_ego_no_action") != "do_ego_no_action":
            raise ValueError(f"L4 {name} has an unsupported ego policy")
    queries = gt.get("l4_queries")
    if not isinstance(queries, list) or len(queries) != 1:
        raise ValueError("L4 requires exactly one matched control/treated query per scene")
    q = queries[0]
    if not has_control_reference(q):
        raise ValueError("L4 reference has no control chain; migrate the split first")
    treated = q.get("counterfactual_chain")
    if not isinstance(treated, dict) or not isinstance(treated.get("chain"), list) or not treated["chain"]:
        raise ValueError("L4 reference has no treated chain")
    iv = treated.get("intervention")
    if not isinstance(iv, dict) or iv.get("trigger_depth") != 1 or iv.get("target_actor_id") in (None, "ego"):
        raise ValueError("L4 reference needs a named non-ego target at trigger depth 1")
    control = q["control_chain"]
    expected = control_intervention(iv)
    metadata = control.get("intervention")
    if not isinstance(metadata, dict) or any(metadata.get(k) != v for k, v in expected.items()):
        raise ValueError("L4 control reference intervention metadata differs from its treated query")
    semantics = control.get("semantics") or {}
    if semantics.get("ego_policy") != "do_ego_no_action":
        raise ValueError("L4 control reference has an incompatible ego policy")
