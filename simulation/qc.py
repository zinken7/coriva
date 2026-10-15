"""
QC gate — accept a scenario GT only if every branch is scene-admissible.

Runs each branch of a built ground_truth through the CTMS validity gates
(schema + semantic, with the scenario observed_state root universe, + scene).
Used by the batch generation pipeline to reject-and-resample bad scenarios so
the 400-scenario set is guaranteed valid rather than merely produced.
"""

from typing import Any, Dict, List, Optional, Tuple

try:
    from ctms.ctms import (
        _pre_normalize_branch, canonicalize_branch, check_schema_validity,
        check_semantic_validity, check_scene_validity, check_kinematic_validity,
        apply_reference_gate_policy,
    )
except ImportError:  # when run as a script with ctms/ on the path
    from ctms import (  # type: ignore
        _pre_normalize_branch, canonicalize_branch, check_schema_validity,
        check_semantic_validity, check_scene_validity, check_kinematic_validity,
        apply_reference_gate_policy,
    )


def _root_universe(gt: Dict[str, Any]) -> Optional[Dict[str, str]]:
    root = gt.get("observed_state")
    if root is None:
        return None
    return {
        str(a.get("id")): a.get("entity_type")
        for a in (root.get("actors") or []) if a.get("id") is not None
    }


def qc_scenario_gt(gt: Dict[str, Any], ontology: Dict[str, Any], config: Dict[str, Any], branch_schema: Dict[str, Any]) -> Tuple[bool, List[Dict[str, Any]]]:
    """Return (accepted, reasons). Accepted iff exactly one factual branch and
    every branch passes schema + semantic + scene validity."""
    reasons: List[Dict[str, Any]] = []
    # A protocol reference (gt["protocol"]["protocol_version"], written by
    # both builders) must sit on the T0 + k*dt grid; a reference without a
    # protocol version (variable dt) is exempt. Same policy helper as
    # run_benchmark.
    config = apply_reference_gate_policy(config, gt)

    branches = gt.get("branches", [])
    factual = [b for b in branches if b.get("is_factual")]
    if len(factual) != 1:
        reasons.append({"reason": "factual_branch_count", "count": len(factual)})

    root_universe = _root_universe(gt)
    for b in branches:
        nb = _pre_normalize_branch(b, ontology)
        schema_ok, schema_msgs = check_schema_validity(nb, branch_schema)
        canon = canonicalize_branch(nb, ontology, config)
        semantic_ok, semantic_msgs = check_semantic_validity(canon, ontology, root_universe=root_universe)
        scene_ok, scene_msgs = check_scene_validity(
            canon, config, map_data=gt.get("scene_geometry"),
            root=gt.get("observed_state"))
        kinematic_ok, kinematic_msgs = check_kinematic_validity(
            canon, config, root=gt.get("observed_state"))
        if not (schema_ok and semantic_ok and scene_ok and kinematic_ok):
            reasons.append({
                "branch_id": b.get("branch_id"),
                "schema_msgs": schema_msgs,
                "semantic_msgs": semantic_msgs,
                "scene_msgs": scene_msgs,
                "kinematic_msgs": kinematic_msgs,
            })

    # L4 is a separate, named query set.  It is deliberately not included in
    # the L3 branch denominator, but its target must still be grounded in the
    # same root universe and exactly one reviewed query is required whenever a
    # protocol dataset declares one.
    l4_queries = gt.get("l4_queries", [])
    root_ids = set(root_universe or {})
    if len(l4_queries) > 1:
        reasons.append({"reason": "multiple_reviewed_l4_queries", "count": len(l4_queries)})
    for query in l4_queries:
        chain = query.get("counterfactual_chain", {}) if isinstance(query, dict) else {}
        intervention = chain.get("intervention", {}) if isinstance(chain, dict) else {}
        target = intervention.get("target_actor_id")
        if not target or str(target) not in root_ids:
            reasons.append({"reason": "l4_target_not_in_root", "target_actor_id": target})
        if target == "ego":
            reasons.append({"reason": "l4_named_query_targets_ego"})
        if not chain.get("chain"):
            reasons.append({"reason": "l4_query_empty_chain"})
        else:
            nb = _pre_normalize_branch(chain, ontology)
            canon = canonicalize_branch(nb, ontology, config)
            kin_ok, kin_msgs = check_kinematic_validity(
                canon, config, root=gt.get("observed_state"))
            if not kin_ok:
                reasons.append({"reason": "l4_kinematic_invalid", "messages": kin_msgs})
        # Matched-pair protocol: a reference that declares the pair must carry
        # a non-empty control chain for the same target that passes the same
        # kinematic check; a legacy reference without it is not rejected here
        # (campaign inventory refuses it).
        control = query.get("control_chain") if isinstance(query, dict) else None
        declares_pair = (gt.get("protocol") or {}).get("l4_pair_protocol") is not None
        if control is None and declares_pair:
            reasons.append({"reason": "l4_control_chain_missing"})
        elif isinstance(control, dict):
            c_interv = control.get("intervention") or {}
            if str(c_interv.get("target_actor_id")) != str(target):
                reasons.append({"reason": "l4_control_target_mismatch",
                                "control_target": c_interv.get("target_actor_id"), "target": target})
            if not control.get("chain"):
                reasons.append({"reason": "l4_control_empty_chain"})
            else:
                nb = _pre_normalize_branch(control, ontology)
                canon = canonicalize_branch(nb, ontology, config)
                kin_ok, kin_msgs = check_kinematic_validity(
                    canon, config, root=gt.get("observed_state"))
                if not kin_ok:
                    reasons.append({"reason": "l4_control_kinematic_invalid", "messages": kin_msgs})
        elif control is not None:
            reasons.append({"reason": "l4_control_chain_malformed"})

    return len(reasons) == 0, reasons
