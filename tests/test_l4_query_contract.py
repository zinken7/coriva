"""The released L4 query must equal the reference it is scored against, or
every honest answer is unmatched (CAS 0)."""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from analysis.evaluation_campaign import l4_query_mismatches


def _gt(itype="npc_sudden_brake", target="npc_1", trigger=1, policy="do_ego_no_action"):
    return {"protocol": {"l4_ego_policy": policy},
            "l4_queries": [{"intervention_type": itype, "counterfactual_chain": {
                "intervention": {"intervention_type": itype, "target_actor_id": target, "trigger_depth": trigger},
                "chain": []}}]}


def _query(itype="npc_sudden_brake", target="npc_1", trigger=1, policy="do_ego_no_action"):
    return {"l3_ego_actions": [], "l4_ego_policy": policy,
            "l4_interventions": [{"intervention_type": itype, "target_actor_id": target, "trigger_depth": trigger}]}


def test_matching_query_passes():
    assert l4_query_mismatches(_gt(), _query(), {"critical_actor_id": "npc_1"}) == []


def test_every_field_of_the_query_tuple_is_compared():
    assert any("L4 query differs" in p for p in l4_query_mismatches(_gt(), _query(itype="npc_cut_in_right")))
    assert any("L4 query differs" in p for p in l4_query_mismatches(_gt(), _query(target="npc_2")))
    assert any("L4 query differs" in p for p in l4_query_mismatches(_gt(), _query(trigger=2)))
    assert any("ego policy" in p for p in l4_query_mismatches(_gt(), _query(policy="do_ego_comfortable_brake")))


def test_manifest_critical_actor_must_be_a_query_target():
    assert any("critical_actor_id" in p for p in l4_query_mismatches(_gt(), _query(), {"critical_actor_id": "npc_9"}))
    assert l4_query_mismatches(_gt(), _query(), {}) == []


def test_reference_without_queries_is_reported():
    assert any("no l4_queries" in p for p in l4_query_mismatches({"protocol": {}}, _query()))
