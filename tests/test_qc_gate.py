"""
QC gate.

A generated/built scenario GT is accepted only if every branch passes the CTMS
validity gates (schema + semantic with the scenario root universe + scene). This
is what guarantees scene-admissible scenarios at scale (reject-and-resample).
"""

import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


@pytest.fixture(scope="module")
def env():
    from ctms.ctms import load_ctms_environment
    return load_ctms_environment(
        "ctms/ontology.yaml", "ctms/ctms_config.yaml",
        "ctms/schema_branch.json", "ctms/schema_counterfactual.json",
    )


@pytest.fixture(scope="module")
def valid_gt():
    import json
    from simulation.build_ground_truth import build_scenario_gt, SCENARIOS
    raw = json.load(open("coriva_bench_examples.json"))
    return build_scenario_gt(raw[0], SCENARIOS[0])


def test_qc_accepts_valid_scenario_gt(valid_gt, env):
    from simulation.qc import qc_scenario_gt
    ok, reasons = qc_scenario_gt(valid_gt, env["ontology"], env["config"], env["branch_schema"])
    assert ok, reasons


def test_qc_rejects_gt_with_hallucinated_actor(valid_gt, env):
    import copy
    from simulation.qc import qc_scenario_gt
    bad = copy.deepcopy(valid_gt)
    # Inject an actor not present in the observed_state root universe.
    bad["branches"][0]["chain"][0].setdefault("actors_update", []).append(
        {"id": "ghost", "entity_type": "vehicle", "role_type": "lead_vehicle",
         "x": 0.0, "y": 0.0, "vx": 0.0, "vy": 0.0, "speed": 0.0}
    )
    ok, reasons = qc_scenario_gt(bad, env["ontology"], env["config"], env["branch_schema"])
    assert ok is False
    assert reasons, "rejection must report reasons"


def test_qc_rejects_gt_without_single_factual(valid_gt, env):
    import copy
    from simulation.qc import qc_scenario_gt
    bad = copy.deepcopy(valid_gt)
    for b in bad["branches"]:
        b["is_factual"] = False
    ok, _reasons = qc_scenario_gt(bad, env["ontology"], env["config"], env["branch_schema"])
    assert ok is False
