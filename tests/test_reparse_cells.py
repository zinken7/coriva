"""Re-parsing rewrites only prediction files, and only for a complete cell."""
import json
import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from scripts.reparse_cells import reparse_cell  # noqa: E402


@pytest.mark.parametrize("version,allow_rescore,complete", [
    ("ctms-scorer-1.2", False, False),
    ("ctms-scorer-1.2", True, True),
    ("ctms-scorer-1.3", True, True),
    ("ctms-scorer-0.9", True, False),
])
def test_rescoring_preserves_compatible_inference_version(tmp_path, version, allow_rescore, complete):
    from analysis.evaluation_campaign import check_cell
    from analysis.parse_prediction import PARSER_VERSION
    from simulation.protocol import PROTOCOL_VERSION
    rows = _cell(tmp_path, ["scenario_1"])
    manifest = dict(condition="video", backend="mock", protocol_version=PROTOCOL_VERSION,
                    scorer_version=version, parser_version=PARSER_VERSION,
                    scorer_input_track="primary", seed=0, decoding="greedy", temperature=0.0)
    path = tmp_path / "scenario_1" / "run_manifest.json"
    path.write_text(json.dumps(manifest))
    before = path.read_bytes()
    result = check_cell(tmp_path, rows, "video", backend="mock", allow_rescore=allow_rescore)
    assert result["complete"] is complete
    assert path.read_bytes() == before
    manifest["seed"] = 17
    path.write_text(json.dumps(manifest))
    assert not check_cell(tmp_path, rows, "video", backend="mock", allow_rescore=True)["complete"]


@pytest.mark.parametrize("version,expected", [("ctms-scorer-1.2", 0), ("ctms-scorer-0.9", 1)])
def test_runner_requires_explicit_compatible_rescoring(tmp_path, monkeypatch, version, expected):
    from run_benchmark import main
    from ctms.ctms import SCORER_VERSION
    gt, pred, report = [tmp_path / name for name in ("gt.json", "pred.json", "report.json")]
    gt.write_text(json.dumps({"scenarios": []}))
    pred.write_text(json.dumps({"scenarios": [], "scorer_version": version}))
    before = pred.read_bytes()
    argv = ["run_benchmark.py", "--gt", str(gt), "--pred", str(pred), "--output", str(report)]
    monkeypatch.setattr(sys, "argv", argv)
    assert main() == 1
    monkeypatch.setattr(sys, "argv", argv + ["--allow-rescore"])
    assert main() == expected
    assert pred.read_bytes() == before
    if expected == 0:
        saved = json.loads(report.read_text())
        assert saved["scorer_version"] == SCORER_VERSION
        assert json.loads(pred.read_text())["scorer_version"] == version

BROKEN = '{"predicted_branches":[{"branch_id":"B1","is_factual":true,"chain":[{"depth":1,' \
         '"actors_update":[{"id":"ego","x":1.0}]}]}],"predicted_counterfactuals":[]'


def _cell(tmp_path, names, text=BROKEN):
    rows = []
    for i, name in enumerate(names, 1):
        d = tmp_path / name; d.mkdir()
        (d / "final_response.txt").write_text(text)
        (d / "run_manifest.json").write_text("{}")
        (d / "l3_pre.json").write_text(json.dumps({"scenario_id": f"carla_{i}", "predicted_branches": []}))
        (d / "l4_pre.json").write_text(json.dumps({"scenario_id": f"carla_{i}", "predicted_counterfactuals": []}))
        rows.append({"name": name, "scenario_id": f"carla_{i}"})
    return rows


def test_reparse_rewrites_changed_predictions_and_nothing_else(tmp_path):
    rows = _cell(tmp_path, ["scenario_1", "scenario_2"])
    before = (tmp_path / "scenario_1" / "final_response.txt").read_bytes()
    res = reparse_cell(tmp_path, rows, write=True)
    assert res["written"] and res["changed"] == ["scenario_1", "scenario_2"] and res["parse_failures"] == 0
    l3 = json.loads((tmp_path / "scenario_1" / "l3_pre.json").read_text())
    assert l3["scenario_id"] == "carla_1" and l3["predicted_branches"][0]["branch_id"] == "B1"
    assert (tmp_path / "scenario_1" / "final_response.txt").read_bytes() == before
    assert (tmp_path / "scenario_1" / "run_manifest.json").read_text() == "{}"
    again = reparse_cell(tmp_path, rows, write=True)
    assert again["changed"] == [] and again["unchanged"] == 2


def test_dry_run_and_incomplete_cell_write_nothing(tmp_path):
    rows = _cell(tmp_path, ["scenario_1"])
    res = reparse_cell(tmp_path, rows, write=False)
    assert res["changed"] == ["scenario_1"] and not res["written"]
    assert json.loads((tmp_path / "scenario_1" / "l3_pre.json").read_text())["predicted_branches"] == []
    rows.append({"name": "scenario_9", "scenario_id": "carla_9"})
    res = reparse_cell(tmp_path, rows, write=True)
    assert res["missing"] == ["scenario_9"] and not res["written"]
    assert json.loads((tmp_path / "scenario_1" / "l3_pre.json").read_text())["predicted_branches"] == []


def test_unparsed_answer_counts_as_failure_and_keeps_empty_predictions(tmp_path):
    rows = _cell(tmp_path, ["scenario_1"], text='{"predicted_branches":[{"x":1.0+2.0}]')
    res = reparse_cell(tmp_path, rows, write=True)
    assert res["parse_failures"] == 1 and res["changed"] == []


def test_stale_score_detects_changed_predictions_and_old_versions(tmp_path):
    from analysis.evaluation_campaign import sha
    from ctms.ctms import L4_PAIR_PROTOCOL, SCORER_VERSION
    from scripts.reparse_cells import stale_score
    rows = _cell(tmp_path, ["scenario_1"])
    d = tmp_path / "scenario_1"
    files = {f"scenario_1/{n}": sha(d / n) for n in ("l3_pre.json", "l4_pre.json", "run_manifest.json")}
    (tmp_path / "score_provenance.json").write_text(json.dumps({"prediction_files": files}))
    (tmp_path / "report.json").write_text(json.dumps(
        {"scorer_version": SCORER_VERSION, "l4_pair_protocol": L4_PAIR_PROTOCOL}))
    from analysis.causal_coherence import COHERENCE_VERSION
    from analysis.intervention_pair import L4_PAIR_VERSION
    (tmp_path / "coherence.json").write_text(json.dumps({"coherence_version": COHERENCE_VERSION}))
    (tmp_path / "l4_pair.json").write_text(json.dumps({"l4_pair_version": L4_PAIR_VERSION}))
    assert stale_score(tmp_path, rows) is None
    reparse_cell(tmp_path, rows, write=True)
    assert "l3_pre.json differs" in stale_score(tmp_path, rows)
    (tmp_path / "report.json").write_text(json.dumps({"scorer_version": SCORER_VERSION}))
    assert "version differs" in stale_score(tmp_path, rows)


def test_stale_score_detects_an_older_coherence_or_pair_version(tmp_path):
    from analysis.causal_coherence import COHERENCE_VERSION
    from analysis.evaluation_campaign import sha
    from analysis.intervention_pair import L4_PAIR_VERSION
    from ctms.ctms import L4_PAIR_PROTOCOL, SCORER_VERSION
    from scripts.reparse_cells import stale_score
    rows = _cell(tmp_path, ["scenario_1"])
    d = tmp_path / "scenario_1"
    files = {f"scenario_1/{n}": sha(d / n) for n in ("l3_pre.json", "l4_pre.json", "run_manifest.json")}
    (tmp_path / "score_provenance.json").write_text(json.dumps({"prediction_files": files}))
    (tmp_path / "report.json").write_text(json.dumps({"scorer_version": SCORER_VERSION, "l4_pair_protocol": L4_PAIR_PROTOCOL}))
    (tmp_path / "coherence.json").write_text(json.dumps({"coherence_version": "coriva-coherence-1.1"}))
    (tmp_path / "l4_pair.json").write_text(json.dumps({"l4_pair_version": L4_PAIR_VERSION}))
    assert "coherence coriva-coherence-1.1 differs" in stale_score(tmp_path, rows)
    (tmp_path / "coherence.json").write_text(json.dumps({"coherence_version": COHERENCE_VERSION}))
    assert stale_score(tmp_path, rows) is None
    (tmp_path / "l4_pair.json").write_text(json.dumps({"l4_pair_version": "coriva-l4-pair-0.9"}))
    assert "l4 pair" in stale_score(tmp_path, rows)
