"""Branch audit v2 contract tests and multi-rater statistic checks."""

import csv
import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from analysis.branch_audit import (  # noqa: E402
    build_audit_sheet, krippendorff_alpha, audit_summary)


# ── Krippendorff nominal α ──────────────────────────────────────────────

def test_alpha_perfect_agreement_is_one():
    units = [["y", "y"], ["n", "n"], ["y", "y"], ["n", "n"]]
    assert krippendorff_alpha(units) == pytest.approx(1.0)


def test_alpha_hand_worked_example():
    # 4 units, 2 raters, binary; one disagreement on unit 4.
    # Coincidence: o[y][y]=4, o[n][n]=2, o[y][n]=o[n][y]=1, n=8;
    # marginals n_y=5, n_n=3; D_o=(1/8)(2)=0.25;
    # D_e=(1/56)(2·15)=0.5357; α = 1 − 0.25/0.5357 = 0.5333.
    units = [["y", "y"], ["y", "y"], ["n", "n"], ["n", "y"]]
    assert krippendorff_alpha(units) == pytest.approx(0.5333, abs=1e-3)


def test_alpha_handles_missing_ratings():
    # A unit rated by only one annotator contributes no pairs; must not crash.
    units = [["y", "y"], ["n", None], ["y", "y"], ["n", "n"]]
    a = krippendorff_alpha(units)
    assert a == pytest.approx(1.0)  # every complete unit agrees


def test_alpha_below_threshold_when_noisy():
    units = [["y", "n"], ["n", "y"], ["y", "y"], ["n", "n"], ["y", "n"]]
    assert krippendorff_alpha(units) < 0.67


def test_alpha_three_raters():
    units = [["y", "y", "y"], ["n", "n", "n"], ["y", "y", "n"]]
    a = krippendorff_alpha(units)
    assert 0.0 < a < 1.0


# ── Sheet generation ─────────────────────────────────────────────────────

@pytest.fixture()
def sheet(tmp_path):
    out = tmp_path / "audit_sheet.csv"
    build_audit_sheet(["pilot30"], str(out), split_name="nuscenes",
                      n_raters=3, limit=5, seed=0)
    with open(out) as f:
        return list(csv.DictReader(f))


@pytest.mark.skip(reason="needs pilot fixture data that is not part of this release")
def test_sheet_has_scene_and_included_rows_only(sheet):
    kinds = {r["item_type"] for r in sheet}
    assert kinds == {"included_branch", "scene_context"}
    scenarios = {r["scenario_id"] for r in sheet}
    assert len(scenarios) == 5


def _gt_by_scenario_id():
    import glob
    import json
    out = {}
    for p in glob.glob("pilot30/scenario_*/ground_truth.json"):
        gt = json.load(open(p))
        out[gt["scenario_id"]] = gt
    return out


@pytest.mark.skip(reason="needs pilot fixture data that is not part of this release")
def test_sheet_included_rows_are_the_gt_branches(sheet):
    gts = _gt_by_scenario_id()
    sid = sheet[0]["scenario_id"]  # a scenario actually sampled into the sheet
    included = {r["item"] for r in sheet
                if r["scenario_id"] == sid and r["item_type"] == "included_branch"}
    assert included == {br["intervention"]["intervention_type"] for br in gts[sid]["branches"]}


@pytest.mark.skip(reason="needs pilot fixture data that is not part of this release")
def test_sheet_has_one_scene_question_per_scene_and_no_exclusions(sheet):
    from collections import Counter
    contexts = Counter(r['scenario_id'] for r in sheet if r['item_type']=='scene_context')
    assert set(contexts.values()) == {1}
    assert len(contexts) == 5
    assert not any('rightly excluded' in r['prompt'] for r in sheet)
    assert all('notes' in r and not r['notes'] for r in sheet)


@pytest.mark.skip(reason="needs pilot fixture data that is not part of this release")
def test_sheet_has_blank_rater_columns(sheet):
    for col in ("rater1", "rater2", "rater3"):
        assert col in sheet[0]
        assert all(r[col] == "" for r in sheet)


# ── Summary of a filled sheet ────────────────────────────────────────────

def _write_filled(path, rows):
    cols = ["scenario_id", "split", "item_type", "item", "rater1", "rater2", "rater3"]
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for r in rows:
            w.writerow(r)


def test_summary_computes_inclusion_rate_and_alpha(tmp_path):
    p = tmp_path / "filled.csv"
    rows = [
        {"scenario_id": "s1", "split": "nuscenes", "item_type": "included_branch",
         "item": "do_ego_emergency_brake", "rater1": "y", "rater2": "y", "rater3": "y"},
        {"scenario_id": "s1", "split": "nuscenes", "item_type": "included_branch",
         "item": "do_ego_no_action", "rater1": "y", "rater2": "y", "rater3": "n"},
        {"scenario_id": "s2", "split": "nuscenes", "item_type": "included_branch",
         "item": "do_ego_swerve_left", "rater1": "y", "rater2": "y", "rater3": "y"},
        {"scenario_id": "s2", "split": "nuscenes", "item_type": "included_branch",
         "item": "do_ego_panic_brake", "rater1": "n", "rater2": "n", "rater3": "y"},
        {"scenario_id": "s1", "split": "nuscenes", "item_type": "excluded_action",
         "item": "do_ego_accelerate_pass", "rater1": "y", "rater2": "y", "rater3": "y"},
    ]
    _write_filled(str(p), rows)
    out = audit_summary(str(p))
    assert out["n_included"] == 4
    assert out["inclusion_plausibility"] == pytest.approx(0.75)
    assert out["n_excluded"] == 1
    assert out["exclusion_correctness"] == pytest.approx(1.0)
    assert "alpha" in out
    assert out["passes_inclusion"] is False   # 0.75 < 0.90
    assert isinstance(out["passes_alpha"], bool)


def _contract_gt(actions=None, factual='do_ego_emergency_brake'):
    actions = actions or ['do_ego_emergency_brake','do_ego_no_action','do_ego_swerve_left']
    gt = dict(scenario_id='example',scenario_type='rear_end_collision_risk',branches=[
        dict(is_factual=a==factual,intervention=dict(intervention_type=a,target_actor_id='ego',trigger_depth=1))
        for a in actions])
    return gt,dict(l3_ego_actions=actions.copy())


def test_protocol_check_catches_agreeing_but_wrong_query_and_gt():
    from analysis.branch_audit import check_branch_contract
    gt,q=_contract_gt(['do_ego_emergency_brake','do_ego_no_action','do_ego_comfortable_brake'])
    result=check_branch_contract(gt,q,'carla')
    assert result['checks']['query_matches_references'] is True
    assert result['checks']['query_matches_protocol'] is False
    assert result['status']=='fail'


def test_real_factual_extra_is_allowed_but_arbitrary_extra_is_not():
    from analysis.branch_audit import check_branch_contract
    actions=['do_ego_emergency_brake','do_ego_no_action','do_ego_swerve_left','do_ego_comfortable_brake']
    gt,q=_contract_gt(actions,factual='do_ego_comfortable_brake')
    assert check_branch_contract(gt,q,'nuscenes')['status']=='pass'
    assert check_branch_contract(gt,q,'carla')['status']=='fail'
    gt['branches'][0]['is_factual']=True
    assert check_branch_contract(gt,q,'nuscenes')['status']=='fail'


def test_missing_and_duplicate_references_fail_contract():
    from analysis.branch_audit import check_branch_contract
    gt,q=_contract_gt();gt['branches'].pop()
    assert check_branch_contract(gt,q,'carla')['status']=='fail'
    gt,q=_contract_gt();gt['branches'].append(gt['branches'][1].copy())
    result=check_branch_contract(gt,q,'carla')
    assert result['status']=='fail' and not result['checks']['one_reference_per_action']


def test_missing_type_not_inferred_from_branch_list():
    from analysis.branch_audit import check_branch_contract
    gt,q=_contract_gt();gt.pop('scenario_type')
    result=check_branch_contract(gt,q,'nuscenes')
    assert result['status']=='not_checked'
    assert result['checks']['query_matches_references'] is True
    assert result['checks']['query_matches_protocol'] is None


def test_author_counts_separate_scene_and_branches_and_keep_uncertainty():
    from analysis.branch_audit import audit_rows,summarize_author_audit
    gt,_=_contract_gt();rows=audit_rows(gt,'carla')
    assert summarize_author_audit(rows)['status']=='awaiting_author_review'
    for row in rows:row.update(rater1='y',notes='Read the supplied states.')
    rows[0].update(rater1='u',notes='Target is obscured at T0.')
    result=summarize_author_audit(rows)
    assert result['status']=='reviewed_by_author'
    assert result['by_item_type']['scene_context']['counts']['u']==1
    assert result['by_item_type']['included_branch']['counts']['y']==3
    assert 'alpha' not in result and 'passes_inclusion' not in result
    rows[0]['item_type']='excluded_action'
    with pytest.raises(ValueError,match='Outdated'):
        summarize_author_audit(rows)
