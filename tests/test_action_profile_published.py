"""action_profile_published: the non-learning control on ALL the
information the prompt publishes.

The prompt prints (a) the scene's scripted hazard manoeuvre
for the L3 references (`build_pilot.declared_hazard_profile`) and (b) the
scene's own L4 intervention magnitudes (`build_pilot.declared_intervention_profile`,
CARLA: `scenario_config.json` env_behavior). `action_profile_kinematic` uses
neither (a documented design property; it is NOT modified), so beating it does
not show a model beats arithmetic on everything it was told. The second control
applies both records and is otherwise identical.
"""
import json
import os
import sys
from pathlib import Path

import pytest

ROOT = os.path.join(os.path.dirname(__file__), "..")
sys.path.insert(0, ROOT)

from analysis.rq3a import make_synthetic_gts  # noqa: E402


def _gt():
    return make_synthetic_gts(1)[0]


def _actor(node, aid):
    return next(a for a in node["actors_update"] if a["id"] == aid)


def _root_actor(gt, aid):
    return next(a for a in gt["observed_state"]["actors"] if a["id"] == aid)


def _l4(gt, itype="npc_sudden_brake", target="npc_1"):
    query = {"intervention_type": itype, "target_actor_id": target, "trigger_depth": 1}
    gt["l4_queries"] = [{"intervention_type": itype,
                         "counterfactual_chain": {"intervention": query, "chain": []}}]
    return query


class TestHazardProfileInL3:
    def test_hazard_actor_follows_declared_profile_in_every_l3_branch(self):
        from analysis.baselines import (action_profile_kinematic_text,
                                        action_profile_published_text)
        gt = _gt()
        # CARLA executed the script in the factual branch too; on a split whose
        # factual reference is the recording the factual branch is unscripted
        # (tests/test_research_review_20260917.py).
        gt["protocol"] = {"split": "carla"}
        hazard = {"actor_id": "npc_1", "accel": -4.0, "lat": 0.0}
        old = json.loads(action_profile_kinematic_text(gt))
        new = json.loads(action_profile_published_text(gt, hazard_profile=hazard))
        v0 = _root_actor(gt, "npc_1")["speed"]
        assert len(new["predicted_branches"]) == len(old["predicted_branches"]) == 3
        for b_old, b_new in zip(old["predicted_branches"], new["predicted_branches"]):
            assert b_new["intervention"] == b_old["intervention"]
            # old control: root velocity throughout
            assert all(_actor(n, "npc_1")["speed"] == pytest.approx(v0, abs=1e-6)
                       for n in b_old["chain"])
            # new control: -4 m/s^2 from T=0, i.e. -2 m/s per 0.5 s step until rest
            for k, n in enumerate(b_new["chain"], start=1):
                assert _actor(n, "npc_1")["speed"] == pytest.approx(max(0.0, v0 - 2.0 * k), abs=1e-6)
            # the ego is untouched by the hazard record
            for n_old, n_new in zip(b_old["chain"], b_new["chain"]):
                assert _actor(n_new, "ego") == _actor(n_old, "ego")

    def test_without_a_hazard_profile_l3_equals_the_existing_control(self):
        from analysis.baselines import (action_profile_kinematic_text,
                                        action_profile_published_text)
        gt = _gt()
        old = json.loads(action_profile_kinematic_text(gt))
        new = json.loads(action_profile_published_text(gt))
        assert [b["chain"] for b in new["predicted_branches"]] == \
            [b["chain"] for b in old["predicted_branches"]]
        assert [b["branch_id"] for b in new["predicted_branches"]] == \
            ["APP_do_ego_emergency_brake", "APP_do_ego_no_action", "APP_do_ego_swerve_left"]

    def test_hazard_profile_for_an_actor_outside_the_root_is_ignored(self):
        from analysis.baselines import (action_profile_kinematic_text,
                                        action_profile_published_text)
        gt = _gt()
        old = json.loads(action_profile_kinematic_text(gt))
        new = json.loads(action_profile_published_text(
            gt, hazard_profile={"actor_id": "ghost", "accel": -4.0, "lat": 0.0}))
        assert [b["chain"] for b in new["predicted_branches"]] == \
            [b["chain"] for b in old["predicted_branches"]]

    def test_hazard_profile_is_not_applied_in_l4(self):
        # The prompt scripts the hazard manoeuvre for the L3 branches only; in
        # L4 the named target follows the intervention and every other actor
        # keeps its root velocity.
        from analysis.baselines import action_profile_published_text
        gt = _gt()
        gt["protocol"] = {"split": "carla"}
        gt["observed_state"]["actors"].append({
            "id": "npc_2", "entity_type": "vehicle", "role_type": "adjacent_vehicle",
            "x": 30.0, "y": 4.0, "vx": 10.0, "vy": 0.0, "speed": 10.0})
        _l4(gt)
        new = json.loads(action_profile_published_text(
            gt, hazard_profile={"actor_id": "npc_2", "accel": -4.0, "lat": 0.0}))
        cf = new["predicted_counterfactuals"][0]
        assert all(_actor(n, "npc_2")["speed"] == pytest.approx(10.0) for n in cf["treated_chain"])
        assert all(_actor(n, "npc_2")["speed"] < 10.0 for b in new["predicted_branches"]
                   for n in b["chain"])


class TestInterventionProfileInL4:
    def test_l4_target_follows_declared_magnitude_not_env_profiles(self):
        from analysis.baselines import (action_profile_kinematic_text,
                                        action_profile_published_text)
        from simulation.kinematic_twin import ENV_PROFILES
        gt = _gt()
        query = _l4(gt)
        profiles = {"npc_sudden_brake": {"accel": -6.0, "lat": 0.0,
                                         "source": "scene configuration executed by the simulator"}}
        old = json.loads(action_profile_kinematic_text(gt))["predicted_counterfactuals"][0]
        new = json.loads(action_profile_published_text(
            gt, intervention_profiles=profiles))["predicted_counterfactuals"][0]
        assert new["intervention"] == old["intervention"] == query
        v0 = _root_actor(gt, "npc_1")["speed"]
        assert ENV_PROFILES["npc_sudden_brake"]["accel"] == -8.0
        assert _actor(old["treated_chain"][0], "npc_1")["speed"] == pytest.approx(v0 - 4.0, abs=1e-6)
        assert _actor(new["treated_chain"][0], "npc_1")["speed"] == pytest.approx(v0 - 3.0, abs=1e-6)
        # fixed ego policy in both
        assert _actor(new["treated_chain"][0], "ego")["speed"] == pytest.approx(_root_actor(gt, "ego")["speed"])

    def test_declared_lateral_profile_is_applied_as_given(self):
        from analysis.baselines import action_profile_published_text
        gt = _gt()
        gt["observed_state"]["actors"].append({
            "id": "ped", "entity_type": "pedestrian", "role_type": "crossing_actor",
            "x": 15.0, "y": 4.0, "vx": 0.0, "vy": 0.0, "speed": 0.0})
        _l4(gt, "pedestrian_enters_road", "ped")
        profiles = {"pedestrian_enters_road": {"accel": 0.0, "lat": -2.0, "source": "scene configuration"}}
        cf = json.loads(action_profile_published_text(
            gt, intervention_profiles=profiles))["predicted_counterfactuals"][0]
        first = _actor(cf["treated_chain"][0], "ped")
        assert first["vy"] == pytest.approx(-2.0) and first["y"] == pytest.approx(3.0)

    @pytest.mark.parametrize("profiles", [None, {}, {"npc_sudden_brake": {"accel": None, "lat": None,
                                                                          "source": "no published profile"}}])
    def test_missing_profile_falls_back_to_the_existing_control(self, profiles):
        from analysis.baselines import (action_profile_kinematic_text,
                                        action_profile_published_text)
        gt = _gt()
        _l4(gt)
        old = json.loads(action_profile_kinematic_text(gt))["predicted_counterfactuals"]
        new = json.loads(action_profile_published_text(
            gt, intervention_profiles=profiles))["predicted_counterfactuals"]
        assert new == old

    def test_road_entry_fallback_keeps_side_resolution(self):
        from analysis.baselines import (action_profile_kinematic_text,
                                        action_profile_published_text)
        for side in (-4.0, 4.0):
            gt = _gt()
            gt["observed_state"]["actors"].append({
                "id": "ped", "entity_type": "pedestrian", "role_type": "crossing_actor",
                "x": 15.0, "y": side, "vx": 0.0, "vy": 0.0, "speed": 0.0})
            _l4(gt, "pedestrian_enters_road", "ped")
            old = json.loads(action_profile_kinematic_text(gt))["predicted_counterfactuals"]
            new = json.loads(action_profile_published_text(gt))["predicted_counterfactuals"]
            assert new == old
            assert abs(_actor(new[0]["treated_chain"][0], "ped")["y"]) < abs(side)


def _add_l4_reference(d, hazard_profile=None, env_behavior=None):
    """Give a `make_protocol_dataset` scene the named-actor L4 reference its
    query.json already declares (npc_sudden_brake on npc_1, rolled by the twin
    as the nuScenes builder would), optionally a template hazard profile
    (nuScenes) or a CARLA-style scene configuration + manifest critical actor."""
    from simulation.kinematic_twin import build_twin_branch
    from simulation.nuscenes_admission import observed_velocity_baseline
    gt = json.loads((d / "ground_truth.json").read_text())
    query = {"intervention_type": "npc_sudden_brake", "target_actor_id": "npc_1", "trigger_depth": 1}
    ref = build_twin_branch("L4", False, "npc_sudden_brake", gt["observed_state"], 6, 0.5, 0.5,
                            intervention_meta=dict(query),
                            env_behavior=observed_velocity_baseline(gt["observed_state"]),
                            environment_intervention=dict(query))
    # Pair protocol: the reference also carries the control (same policy,
    # every actor at its root velocity, no intervention).
    from simulation.l4_control import control_intervention, control_query_entry, mark_pair_protocol
    control = build_twin_branch("L4_control", False, "do_ego_no_action", gt["observed_state"], 6, 0.5, 0.5,
                                intervention_meta=control_intervention(query),
                                env_behavior=observed_velocity_baseline(gt["observed_state"]))
    gt["l4_queries"] = [{"intervention_type": "npc_sudden_brake",
                         "counterfactual_chain": {"intervention": query, "chain": ref["chain"]},
                         "control_chain": control_query_entry(query, control["chain"], "nuscenes")}]
    mark_pair_protocol(gt.setdefault("protocol", {}), "nuscenes")
    if hazard_profile is not None:
        gt["scenario_template"]["hazard_profile"] = hazard_profile
    (d / "ground_truth.json").write_text(json.dumps(gt))
    if env_behavior is not None:
        (d / "scenario_config.json").write_text(json.dumps({"env_behavior": env_behavior}))
        manifest = json.loads((d / "input_manifest.json").read_text())
        manifest["critical_actor_id"] = "npc_1"
        (d / "input_manifest.json").write_text(json.dumps(manifest))


def _score(tmp_path, base, split, backend):
    from analysis.evaluation_campaign import dataset_inventory, score_cell, write_json, write_campaign_identity
    from analysis.run_model import run_model
    rows = dataset_inventory(base, 2, split)
    out = tmp_path / f"run_{backend}"
    out.mkdir()
    write_json(out / "frozen.json", {})
    cell = dict(split=split, backend=backend, condition="video", out_dir=str(out / backend))
    cfg = {"campaign_dir": str(out), "splits": {split: {"path": str(base)}}}
    write_campaign_identity(cfg, cell)
    stats = run_model(str(base), cell["out_dir"], backend=backend)
    assert stats["scenarios"] == 2 and not stats["failed"]
    score_cell(cfg, cell, {"datasets": {split: rows}})
    report = json.loads((Path(cell["out_dir"]) / "report.json").read_text())
    return Path(cell["out_dir"]), report


def _assert_all_l3_valid(report):
    assert len(report["per_sample"]) == 2
    for sample in report["per_sample"]:
        diag = sample["layer3"]["diagnostics"]
        assert diag["pred_unique_count"] == 3
        assert diag["pred_valid_count"] == diag["pred_unique_count"], diag
        assert sample["scores"]["Causal_Foresight_F1"] > 0.0


class TestRunnerAndScorer:
    def test_nuscenes_fixture_passes_the_scorer_gates_with_the_template_hazard(self, tmp_path):
        from test_evaluation_campaign import make_protocol_dataset
        base = make_protocol_dataset(tmp_path, "nuscenes")
        for d in sorted(base.glob("scenario_*")):
            _add_l4_reference(d, hazard_profile={"accel": -4.0, "lat": 0.0})
        out_dir, report = _score(tmp_path, base, "nuscenes", "action_profile_published")
        _assert_all_l3_valid(report)
        gt = json.loads((base / "scenario_1" / "ground_truth.json").read_text())
        v0 = _root_actor(gt, "npc_1")["speed"]
        l3 = json.loads((out_dir / "scenario_1" / "l3_pre.json").read_text())
        assert len(l3["predicted_branches"]) == 3
        # nuScenes: the factual branch's reference is the recording, so the
        # control scripts the hazard in the generated branches only.
        for b in l3["predicted_branches"]:
            expected = v0 if b["is_factual"] else v0 - 2.0
            assert _actor(b["chain"][0], "npc_1")["speed"] == pytest.approx(expected, abs=1e-6)
        assert sum(b["is_factual"] for b in l3["predicted_branches"]) == 1
        # nuScenes L4: the declared profile is the twin's own (-8), so the L4
        # chain equals the existing control's and the reference's
        old_dir, _ = _score(tmp_path, base, "nuscenes", "action_profile_kinematic")
        new_l4 = json.loads((out_dir / "scenario_1" / "l4_pre.json").read_text())
        assert new_l4 == json.loads((old_dir / "scenario_1" / "l4_pre.json").read_text())
        cf = new_l4["predicted_counterfactuals"][0]
        assert cf["intervention"]["intervention_type"] == "npc_sudden_brake"
        assert _actor(cf["treated_chain"][0], "npc_1")["speed"] == pytest.approx(v0 - 4.0, abs=1e-6)
        assert cf["treated_chain"] == gt["l4_queries"][0]["counterfactual_chain"]["chain"]
        assert all(s["scores"]["Counterfactual_Alignment_Score"] == pytest.approx(1.0)
                   for s in report["per_sample"])

    def test_carla_style_scene_configuration_drives_l4_and_l3_hazard(self, tmp_path):
        from test_evaluation_campaign import make_protocol_dataset
        base = make_protocol_dataset(tmp_path, "carla")
        for d in sorted(base.glob("scenario_*")):
            _add_l4_reference(d, env_behavior={"npc_1": {"accel": -6.0, "lat": 0.0}})
        out_dir, report = _score(tmp_path, base, "carla", "action_profile_published")
        _assert_all_l3_valid(report)
        old_dir, _ = _score(tmp_path, base, "carla", "action_profile_kinematic")
        for name in ("scenario_1", "scenario_2"):
            gt = json.loads((base / name / "ground_truth.json").read_text())
            v0 = _root_actor(gt, "npc_1")["speed"]
            new_l4 = json.loads((out_dir / name / "l4_pre.json").read_text())["predicted_counterfactuals"][0]
            old_l4 = json.loads((old_dir / name / "l4_pre.json").read_text())["predicted_counterfactuals"][0]
            assert new_l4["intervention"] == old_l4["intervention"]
            assert new_l4["intervention"]["intervention_type"] == "npc_sudden_brake"
            # -6 m/s^2 over the first 0.5 s step = -3 m/s (the old control: -8 -> -4 m/s)
            assert _actor(new_l4["treated_chain"][0], "npc_1")["speed"] == pytest.approx(v0 - 3.0, abs=1e-6)
            assert _actor(old_l4["treated_chain"][0], "npc_1")["speed"] == pytest.approx(v0 - 4.0, abs=1e-6)
            # the critical actor's env_behavior is the published L3 hazard manoeuvre
            new_l3 = json.loads((out_dir / name / "l3_pre.json").read_text())["predicted_branches"]
            old_l3 = json.loads((old_dir / name / "l3_pre.json").read_text())["predicted_branches"]
            assert len(new_l3) == 3
            for b_new, b_old in zip(new_l3, old_l3):
                assert _actor(b_new["chain"][0], "npc_1")["speed"] == pytest.approx(v0 - 3.0, abs=1e-6)
                assert _actor(b_old["chain"][0], "npc_1")["speed"] == pytest.approx(v0, abs=1e-6)

    def test_legacy_dataset_without_released_query_behaves_like_the_existing_control(self, tmp_path):
        from analysis.run_model import run_model
        gt_dir = tmp_path / "gt"
        for i, gt in enumerate(make_synthetic_gts(2), start=1):
            gt["scenario_id"] = f"nuscenes_{i:04d}"
            d = gt_dir / f"scenario_{i}"
            d.mkdir(parents=True)
            (d / "ground_truth.json").write_text(json.dumps(gt))
        a = run_model(str(gt_dir), str(tmp_path / "a"), backend="action_profile_published")
        b = run_model(str(gt_dir), str(tmp_path / "b"), backend="action_profile_kinematic")
        assert a["scenarios"] == b["scenarios"] == 2 and not a["failed"]
        for i in (1, 2):
            new = json.loads((tmp_path / "a" / f"scenario_{i}" / "l3_pre.json").read_text())
            old = json.loads((tmp_path / "b" / f"scenario_{i}" / "l3_pre.json").read_text())
            assert [x["chain"] for x in new["predicted_branches"]] == \
                [x["chain"] for x in old["predicted_branches"]]
            assert (tmp_path / "a" / f"scenario_{i}" / "l4_pre.json").read_text() == \
                (tmp_path / "b" / f"scenario_{i}" / "l4_pre.json").read_text()

    def test_backend_is_a_campaign_baseline_and_defines_every_axis(self, tmp_path):
        from analysis.evaluation_campaign import BASELINES
        from analysis.causal_coherence import run_coherence
        from analysis.run_model import run_model, GT_BACKENDS
        assert "action_profile_published" in BASELINES and "action_profile_published" in GT_BACKENDS
        from test_evaluation_campaign import make_protocol_dataset
        base = make_protocol_dataset(tmp_path, "nuscenes")
        for d in sorted(base.glob("scenario_*")):
            _add_l4_reference(d, hazard_profile={"accel": -4.0, "lat": 0.0})
        res = tmp_path / "res"
        run_model(str(base), str(res), backend="action_profile_published")
        report = run_coherence(str(base), pred_dir=str(res))
        for axis in ("monotonicity", "responsiveness", "action_direction", "speed_response"):
            assert report["axes"][axis]["n"] == 2, axis
