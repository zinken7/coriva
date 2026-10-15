"""Reference baselines (Chapter 5 of the dissertation) — GT-derived controls.

Each baseline removes one capability so leaderboard scores can be attributed:
  copy_factual            — factual chain as every branch and every CF chain
                            (tests that repetition cannot game set matching)
  kinematic_extrapolation — constant-velocity rollout from the root state,
                            ignoring the intervention (plain kinematics, no video)
  random_valid            — schema-valid random ontology rollouts (chance floor)

All are pure text producers like _oracle_text: gt dict -> JSON string that
parse_model_output can consume, so the whole downstream pipeline is identical.
"""

import json
import os
import sys

import pytest

ROOT = os.path.join(os.path.dirname(__file__), "..")
sys.path.insert(0, ROOT)

from analysis.rq3a import make_synthetic_gts


def _gt():
    return make_synthetic_gts(1)[0]


def _factual(gt):
    return next(b for b in gt["branches"] if b.get("is_factual"))


def _make_gt_dir(tmp_path, n=2):
    gt_dir = tmp_path / "gt"
    for i, gt in enumerate(make_synthetic_gts(n), start=1):
        gt["scenario_id"] = f"nuscenes_{i:04d}"
        d = gt_dir / f"scenario_{i}"
        d.mkdir(parents=True)
        with open(d / "ground_truth.json", "w") as f:
            json.dump(gt, f)
    return gt_dir


class TestRunnerWiring:
    def test_copy_factual_backend_writes_prediction_files(self, tmp_path):
        from analysis.run_model import run_model
        gt_dir = _make_gt_dir(tmp_path, 2)
        res = tmp_path / "res"
        stats = run_model(str(gt_dir), str(res), backend="copy_factual")
        assert stats["scenarios"] == 2 and not stats["failed"]
        l3 = json.load(open(res / "scenario_1" / "l3_pre.json"))
        gt = json.load(open(gt_dir / "scenario_1" / "ground_truth.json"))
        factual = _factual(gt)
        assert all(b["chain"] == factual["chain"]
                   for b in l3["predicted_branches"])


def _tiny_gt():
    """Hand-checkable scenario: ego at origin moving +x at 10 m/s, one lead
    npc ahead at 5 m/s, three factual steps of 0.5 s."""
    def node(depth, ego_x, npc_x):
        return {"depth": depth, "timestamp": 0.5 * depth,
                "event_type": "ego_brake", "action_type": "ego_emergency_brake",
                "actors_update": [
                    {"id": "ego", "entity_type": "ego_vehicle",
                     "role_type": "ego_agent", "x": ego_x, "y": 0.0,
                     "vx": 10.0, "vy": 0.0, "speed": 10.0},
                    {"id": "npc_1", "entity_type": "vehicle",
                     "role_type": "lead_vehicle", "x": npc_x, "y": 0.0,
                     "vx": 5.0, "vy": 0.0, "speed": 5.0}]}
    return {
        "scenario_id": "tiny_0001",
        "sample_interval_sec": 0.5,
        "observed_state": {"actors": [
            {"id": "ego", "entity_type": "ego_vehicle", "role_type": "ego_agent",
             "x": 0.0, "y": 0.0, "vx": 10.0, "vy": 0.0, "speed": 10.0},
            {"id": "npc_1", "entity_type": "vehicle", "role_type": "lead_vehicle",
             "x": 20.0, "y": 0.0, "vx": 5.0, "vy": 0.0, "speed": 5.0}]},
        "branches": [{"branch_id": "F", "is_factual": True, "intervention": None,
                      "chain": [node(1, 4.0, 22.0), node(2, 7.0, 24.0),
                                node(3, 9.0, 26.0)]}],
        "counterfactuals": [{"intervention_type": "do_ego_emergency_brake",
                             "counterfactual_chain": {
                                 "intervention": {"type": "do_ego_emergency_brake",
                                                  "target": "ego", "depth": 1},
                                 "chain": [node(1, 2.0, 22.0)]}}],
    }


class TestKinematicExtrapolation:
    def test_positions_advance_at_constant_root_velocity(self):
        from analysis.baselines import kinematic_extrapolation_text
        obj = json.loads(kinematic_extrapolation_text(_tiny_gt()))
        assert len(obj["predicted_branches"]) == 1
        chain = obj["predicted_branches"][0]["chain"]
        assert len(chain) == 3
        by_id = {a["id"]: a for a in chain[2]["actors_update"]}
        assert by_id["ego"]["x"] == 15.0 and by_id["ego"]["y"] == 0.0
        assert by_id["npc_1"]["x"] == 27.5
        assert by_id["ego"]["speed"] == 10.0

    def test_mirrors_factual_chain_timestamps(self):
        from analysis.baselines import kinematic_extrapolation_text
        gt = _tiny_gt()
        obj = json.loads(kinematic_extrapolation_text(gt))
        ts = [n["timestamp"] for n in obj["predicted_branches"][0]["chain"]]
        assert ts == [0.5, 1.0, 1.5]

    def test_ignores_intervention_same_chain_for_every_cf(self):
        from analysis.baselines import kinematic_extrapolation_text
        gt = _tiny_gt()
        obj = json.loads(kinematic_extrapolation_text(gt))
        assert len(obj["predicted_counterfactuals"]) == 1
        cf = obj["predicted_counterfactuals"][0]
        assert cf["treated_chain"] == obj["predicted_branches"][0]["chain"]
        assert cf["intervention"]["type"] == "do_ego_emergency_brake"

    def test_nodes_are_nonterminal_maintain_course(self):
        from analysis.baselines import kinematic_extrapolation_text
        obj = json.loads(kinematic_extrapolation_text(_tiny_gt()))
        for n in obj["predicted_branches"][0]["chain"]:
            assert n["event_type"] == "maintain_course"
            assert "terminal" not in n


class TestRunnerWiringKinematic:
    def test_kinematic_backend_is_registered(self, tmp_path):
        from analysis.run_model import run_model
        gt_dir = _make_gt_dir(tmp_path, 1)
        stats = run_model(str(gt_dir), str(tmp_path / "r"),
                          backend="kinematic_extrapolation")
        assert stats["scenarios"] == 1 and not stats["failed"]


class TestRandomValid:
    def test_deterministic_per_seed_and_varies_across_seeds(self):
        from analysis.baselines import random_valid_text
        gt = _tiny_gt()
        assert random_valid_text(gt, seed=0) == random_valid_text(gt, seed=0)
        assert random_valid_text(gt, seed=0) != random_valid_text(gt, seed=1)

    def test_branch_and_chain_counts_within_scenario_bounds(self):
        from analysis.baselines import random_valid_text
        obj = json.loads(random_valid_text(_tiny_gt(), seed=3))
        assert 2 <= len(obj["predicted_branches"]) <= 4
        for b in obj["predicted_branches"]:
            assert 3 <= len(b["chain"]) <= 6

    def test_actors_come_only_from_root_universe(self):
        from analysis.baselines import random_valid_text
        gt = _tiny_gt()
        universe = {a["id"] for a in gt["observed_state"]["actors"]}
        obj = json.loads(random_valid_text(gt, seed=7))
        for b in obj["predicted_branches"]:
            for n in b["chain"]:
                assert {a["id"] for a in n["actors_update"]} <= universe

    def test_every_branch_passes_the_real_schema(self):
        from analysis.baselines import random_valid_text
        from ctms.ctms import load_ctms_environment
        from jsonschema import Draft202012Validator
        env = load_ctms_environment("ctms/ontology.yaml", "ctms/ctms_config.yaml",
                                    "ctms/schema_branch.json",
                                    "ctms/schema_counterfactual.json")
        validator = Draft202012Validator(env["branch_schema"])
        obj = json.loads(random_valid_text(_tiny_gt(), seed=11))
        for b in obj["predicted_branches"]:
            errs = [e.message for e in validator.iter_errors(b)]
            assert not errs, errs

    def test_one_cf_chain_per_gt_intervention(self):
        from analysis.baselines import random_valid_text
        gt = _tiny_gt()
        obj = json.loads(random_valid_text(gt, seed=5))
        assert len(obj["predicted_counterfactuals"]) == len(gt["counterfactuals"])
        assert obj["predicted_counterfactuals"][0]["intervention"]["type"] == \
            "do_ego_emergency_brake"


class TestRunnerWiringRandom:
    def test_random_valid_backend_seed_controls_output(self, tmp_path):
        from analysis.run_model import run_model
        gt_dir = _make_gt_dir(tmp_path, 1)
        run_model(str(gt_dir), str(tmp_path / "a"), backend="random_valid", seed=1)
        run_model(str(gt_dir), str(tmp_path / "b"), backend="random_valid", seed=1)
        run_model(str(gt_dir), str(tmp_path / "c"), backend="random_valid", seed=2)
        a = open(tmp_path / "a" / "scenario_1" / "l3_pre.json").read()
        b = open(tmp_path / "b" / "scenario_1" / "l3_pre.json").read()
        c = open(tmp_path / "c" / "scenario_1" / "l3_pre.json").read()
        assert a == b
        assert a != c


class TestCopyFactual:
    def test_every_predicted_branch_is_the_factual_chain(self):
        from analysis.baselines import copy_factual_text
        gt = _gt()
        obj = json.loads(copy_factual_text(gt))
        factual = _factual(gt)
        assert len(obj["predicted_branches"]) == len(gt["branches"])
        for b in obj["predicted_branches"]:
            assert b["chain"] == factual["chain"]

    def test_branch_ids_are_distinct_like_a_real_submission(self):
        from analysis.baselines import copy_factual_text
        obj = json.loads(copy_factual_text(_gt()))
        ids = [b["branch_id"] for b in obj["predicted_branches"]]
        assert len(set(ids)) == len(ids)

    def test_counterfactuals_echo_factual_chain_per_gt_intervention(self):
        from analysis.baselines import copy_factual_text
        gt = _gt()
        obj = json.loads(copy_factual_text(gt))
        factual = _factual(gt)
        assert len(obj["predicted_counterfactuals"]) == len(gt["counterfactuals"])
        for pred_cf, gt_cf in zip(obj["predicted_counterfactuals"],
                                  gt["counterfactuals"]):
            assert pred_cf["treated_chain"] == factual["chain"]
            assert pred_cf["intervention"] == \
                gt_cf["counterfactual_chain"]["intervention"]


# ── action_profile_kinematic ─────────────────────────────────────────────
# The obvious strong non-learning control: apply the PUBLISHED ego action
# profile to the ego, keep every other actor at its root velocity, and for L4
# apply the published intervention profile to the named target under the
# fixed no-action ego policy. It reads only public inputs (root state, query
# action list, chain length/dt) plus the twin's published profiles; never the
# reference chains. It defines every coherence axis, so it is the floor each
# model must beat before "reasoning" is claimed.

def _ego_last(branch):
    return next(a for a in branch["chain"][-1]["actors_update"]
                if a["role_type"] == "ego_agent")


class TestActionProfileKinematic:
    def test_emits_one_distinct_branch_per_declared_action(self):
        from analysis.baselines import action_profile_kinematic_text
        gt = _gt()
        out = json.loads(action_profile_kinematic_text(gt))
        declared = [b["intervention"] for b in gt["branches"]]
        assert [b["intervention"] for b in out["predicted_branches"]] == declared
        assert sum(b["is_factual"] for b in out["predicted_branches"]) == 1
        chains = [json.dumps(b["chain"], sort_keys=True) for b in out["predicted_branches"]]
        assert len(set(chains)) == len(chains)

    def test_braking_ends_slower_and_swerve_left_ends_left_of_no_action(self):
        from analysis.baselines import action_profile_kinematic_text
        out = json.loads(action_profile_kinematic_text(_gt()))
        by_action = {b["intervention"]["intervention_type"]: b for b in out["predicted_branches"]}
        brake, none, left = (_ego_last(by_action[k]) for k in
                             ("do_ego_emergency_brake", "do_ego_no_action", "do_ego_swerve_left"))
        assert brake["speed"] < none["speed"] - 1.0
        assert brake["x"] < none["x"] - 1.0
        assert left["y"] < none["y"] - 1.0

    def test_other_actors_keep_root_velocity_not_reference_behaviour(self):
        # The synthetic GT's lead vehicle brakes (env_accels npc_1 = -2.0); the
        # baseline may not read that private behaviour.
        from analysis.baselines import action_profile_kinematic_text
        gt = _gt()
        out = json.loads(action_profile_kinematic_text(gt))
        npc_root = next(a for a in gt["observed_state"]["actors"] if a["id"] == "npc_1")
        for b in out["predicted_branches"]:
            for node in b["chain"]:
                npc = next(a for a in node["actors_update"] if a["id"] == "npc_1")
                assert npc["speed"] == pytest.approx(npc_root["speed"], abs=1e-6)

    def test_l4_applies_published_profile_to_named_target_under_no_action_ego(self):
        from analysis.baselines import action_profile_kinematic_text
        gt = _gt()
        query = {"intervention_type": "npc_sudden_brake", "target_actor_id": "npc_1",
                 "trigger_depth": 1}
        gt["l4_queries"] = [{"counterfactual_chain": {"intervention": query, "chain": []}}]
        out = json.loads(action_profile_kinematic_text(gt))
        assert len(out["predicted_counterfactuals"]) == 1
        cf = out["predicted_counterfactuals"][0]
        assert cf["intervention"] == query
        npc_root = next(a for a in gt["observed_state"]["actors"] if a["id"] == "npc_1")
        npc_last = next(a for a in cf["treated_chain"][-1]["actors_update"] if a["id"] == "npc_1")
        assert npc_last["speed"] < npc_root["speed"] - 1.0
        ego_root = next(a for a in gt["observed_state"]["actors"] if a["id"] == "ego")
        ego_first = next(a for a in cf["treated_chain"][0]["actors_update"] if a["id"] == "ego")
        assert ego_first["speed"] == pytest.approx(ego_root["speed"], abs=1e-6)

    def test_passes_every_l3_gate_against_the_scorer(self):
        from analysis.baselines import action_profile_kinematic_text
        from ctms.ctms import load_ctms_environment, score_layer3
        env = load_ctms_environment(os.path.join(ROOT, "ctms/ontology.yaml"),
                                    os.path.join(ROOT, "ctms/ctms_config.yaml"),
                                    os.path.join(ROOT, "ctms/schema_branch.json"),
                                    os.path.join(ROOT, "ctms/schema_counterfactual.json"))
        gt = _gt()
        pred = json.loads(action_profile_kinematic_text(gt))["predicted_branches"]
        r = score_layer3(gt["branches"], pred, env["ontology"], env["config"],
                         env["branch_schema"], root=gt["observed_state"])
        assert r["diagnostics"]["pred_valid_count"] == len(pred)
        assert r["F1_cf"] > 0.0

    def test_runner_backend_defines_every_coherence_axis(self, tmp_path):
        from analysis.run_model import run_model
        from analysis.causal_coherence import run_coherence
        gt_dir = _make_gt_dir(tmp_path, 2)
        res = tmp_path / "res"
        stats = run_model(str(gt_dir), str(res), backend="action_profile_kinematic")
        assert stats["scenarios"] == 2 and not stats["failed"]
        report = run_coherence(str(gt_dir), pred_dir=str(res))
        for axis in ("monotonicity", "responsiveness", "action_direction", "speed_response"):
            assert report["axes"][axis]["n"] == 2, axis
            assert report["axes"][axis]["mean"] == pytest.approx(1.0), axis


    def test_control_ordering_holds_at_the_same_depth_when_a_branch_ends_early(self):
        # The no-action branch reaches the lead within the first second
        # (contact at about 0.8 s, so the collision node is the 0.5 s sample)
        # while the emergency brake avoids it and stops later at x = 14 m.
        # Under coherence 1.1 the endpoint comparison read the collision-cut
        # branch as travelling less and scored the control 0.5; coherence 1.2
        # compares forward travel at the same depth, where the ordering the
        # profiles imply holds. Responsiveness still reads endpoints and the
        # other axes their own tolerances, so the control's MEASURED values
        # remain the floor on every axis; never assume its ceiling.
        from simulation.nuscenes_adapter import infer_scenario_spec
        from simulation.kinematic_twin import build_twin_gt
        from analysis.baselines import action_profile_kinematic_text
        from analysis.causal_coherence import pairs_from_prediction, coherence_from_pairs
        ego = {"id": "ego", "entity_type": "ego_vehicle", "role_type": "ego_agent",
               "x": 0, "y": 0, "vx": 15.0, "vy": 0, "speed": 15.0, "length": 4.0, "width": 1.8}
        lead = {"id": "lead", "entity_type": "vehicle", "role_type": "lead_vehicle",
                "x": 9.0, "y": 0.3, "vx": 9.0, "vy": 0, "speed": 9.0, "length": 4.5, "width": 1.9}
        root = {"depth": 0, "timestamp": 0.0, "event_type": "observed_state", "actors": [ego, lead]}
        gt = build_twin_gt(root, infer_scenario_spec("c", root))
        pred = json.loads(action_profile_kinematic_text(gt))
        ends = {b["intervention"]["intervention_type"]: b["chain"][-1] for b in pred["predicted_branches"]}
        assert ends["do_ego_no_action"]["event_type"] == "collision"
        assert ends["do_ego_emergency_brake"]["event_type"] != "collision"
        out = coherence_from_pairs(pairs_from_prediction(pred), gt["observed_state"])
        assert out["monotonicity"] == pytest.approx(1.0)
        assert out["responsiveness"] is not None

    def test_other_actors_keep_lateral_velocity_too(self):
        # `step_actor` overwrites vy with the lateral command, so a zero command
        # would silently freeze every non-ego actor's lateral motion.
        from analysis.baselines import action_profile_kinematic_text
        gt = _gt()
        npc = next(a for a in gt["observed_state"]["actors"] if a["id"] == "npc_1")
        npc["vy"] = 0.5; npc["speed"] = (npc["vx"] ** 2 + 0.25) ** 0.5
        out = json.loads(action_profile_kinematic_text(gt))
        for b in out["predicted_branches"]:
            ys = [next(a for a in n["actors_update"] if a["id"] == "npc_1") for n in b["chain"]]
            assert all(a["vy"] == pytest.approx(0.5) for a in ys)
            assert ys[0]["y"] == pytest.approx(npc["y"] + 0.25)

    def test_l4_road_entry_moves_toward_the_lane_from_either_side(self):
        from analysis.baselines import action_profile_kinematic_text
        for side in (-4.0, 4.0):
            gt = _gt()
            gt["observed_state"]["actors"].append({"id": "ped", "entity_type": "pedestrian",
                "role_type": "crossing_actor", "x": 15.0, "y": side, "vx": 0.0, "vy": 0.0,
                "speed": 0.0, "length": 0.6, "width": 0.6})
            query = {"intervention_type": "pedestrian_enters_road", "target_actor_id": "ped", "trigger_depth": 1}
            gt["l4_queries"] = [{"counterfactual_chain": {"intervention": query, "chain": []}}]
            cf = json.loads(action_profile_kinematic_text(gt))["predicted_counterfactuals"][0]
            first = next(a for a in cf["treated_chain"][0]["actors_update"] if a["id"] == "ped")
            assert abs(first["y"]) < abs(side), (side, first["y"])  # closer to y=0
