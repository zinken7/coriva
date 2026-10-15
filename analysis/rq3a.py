"""
RQ3a — metric stability study for the CTMS Layer-3 structured score (F1_cf).

Pure-Python, deterministic, self-contained (no nuScenes / no VLM). Builds a
synthetic ground truth via the kinematic twin, derives a quality ladder of
predictions, and measures:

  * order-invariance      — permuting branch order must not change the score
  * format-invariance     — entity-type aliases must be canonicalized away
  * ranking stability     — bounded coordinate jitter must not flip the ranking

CLI: python3 analysis/rq3a.py   (prints a stability report)
"""

import copy
import math
import os
import random
import sys
from typing import Any, Dict, List, Tuple

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from simulation.kinematic_twin import build_twin_gt
from ctms.ctms import load_ctms_environment, score_layer3

# Entity aliases that ontology.synonyms maps back to canonical labels.
_ENTITY_ALIASES = {"vehicle": "car", "ego_vehicle": "ego", "pedestrian": "person", "bicycle": "bike"}


def make_study_gt() -> Dict[str, Any]:
    """A synthetic rear-end scenario GT (ego + lead vehicle), built via the twin."""
    root = {
        "depth": 0, "timestamp": 0.0, "event_type": "observed_state",
        "actors": [
            {"id": "ego", "entity_type": "ego_vehicle", "role_type": "ego_agent",
             "x": 0.0, "y": 0.0, "vx": 20.0, "vy": 0.0, "speed": 20.0},
            {"id": "npc_1", "entity_type": "vehicle", "role_type": "lead_vehicle",
             "x": 60.0, "y": 0.0, "vx": 20.0, "vy": 0.0, "speed": 20.0},
        ],
    }
    spec = {
        "scenario_id": "rq3a_study", "scenario_type": "rear_end_collision_risk",
        "n_steps": 6, "dt": 0.5, "sample_interval": 0.5, "env_accels": {"npc_1": -2.0},
        "branches": [
            {"branch_id": "Factual_AEB", "is_factual": True, "intervention_type": "do_ego_emergency_brake"},
            {"branch_id": "CF_NoAction", "is_factual": False, "intervention_type": "do_ego_no_action"},
            {"branch_id": "CF_Swerve", "is_factual": False, "intervention_type": "do_ego_swerve_left"},
        ],
    }
    return build_twin_gt(root, spec)


def score_f1(gt: Dict[str, Any], pred_branches: List[Dict[str, Any]], env: Dict[str, Any]) -> float:
    """Layer-3 Causal Foresight F1 of pred_branches against the GT."""
    r = score_layer3(gt["branches"], pred_branches, env["ontology"], env["config"],
                     env["branch_schema"], root=gt.get("observed_state"))
    return r["F1_cf"]


# ── perturbations ─────────────────────────────────────────────────────
def shuffle_branches(branches, rng: random.Random):
    b = copy.deepcopy(branches)
    rng.shuffle(b)
    return b


def reformat_entity_aliases(branches):
    """Replace canonical entity_type labels with synonym aliases (car, ego, ...)."""
    b = copy.deepcopy(branches)
    for br in b:
        for node in br.get("chain", []):
            for a in node.get("actors_update", []):
                et = a.get("entity_type")
                if et in _ENTITY_ALIASES:
                    a["entity_type"] = _ENTITY_ALIASES[et]
    return b


def jitter_branches(branches, sigma: float, rng: random.Random):
    """Add bounded Gaussian noise to positions/velocities; speed kept consistent."""
    b = copy.deepcopy(branches)
    for br in b:
        for node in br.get("chain", []):
            for a in node.get("actors_update", []):
                for k in ("x", "y", "vx", "vy"):
                    if a.get(k) is not None:
                        a[k] = round(a[k] + rng.gauss(0.0, sigma), 3)
                if a.get("vx") is not None and a.get("vy") is not None:
                    a["speed"] = round(math.hypot(a["vx"], a["vy"]), 3)
    return b


# ── quality ladder (structural degradation -> known ranking) ──────────
def _drop_last_node_of_nonfactual(branches):
    b = copy.deepcopy(branches)
    for br in b:
        if not br.get("is_factual") and len(br.get("chain", [])) > 1:
            br["chain"] = br["chain"][:-1]
            break
    return b


def _drop_n_nonfactual(branches, n):
    b = copy.deepcopy(branches)
    kept, dropped = [], 0
    for br in b:
        if not br.get("is_factual") and dropped < n:
            dropped += 1
            continue
        kept.append(br)
    return kept


def build_quality_ladder(gt) -> List[Tuple[str, List[Dict[str, Any]]]]:
    """Predictions of decreasing quality with a known ranking m0 > m1 > m2 > m3."""
    base = gt["branches"]
    return [
        ("m0_perfect", copy.deepcopy(base)),
        ("m1_truncated", _drop_last_node_of_nonfactual(base)),
        ("m2_missing_one", _drop_n_nonfactual(base, 1)),
        ("m3_missing_two", _drop_n_nonfactual(base, 2)),
    ]


# ── ranking metrics ───────────────────────────────────────────────────
def kendall_tau(order_a: List[str], order_b: List[str]) -> float:
    """Kendall rank correlation between two orderings of the same items."""
    pos = {name: i for i, name in enumerate(order_b)}
    items = [pos[name] for name in order_a]
    n = len(items)
    if n < 2:
        return 1.0
    concordant = discordant = 0
    for i in range(n):
        for j in range(i + 1, n):
            if items[i] < items[j]:
                concordant += 1
            else:
                discordant += 1
    return (concordant - discordant) / (n * (n - 1) / 2)


def _ranking(gt, ladder, env) -> List[str]:
    scored = [(name, score_f1(gt, preds, env)) for name, preds in ladder]
    scored.sort(key=lambda kv: kv[1], reverse=True)
    return [name for name, _ in scored]


def run_stability_study(gt, env, sigma: float = 0.25, seeds: int = 8) -> Dict[str, Any]:
    """Jitter each ladder model independently across `seeds` and check the
    ranking against the unperturbed baseline."""
    ladder = build_quality_ladder(gt)
    base_order = _ranking(gt, ladder, env)

    taus, inversions = [], 0
    for seed in range(seeds):
        rng = random.Random(seed)
        jittered = [(name, jitter_branches(preds, sigma, rng)) for name, preds in ladder]
        order = _ranking(gt, jittered, env)
        tau = kendall_tau(order, base_order)
        taus.append(tau)
        if order != base_order:
            inversions += 1

    return {
        "base_order": base_order,
        "base_scores": {name: score_f1(gt, preds, env) for name, preds in ladder},
        "sigma": sigma, "seeds": seeds,
        "min_kendall_tau": min(taus), "mean_kendall_tau": sum(taus) / len(taus),
        "rank_inversions": inversions,
    }


# ── dataset-scale fleet ranking stability ─────────────────────────────
def make_synthetic_gts(n: int) -> List[Dict[str, Any]]:
    """A deterministic set of distinct rear-end GTs (varying ego speed and gap)."""
    gts = []
    for i in range(n):
        ego_vx = 12.0 + 2.0 * (i % 7)
        gap = 40.0 + 5.0 * (i % 6)
        root = {
            "depth": 0, "timestamp": 0.0, "event_type": "observed_state",
            "actors": [
                {"id": "ego", "entity_type": "ego_vehicle", "role_type": "ego_agent",
                 "x": 0.0, "y": 0.0, "vx": ego_vx, "vy": 0.0, "speed": ego_vx},
                {"id": "npc_1", "entity_type": "vehicle", "role_type": "lead_vehicle",
                 "x": gap, "y": 0.0, "vx": ego_vx, "vy": 0.0, "speed": ego_vx},
            ],
        }
        spec = {
            "scenario_id": f"syn_{i:03d}", "scenario_type": "rear_end_collision_risk",
            "n_steps": 6, "dt": 0.5, "sample_interval": 0.5, "env_accels": {"npc_1": -2.0},
            "branches": [
                {"branch_id": "Factual_AEB", "is_factual": True, "intervention_type": "do_ego_emergency_brake"},
                {"branch_id": "CF_NoAction", "is_factual": False, "intervention_type": "do_ego_no_action"},
                {"branch_id": "CF_Swerve", "is_factual": False, "intervention_type": "do_ego_swerve_left"},
            ],
        }
        gts.append(build_twin_gt(root, spec))
    return gts


# A fixed fleet of "models": prediction policies of decreasing fidelity.
MODEL_POLICIES = {
    "perfect": lambda b, rng: copy.deepcopy(b),
    "mild": lambda b, rng: jitter_branches(b, 0.3, rng),
    "moderate": lambda b, rng: jitter_branches(_drop_last_node_of_nonfactual(b), 0.8, rng),
    "heavy": lambda b, rng: jitter_branches(_drop_n_nonfactual(b, 1), 1.2, rng),
}


def fleet_scores(gts, env, seed: int) -> Dict[str, float]:
    """Mean Layer-3 F1 of each model policy across all scenarios (seeded)."""
    rng = random.Random(seed)
    out: Dict[str, float] = {}
    for name, policy in MODEL_POLICIES.items():
        scores = [score_f1(gt, policy(gt["branches"], rng), env) for gt in gts]
        out[name] = sum(scores) / len(scores)
    return out


def _order_by_score(scores: Dict[str, float]) -> List[str]:
    return [name for name, _ in sorted(scores.items(), key=lambda kv: kv[1], reverse=True)]


def run_fleet_stability(gts, env, seeds: int = 10) -> Dict[str, Any]:
    """Rank the model fleet by mean score at seed 0, then check the ranking holds
    across `seeds` further random seeds (stochastic policies + jitter)."""
    base_scores = fleet_scores(gts, env, 0)
    base_order = _order_by_score(base_scores)
    taus, inversions = [], 0
    for s in range(1, seeds + 1):
        order = _order_by_score(fleet_scores(gts, env, s))
        taus.append(kendall_tau(order, base_order))
        if order != base_order:
            inversions += 1
    return {
        "n_scenarios": len(gts), "seeds": seeds,
        "base_order": base_order,
        "base_scores": {k: round(v, 4) for k, v in base_scores.items()},
        "min_kendall_tau": min(taus), "mean_kendall_tau": sum(taus) / len(taus),
        "rank_inversions": inversions,
    }


def _report(label, gts, env):
    r = run_fleet_stability(gts, env)
    print(f"\nRQ3a fleet ranking stability — {label} ({r['n_scenarios']} scenarios)")
    print("  base ranking :", r["base_order"])
    print("  mean scores  :", r["base_scores"])
    print(f"  Kendall tau  : min={r['min_kendall_tau']:.3f} mean={r['mean_kendall_tau']:.3f}")
    print(f"  rank inversions across {r['seeds']} seeds: {r['rank_inversions']}")
    return r


def main():
    env = load_ctms_environment("ctms/ontology.yaml", "ctms/ctms_config.yaml",
                                "ctms/schema_branch.json", "ctms/schema_counterfactual.json")
    _report("synthetic", make_synthetic_gts(12), env)

    # If real nuScenes twin GTs can be built, report on those too.
    if os.path.isdir(os.path.join("data", "nuscenes", "v1.0-mini")):
        try:
            from simulation.build_nuscenes_dataset import build_dataset
            import tempfile
            with tempfile.TemporaryDirectory() as td:
                res = build_dataset("data/nuscenes", "v1.0-mini", limit=20, out_dir=td, env=env)
                import json as _json
                gts = [_json.load(open(p)) for p in res["paths"]]
            if gts:
                _report("nuScenes-mini twin", gts, env)
        except Exception as e:  # noqa: BLE001 — best-effort real-data report
            print(f"\n(nuScenes report skipped: {e})")


if __name__ == "__main__":
    main()
