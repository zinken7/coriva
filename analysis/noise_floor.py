"""Metric noise floor delta_noise (outline VI.2b, dissertation 3.6.2).

The twin's fidelity error is the instrument's error. To turn it into a bound
on score interpretation, ground-truth trajectories are perturbed with
displacements resampled from the empirical fidelity-error distribution (per
actor and depth, random direction), a fixed prediction set is re-scored
against each perturbed ground truth, and delta_noise is the standard
deviation of the resulting mean scores. Leaderboard differences smaller than
delta_noise must not be presented as ranking claims.

Usage:
    from analysis.noise_floor import estimate_noise_floor
    out = estimate_noise_floor(gts, epsilons=[...], env=env)
    print(out["delta_noise"])
"""

import copy
import math
import random
import statistics
from typing import Any, Dict, List, Optional

from analysis.rq3a import score_f1


def perturb_gt(gt: Dict[str, Any], epsilons: List[float],
               rng: random.Random) -> Dict[str, Any]:
    """Displace every actor state in every branch chain by a vector whose
    magnitude is resampled from `epsilons` (per actor and depth) and whose
    direction is uniform. Structure, ids and all non-spatial fields are
    unchanged."""
    pert = copy.deepcopy(gt)
    for branch in pert.get("branches", []):
        for node in branch.get("chain", []):
            for actor in node.get("actors_update", []):
                eps = rng.choice(epsilons)
                ang = rng.uniform(0.0, 2.0 * math.pi)
                actor["x"] = round(float(actor.get("x", 0.0))
                                   + eps * math.cos(ang), 3)
                actor["y"] = round(float(actor.get("y", 0.0))
                                   + eps * math.sin(ang), 3)
    return pert


def estimate_noise_floor(gts: List[Dict[str, Any]], epsilons: List[float],
                         env: Dict[str, Any], n_trials: int = 20,
                         seed: int = 0,
                         predictions: Optional[List[List[Dict[str, Any]]]] = None
                         ) -> Dict[str, Any]:
    """delta_noise over the Causal Foresight F1 axis. The fixed prediction set
    defaults to the oracle (each scenario's own unperturbed branches), so the
    trials measure exactly how far instrument noise alone can move the score
    of one and the same prediction."""
    if predictions is None:
        predictions = [copy.deepcopy(gt["branches"]) for gt in gts]

    rng = random.Random(seed)
    trial_scores: List[float] = []
    for _ in range(n_trials):
        scores = [score_f1(perturb_gt(gt, epsilons, rng), pred, env)
                  for gt, pred in zip(gts, predictions)]
        trial_scores.append(sum(scores) / len(scores))

    delta = statistics.stdev(trial_scores) if len(trial_scores) > 1 else 0.0
    return {"delta_noise": delta, "trial_scores": trial_scores,
            "n_trials": n_trials, "n_scenarios": len(gts),
            "epsilon_sample_size": len(epsilons)}
