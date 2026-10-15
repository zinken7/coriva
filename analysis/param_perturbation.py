"""Parameter-perturbation check (dissertation 5.4, outline VI.3).

The metric's free parameters are defaults, not fitted constants: gamma
(Layer 2 depth discount), layer1.cat_weight (categorical/continuous split)
and the pairwise weights (lambda, mu, nu). This module perturbs each around
its default, re-ranks the synthetic fleet with the same degradation seeds,
and reports Kendall tau against the default-config ranking. Leaderboard
order must survive; absolute scores may move.

Usage:
    from analysis.param_perturbation import run_param_perturbation
    out = run_param_perturbation(gts, env)
    print(out["min_tau"], out["variants"])
"""

import copy
from typing import Any, Dict, List, Optional

from analysis.rq3a import fleet_scores, kendall_tau, _order_by_score

# One dotted config key -> values to try around the default.
DEFAULT_GRID: Dict[str, List[float]] = {
    "layer2.gamma": [0.8, 0.95],
    "layer1.cat_weight": [0.3, 0.7],
    "pairwise_weights.lambda_chain": [0.6, 0.8],
    "pairwise_weights.mu_split": [0.1, 0.3],
    "pairwise_weights.nu_term": [0.05, 0.2],
}


def _with_param(env: Dict[str, Any], dotted_key: str, value: float) -> Dict[str, Any]:
    """Copy of env whose config has dotted_key set to value; the original env
    is never mutated."""
    env2 = dict(env)
    env2["config"] = copy.deepcopy(env["config"])
    node = env2["config"]
    keys = dotted_key.split(".")
    for k in keys[:-1]:
        node = node.setdefault(k, {})
    node[keys[-1]] = value
    return env2


def run_param_perturbation(gts: List[Dict[str, Any]], env: Dict[str, Any],
                           grid: Optional[Dict[str, List[float]]] = None,
                           seeds: int = 3) -> Dict[str, Any]:
    """For every (param, value) in the grid, rank the fleet under the modified
    config and compare against the default-config ranking at the same
    degradation seed, so the config change is the only difference."""
    grid = DEFAULT_GRID if grid is None else grid
    default_orders = {s: _order_by_score(fleet_scores(gts, env, s))
                      for s in range(seeds)}

    variants = []
    for param, values in grid.items():
        for value in values:
            env_v = _with_param(env, param, value)
            taus = [kendall_tau(_order_by_score(fleet_scores(gts, env_v, s)),
                                default_orders[s])
                    for s in range(seeds)]
            variants.append({"param": param, "value": value,
                             "min_tau": min(taus),
                             "mean_tau": sum(taus) / len(taus)})

    return {"variants": variants,
            "min_tau": min(v["min_tau"] for v in variants),
            "seeds": seeds, "n_scenarios": len(gts)}
