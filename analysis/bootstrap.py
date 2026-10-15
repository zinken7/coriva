"""Bootstrap statistics for leaderboard reporting (dissertation 5.3).

Scores carry 95% bootstrap confidence intervals over scenarios, and
model-to-model differences are tested by paired bootstrap over the same
scenarios. Percentile intervals; deterministic given a seed.
"""

import random
import statistics
from typing import Any, Dict, List


def _percentile(sorted_vals: List[float], q: float) -> float:
    """Linear-interpolation percentile on a pre-sorted list, q in [0, 1]."""
    if len(sorted_vals) == 1:
        return sorted_vals[0]
    pos = q * (len(sorted_vals) - 1)
    lo = int(pos)
    hi = min(lo + 1, len(sorted_vals) - 1)
    frac = pos - lo
    return sorted_vals[lo] * (1 - frac) + sorted_vals[hi] * frac


def bootstrap_ci(scores: List[float], n_boot: int = 10_000,
                 alpha: float = 0.05, seed: int = 0) -> Dict[str, float]:
    """Percentile bootstrap CI for the mean of per-scenario scores."""
    rng = random.Random(seed)
    n = len(scores)
    means = sorted(statistics.fmean(rng.choices(scores, k=n))
                   for _ in range(n_boot))
    return {"mean": statistics.fmean(scores),
            "lo": _percentile(means, alpha / 2),
            "hi": _percentile(means, 1 - alpha / 2)}


def paired_bootstrap_diff(a: List[float], b: List[float],
                          n_boot: int = 10_000, alpha: float = 0.05,
                          seed: int = 0) -> Dict[str, Any]:
    """Paired bootstrap over per-scenario score differences a[i] - b[i].
    `significant` means the (1 - alpha) percentile interval excludes zero."""
    if len(a) != len(b):
        raise ValueError(f"paired scores must align: {len(a)} vs {len(b)}")
    diffs = [x - y for x, y in zip(a, b)]
    rng = random.Random(seed)
    n = len(diffs)
    means = sorted(statistics.fmean(rng.choices(diffs, k=n))
                   for _ in range(n_boot))
    lo = _percentile(means, alpha / 2)
    hi = _percentile(means, 1 - alpha / 2)
    return {"mean_diff": statistics.fmean(diffs), "lo": lo, "hi": hi,
            "significant": lo > 0.0 or hi < 0.0}
