"""Per-axis power analysis.

Before the main run, pilot variance fixes the minimum difference the benchmark
can resolve for each primary coherence axis. The same calculation may be used
for secondary reference axes, but those results remain diagnostic.
"""

import math
from statistics import NormalDist


def min_detectable_difference(std_diff: float, n: int, alpha: float = 0.05,
                              power: float = 0.8) -> float:
    """Minimum detectable mean paired difference for a two-sided test:
    (z_{1-alpha/2} + z_{power}) * std_diff / sqrt(n)."""
    z = NormalDist().inv_cdf
    return (z(1 - alpha / 2) + z(power)) * std_diff / math.sqrt(n)
