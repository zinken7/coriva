"""Reporting statistics: bootstrap CIs over per-scenario scores, paired
bootstrap for model-to-model differences, and the minimum detectable difference
for the power analysis."""

import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


class TestBootstrapCI:
    def test_mean_lies_inside_interval(self):
        from analysis.bootstrap import bootstrap_ci
        scores = [0.2, 0.4, 0.6, 0.8, 0.5, 0.3, 0.7]
        out = bootstrap_ci(scores, n_boot=2000, seed=0)
        assert out["lo"] <= out["mean"] <= out["hi"]

    def test_constant_scores_give_zero_width(self):
        from analysis.bootstrap import bootstrap_ci
        out = bootstrap_ci([0.5] * 10, n_boot=500, seed=0)
        assert out["lo"] == pytest.approx(0.5)
        assert out["hi"] == pytest.approx(0.5)

    def test_deterministic_given_seed(self):
        from analysis.bootstrap import bootstrap_ci
        s = [0.1, 0.9, 0.4, 0.6]
        assert bootstrap_ci(s, n_boot=500, seed=7) == \
            bootstrap_ci(s, n_boot=500, seed=7)


class TestPairedBootstrap:
    def test_clear_difference_is_significant(self):
        from analysis.bootstrap import paired_bootstrap_diff
        a = [0.7, 0.8, 0.75, 0.72, 0.78, 0.74, 0.76, 0.73]
        b = [x - 0.3 for x in a]
        out = paired_bootstrap_diff(a, b, n_boot=2000, seed=0)
        assert out["mean_diff"] == pytest.approx(0.3)
        assert out["lo"] > 0.0
        assert out["significant"] is True

    def test_identical_scores_not_significant(self):
        from analysis.bootstrap import paired_bootstrap_diff
        a = [0.5, 0.6, 0.4, 0.55]
        out = paired_bootstrap_diff(a, list(a), n_boot=500, seed=0)
        assert out["mean_diff"] == pytest.approx(0.0)
        assert out["significant"] is False

    def test_mismatched_lengths_raise(self):
        from analysis.bootstrap import paired_bootstrap_diff
        with pytest.raises(ValueError):
            paired_bootstrap_diff([0.1, 0.2], [0.1], n_boot=10, seed=0)


class TestPowerAnalysis:
    def test_textbook_value(self):
        """sigma=1, n=100, alpha=.05, power=.8 ->
        (1.95996 + 0.84162) / 10 = 0.28016."""
        from analysis.power_analysis import min_detectable_difference
        assert min_detectable_difference(1.0, 100) == pytest.approx(0.28016,
                                                                    abs=1e-4)

    def test_quadrupling_n_halves_mdd(self):
        from analysis.power_analysis import min_detectable_difference
        assert min_detectable_difference(0.4, 600) == pytest.approx(
            2 * min_detectable_difference(0.4, 2400))
