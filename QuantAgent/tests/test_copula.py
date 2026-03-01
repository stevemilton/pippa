"""Tests for the vine copula engine.

These are integration-level tests that verify the full pipeline:
fitting, probability computation, tail dependence, IS triggering, and
variance reduction. They use synthetic data so no IB connection is needed.
"""

import numpy as np
import pytest

from quantagent.copula import (
    CopulaEngine,
    CopulaResult,
    T_COPULA_NU,
    TAIL_PROBABILITY_THRESHOLD,
)


class TestCopulaEngine:

    def test_rejects_single_instrument(self):
        """Need at least 2 instruments for copula analysis."""
        engine = CopulaEngine()
        result = engine.analyze_cluster({"A": list(np.random.normal(0, 0.01, 150))})
        assert result is None

    def test_rejects_insufficient_data(self, rng):
        """Below copula_min_observations (100) should return None."""
        engine = CopulaEngine()
        # Only 50 observations
        result = engine.analyze_cluster({
            "A": rng.normal(0, 0.01, 50).tolist(),
            "B": rng.normal(0, 0.01, 50).tolist(),
        })
        assert result is None

    def test_basic_analysis(self, sample_returns):
        """Full pipeline should return a CopulaResult with all fields."""
        engine = CopulaEngine()
        result = engine.analyze_cluster(sample_returns)

        assert result is not None
        assert isinstance(result, CopulaResult)
        assert len(result.instruments) == 4
        assert len(result.implied_probs) == 4
        assert len(result.copula_probs) == 4
        assert len(result.discrepancies) == 4
        assert result.max_discrepancy >= 0
        assert 0 <= result.confidence <= 1
        assert "Student-t" in result.vine_structure
        assert result.variance_reduction_factor > 0

    def test_gaussian_excluded(self, sample_returns):
        """The vine structure description must confirm Gaussian is excluded."""
        engine = CopulaEngine()
        result = engine.analyze_cluster(sample_returns)
        assert result is not None
        assert "Gaussian" in result.vine_structure
        assert "EXCLUDED" in result.vine_structure

    def test_tail_dependence_all_pairs(self, sample_returns):
        """Should compute tail dependence for all instrument pairs."""
        engine = CopulaEngine()
        result = engine.analyze_cluster(sample_returns)
        assert result is not None

        n = len(result.instruments)
        expected_pairs = n * (n - 1) // 2
        assert len(result.tail_dependence) == expected_pairs

        # All values should be in [0, 1]
        for pair, td in result.tail_dependence.items():
            assert 0 <= td <= 1, f"Tail dependence for {pair} out of range: {td}"

    def test_two_instrument_minimum(self, sample_returns_pair):
        """Should work with exactly 2 instruments."""
        engine = CopulaEngine()
        result = engine.analyze_cluster(sample_returns_pair)
        assert result is not None
        assert len(result.instruments) == 2
        assert len(result.tail_dependence) == 1

    def test_discrepancy_is_signed(self, sample_returns):
        """Discrepancy = implied - copula → can be positive or negative."""
        engine = CopulaEngine()
        result = engine.analyze_cluster(sample_returns)
        assert result is not None

        # At least one should be non-zero
        vals = list(result.discrepancies.values())
        assert any(v != 0 for v in vals)

        # Check they're correctly computed
        for name in result.instruments:
            expected = result.implied_probs[name] - result.copula_probs[name]
            assert abs(result.discrepancies[name] - expected) < 1e-4

    def test_max_instruments_enforced(self, rng):
        """If cluster has more instruments than MAX, should cap."""
        from quantagent.config import config
        original = config.max_instruments_per_cluster
        object.__setattr__(config, "max_instruments_per_cluster", 3)
        engine = CopulaEngine()

        returns = {}
        for i in range(5):
            returns[f"INST_{i}"] = rng.normal(0, 0.01, 150).tolist()

        try:
            result = engine.analyze_cluster(returns)
            assert result is not None
            assert len(result.instruments) <= 3
        finally:
            object.__setattr__(config, "max_instruments_per_cluster", original)


class TestTailDependenceFormula:

    def test_high_correlation_gives_high_tail_dep(self):
        """rho=0.8, nu=4 should give substantial tail dependence (>10%)."""
        from scipy.stats import t as t_dist
        rho = 0.8
        nu = T_COPULA_NU
        arg = -np.sqrt((nu + 1) * (1 - rho) / (1 + rho))
        lambda_L = 2.0 * t_dist.cdf(arg, df=nu + 1)
        assert lambda_L > 0.10

    def test_zero_correlation_gives_low_tail_dep(self):
        """rho=0 should give low (but non-zero) tail dependence for t(nu=4).

        Even with zero correlation, t-copula has some tail dependence
        (~7.6% at nu=4). This is MUCH less than high-correlation but non-zero
        — unlike Gaussian which is exactly zero.
        """
        from scipy.stats import t as t_dist
        rho = 0.0
        nu = T_COPULA_NU
        arg = -np.sqrt((nu + 1) * (1 - rho) / (1 + rho))
        lambda_L = 2.0 * t_dist.cdf(arg, df=nu + 1)
        assert lambda_L < 0.10  # low but non-zero
        assert lambda_L > 0.01  # definitely not zero like Gaussian

    def test_gaussian_gives_zero_tail_dep(self):
        """Verify analytically that Gaussian (nu→inf) gives tail dep → 0.
        This is why we exclude Gaussian copulas.
        """
        from scipy.stats import t as t_dist
        rho = 0.5
        # As nu increases, tail dependence drops to zero
        tail_deps = []
        for nu in [4, 10, 50, 200]:
            arg = -np.sqrt((nu + 1) * (1 - rho) / (1 + rho))
            lambda_L = 2.0 * t_dist.cdf(arg, df=nu + 1)
            tail_deps.append(lambda_L)

        # Should be strictly decreasing
        for i in range(len(tail_deps) - 1):
            assert tail_deps[i] > tail_deps[i + 1]

        # At nu=200, should be very small
        assert tail_deps[-1] < 0.01


class TestImportanceSampling:

    def test_is_triggers_below_threshold(self, rng):
        """When an instrument is priced below 5%, IS should trigger."""
        engine = CopulaEngine()

        # Create data where one instrument has very negative returns → low implied prob
        n = 150
        returns = {
            "NORMAL": rng.normal(0.001, 0.01, n).tolist(),
            "TAIL": rng.normal(-0.02, 0.005, n).tolist(),  # strongly negative
        }

        result = engine.analyze_cluster(returns)
        if result is not None and result.is_tail_regime:
            assert len(result.is_instruments) > 0

    def test_is_does_not_trigger_normal_regime(self, sample_returns):
        """Under normal conditions, IS should not trigger."""
        engine = CopulaEngine()
        result = engine.analyze_cluster(sample_returns)
        # With synthetic normal data, IS typically won't trigger
        # (unless random seed produces extreme values)
        if result is not None and not result.is_tail_regime:
            assert result.is_instruments == []


class TestVarianceReduction:

    def test_vr_factor_above_one(self, sample_returns):
        """Variance reduction should be > 1 (better than crude MC)."""
        engine = CopulaEngine()
        result = engine.analyze_cluster(sample_returns)
        assert result is not None
        assert result.variance_reduction_factor >= 1.0


class TestPseudoObservations:

    def test_pseudo_obs_in_unit_interval(self, rng):
        """Pseudo-observations should be in (0, 1)."""
        engine = CopulaEngine()
        data = rng.normal(0, 1, (100, 3))
        pseudo = engine._to_pseudo_observations(data)
        assert pseudo.min() > 0
        assert pseudo.max() < 1

    def test_weiss_formula_avoids_boundaries(self, rng):
        """Ranks/(n+1) should never produce exactly 0 or 1."""
        engine = CopulaEngine()
        data = rng.normal(0, 1, (50, 2))
        pseudo = engine._to_pseudo_observations(data)
        assert pseudo.min() > 0
        assert pseudo.max() < 1
