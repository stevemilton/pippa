"""Tests for the mathematical models (user-provided implementations).

These tests verify the math is correct — they do NOT modify the implementations.
"""

import numpy as np
import pytest

from quantagent.math_models import (
    simulate_binary_contract,
    brier_score,
    rare_event_IS,
    PredictionMarketParticleFilter,
    simulate_correlated_outcomes_t,
    stratified_binary_mc,
    PredictionMarketABM,
)


class TestSimulateBinaryContract:

    def test_atm_option_near_half(self):
        """An ATM binary contract should be near 0.5 probability."""
        np.random.seed(42)
        result = simulate_binary_contract(S0=100, K=100, mu=0.0, sigma=0.2, T=1.0)
        assert 0.35 < result["probability"] < 0.65
        assert result["std_error"] > 0
        assert result["ci_95"][0] < result["probability"] < result["ci_95"][1]

    def test_deep_itm(self):
        """Very low strike → probability near 1."""
        np.random.seed(42)
        result = simulate_binary_contract(S0=100, K=50, mu=0.05, sigma=0.2, T=1.0)
        assert result["probability"] > 0.90

    def test_deep_otm(self):
        """Very high strike → probability near 0."""
        np.random.seed(42)
        result = simulate_binary_contract(S0=100, K=200, mu=0.0, sigma=0.2, T=1.0)
        assert result["probability"] < 0.10


class TestBrierScore:

    def test_perfect_calibration(self):
        """Predict exactly the outcome → Brier = 0."""
        assert brier_score([1.0, 0.0, 1.0], [1.0, 0.0, 1.0]) == 0.0

    def test_worst_calibration(self):
        """Predict opposite of outcome → Brier = 1."""
        assert brier_score([1.0, 0.0], [0.0, 1.0]) == 1.0

    def test_coin_flip_calibration(self):
        """Predict 0.5 for everything → Brier = 0.25."""
        bs = brier_score([0.5, 0.5, 0.5, 0.5], [1, 0, 1, 0])
        assert abs(bs - 0.25) < 1e-10

    def test_good_calibration_below_target(self):
        """Predictions close to outcomes → Brier below 0.10 target."""
        preds = [0.9, 0.1, 0.8, 0.2, 0.7, 0.3]
        outcomes = [1, 0, 1, 0, 1, 0]
        bs = brier_score(preds, outcomes)
        assert bs < 0.10


class TestRareEventIS:

    def test_crash_probability_is_positive(self):
        np.random.seed(42)
        result = rare_event_IS(S0=100, K_crash=0.3, sigma=0.2, T=1/12, N_paths=100_000)
        assert result["p_IS"] > 0
        assert result["se_IS"] > 0
        assert result["se_IS"] < result["p_IS"]  # SE should be smaller than estimate

    def test_deeper_crash_is_rarer(self):
        """A 40% crash should be rarer than a 20% crash."""
        np.random.seed(42)
        r20 = rare_event_IS(S0=100, K_crash=0.2, sigma=0.2, T=1/12, N_paths=100_000)
        r40 = rare_event_IS(S0=100, K_crash=0.4, sigma=0.2, T=1/12, N_paths=100_000)
        assert r40["p_IS"] < r20["p_IS"]


class TestParticleFilter:

    def test_converges_to_true_value(self):
        """Particle filter should converge near the true probability."""
        np.random.seed(42)
        true_prob = 0.7
        pf = PredictionMarketParticleFilter(N_particles=5000, prior_prob=0.5)

        # Feed observations near the true prob
        for _ in range(50):
            obs = true_prob + np.random.normal(0, 0.02)
            pf.update(obs)

        est = pf.estimate()
        assert abs(est - true_prob) < 0.10

    def test_credible_interval_contains_estimate(self):
        np.random.seed(42)
        pf = PredictionMarketParticleFilter(N_particles=5000, prior_prob=0.5)
        for _ in range(20):
            pf.update(0.6)
        lo, hi = pf.credible_interval()
        est = pf.estimate()
        assert lo <= est <= hi


class TestSimulateCorrelatedOutcomesT:

    def test_output_shape(self):
        probs = [0.5, 0.5, 0.5]
        corr = np.eye(3)
        outcomes = simulate_correlated_outcomes_t(probs, corr, nu=4, N=1000)
        assert outcomes.shape == (1000, 3)

    def test_outcomes_are_binary(self):
        probs = [0.3, 0.7]
        corr = np.eye(2)
        outcomes = simulate_correlated_outcomes_t(probs, corr, nu=4, N=1000)
        assert set(outcomes.flatten()) <= {0, 1}

    def test_marginal_probs_roughly_correct(self):
        """Each marginal should be close to the input probability."""
        np.random.seed(42)
        probs = [0.3, 0.7]
        corr = np.array([[1.0, 0.5], [0.5, 1.0]])
        outcomes = simulate_correlated_outcomes_t(probs, corr, nu=4, N=100_000)
        for i, p in enumerate(probs):
            empirical = outcomes[:, i].mean()
            assert abs(empirical - p) < 0.03, f"Marginal {i}: expected {p}, got {empirical}"


class TestStratifiedBinaryMC:

    def test_returns_mean_and_se(self):
        np.random.seed(42)
        p, se = stratified_binary_mc(S0=100, K=100, sigma=0.2, T=1/12, J=20, N_total=100_000)
        assert 0.0 < p < 1.0
        assert se > 0

    def test_estimate_close_to_analytical(self):
        """Stratified estimate for ATM should be close to analytical value (~0.5)."""
        np.random.seed(42)
        # For ATM (S0=K), near-zero drift, short T, the probability should be near 0.5
        p_strat, se_strat = stratified_binary_mc(
            S0=100, K=100, sigma=0.2, T=1/12, J=20, N_total=100_000
        )
        # Should be close to 0.5 for ATM with zero drift
        assert 0.40 < p_strat < 0.60
        # SE should be positive
        assert se_strat > 0


class TestPredictionMarketABM:

    def test_price_stays_bounded(self):
        """ABM prices should remain in (0, 1)."""
        np.random.seed(42)
        abm = PredictionMarketABM(true_prob=0.7)
        prices = abm.run(n_steps=500)
        assert all(0 < p < 1 for p in prices)

    def test_price_converges_toward_truth(self):
        """Over many steps, price should drift toward true_prob."""
        np.random.seed(42)
        abm = PredictionMarketABM(true_prob=0.8, n_informed=20, n_noise=10)
        prices = abm.run(n_steps=2000)
        # Final price should be closer to 0.8 than the starting 0.5
        assert abs(prices[-1] - 0.8) < abs(0.5 - 0.8)

    def test_volume_increases(self):
        np.random.seed(42)
        abm = PredictionMarketABM(true_prob=0.6)
        abm.run(n_steps=100)
        assert abm.volume > 0
