"""
Mathematical techniques for QuantAgent.

Contains working implementations for:
- Binary contract simulation (Monte Carlo)
- Brier score calibration
- Rare event importance sampling
- Prediction market particle filter
- Correlated outcome simulation (t-copula)
- Stratified Monte Carlo
- Prediction market agent-based model
"""

import numpy as np
from scipy.special import expit, logit
from scipy.stats import t as t_dist, norm


def simulate_binary_contract(S0, K, mu, sigma, T, N_paths=100_000):
    Z = np.random.standard_normal(N_paths)
    S_T = S0 * np.exp((mu - 0.5 * sigma**2) * T + sigma * np.sqrt(T) * Z)
    payoffs = (S_T > K).astype(float)
    p_hat = payoffs.mean()
    se = np.sqrt(p_hat * (1 - p_hat) / N_paths)
    ci_lower = p_hat - 1.96 * se
    ci_upper = p_hat + 1.96 * se
    return {'probability': p_hat, 'std_error': se, 'ci_95': (ci_lower, ci_upper)}


def brier_score(predictions, outcomes):
    return np.mean((np.array(predictions) - np.array(outcomes))**2)


def rare_event_IS(S0, K_crash, sigma, T, N_paths=100_000):
    K = S0 * (1 - K_crash)
    mu_original = -0.5 * sigma**2
    log_threshold = np.log(K / S0)
    mu_tilt = log_threshold / T
    Z = np.random.standard_normal(N_paths)
    log_returns_tilted = mu_tilt * T + sigma * np.sqrt(T) * Z
    S_T_tilted = S0 * np.exp(log_returns_tilted)
    log_LR = (
        -0.5 * ((log_returns_tilted - mu_original * T) / (sigma * np.sqrt(T)))**2
        + 0.5 * ((log_returns_tilted - mu_tilt * T) / (sigma * np.sqrt(T)))**2
    )
    LR = np.exp(log_LR)
    payoffs = (S_T_tilted < K).astype(float)
    is_estimates = payoffs * LR
    p_IS = is_estimates.mean()
    se_IS = is_estimates.std() / np.sqrt(N_paths)
    return {'p_IS': p_IS, 'se_IS': se_IS}


class PredictionMarketParticleFilter:
    def __init__(self, N_particles=5000, prior_prob=0.5, process_vol=0.05, obs_noise=0.03):
        self.N = N_particles
        self.process_vol = process_vol
        self.obs_noise = obs_noise
        logit_prior = logit(prior_prob)
        self.logit_particles = logit_prior + np.random.normal(0, 0.5, N_particles)
        self.weights = np.ones(N_particles) / N_particles

    def update(self, observed_price):
        noise = np.random.normal(0, self.process_vol, self.N)
        self.logit_particles += noise
        prob_particles = expit(self.logit_particles)
        log_likelihood = -0.5 * ((observed_price - prob_particles) / self.obs_noise)**2
        log_weights = np.log(self.weights + 1e-300) + log_likelihood
        log_weights -= log_weights.max()
        self.weights = np.exp(log_weights)
        self.weights /= self.weights.sum()
        ess = 1.0 / np.sum(self.weights**2)
        if ess < self.N / 2:
            self._systematic_resample()

    def _systematic_resample(self):
        cumsum = np.cumsum(self.weights)
        u = (np.arange(self.N) + np.random.uniform()) / self.N
        indices = np.searchsorted(cumsum, u)
        self.logit_particles = self.logit_particles[indices]
        self.weights = np.ones(self.N) / self.N

    def estimate(self):
        return np.average(expit(self.logit_particles), weights=self.weights)

    def credible_interval(self, alpha=0.05):
        probs = expit(self.logit_particles)
        sorted_idx = np.argsort(probs)
        sorted_probs = probs[sorted_idx]
        sorted_weights = self.weights[sorted_idx]
        cumw = np.cumsum(sorted_weights)
        lower = sorted_probs[np.searchsorted(cumw, alpha/2)]
        upper = sorted_probs[np.searchsorted(cumw, 1 - alpha/2)]
        return lower, upper


def simulate_correlated_outcomes_t(probs, corr_matrix, nu=4, N=100_000):
    d = len(probs)
    L = np.linalg.cholesky(corr_matrix)
    Z = np.random.standard_normal((N, d))
    X = Z @ L.T
    S = np.random.chisquare(nu, N) / nu
    T = X / np.sqrt(S[:, None])
    U = t_dist.cdf(T, nu)
    outcomes = (U < np.array(probs)).astype(int)
    return outcomes


def stratified_binary_mc(S0, K, sigma, T, J=10, N_total=100_000):
    n_per_stratum = N_total // J
    estimates = []
    for j in range(J):
        U = np.random.uniform(j/J, (j+1)/J, n_per_stratum)
        Z = norm.ppf(U)
        S_T = S0 * np.exp((-0.5*sigma**2)*T + sigma*np.sqrt(T)*Z)
        stratum_mean = (S_T > K).mean()
        estimates.append(stratum_mean)
    p_stratified = np.mean(estimates)
    se_stratified = np.std(estimates) / np.sqrt(J)
    return p_stratified, se_stratified


class PredictionMarketABM:
    def __init__(self, true_prob, n_informed=10, n_noise=50, n_mm=5):
        self.true_prob = true_prob
        self.price = 0.50
        self.best_bid = 0.49
        self.best_ask = 0.51
        self.n_informed = n_informed
        self.n_noise = n_noise
        self.n_mm = n_mm
        self.volume = 0
        self.price_history = [self.price]

    def step(self):
        total = self.n_informed + self.n_noise + self.n_mm
        r = np.random.random()
        if r < self.n_informed / total:
            self._informed_trade()
        elif r < (self.n_informed + self.n_noise) / total:
            self._noise_trade()
        else:
            self._mm_update()
        self.price_history.append(self.price)

    def _informed_trade(self):
        signal = self.true_prob + np.random.normal(0, 0.02)
        if signal > self.best_ask + 0.01:
            size = min(0.1, abs(signal - self.price) * 2)
            self.price += size * self._kyle_lambda()
            self.volume += size
        elif signal < self.best_bid - 0.01:
            size = min(0.1, abs(self.price - signal) * 2)
            self.price -= size * self._kyle_lambda()
            self.volume += size
        self.price = np.clip(self.price, 0.01, 0.99)
        self._update_book()

    def _noise_trade(self):
        direction = np.random.choice([-1, 1])
        size = np.random.exponential(0.02)
        self.price += direction * size * self._kyle_lambda()
        self.price = np.clip(self.price, 0.01, 0.99)
        self.volume += size
        self._update_book()

    def _mm_update(self):
        spread = max(0.02, 0.05 * (1 - self.volume / 100))
        self.best_bid = self.price - spread / 2
        self.best_ask = self.price + spread / 2

    def _kyle_lambda(self):
        sigma_v = abs(self.true_prob - self.price) + 0.05
        sigma_u = 0.1 * np.sqrt(self.n_noise)
        return sigma_v / (2 * sigma_u)

    def _update_book(self):
        spread = self.best_ask - self.best_bid
        self.best_bid = self.price - spread / 2
        self.best_ask = self.price + spread / 2

    def run(self, n_steps=1000):
        for _ in range(n_steps):
            self.step()
        return np.array(self.price_history)
