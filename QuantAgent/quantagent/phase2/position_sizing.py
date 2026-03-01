"""Phase 2 stub: Kelly criterion position sizing.

Architecture hook for future implementation. When ready:
- Implement full Kelly and fractional Kelly
- Integrate with the copula confidence scores
- Account for correlation between positions
"""


def kelly_fraction(win_prob: float, win_loss_ratio: float) -> float:
    """Full Kelly criterion: f* = p - q/b

    Args:
        win_prob: Probability of winning (0-1)
        win_loss_ratio: Ratio of average win to average loss

    Returns:
        Optimal fraction of bankroll to risk.
    """
    raise NotImplementedError("Phase 2: Kelly criterion not yet implemented")


def fractional_kelly(win_prob: float, win_loss_ratio: float, fraction: float = 0.25) -> float:
    """Conservative fractional Kelly.

    Args:
        fraction: What fraction of full Kelly to use (default 25%)
    """
    raise NotImplementedError("Phase 2: Fractional Kelly not yet implemented")


def portfolio_kelly(expected_returns: list[float], covariance_matrix: list[list[float]]) -> list[float]:
    """Multi-asset Kelly using mean-variance optimization.

    Takes correlation between positions into account.
    """
    raise NotImplementedError("Phase 2: Portfolio Kelly not yet implemented")
