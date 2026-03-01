"""Phase 2 stub: Trade execution via IBKR with ABM market impact check.

Architecture hooks:
- place_order: Execute trades via ib_insync
- estimate_market_impact: Use PredictionMarketABM to simulate price impact
  before execution. This is how we avoid moving the price against ourselves
  on thin markets.
"""

from dataclasses import dataclass

from quantagent.math_models import PredictionMarketABM


@dataclass
class OrderRequest:
    instrument: str
    size: float
    side: str  # "BUY" or "SELL"
    order_type: str = "LMT"
    limit_price: float | None = None


@dataclass
class MarketImpactEstimate:
    """Result of ABM market impact simulation."""
    instrument: str
    proposed_size: float
    side: str
    estimated_slippage: float      # price impact as fraction
    estimated_price_after: float
    price_path: list[float]        # full simulated price trajectory
    is_safe: bool                  # whether impact is acceptable
    max_safe_size: float | None    # suggested max if impact too high


def estimate_market_impact(
    instrument: str,
    current_price: float,
    true_prob: float,
    proposed_size: float,
    side: str,
    n_noise_traders: int = 50,
    n_market_makers: int = 5,
    n_steps: int = 500,
) -> MarketImpactEstimate:
    """PHASE 2 STUB — Pre-trade ABM market impact check.

    Uses PredictionMarketABM to estimate how much the market will move
    against us for a given position size on a given instrument.

    This is how we avoid moving the price against ourselves on thin markets.

    When implemented:
    1. Initialize ABM with current market state
    2. Inject a large informed order matching our proposed trade
    3. Measure price impact over the simulation
    4. If impact exceeds threshold, reduce size or reject

    The interface is defined; the implementation will wire it up.
    """
    raise NotImplementedError(
        "Phase 2: ABM market impact estimation not yet implemented. "
        "The interface is defined — implementation will use PredictionMarketABM "
        "to simulate price impact before execution."
    )


async def place_order(request: OrderRequest) -> dict:
    """STUB — not yet implemented."""
    raise NotImplementedError(
        "Phase 2: Trade execution is not yet implemented. "
        "This stub exists to define the interface."
    )
