"""Phase 2 stub: Bootstrap particle filter for live event tracking.

Architecture hook for future implementation. When ready:
- Use PredictionMarketParticleFilter from math_models as the base
- Add real-time market data feed integration
- Track implied probabilities across events in real time
"""


async def run_particle_filter(instrument: str, observed_prices: list[float]) -> dict:
    """STUB — not yet implemented.

    Will use PredictionMarketParticleFilter from quantagent.math_models
    to track live implied probabilities.
    """
    raise NotImplementedError("Phase 2: Bootstrap particle filter not yet implemented")
