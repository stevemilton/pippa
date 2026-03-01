"""Phase 2 stub: Betfair API as second data source.

Architecture hook for future implementation. When ready:
- Connect to Betfair Exchange Streaming API
- Pull odds for correlated event markets
- Feed into copula analysis alongside IBKR data
"""


class BetfairClient:
    """STUB — Betfair API client for exchange data."""

    def __init__(self, app_key: str, session_token: str):
        raise NotImplementedError("Phase 2: Betfair integration not yet implemented")

    async def get_market_odds(self, market_id: str) -> dict:
        raise NotImplementedError

    async def get_event_markets(self, event_type: str) -> list[dict]:
        raise NotImplementedError
