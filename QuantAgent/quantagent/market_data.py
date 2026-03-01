"""IBKR connection and market data retrieval via ib_insync."""

import logging
import asyncio
from datetime import datetime, timezone
from typing import Any

from ib_insync import IB, Contract, util

from quantagent.config import config
from quantagent import database

logger = logging.getLogger(__name__)


class MarketDataClient:
    """Manages the IBKR connection and fetches live / historical data."""

    def __init__(self):
        self.ib = IB()
        self._connected = False

    async def connect(self):
        """Connect to IB Gateway. Retries once on failure."""
        if self._connected and self.ib.isConnected():
            return
        try:
            await self.ib.connectAsync(
                host=config.ibkr_host,
                port=config.ibkr_port,
                clientId=config.ibkr_client_id,
                readonly=True,
            )
            self._connected = True
            logger.info("Connected to IB Gateway at %s:%s", config.ibkr_host, config.ibkr_port)
        except Exception as e:
            self._connected = False
            logger.error("IB Gateway connection failed: %s", e)
            database.log_error("market_data", f"Connection failed: {e}")
            raise

    async def ensure_connected(self):
        """Reconnect if the connection dropped."""
        if not self.ib.isConnected():
            logger.warning("IB Gateway disconnected — reconnecting")
            self._connected = False
            await self.connect()

    async def qualify_contracts(self, contracts: dict[str, Contract]) -> dict[str, Contract]:
        """Resolve ambiguous contracts with IBKR. Returns only successfully qualified ones."""
        await self.ensure_connected()
        qualified = {}
        for name, contract in contracts.items():
            try:
                resolved = await self.ib.qualifyContractsAsync(contract)
                if resolved:
                    qualified[name] = resolved[0]
                    logger.debug("Qualified %s -> conId=%s", name, resolved[0].conId)
                else:
                    logger.warning("Could not qualify contract: %s", name)
                    database.log_error("market_data", f"Contract qualification failed: {name}")
            except Exception as e:
                logger.warning("Error qualifying %s: %s", name, e)
                database.log_error("market_data", f"Qualify error for {name}: {e}")
        return qualified

    async def get_snapshot(self, contracts: dict[str, Contract]) -> dict[str, dict[str, Any]]:
        """Get current market snapshot for a set of named contracts.

        Returns dict of instrument_name -> {bid, ask, last, mid, timestamp}.
        Skips instruments with stale or unavailable data.
        """
        await self.ensure_connected()
        result = {}

        for name, contract in contracts.items():
            try:
                self.ib.reqMktData(contract, "", False, False)
            except Exception as e:
                logger.warning("reqMktData failed for %s: %s", name, e)
                continue

        # Wait for tickers to populate — poll up to 5s instead of blind sleep.
        # Most data arrives within 1-2s; we bail early once all tickers have prices.
        for _wait in range(10):
            await asyncio.sleep(0.5)
            all_ready = True
            for contract in contracts.values():
                t = self.ib.ticker(contract)
                if t is None or (t.bid in (None, 0, -1) and t.ask in (None, 0, -1)
                                 and t.last in (None, 0, -1)):
                    all_ready = False
                    break
            if all_ready:
                break

        for name, contract in contracts.items():
            ticker = self.ib.ticker(contract)
            if ticker is None:
                logger.warning("No ticker for %s — skipping", name)
                continue

            bid = ticker.bid if ticker.bid and ticker.bid > 0 else None
            ask = ticker.ask if ticker.ask and ticker.ask > 0 else None
            last = ticker.last if ticker.last and ticker.last > 0 else None

            if bid is None and ask is None and last is None:
                logger.warning("No price data for %s — skipping", name)
                database.log_error("market_data", f"No price data for {name}")
                continue

            mid = None
            if bid is not None and ask is not None:
                mid = (bid + ask) / 2

            result[name] = {
                "bid": bid,
                "ask": ask,
                "last": last,
                "mid": mid or last,
                "spread": (ask - bid) if (bid and ask) else None,
                "volume": ticker.volume if ticker.volume else 0,
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }

        # Cancel subscriptions
        for contract in contracts.values():
            try:
                self.ib.cancelMktData(contract)
            except Exception:
                pass

        return result

    async def get_historical_closes(
        self, contract: Contract, days: int = 60, bar_size: str = "1 day"
    ) -> list[float]:
        """Fetch historical daily closes for copula fitting."""
        await self.ensure_connected()
        try:
            bars = await self.ib.reqHistoricalDataAsync(
                contract,
                endDateTime="",
                durationStr=f"{days} D",
                barSizeSetting=bar_size,
                whatToShow="TRADES",
                useRTH=True,
                formatDate=1,
            )
            if not bars:
                logger.warning("No historical data for %s", contract.symbol)
                return []
            return [b.close for b in bars]
        except Exception as e:
            logger.error("Historical data error for %s: %s", contract.symbol, e)
            database.log_error("market_data", f"Historical data error: {e}")
            return []

    async def get_historical_returns(
        self, contracts: dict[str, Contract], days: int = 60
    ) -> dict[str, list[float]]:
        """Fetch log returns for multiple contracts (used by copula engine)."""
        import numpy as np

        returns = {}
        for name, contract in contracts.items():
            closes = await self.get_historical_closes(contract, days=days)
            if len(closes) < 10:
                logger.warning("Insufficient data for %s (%d bars)", name, len(closes))
                continue
            prices = np.array(closes)
            log_rets = np.diff(np.log(prices))
            returns[name] = log_rets.tolist()
        return returns

    async def get_positions(self) -> list[dict]:
        """Return current open positions."""
        await self.ensure_connected()
        positions = self.ib.positions()
        return [
            {
                "symbol": p.contract.symbol,
                "exchange": p.contract.exchange,
                "position": p.position,
                "avg_cost": p.avgCost,
                "contract_type": p.contract.secType,
            }
            for p in positions
        ]

    def disconnect(self):
        if self.ib.isConnected():
            self.ib.disconnect()
            self._connected = False
            logger.info("Disconnected from IB Gateway")
