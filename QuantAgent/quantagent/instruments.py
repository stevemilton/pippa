"""Instrument cluster definitions with automatic futures rollover.

Futures contracts expire quarterly. This module computes the correct
front-month contract date at build time, so qualifyContractsAsync always
resolves unambiguously. Rolls to the next contract 5 business days before
expiry to avoid illiquid final-week pricing.
"""

import logging
from dataclasses import dataclass, field
from datetime import date, timedelta

from ib_insync import Contract, Future, Forex, Stock, Bond

from quantagent.config import config

logger = logging.getLogger(__name__)

# Roll this many business days before expiry to avoid thin liquidity
ROLL_DAYS_BEFORE_EXPIRY = 5


def _next_quarterly_expiry(ref_date: date | None = None) -> str:
    """Compute the next quarterly futures expiry (Mar/Jun/Sep/Dec).

    Returns YYYYMM format string for lastTradeDateOrContractMonth.
    If we're within ROLL_DAYS_BEFORE_EXPIRY of the current quarter's
    expiry, rolls forward to the next quarter.
    """
    today = ref_date or date.today()

    # Quarterly months: March, June, September, December
    quarterly_months = [3, 6, 9, 12]

    # Find the next quarterly month
    for m in quarterly_months:
        year = today.year
        if m < today.month:
            continue

        # Third Friday of the month is a common expiry convention.
        # For safety, assume expiry is around the 15th-21st.
        # We use the 15th as a conservative expiry estimate.
        expiry_estimate = date(year, m, 15)

        # If we're past the roll date (expiry minus N business days),
        # skip to the next quarterly month
        roll_date = _subtract_business_days(expiry_estimate, ROLL_DAYS_BEFORE_EXPIRY)
        if today >= roll_date:
            continue

        return f"{year}{m:02d}"

    # Wrapped around to next year
    return f"{today.year + 1}03"


def _subtract_business_days(d: date, n: int) -> date:
    """Subtract n business days from date d."""
    current = d
    remaining = n
    while remaining > 0:
        current -= timedelta(days=1)
        if current.weekday() < 5:  # Mon-Fri
            remaining -= 1
    return current


def _next_monthly_expiry(ref_date: date | None = None) -> str:
    """Compute the next monthly expiry for monthly contracts (e.g. SONIA).

    Returns YYYYMM format.
    """
    today = ref_date or date.today()
    expiry_estimate = date(today.year, today.month, 15)
    roll_date = _subtract_business_days(expiry_estimate, ROLL_DAYS_BEFORE_EXPIRY)

    if today >= roll_date:
        # Roll to next month
        if today.month == 12:
            return f"{today.year + 1}01"
        return f"{today.year}{today.month + 1:02d}"

    return f"{today.year}{today.month:02d}"


@dataclass
class InstrumentCluster:
    """A group of correlated instruments to analyze together."""
    name: str
    description: str
    contracts: dict[str, Contract] = field(default_factory=dict)


def build_uk_macro_cluster() -> InstrumentCluster:
    """UK gilts, FTSE 100, GBP/USD, BoE rate futures.

    Auto-rolls futures to the correct front-month contract.
    """
    cluster = InstrumentCluster(
        name="uk_macro",
        description="UK gilts, FTSE 100, GBP/USD, and BoE rate futures",
    )

    quarterly = _next_quarterly_expiry()
    monthly = _next_monthly_expiry()
    logger.info("uk_macro: quarterly expiry=%s, monthly expiry=%s", quarterly, monthly)

    # FTSE 100 index future (ICE) — quarterly
    cluster.contracts["FTSE100"] = Future(
        symbol="Z",
        lastTradeDateOrContractMonth=quarterly,
        exchange="ICEEU",
        currency="GBP",
        includeExpired=False,
    )

    # UK Long Gilt future (ICE) — quarterly
    cluster.contracts["GILT"] = Future(
        symbol="R",
        lastTradeDateOrContractMonth=quarterly,
        exchange="ICEEU",
        currency="GBP",
        includeExpired=False,
    )

    # GBP/USD forex — no expiry
    cluster.contracts["GBPUSD"] = Forex("GBPUSD")

    # Short Sterling future (BoE rate expectations) — quarterly
    cluster.contracts["SHORT_STERLING"] = Future(
        symbol="L",
        lastTradeDateOrContractMonth=quarterly,
        exchange="ICEEU",
        currency="GBP",
        includeExpired=False,
    )

    # SONIA future (BoE rate expectations) — monthly
    cluster.contracts["SONIA"] = Future(
        symbol="SO3",
        lastTradeDateOrContractMonth=monthly,
        exchange="ICEEU",
        currency="GBP",
        includeExpired=False,
    )

    return cluster


def build_us_rates_cluster() -> InstrumentCluster:
    """US Treasury complex — rates and equity sensitivity.

    Auto-rolls futures to the correct front-month contract.
    """
    cluster = InstrumentCluster(
        name="us_rates",
        description="US Treasuries, S&P 500, and Fed Funds futures",
    )

    quarterly = _next_quarterly_expiry()
    logger.info("us_rates: quarterly expiry=%s", quarterly)

    # E-mini S&P 500 — quarterly
    cluster.contracts["ES"] = Future(
        symbol="ES",
        lastTradeDateOrContractMonth=quarterly,
        exchange="CME",
        currency="USD",
    )

    # 10-Year T-Note — quarterly
    cluster.contracts["ZN"] = Future(
        symbol="ZN",
        lastTradeDateOrContractMonth=quarterly,
        exchange="CBOT",
        currency="USD",
    )

    # 30-Year T-Bond — quarterly
    cluster.contracts["ZB"] = Future(
        symbol="ZB",
        lastTradeDateOrContractMonth=quarterly,
        exchange="CBOT",
        currency="USD",
    )

    # EUR/USD forex — no expiry
    cluster.contracts["EURUSD"] = Forex("EURUSD")

    return cluster


# Registry of all available clusters
CLUSTERS: dict[str, callable] = {
    "uk_macro": build_uk_macro_cluster,
    "us_rates": build_us_rates_cluster,
}


def get_active_clusters() -> list[InstrumentCluster]:
    """Build and return all active clusters."""
    return [builder() for builder in CLUSTERS.values()]
