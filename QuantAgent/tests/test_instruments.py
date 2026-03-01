"""Tests for instrument cluster definitions and futures auto-rollover."""

from datetime import date
import pytest

from quantagent.instruments import (
    _next_quarterly_expiry,
    _next_monthly_expiry,
    _subtract_business_days,
    build_uk_macro_cluster,
    build_us_rates_cluster,
    get_active_clusters,
    ROLL_DAYS_BEFORE_EXPIRY,
)


# ── Quarterly expiry ───────────────────────────────────────────────────────

class TestNextQuarterlyExpiry:

    def test_well_before_march_expiry(self):
        """Jan 10 → should get 202503 (March still far away)."""
        result = _next_quarterly_expiry(date(2025, 1, 10))
        assert result == "202503"

    def test_just_after_march_roll(self):
        """March 10 is within 5 bdays of the 15th → should skip to June."""
        result = _next_quarterly_expiry(date(2025, 3, 10))
        assert result == "202506"

    def test_well_before_june_expiry(self):
        """April 1 → should get 202506."""
        result = _next_quarterly_expiry(date(2025, 4, 1))
        assert result == "202506"

    def test_december_wraps_to_next_year(self):
        """Dec 12 is near December expiry → should wrap to next year March."""
        result = _next_quarterly_expiry(date(2025, 12, 12))
        assert result == "202603"

    def test_exact_roll_boundary(self):
        """The exact roll date should trigger a skip to next quarter."""
        # 5 business days before March 15 = roughly March 7-10 depending on year
        roll = _subtract_business_days(date(2025, 3, 15), ROLL_DAYS_BEFORE_EXPIRY)
        result = _next_quarterly_expiry(roll)
        assert result == "202506"

    def test_one_day_before_roll_boundary(self):
        """One day before roll should still return current quarter."""
        roll = _subtract_business_days(date(2025, 6, 15), ROLL_DAYS_BEFORE_EXPIRY)
        one_before = _subtract_business_days(roll, 1)
        # Should be before the roll boundary, but check it's the same or prior quarter
        # The day before the roll date of June should still return June
        result = _next_quarterly_expiry(one_before)
        # one_before is a day before the roll date, so June is still valid
        assert result == "202506"


# ── Monthly expiry ─────────────────────────────────────────────────────────

class TestNextMonthlyExpiry:

    def test_early_in_month(self):
        """Jan 3 → should get 202501 (far from the 15th)."""
        result = _next_monthly_expiry(date(2025, 1, 3))
        assert result == "202501"

    def test_near_expiry_rolls_forward(self):
        """Jan 12 is near the 15th → should roll to February."""
        result = _next_monthly_expiry(date(2025, 1, 12))
        assert result == "202502"

    def test_december_wraps(self):
        """Dec 14 near expiry → wraps to next year January."""
        result = _next_monthly_expiry(date(2025, 12, 14))
        assert result == "202601"


# ── Business day subtraction ───────────────────────────────────────────────

class TestSubtractBusinessDays:

    def test_basic_weekday(self):
        """Wednesday minus 2 bdays = Monday."""
        result = _subtract_business_days(date(2025, 1, 15), 2)  # Wed
        assert result == date(2025, 1, 13)  # Mon

    def test_skips_weekend(self):
        """Monday minus 1 bday = Friday."""
        result = _subtract_business_days(date(2025, 1, 13), 1)  # Mon
        assert result == date(2025, 1, 10)  # Fri

    def test_five_bdays_crosses_weekend(self):
        """Friday minus 5 bdays = previous Friday."""
        result = _subtract_business_days(date(2025, 1, 17), 5)  # Fri
        assert result == date(2025, 1, 10)  # Fri

    def test_zero_days(self):
        """Subtracting 0 business days returns the same date."""
        d = date(2025, 1, 15)
        assert _subtract_business_days(d, 0) == d


# ── Cluster building ──────────────────────────────────────────────────────

class TestClusterBuilding:

    def test_uk_macro_has_expected_instruments(self):
        cluster = build_uk_macro_cluster()
        assert cluster.name == "uk_macro"
        expected = {"FTSE100", "GILT", "GBPUSD", "SHORT_STERLING", "SONIA"}
        assert set(cluster.contracts.keys()) == expected

    def test_us_rates_has_expected_instruments(self):
        cluster = build_us_rates_cluster()
        assert cluster.name == "us_rates"
        expected = {"ES", "ZN", "ZB", "EURUSD"}
        assert set(cluster.contracts.keys()) == expected

    def test_futures_have_expiry_set(self):
        """All futures must have lastTradeDateOrContractMonth set."""
        cluster = build_uk_macro_cluster()
        for name, contract in cluster.contracts.items():
            if hasattr(contract, "lastTradeDateOrContractMonth"):
                ltdom = contract.lastTradeDateOrContractMonth
                if ltdom:  # Forex won't have this
                    assert len(ltdom) == 6, f"{name} expiry should be YYYYMM, got {ltdom}"
                    assert ltdom[:4].isdigit(), f"{name} expiry year invalid: {ltdom}"

    def test_get_active_clusters(self):
        clusters = get_active_clusters()
        assert len(clusters) == 2
        names = {c.name for c in clusters}
        assert names == {"uk_macro", "us_rates"}
