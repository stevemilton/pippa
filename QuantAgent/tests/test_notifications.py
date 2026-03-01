"""Tests for notification formatting (no actual Telegram calls)."""

import pytest

from quantagent.notifications import (
    format_mispricing_alert,
    format_weekly_summary,
    format_error_alert,
    format_startup_message,
)


class TestMispricingAlert:

    def test_basic_format(self):
        msg = format_mispricing_alert(
            cluster_name="uk_macro",
            instruments=["FTSE100", "GILT"],
            implied_probs={"FTSE100": 0.6500, "GILT": 0.4500},
            copula_probs={"FTSE100": 0.5500, "GILT": 0.5000},
            discrepancies={"FTSE100": 0.1000, "GILT": -0.0500},
            max_discrepancy=0.1000,
            confidence=0.75,
            reasoning="FTSE100 implied probability diverges from copula model.",
        )
        assert "UK_MACRO" in msg
        assert "MISPRICING" in msg
        assert "Student-t" in msg
        assert "FTSE100" in msg
        assert "0.6500" in msg

    def test_includes_tail_dependence(self):
        msg = format_mispricing_alert(
            cluster_name="uk_macro",
            instruments=["FTSE100"],
            implied_probs={"FTSE100": 0.65},
            copula_probs={"FTSE100": 0.55},
            discrepancies={"FTSE100": 0.10},
            max_discrepancy=0.10,
            confidence=0.75,
            reasoning="Test",
            tail_dependence={"FTSE100-GILT": 0.1823},
        )
        assert "Tail Dependence" in msg
        assert "0.1823" in msg

    def test_includes_importance_sampling(self):
        msg = format_mispricing_alert(
            cluster_name="uk_macro",
            instruments=["FTSE100"],
            implied_probs={"FTSE100": 0.03},
            copula_probs={"FTSE100": 0.05},
            discrepancies={"FTSE100": -0.02},
            max_discrepancy=0.02,
            confidence=0.6,
            reasoning="Tail regime",
            is_tail_regime=True,
            is_instruments=["FTSE100"],
        )
        assert "Importance sampling" in msg
        assert "FTSE100" in msg

    def test_includes_variance_reduction(self):
        msg = format_mispricing_alert(
            cluster_name="uk_macro",
            instruments=["FTSE100"],
            implied_probs={"FTSE100": 0.65},
            copula_probs={"FTSE100": 0.55},
            discrepancies={"FTSE100": 0.10},
            max_discrepancy=0.10,
            confidence=0.75,
            reasoning="Test",
            variance_reduction=350.0,
        )
        assert "Variance reduction" in msg
        assert "350" in msg


class TestWeeklySummary:

    def test_format(self):
        msg = format_weekly_summary(
            cluster_brier={
                "uk_macro": {"brier_score": 0.08, "n_resolved": 10, "n_predictions": 15, "on_target": True},
                "us_rates": {"brier_score": 0.12, "n_resolved": 5, "n_predictions": 8, "on_target": False},
            },
            scan_stats={"total_scans": 100, "total_mispricings": 5, "error_scans": 2},
        )
        assert "WEEKLY SUMMARY" in msg
        assert "BRIER" in msg
        assert "ON TARGET" in msg
        assert "ABOVE TARGET" in msg
        assert "uk_macro" in msg
        assert "us_rates" in msg

    def test_no_resolved_cluster(self):
        msg = format_weekly_summary(
            cluster_brier={"uk_macro": {"brier_score": None, "n_resolved": 0, "n_predictions": 3, "on_target": None}},
            scan_stats={"total_scans": 10, "total_mispricings": 3, "error_scans": 0},
        )
        assert "No resolved predictions" in msg


class TestErrorAlert:

    def test_format(self):
        msg = format_error_alert("scanner", "IB Gateway disconnected")
        assert "ERROR" in msg
        assert "SCANNER" in msg
        assert "disconnected" in msg


class TestStartupMessage:

    def test_format(self):
        msg = format_startup_message(["uk_macro", "us_rates"], paper=True)
        assert "PAPER" in msg
        assert "Student-t" in msg
        assert "Gaussian excluded" in msg
        assert "uk_macro" in msg
        assert "5%" in msg
        assert "100-500x" in msg
