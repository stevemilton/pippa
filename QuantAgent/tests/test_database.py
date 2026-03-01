"""Tests for SQLite database logging, Brier score, and prediction tracking."""

import json
import pytest

from quantagent import database


class TestScanLogging:

    def test_log_scan_returns_id(self):
        scan_id = database.log_scan(
            cluster_name="uk_macro",
            instruments=["FTSE100", "GILT"],
            mispricings_found=0,
        )
        assert isinstance(scan_id, int)
        assert scan_id > 0

    def test_scan_stats(self):
        database.log_scan("uk_macro", ["FTSE100"], 2)
        database.log_scan("uk_macro", ["FTSE100"], 1)
        database.log_scan("uk_macro", ["FTSE100"], 0, status="error", error_message="test")

        stats = database.get_scan_stats(hours=1)
        assert stats["total_scans"] == 3
        assert stats["total_mispricings"] == 3
        assert stats["error_scans"] == 1


class TestFindingLogging:

    def test_log_finding_creates_predictions(self):
        """Every finding should auto-create prediction records for Brier tracking."""
        scan_id = database.log_scan("uk_macro", ["FTSE100", "GILT"], 1)

        implied = {"FTSE100": 0.65, "GILT": 0.45}
        copula = {"FTSE100": 0.55, "GILT": 0.50}

        finding_id = database.log_finding(
            scan_id=scan_id,
            instruments=["FTSE100", "GILT"],
            implied_probs=implied,
            copula_probs=copula,
            discrepancy=0.10,
            confidence=0.75,
            reasoning="Test finding",
            tail_dependence={"FTSE100-GILT": 0.18},
            variance_reduction=250.0,
            is_tail_regime=False,
        )
        assert isinstance(finding_id, int)

        # Check predictions were created
        preds = database.get_unresolved_predictions(hours=1)
        assert len(preds) == 2
        instruments_found = {p["instrument"] for p in preds}
        assert instruments_found == {"FTSE100", "GILT"}

    def test_recent_findings(self):
        scan_id = database.log_scan("uk_macro", ["FTSE100"], 1)
        database.log_finding(
            scan_id=scan_id,
            instruments=["FTSE100"],
            implied_probs={"FTSE100": 0.6},
            copula_probs={"FTSE100": 0.5},
            discrepancy=0.10,
            confidence=0.7,
            reasoning="Test",
        )
        findings = database.get_recent_findings(hours=1)
        assert len(findings) == 1
        assert findings[0]["discrepancy"] == 0.10


class TestBrierScore:

    def _create_prediction(self, cluster="uk_macro", instrument="FTSE100", prob=0.7):
        scan_id = database.log_scan(cluster, [instrument], 1)
        database.log_finding(
            scan_id=scan_id,
            instruments=[instrument],
            implied_probs={instrument: prob},
            copula_probs={instrument: 0.5},
            discrepancy=abs(prob - 0.5),
            confidence=0.8,
            reasoning="Test prediction",
        )

    def test_no_resolved_predictions(self):
        """Before any outcomes are recorded, Brier should be None."""
        bs = database.compute_brier_score()
        assert bs["brier_score"] is None
        assert bs["n_resolved"] == 0

    def test_perfect_predictions(self):
        """Predicted 0.9 and outcome was 1, predicted 0.1 and outcome was 0."""
        self._create_prediction(prob=0.9)
        self._create_prediction(prob=0.1, instrument="GILT")

        preds = database.get_unresolved_predictions(hours=1)
        # Record perfect outcomes
        for p in preds:
            if p["predicted_prob"] > 0.5:
                database.record_outcome(p["id"], 1.0)
            else:
                database.record_outcome(p["id"], 0.0)

        bs = database.compute_brier_score()
        assert bs["brier_score"] is not None
        # Brier for (0.9, 1) = 0.01, (0.1, 0) = 0.01 → mean = 0.01
        assert bs["brier_score"] < 0.05
        assert bs["on_target"] is True

    def test_terrible_predictions(self):
        """Predicted 0.9 but outcome was 0 — Brier should be high."""
        self._create_prediction(prob=0.9)

        preds = database.get_unresolved_predictions(hours=1)
        database.record_outcome(preds[0]["id"], 0.0)

        bs = database.compute_brier_score()
        assert bs["brier_score"] is not None
        # Brier for (0.9, 0) = 0.81
        assert bs["brier_score"] > 0.5
        assert bs["on_target"] is False

    def test_brier_per_cluster(self):
        """Brier scores should be computed per cluster."""
        self._create_prediction(cluster="uk_macro", prob=0.8)
        self._create_prediction(cluster="us_rates", instrument="ES", prob=0.2)

        preds = database.get_unresolved_predictions(hours=1)
        for p in preds:
            if p["cluster_name"] == "uk_macro":
                database.record_outcome(p["id"], 1.0)  # good prediction
            else:
                database.record_outcome(p["id"], 1.0)  # bad prediction (said 0.2, got 1)

        bs_uk = database.compute_brier_score("uk_macro")
        bs_us = database.compute_brier_score("us_rates")

        assert bs_uk["brier_score"] < bs_us["brier_score"]

    def test_snapshot_brier_scores(self):
        self._create_prediction(prob=0.7)
        preds = database.get_unresolved_predictions(hours=1)
        database.record_outcome(preds[0]["id"], 1.0)

        results = database.snapshot_brier_scores()
        assert "uk_macro" in results
        assert results["uk_macro"]["brier_score"] is not None

        # Check it was persisted
        history = database.get_brier_history("uk_macro")
        assert len(history) == 1


class TestOutcomeRecording:

    def test_record_outcome(self):
        scan_id = database.log_scan("uk_macro", ["FTSE100"], 1)
        database.log_finding(
            scan_id=scan_id,
            instruments=["FTSE100"],
            implied_probs={"FTSE100": 0.6},
            copula_probs={"FTSE100": 0.5},
            discrepancy=0.10,
            confidence=0.7,
            reasoning="Test",
        )

        preds = database.get_unresolved_predictions(hours=1)
        assert len(preds) == 1
        assert preds[0]["resolved"] == 0

        database.record_outcome(preds[0]["id"], 1.0)

        # Should now be resolved
        unresolved = database.get_unresolved_predictions(hours=1)
        assert len(unresolved) == 0


class TestErrorLogging:

    def test_log_error(self):
        database.log_error("test_component", "Something broke", details="stacktrace here")
        conn = database._get_conn()
        row = conn.execute("SELECT * FROM errors ORDER BY id DESC LIMIT 1").fetchone()
        assert row["component"] == "test_component"
        assert "broke" in row["message"]
