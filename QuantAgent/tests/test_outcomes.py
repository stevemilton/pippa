"""Tests for outcome recording and auto-resolution."""

import json
import pytest
from datetime import datetime, timezone, timedelta

from quantagent import database
from quantagent.outcomes import (
    auto_resolve_predictions,
    record_manual,
    _check_outcome_from_db,
    _get_finding_for_prediction,
)


class TestManualRecording:

    def test_record_manual(self):
        scan_id = database.log_scan("uk_macro", ["FTSE100"], 1)
        database.log_finding(
            scan_id=scan_id,
            instruments=["FTSE100"],
            implied_probs={"FTSE100": 0.7},
            copula_probs={"FTSE100": 0.5},
            discrepancy=0.20,
            confidence=0.8,
            reasoning="Test",
        )

        preds = database.get_unresolved_predictions(hours=1)
        assert len(preds) == 1

        record_manual(preds[0]["id"], 1.0)

        # Now resolved
        unresolved = database.get_unresolved_predictions(hours=1)
        assert len(unresolved) == 0


class TestGetFindingForPrediction:

    def test_existing_finding(self):
        scan_id = database.log_scan("uk_macro", ["FTSE100"], 1)
        finding_id = database.log_finding(
            scan_id=scan_id,
            instruments=["FTSE100"],
            implied_probs={"FTSE100": 0.6},
            copula_probs={"FTSE100": 0.5},
            discrepancy=0.10,
            confidence=0.7,
            reasoning="Test",
        )

        finding = _get_finding_for_prediction(finding_id)
        assert finding is not None
        assert finding["discrepancy"] == 0.10

    def test_missing_finding(self):
        finding = _get_finding_for_prediction(99999)
        assert finding is None


class TestAutoResolver:

    def test_no_unresolved(self):
        """When there are no unresolved predictions, should return zeros."""
        result = auto_resolve_predictions(lookback_hours=24)
        assert result["resolved"] == 0
        assert result["skipped"] == 0

    def test_skips_too_young_predictions(self):
        """Predictions younger than lookback_hours should be skipped."""
        scan_id = database.log_scan("uk_macro", ["FTSE100"], 1)
        database.log_finding(
            scan_id=scan_id,
            instruments=["FTSE100"],
            implied_probs={"FTSE100": 0.7},
            copula_probs={"FTSE100": 0.5},
            discrepancy=0.20,
            confidence=0.8,
            reasoning="Test",
        )

        # Prediction just created, lookback is 24h → should skip
        result = auto_resolve_predictions(lookback_hours=24)
        assert result["skipped"] > 0
        assert result["resolved"] == 0


class TestCheckOutcomeFromDb:

    def test_no_later_findings_returns_none(self):
        """If there are no subsequent findings, outcome is unknown."""
        pred = {
            "id": 1,
            "finding_id": 1,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "instrument": "FTSE100",
            "cluster_name": "uk_macro",
            "predicted_prob": 0.7,
        }
        # No findings in DB yet → should return None
        outcome = _check_outcome_from_db(pred, lookback_hours=24)
        assert outcome is None
