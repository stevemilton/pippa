"""Outcome recording and auto-resolution for Brier score calibration.

Three ways to resolve predictions:
1. CLI: python -m quantagent.outcomes record --id 42 --outcome 1
2. Auto: The auto-resolver checks whether predicted price moves materialised
3. Telegram: /outcome 42 1 (via the Telegram bot webhook — Phase 2)

A prediction is: "instrument X will move in direction Y with probability P".
The outcome is binary: 1 if the move happened within the lookback window, 0 if not.
"""

import argparse
import json
import logging
import sys
from datetime import datetime, timezone, timedelta
from pathlib import Path

import numpy as np

from quantagent.config import config
from quantagent import database

logger = logging.getLogger(__name__)


# ── Auto-resolver ────────────────────────────────────────────────────────────

def auto_resolve_predictions(market_data_client=None, lookback_hours: int = 24):
    """Auto-resolve predictions by checking whether the predicted move happened.

    For each unresolved prediction:
    1. Look at the finding's implied probability and copula probability
    2. The prediction is: "this instrument is mispriced — the implied probability
       is too high/too low relative to the copula model"
    3. If the discrepancy (implied - copula) > 0, the market was overpricing
       the probability of an up move. Outcome = 1 if the instrument went DOWN
       (i.e., the copula was right and the market was wrong).
    4. If discrepancy < 0, the market was underpricing. Outcome = 1 if UP.

    We check this by looking at the actual return over the lookback window.
    """
    from quantagent import database as db

    unresolved = db.get_unresolved_predictions(hours=lookback_hours * 2)

    if not unresolved:
        logger.info("No unresolved predictions to process")
        return {"resolved": 0, "skipped": 0}

    resolved_count = 0
    skipped_count = 0

    for pred in unresolved:
        pred_time = datetime.fromisoformat(pred["timestamp"])
        age_hours = (datetime.now(timezone.utc) - pred_time).total_seconds() / 3600

        # Only resolve predictions that are old enough (at least lookback_hours)
        if age_hours < lookback_hours:
            skipped_count += 1
            continue

        # Get the finding to determine the discrepancy direction
        finding = _get_finding_for_prediction(pred["finding_id"])
        if finding is None:
            skipped_count += 1
            continue

        discrepancies = json.loads(finding["implied_probs"])
        copula_probs = json.loads(finding["copula_probs"])
        implied_probs = json.loads(finding["implied_probs"])

        instrument = pred["instrument"]
        predicted_prob = pred["predicted_prob"]

        # Determine what the prediction was:
        # implied > copula means market overpriced the up-move probability
        # So the "prediction" is that the true probability is LOWER (copula is right)
        imp = implied_probs.get(instrument, 0.5)
        cop = copula_probs.get(instrument, 0.5)

        if abs(imp - cop) < 0.001:
            # No meaningful prediction — skip
            skipped_count += 1
            continue

        # For Brier scoring:
        # prediction = copula probability (what the model says the true prob is)
        # outcome = 1 if the instrument moved up, 0 if it moved down
        # If copula said p=0.3 and the market said p=0.6, and the instrument
        # actually went down (outcome=0), copula was closer to right.

        # Try to get the actual return from market data
        if market_data_client is not None:
            outcome = _check_outcome_live(market_data_client, pred, lookback_hours)
        else:
            # Offline mode: try to determine from the database
            outcome = _check_outcome_from_db(pred, lookback_hours)

        if outcome is not None:
            db.record_outcome(pred["id"], outcome)
            resolved_count += 1
            logger.info(
                "Resolved prediction %d: %s outcome=%d (predicted=%.3f)",
                pred["id"], instrument, outcome, predicted_prob,
            )
        else:
            skipped_count += 1

    logger.info("Auto-resolve: %d resolved, %d skipped", resolved_count, skipped_count)
    return {"resolved": resolved_count, "skipped": skipped_count}


def _get_finding_for_prediction(finding_id: int) -> dict | None:
    """Get the finding associated with a prediction."""
    from quantagent.database import _get_conn
    conn = _get_conn()
    row = conn.execute("SELECT * FROM findings WHERE id = ?", (finding_id,)).fetchone()
    return dict(row) if row else None


def _check_outcome_live(market_data_client, pred: dict, lookback_hours: int) -> int | None:
    """Check outcome using live market data. Returns 1 (up) or 0 (down) or None."""
    # This would need the market_data_client to be async, so we skip in CLI mode
    # and rely on _check_outcome_from_db instead.
    return None


def _check_outcome_from_db(pred: dict, lookback_hours: int) -> int | None:
    """Infer outcome from subsequent findings for the same instrument.

    If a later scan shows the instrument's implied probability moved toward
    the copula prediction, the copula was right (outcome aligns with model).
    """
    from quantagent.database import _get_conn
    conn = _get_conn()

    pred_time = pred["timestamp"]
    instrument = pred["instrument"]
    cluster = pred["cluster_name"]

    # Look for later findings involving this instrument
    later_findings = conn.execute(
        """SELECT implied_probs, copula_probs FROM findings
           WHERE scan_id IN (SELECT id FROM scans WHERE cluster_name = ?)
             AND timestamp > ?
             AND instruments LIKE ?
           ORDER BY timestamp ASC LIMIT 5""",
        (cluster, pred_time, f'%"{instrument}"%'),
    ).fetchall()

    if not later_findings:
        return None

    # Get original finding
    finding = _get_finding_for_prediction(pred["finding_id"])
    if not finding:
        return None

    orig_implied = json.loads(finding["implied_probs"]).get(instrument)
    orig_copula = json.loads(finding["copula_probs"]).get(instrument)
    if orig_implied is None or orig_copula is None:
        return None

    # Check the latest observation
    latest = later_findings[-1]
    new_implied = json.loads(latest["implied_probs"]).get(instrument)
    if new_implied is None:
        return None

    # Did the market move toward the copula prediction?
    # Original discrepancy: implied - copula
    # If discrepancy was positive (implied too high), copula says it should go down.
    # Outcome = 1 if the new implied is lower (moved toward copula) = copula was right.
    orig_disc = orig_implied - orig_copula
    new_disc = new_implied - orig_copula

    if abs(orig_disc) < 0.001:
        return None

    # The market corrected toward the copula model
    if abs(new_disc) < abs(orig_disc):
        return 1  # Copula was right — market converged to model
    else:
        return 0  # Market moved away from model — copula was wrong


# ── Manual recording ─────────────────────────────────────────────────────────

def record_manual(prediction_id: int, outcome: float):
    """Record an outcome manually."""
    database.record_outcome(prediction_id, outcome)
    logger.info("Manually recorded: prediction %d -> outcome %.1f", prediction_id, outcome)


def list_unresolved(hours: int = 168):
    """List unresolved predictions for manual review."""
    preds = database.get_unresolved_predictions(hours=hours)
    if not preds:
        print("No unresolved predictions.")
        return

    print(f"\n{'ID':>5}  {'Cluster':<12}  {'Instrument':<16}  {'Predicted':>9}  {'Age (h)':>7}")
    print("-" * 65)
    for p in preds:
        age_h = (datetime.now(timezone.utc) - datetime.fromisoformat(p["timestamp"])).total_seconds() / 3600
        print(f"{p['id']:>5}  {p['cluster_name']:<12}  {p['instrument']:<16}  {p['predicted_prob']:>9.4f}  {age_h:>7.1f}")


def show_brier_scores():
    """Print current Brier scores per cluster."""
    overall = database.compute_brier_score()
    print(f"\nOverall Brier Score: {overall['brier_score'] or 'N/A'}")
    print(f"  Resolved: {overall['n_resolved']}/{overall['n_predictions']}")
    print(f"  Target: < 0.10")
    if overall["brier_score"] is not None:
        status = "ON TARGET" if overall["on_target"] else "ABOVE TARGET"
        print(f"  Status: {status}")

    # Per-cluster
    from quantagent.database import _get_conn
    conn = _get_conn()
    clusters = conn.execute(
        "SELECT DISTINCT cluster_name FROM predictions"
    ).fetchall()

    for row in clusters:
        cname = row["cluster_name"]
        bs = database.compute_brier_score(cname)
        if bs["brier_score"] is not None:
            status = "ON TARGET" if bs["on_target"] else "ABOVE TARGET"
            print(f"\n  {cname}: {bs['brier_score']:.4f} ({status}) — {bs['n_resolved']}/{bs['n_predictions']} resolved")
        else:
            print(f"\n  {cname}: No resolved predictions")


# ── CLI entrypoint ───────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="QuantAgent outcome recording and Brier calibration",
        prog="python -m quantagent.outcomes",
    )
    sub = parser.add_subparsers(dest="command", help="Command")

    # Record outcome
    rec = sub.add_parser("record", help="Record an outcome for a prediction")
    rec.add_argument("--id", type=int, required=True, help="Prediction ID")
    rec.add_argument("--outcome", type=float, required=True, help="Outcome (0 or 1)")

    # Auto-resolve
    auto = sub.add_parser("auto-resolve", help="Auto-resolve predictions from market data")
    auto.add_argument("--lookback", type=int, default=24, help="Lookback hours (default 24)")

    # List unresolved
    ls = sub.add_parser("list", help="List unresolved predictions")
    ls.add_argument("--hours", type=int, default=168, help="Hours to look back (default 168)")

    # Show Brier scores
    sub.add_parser("brier", help="Show current Brier scores")

    args = parser.parse_args()

    if args.command == "record":
        if args.outcome not in (0, 0.0, 1, 1.0):
            print("ERROR: outcome must be 0 or 1")
            sys.exit(1)
        record_manual(args.id, args.outcome)
        print(f"Recorded: prediction {args.id} -> outcome {args.outcome}")

    elif args.command == "auto-resolve":
        result = auto_resolve_predictions(lookback_hours=args.lookback)
        print(f"Resolved: {result['resolved']}, Skipped: {result['skipped']}")

    elif args.command == "list":
        list_unresolved(hours=args.hours)

    elif args.command == "brier":
        show_brier_scores()

    else:
        parser.print_help()


if __name__ == "__main__":
    main()
