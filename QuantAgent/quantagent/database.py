"""SQLite logging for scan results, mispricings, and Brier score calibration.

Every mispricing flagged is a prediction. When the outcome is known, we record
it and compute running Brier score per cluster. Target: below 0.10.
"""

import json
import sqlite3
import threading
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from quantagent.config import config
from quantagent.math_models import brier_score

_local = threading.local()


def _get_conn() -> sqlite3.Connection:
    """One connection per thread, auto-creates tables on first use."""
    if not hasattr(_local, "conn"):
        db_path = Path(config.db_path)
        db_path.parent.mkdir(parents=True, exist_ok=True)
        _local.conn = sqlite3.connect(str(db_path))
        _local.conn.row_factory = sqlite3.Row
        _local.conn.execute("PRAGMA journal_mode=WAL")
        _local.conn.execute("PRAGMA foreign_keys=ON")
        _init_tables(_local.conn)
    return _local.conn


def _init_tables(conn: sqlite3.Connection):
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS scans (
            id              INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp       TEXT    NOT NULL,
            cluster_name    TEXT    NOT NULL,
            instruments     TEXT    NOT NULL,
            mispricings_found INTEGER NOT NULL DEFAULT 0,
            status          TEXT    NOT NULL DEFAULT 'ok',
            error_message   TEXT,
            claude_reasoning TEXT
        );

        CREATE TABLE IF NOT EXISTS findings (
            id              INTEGER PRIMARY KEY AUTOINCREMENT,
            scan_id         INTEGER NOT NULL REFERENCES scans(id),
            timestamp       TEXT    NOT NULL,
            instruments     TEXT    NOT NULL,
            implied_probs   TEXT    NOT NULL,
            copula_probs    TEXT    NOT NULL,
            discrepancy     REAL    NOT NULL,
            confidence      REAL    NOT NULL,
            reasoning       TEXT    NOT NULL,
            notified        INTEGER NOT NULL DEFAULT 0,
            tail_dependence TEXT,
            variance_reduction REAL,
            is_tail_regime  INTEGER NOT NULL DEFAULT 0,
            is_instruments  TEXT
        );

        -- Brier score calibration: every finding is a prediction.
        -- When we know the outcome, we record it here.
        CREATE TABLE IF NOT EXISTS predictions (
            id              INTEGER PRIMARY KEY AUTOINCREMENT,
            finding_id      INTEGER NOT NULL REFERENCES findings(id),
            cluster_name    TEXT    NOT NULL,
            instrument      TEXT    NOT NULL,
            predicted_prob  REAL    NOT NULL,
            timestamp       TEXT    NOT NULL,
            outcome         REAL,
            outcome_timestamp TEXT,
            resolved        INTEGER NOT NULL DEFAULT 0
        );

        -- Weekly Brier score snapshots
        CREATE TABLE IF NOT EXISTS brier_scores (
            id              INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp       TEXT    NOT NULL,
            cluster_name    TEXT    NOT NULL,
            brier_score     REAL    NOT NULL,
            n_predictions   INTEGER NOT NULL,
            n_resolved      INTEGER NOT NULL,
            target          REAL    NOT NULL DEFAULT 0.10
        );

        CREATE TABLE IF NOT EXISTS errors (
            id              INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp       TEXT    NOT NULL,
            component       TEXT    NOT NULL,
            message         TEXT    NOT NULL,
            details         TEXT
        );

        CREATE INDEX IF NOT EXISTS idx_scans_ts ON scans(timestamp);
        CREATE INDEX IF NOT EXISTS idx_findings_ts ON findings(timestamp);
        CREATE INDEX IF NOT EXISTS idx_findings_disc ON findings(discrepancy);
        CREATE INDEX IF NOT EXISTS idx_predictions_cluster ON predictions(cluster_name);
        CREATE INDEX IF NOT EXISTS idx_predictions_unresolved
            ON predictions(resolved) WHERE resolved = 0;
        CREATE INDEX IF NOT EXISTS idx_brier_ts ON brier_scores(timestamp);
    """)
    conn.commit()


# ── Scan logging ─────────────────────────────────────────────────────────────

def log_scan(cluster_name: str, instruments: list[str], mispricings_found: int,
             status: str = "ok", error_message: str | None = None,
             claude_reasoning: str | None = None) -> int:
    """Log a scan. Returns the scan id."""
    conn = _get_conn()
    cur = conn.execute(
        """INSERT INTO scans (timestamp, cluster_name, instruments, mispricings_found,
                              status, error_message, claude_reasoning)
           VALUES (?, ?, ?, ?, ?, ?, ?)""",
        (
            datetime.now(timezone.utc).isoformat(),
            cluster_name,
            json.dumps(instruments),
            mispricings_found,
            status,
            error_message,
            claude_reasoning,
        ),
    )
    conn.commit()
    return cur.lastrowid


# ── Finding logging ──────────────────────────────────────────────────────────

def log_finding(scan_id: int, instruments: list[str], implied_probs: dict,
                copula_probs: dict, discrepancy: float, confidence: float,
                reasoning: str, tail_dependence: dict | None = None,
                variance_reduction: float | None = None,
                is_tail_regime: bool = False,
                is_instruments: list[str] | None = None) -> int:
    """Log a mispricing finding. Returns the finding id."""
    conn = _get_conn()
    cur = conn.execute(
        """INSERT INTO findings (scan_id, timestamp, instruments, implied_probs,
                                 copula_probs, discrepancy, confidence, reasoning,
                                 tail_dependence, variance_reduction, is_tail_regime,
                                 is_instruments)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        (
            scan_id,
            datetime.now(timezone.utc).isoformat(),
            json.dumps(instruments),
            json.dumps(implied_probs),
            json.dumps(copula_probs),
            discrepancy,
            confidence,
            reasoning,
            json.dumps(tail_dependence) if tail_dependence else None,
            variance_reduction,
            1 if is_tail_regime else 0,
            json.dumps(is_instruments) if is_instruments else None,
        ),
    )
    finding_id = cur.lastrowid
    conn.commit()

    # Auto-create prediction records for Brier tracking
    _create_predictions_from_finding(finding_id, scan_id, instruments, implied_probs)

    return finding_id


def _create_predictions_from_finding(finding_id: int, scan_id: int,
                                     instruments: list[str], implied_probs: dict):
    """Every finding is a prediction. Record each instrument's predicted probability."""
    conn = _get_conn()
    # Get cluster name from the scan
    row = conn.execute("SELECT cluster_name FROM scans WHERE id = ?", (scan_id,)).fetchone()
    cluster_name = row["cluster_name"] if row else "unknown"

    now = datetime.now(timezone.utc).isoformat()
    for inst in instruments:
        prob = implied_probs.get(inst)
        if prob is not None:
            conn.execute(
                """INSERT INTO predictions (finding_id, cluster_name, instrument,
                                            predicted_prob, timestamp)
                   VALUES (?, ?, ?, ?, ?)""",
                (finding_id, cluster_name, inst, prob, now),
            )
    conn.commit()


def mark_notified(finding_id: int):
    conn = _get_conn()
    conn.execute("UPDATE findings SET notified = 1 WHERE id = ?", (finding_id,))
    conn.commit()


# ── Brier score ──────────────────────────────────────────────────────────────

def record_outcome(prediction_id: int, outcome: float):
    """Record the actual outcome for a prediction (0 or 1)."""
    conn = _get_conn()
    conn.execute(
        """UPDATE predictions SET outcome = ?, outcome_timestamp = ?, resolved = 1
           WHERE id = ?""",
        (outcome, datetime.now(timezone.utc).isoformat(), prediction_id),
    )
    conn.commit()


def compute_brier_score(cluster_name: str | None = None) -> dict:
    """Compute running Brier score for resolved predictions.

    Returns dict with brier_score, n_predictions, n_resolved, on_target.
    Target is below 0.10.
    """
    conn = _get_conn()
    if cluster_name:
        rows = conn.execute(
            """SELECT predicted_prob, outcome FROM predictions
               WHERE cluster_name = ? AND resolved = 1""",
            (cluster_name,),
        ).fetchall()
    else:
        rows = conn.execute(
            "SELECT predicted_prob, outcome FROM predictions WHERE resolved = 1"
        ).fetchall()

    if not rows:
        return {
            "brier_score": None,
            "n_predictions": 0,
            "n_resolved": 0,
            "on_target": None,
            "target": 0.10,
        }

    predictions = [r["predicted_prob"] for r in rows]
    outcomes = [r["outcome"] for r in rows]
    bs = float(brier_score(predictions, outcomes))

    # Also count total predictions (including unresolved)
    if cluster_name:
        total = conn.execute(
            "SELECT COUNT(*) as c FROM predictions WHERE cluster_name = ?",
            (cluster_name,),
        ).fetchone()["c"]
    else:
        total = conn.execute("SELECT COUNT(*) as c FROM predictions").fetchone()["c"]

    return {
        "brier_score": round(bs, 4),
        "n_predictions": total,
        "n_resolved": len(rows),
        "on_target": bs < 0.10,
        "target": 0.10,
    }


def snapshot_brier_scores():
    """Take a snapshot of Brier scores for all clusters. Call weekly."""
    conn = _get_conn()
    clusters = conn.execute(
        "SELECT DISTINCT cluster_name FROM predictions WHERE resolved = 1"
    ).fetchall()

    now = datetime.now(timezone.utc).isoformat()
    results = {}

    for row in clusters:
        cname = row["cluster_name"]
        bs_data = compute_brier_score(cname)
        if bs_data["brier_score"] is not None:
            conn.execute(
                """INSERT INTO brier_scores (timestamp, cluster_name, brier_score,
                                             n_predictions, n_resolved)
                   VALUES (?, ?, ?, ?, ?)""",
                (now, cname, bs_data["brier_score"],
                 bs_data["n_predictions"], bs_data["n_resolved"]),
            )
            results[cname] = bs_data

    conn.commit()
    return results


def get_brier_history(cluster_name: str, limit: int = 52) -> list[dict]:
    """Get Brier score history for a cluster (weekly snapshots)."""
    conn = _get_conn()
    rows = conn.execute(
        """SELECT * FROM brier_scores WHERE cluster_name = ?
           ORDER BY timestamp DESC LIMIT ?""",
        (cluster_name, limit),
    ).fetchall()
    return [dict(r) for r in rows]


def get_unresolved_predictions(hours: int = 168) -> list[dict]:
    """Get predictions that need outcomes recorded."""
    conn = _get_conn()
    rows = conn.execute(
        """SELECT * FROM predictions
           WHERE resolved = 0 AND timestamp > datetime('now', ?)
           ORDER BY timestamp ASC""",
        (f"-{hours} hours",),
    ).fetchall()
    return [dict(r) for r in rows]


# ── Error logging ────────────────────────────────────────────────────────────

def log_error(component: str, message: str, details: str | None = None):
    """Log an error for diagnostics."""
    conn = _get_conn()
    conn.execute(
        "INSERT INTO errors (timestamp, component, message, details) VALUES (?, ?, ?, ?)",
        (datetime.now(timezone.utc).isoformat(), component, message, details),
    )
    conn.commit()


# ── Query helpers ────────────────────────────────────────────────────────────

def get_recent_findings(hours: int = 24, limit: int = 50) -> list[dict]:
    """Retrieve recent findings for context."""
    conn = _get_conn()
    rows = conn.execute(
        """SELECT * FROM findings
           WHERE timestamp > datetime('now', ?)
           ORDER BY timestamp DESC LIMIT ?""",
        (f"-{hours} hours", limit),
    ).fetchall()
    return [dict(r) for r in rows]


def get_scan_stats(hours: int = 24) -> dict:
    """Summary stats for recent scans."""
    conn = _get_conn()
    row = conn.execute(
        """SELECT COUNT(*) as total_scans,
                  SUM(mispricings_found) as total_mispricings,
                  SUM(CASE WHEN status != 'ok' THEN 1 ELSE 0 END) as error_scans
           FROM scans WHERE timestamp > datetime('now', ?)""",
        (f"-{hours} hours",),
    ).fetchone()
    return dict(row) if row else {"total_scans": 0, "total_mispricings": 0, "error_scans": 0}
