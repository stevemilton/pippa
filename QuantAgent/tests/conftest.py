"""Shared fixtures for QuantAgent tests."""

import os
import sqlite3
import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest
import numpy as np


@pytest.fixture(autouse=True)
def _isolate_db(tmp_path):
    """Every test gets its own ephemeral SQLite database.

    Uses object.__setattr__ to bypass the frozen dataclass restriction
    on Config, and restores the original value after the test.
    """
    from quantagent.config import config
    import quantagent.database as db

    db_file = tmp_path / "test_quantagent.db"
    original_db_path = config.db_path

    # Bypass frozen dataclass
    object.__setattr__(config, "db_path", str(db_file))

    # Force database module to create a new connection on next call
    if hasattr(db._local, "conn"):
        try:
            db._local.conn.close()
        except Exception:
            pass
        del db._local.conn

    yield

    # Teardown: close connection and restore original path
    if hasattr(db._local, "conn"):
        try:
            db._local.conn.close()
        except Exception:
            pass
        del db._local.conn

    object.__setattr__(config, "db_path", original_db_path)


@pytest.fixture
def rng():
    """Deterministic RNG for reproducible tests."""
    return np.random.default_rng(42)


@pytest.fixture
def sample_returns(rng):
    """Generate 150 days of synthetic log returns for 4 correlated instruments.

    Uses a factor model so there's genuine correlation for the copula to find.
    """
    n = 150
    d = 4
    names = ["FTSE100", "GILT", "GBPUSD", "SHORT_STERLING"]

    # Common factor + idiosyncratic
    factor = rng.normal(0, 0.01, n)
    returns = {}
    betas = [0.8, -0.5, 0.3, -0.4]
    vols = [0.012, 0.008, 0.006, 0.005]

    for i, name in enumerate(names):
        idio = rng.normal(0, vols[i], n)
        returns[name] = (betas[i] * factor + idio).tolist()

    return returns


@pytest.fixture
def sample_returns_pair(rng):
    """Minimal 2-instrument returns for edge-case tests."""
    n = 120
    factor = rng.normal(0, 0.01, n)
    return {
        "A": (0.6 * factor + rng.normal(0, 0.008, n)).tolist(),
        "B": (-0.4 * factor + rng.normal(0, 0.008, n)).tolist(),
    }
