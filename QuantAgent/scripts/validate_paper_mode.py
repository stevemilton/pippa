#!/usr/bin/env python3
"""Paper-mode validation script.

Run this for 1 week before going live. It performs end-to-end checks:
1. Connects to IB Gateway (paper account)
2. Qualifies all contracts in every cluster
3. Fetches market data snapshots
4. Runs one copula scan per cluster
5. Verifies database writes (scans, findings, predictions)
6. Checks Telegram delivery
7. Runs auto-resolver
8. Reports Brier score state

Usage:
    python scripts/validate_paper_mode.py [--skip-telegram] [--skip-copula]

Exit codes:
    0 = all checks passed
    1 = one or more checks failed
"""

import argparse
import asyncio
import json
import logging
import sys
import time
from pathlib import Path

# Ensure project root is on path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from quantagent.config import config
from quantagent.market_data import MarketDataClient
from quantagent.copula import CopulaEngine
from quantagent.instruments import get_active_clusters
from quantagent import database, notifications
from quantagent.outcomes import auto_resolve_predictions

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-8s [%(name)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("validate")

PASS = "\033[92m✓ PASS\033[0m"
FAIL = "\033[91m✗ FAIL\033[0m"
SKIP = "\033[93m○ SKIP\033[0m"


class ValidationRunner:
    def __init__(self, skip_telegram=False, skip_copula=False):
        self.skip_telegram = skip_telegram
        self.skip_copula = skip_copula
        self.results: list[tuple[str, bool, str]] = []
        self.market_client = MarketDataClient()
        self.copula_engine = CopulaEngine()
        self.clusters = {}

    def record(self, name: str, passed: bool, detail: str = ""):
        status = PASS if passed else FAIL
        logger.info("%s %s %s", status, name, detail)
        self.results.append((name, passed, detail))

    def record_skip(self, name: str, reason: str):
        logger.info("%s %s (%s)", SKIP, name, reason)

    async def run_all(self):
        logger.info("=" * 60)
        logger.info("QuantAgent Paper-Mode Validation")
        logger.info("=" * 60)
        logger.info("")

        # ── 1. Config check ────────────────────────────────────────
        logger.info("── Step 1: Configuration ──")
        self._check_config()

        # ── 2. Build clusters ──────────────────────────────────────
        logger.info("")
        logger.info("── Step 2: Instrument Clusters ──")
        self._check_clusters()

        # ── 3. IB Gateway connection ───────────────────────────────
        logger.info("")
        logger.info("── Step 3: IB Gateway Connection ──")
        await self._check_ib_connection()

        if not self.market_client._connected:
            logger.error("Cannot continue without IB Gateway connection")
            self._print_summary()
            return

        # ── 4. Contract qualification ──────────────────────────────
        logger.info("")
        logger.info("── Step 4: Contract Qualification ──")
        qualified = await self._check_contract_qualification()

        # ── 5. Market data snapshot ────────────────────────────────
        logger.info("")
        logger.info("── Step 5: Market Data Snapshot ──")
        await self._check_market_data(qualified)

        # ── 6. Copula analysis ─────────────────────────────────────
        if not self.skip_copula:
            logger.info("")
            logger.info("── Step 6: Copula Analysis ──")
            await self._check_copula_analysis(qualified)
        else:
            self.record_skip("Copula analysis", "skipped via --skip-copula")

        # ── 7. Database writes ─────────────────────────────────────
        logger.info("")
        logger.info("── Step 7: Database Integrity ──")
        self._check_database()

        # ── 8. Telegram ────────────────────────────────────────────
        if not self.skip_telegram:
            logger.info("")
            logger.info("── Step 8: Telegram Delivery ──")
            await self._check_telegram()
        else:
            self.record_skip("Telegram", "skipped via --skip-telegram")

        # ── 9. Auto-resolver ───────────────────────────────────────
        logger.info("")
        logger.info("── Step 9: Auto-Resolver ──")
        self._check_auto_resolver()

        # ── 10. Brier score state ──────────────────────────────────
        logger.info("")
        logger.info("── Step 10: Brier Score State ──")
        self._check_brier_state()

        # ── Done ───────────────────────────────────────────────────
        self.market_client.disconnect()
        self._print_summary()

    def _check_config(self):
        try:
            config.validate()
            self.record("Config: required env vars present", True)
        except EnvironmentError as e:
            self.record("Config: required env vars present", False, str(e))

        self.record(
            "Config: paper_trading is True",
            config.paper_trading,
            f"paper_trading={config.paper_trading}",
        )
        self.record(
            "Config: mispricing_threshold set",
            config.mispricing_threshold > 0,
            f"threshold={config.mispricing_threshold}",
        )

    def _check_clusters(self):
        try:
            clusters_list = get_active_clusters()
            self.clusters = {c.name: c for c in clusters_list}

            self.record(
                "Clusters: built successfully",
                len(self.clusters) > 0,
                f"clusters={list(self.clusters.keys())}",
            )

            for name, cluster in self.clusters.items():
                n = len(cluster.contracts)
                self.record(
                    f"Cluster '{name}': has contracts",
                    n > 0,
                    f"{n} instruments",
                )

                # Check futures have expiry
                for iname, contract in cluster.contracts.items():
                    ltdom = getattr(contract, "lastTradeDateOrContractMonth", None)
                    if ltdom:
                        self.record(
                            f"  {iname}: expiry set",
                            len(ltdom) == 6,
                            f"expiry={ltdom}",
                        )
        except Exception as e:
            self.record("Clusters: built successfully", False, str(e))

    async def _check_ib_connection(self):
        try:
            await self.market_client.connect()
            self.record(
                "IB Gateway: connected",
                True,
                f"{config.ibkr_host}:{config.ibkr_port}",
            )
        except Exception as e:
            self.record("IB Gateway: connected", False, str(e))

    async def _check_contract_qualification(self) -> dict:
        all_qualified = {}
        for name, cluster in self.clusters.items():
            try:
                q = await self.market_client.qualify_contracts(cluster.contracts)
                all_qualified[name] = q
                n_ok = len(q)
                n_total = len(cluster.contracts)
                self.record(
                    f"Qualify '{name}'",
                    n_ok > 0,
                    f"{n_ok}/{n_total} contracts qualified",
                )
                for iname in cluster.contracts:
                    if iname not in q:
                        logger.warning("  FAILED to qualify: %s", iname)
            except Exception as e:
                self.record(f"Qualify '{name}'", False, str(e))
        return all_qualified

    async def _check_market_data(self, qualified: dict):
        for name, contracts in qualified.items():
            if not contracts:
                continue
            try:
                snapshot = await self.market_client.get_snapshot(contracts)
                n_ok = len(snapshot)
                n_total = len(contracts)
                self.record(
                    f"Snapshot '{name}'",
                    n_ok > 0,
                    f"{n_ok}/{n_total} instruments have prices",
                )
                for iname, data in snapshot.items():
                    mid = data.get("mid")
                    spread = data.get("spread")
                    logger.info("  %s: mid=%.4f spread=%s", iname, mid or 0, spread)
            except Exception as e:
                self.record(f"Snapshot '{name}'", False, str(e))

    async def _check_copula_analysis(self, qualified: dict):
        for name, contracts in qualified.items():
            if not contracts:
                continue
            try:
                returns = await self.market_client.get_historical_returns(
                    contracts, days=config.copula_lookback_days
                )
                self.record(
                    f"Historical data '{name}'",
                    len(returns) >= 2,
                    f"{len(returns)} instruments with return data",
                )

                if len(returns) >= 2:
                    t0 = time.monotonic()
                    result = self.copula_engine.analyze_cluster(returns)
                    elapsed = time.monotonic() - t0

                    if result:
                        self.record(
                            f"Copula '{name}'",
                            True,
                            f"max_disc={result.max_discrepancy:.4f} "
                            f"conf={result.confidence:.2f} "
                            f"vr={result.variance_reduction_factor:.0f}x "
                            f"tail_regime={result.is_tail_regime} "
                            f"({elapsed:.1f}s)",
                        )
                        logger.info("  Vine: %s", result.vine_structure.split('\n')[0])
                        for inst, disc in result.discrepancies.items():
                            logger.info("  %s: disc=%.4f", inst, disc)
                    else:
                        self.record(f"Copula '{name}'", False, "returned None")
            except Exception as e:
                self.record(f"Copula '{name}'", False, str(e))

    def _check_database(self):
        try:
            # Write a test scan
            scan_id = database.log_scan(
                "validation_test", ["TEST"], 0, status="ok",
                claude_reasoning="Validation run",
            )
            self.record("DB: write scan", isinstance(scan_id, int), f"id={scan_id}")

            # Read it back
            stats = database.get_scan_stats(hours=1)
            self.record("DB: read stats", stats["total_scans"] > 0, json.dumps(stats))

            # Error logging
            database.log_error("validation", "Test error — ignore")
            self.record("DB: write error", True)

        except Exception as e:
            self.record("DB: integrity", False, str(e))

    async def _check_telegram(self):
        try:
            msg = (
                "<b>QuantAgent Validation</b>\n\n"
                "This is a test message from the paper-mode validation script.\n"
                "If you received this, Telegram delivery is working."
            )
            ok = await notifications.send_message(msg)
            self.record("Telegram: send test message", ok)
        except Exception as e:
            self.record("Telegram: send test message", False, str(e))

    def _check_auto_resolver(self):
        try:
            result = auto_resolve_predictions(lookback_hours=168)
            self.record(
                "Auto-resolver: runs without error",
                True,
                f"resolved={result['resolved']} skipped={result['skipped']}",
            )
        except Exception as e:
            self.record("Auto-resolver: runs without error", False, str(e))

    def _check_brier_state(self):
        try:
            bs = database.compute_brier_score()
            self.record(
                "Brier: compute runs",
                True,
                f"score={bs['brier_score']} resolved={bs['n_resolved']}/{bs['n_predictions']}",
            )
            if bs["brier_score"] is not None:
                status = "ON TARGET" if bs["on_target"] else "ABOVE TARGET"
                logger.info("  Overall Brier: %.4f (%s)", bs["brier_score"], status)
            else:
                logger.info("  No resolved predictions yet (expected for new install)")
        except Exception as e:
            self.record("Brier: compute runs", False, str(e))

    def _print_summary(self):
        logger.info("")
        logger.info("=" * 60)
        logger.info("VALIDATION SUMMARY")
        logger.info("=" * 60)

        passed = sum(1 for _, ok, _ in self.results if ok)
        failed = sum(1 for _, ok, _ in self.results if not ok)
        total = len(self.results)

        for name, ok, detail in self.results:
            status = PASS if ok else FAIL
            suffix = f" — {detail}" if detail else ""
            logger.info("  %s %s%s", status, name, suffix)

        logger.info("")
        logger.info("Results: %d/%d passed, %d failed", passed, total, failed)

        if failed == 0:
            logger.info("\033[92mAll checks passed — ready for paper trading!\033[0m")
        else:
            logger.info("\033[91m%d checks failed — fix before going live\033[0m", failed)

        sys.exit(0 if failed == 0 else 1)


def main():
    parser = argparse.ArgumentParser(
        description="QuantAgent paper-mode validation",
        prog="python scripts/validate_paper_mode.py",
    )
    parser.add_argument(
        "--skip-telegram",
        action="store_true",
        help="Skip Telegram delivery check",
    )
    parser.add_argument(
        "--skip-copula",
        action="store_true",
        help="Skip copula analysis (faster validation)",
    )
    args = parser.parse_args()

    runner = ValidationRunner(
        skip_telegram=args.skip_telegram,
        skip_copula=args.skip_copula,
    )
    asyncio.run(runner.run_all())


if __name__ == "__main__":
    main()
