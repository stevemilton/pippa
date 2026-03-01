"""QuantAgent main entry point — scan loop with weekly Brier summary."""

import asyncio
import logging
import signal
import sys
from datetime import datetime, timezone

from quantagent.config import config
from quantagent.market_data import MarketDataClient
from quantagent.copula import CopulaEngine
from quantagent.instruments import get_active_clusters
from quantagent.tools import ToolExecutor
from quantagent.agent import Agent
from quantagent import database, notifications

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-8s [%(name)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("quantagent.log"),
    ],
)
logger = logging.getLogger("quantagent")

# Quiet down noisy libraries
logging.getLogger("ib_insync").setLevel(logging.WARNING)
logging.getLogger("httpx").setLevel(logging.WARNING)


class Scanner:
    """Main scan loop that coordinates the agent across all clusters."""

    def __init__(self):
        self.market_client = MarketDataClient()
        self.copula_engine = CopulaEngine()
        self.clusters = {}
        self.agent: Agent | None = None
        self._running = False
        self._scan_count = 0
        self._last_weekly_summary: datetime | None = None

    async def start(self):
        """Initialize and run the scan loop."""
        logger.info("QuantAgent starting up...")
        logger.info("Copula: Student-t (nu=4) — Gaussian EXCLUDED")
        logger.info("Tail IS threshold: <5%% probability")
        logger.info("Variance reduction: antithetic + stratified (target 100-500x)")
        config.validate()

        # Build instrument clusters
        active_clusters = get_active_clusters()
        self.clusters = {c.name: c for c in active_clusters}
        logger.info("Active clusters: %s", list(self.clusters.keys()))

        # Connect to IBKR
        try:
            await self.market_client.connect()
        except Exception as e:
            logger.error("Failed to connect to IB Gateway: %s", e)
            await notifications.send_message(
                notifications.format_error_alert("startup", f"IB Gateway connection failed: {e}")
            )
            raise

        # Initialize tool executor and agent
        executor = ToolExecutor(self.market_client, self.copula_engine, self.clusters)
        self.agent = Agent(executor)

        # Send startup notification
        await notifications.send_message(
            notifications.format_startup_message(
                list(self.clusters.keys()), config.paper_trading
            )
        )

        # Run scan loop
        self._running = True
        logger.info(
            "Scan loop starting — interval=%dm, threshold=%.3f, paper=%s",
            config.scan_interval_minutes, config.mispricing_threshold, config.paper_trading,
        )

        while self._running:
            await self._scan_all_clusters()
            self._scan_count += 1

            # Check if it's time for the weekly summary (every Sunday at ~20:00 UTC)
            await self._maybe_send_weekly_summary()

            logger.info("Sleeping %d minutes until next scan...", config.scan_interval_minutes)
            try:
                await asyncio.sleep(config.scan_interval_minutes * 60)
            except asyncio.CancelledError:
                break

    async def _scan_all_clusters(self):
        """Run a scan across all active clusters."""
        for cluster_name in self.clusters:
            try:
                logger.info("Scanning cluster: %s", cluster_name)
                result = await self.agent.scan_cluster(cluster_name)
                logger.info(
                    "Cluster %s: %d mispricings found (%.1fs)",
                    cluster_name,
                    result["mispricings_found"],
                    result["elapsed_seconds"],
                )
            except Exception as e:
                logger.error("Scan failed for %s: %s", cluster_name, e, exc_info=True)
                database.log_scan(
                    cluster_name=cluster_name,
                    instruments=list(self.clusters[cluster_name].contracts.keys()),
                    mispricings_found=0,
                    status="error",
                    error_message=str(e),
                )
                database.log_error("scanner", f"Cluster {cluster_name} scan failed: {e}")
                try:
                    await notifications.send_message(
                        notifications.format_error_alert(
                            "scanner", f"Cluster {cluster_name} failed: {e}"
                        )
                    )
                except Exception:
                    pass

    async def _maybe_send_weekly_summary(self):
        """Send weekly Brier score summary on Sundays around 20:00 UTC.

        Brier score is the only honest measure of whether the model has edge.
        Surface it in every weekly Telegram summary. Target below 0.10.
        """
        now = datetime.now(timezone.utc)

        # Sunday = weekday 6, target window 20:00-20:59 UTC
        # (before IB Gateway restart at 22:00).
        # Use a time window instead of exact hour match so short scan intervals
        # don't skip the window entirely.
        if now.weekday() != 6 or not (20 <= now.hour <= 20):
            return

        # Don't send twice in the same day
        if (self._last_weekly_summary
                and self._last_weekly_summary.date() == now.date()):
            return

        logger.info("Generating weekly Brier score summary...")

        try:
            # Snapshot Brier scores to the database
            cluster_brier = database.snapshot_brier_scores()
            scan_stats = database.get_scan_stats(hours=168)  # 7 days

            # Ask Claude to generate the full summary
            summary = await self.agent.generate_weekly_summary(
                list(self.clusters.keys())
            )

            if summary:
                await notifications.send_message(summary)
            else:
                # Fallback to formatted summary
                msg = notifications.format_weekly_summary(cluster_brier, scan_stats)
                await notifications.send_message(msg)

            self._last_weekly_summary = now
            logger.info("Weekly summary sent")

        except Exception as e:
            logger.error("Failed to generate weekly summary: %s", e, exc_info=True)
            database.log_error("scanner", f"Weekly summary failed: {e}")

    def stop(self):
        """Signal the scan loop to stop."""
        logger.info("Shutdown requested")
        self._running = False
        self.market_client.disconnect()


def main():
    scanner = Scanner()

    def handle_signal(sig, _frame):
        logger.info("Received signal %s", signal.Signals(sig).name)
        scanner.stop()

    signal.signal(signal.SIGTERM, handle_signal)
    signal.signal(signal.SIGINT, handle_signal)

    try:
        asyncio.run(scanner.start())
    except KeyboardInterrupt:
        logger.info("Interrupted — shutting down")
    except Exception as e:
        logger.critical("Fatal error: %s", e, exc_info=True)
        database.log_error("main", f"Fatal: {e}")
        sys.exit(1)
    finally:
        scanner.stop()
        logger.info("QuantAgent stopped")


if __name__ == "__main__":
    main()
