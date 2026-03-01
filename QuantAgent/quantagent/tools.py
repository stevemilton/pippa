"""Tool definitions and execution for the Claude agent.

Each tool is defined as an Anthropic tool_use schema + an execute function.
The agent calls these tools; this module dispatches and returns results.
"""

import json
import logging
from typing import Any

from quantagent.market_data import MarketDataClient
from quantagent.copula import CopulaEngine, CopulaResult
from quantagent.instruments import InstrumentCluster
from quantagent import database, notifications
from quantagent.config import config

logger = logging.getLogger(__name__)

# ── Tool schemas for the Anthropic API ────────────────────────────────────────

TOOL_DEFINITIONS = [
    {
        "name": "get_market_data",
        "description": (
            "Fetch current market prices for a list of instruments in a cluster. "
            "Returns bid, ask, last, mid, spread, and volume for each instrument."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "cluster_name": {
                    "type": "string",
                    "description": "Name of the instrument cluster to fetch data for",
                },
            },
            "required": ["cluster_name"],
        },
    },
    {
        "name": "run_copula_analysis",
        "description": (
            "Run Student-t vine copula analysis (nu=4, Gaussian EXCLUDED) on a cluster. "
            "Fetches historical returns, fits a vine copula with mandatory tail dependence, "
            "and identifies probability inconsistencies. Auto-switches to importance sampling "
            "for instruments priced below 5%. Applies antithetic variates + stratified sampling "
            "for 100-500x variance reduction. Returns implied probs, copula probs, discrepancies, "
            "tail dependence coefficients, variance reduction factor, and confidence."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "cluster_name": {
                    "type": "string",
                    "description": "Name of the instrument cluster to analyze",
                },
            },
            "required": ["cluster_name"],
        },
    },
    {
        "name": "get_positions",
        "description": "Get current open positions from the IBKR account.",
        "input_schema": {
            "type": "object",
            "properties": {},
        },
    },
    {
        "name": "log_finding",
        "description": (
            "Log a mispricing finding to the database. Every finding is automatically "
            "recorded as a prediction for Brier score calibration. Call this when you've "
            "identified a meaningful inconsistency that exceeds the threshold."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "scan_id": {
                    "type": "integer",
                    "description": "The scan ID this finding belongs to",
                },
                "instruments": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "List of instrument names involved",
                },
                "implied_probs": {
                    "type": "object",
                    "description": "Market-implied probabilities per instrument",
                },
                "copula_probs": {
                    "type": "object",
                    "description": "Copula model probabilities per instrument",
                },
                "discrepancy": {
                    "type": "number",
                    "description": "Maximum discrepancy value",
                },
                "confidence": {
                    "type": "number",
                    "description": "Confidence level 0-1",
                },
                "reasoning": {
                    "type": "string",
                    "description": "Plain English explanation of the finding",
                },
                "tail_dependence": {
                    "type": "object",
                    "description": "Pairwise tail dependence coefficients",
                },
                "variance_reduction": {
                    "type": "number",
                    "description": "Variance reduction factor achieved",
                },
                "is_tail_regime": {
                    "type": "boolean",
                    "description": "Whether importance sampling was triggered",
                },
                "is_instruments": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Instruments that triggered importance sampling",
                },
            },
            "required": ["scan_id", "instruments", "implied_probs", "copula_probs",
                         "discrepancy", "confidence", "reasoning"],
        },
    },
    {
        "name": "send_notification",
        "description": (
            "Send a Telegram notification message. Use for mispricing alerts "
            "and important status updates."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "message": {
                    "type": "string",
                    "description": "HTML-formatted message to send via Telegram",
                },
            },
            "required": ["message"],
        },
    },
    {
        "name": "get_scan_history",
        "description": "Retrieve recent scan results and findings for context.",
        "input_schema": {
            "type": "object",
            "properties": {
                "hours": {
                    "type": "integer",
                    "description": "Number of hours to look back (default 24)",
                    "default": 24,
                },
            },
        },
    },
    {
        "name": "get_brier_scores",
        "description": (
            "Get running Brier score calibration data. Every mispricing finding is a "
            "prediction. This tool returns the Brier score per cluster and overall. "
            "Target is below 0.10. This is the only honest measure of whether the "
            "model has edge. Include in every weekly Telegram summary."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "cluster_name": {
                    "type": "string",
                    "description": "Cluster to get Brier score for. Omit for overall.",
                },
            },
        },
    },
    # ── Phase 2 stubs ─────────────────────────────────────────────────────────
    {
        "name": "place_order",
        "description": "[PHASE 2 — NOT YET IMPLEMENTED] Place a trade order via IBKR.",
        "input_schema": {
            "type": "object",
            "properties": {
                "instrument": {"type": "string"},
                "size": {"type": "number"},
                "side": {"type": "string", "enum": ["BUY", "SELL"]},
            },
            "required": ["instrument", "size", "side"],
        },
    },
    {
        "name": "estimate_market_impact",
        "description": (
            "[PHASE 2 — NOT YET IMPLEMENTED] Pre-trade ABM check. Uses "
            "PredictionMarketABM to estimate market impact before execution. "
            "Prevents moving the price against ourselves on thin markets."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "instrument": {"type": "string"},
                "proposed_size": {"type": "number"},
                "side": {"type": "string", "enum": ["BUY", "SELL"]},
                "current_price": {"type": "number"},
            },
            "required": ["instrument", "proposed_size", "side", "current_price"],
        },
    },
]


class ToolExecutor:
    """Executes tool calls from the Claude agent."""

    def __init__(
        self,
        market_client: MarketDataClient,
        copula_engine: CopulaEngine,
        clusters: dict[str, InstrumentCluster],
    ):
        self.market = market_client
        self.copula = copula_engine
        self.clusters = clusters
        self._qualified: dict[str, dict] = {}

    async def execute(self, tool_name: str, tool_input: dict[str, Any]) -> str:
        """Dispatch a tool call and return a JSON string result."""
        try:
            handler = getattr(self, f"_tool_{tool_name}", None)
            if handler is None:
                return json.dumps({"error": f"Unknown tool: {tool_name}"})
            result = await handler(tool_input)
            return json.dumps(result, default=str)
        except Exception as e:
            logger.error("Tool %s failed: %s", tool_name, e, exc_info=True)
            database.log_error("tools", f"Tool {tool_name} failed: {e}")
            return json.dumps({"error": str(e)})

    async def _ensure_qualified(self, cluster_name: str) -> dict:
        """Qualify contracts for a cluster (cached)."""
        if cluster_name not in self._qualified:
            cluster = self.clusters.get(cluster_name)
            if not cluster:
                raise ValueError(f"Unknown cluster: {cluster_name}")
            self._qualified[cluster_name] = await self.market.qualify_contracts(
                cluster.contracts
            )
        return self._qualified[cluster_name]

    async def _tool_get_market_data(self, inp: dict) -> dict:
        cluster_name = inp["cluster_name"]
        contracts = await self._ensure_qualified(cluster_name)
        if not contracts:
            return {"error": f"No qualified contracts for cluster {cluster_name}"}
        snapshot = await self.market.get_snapshot(contracts)
        return {"cluster": cluster_name, "data": snapshot}

    async def _tool_run_copula_analysis(self, inp: dict) -> dict:
        cluster_name = inp["cluster_name"]
        contracts = await self._ensure_qualified(cluster_name)
        if not contracts:
            return {"error": f"No qualified contracts for cluster {cluster_name}"}

        returns = await self.market.get_historical_returns(
            contracts, days=config.copula_lookback_days
        )
        if len(returns) < 2:
            return {"error": "Insufficient instruments with data for copula analysis"}

        result = self.copula.analyze_cluster(returns)
        if result is None:
            return {"status": "no_result", "message": "Copula analysis returned no result"}

        return {
            "status": "ok",
            "copula_type": "Student-t (nu=4) — Gaussian EXCLUDED",
            "instruments": result.instruments,
            "implied_probs": result.implied_probs,
            "copula_probs": result.copula_probs,
            "discrepancies": result.discrepancies,
            "max_discrepancy": result.max_discrepancy,
            "confidence": result.confidence,
            "vine_structure": result.vine_structure,
            "log_likelihood": result.log_likelihood,
            "aic": result.aic,
            "tail_dependence": result.tail_dependence,
            "variance_reduction_factor": result.variance_reduction_factor,
            "is_tail_regime": result.is_tail_regime,
            "is_instruments": result.is_instruments,
            "threshold": config.mispricing_threshold,
            "exceeds_threshold": result.max_discrepancy >= config.mispricing_threshold,
        }

    async def _tool_get_positions(self, _inp: dict) -> dict:
        positions = await self.market.get_positions()
        return {"positions": positions}

    async def _tool_log_finding(self, inp: dict) -> dict:
        finding_id = database.log_finding(
            scan_id=inp["scan_id"],
            instruments=inp["instruments"],
            implied_probs=inp["implied_probs"],
            copula_probs=inp["copula_probs"],
            discrepancy=inp["discrepancy"],
            confidence=inp["confidence"],
            reasoning=inp["reasoning"],
            tail_dependence=inp.get("tail_dependence"),
            variance_reduction=inp.get("variance_reduction"),
            is_tail_regime=inp.get("is_tail_regime", False),
            is_instruments=inp.get("is_instruments"),
        )
        return {"finding_id": finding_id, "logged": True,
                "note": "Prediction recorded for Brier score tracking"}

    async def _tool_send_notification(self, inp: dict) -> dict:
        success = await notifications.send_message(inp["message"])
        if success:
            return {"sent": True}
        return {"sent": False, "error": "Telegram send failed — check logs"}

    async def _tool_get_scan_history(self, inp: dict) -> dict:
        hours = inp.get("hours", 24)
        findings = database.get_recent_findings(hours=hours)
        stats = database.get_scan_stats(hours=hours)
        return {"stats": stats, "recent_findings": findings[:20]}

    async def _tool_get_brier_scores(self, inp: dict) -> dict:
        cluster_name = inp.get("cluster_name")
        bs_data = database.compute_brier_score(cluster_name)
        return {
            "brier_data": bs_data,
            "interpretation": (
                f"Brier score: {bs_data['brier_score']} "
                f"({'ON TARGET' if bs_data.get('on_target') else 'ABOVE TARGET'} — "
                f"target < 0.10). "
                f"Based on {bs_data['n_resolved']}/{bs_data['n_predictions']} resolved predictions."
                if bs_data["brier_score"] is not None
                else "No resolved predictions yet — need outcomes recorded."
            ),
        }

    async def _tool_place_order(self, _inp: dict) -> dict:
        return {
            "error": "PHASE 2 — not yet implemented. "
            "place_order is stubbed for future execution capability."
        }

    async def _tool_estimate_market_impact(self, _inp: dict) -> dict:
        return {
            "error": "PHASE 2 — not yet implemented. "
            "estimate_market_impact will use PredictionMarketABM to simulate "
            "price impact before execution on thin markets."
        }
