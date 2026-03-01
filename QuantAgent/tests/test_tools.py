"""Tests for tool definitions and dispatch.

These tests verify:
1. All tool definitions have matching handlers
2. Tool schemas are valid
3. Brier score tool works correctly
4. Phase 2 stubs return proper errors
"""

import json
import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from quantagent.tools import TOOL_DEFINITIONS, ToolExecutor
from quantagent import database


class TestToolDefinitions:

    def test_all_tools_have_handlers(self):
        """Every tool in TOOL_DEFINITIONS must have a _tool_<name> handler."""
        executor = ToolExecutor.__new__(ToolExecutor)
        for tool in TOOL_DEFINITIONS:
            handler_name = f"_tool_{tool['name']}"
            assert hasattr(executor, handler_name), (
                f"Tool '{tool['name']}' defined but no handler '{handler_name}' found"
            )

    def test_all_tools_have_required_fields(self):
        for tool in TOOL_DEFINITIONS:
            assert "name" in tool
            assert "description" in tool
            assert "input_schema" in tool
            assert tool["input_schema"]["type"] == "object"

    def test_no_duplicate_tool_names(self):
        names = [t["name"] for t in TOOL_DEFINITIONS]
        assert len(names) == len(set(names)), "Duplicate tool names found"

    def test_tool_count(self):
        """Sanity check: we expect 9 tools (7 active + 2 phase 2 stubs)."""
        assert len(TOOL_DEFINITIONS) == 9


class TestToolExecution:

    @pytest.fixture
    def executor(self):
        """Create a ToolExecutor with mocked dependencies."""
        market = AsyncMock()
        copula = MagicMock()
        clusters = {}
        return ToolExecutor(market, copula, clusters)

    @pytest.mark.asyncio
    async def test_unknown_tool(self, executor):
        result = await executor.execute("nonexistent_tool", {})
        data = json.loads(result)
        assert "error" in data
        assert "Unknown tool" in data["error"]

    @pytest.mark.asyncio
    async def test_log_finding_tool(self, executor):
        """log_finding should persist to DB and return finding_id."""
        # First create a scan
        scan_id = database.log_scan("uk_macro", ["FTSE100"], 0)

        result = await executor.execute("log_finding", {
            "scan_id": scan_id,
            "instruments": ["FTSE100"],
            "implied_probs": {"FTSE100": 0.65},
            "copula_probs": {"FTSE100": 0.55},
            "discrepancy": 0.10,
            "confidence": 0.75,
            "reasoning": "Test mispricing",
        })
        data = json.loads(result)
        assert data["logged"] is True
        assert "finding_id" in data
        assert "Brier" in data["note"]

    @pytest.mark.asyncio
    async def test_get_brier_scores_tool(self, executor):
        result = await executor.execute("get_brier_scores", {})
        data = json.loads(result)
        assert "brier_data" in data
        assert "interpretation" in data

    @pytest.mark.asyncio
    async def test_get_scan_history_tool(self, executor):
        result = await executor.execute("get_scan_history", {"hours": 24})
        data = json.loads(result)
        assert "stats" in data
        assert "recent_findings" in data

    @pytest.mark.asyncio
    async def test_phase2_place_order_stub(self, executor):
        result = await executor.execute("place_order", {
            "instrument": "FTSE100",
            "size": 1,
            "side": "BUY",
        })
        data = json.loads(result)
        assert "PHASE 2" in data["error"]

    @pytest.mark.asyncio
    async def test_phase2_estimate_market_impact_stub(self, executor):
        result = await executor.execute("estimate_market_impact", {
            "instrument": "FTSE100",
            "proposed_size": 1,
            "side": "BUY",
            "current_price": 100.0,
        })
        data = json.loads(result)
        assert "PHASE 2" in data["error"]

    @pytest.mark.asyncio
    async def test_send_notification_tool(self, executor):
        """Notification should handle missing Telegram config gracefully."""
        with patch("quantagent.notifications.send_message", new_callable=AsyncMock) as mock_send:
            mock_send.return_value = False
            result = await executor.execute("send_notification", {
                "message": "Test notification",
            })
            data = json.loads(result)
            # Should not crash even if Telegram is not configured
            assert isinstance(data, dict)

    @pytest.mark.asyncio
    async def test_tool_error_handling(self, executor):
        """Tools should catch exceptions and return error JSON."""
        # Force an error by making market client raise
        executor.market.get_snapshot.side_effect = Exception("IB disconnected")
        executor.clusters["uk_macro"] = MagicMock()
        executor._qualified["uk_macro"] = {"FTSE100": MagicMock()}

        result = await executor.execute("get_market_data", {"cluster_name": "uk_macro"})
        data = json.loads(result)
        assert "error" in data
