"""Claude agent orchestration via Anthropic tool_use API.

Claude is the brain: it calls tools to gather data, runs copula analysis,
interprets results, decides what to flag, and explains its reasoning.
"""

import json
import logging
import time

import anthropic

from quantagent.config import config
from quantagent.tools import TOOL_DEFINITIONS, ToolExecutor
from quantagent import database

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """\
You are QuantAgent, a quantitative trading scanner running on a Hetzner VPS \
connected to Interactive Brokers. Your job is to find correlated market \
mispricings using vine copula analysis.

MATHEMATICAL MODEL (non-negotiable):
- The copula engine uses Student-t with nu=4. Gaussian is EXCLUDED because it \
  sets tail dependence to zero and underestimates joint extreme outcomes by 2-5x. \
  The t(nu=4) gives ~18% tail dependence on correlated instruments. This is where \
  the edge lives.
- When any instrument is priced below 5% probability, the system automatically \
  switches from crude Monte Carlo to importance sampling with exponential tilting. \
  Crude MC on tail contracts gives useless estimates.
- Antithetic variates and stratified sampling are applied alongside the copula \
  analysis, targeting 100-500x variance reduction over crude MC. Check the \
  variance_reduction_factor field in results.

For each scan cycle, you will:
1. Fetch current market data for the given instrument cluster.
2. Run vine copula analysis (Student-t, nu=4) to detect joint probability \
   inconsistencies.
3. The mispricing score is the divergence between what each market implies \
   independently vs what the joint t-copula model says given the dependency \
   structure. Flag anything above MISPRICING_THRESHOLD.
4. For any mispricing found:
   a. Log it to the database (this auto-creates a prediction for Brier calibration).
   b. Send a Telegram notification explaining:
      - Which instruments are inconsistent
      - What each market implies the probability is (independently)
      - What the copula model says it should be (given dependencies)
      - Size of the discrepancy and confidence level
      - Tail dependence coefficients (lambda_L) for the pair
      - Whether importance sampling was triggered (and for which instruments)
      - Variance reduction factor achieved
      - Your plain English reasoning: why this matters and what to do about it
5. If no mispricing is found, report the scan as clean — no notification needed.

BRIER SCORE CALIBRATION:
Every mispricing you flag is a prediction. Outcomes will be recorded when known. \
Running Brier score is computed per cluster — target is below 0.10. This is the \
only honest measure of whether the model has edge. When asked for weekly summaries, \
always include Brier scores.

Guidelines:
- Be precise with numbers. Show 4 decimal places for probabilities.
- Your reasoning should be useful to a trader — explain the economic logic, \
  not just the statistical finding.
- Consider whether a discrepancy reflects a real opportunity or could be \
  caused by stale data, illiquid markets, or market microstructure noise.
- When confidence is low, say so explicitly and explain why.
- When tail dependence is high (>15%), emphasize the joint tail risk.
- When importance sampling was triggered, note which instruments and why.
- Never invent data. If a tool returns an error, report that honestly.

Current configuration:
- Mispricing threshold: {threshold}
- Paper trading mode: {paper}
- Copula: Student-t, nu=4 (Gaussian excluded)
"""

MAX_TOOL_ROUNDS = 10


class Agent:
    """Claude agent that orchestrates scanning via tool_use."""

    def __init__(self, tool_executor: ToolExecutor):
        self.client = anthropic.Anthropic(api_key=config.anthropic_api_key)
        self.executor = tool_executor
        self.model = "claude-sonnet-4-6-20250514"

    async def scan_cluster(self, cluster_name: str) -> dict:
        """Run a full scan cycle for one cluster.

        Returns a summary dict with scan_id, mispricings_found, reasoning.
        """
        start = time.monotonic()

        system = SYSTEM_PROMPT.format(
            threshold=config.mispricing_threshold,
            paper=config.paper_trading,
        )

        user_message = (
            f"Run a scan cycle for the '{cluster_name}' cluster.\n\n"
            f"1. Fetch current market data.\n"
            f"2. Run the copula analysis (Student-t nu=4, Gaussian excluded).\n"
            f"3. If you find a mispricing above threshold ({config.mispricing_threshold}), "
            f"log it and send a Telegram notification with your full analysis including "
            f"tail dependence, importance sampling status, and variance reduction factor.\n"
            f"4. If nothing found, just confirm the scan was clean."
        )

        messages = [{"role": "user", "content": user_message}]

        mispricings_found = 0
        reasoning = ""
        scan_id = None

        for round_num in range(MAX_TOOL_ROUNDS):
            logger.debug("Agent round %d for cluster %s", round_num + 1, cluster_name)

            try:
                response = self.client.messages.create(
                    model=self.model,
                    max_tokens=4096,
                    system=system,
                    tools=TOOL_DEFINITIONS,
                    messages=messages,
                )
            except anthropic.APIError as e:
                logger.error("Anthropic API error: %s", e)
                database.log_error("agent", f"API error: {e}")
                break

            assistant_content = response.content
            messages.append({"role": "assistant", "content": assistant_content})

            if response.stop_reason == "end_turn":
                for block in assistant_content:
                    if hasattr(block, "text"):
                        reasoning = block.text
                break

            tool_results = []
            for block in assistant_content:
                if block.type == "tool_use":
                    tool_name = block.name
                    tool_input = block.input
                    logger.info("Agent calling tool: %s(%s)", tool_name, json.dumps(tool_input)[:200])

                    result_str = await self.executor.execute(tool_name, tool_input)

                    if tool_name == "log_finding":
                        result_data = json.loads(result_str)
                        if result_data.get("logged"):
                            mispricings_found += 1

                    tool_results.append({
                        "type": "tool_result",
                        "tool_use_id": block.id,
                        "content": result_str,
                    })

            if tool_results:
                messages.append({"role": "user", "content": tool_results})
            else:
                break

        elapsed = time.monotonic() - start

        scan_id = database.log_scan(
            cluster_name=cluster_name,
            instruments=list(self.executor.clusters.get(cluster_name, {}).contracts.keys())
            if cluster_name in self.executor.clusters else [],
            mispricings_found=mispricings_found,
            status="ok",
            claude_reasoning=reasoning[:2000] if reasoning else None,
        )

        logger.info(
            "Scan complete: cluster=%s mispricings=%d elapsed=%.1fs",
            cluster_name, mispricings_found, elapsed,
        )

        return {
            "scan_id": scan_id,
            "cluster_name": cluster_name,
            "mispricings_found": mispricings_found,
            "reasoning": reasoning,
            "elapsed_seconds": round(elapsed, 1),
        }

    async def generate_weekly_summary(self, cluster_names: list[str]) -> str:
        """Generate a weekly summary including Brier scores. Called by the main loop."""
        system = SYSTEM_PROMPT.format(
            threshold=config.mispricing_threshold,
            paper=config.paper_trading,
        )

        user_message = (
            "Generate a weekly summary for Telegram. For each cluster, include:\n"
            "1. Number of scans and mispricings found this week\n"
            "2. The running Brier score (get it via get_brier_scores tool)\n"
            "3. Whether we're on target (below 0.10)\n"
            "4. Any notable patterns or recurring mispricings\n"
            "5. Your assessment of model performance\n\n"
            f"Clusters: {', '.join(cluster_names)}\n\n"
            "Format the output as an HTML Telegram message. Always include Brier "
            "scores — this is the only honest measure of edge."
        )

        messages = [{"role": "user", "content": user_message}]
        summary = ""

        for round_num in range(MAX_TOOL_ROUNDS):
            try:
                response = self.client.messages.create(
                    model=self.model,
                    max_tokens=4096,
                    system=system,
                    tools=TOOL_DEFINITIONS,
                    messages=messages,
                )
            except anthropic.APIError as e:
                logger.error("Anthropic API error in weekly summary: %s", e)
                break

            assistant_content = response.content
            messages.append({"role": "assistant", "content": assistant_content})

            if response.stop_reason == "end_turn":
                for block in assistant_content:
                    if hasattr(block, "text"):
                        summary = block.text
                break

            tool_results = []
            for block in assistant_content:
                if block.type == "tool_use":
                    result_str = await self.executor.execute(block.name, block.input)
                    tool_results.append({
                        "type": "tool_result",
                        "tool_use_id": block.id,
                        "content": result_str,
                    })

            if tool_results:
                messages.append({"role": "user", "content": tool_results})
            else:
                break

        return summary
