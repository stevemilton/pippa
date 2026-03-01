"""Telegram notification module."""

import logging
import httpx

from quantagent.config import config
from quantagent import database

logger = logging.getLogger(__name__)

TELEGRAM_API = "https://api.telegram.org"


async def send_message(text: str, parse_mode: str = "HTML") -> bool:
    """Send a message to the configured Telegram chat.

    Returns True on success, False on failure (with error logged).
    """
    if not config.telegram_bot_token or not config.telegram_chat_id:
        logger.warning("Telegram not configured — skipping notification")
        return False

    url = f"{TELEGRAM_API}/bot{config.telegram_bot_token}/sendMessage"
    # Telegram max message length is 4096 chars
    truncated = text[:4000] + "\n\n[truncated]" if len(text) > 4000 else text

    payload = {
        "chat_id": config.telegram_chat_id,
        "text": truncated,
        "parse_mode": parse_mode,
        "disable_web_page_preview": True,
    }

    try:
        async with httpx.AsyncClient(timeout=15) as client:
            resp = await client.post(url, json=payload)
            resp.raise_for_status()
            logger.info("Telegram notification sent (%d chars)", len(truncated))
            return True
    except httpx.HTTPStatusError as e:
        logger.error("Telegram API error %s: %s", e.response.status_code, e.response.text)
        database.log_error("notifications", f"Telegram HTTP {e.response.status_code}: {e.response.text}")
        return False
    except Exception as e:
        logger.error("Telegram send failed: %s", e)
        database.log_error("notifications", f"Telegram error: {e}")
        return False


def format_mispricing_alert(
    cluster_name: str,
    instruments: list[str],
    implied_probs: dict,
    copula_probs: dict,
    discrepancies: dict,
    max_discrepancy: float,
    confidence: float,
    reasoning: str,
    tail_dependence: dict | None = None,
    variance_reduction: float | None = None,
    is_tail_regime: bool = False,
    is_instruments: list[str] | None = None,
) -> str:
    """Format a mispricing finding into a Telegram-friendly HTML message."""
    lines = [
        f"<b>MISPRICING DETECTED — {cluster_name.upper()}</b>",
        f"<b>Model:</b> Student-t copula (nu=4)",
        "",
        f"<b>Discrepancy:</b> {max_discrepancy:.4f} ({max_discrepancy*100:.1f}%)",
        f"<b>Confidence:</b> {confidence:.1%}",
    ]

    if variance_reduction is not None:
        lines.append(f"<b>Variance reduction:</b> {variance_reduction:.0f}x")

    if is_tail_regime:
        is_names = ", ".join(is_instruments or [])
        lines.append(f"<b>Importance sampling:</b> YES ({is_names})")

    lines.extend(["", "<b>Instrument Details:</b>"])

    for inst in instruments:
        imp = implied_probs.get(inst, "N/A")
        cop = copula_probs.get(inst, "N/A")
        disc = discrepancies.get(inst, 0)
        imp_str = f"{imp:.4f}" if isinstance(imp, float) else str(imp)
        cop_str = f"{cop:.4f}" if isinstance(cop, float) else str(cop)
        arrow = "+" if disc > 0 else ""
        lines.append(f"  {inst}: implied={imp_str} copula={cop_str} ({arrow}{disc:.4f})")

    if tail_dependence:
        lines.extend(["", "<b>Tail Dependence (lambda_L):</b>"])
        for pair, td in tail_dependence.items():
            flag = " !!!" if td > 0.15 else ""
            lines.append(f"  {pair}: {td:.4f}{flag}")

    lines.extend(["", "<b>Analysis:</b>", reasoning])

    return "\n".join(lines)


def format_weekly_summary(
    cluster_brier: dict[str, dict],
    scan_stats: dict,
) -> str:
    """Format weekly summary with Brier scores for Telegram."""
    lines = [
        "<b>WEEKLY SUMMARY — QuantAgent</b>",
        "",
        f"<b>Total scans:</b> {scan_stats.get('total_scans', 0)}",
        f"<b>Mispricings flagged:</b> {scan_stats.get('total_mispricings', 0)}",
        f"<b>Errors:</b> {scan_stats.get('error_scans', 0)}",
        "",
        "<b>BRIER SCORE CALIBRATION</b>",
        "<i>Target: below 0.10 (lower is better)</i>",
        "",
    ]

    for cluster_name, bs_data in cluster_brier.items():
        bs = bs_data.get("brier_score")
        n_res = bs_data.get("n_resolved", 0)
        n_total = bs_data.get("n_predictions", 0)
        on_target = bs_data.get("on_target")

        if bs is not None:
            status = "ON TARGET" if on_target else "ABOVE TARGET"
            lines.append(
                f"  <b>{cluster_name}:</b> {bs:.4f} ({status}) "
                f"— {n_res}/{n_total} resolved"
            )
        else:
            lines.append(f"  <b>{cluster_name}:</b> No resolved predictions yet")

    return "\n".join(lines)


def format_error_alert(component: str, message: str) -> str:
    """Format an error for Telegram notification."""
    return f"<b>ERROR — {component.upper()}</b>\n\n{message}"


def format_startup_message(clusters: list[str], paper: bool) -> str:
    """Startup notification."""
    mode = "PAPER" if paper else "LIVE"
    return (
        f"<b>QuantAgent Started</b>\n\n"
        f"Mode: {mode}\n"
        f"Copula: Student-t (nu=4, Gaussian excluded)\n"
        f"Clusters: {', '.join(clusters)}\n"
        f"Scan interval: {config.scan_interval_minutes}m\n"
        f"Threshold: {config.mispricing_threshold}\n"
        f"Tail IS threshold: <5% probability\n"
        f"Variance reduction: antithetic + stratified (target 100-500x)"
    )
