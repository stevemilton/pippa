"""Configuration loaded from .env file."""

import os
from pathlib import Path
from dataclasses import dataclass

from dotenv import load_dotenv


def _load_env():
    """Load .env from project root."""
    env_path = Path(__file__).resolve().parent.parent / ".env"
    load_dotenv(env_path)


_load_env()


@dataclass(frozen=True)
class Config:
    # IBKR
    ibkr_host: str = os.getenv("IBKR_HOST", "127.0.0.1")
    ibkr_port: int = int(os.getenv("IBKR_PORT", "4002"))
    ibkr_client_id: int = int(os.getenv("IBKR_CLIENT_ID", "1"))

    # Telegram
    telegram_bot_token: str = os.getenv("TELEGRAM_BOT_TOKEN", "")
    telegram_chat_id: str = os.getenv("TELEGRAM_CHAT_ID", "")

    # Anthropic
    anthropic_api_key: str = os.getenv("ANTHROPIC_API_KEY", "")

    # Scanner
    scan_interval_minutes: int = int(os.getenv("SCAN_INTERVAL_MINUTES", "5"))
    mispricing_threshold: float = float(os.getenv("MISPRICING_THRESHOLD", "0.03"))
    max_instruments_per_cluster: int = int(os.getenv("MAX_INSTRUMENTS_PER_CLUSTER", "6"))
    paper_trading: bool = os.getenv("PAPER_TRADING", "true").lower() == "true"

    # Database
    db_path: str = os.getenv("DB_PATH", str(Path(__file__).resolve().parent.parent / "quantagent.db"))

    # Copula
    copula_lookback_days: int = int(os.getenv("COPULA_LOOKBACK_DAYS", "60"))
    copula_min_observations: int = int(os.getenv("COPULA_MIN_OBSERVATIONS", "100"))

    def validate(self):
        """Raise if critical config is missing."""
        missing = []
        if not self.telegram_bot_token:
            missing.append("TELEGRAM_BOT_TOKEN")
        if not self.telegram_chat_id:
            missing.append("TELEGRAM_CHAT_ID")
        if not self.anthropic_api_key:
            missing.append("ANTHROPIC_API_KEY")
        if missing:
            raise EnvironmentError(f"Missing required env vars: {', '.join(missing)}")


config = Config()
