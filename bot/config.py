import os
from dataclasses import dataclass, field
from typing import List
from dotenv import load_dotenv

load_dotenv()


@dataclass(frozen=True)
class BotConfig:
    # Credentials
    discord_webhook_url: str
    twilio_sid: str
    twilio_token: str
    twilio_from: str
    twilio_to: str

    # Account / risk
    account_size: float
    risk_pct: float

    # Signal thresholds
    confidence_threshold: int
    min_iv_rank: int

    # Scheduling
    scan_interval_minutes: int

    # Market hours (ET)
    market_open_hour: int = 9
    market_open_minute: int = 30
    market_close_hour: int = 16
    market_close_minute: int = 0
    timezone: str = "America/New_York"

    # Tickers
    spy: str = "QQQ"
    vix: str = "^VIX"
    tlt: str = "TLT"
    uup: str = "UUP"
    sector_etfs: List[str] = field(default_factory=lambda: [
        "XLK", "XLV", "XLF", "XLE", "XLI",
        "XLY", "XLP", "XLB", "XLU", "XLRE", "XLC",
    ])

    # Options preferences
    target_dte_min: int = 7
    target_dte_max: int = 45
    prefer_weekly_dte: int = 14
    prefer_monthly_dte: int = 35
    max_spread_width: float = 5.0
    min_open_interest: int = 200
    min_volume: int = 50

    # Alert cooldown
    cooldown_hours: int = 2
    cooldown_confidence_jump: int = 10


def load_config() -> BotConfig:
    missing = [k for k in ["DISCORD_WEBHOOK_URL"] if not os.getenv(k)]
    if missing:
        raise EnvironmentError(f"Missing required env vars: {missing}")
    return BotConfig(
        discord_webhook_url=os.environ["DISCORD_WEBHOOK_URL"],
        twilio_sid=os.getenv("TWILIO_ACCOUNT_SID", ""),
        twilio_token=os.getenv("TWILIO_AUTH_TOKEN", ""),
        twilio_from=os.getenv("TWILIO_FROM_WHATSAPP", ""),
        twilio_to=os.getenv("TWILIO_TO_WHATSAPP", ""),
        account_size=float(os.getenv("ACCOUNT_SIZE", 25000)),
        risk_pct=float(os.getenv("RISK_PCT", 0.015)),
        confidence_threshold=int(os.getenv("CONFIDENCE_THRESHOLD", 75)),
        min_iv_rank=int(os.getenv("MIN_IV_RANK", 20)),
        scan_interval_minutes=int(os.getenv("SCAN_INTERVAL_MINUTES", 20)),
    )


CONFIG = load_config()
