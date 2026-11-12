"""
Structured logger for the QQQ options trading bot.
Mirrors the SPY bot's logger — every signal evaluation, score breakdown,
alert, and error is written to both console and a dated log file.
"""

import logging
import sys
from datetime import datetime
from pathlib import Path


import os
LOG_DIR = Path(os.getenv("DATA_DIR") or Path(__file__).parent.parent) / "logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)

_fmt = "%(asctime)s | %(levelname)-8s | %(name)-22s | %(message)s"
_date_fmt = "%Y-%m-%d %H:%M:%S"


def _build_logger(name: str) -> logging.Logger:
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger

    logger.setLevel(logging.DEBUG)
    logger.propagate = False

    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.INFO)
    ch.setFormatter(logging.Formatter(_fmt, _date_fmt))
    logger.addHandler(ch)

    today = datetime.now().strftime("%Y-%m-%d")
    fh = logging.FileHandler(LOG_DIR / f"bot_{today}.log", encoding="utf-8")
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(logging.Formatter(_fmt, _date_fmt))
    logger.addHandler(fh)

    return logger


def _configure_root() -> None:
    """Send every stdlib logger (bot.scheduler, bot.alert_sender, ...) to console + dated file.
    Without this the scheduler's logger.info/exception calls were silently dropped."""
    root = logging.getLogger()
    if getattr(root, "_bot_configured", False):
        return
    root.setLevel(logging.INFO)
    ch = logging.StreamHandler(sys.stdout)
    ch.setFormatter(logging.Formatter(_fmt, _date_fmt))
    root.addHandler(ch)
    today = datetime.now().strftime("%Y-%m-%d")
    fh = logging.FileHandler(LOG_DIR / f"bot_{today}.log", encoding="utf-8")
    fh.setFormatter(logging.Formatter(_fmt, _date_fmt))
    root.addHandler(fh)
    root._bot_configured = True


_configure_root()

# ── module-level loggers ──────────────────────────────────────────────────────
main      = _build_logger("main")
signals   = _build_logger("signals")
technical = _build_logger("technical")
macro     = _build_logger("macro")
options   = _build_logger("options")
scheduler = _build_logger("scheduler")
alerts    = _build_logger("alerts")
trades    = _build_logger("trades")
