#!/usr/bin/env python3
"""
QQQ Options Trading Bot
Scans SPY options every 20 min during market hours and sends Discord alerts.

Usage:
  python main.py          — start the bot
  python main.py stats    — print win-rate summary and exit
"""
import logging
import os
import sys

from bot.config import CONFIG
from bot.trade_logger import get_win_rate_summary, init_db


def setup_logging() -> None:
    level = getattr(logging, os.getenv("LOG_LEVEL", "INFO").upper(), logging.INFO)
    fmt = "%(asctime)s [%(levelname)-8s] %(name)s — %(message)s"
    handlers = [
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("qqq_bot.log", encoding="utf-8"),
    ]
    logging.basicConfig(level=level, format=fmt, handlers=handlers)
    # Quiet noisy third-party loggers
    logging.getLogger("yfinance").setLevel(logging.WARNING)
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    logging.getLogger("apscheduler").setLevel(logging.WARNING)


def print_banner() -> None:
    init_db()
    summary = get_win_rate_summary()
    print("=" * 60)
    print("  QQQ Options Trading Bot")
    print(f"  Account: ${CONFIG.account_size:>10,.0f}  |  Risk/trade: {CONFIG.risk_pct*100:.1f}%")
    print(f"  Confidence threshold: {CONFIG.confidence_threshold}%")
    print(f"  Scan interval: every {CONFIG.scan_interval_minutes} minutes")
    total = summary.get("total_trades", 0)
    if total and total > 0:
        print(
            f"  Historical:  {summary.get('win_rate_pct', 0)}% win rate  "
            f"({summary.get('wins', 0)}W / {summary.get('losses', 0)}L)  "
            f"P&L: ${summary.get('total_pnl', 0):+,.2f}"
        )
    else:
        print("  Historical:  No closed trades yet")
    print("=" * 60)
    print()


if __name__ == "__main__":
    setup_logging()

    if len(sys.argv) > 1 and sys.argv[1] == "stats":
        init_db()
        s = get_win_rate_summary()
        print("\nQQQ Bot — Trade Statistics")
        print("-" * 40)
        for k, v in s.items():
            print(f"  {k:<20}: {v}")
        sys.exit(0)

    print_banner()

    from bot.scheduler import start_scheduler
    try:
        start_scheduler()
    except (KeyboardInterrupt, SystemExit):
        print("\nBot stopped.")
