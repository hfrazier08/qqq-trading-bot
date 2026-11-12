"""Record the real quotes behind every paper entry/exit so the backtest's cost assumptions can be checked."""
import csv
import logging
from datetime import datetime
from pathlib import Path
from typing import List, Optional

import pytz

from bot.config import DATA_DIR

logger = logging.getLogger(__name__)
ET = pytz.timezone("America/New_York")
FILLS_FILE = Path(DATA_DIR) / "fills.csv"
BACKTEST_COST_BPS_PER_LEG_SIDE = 1.0        # what backtest/strategy_lab.py assumed (1bp of spot, per leg, per side)
FIELDS = ["time_et", "event", "ticker", "expiration", "long_strike", "short_strike", "spot",
          "long_bid", "long_ask", "short_bid", "short_ask", "mid", "executed", "slippage_per_share",
          "assumed_per_side"]


def log_fill(event: str, trade: dict, quote: dict, executed: float, spot: float) -> Optional[dict]:
    """event = 'entry' (executed = debit paid) or 'exit' (executed = value received). Costs are per share."""
    try:
        mid = (quote["long_bid"] + quote["long_ask"]) / 2 - (quote["short_bid"] + quote["short_ask"]) / 2
        slip = (executed - mid) if event == "entry" else (mid - executed)
        row = {
            "time_et": datetime.now(ET).strftime("%Y-%m-%d %H:%M"), "event": event, "ticker": trade["ticker"],
            "expiration": trade["expiration"], "long_strike": trade["strike"], "short_strike": trade.get("short_strike"),
            "spot": round(spot, 2), **{k: quote[k] for k in ("long_bid", "long_ask", "short_bid", "short_ask")},
            "mid": round(mid, 3), "executed": round(executed, 3), "slippage_per_share": round(slip, 3),
            "assumed_per_side": round(2 * BACKTEST_COST_BPS_PER_LEG_SIDE / 1e4 * spot, 3),
        }
        new = not FILLS_FILE.exists()
        with open(FILLS_FILE, "a", newline="") as f:
            w = csv.DictWriter(f, fieldnames=FIELDS)
            if new:
                w.writeheader()
            w.writerow(row)
        return row
    except Exception:
        logger.exception("Could not log fill")
        return None


def read_fills() -> List[dict]:
    if not FILLS_FILE.exists():
        return []
    with open(FILLS_FILE) as f:
        return list(csv.DictReader(f))


def fills_summary() -> str:
    rows = read_fills()
    if len(rows) < 2:
        return ""
    obs = sum(float(r["slippage_per_share"]) for r in rows) / len(rows)
    ass = sum(float(r["assumed_per_side"]) for r in rows) / len(rows)
    verdict = "worse than" if obs > ass * 1.25 else "better than" if obs < ass * 0.75 else "in line with"
    return (f"🧾 **Fill check** ({len(rows)} fills): real cost ${obs:.2f}/share per side vs "
            f"${ass:.2f} the backtest assumed — {verdict} the model")
