"""Compare the bot's results with simply holding the underlying over the same dates."""
import logging
from datetime import date, timedelta
from typing import Optional

import yfinance as yf

logger = logging.getLogger(__name__)


def hold_return_pct(ticker: str, start: date, end: Optional[date] = None) -> Optional[float]:
    """% change from the last close before `start` to the latest close (None if data unavailable)."""
    try:
        end = end or date.today()
        hist = yf.Ticker(ticker).history(start=start - timedelta(days=10), end=end + timedelta(days=1))["Close"].dropna()
        if hist.empty:
            return None
        before = hist[[d.date() < start for d in hist.index]]
        base = float(before.iloc[-1]) if len(before) else float(hist.iloc[0])
        return (float(hist.iloc[-1]) / base - 1) * 100
    except Exception as e:
        logger.warning(f"Benchmark lookup failed: {e}")
        return None


def benchmark_line(ticker: str, start: date, bot_pnl: float, account_size: float) -> str:
    hold = hold_return_pct(ticker, start)
    bot_pct = bot_pnl / account_size * 100 if account_size else 0.0
    if hold is None:
        return f"📏 **vs holding {ticker}:** n/a (price data unavailable)"
    diff = bot_pct - hold
    verdict = "ahead" if diff > 0 else "behind" if diff < 0 else "even"
    return (f"📏 **vs holding {ticker}:** bot {bot_pct:+.1f}% of account  |  {ticker} {hold:+.1f}%  "
            f"→ {abs(diff):.1f} pts {verdict}")
