"""
Trend-gated 60-day bull call spread (the one setup that held up in backtests/strategy_lab.py).

Rules (all fixed in advance, see backtest/strategy_lab.py):
  * ENTER only when flat AND the underlying closes/trades above its 200-day average. No bearish trades.
  * Buy the first strike at/above spot, sell the strike ~4% higher, expiry nearest 60 days out.
  * EXIT at +100% of the debit, -50% of the debit, or when 3 days remain to expiration.
This is long-market exposure with defined risk, NOT a predictive signal. Expect ~half the trades to lose
and individual losses of up to ~70% of the premium.
"""
import logging
from dataclasses import dataclass
from datetime import datetime
from typing import Optional

import pandas as pd
import pytz
import yfinance as yf

logger = logging.getLogger(__name__)
ET = pytz.timezone("America/New_York")

TP_PCT = 100.0
SL_PCT = -50.0


@dataclass
class CallSpread:
    expiration: str
    expiration_label: str
    dte: int
    strike: float            # long (lower) strike — named `strike` so trade_logger works unchanged
    short_strike: float
    long_ask: float
    long_bid: float
    short_ask: float
    short_bid: float
    ask_price: float         # net debit per share (buy long at ask, sell short at bid)
    bid_price: float         # what the spread would fetch if closed right now
    width: float
    max_profit_per_share: float
    breakeven: float
    round_trip_cost: float   # per share, both legs, in+out
    contracts: int
    cost_per_contract: float
    total_cost: float
    account_size: float
    risk_pct_used: float
    spy_price: float
    direction: str = "call"
    option_type: str = "Call Spread"
    dte_target: int = 60


def construct_call_spread(
    calls: pd.DataFrame, spot: float, expirations: list,
    account_size: float, risk_pct: float,
    dte_min: int = 50, dte_max: int = 75, dte_target: int = 60, width_pct: float = 0.04,
) -> Optional[CallSpread]:
    if calls is None or calls.empty or not {"strike", "bid", "ask", "expiration"}.issubset(calls.columns):
        logger.warning("Spread: no call chain available")
        return None
    today = datetime.now(ET).date()
    cands = []
    for e in expirations:
        try:
            dte = (datetime.strptime(e, "%Y-%m-%d").date() - today).days
        except ValueError:
            continue
        if dte_min <= dte <= dte_max:
            cands.append((abs(dte - dte_target), e, dte))
    if not cands:
        logger.warning(f"Spread: no expiration between {dte_min} and {dte_max} days")
        return None
    _, exp, dte = sorted(cands)[0]
    chain = calls[calls["expiration"] == exp].sort_values("strike")
    chain = chain[(chain["ask"] > 0) & (chain["bid"] > 0)]
    longs = chain[chain["strike"] >= spot]
    if longs.empty:
        return None
    lrow = longs.iloc[0]
    target_short = float(lrow["strike"]) + spot * width_pct
    shorts = chain[chain["strike"] > lrow["strike"]]
    if shorts.empty:
        return None
    srow = shorts.iloc[(shorts["strike"] - target_short).abs().argsort()[:1]].iloc[0]

    la, lb, sa, sb = float(lrow["ask"]), float(lrow["bid"]), float(srow["ask"]), float(srow["bid"])
    debit = round(la - sb, 2)
    width = float(srow["strike"] - lrow["strike"])
    if debit <= 0.05 or debit >= width:
        logger.warning(f"Spread: unusable pricing (debit {debit:.2f}, width {width:.2f})")
        return None
    contracts = max(1, int(account_size * risk_pct // (debit * 100)))
    total = round(contracts * debit * 100, 2)
    return CallSpread(
        expiration=exp, expiration_label=datetime.strptime(exp, "%Y-%m-%d").strftime("%b %-d, %Y"), dte=dte,
        strike=float(lrow["strike"]), short_strike=float(srow["strike"]),
        long_ask=la, long_bid=lb, short_ask=sa, short_bid=sb,
        ask_price=debit, bid_price=round(max(lb - sa, 0), 2), width=width,
        max_profit_per_share=round(width - debit, 2), breakeven=round(float(lrow["strike"]) + debit, 2),
        round_trip_cost=round((la - lb) + (sa - sb), 2), contracts=contracts,
        cost_per_contract=round(debit * 100, 2), total_cost=total, account_size=account_size,
        risk_pct_used=round(total / account_size * 100, 1), spy_price=spot,
    )


def spread_value(ticker: str, expiration: str, long_strike: float, short_strike: float) -> Optional[float]:
    """Per-share value you'd get closing now: sell the long at the bid, buy back the short at the ask."""
    try:
        calls = yf.Ticker(ticker).option_chain(expiration).calls
        l = calls[calls["strike"] == long_strike]
        s = calls[calls["strike"] == short_strike]
        if l.empty or s.empty:
            return None
        l, s = l.iloc[0], s.iloc[0]

        def px(row, side):
            v = float(row.get(side, 0) or 0)
            if v > 0:
                return v
            bid, ask = float(row.get("bid", 0) or 0), float(row.get("ask", 0) or 0)
            if bid > 0 and ask > 0:
                return (bid + ask) / 2
            return float(row.get("lastPrice", 0) or 0)

        value = px(l, "bid") - px(s, "ask")
        width = short_strike - long_strike
        return round(min(max(value, 0.0), width), 2)
    except Exception as e:
        logger.warning(f"Spread value lookup failed: {e}")
        return None
