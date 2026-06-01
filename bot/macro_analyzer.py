import logging
from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class MacroSignals:
    # VIX
    vix_current: float
    vix_regime: str        # "calm", "normal", "elevated", "panic"
    vix_trend: str         # "rising", "falling", "flat"
    vix_spike: bool

    # Breadth
    sectors_above_50d: int
    sectors_total: int
    breadth_pct: float
    breadth_signal: str    # "strong", "neutral", "weak"

    # Bond / dollar
    tlt_trend: str         # "rising", "falling", "flat"
    tlt_vs_50d_pct: float
    uup_trend: str
    uup_vs_50d_pct: float

    # Composite
    macro_score: int
    macro_bias: str        # "bullish", "bearish", "neutral"


def analyze_macro(
    vix_current: float,
    vix_1d: pd.DataFrame,
    sector_etf_data: Dict[str, pd.DataFrame],
    tlt_1d: pd.DataFrame,
    uup_1d: pd.DataFrame,
) -> MacroSignals:
    vix_regime = _vix_regime(vix_current)
    vix_trend, vix_spike = _vix_trend(vix_1d, vix_current)
    sectors_above, sectors_total, breadth_pct, breadth_signal = _market_breadth(sector_etf_data)
    tlt_trend, tlt_vs_50d = _etf_trend(tlt_1d)
    uup_trend, uup_vs_50d = _etf_trend(uup_1d)
    score, bias = _score_macro(
        vix_current, vix_regime, vix_trend, vix_spike,
        breadth_pct, breadth_signal,
        tlt_trend, uup_trend,
    )

    return MacroSignals(
        vix_current=round(vix_current, 2),
        vix_regime=vix_regime,
        vix_trend=vix_trend,
        vix_spike=vix_spike,
        sectors_above_50d=sectors_above,
        sectors_total=sectors_total,
        breadth_pct=round(breadth_pct, 1),
        breadth_signal=breadth_signal,
        tlt_trend=tlt_trend,
        tlt_vs_50d_pct=round(tlt_vs_50d, 2),
        uup_trend=uup_trend,
        uup_vs_50d_pct=round(uup_vs_50d, 2),
        macro_score=score,
        macro_bias=bias,
    )


def _vix_regime(vix: float) -> str:
    if vix < 15:
        return "calm"
    if vix < 25:
        return "normal"
    if vix < 35:
        return "elevated"
    return "panic"


def _vix_trend(vix_1d: pd.DataFrame, current: float) -> Tuple[str, bool]:
    if vix_1d.empty or len(vix_1d) < 6:
        return "flat", False
    closes = vix_1d["Close"].dropna()
    ma5 = float(closes.rolling(5).mean().iloc[-1])
    trend = "flat"
    if current > ma5 * 1.02:
        trend = "rising"
    elif current < ma5 * 0.98:
        trend = "falling"

    # Spike: >20% move in 3 bars
    spike = False
    if len(closes) >= 4:
        three_ago = float(closes.iloc[-4])
        if three_ago > 0 and (current - three_ago) / three_ago > 0.20:
            spike = True

    return trend, spike


def _market_breadth(sector_data: Dict[str, pd.DataFrame]) -> Tuple[int, int, float, str]:
    above = 0
    total = 0
    for ticker, df in sector_data.items():
        if df.empty or len(df) < 50:
            continue
        total += 1
        sma50 = df["Close"].rolling(50).mean().iloc[-1]
        if df["Close"].iloc[-1] > sma50:
            above += 1
    if total == 0:
        return 0, 0, 50.0, "neutral"
    pct = above / total * 100
    if pct > 70:
        signal = "strong"
    elif pct < 40:
        signal = "weak"
    else:
        signal = "neutral"
    return above, total, pct, signal


def _etf_trend(df: pd.DataFrame) -> Tuple[str, float]:
    if df.empty or len(df) < 20:
        return "flat", 0.0
    close = df["Close"].dropna()
    sma20 = float(close.rolling(20).mean().iloc[-1])
    current = float(close.iloc[-1])
    if sma20 == 0:
        return "flat", 0.0
    pct = (current - sma20) / sma20 * 100
    if pct > 1.0:
        trend = "rising"
    elif pct < -1.0:
        trend = "falling"
    else:
        trend = "flat"
    return trend, pct


def _score_macro(
    vix: float,
    vix_regime: str,
    vix_trend: str,
    vix_spike: bool,
    breadth_pct: float,
    breadth_signal: str,
    tlt_trend: str,
    uup_trend: str,
) -> Tuple[int, str]:
    score = 0
    bullish_pts = 0
    bearish_pts = 0

    # VIX regime (max 12)
    if vix_regime == "calm":
        score += 12; bullish_pts += 12
    elif vix_regime == "normal":
        score += 8; bullish_pts += 6
    elif vix_regime == "elevated":
        score += 5; bearish_pts += 5
    else:  # panic
        score -= 10; bearish_pts += 15

    # VIX trend (max 5)
    if vix_trend == "falling":
        score += 5; bullish_pts += 5
    elif vix_trend == "rising":
        score += 2; bearish_pts += 5

    # VIX spike penalty
    if vix_spike:
        score -= 8; bearish_pts += 8

    # Breadth (max 8)
    if breadth_signal == "strong":
        score += 8; bullish_pts += 8
    elif breadth_signal == "neutral":
        score += 4; bullish_pts += 2
    else:
        score -= 5; bearish_pts += 8

    # Bond/dollar (max 5)
    # TLT falling + UUP neutral/falling = risk-on = bullish for equities
    if tlt_trend == "falling" and uup_trend != "rising":
        score += 5; bullish_pts += 5
    # TLT rising = risk-off = bearish
    elif tlt_trend == "rising":
        score += 2; bearish_pts += 4
    # UUP rising aggressively = headwind for equities
    if uup_trend == "rising":
        score -= 2; bearish_pts += 2

    score = max(0, min(30, score))

    if bullish_pts > bearish_pts * 1.3:
        bias = "bullish"
    elif bearish_pts > bullish_pts * 1.3:
        bias = "bearish"
    else:
        bias = "neutral"

    return score, bias
