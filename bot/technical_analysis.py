import logging
from dataclasses import dataclass, field
from typing import List, Tuple

import numpy as np
import pandas as pd
import ta as ta_lib
from ta.momentum import RSIIndicator
from ta.trend import MACD as MACDIndicator
from ta.volatility import BollingerBands, AverageTrueRange

logger = logging.getLogger(__name__)


@dataclass
class TechnicalSignals:
    # Trend
    above_50d_ma: bool
    above_200d_ma: bool
    ma_50: float
    ma_200: float
    golden_cross: bool
    death_cross: bool
    price_vs_50d_pct: float

    # Momentum
    rsi_1d: float
    rsi_1h: float
    macd_signal: str        # "bullish_cross", "bearish_cross", "bullish", "bearish", "neutral"
    macd_histogram: float
    macd_divergence: str    # "bullish_div", "bearish_div", "none"

    # Structure
    support_levels: List[float]
    resistance_levels: List[float]
    nearest_support: float
    nearest_resistance: float
    distance_to_support_pct: float
    distance_to_resistance_pct: float

    # Patterns
    breakout_signal: str    # "bullish_breakout", "bearish_breakdown", "none"
    rejection_signal: str   # "resistance_rejection", "support_bounce", "none"

    # Volatility
    atr_14d: float
    bb_position: float
    bb_squeeze: bool

    # Composite
    technical_score: int
    technical_bias: str


def compute_technical_signals(
    spy_1h: pd.DataFrame,
    spy_1d: pd.DataFrame,
    spy_1wk: pd.DataFrame,
    current_price: float,
) -> TechnicalSignals:
    if spy_1d.empty or len(spy_1d) < 20:
        return _empty_signals(current_price)

    # --- Moving averages ---
    close_1d = spy_1d["Close"].dropna()
    ma_50 = float(close_1d.rolling(50).mean().iloc[-1]) if len(close_1d) >= 50 else current_price
    ma_200 = float(close_1d.rolling(200).mean().iloc[-1]) if len(close_1d) >= 200 else current_price
    above_50 = current_price > ma_50
    above_200 = current_price > ma_200
    price_vs_50_pct = (current_price - ma_50) / ma_50 * 100

    # Golden/death cross: did 50d cross 200d in the last 10 bars?
    golden_cross = False
    death_cross = False
    if len(close_1d) >= 210:
        ma50_series = close_1d.rolling(50).mean()
        ma200_series = close_1d.rolling(200).mean()
        diff = ma50_series - ma200_series
        recent = diff.iloc[-11:]
        if (recent.iloc[-1] > 0) and any(recent.iloc[:-1] <= 0):
            golden_cross = True
        elif (recent.iloc[-1] < 0) and any(recent.iloc[:-1] >= 0):
            death_cross = True

    # --- RSI ---
    rsi_1d = _safe_rsi(close_1d, 14)
    rsi_1h = _safe_rsi(spy_1h["Close"].dropna(), 14) if not spy_1h.empty else 50.0

    # --- MACD (daily) ---
    macd_signal_str, macd_hist, macd_div = _compute_macd(close_1d)

    # --- Support / Resistance ---
    supports, resistances = _compute_support_resistance(spy_1d, current_price)
    nearest_support = supports[0] if supports else current_price * 0.97
    nearest_resistance = resistances[0] if resistances else current_price * 1.03
    dist_support_pct = (current_price - nearest_support) / current_price * 100
    dist_resist_pct = (nearest_resistance - current_price) / current_price * 100

    # --- Breakout / rejection ---
    breakout, rejection = _detect_patterns(
        spy_1d, current_price, nearest_resistance, nearest_support
    )

    # --- Bollinger Bands ---
    bb_pos, bb_squeeze = _compute_bb(close_1d)

    # --- ATR ---
    atr_14d = _compute_atr(spy_1d, 14)

    # --- Score ---
    score, bias = _score_technicals(
        above_50, above_200, golden_cross, death_cross, price_vs_50_pct,
        rsi_1d, macd_signal_str, macd_div,
        dist_support_pct, dist_resist_pct, breakout, rejection, bb_squeeze,
    )

    return TechnicalSignals(
        above_50d_ma=above_50,
        above_200d_ma=above_200,
        ma_50=round(ma_50, 2),
        ma_200=round(ma_200, 2),
        golden_cross=golden_cross,
        death_cross=death_cross,
        price_vs_50d_pct=round(price_vs_50_pct, 2),
        rsi_1d=round(rsi_1d, 1),
        rsi_1h=round(rsi_1h, 1),
        macd_signal=macd_signal_str,
        macd_histogram=round(macd_hist, 4),
        macd_divergence=macd_div,
        support_levels=supports,
        resistance_levels=resistances,
        nearest_support=round(nearest_support, 2),
        nearest_resistance=round(nearest_resistance, 2),
        distance_to_support_pct=round(dist_support_pct, 2),
        distance_to_resistance_pct=round(dist_resist_pct, 2),
        breakout_signal=breakout,
        rejection_signal=rejection,
        atr_14d=round(atr_14d, 2),
        bb_position=round(bb_pos, 3),
        bb_squeeze=bb_squeeze,
        technical_score=score,
        technical_bias=bias,
    )


def _safe_rsi(series: pd.Series, period: int) -> float:
    if len(series) < period + 1:
        return 50.0
    try:
        rsi = RSIIndicator(close=series, window=period).rsi()
        val = rsi.dropna().iloc[-1]
        return float(val) if not np.isnan(val) else 50.0
    except Exception:
        return 50.0


def _compute_macd(close: pd.Series) -> Tuple[str, float, str]:
    if len(close) < 35:
        return "neutral", 0.0, "none"
    try:
        macd_obj = MACDIndicator(close=close, window_slow=26, window_fast=12, window_sign=9)
        hist = macd_obj.macd_diff().dropna()
        if hist.empty:
            return "neutral", 0.0, "none"

        hist_val = float(hist.iloc[-1])
        prev_hist = float(hist.iloc[-2]) if len(hist) >= 2 else hist_val

        if prev_hist < 0 and hist_val > 0:
            signal_str = "bullish_cross"
        elif prev_hist > 0 and hist_val < 0:
            signal_str = "bearish_cross"
        elif hist_val > 0:
            signal_str = "bullish"
        elif hist_val < 0:
            signal_str = "bearish"
        else:
            signal_str = "neutral"

        # Simple divergence detection
        divergence = "none"
        if len(hist) >= 20 and len(close) >= 20:
            price_high = close.iloc[-20:].max()
            macd_high = hist.iloc[-20:].max()
            price_prev_high = close.iloc[-40:-20].max() if len(close) >= 40 else price_high
            macd_prev_high = hist.iloc[-40:-20].max() if len(hist) >= 40 else macd_high
            if price_high > price_prev_high and macd_high < macd_prev_high:
                divergence = "bearish_div"
            price_low = close.iloc[-20:].min()
            macd_low = hist.iloc[-20:].min()
            price_prev_low = close.iloc[-40:-20].min() if len(close) >= 40 else price_low
            macd_prev_low = hist.iloc[-40:-20].min() if len(hist) >= 40 else macd_low
            if price_low < price_prev_low and macd_low > macd_prev_low:
                divergence = "bullish_div"

        return signal_str, hist_val, divergence
    except Exception as e:
        logger.debug(f"MACD error: {e}")
        return "neutral", 0.0, "none"


def _compute_support_resistance(
    df: pd.DataFrame,
    current_price: float,
    lookback: int = 60,
    n_levels: int = 3,
    cluster_pct: float = 0.005,
) -> Tuple[List[float], List[float]]:
    df = df.tail(lookback)
    highs = df["High"].values
    lows = df["Low"].values

    pivot_highs = []
    pivot_lows = []
    for i in range(2, len(df) - 2):
        if highs[i] > highs[i-2] and highs[i] > highs[i-1] and highs[i] > highs[i+1] and highs[i] > highs[i+2]:
            pivot_highs.append(highs[i])
        if lows[i] < lows[i-2] and lows[i] < lows[i-1] and lows[i] < lows[i+1] and lows[i] < lows[i+2]:
            pivot_lows.append(lows[i])

    def cluster(levels):
        if not levels:
            return []
        levels = sorted(levels)
        clusters = [[levels[0]]]
        for lvl in levels[1:]:
            if abs(lvl - clusters[-1][-1]) / clusters[-1][-1] <= cluster_pct:
                clusters[-1].append(lvl)
            else:
                clusters.append([lvl])
        return [round(np.mean(c), 2) for c in clusters]

    resistances = sorted(
        [l for l in cluster(pivot_highs) if l > current_price]
    )[:n_levels]
    supports = sorted(
        [l for l in cluster(pivot_lows) if l < current_price],
        reverse=True,
    )[:n_levels]

    # Fallback to MA-based levels if no pivots found
    close = df["Close"]
    if not supports:
        supports = [round(current_price * 0.97, 2)]
    if not resistances:
        resistances = [round(current_price * 1.03, 2)]

    return supports, resistances


def _detect_patterns(
    df: pd.DataFrame,
    current_price: float,
    nearest_resistance: float,
    nearest_support: float,
) -> Tuple[str, str]:
    if len(df) < 5:
        return "none", "none"

    recent = df.tail(5)
    avg_vol = df["Volume"].rolling(20).mean().iloc[-1] if len(df) >= 20 else df["Volume"].mean()

    breakout = "none"
    rejection = "none"

    # Bullish breakout: price closed above resistance with above-avg volume
    last_close = float(recent["Close"].iloc[-1])
    last_vol = float(recent["Volume"].iloc[-1])
    prev_close = float(recent["Close"].iloc[-2]) if len(recent) >= 2 else last_close

    if prev_close < nearest_resistance and last_close > nearest_resistance * 1.01:
        if last_vol > avg_vol * 1.1:
            breakout = "bullish_breakout"
    elif prev_close > nearest_support and last_close < nearest_support * 0.99:
        if last_vol > avg_vol * 1.1:
            breakout = "bearish_breakdown"

    # Rejection: price touched resistance/support and reversed
    recent_high = float(recent["High"].max())
    recent_low = float(recent["Low"].min())
    if abs(recent_high - nearest_resistance) / nearest_resistance < 0.005 and last_close < nearest_resistance * 0.995:
        rejection = "resistance_rejection"
    elif abs(recent_low - nearest_support) / nearest_support < 0.005 and last_close > nearest_support * 1.005:
        rejection = "support_bounce"

    return breakout, rejection


def _compute_bb(close: pd.Series, period: int = 20) -> Tuple[float, bool]:
    if len(close) < period + 1:
        return 0.5, False
    try:
        bb = BollingerBands(close=close, window=period, window_dev=2)
        upper = bb.bollinger_hband().dropna()
        lower = bb.bollinger_lband().dropna()
        mid = bb.bollinger_mavg().dropna()

        if upper.empty or lower.empty or mid.empty:
            return 0.5, False

        u, l, m = float(upper.iloc[-1]), float(lower.iloc[-1]), float(mid.iloc[-1])
        if u == l:
            return 0.5, False
        pos = (float(close.iloc[-1]) - l) / (u - l)

        # Bandwidth squeeze: reindex to align
        idx = upper.index.intersection(lower.index).intersection(mid.index)
        bandwidth = (upper[idx] - lower[idx]) / mid[idx].replace(0, np.nan)
        squeeze = False
        clean = bandwidth.dropna()
        if len(clean) >= 126:
            squeeze = float(clean.iloc[-1]) < float(clean.quantile(0.20))

        return float(np.clip(pos, 0, 1)), squeeze
    except Exception:
        return 0.5, False


def _compute_atr(df: pd.DataFrame, period: int = 14) -> float:
    if len(df) < period + 1:
        return 0.0
    try:
        atr = AverageTrueRange(high=df["High"], low=df["Low"], close=df["Close"], window=period)
        vals = atr.average_true_range().dropna()
        return float(vals.iloc[-1]) if not vals.empty else 0.0
    except Exception:
        return 0.0


def _score_technicals(
    above_50: bool,
    above_200: bool,
    golden_cross: bool,
    death_cross: bool,
    price_vs_50_pct: float,
    rsi_1d: float,
    macd_signal: str,
    macd_div: str,
    dist_support_pct: float,
    dist_resist_pct: float,
    breakout: str,
    rejection: str,
    bb_squeeze: bool,
) -> Tuple[int, str]:
    score = 0
    bullish_pts = 0
    bearish_pts = 0

    # Trend — max 25
    if above_50:
        score += 8; bullish_pts += 8
    else:
        bearish_pts += 8
    if above_200:
        score += 10; bullish_pts += 10
    else:
        bearish_pts += 10
    if golden_cross:
        score += 5; bullish_pts += 5
    if death_cross:
        score -= 5; bearish_pts += 5
    if -2 < price_vs_50_pct < 5:  # healthy, not overextended
        score += 3; bullish_pts += 1

    # RSI — max 15
    if 40 <= rsi_1d <= 65:
        score += 8; bullish_pts += 4
    elif rsi_1d < 30:
        score += 6; bearish_pts += 2  # oversold = potential reversal up
    elif rsi_1d > 70:
        score += 4; bearish_pts += 4  # overbought = caution

    # MACD — max 20
    if macd_signal == "bullish_cross":
        score += 15; bullish_pts += 15
    elif macd_signal == "bearish_cross":
        score += 10; bearish_pts += 15
    elif macd_signal == "bullish":
        score += 8; bullish_pts += 8
    elif macd_signal == "bearish":
        score += 5; bearish_pts += 8

    if macd_div == "bullish_div":
        score += 5; bullish_pts += 5
    elif macd_div == "bearish_div":
        score += 3; bearish_pts += 5

    # S/R positioning — max 10
    if dist_support_pct < 1.5:  # near support = potential bounce
        score += 5; bullish_pts += 3
    if dist_resist_pct < 1.5:  # near resistance = potential rejection
        score += 3; bearish_pts += 3

    # Breakout / patterns — max 15
    if breakout == "bullish_breakout":
        score += 12; bullish_pts += 12
    elif breakout == "bearish_breakdown":
        score += 8; bearish_pts += 12
    if rejection == "support_bounce":
        score += 5; bullish_pts += 5
    elif rejection == "resistance_rejection":
        score += 4; bearish_pts += 5

    # BB squeeze bonus (pre-breakout state)
    if bb_squeeze:
        score += 3

    # Deductions
    if price_vs_50_pct > 8:  # very extended
        score -= 4

    score = max(0, min(50, score))

    if bullish_pts > bearish_pts * 1.3:
        bias = "bullish"
    elif bearish_pts > bullish_pts * 1.3:
        bias = "bearish"
    else:
        bias = "neutral"

    return score, bias


def _empty_signals(price: float) -> TechnicalSignals:
    return TechnicalSignals(
        above_50d_ma=False, above_200d_ma=False,
        ma_50=price, ma_200=price,
        golden_cross=False, death_cross=False, price_vs_50d_pct=0.0,
        rsi_1d=50.0, rsi_1h=50.0,
        macd_signal="neutral", macd_histogram=0.0, macd_divergence="none",
        support_levels=[price * 0.97], resistance_levels=[price * 1.03],
        nearest_support=price * 0.97, nearest_resistance=price * 1.03,
        distance_to_support_pct=3.0, distance_to_resistance_pct=3.0,
        breakout_signal="none", rejection_signal="none",
        atr_14d=0.0, bb_position=0.5, bb_squeeze=False,
        technical_score=0, technical_bias="neutral",
    )
