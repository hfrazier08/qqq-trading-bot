import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional
from datetime import datetime

import pytz

from bot.technical_analysis import TechnicalSignals
from bot.options_analyzer import OptionsAnalysis
from bot.macro_analyzer import MacroSignals

logger = logging.getLogger(__name__)

ET = pytz.timezone("America/New_York")

# Valid signal strings
BUY_CALL = "BUY_CALL_SPREAD"
BUY_PUT = "BUY_PUT_SPREAD"
EXIT_LONG = "EXIT_LONG"
EXIT_SHORT = "EXIT_SHORT"
HOLD = "HOLD"


@dataclass
class TradingSignal:
    timestamp: str
    signal: str
    direction: str              # "bullish", "bearish", "neutral"
    confidence: int             # 0-100
    confidence_breakdown: Dict[str, int]
    signal_reasons: List[str]
    risk_level: str             # "low", "moderate", "high", "extreme"
    spy_price: float = 0.0


# Module-level cooldown state
_last_signal: Optional[str] = None
_last_signal_time: Optional[datetime] = None
_last_confidence: int = 0


def generate_signal(
    technical: TechnicalSignals,
    macro: MacroSignals,
    options: OptionsAnalysis,
    threshold: int = 75,
    current_position: Optional[str] = None,
    spy_price: float = 0.0,
) -> TradingSignal:
    global _last_signal, _last_signal_time, _last_confidence

    tech_score = technical.technical_score   # 0-50
    macro_score = macro.macro_score          # 0-30
    opts_score = options.options_score       # 0-20
    total = tech_score + macro_score + opts_score  # 0-100

    tech_bias = technical.technical_bias
    macro_bias = macro.macro_bias

    reasons = _build_reasons(technical, macro, options, total)
    risk_level = _risk_level(macro.vix_regime, macro.vix_spike)

    # Layers must agree for an actionable signal
    layers_agree = (tech_bias == macro_bias) and tech_bias in ("bullish", "bearish")

    # EXIT logic (overrides everything)
    if current_position == "long_call_spread":
        if total < 40 and tech_bias == "bearish":
            return _make_signal(EXIT_LONG, "bearish", total, tech_score, macro_score, opts_score, reasons, risk_level, spy_price)
    if current_position == "long_put_spread":
        if total < 40 and tech_bias == "bullish":
            return _make_signal(EXIT_SHORT, "bullish", total, tech_score, macro_score, opts_score, reasons, risk_level, spy_price)

    # Panic — don't open new positions
    if macro.vix_regime == "panic":
        reasons.insert(0, f"HOLD — VIX in panic regime ({macro.vix_current:.1f})")
        return _make_signal(HOLD, "neutral", total, tech_score, macro_score, opts_score, reasons, risk_level, spy_price)

    if total < threshold or not layers_agree:
        return _make_signal(HOLD, tech_bias if tech_bias != "neutral" else "neutral", total, tech_score, macro_score, opts_score, reasons, risk_level, spy_price)

    # Determine direction
    direction = tech_bias  # "bullish" or "bearish"
    signal = BUY_CALL if direction == "bullish" else BUY_PUT

    # Cooldown: suppress identical signal within cooldown window unless confidence jumped
    now = datetime.now(ET)
    if (
        _last_signal == signal
        and _last_signal_time is not None
        and (now - _last_signal_time).total_seconds() < 7200  # 2 hours
        and (total - _last_confidence) < 10
    ):
        logger.info(f"Cooldown active — suppressing {signal} (conf={total})")
        return _make_signal(HOLD, direction, total, tech_score, macro_score, opts_score, reasons, risk_level, spy_price)

    _last_signal = signal
    _last_signal_time = now
    _last_confidence = total

    return _make_signal(signal, direction, total, tech_score, macro_score, opts_score, reasons, risk_level, spy_price)


def _make_signal(
    signal: str, direction: str, total: int,
    tech: int, macro: int, opts: int,
    reasons: List[str], risk_level: str, price: float,
) -> TradingSignal:
    return TradingSignal(
        timestamp=datetime.now(ET).strftime("%Y-%m-%d %H:%M ET"),
        signal=signal,
        direction=direction,
        confidence=total,
        confidence_breakdown={"technical": tech, "macro": macro, "options": opts},
        signal_reasons=reasons[:8],
        risk_level=risk_level,
        spy_price=price,
    )


def _risk_level(vix_regime: str, vix_spike: bool) -> str:
    if vix_spike or vix_regime == "panic":
        return "extreme"
    if vix_regime == "elevated":
        return "high"
    if vix_regime == "normal":
        return "moderate"
    return "low"


def _build_reasons(
    t: TechnicalSignals,
    m: MacroSignals,
    o: OptionsAnalysis,
    total: int,
) -> List[str]:
    reasons = []

    # Technical
    trend_parts = []
    if t.above_200d_ma:
        trend_parts.append(f"above 200d MA (${t.ma_200:.2f})")
    else:
        trend_parts.append(f"below 200d MA (${t.ma_200:.2f})")
    if t.above_50d_ma:
        trend_parts.append(f"above 50d MA (${t.ma_50:.2f})")
    if trend_parts:
        reasons.append("SPY " + " and ".join(trend_parts))

    if t.golden_cross:
        reasons.append("Golden cross: 50d MA crossed above 200d MA")
    if t.death_cross:
        reasons.append("Death cross: 50d MA crossed below 200d MA")

    rsi_desc = "oversold" if t.rsi_1d < 30 else ("overbought" if t.rsi_1d > 70 else "healthy range")
    reasons.append(f"RSI(14) daily: {t.rsi_1d:.1f} ({rsi_desc})")

    if t.macd_signal in ("bullish_cross", "bearish_cross"):
        reasons.append(f"MACD {t.macd_signal.replace('_', ' ')} on daily chart")
    elif t.macd_signal in ("bullish", "bearish"):
        reasons.append(f"MACD histogram {t.macd_signal} (no crossover yet)")

    if t.breakout_signal != "none":
        reasons.append(f"Pattern: {t.breakout_signal.replace('_', ' ')}")
    if t.rejection_signal != "none":
        reasons.append(f"Pattern: {t.rejection_signal.replace('_', ' ')}")

    # Macro
    reasons.append(f"VIX: {m.vix_current:.1f} ({m.vix_regime}) — trend {m.vix_trend}")
    reasons.append(f"Market breadth: {m.sectors_above_50d}/{m.sectors_total} sectors above 50d MA ({m.breadth_pct:.0f}% — {m.breadth_signal})")
    if m.vix_spike:
        reasons.append("WARNING: VIX spike detected (>20% in 3 bars)")

    # Options
    reasons.append(f"IV Rank: {o.iv_rank:.1f} ({o.iv_environment}) — VIX range ${o.iv_52w_low:.1f}–${o.iv_52w_high:.1f}")
    if o.put_call_skew > 0.02:
        reasons.append(f"Put/call skew: {o.put_call_skew:.3f} — elevated put demand (fear)")
    elif o.put_call_skew < -0.02:
        reasons.append(f"Put/call skew: {o.put_call_skew:.3f} — call demand elevated (complacency)")

    reasons.append(f"Confidence: {total}/100 (tech={t.technical_score} macro={m.macro_score} opts={o.options_score})")
    return reasons
