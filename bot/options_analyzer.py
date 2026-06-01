import logging
from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
import pandas as pd
from scipy.stats import norm

logger = logging.getLogger(__name__)


@dataclass
class OptionsAnalysis:
    iv_rank: float
    iv_current: float
    iv_52w_high: float
    iv_52w_low: float
    iv_environment: str     # "low", "moderate", "high", "extreme"
    put_call_skew: float    # positive = fear (puts more expensive)
    term_structure: str     # "contango", "backwardation", "flat"
    best_calls: pd.DataFrame
    best_puts: pd.DataFrame
    options_score: int
    preferred_direction: str  # "calls", "puts", "neutral"


def analyze_options(
    calls: pd.DataFrame,
    puts: pd.DataFrame,
    spy_price: float,
    expirations: list,
    vix_1d: pd.DataFrame,
) -> OptionsAnalysis:
    iv_rank, iv_current, iv_52w_high, iv_52w_low = compute_iv_rank_via_vix(vix_1d)
    iv_env = _iv_environment(iv_rank)

    best_calls = _filter_and_enrich(calls, spy_price, "call") if not calls.empty else pd.DataFrame()
    best_puts = _filter_and_enrich(puts, spy_price, "put") if not puts.empty else pd.DataFrame()

    skew = _compute_skew(calls, puts, spy_price)
    term_structure = _compute_term_structure(calls, spy_price)
    score, direction = _score_options(iv_rank, skew, term_structure, best_calls)

    return OptionsAnalysis(
        iv_rank=round(iv_rank, 1),
        iv_current=round(iv_current, 2),
        iv_52w_high=round(iv_52w_high, 2),
        iv_52w_low=round(iv_52w_low, 2),
        iv_environment=iv_env,
        put_call_skew=round(skew, 3),
        term_structure=term_structure,
        best_calls=best_calls,
        best_puts=best_puts,
        options_score=score,
        preferred_direction=direction,
    )


def compute_iv_rank_via_vix(vix_1d: pd.DataFrame) -> Tuple[float, float, float, float]:
    """IV Rank using VIX as SPY's implied volatility proxy (CBOE definition)."""
    if vix_1d.empty:
        return 50.0, 20.0, 30.0, 12.0
    closes = vix_1d["Close"].dropna()
    if len(closes) < 10:
        return 50.0, float(closes.iloc[-1]), float(closes.max()), float(closes.min())
    current = float(closes.iloc[-1])
    high_52w = float(closes.max())
    low_52w = float(closes.min())
    rng = high_52w - low_52w
    iv_rank = (current - low_52w) / rng * 100 if rng > 0 else 50.0
    return round(iv_rank, 1), current, high_52w, low_52w


def _iv_environment(iv_rank: float) -> str:
    if iv_rank < 20:
        return "low"
    if iv_rank < 50:
        return "moderate"
    if iv_rank < 80:
        return "high"
    return "extreme"


def _filter_and_enrich(
    chain: pd.DataFrame,
    spy_price: float,
    option_type: str,
    min_oi: int = 200,
    min_vol: int = 50,
    max_spread_pct: float = 0.15,
) -> pd.DataFrame:
    if chain.empty:
        return pd.DataFrame()

    df = chain.copy()
    required = {"strike", "bid", "ask", "openInterest", "volume", "impliedVolatility", "dte"}
    if not required.issubset(df.columns):
        return pd.DataFrame()

    df["mid"] = (df["bid"] + df["ask"]) / 2
    df["spread_pct"] = (df["ask"] - df["bid"]) / df["mid"].replace(0, np.nan)

    mask = (
        (df["openInterest"] >= min_oi) &
        (df["volume"] >= min_vol) &
        (df["spread_pct"] <= max_spread_pct) &
        (df["mid"] > 0.05)
    )
    df = df[mask].copy()
    if df.empty:
        return df

    # Add Greeks
    risk_free = 0.05
    greeks_rows = []
    for _, row in df.iterrows():
        T = max(row["dte"] / 365.0, 1 / 365.0)
        sigma = row["impliedVolatility"]
        if sigma <= 0 or np.isnan(sigma):
            sigma = 0.20
        g = black_scholes_greeks(spy_price, row["strike"], T, risk_free, sigma, option_type)
        greeks_rows.append(g)

    greeks_df = pd.DataFrame(greeks_rows, index=df.index)
    df = pd.concat([df, greeks_df], axis=1)
    return df.reset_index(drop=True)


def black_scholes_greeks(
    S: float, K: float, T: float, r: float, sigma: float, option_type: str
) -> dict:
    T = max(T, 1 / 365.0)
    try:
        d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
        d2 = d1 - sigma * np.sqrt(T)
        if option_type == "call":
            delta = norm.cdf(d1)
            price = S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
        else:
            delta = norm.cdf(d1) - 1
            price = K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)
        gamma = norm.pdf(d1) / (S * sigma * np.sqrt(T))
        theta = (-(S * norm.pdf(d1) * sigma) / (2 * np.sqrt(T))) / 365
        vega = S * norm.pdf(d1) * np.sqrt(T) / 100
        return {"bs_delta": round(delta, 4), "bs_gamma": round(gamma, 6),
                "bs_theta": round(theta, 4), "bs_vega": round(vega, 4),
                "bs_price": round(price, 3)}
    except Exception:
        return {"bs_delta": 0.0, "bs_gamma": 0.0, "bs_theta": 0.0, "bs_vega": 0.0, "bs_price": 0.0}


def _compute_skew(calls: pd.DataFrame, puts: pd.DataFrame, spy_price: float) -> float:
    """Put/call skew: 5% OTM put IV minus 5% OTM call IV. Positive = fear."""
    try:
        otm_call_strike = spy_price * 1.05
        otm_put_strike = spy_price * 0.95

        if not calls.empty and "strike" in calls.columns and "impliedVolatility" in calls.columns:
            call_row = calls.iloc[(calls["strike"] - otm_call_strike).abs().argsort()[:1]]
            call_iv = float(call_row["impliedVolatility"].iloc[0])
        else:
            call_iv = 0.20

        if not puts.empty and "strike" in puts.columns and "impliedVolatility" in puts.columns:
            put_row = puts.iloc[(puts["strike"] - otm_put_strike).abs().argsort()[:1]]
            put_iv = float(put_row["impliedVolatility"].iloc[0])
        else:
            put_iv = 0.20

        return put_iv - call_iv
    except Exception:
        return 0.0


def _compute_term_structure(calls: pd.DataFrame, spy_price: float) -> str:
    """Compare near-term vs far-term ATM IV. Contango = normal (near < far)."""
    if calls.empty or "dte" not in calls.columns or "impliedVolatility" not in calls.columns:
        return "flat"
    try:
        atm = calls[calls["strike"].between(spy_price * 0.99, spy_price * 1.01)]
        if atm.empty:
            return "flat"
        near = atm[atm["dte"] <= atm["dte"].median()]["impliedVolatility"].mean()
        far = atm[atm["dte"] > atm["dte"].median()]["impliedVolatility"].mean()
        if np.isnan(near) or np.isnan(far):
            return "flat"
        diff = far - near
        if diff > 0.005:
            return "contango"
        if diff < -0.005:
            return "backwardation"
        return "flat"
    except Exception:
        return "flat"


def _score_options(
    iv_rank: float,
    skew: float,
    term_structure: str,
    best_calls: pd.DataFrame,
) -> Tuple[int, str]:
    score = 0

    # IV Rank contribution (max 8)
    if 20 <= iv_rank <= 50:
        score += 8   # sweet spot for debit spreads
    elif 50 < iv_rank <= 80:
        score += 6   # high IV — credit spreads or wider debit
    elif iv_rank > 80:
        score += 4   # extreme — caution
    else:
        score += 2   # low IV

    # Skew (max 6)
    skew_abs = abs(skew)
    if skew_abs > 0.03:
        score += 6   # strong directional skew is informative
    elif skew_abs > 0.01:
        score += 3
    else:
        score += 1

    # Term structure (max 4)
    if term_structure == "contango":
        score += 4   # normal market — spread buyers are fine
    elif term_structure == "backwardation":
        score += 2   # stress — favors puts
    else:
        score += 2

    # Liquidity bonus (max 2)
    if not best_calls.empty and len(best_calls) >= 5:
        score += 2

    score = max(0, min(20, score))

    # Direction preference
    if skew > 0.02:
        direction = "puts"    # fear = puts more expensive
    elif skew < -0.02:
        direction = "calls"   # complacency = calls cheaper
    else:
        direction = "neutral"

    return score, direction
