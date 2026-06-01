import logging
import math
from dataclasses import dataclass
from datetime import date, datetime
from typing import Optional

import numpy as np
import pandas as pd

from bot.options_analyzer import black_scholes_greeks

logger = logging.getLogger(__name__)


@dataclass
class SingleLegTrade:
    direction: str           # "call" or "put"
    signal_timestamp: str

    expiration: str          # "YYYY-MM-DD"
    expiration_label: str    # e.g. "Jun 6, 2025"
    dte: int

    strike: float
    option_type: str         # "Call" or "Put"

    ask_price: float         # what you'll pay per contract
    bid_price: float
    mid_price: float

    breakeven: float         # strike + ask (calls) or strike - ask (puts)
    profit_target: float     # breakeven + 50% of ask (take profit level)
    stop_loss_price: float   # exit if option drops to 50% of ask paid

    delta: float
    theta: float
    implied_volatility: float

    contracts: int
    cost_per_contract: float   # ask * 100
    total_cost: float          # contracts * cost_per_contract
    account_size: float
    risk_pct_used: float

    # For display
    spy_price: float


def construct_trade(
    signal_direction: str,
    spy_price: float,
    calls: pd.DataFrame,
    puts: pd.DataFrame,
    expirations: list,
    account_size: float,
    risk_pct: float,
    target_dte_min: int = 7,
    target_dte_max: int = 45,
    signal_timestamp: str = "",
    **kwargs,   # absorb any legacy keyword args
) -> Optional[SingleLegTrade]:
    try:
        exp = _select_expiration(expirations, target_dte_min, target_dte_max)
        if not exp:
            logger.warning("No suitable expiration found")
            return None

        dte = (datetime.strptime(exp, "%Y-%m-%d").date() - date.today()).days
        exp_label = datetime.strptime(exp, "%Y-%m-%d").strftime("%b %-d, %Y")

        if signal_direction == "bullish":
            chain = calls[calls["expiration"] == exp] if not calls.empty else pd.DataFrame()
            return _build_call(chain, spy_price, dte, exp, exp_label, risk_pct, account_size, signal_timestamp)
        else:
            chain = puts[puts["expiration"] == exp] if not puts.empty else pd.DataFrame()
            return _build_put(chain, spy_price, dte, exp, exp_label, risk_pct, account_size, signal_timestamp)

    except Exception as e:
        logger.error(f"Trade construction failed: {e}")
        return None


def _select_expiration(
    expirations: list,
    target_min: int,
    target_max: int,
    prefer_dte: int = 14,
) -> Optional[str]:
    today = date.today()
    candidates = []
    for exp_str in expirations:
        try:
            exp_date = datetime.strptime(exp_str, "%Y-%m-%d").date()
            dte = (exp_date - today).days
            if target_min <= dte <= target_max:
                candidates.append((exp_str, dte))
        except ValueError:
            continue
    if not candidates:
        return None
    candidates.sort(key=lambda x: abs(x[1] - prefer_dte))
    return candidates[0][0]


def _build_call(
    chain: pd.DataFrame,
    spy_price: float,
    dte: int,
    expiration: str,
    exp_label: str,
    risk_pct: float,
    account_size: float,
    signal_timestamp: str,
) -> Optional[SingleLegTrade]:
    if chain.empty or "strike" not in chain.columns:
        return None

    strikes = sorted(chain["strike"].unique())

    # Pick ATM or 1 strike OTM call
    atm_candidates = [s for s in strikes if s >= spy_price]
    if not atm_candidates:
        return None
    strike = atm_candidates[0]  # first strike at or just above current price

    row = _get_row(chain, strike)
    if row is None:
        return None

    ask = float(row.get("ask", 0) or 0)
    bid = float(row.get("bid", 0) or 0)
    mid = round((ask + bid) / 2, 2) if ask > 0 else float(row.get("lastPrice", 0) or 0)
    if ask <= 0:
        ask = mid

    breakeven = round(strike + ask, 2)
    profit_target = round(breakeven + ask * 0.5, 2)   # 50% gain on option price
    stop_loss_price = round(ask * 0.5, 2)              # exit if option loses 50%

    # Greeks
    T = max(dte / 365.0, 1 / 365.0)
    sigma = float(row.get("impliedVolatility", 0.20) or 0.20)
    if sigma <= 0:
        sigma = 0.20
    greeks = black_scholes_greeks(spy_price, strike, T, 0.05, sigma, "call")

    contracts = _size_contracts(ask, account_size, risk_pct)
    cost_per = round(ask * 100, 2)
    total = round(contracts * cost_per, 2)

    return SingleLegTrade(
        direction="call",
        signal_timestamp=signal_timestamp,
        expiration=expiration,
        expiration_label=exp_label,
        dte=dte,
        strike=strike,
        option_type="Call",
        ask_price=round(ask, 2),
        bid_price=round(bid, 2),
        mid_price=mid,
        breakeven=breakeven,
        profit_target=profit_target,
        stop_loss_price=stop_loss_price,
        delta=round(greeks.get("bs_delta", 0.5), 3),
        theta=round(greeks.get("bs_theta", 0), 4),
        implied_volatility=round(sigma * 100, 1),
        contracts=contracts,
        cost_per_contract=cost_per,
        total_cost=total,
        account_size=account_size,
        risk_pct_used=round(total / account_size * 100, 2),
        spy_price=spy_price,
    )


def _build_put(
    chain: pd.DataFrame,
    spy_price: float,
    dte: int,
    expiration: str,
    exp_label: str,
    risk_pct: float,
    account_size: float,
    signal_timestamp: str,
) -> Optional[SingleLegTrade]:
    if chain.empty or "strike" not in chain.columns:
        return None

    strikes = sorted(chain["strike"].unique(), reverse=True)

    # Pick ATM or 1 strike OTM put
    atm_candidates = [s for s in strikes if s <= spy_price]
    if not atm_candidates:
        return None
    strike = atm_candidates[0]  # first strike at or just below current price

    row = _get_row(chain, strike)
    if row is None:
        return None

    ask = float(row.get("ask", 0) or 0)
    bid = float(row.get("bid", 0) or 0)
    mid = round((ask + bid) / 2, 2) if ask > 0 else float(row.get("lastPrice", 0) or 0)
    if ask <= 0:
        ask = mid

    breakeven = round(strike - ask, 2)
    profit_target = round(breakeven - ask * 0.5, 2)
    stop_loss_price = round(ask * 0.5, 2)

    T = max(dte / 365.0, 1 / 365.0)
    sigma = float(row.get("impliedVolatility", 0.20) or 0.20)
    if sigma <= 0:
        sigma = 0.20
    greeks = black_scholes_greeks(spy_price, strike, T, 0.05, sigma, "put")

    contracts = _size_contracts(ask, account_size, risk_pct)
    cost_per = round(ask * 100, 2)
    total = round(contracts * cost_per, 2)

    return SingleLegTrade(
        direction="put",
        signal_timestamp=signal_timestamp,
        expiration=expiration,
        expiration_label=exp_label,
        dte=dte,
        strike=strike,
        option_type="Put",
        ask_price=round(ask, 2),
        bid_price=round(bid, 2),
        mid_price=mid,
        breakeven=breakeven,
        profit_target=profit_target,
        stop_loss_price=stop_loss_price,
        delta=round(greeks.get("bs_delta", -0.5), 3),
        theta=round(greeks.get("bs_theta", 0), 4),
        implied_volatility=round(sigma * 100, 1),
        contracts=contracts,
        cost_per_contract=cost_per,
        total_cost=total,
        account_size=account_size,
        risk_pct_used=round(total / account_size * 100, 2),
        spy_price=spy_price,
    )


def _get_row(chain: pd.DataFrame, strike: float) -> Optional[dict]:
    rows = chain[chain["strike"] == strike]
    if rows.empty:
        return None
    return rows.iloc[0].to_dict()


def _size_contracts(ask: float, account_size: float, risk_pct: float) -> int:
    if ask <= 0:
        return 1
    max_risk = account_size * risk_pct
    contracts = int(math.floor(max_risk / (ask * 100)))
    return max(1, min(contracts, 10))
