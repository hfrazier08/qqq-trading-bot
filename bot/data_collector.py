import logging
import time
from dataclasses import dataclass, field
from datetime import datetime, date
from typing import Dict, List, Optional

import pandas as pd
import yfinance as yf
import pytz

from bot.config import CONFIG

logger = logging.getLogger(__name__)

ET = pytz.timezone("America/New_York")


@dataclass
class MarketSnapshot:
    timestamp: datetime
    spy_price: float
    spy_1h: pd.DataFrame
    spy_4h: pd.DataFrame
    spy_1d: pd.DataFrame
    spy_1wk: pd.DataFrame
    vix_current: float
    vix_1d: pd.DataFrame
    sector_etf_prices: Dict[str, pd.DataFrame]
    tlt_1d: pd.DataFrame
    uup_1d: pd.DataFrame
    options_calls: pd.DataFrame
    options_puts: pd.DataFrame
    available_expirations: List[str]


def fetch_market_snapshot() -> MarketSnapshot:
    logger.info("Fetching market snapshot...")
    spy = yf.Ticker(CONFIG.spy)

    spy_1h = _retry(lambda: _fetch_spy_1h(spy))
    spy_1d = _retry(lambda: _fetch_history(spy, period="1y", interval="1d"))
    spy_1wk = _retry(lambda: _fetch_history(spy, period="3y", interval="1wk"))
    spy_4h = _resample_to_4h(spy_1h)

    spy_price = _fetch_live_price(spy, spy_1d)

    vix = yf.Ticker(CONFIG.vix)
    vix_1d = _retry(lambda: _fetch_history(vix, period="1y", interval="1d"))
    vix_current = _fetch_live_price(vix, vix_1d)

    sector_data = _retry(lambda: _fetch_sector_etfs())
    tlt_1d = sector_data.pop("TLT", pd.DataFrame())
    uup_1d = sector_data.pop("UUP", pd.DataFrame())

    calls, puts, expirations = _retry(lambda: _fetch_options_chain(spy))

    logger.info(
        f"Snapshot: {CONFIG.spy}={spy_price:.2f} VIX={vix_current:.2f} "
        f"expirations={len(expirations)}"
    )
    return MarketSnapshot(
        timestamp=datetime.now(ET),
        spy_price=spy_price,
        spy_1h=spy_1h,
        spy_4h=spy_4h,
        spy_1d=spy_1d,
        spy_1wk=spy_1wk,
        vix_current=vix_current,
        vix_1d=vix_1d,
        sector_etf_prices=sector_data,
        tlt_1d=tlt_1d,
        uup_1d=uup_1d,
        options_calls=calls,
        options_puts=puts,
        available_expirations=expirations,
    )


def _fetch_live_price(ticker: yf.Ticker, fallback_df: pd.DataFrame) -> float:
    """Prefer a live/last quote over the (possibly several-minutes-delayed)
    daily bar close. fast_info pulls a near-real-time last price; the daily
    bar is only used if that lookup fails entirely."""
    try:
        last = ticker.fast_info.get("lastPrice")
        if last:
            return float(last)
    except Exception as e:
        logger.warning(f"Live price lookup failed, falling back to daily close: {e}")
    return float(fallback_df["Close"].dropna().iloc[-1])


def _fetch_spy_1h(ticker: yf.Ticker) -> pd.DataFrame:
    df = ticker.history(period="60d", interval="1h")
    return _normalize_df(df)


def _fetch_history(ticker: yf.Ticker, period: str, interval: str) -> pd.DataFrame:
    df = ticker.history(period=period, interval=interval)
    return _normalize_df(df)


def _normalize_df(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    if df.index.tz is not None:
        df.index = df.index.tz_convert("America/New_York").tz_localize(None)
    return df[["Open", "High", "Low", "Close", "Volume"]].copy()


def _resample_to_4h(df_1h: pd.DataFrame) -> pd.DataFrame:
    if df_1h.empty:
        return df_1h
    return (
        df_1h.resample("4h")
        .agg({"Open": "first", "High": "max", "Low": "min", "Close": "last", "Volume": "sum"})
        .dropna()
    )


def _fetch_sector_etfs() -> Dict[str, pd.DataFrame]:
    tickers = CONFIG.sector_etfs + [CONFIG.tlt, CONFIG.uup]
    raw = yf.download(tickers, period="1y", interval="1d", group_by="ticker", progress=False)
    result: Dict[str, pd.DataFrame] = {}
    for t in tickers:
        try:
            df = raw[t][["Open", "High", "Low", "Close", "Volume"]].copy().dropna()
            if df.index.tz is not None:
                df.index = df.index.tz_convert("America/New_York").tz_localize(None)
            result[t] = df
        except (KeyError, TypeError):
            logger.warning(f"No data for {t}")
            result[t] = pd.DataFrame()
    return result


def _fetch_options_chain(spy: yf.Ticker):
    today = date.today()
    all_expirations = spy.options  # tuple of date strings
    target_exps = []
    for exp_str in all_expirations:
        try:
            exp_date = datetime.strptime(exp_str, "%Y-%m-%d").date()
            dte = (exp_date - today).days
            if CONFIG.target_dte_min <= dte <= CONFIG.target_dte_max:
                target_exps.append(exp_str)
        except ValueError:
            continue

    all_calls: List[pd.DataFrame] = []
    all_puts: List[pd.DataFrame] = []

    for exp_str in target_exps:
        try:
            chain = spy.option_chain(exp_str)
            calls = chain.calls.copy()
            puts = chain.puts.copy()
            calls["expiration"] = exp_str
            puts["expiration"] = exp_str
            exp_date = datetime.strptime(exp_str, "%Y-%m-%d").date()
            calls["dte"] = (exp_date - today).days
            puts["dte"] = (exp_date - today).days
            all_calls.append(calls)
            all_puts.append(puts)
            time.sleep(0.2)
        except Exception as e:
            logger.warning(f"Failed to fetch chain for {exp_str}: {e}")

    calls_df = pd.concat(all_calls, ignore_index=True) if all_calls else pd.DataFrame()
    puts_df = pd.concat(all_puts, ignore_index=True) if all_puts else pd.DataFrame()
    return calls_df, puts_df, target_exps


def _retry(fn, attempts: int = 3, delay: float = 1.5):
    last_exc = None
    for attempt in range(attempts):
        try:
            return fn()
        except Exception as e:
            last_exc = e
            logger.warning(f"Attempt {attempt + 1}/{attempts} failed: {e}")
            if attempt < attempts - 1:
                time.sleep(delay * (attempt + 1))
    raise last_exc
