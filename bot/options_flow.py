"""
Options Flow Tracker
Detects unusual options activity — whale-sized orders, sweeps, and large premium blocks.
Uses yfinance options chain data to find volume/OI anomalies.
"""

import logging
from dataclasses import dataclass
from datetime import date, datetime
from typing import List, Optional

import pandas as pd
import yfinance as yf

logger = logging.getLogger(__name__)

# Discord webhook for the sentiment/flow channel
FLOW_WEBHOOK_URL = "https://discord.com/api/webhooks/1510817097159151656/3m4776CwYjFJoQ8mhGR67HCmqZu9fs82gLR9X0iKikDB5Iox_07sukVvjXTic_KEg-5Y"

# Tickers to monitor for unusual flow
FLOW_TICKERS = ["SPY", "QQQ"]

# Thresholds
MIN_VOLUME_OI_RATIO = 2.0      # volume must be 2x open interest to flag
MIN_TOTAL_PREMIUM = 500_000    # minimum $500k in premium to flag as whale
MIN_VOLUME = 500               # minimum raw volume
MAX_DTE = 45                   # ignore LEAPS (too far out)
MIN_DTE = 1                    # ignore same-day expiry


@dataclass
class UnusualFlow:
    ticker: str
    option_type: str        # "call" or "put"
    strike: float
    expiration: str
    dte: int
    volume: int
    open_interest: int
    vol_oi_ratio: float
    ask: float
    total_premium: float    # volume * ask * 100
    sentiment: str          # "bullish" or "bearish"
    size: str               # "large" / "whale" / "mega"
    implied_volatility: float


def fetch_unusual_flow(ticker: str) -> List[UnusualFlow]:
    """
    Scans the full options chain for a ticker and returns unusual activity.
    Flags options where volume >> open interest and total premium is large.
    """
    try:
        t = yf.Ticker(ticker)
        expirations = t.options
        today = date.today()
        unusual = []

        for exp_str in expirations:
            exp_date = datetime.strptime(exp_str, "%Y-%m-%d").date()
            dte = (exp_date - today).days
            if not (MIN_DTE <= dte <= MAX_DTE):
                continue

            try:
                chain = t.option_chain(exp_str)
            except Exception:
                continue

            for opt_type, df in [("call", chain.calls), ("put", chain.puts)]:
                if df.empty:
                    continue

                df = df.copy()
                df["dte"] = dte
                df["opt_type"] = opt_type

                # Filter for meaningful volume
                df = df[df["volume"] >= MIN_VOLUME].copy()
                if df.empty:
                    continue

                # Calculate vol/OI ratio
                df["vol_oi_ratio"] = df["volume"] / df["openInterest"].replace(0, 1)

                # Calculate total premium
                df["ask"] = df["ask"].fillna(0)
                df["total_premium"] = df["volume"] * df["ask"] * 100

                # Filter for unusual activity
                unusual_df = df[
                    (df["vol_oi_ratio"] >= MIN_VOLUME_OI_RATIO) &
                    (df["total_premium"] >= MIN_TOTAL_PREMIUM)
                ].copy()

                for _, row in unusual_df.iterrows():
                    total = float(row["total_premium"])
                    if total >= 5_000_000:
                        size = "🐋 MEGA WHALE"
                    elif total >= 2_000_000:
                        size = "🦈 WHALE"
                    else:
                        size = "🔥 LARGE BLOCK"

                    unusual.append(UnusualFlow(
                        ticker=ticker,
                        option_type=opt_type,
                        strike=float(row["strike"]),
                        expiration=exp_str,
                        dte=dte,
                        volume=int(row["volume"]),
                        open_interest=int(row["openInterest"]),
                        vol_oi_ratio=round(float(row["vol_oi_ratio"]), 1),
                        ask=round(float(row["ask"]), 2),
                        total_premium=round(total, 0),
                        sentiment="bullish" if opt_type == "call" else "bearish",
                        size=size,
                        implied_volatility=round(float(row.get("impliedVolatility", 0)) * 100, 1),
                    ))

        # Sort by total premium descending
        unusual.sort(key=lambda x: x.total_premium, reverse=True)
        return unusual[:10]  # top 10

    except Exception as e:
        logger.exception(f"Options flow fetch error for {ticker}: {e}")
        return []


def format_flow_alert(flows: List[UnusualFlow], ticker: str, spy_price: float) -> Optional[str]:
    """Formats the flow data into a Discord message."""
    if not flows:
        return None

    calls = [f for f in flows if f.option_type == "call"]
    puts  = [f for f in flows if f.option_type == "put"]

    total_call_premium = sum(f.total_premium for f in calls)
    total_put_premium  = sum(f.total_premium for f in puts)
    total_premium      = total_call_premium + total_put_premium

    if total_call_premium > total_put_premium * 1.5:
        bias = "🟢 BULLISH FLOW"
        bias_note = "Smart money is buying CALLS — betting price goes UP"
    elif total_put_premium > total_call_premium * 1.5:
        bias = "🔴 BEARISH FLOW"
        bias_note = "Smart money is buying PUTS — betting price goes DOWN"
    else:
        bias = "🟡 MIXED FLOW"
        bias_note = "Mixed signals — both calls and puts being bought"

    now_et = datetime.now().strftime("%I:%M %p ET")
    lines = [
        f"🐋 **{ticker} Options Flow — {now_et}**",
        f"**{ticker} Price:** ${spy_price:.2f}",
        f"**Overall Bias:** {bias}",
        f"📌 {bias_note}",
        f"",
        f"💰 **Total Call Premium:** ${total_call_premium/1e6:.2f}M",
        f"💸 **Total Put Premium:** ${total_put_premium/1e6:.2f}M",
        f"📊 **Total Flow:** ${total_premium/1e6:.2f}M",
        f"",
        f"**Top Unusual Orders:**",
    ]

    for f in flows[:6]:
        sentiment_emoji = "📈" if f.sentiment == "bullish" else "📉"
        exp_fmt = datetime.strptime(f.expiration, "%Y-%m-%d").strftime("%b %d")
        lines.append(
            f"{sentiment_emoji} {f.size} | **{f.ticker} ${f.strike:.0f} {f.option_type.upper()}** "
            f"exp {exp_fmt} ({f.dte}d) | "
            f"Vol: {f.volume:,} / OI: {f.open_interest:,} ({f.vol_oi_ratio:.1f}x) | "
            f"Premium: **${f.total_premium/1e6:.2f}M**"
        )

    lines += [
        f"",
        f"⚠️ *Options flow is not a guaranteed signal. Large orders can be hedges.*"
    ]

    return "\n".join(lines)


def send_flow_alert(message: str) -> bool:
    """Sends the flow alert to the sentiment Discord channel."""
    import requests
    try:
        resp = requests.post(
            FLOW_WEBHOOK_URL,
            json={"username": "Options Flow Bot 🐋", "content": message},
            timeout=10
        )
        return resp.status_code == 204
    except Exception as e:
        logger.error(f"Flow alert send error: {e}")
        return False


def run_flow_scan(tickers: list = None) -> None:
    """Main entry point — scans all tickers and sends alerts if unusual flow found."""
    if tickers is None:
        tickers = FLOW_TICKERS

    try:
        import yfinance as yf
        all_flows = []

        for ticker in tickers:
            logger.info(f"Scanning options flow for {ticker}...")
            price_info = yf.Ticker(ticker).fast_info
            price = float(price_info.get("lastPrice", 0) or 0)
            flows = fetch_unusual_flow(ticker)

            if flows:
                message = format_flow_alert(flows, ticker, price)
                if message:
                    sent = send_flow_alert(message)
                    logger.info(f"Flow alert for {ticker}: {len(flows)} unusual orders, sent={sent}")
                    all_flows.extend(flows)
            else:
                logger.info(f"No unusual flow detected for {ticker}")

        if not all_flows:
            # Send a quiet no-activity update
            send_flow_alert(
                f"🔍 **Options Flow Scan — {datetime.now().strftime('%I:%M %p ET')}**\n"
                f"No unusual whale activity detected on SPY or QQQ right now.\n"
                f"Market flow appears normal 📊"
            )

    except Exception as e:
        logger.exception(f"Flow scan error: {e}")
