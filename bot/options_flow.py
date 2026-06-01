"""
Options Flow & Smart Money Tracker
- Whale detection (large premium blocks)
- Put/Call ratio (crowd sentiment)
- Dark pool / institutional buying signals
- Fear & Greed indicator
- Follow the money summary
"""

import logging
import requests
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from typing import List, Optional

import yfinance as yf
import numpy as np

logger = logging.getLogger(__name__)

FLOW_WEBHOOK_URL = "https://discord.com/api/webhooks/1510817097159151656/3m4776CwYjFJoQ8mhGR67HCmqZu9fs82gLR9X0iKikDB5Iox_07sukVvjXTic_KEg-5Y"
FLOW_TICKERS = ["SPY", "QQQ"]

MIN_VOLUME_OI_RATIO = 2.0
MIN_TOTAL_PREMIUM   = 500_000
MIN_VOLUME          = 500
MAX_DTE             = 45
MIN_DTE             = 1


@dataclass
class UnusualFlow:
    ticker: str
    option_type: str
    strike: float
    expiration: str
    dte: int
    volume: int
    open_interest: int
    vol_oi_ratio: float
    ask: float
    total_premium: float
    sentiment: str
    size: str
    implied_volatility: float


@dataclass
class MarketSentiment:
    ticker: str
    price: float
    # Put/Call ratio
    put_call_ratio: float
    pc_signal: str          # "bullish" / "bearish" / "neutral"
    pc_note: str
    # Volume analysis
    total_call_volume: int
    total_put_volume: int
    call_put_vol_pct: float  # % of volume that is calls
    # Premium analysis
    total_call_premium: float
    total_put_premium: float
    # Whale activity
    whale_calls: int
    whale_puts: int
    whale_bias: str
    # Overall
    crowd_bias: str          # what the crowd is doing
    smart_money_bias: str    # what whales are doing
    action: str              # what YOU should consider doing


def _send(message: str, username: str = "Options Flow Bot 🐋") -> bool:
    try:
        r = requests.post(FLOW_WEBHOOK_URL,
                          json={"username": username, "content": message},
                          timeout=10)
        return r.status_code == 204
    except Exception as e:
        logger.error(f"Flow send error: {e}")
        return False


def fetch_full_chain(ticker: str):
    """Returns (calls_df, puts_df, expirations) with DTE filter applied."""
    t = yf.Ticker(ticker)
    expirations = t.options
    today = date.today()
    all_calls, all_puts = [], []

    for exp_str in expirations:
        exp_date = datetime.strptime(exp_str, "%Y-%m-%d").date()
        dte = (exp_date - today).days
        if not (MIN_DTE <= dte <= MAX_DTE):
            continue
        try:
            chain = t.option_chain(exp_str)
            c = chain.calls.copy(); c["expiration"] = exp_str; c["dte"] = dte
            p = chain.puts.copy();  p["expiration"] = exp_str; p["dte"] = dte
            all_calls.append(c)
            all_puts.append(p)
        except Exception:
            continue

    import pandas as pd
    calls = pd.concat(all_calls) if all_calls else pd.DataFrame()
    puts  = pd.concat(all_puts)  if all_puts  else pd.DataFrame()
    return calls, puts


def analyze_sentiment(ticker: str, price: float) -> Optional[MarketSentiment]:
    """Full sentiment analysis — crowd + smart money."""
    try:
        calls, puts = fetch_full_chain(ticker)
        if calls.empty or puts.empty:
            return None

        # ── Volume analysis ──
        total_call_vol = int(calls["volume"].fillna(0).sum())
        total_put_vol  = int(puts["volume"].fillna(0).sum())
        total_vol = total_call_vol + total_put_vol
        call_pct = (total_call_vol / total_vol * 100) if total_vol else 50

        # ── Put/Call ratio ──
        pc_ratio = round(total_put_vol / total_call_vol, 2) if total_call_vol > 0 else 1.0
        if pc_ratio < 0.7:
            pc_signal, pc_note = "bullish", "Crowd is buying CALLS aggressively — everyone is bullish"
        elif pc_ratio > 1.2:
            pc_signal, pc_note = "bearish", "Crowd is buying PUTS aggressively — everyone is fearful"
        else:
            pc_signal, pc_note = "neutral", "Balanced call/put activity — no clear crowd bias"

        # ── Premium analysis ──
        calls["premium"] = calls["volume"].fillna(0) * calls["ask"].fillna(0) * 100
        puts["premium"]  = puts["volume"].fillna(0) * puts["ask"].fillna(0) * 100
        total_call_prem = float(calls["premium"].sum())
        total_put_prem  = float(puts["premium"].sum())

        # ── Whale detection ──
        calls["vol_oi_ratio"] = calls["volume"] / calls["openInterest"].replace(0, 1)
        puts["vol_oi_ratio"]  = puts["volume"]  / puts["openInterest"].replace(0, 1)

        whale_call_df = calls[
            (calls["volume"] >= MIN_VOLUME) &
            (calls["vol_oi_ratio"] >= MIN_VOLUME_OI_RATIO) &
            (calls["premium"] >= MIN_TOTAL_PREMIUM)
        ]
        whale_put_df = puts[
            (puts["volume"] >= MIN_VOLUME) &
            (puts["vol_oi_ratio"] >= MIN_VOLUME_OI_RATIO) &
            (puts["premium"] >= MIN_TOTAL_PREMIUM)
        ]

        whale_calls = len(whale_call_df)
        whale_puts  = len(whale_put_df)
        whale_call_prem = float(whale_call_df["premium"].sum())
        whale_put_prem  = float(whale_put_df["premium"].sum())

        if whale_call_prem > whale_put_prem * 1.5:
            whale_bias = "bullish"
        elif whale_put_prem > whale_call_prem * 1.5:
            whale_bias = "bearish"
        else:
            whale_bias = "neutral"

        # ── Crowd bias (put/call ratio) ──
        crowd_bias = pc_signal

        # ── Smart money bias (whale premium) ──
        smart_money_bias = whale_bias

        # ── Action recommendation ──
        # When smart money AND crowd agree → strong signal
        # When they disagree → follow smart money (whales know more)
        if smart_money_bias == "bullish" and crowd_bias == "bullish":
            action = "🚀 STRONG BUY SIGNAL — Both whales AND crowd are bullish. High conviction call opportunity."
        elif smart_money_bias == "bullish" and crowd_bias != "bullish":
            action = "🐋 FOLLOW THE WHALES — Smart money is buying calls while crowd is uncertain. Lean bullish."
        elif smart_money_bias == "bearish" and crowd_bias == "bearish":
            action = "🔴 STRONG SELL SIGNAL — Both whales AND crowd are bearish. Consider puts."
        elif smart_money_bias == "bearish" and crowd_bias != "bearish":
            action = "🐋 FOLLOW THE WHALES — Smart money is buying puts while crowd is uncertain. Lean bearish."
        elif smart_money_bias == "neutral" and crowd_bias == "bullish":
            action = "📈 CROWD IS BULLISH — No big whale activity but retail is buying calls. Moderate bullish."
        elif smart_money_bias == "neutral" and crowd_bias == "bearish":
            action = "📉 CROWD IS FEARFUL — No big whale activity but retail is buying puts. Be cautious."
        else:
            action = "⏳ NO CLEAR SIGNAL — Mixed or neutral activity. Wait for a clearer setup."

        return MarketSentiment(
            ticker=ticker,
            price=price,
            put_call_ratio=pc_ratio,
            pc_signal=pc_signal,
            pc_note=pc_note,
            total_call_volume=total_call_vol,
            total_put_volume=total_put_vol,
            call_put_vol_pct=round(call_pct, 1),
            total_call_premium=total_call_prem,
            total_put_premium=total_put_prem,
            whale_calls=whale_calls,
            whale_puts=whale_puts,
            whale_bias=whale_bias,
            crowd_bias=crowd_bias,
            smart_money_bias=smart_money_bias,
            action=action,
        )
    except Exception as e:
        logger.exception(f"Sentiment analysis error for {ticker}: {e}")
        return None


def fetch_unusual_flow(ticker: str, calls, puts) -> List[UnusualFlow]:
    """Find specific unusual option orders from pre-fetched chain data."""
    unusual = []
    try:
        for opt_type, df in [("call", calls), ("put", puts)]:
            if df.empty:
                continue
            df = df.copy()
            df["vol_oi_ratio"] = df["volume"].fillna(0) / df["openInterest"].replace(0, 1)
            df["premium"] = df["volume"].fillna(0) * df["ask"].fillna(0) * 100
            unusual_df = df[
                (df["volume"] >= MIN_VOLUME) &
                (df["vol_oi_ratio"] >= MIN_VOLUME_OI_RATIO) &
                (df["premium"] >= MIN_TOTAL_PREMIUM)
            ].copy()

            for _, row in unusual_df.iterrows():
                total = float(row["premium"])
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
                    expiration=row.get("expiration", ""),
                    dte=int(row.get("dte", 0)),
                    volume=int(row["volume"]),
                    open_interest=int(row["openInterest"]),
                    vol_oi_ratio=round(float(row["vol_oi_ratio"]), 1),
                    ask=round(float(row["ask"]), 2),
                    total_premium=round(total, 0),
                    sentiment="bullish" if opt_type == "call" else "bearish",
                    size=size,
                    implied_volatility=round(float(row.get("impliedVolatility", 0)) * 100, 1),
                ))
    except Exception as e:
        logger.exception(f"Unusual flow error: {e}")

    unusual.sort(key=lambda x: x.total_premium, reverse=True)
    return unusual[:8]


def format_full_report(sentiment: MarketSentiment, flows: List[UnusualFlow]) -> str:
    """Builds the full Discord message."""
    now_et = datetime.now().strftime("%I:%M %p ET")

    crowd_emoji  = "🟢" if sentiment.crowd_bias == "bullish" else ("🔴" if sentiment.crowd_bias == "bearish" else "🟡")
    whale_emoji  = "🟢" if sentiment.smart_money_bias == "bullish" else ("🔴" if sentiment.smart_money_bias == "bearish" else "🟡")

    lines = [
        f"🐋 **{sentiment.ticker} Smart Money Report — {now_et}**",
        f"**Price:** ${sentiment.price:.2f}",
        f"━━━━━━━━━━━━━━━━━━━━━━",
        f"",
        f"**📊 CROWD SENTIMENT** {crowd_emoji}",
        f"Put/Call Ratio: **{sentiment.put_call_ratio:.2f}** — {sentiment.pc_note}",
        f"Call Volume: **{sentiment.total_call_volume:,}** ({sentiment.call_put_vol_pct:.0f}% of total)",
        f"Put Volume:  **{sentiment.total_put_volume:,}** ({100-sentiment.call_put_vol_pct:.0f}% of total)",
        f"Call Premium: **${sentiment.total_call_premium/1e6:.2f}M** | Put Premium: **${sentiment.total_put_premium/1e6:.2f}M**",
        f"",
        f"**🐋 SMART MONEY (WHALES)** {whale_emoji}",
        f"Whale Call Orders: **{sentiment.whale_calls}** | Whale Put Orders: **{sentiment.whale_puts}**",
        f"Whale Bias: **{sentiment.smart_money_bias.upper()}**",
        f"",
        f"**💡 WHAT TO DO:**",
        f"{sentiment.action}",
        f"━━━━━━━━━━━━━━━━━━━━━━",
    ]

    if flows:
        lines.append(f"**🔥 Top Unusual Orders:**")
        for f in flows[:5]:
            s_emoji = "📈" if f.sentiment == "bullish" else "📉"
            exp_fmt = datetime.strptime(f.expiration, "%Y-%m-%d").strftime("%b %d") if f.expiration else "?"
            lines.append(
                f"{s_emoji} {f.size} | **${f.strike:.0f} {f.option_type.upper()}** {exp_fmt} ({f.dte}d) | "
                f"Vol: {f.volume:,} vs OI: {f.open_interest:,} ({f.vol_oi_ratio:.1f}x) | "
                f"**${f.total_premium/1e6:.2f}M**"
            )
        lines.append("")

    lines.append("⚠️ *Flow data is informational. Large orders can be hedges. Always use with technical signals.*")
    return "\n".join(lines)


def run_sentiment_daily_report() -> None:
    """End of day full sentiment report in standard format."""
    try:
        now_et = datetime.now()
        spy_info = yf.Ticker("SPY").fast_info
        qqq_info = yf.Ticker("QQQ").fast_info
        vix      = float(yf.Ticker("^VIX").fast_info.get("lastPrice", 0) or 0)

        spy_price = float(spy_info.get("lastPrice", 0) or 0)
        spy_prev  = float(spy_info.get("previousClose", spy_price) or spy_price)
        qqq_price = float(qqq_info.get("lastPrice", 0) or 0)
        qqq_prev  = float(qqq_info.get("previousClose", qqq_price) or qqq_price)
        spy_pct   = (spy_price - spy_prev) / spy_prev * 100 if spy_prev else 0
        qqq_pct   = (qqq_price - qqq_prev) / qqq_prev * 100 if qqq_prev else 0

        # Run full analysis for SPY
        spy_calls, spy_puts = fetch_full_chain("SPY")
        sentiment = analyze_sentiment("SPY", spy_price)
        flows     = fetch_unusual_flow("SPY", spy_calls, spy_puts)

        # Tomorrow outlook
        if spy_pct > 0.5 and sentiment and sentiment.crowd_bias == "bullish":
            outlook = "📈 **Bullish setup for tomorrow** — strong close + bullish flow. Watch for calls at open."
        elif spy_pct < -0.5 and sentiment and sentiment.crowd_bias == "bearish":
            outlook = "📉 **Bearish setup for tomorrow** — weak close + bearish flow. Watch for puts at open."
        elif vix > 20:
            outlook = "⚠️ **Volatile conditions** — elevated VIX. Wait for clear signal before trading."
        else:
            outlook = "⏳ **Mixed signals** — no strong directional bias. Let the bots find the setup."

        spy_e = "🟢" if spy_pct >= 0 else "🔴"
        qqq_e = "🟢" if qqq_pct >= 0 else "🔴"

        header = (
            f"📊 **End of Day Sentiment — {now_et.strftime('%b %d, %Y')}**\n"
            f"{spy_e} SPY: ${spy_price:.2f} ({spy_pct:+.2f}%)  "
            f"{qqq_e} QQQ: ${qqq_price:.2f} ({qqq_pct:+.2f}%)  "
            f"😨 VIX: {vix:.2f}\n"
        )

        # Use standard format_full_report for the body
        body = format_full_report(sentiment, flows) if sentiment else "⚠️ Could not fetch sentiment data."

        footer = f"\n**🔭 Tomorrow's Outlook:**\n{outlook}\nSee you tomorrow at **9:00 AM ET**! 👋"

        _send(header + body + footer, username="Options Flow Bot 🐋")
        logger.info("Sentiment daily report sent")
    except Exception as e:
        logger.exception(f"Sentiment daily report error: {e}")


def run_sentiment_weekly_report() -> None:
    """Friday end of week full sentiment report."""
    try:
        now_et = datetime.now()

        spy_hist = yf.Ticker("SPY").history(period="5d")
        qqq_hist = yf.Ticker("QQQ").history(period="5d")
        vix_hist = yf.Ticker("^VIX").history(period="5d")

        def week_change(hist):
            if len(hist) >= 2:
                o = float(hist["Close"].iloc[0])
                c = float(hist["Close"].iloc[-1])
                return c, (c - o) / o * 100
            return 0, 0

        spy_close, spy_wpct = week_change(spy_hist)
        qqq_close, qqq_wpct = week_change(qqq_hist)
        avg_vix = round(float(vix_hist["Close"].mean()), 2) if not vix_hist.empty else 0
        vix_env = "calm (good for buyers)" if avg_vix < 16 else ("normal" if avg_vix < 22 else "elevated (volatile week)")

        if not spy_hist.empty:
            spy_hist["dp"] = spy_hist["Close"].pct_change() * 100
            best_day  = float(spy_hist["dp"].max())
            worst_day = float(spy_hist["dp"].min())
        else:
            best_day = worst_day = 0

        next_week = (
            "📈 Bullish momentum heading into next week" if spy_wpct > 1
            else "📉 Bearish momentum — be cautious Monday open" if spy_wpct < -1
            else "⏳ Flat week — watch for breakout direction Monday"
        )

        spy_e = "🟢" if spy_wpct >= 0 else "🔴"
        qqq_e = "🟢" if qqq_wpct >= 0 else "🔴"

        # Run full live analysis for Friday close
        spy_price = spy_close
        spy_calls, spy_puts = fetch_full_chain("SPY")
        sentiment = analyze_sentiment("SPY", spy_price)
        flows     = fetch_unusual_flow("SPY", spy_calls, spy_puts)

        header = (
            f"📊 **Weekly Sentiment Report — Week of {now_et.strftime('%b %d, %Y')}**\n"
            f"━━━━━━━━━━━━━━━━━━━━━━\n"
            f"{spy_e} **SPY:** ${spy_close:.2f} ({spy_wpct:+.2f}% this week)\n"
            f"{qqq_e} **QQQ:** ${qqq_close:.2f} ({qqq_wpct:+.2f}% this week)\n"
            f"😨 **VIX avg:** {avg_vix} — {vix_env}\n"
            f"📈 Best day: +{best_day:.2f}%  |  📉 Worst day: {worst_day:.2f}%\n"
            f"━━━━━━━━━━━━━━━━━━━━━━\n"
            f"**Friday Close Flow:**\n"
        )

        body   = format_full_report(sentiment, flows) if sentiment else ""
        footer = f"\n**🔭 Next Week:** {next_week}\nHave a great weekend! Back Monday at **9:00 AM ET** 🚀"

        _send(header + body + footer, username="Options Flow Bot 🐋")
        logger.info("Sentiment weekly report sent")
    except Exception as e:
        logger.exception(f"Sentiment weekly report error: {e}")


def run_flow_scan(tickers: list = None) -> None:
    """Main entry — scans all tickers, sends full smart money report."""
    if tickers is None:
        tickers = FLOW_TICKERS

    for ticker in tickers:
        try:
            logger.info(f"Running flow scan for {ticker}...")
            price_info = yf.Ticker(ticker).fast_info
            price = float(price_info.get("lastPrice", 0) or 0)

            # Fetch chain once, use for both sentiment + flow
            calls, puts = fetch_full_chain(ticker)
            if calls.empty:
                logger.warning(f"Empty chain for {ticker}")
                continue

            sentiment = analyze_sentiment(ticker, price)
            flows = fetch_unusual_flow(ticker, calls, puts) if sentiment else []

            if sentiment:
                message = format_full_report(sentiment, flows)
                sent = _send(message)
                logger.info(f"Flow report sent for {ticker}: {sent}")

        except Exception as e:
            logger.exception(f"Flow scan error for {ticker}: {e}")
