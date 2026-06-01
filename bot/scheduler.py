import json
import logging
import os
from datetime import datetime
from typing import Optional

import pytz
import yfinance as yf
from apscheduler.schedulers.blocking import BlockingScheduler
from apscheduler.triggers.interval import IntervalTrigger
from apscheduler.triggers.cron import CronTrigger

from bot.config import CONFIG
from bot.data_collector import fetch_market_snapshot
from bot.technical_analysis import compute_technical_signals
from bot.options_analyzer import analyze_options
from bot.macro_analyzer import analyze_macro
from bot.signal_generator import generate_signal, HOLD
from bot.trade_constructor import construct_trade
from bot.alert_formatter import format_discord_embed, format_whatsapp_message
from bot.alert_sender import send_all_alerts, send_discord_alert
from bot.trade_logger import init_db, log_signal, log_trade_entry
from bot.options_flow import run_flow_scan, run_sentiment_daily_report, run_sentiment_weekly_report

logger = logging.getLogger(__name__)
ET = pytz.timezone("America/New_York")

_current_position: Optional[str] = None

_daily_stats = {
    "scans": 0,
    "highest_confidence": 0,
    "trades_fired": 0,
    "open_price": None,
    "milestone_70_sent": False,
}

# ── Active paper trade (set when a real signal fires) ──
# {ticker, direction, strike, expiration, entry_price, contracts,
#  total_cost, entry_time, profit_target, stop_loss_price,
#  tp_alert_sent, sl_alert_sent}
_active_paper_trade: Optional[dict] = None

PAPER_TRADE_LOG = "paper_trades.json"


def _load_paper_trades() -> list:
    if os.path.exists(PAPER_TRADE_LOG):
        try:
            with open(PAPER_TRADE_LOG) as f:
                return json.load(f)
        except Exception:
            pass
    return []


def _save_paper_trade(record: dict) -> None:
    trades = _load_paper_trades()
    trades.append(record)
    with open(PAPER_TRADE_LOG, "w") as f:
        json.dump(trades, f, indent=2)


def is_market_open() -> bool:
    now = datetime.now(ET)
    if now.weekday() >= 5:
        return False
    open_time = now.replace(hour=CONFIG.market_open_hour, minute=CONFIG.market_open_minute, second=0, microsecond=0)
    close_time = now.replace(hour=CONFIG.market_close_hour, minute=CONFIG.market_close_minute, second=0, microsecond=0)
    return open_time <= now <= close_time


def _get_snapshot_and_signal():
    snapshot = fetch_market_snapshot()
    technical = compute_technical_signals(snapshot.spy_1h, snapshot.spy_1d, snapshot.spy_1wk, snapshot.spy_price)
    macro = analyze_macro(snapshot.vix_current, snapshot.vix_1d, snapshot.sector_etf_prices, snapshot.tlt_1d, snapshot.uup_1d)
    options = analyze_options(snapshot.options_calls, snapshot.options_puts, snapshot.spy_price, snapshot.available_expirations, snapshot.vix_1d)
    signal = generate_signal(technical, macro, options, threshold=CONFIG.confidence_threshold, current_position=_current_position, spy_price=snapshot.spy_price)
    return snapshot, technical, macro, options, signal


def _fetch_current_option_price(ticker: str, expiration: str, strike: float, option_type: str) -> Optional[float]:
    """Fetch the current mid price of a specific option from yfinance."""
    try:
        t = yf.Ticker(ticker)
        chain = t.option_chain(expiration)
        df = chain.calls if option_type == "call" else chain.puts
        row = df[df["strike"] == strike]
        if row.empty:
            return None
        bid = float(row.iloc[0].get("bid", 0) or 0)
        ask = float(row.iloc[0].get("ask", 0) or 0)
        if ask > 0:
            return round((bid + ask) / 2, 2)
        return float(row.iloc[0].get("lastPrice", 0) or 0)
    except Exception as e:
        logger.warning(f"Could not fetch option price: {e}")
        return None


# ─────────────────────────────────────────────
# STOP LOSS / TAKE PROFIT MONITOR — every 30 min
# ─────────────────────────────────────────────
def run_position_monitor() -> None:
    global _active_paper_trade

    if not is_market_open() or _active_paper_trade is None:
        return

    trade = _active_paper_trade
    current_price = _fetch_current_option_price(
        trade["ticker"], trade["expiration"],
        trade["strike"], trade["direction"]
    )

    if current_price is None:
        logger.warning("Position monitor: could not fetch current option price")
        return

    entry = trade["entry_price"]
    pct_change = ((current_price - entry) / entry) * 100
    contracts = trade["contracts"]
    total_cost = trade["total_cost"]
    current_value = round(current_price * contracts * 100, 2)
    pnl = round(current_value - total_cost, 2)
    option_type = "Call" if trade["direction"] == "call" else "Put"
    strike = trade["strike"]
    ticker = trade["ticker"]
    now_et = datetime.now(ET).strftime("%I:%M %p ET")

    logger.info(f"Position monitor: {ticker} ${strike} {option_type} | entry=${entry:.2f} current=${current_price:.2f} | {pct_change:+.1f}%")

    # ── TAKE PROFIT: up 50%+ ──
    if pct_change >= 50 and not trade.get("tp_alert_sent"):
        _active_paper_trade["tp_alert_sent"] = True
        send_discord_alert({
            "username": f"{ticker} Options Bot",
            "content": (
                f"🚨 **EXIT NOW — TAKE PROFIT!**\n"
                f"**{ticker} ${strike:.0f} {option_type}** is up **{pct_change:.0f}%** 🎉\n"
                f"Entry: **${entry:.2f}** → Now: **${current_price:.2f}**\n"
                f"💰 **P&L: +${pnl:,.0f}** on {contracts} contract(s)\n"
                f"📋 Go to Robinhood → find your {ticker} ${strike:.0f} {option_type} → **Sell to Close**\n"
                f"⏰ {now_et} — don't wait, lock in the profit!"
            )
        })
        # Log closed paper trade
        _active_paper_trade["exit_price"] = current_price
        _active_paper_trade["exit_time"] = now_et
        _active_paper_trade["exit_reason"] = "take_profit"
        _active_paper_trade["pnl"] = pnl
        _active_paper_trade["pnl_pct"] = round(pct_change, 1)
        _active_paper_trade["outcome"] = "win"
        _save_paper_trade(_active_paper_trade)
        _active_paper_trade = None
        logger.info("Take profit alert sent — paper trade closed")
        return

    # ── STOP LOSS: down 50%+ ──
    if pct_change <= -50 and not trade.get("sl_alert_sent"):
        _active_paper_trade["sl_alert_sent"] = True
        send_discord_alert({
            "username": f"{ticker} Options Bot",
            "content": (
                f"🛑 **STOP LOSS HIT — EXIT NOW**\n"
                f"**{ticker} ${strike:.0f} {option_type}** is down **{abs(pct_change):.0f}%** 📉\n"
                f"Entry: **${entry:.2f}** → Now: **${current_price:.2f}**\n"
                f"💸 **P&L: -${abs(pnl):,.0f}** on {contracts} contract(s)\n"
                f"📋 Go to Robinhood → find your {ticker} ${strike:.0f} {option_type} → **Sell to Close**\n"
                f"⏰ {now_et} — exit now to protect remaining capital!"
            )
        })
        _active_paper_trade["exit_price"] = current_price
        _active_paper_trade["exit_time"] = now_et
        _active_paper_trade["exit_reason"] = "stop_loss"
        _active_paper_trade["pnl"] = pnl
        _active_paper_trade["pnl_pct"] = round(pct_change, 1)
        _active_paper_trade["outcome"] = "loss"
        _save_paper_trade(_active_paper_trade)
        _active_paper_trade = None
        logger.info("Stop loss alert sent — paper trade closed")
        return

    # ── Regular update every 30 min ──
    direction_emoji = "📈" if pct_change >= 0 else "📉"
    send_discord_alert({
        "username": f"{ticker} Options Bot",
        "content": (
            f"{direction_emoji} **Open Trade Update** — {now_et}\n"
            f"**{ticker} ${strike:.0f} {option_type}** | Entry: ${entry:.2f} → Now: **${current_price:.2f}** ({pct_change:+.1f}%)\n"
            f"P&L: **{'+'if pnl>=0 else ''}${pnl:,.0f}**  |  "
            f"Take profit at: **${trade['profit_target']:.2f}** (+50%)  |  "
            f"Stop loss at: **${trade['stop_loss_price']:.2f}** (-50%)"
        )
    })


# ─────────────────────────────────────────────
# PRE-MARKET — 9:00 AM ET
# ─────────────────────────────────────────────
def run_premarket_alert() -> None:
    if datetime.now(ET).weekday() >= 5:
        return
    try:
        es = yf.Ticker("ES=F")
        spy = yf.Ticker(CONFIG.spy)
        vix = yf.Ticker("^VIX")
        es_price = es.fast_info.get("lastPrice", 0) or 0
        vix_price = vix.fast_info.get("lastPrice", 0) or 0
        spy_prev = spy.fast_info.get("previousClose", 0) or 0
        spy_last = spy.fast_info.get("lastPrice", spy_prev) or spy_prev
        change = spy_last - spy_prev
        change_pct = (change / spy_prev * 100) if spy_prev else 0
        direction = "higher ☀️" if change >= 0 else "lower 🌧️"
        vix_note = "calm" if vix_price < 15 else ("normal" if vix_price < 25 else "elevated ⚠️")
        send_discord_alert({
            "username": f"{CONFIG.spy} Options Bot",
            "content": (
                f"🌅 **Pre-Market Report — 9:00 AM ET**\n"
                f"**{CONFIG.spy}** is pointing {direction} — ${spy_last:.2f} ({change_pct:+.2f}%)\n"
                f"**S&P Futures (ES):** ${es_price:.2f}  |  **VIX:** {vix_price:.2f} ({vix_note})\n"
                f"{'📈 Bullish open expected — calls may be in play' if change >= 0 else '📉 Bearish open expected — puts may be in play'}\n"
                f"⏰ Market opens in **30 minutes**"
            )
        })
        logger.info("Pre-market alert sent")
    except Exception as e:
        logger.exception(f"Pre-market error: {e}")


# ─────────────────────────────────────────────
# OPENING BELL — 9:30 AM ET
# ─────────────────────────────────────────────
def run_opening_bell() -> None:
    global _daily_stats
    if datetime.now(ET).weekday() >= 5:
        return
    _daily_stats = {"scans": 0, "highest_confidence": 0, "trades_fired": 0, "open_price": None, "milestone_70_sent": False}
    try:
        snapshot = fetch_market_snapshot()
        _daily_stats["open_price"] = snapshot.spy_price
        send_discord_alert({
            "username": f"{CONFIG.spy} Options Bot",
            "content": (
                f"🔔 **Market is open! Let's get this money!** 💰\n"
                f"**{CONFIG.spy}:** ${snapshot.spy_price:.2f}  |  **VIX:** {snapshot.vix_current:.2f}\n"
                f"Bots are live and scanning every {CONFIG.scan_interval_minutes} minutes. "
                f"Signal fires at **75%+ confidence** 🚀"
            )
        })
        logger.info("Opening bell sent")
    except Exception as e:
        logger.exception(f"Opening bell error: {e}")


# ─────────────────────────────────────────────
# MAIN SCAN CYCLE — every 20 min
# ─────────────────────────────────────────────
def run_scan_cycle() -> None:
    global _current_position, _daily_stats, _active_paper_trade

    if not is_market_open():
        logger.debug("Market closed — skipping scan")
        return

    logger.info("=" * 50)
    logger.info(f"Starting {CONFIG.spy} scan cycle")

    try:
        snapshot, technical, macro, options, signal = _get_snapshot_and_signal()

        _daily_stats["scans"] += 1
        if signal.confidence > _daily_stats["highest_confidence"]:
            _daily_stats["highest_confidence"] = signal.confidence
        if _daily_stats["open_price"] is None:
            _daily_stats["open_price"] = snapshot.spy_price

        logger.info(
            f"Signal: {signal.signal} | Confidence: {signal.confidence}% | "
            f"Tech: {technical.technical_bias} | Macro: {macro.macro_bias}"
        )

        # ── Confidence milestone at 70% ──
        if signal.confidence >= 70 and not _daily_stats["milestone_70_sent"]:
            _daily_stats["milestone_70_sent"] = True
            filled = int(signal.confidence / 10)
            bar = "🟩" * filled + "⬜" * (10 - filled)
            send_discord_alert({
                "username": f"{CONFIG.spy} Options Bot",
                "content": (
                    f"⚠️ **Getting close! Confidence at {signal.confidence}%** — just {75 - signal.confidence} point(s) from firing!\n"
                    f"**{CONFIG.spy}:** ${snapshot.spy_price:.2f}  |  **VIX:** {snapshot.vix_current:.2f}\n"
                    f"{bar}\n"
                    f"📌 Stay ready — next scan in {CONFIG.scan_interval_minutes} minutes could send a trade alert! 👀"
                )
            })
        if signal.confidence < 68:
            _daily_stats["milestone_70_sent"] = False

        alert_sent = False
        trade = None

        if signal.signal != HOLD:
            trade = construct_trade(
                signal_direction=signal.direction,
                spy_price=snapshot.spy_price,
                calls=options.best_calls,
                puts=options.best_puts,
                expirations=snapshot.available_expirations,
                account_size=CONFIG.account_size,
                risk_pct=CONFIG.risk_pct,
                target_dte_min=CONFIG.target_dte_min,
                target_dte_max=CONFIG.target_dte_max,
                signal_timestamp=signal.timestamp,
                max_spread_width=CONFIG.max_spread_width,
            )
            discord_payload = format_discord_embed(signal, trade, snapshot.spy_price)
            whatsapp_msg = format_whatsapp_message(signal, trade, snapshot.spy_price)
            results = send_all_alerts(discord_payload, whatsapp_msg)
            alert_sent = results.get("discord", False)

            if alert_sent and trade:
                _daily_stats["trades_fired"] += 1
                logger.info(f"Trade alert sent — {signal.signal} conf={signal.confidence}%")

                # ── Start paper trade tracker ──
                _active_paper_trade = {
                    "ticker": CONFIG.spy,
                    "direction": trade.direction,
                    "strike": trade.strike,
                    "expiration": trade.expiration,
                    "entry_price": trade.ask_price,
                    "contracts": trade.contracts,
                    "total_cost": trade.total_cost,
                    "profit_target": trade.ask_price * 1.5,
                    "stop_loss_price": trade.ask_price * 0.5,
                    "entry_time": signal.timestamp,
                    "confidence": signal.confidence,
                    "tp_alert_sent": False,
                    "sl_alert_sent": False,
                }
                send_discord_alert({
                    "username": f"{CONFIG.spy} Options Bot",
                    "content": (
                        f"📋 **Paper Trade Started** — tracking this position automatically\n"
                        f"**{CONFIG.spy} ${trade.strike:.0f} {trade.option_type}** | Entry: **${trade.ask_price:.2f}**\n"
                        f"✅ Take profit alert at: **${trade.ask_price * 1.5:.2f}** (+50%)\n"
                        f"🛑 Stop loss alert at: **${trade.ask_price * 0.5:.2f}** (-50%)\n"
                        f"I'll update you every 30 min until you exit 👀"
                    )
                })

                if signal.signal in ("BUY_CALL", "BUY_CALL_SPREAD"):
                    _current_position = "long_call"
                elif signal.signal in ("BUY_PUT", "BUY_PUT_SPREAD"):
                    _current_position = "long_put"
                elif signal.signal in ("EXIT_LONG", "EXIT_SHORT"):
                    _current_position = None

        signal_id = log_signal(signal, snapshot.spy_price, snapshot.vix_current, options.iv_rank, macro.breadth_pct, alert_sent=alert_sent)
        if trade and alert_sent:
            log_trade_entry(signal_id, trade)

    except Exception as e:
        logger.exception(f"Scan cycle error: {e}")


# ─────────────────────────────────────────────
# HOURLY TREND UPDATE
# ─────────────────────────────────────────────
def run_trend_update() -> None:
    if not is_market_open():
        return
    try:
        snapshot, technical, macro, options, signal = _get_snapshot_and_signal()
        tech = technical.technical_bias
        mac = macro.macro_bias
        conf = signal.confidence
        price = snapshot.spy_price
        vix = snapshot.vix_current

        if tech == "bullish" and mac == "bullish":
            emoji, trend, action = "📈", "Market is rising", "conditions improving — watching for buy signal"
        elif tech == "bearish" or mac == "bearish":
            emoji, trend, action = "📉", "Market is dropping", "don't buy now — waiting for reversal"
        elif conf >= 65:
            emoji, trend, action = "🟡", "Market is mixed", "close to a signal — stay ready"
        else:
            emoji, trend, action = "⏳", "Market is flat", "no clear direction yet — holding off"

        filled = int(conf / 10)
        bar = "🟩" * filled + "⬜" * (10 - filled)
        now_et = datetime.now(ET).strftime("%I:%M %p ET")

        send_discord_alert({
            "username": f"{CONFIG.spy} Options Bot",
            "content": (
                f"{emoji} **{trend}** — {now_et}\n"
                f"**{CONFIG.spy}:** ${price:.2f}  |  **VIX:** {vix:.2f}  |  **Confidence:** {conf}%\n"
                f"{bar}\n"
                f"📌 {action.capitalize()}"
            )
        })
        logger.info(f"Trend update sent: {conf}%")
    except Exception as e:
        logger.exception(f"Trend update error: {e}")


# ─────────────────────────────────────────────
# CLOSING BELL + DAILY SUMMARY — 4:00 PM ET
# ─────────────────────────────────────────────
def run_closing_bell() -> None:
    global _active_paper_trade
    if datetime.now(ET).weekday() >= 5:
        return
    try:
        ticker = yf.Ticker(CONFIG.spy)
        close_price = ticker.fast_info.get("lastPrice", 0) or 0
        open_price = _daily_stats.get("open_price") or close_price
        day_change = close_price - open_price
        day_pct = (day_change / open_price * 100) if open_price else 0
        direction = "▲" if day_change >= 0 else "▼"
        color_emoji = "🟢" if day_change >= 0 else "🔴"
        trades = _daily_stats["trades_fired"]
        scans = _daily_stats["scans"]
        high_conf = _daily_stats["highest_confidence"]
        trade_line = f"✅ {trades} trade alert(s) fired today!" if trades > 0 else "⏳ No trade alerts fired today"

        # Paper trade summary
        paper_trades = _load_paper_trades()
        today = datetime.now(ET).strftime("%Y-%m-%d")
        todays_trades = [t for t in paper_trades if t.get("entry_time", "").startswith(today)]
        wins = sum(1 for t in todays_trades if t.get("outcome") == "win")
        losses = sum(1 for t in todays_trades if t.get("outcome") == "loss")
        total_pnl = sum(t.get("pnl", 0) for t in todays_trades)
        paper_line = f"📊 Paper trades: {wins}W / {losses}L | P&L: {'+'if total_pnl>=0 else ''}${total_pnl:,.0f}" if todays_trades else "📊 No paper trades closed today"

        # Close any open paper trade at end of day
        if _active_paper_trade:
            current = _fetch_current_option_price(
                _active_paper_trade["ticker"], _active_paper_trade["expiration"],
                _active_paper_trade["strike"], _active_paper_trade["direction"]
            )
            if current:
                entry = _active_paper_trade["entry_price"]
                pct = ((current - entry) / entry) * 100
                eod_pnl = round((current - entry) * _active_paper_trade["contracts"] * 100, 2)
                send_discord_alert({
                    "username": f"{CONFIG.spy} Options Bot",
                    "content": (
                        f"🌙 **End of Day — Open Trade Status**\n"
                        f"**{CONFIG.spy} ${_active_paper_trade['strike']:.0f}** still open | "
                        f"${entry:.2f} → ${current:.2f} ({pct:+.1f}%)\n"
                        f"Unrealized P&L: **{'+'if eod_pnl>=0 else ''}${eod_pnl:,.0f}**\n"
                        f"Position carries overnight — monitor tomorrow at open."
                    )
                })

        send_discord_alert({
            "username": f"{CONFIG.spy} Options Bot",
            "content": (
                f"🔔 **Market Closed — Daily Summary**\n"
                f"━━━━━━━━━━━━━━━━━━━━━━\n"
                f"{color_emoji} **{CONFIG.spy}:** ${close_price:.2f} {direction} {abs(day_pct):.2f}% today\n"
                f"🔍 Scans ran: **{scans}**  |  Highest confidence: **{high_conf}%**\n"
                f"{trade_line}\n"
                f"{paper_line}\n"
                f"━━━━━━━━━━━━━━━━━━━━━━\n"
                f"See you tomorrow at **9:00 AM ET**! 👋"
            )
        })
        logger.info("Closing bell sent")
    except Exception as e:
        logger.exception(f"Closing bell error: {e}")


# ─────────────────────────────────────────────
# ECONOMIC CALENDAR WARNING — 8:00 AM daily
# ─────────────────────────────────────────────
def run_economic_calendar_check() -> None:
    """Checks for high-impact economic events today or tomorrow and warns users."""
    if datetime.now(ET).weekday() >= 5:
        return
    try:
        import requests as req
        from datetime import timedelta

        # ForexFactory free JSON calendar feed
        url = "https://nfs.faireconomy.media/ff_calendar_thisweek.json"
        resp = req.get(url, timeout=10)
        if resp.status_code != 200:
            logger.warning(f"Economic calendar fetch failed: {resp.status_code}")
            return

        events = resp.json()
        now_et = datetime.now(ET)
        today_str = now_et.strftime("%Y-%m-%d")
        tomorrow_str = (now_et + timedelta(days=1)).strftime("%Y-%m-%d")

        # High-impact keywords to watch for
        HIGH_IMPACT_KEYWORDS = [
            "CPI", "PPI", "FOMC", "Fed", "Federal Reserve", "Interest Rate",
            "NFP", "Non-Farm", "Jobs", "Unemployment", "GDP", "Inflation",
            "Retail Sales", "PCE", "Powell"
        ]

        warnings_today = []
        warnings_tomorrow = []

        for event in events:
            if event.get("impact", "").lower() != "high":
                continue
            title = event.get("title", "")
            date_str = event.get("date", "")[:10]
            time_str = event.get("date", "")[11:16] if len(event.get("date", "")) > 10 else "TBD"
            currency = event.get("currency", "")

            # Only care about USD events
            if currency != "USD":
                continue

            # Check if it matches high impact keywords
            if any(kw.lower() in title.lower() for kw in HIGH_IMPACT_KEYWORDS):
                if date_str == today_str:
                    warnings_today.append(f"• **{title}** at {time_str} ET")
                elif date_str == tomorrow_str:
                    warnings_tomorrow.append(f"• **{title}** tomorrow")

        if warnings_today:
            send_discord_alert({
                "username": f"{CONFIG.spy} Options Bot",
                "content": (
                    f"🚨 **HIGH-IMPACT EVENT TODAY — BE CAREFUL**\n"
                    f"━━━━━━━━━━━━━━━━━━━━━━\n"
                    + "\n".join(warnings_today) + "\n"
                    f"━━━━━━━━━━━━━━━━━━━━━━\n"
                    f"⚠️ **Recommendation:** Avoid opening new positions today.\n"
                    f"These events cause massive price swings — options can lose 80%+ in minutes.\n"
                    f"📌 Wait until AFTER the event for a clearer signal."
                )
            })
            logger.info(f"Economic warning sent: {len(warnings_today)} events today")

        elif warnings_tomorrow:
            send_discord_alert({
                "username": f"{CONFIG.spy} Options Bot",
                "content": (
                    f"📅 **High-Impact Event Tomorrow — Plan Ahead**\n"
                    + "\n".join(warnings_tomorrow) + "\n"
                    f"⚠️ Consider closing any open positions before tomorrow.\n"
                    f"📌 Wait for the data to drop before entering new trades."
                )
            })
            logger.info(f"Economic warning sent: {len(warnings_tomorrow)} events tomorrow")

        else:
            logger.info("Economic calendar: no high-impact events today or tomorrow")

    except Exception as e:
        logger.exception(f"Economic calendar error: {e}")


# ─────────────────────────────────────────────
# WEEKLY PERFORMANCE REPORT — Friday 4:00 PM
# ─────────────────────────────────────────────
def _build_money_report(period: str, trades: list, now_et: datetime) -> str:
    """Shared helper that builds a P&L report string for any period."""
    closed = [t for t in trades if t.get("outcome") in ("win", "loss", "breakeven")]
    wins   = [t for t in closed if t.get("outcome") == "win"]
    losses = [t for t in closed if t.get("outcome") == "loss"]
    open_t = [t for t in trades if t.get("outcome") == "open"]
    total_pnl = sum(t.get("pnl", 0) for t in closed)
    win_rate  = round(len(wins) / len(closed) * 100) if closed else 0
    avg_win   = round(sum(t.get("pnl",0) for t in wins)   / len(wins),   2) if wins   else 0
    avg_loss  = round(sum(t.get("pnl",0) for t in losses) / len(losses), 2) if losses else 0

    # Ticker performance
    try:
        t = yf.Ticker(CONFIG.spy)
        hist = t.history(period="5d" if period == "week" else "1d")
        if len(hist) >= 2:
            p_open  = float(hist["Close"].iloc[0])
            p_close = float(hist["Close"].iloc[-1])
            p_pct   = (p_close - p_open) / p_open * 100
            ticker_line = f"**{CONFIG.spy}:** ${p_close:.2f} ({'+'if p_pct>=0 else ''}{p_pct:.2f}% this {period})"
        else:
            ticker_line = f"**{CONFIG.spy}:** N/A"
    except Exception:
        ticker_line = f"**{CONFIG.spy}:** N/A"

    trade_lines = ""
    for t in closed[-5:]:
        emoji = "✅" if t.get("outcome") == "win" else "❌"
        pnl   = t.get("pnl", 0)
        trade_lines += f"{emoji} ${t.get('strike',0):.0f} {t.get('direction','').title()} | {'+'if pnl>=0 else ''}${pnl:,.0f} ({t.get('pnl_pct',0):+.0f}%)\n"
    if not trade_lines:
        trade_lines = "No closed trades this period\n"

    pnl_emoji = "💰" if total_pnl >= 0 else "📉"
    account   = CONFIG.account_size
    pnl_pct   = round(total_pnl / account * 100, 1) if account else 0

    return (
        f"{ticker_line}\n"
        f"━━━━━━━━━━━━━━━━━━━━━━\n"
        f"📊 **Trades:** {len(trades)} fired | {len(closed)} closed | {len(open_t)} open\n"
        f"✅ **Wins:** {len(wins)}  ❌ **Losses:** {len(losses)}  🎯 **Win Rate:** {win_rate}%\n"
        f"💵 **Avg Win:** +${avg_win:,.0f}  |  **Avg Loss:** -${abs(avg_loss):,.0f}\n"
        f"{pnl_emoji} **Total P&L:** {'+'if total_pnl>=0 else ''}${total_pnl:,.0f} ({pnl_pct:+.1f}% of account)\n"
        f"💼 **Account:** ${account:,.0f}\n"
        f"━━━━━━━━━━━━━━━━━━━━━━\n"
        f"**Trade breakdown:**\n{trade_lines}"
    )


def run_weekly_report() -> None:
    """Sends a full week P&L recap every Friday at market close."""
    try:
        from datetime import timedelta
        now_et = datetime.now(ET)
        week_start_str = (now_et - timedelta(days=now_et.weekday())).strftime("%Y-%m-%d")
        all_trades  = _load_paper_trades()
        week_trades = [t for t in all_trades if t.get("entry_time", "") >= week_start_str]

        body = _build_money_report("week", week_trades, now_et)
        send_discord_alert({
            "username": f"{CONFIG.spy} Options Bot",
            "content": (
                f"📋 **{CONFIG.spy} Weekly Money Report — {now_et.strftime('%b %d, %Y')}**\n"
                f"{body}"
                f"🔍 Scans this week: **{_daily_stats['scans']}** (today only)\n"
                f"━━━━━━━━━━━━━━━━━━━━━━\n"
                f"Have a great weekend! See you Monday at **9:00 AM ET** 🚀"
            )
        })
        logger.info("Weekly money report sent")
    except Exception as e:
        logger.exception(f"Weekly report error: {e}")


def run_daily_money_report() -> None:
    """Sends a daily P&L report at market close (Mon-Thu). Friday uses weekly report."""
    if datetime.now(ET).weekday() >= 4:  # Skip Fri (weekly handles it), Sat, Sun
        return
    try:
        now_et   = datetime.now(ET)
        today    = now_et.strftime("%Y-%m-%d")
        all_trades   = _load_paper_trades()
        today_trades = [t for t in all_trades if t.get("entry_time", "").startswith(today)]

        body = _build_money_report("day", today_trades, now_et)
        send_discord_alert({
            "username": f"{CONFIG.spy} Options Bot",
            "content": (
                f"💵 **{CONFIG.spy} Daily Money Report — {now_et.strftime('%b %d, %Y')}**\n"
                f"{body}"
                f"🔍 Scans today: **{_daily_stats['scans']}** | "
                f"Highest confidence: **{_daily_stats['highest_confidence']}%**\n"
                f"━━━━━━━━━━━━━━━━━━━━━━\n"
                f"See you tomorrow at **9:00 AM ET**! 👋"
            )
        })
        logger.info("Daily money report sent")
    except Exception as e:
        logger.exception(f"Daily money report error: {e}")


# ─────────────────────────────────────────────
# SCHEDULER SETUP
# ─────────────────────────────────────────────
def start_scheduler() -> None:
    init_db()
    scheduler = BlockingScheduler(timezone=ET)

    scheduler.add_job(run_scan_cycle, trigger=IntervalTrigger(minutes=CONFIG.scan_interval_minutes),
                      id="scan", max_instances=1, coalesce=True, misfire_grace_time=60)

    scheduler.add_job(run_position_monitor, trigger=IntervalTrigger(minutes=30),
                      id="position_monitor", max_instances=1, coalesce=True, misfire_grace_time=60)

    scheduler.add_job(run_trend_update, trigger=IntervalTrigger(hours=1),
                      id="trend_update", max_instances=1, coalesce=True, misfire_grace_time=120)

    # QQQ-only alerts — pre-market, opening bell, closing bell go to QQQ channel
    scheduler.add_job(run_premarket_alert, trigger=CronTrigger(hour=9, minute=1, day_of_week="mon-fri", timezone=ET),
                      id="premarket", max_instances=1)

    scheduler.add_job(run_opening_bell, trigger=CronTrigger(hour=9, minute=31, day_of_week="mon-fri", timezone=ET),
                      id="opening_bell", max_instances=1)

    scheduler.add_job(run_closing_bell, trigger=CronTrigger(hour=16, minute=3, day_of_week="mon-fri", timezone=ET),
                      id="closing_bell", max_instances=1)

    # QQQ weekly + daily money reports — go to QQQ channel only
    scheduler.add_job(run_weekly_report, trigger=CronTrigger(hour=16, minute=5, day_of_week="fri", timezone=ET),
                      id="weekly_report", max_instances=1)

    scheduler.add_job(run_daily_money_report, trigger=CronTrigger(hour=16, minute=4, day_of_week="mon-thu", timezone=ET),
                      id="daily_money_report", max_instances=1)

    # NOTE: Economic calendar, flow scan, and sentiment reports
    # are handled by SPY bot only — no duplicates here

    logger.info(f"Scheduler started — scanning every {CONFIG.scan_interval_minutes} minutes")
    run_scan_cycle()
    scheduler.start()
