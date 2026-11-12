import json
import logging
import os
import time
from datetime import datetime, date
from pathlib import Path
from typing import Optional

import pytz
import yfinance as yf
from apscheduler.schedulers.blocking import BlockingScheduler
from apscheduler.triggers.interval import IntervalTrigger
from apscheduler.triggers.cron import CronTrigger

from bot.config import CONFIG, DATA_DIR
from bot.data_collector import fetch_market_snapshot
from bot.technical_analysis import compute_technical_signals
from bot.options_analyzer import analyze_options
from bot.macro_analyzer import analyze_macro
from bot.signal_generator import generate_signal, reset_cooldown, HOLD, EXIT_LONG, EXIT_SHORT
from bot.trade_constructor import construct_trade
from bot.alert_formatter import format_discord_embed, format_whatsapp_message
from bot.alert_sender import send_all_alerts, send_discord_alert
from bot.trade_logger import init_db, log_signal, log_trade_entry, update_trade_exit
# Flow/sentiment handled by SPY bot only — not imported here

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
# {ticker, direction, strike, expiration, entry_price, contracts, total_cost,
#  entry_time, profit_target, stop_loss_price, confidence, db_trade_id, ...}
# Persisted to disk so a restart / sleep / crash never forgets an open position.
_active_paper_trade: Optional[dict] = None

PAPER_TRADE_LOG = str(DATA_DIR / "paper_trades.json")      # closed trades (the win/loss ledger)
ACTIVE_TRADE_FILE = str(DATA_DIR / "active_trade.json")    # the one open trade, if any

TP_PCT = 50.0    # take profit at +50% on the option premium
SL_PCT = -50.0   # stop loss at -50%

_confidence_history: list = []   # last 5 scan confidences, shown in trend updates
_monitor_ticks = 0   # position monitor runs every 5 min; user-facing update every 6th tick (30 min)


def _load_paper_trades() -> list:
    if os.path.exists(PAPER_TRADE_LOG):
        try:
            with open(PAPER_TRADE_LOG) as f:
                return json.load(f)
        except Exception:
            logger.exception("Could not read paper trade ledger")
    return []


def _save_paper_trade(record: dict) -> None:
    trades = _load_paper_trades()
    trades.append(record)
    tmp = PAPER_TRADE_LOG + ".tmp"
    with open(tmp, "w") as f:
        json.dump(trades, f, indent=2)
    os.replace(tmp, PAPER_TRADE_LOG)


def _persist_active_trade() -> None:
    if _active_paper_trade is None:
        if os.path.exists(ACTIVE_TRADE_FILE):
            os.remove(ACTIVE_TRADE_FILE)
        return
    with open(ACTIVE_TRADE_FILE, "w") as f:
        json.dump(_active_paper_trade, f, indent=2)


def _restore_active_trade() -> None:
    global _active_paper_trade, _current_position
    if not os.path.exists(ACTIVE_TRADE_FILE):
        return
    try:
        with open(ACTIVE_TRADE_FILE) as f:
            _active_paper_trade = json.load(f)
        _current_position = _position_label(_active_paper_trade["direction"])
        t = _active_paper_trade
        logger.info(f"Restored open paper trade: {t['ticker']} ${t['strike']:.0f} {t['direction']} "
                    f"exp {t['expiration']} entry ${t['entry_price']:.2f}")
    except Exception:
        logger.exception("Could not restore active trade")
        _active_paper_trade = None


def _position_label(direction: str) -> str:
    # names must match what signal_generator's EXIT logic looks for
    return "long_call_spread" if direction == "call" else "long_put_spread"


def _close_paper_trade(exit_price: float, reason: str) -> dict:
    """Single place where a trade closes: computes P&L, win/loss, writes ledger + DB, clears state."""
    global _active_paper_trade, _current_position
    t = _active_paper_trade
    entry = t["entry_price"]
    contracts = t["contracts"]
    pnl = round((exit_price - entry) * contracts * 100, 2)      # long option: exit - entry
    pnl_pct = round((exit_price - entry) / entry * 100, 1) if entry else 0.0
    outcome = "win" if pnl > 0 else ("loss" if pnl < 0 else "breakeven")
    now = datetime.now(ET)
    t.update({
        "exit_price": exit_price,
        "exit_time": now.strftime("%Y-%m-%d %I:%M %p ET"),
        "exit_date": now.strftime("%Y-%m-%d"),
        "exit_reason": reason,
        "pnl": pnl,
        "pnl_pct": pnl_pct,
        "outcome": outcome,
    })
    _save_paper_trade(t)
    if t.get("db_trade_id"):
        try:
            update_trade_exit(t["db_trade_id"], exit_price, reason)
        except Exception:
            logger.exception("Could not update trade exit in DB")
    closed = t
    _active_paper_trade = None
    _current_position = None
    _persist_active_trade()
    logger.info(f"Paper trade closed: {reason} | {outcome.upper()} | ${pnl:+,.2f} ({pnl_pct:+.1f}%)")
    return closed


def is_market_open() -> bool:
    now = datetime.now(ET)
    if now.weekday() >= 5:
        return False
    open_time = now.replace(hour=CONFIG.market_open_hour, minute=CONFIG.market_open_minute, second=0, microsecond=0)
    close_time = now.replace(hour=CONFIG.market_close_hour, minute=CONFIG.market_close_minute, second=0, microsecond=0)
    return open_time <= now <= close_time


def _get_snapshot_and_signal(dry_run: bool = False):
    snapshot = fetch_market_snapshot()
    technical = compute_technical_signals(snapshot.spy_1h, snapshot.spy_1d, snapshot.spy_1wk, snapshot.spy_price)
    macro = analyze_macro(snapshot.vix_current, snapshot.vix_1d, snapshot.sector_etf_prices, snapshot.tlt_1d, snapshot.uup_1d)
    options = analyze_options(snapshot.options_calls, snapshot.options_puts, snapshot.spy_price, snapshot.available_expirations, snapshot.vix_1d)
    signal = generate_signal(technical, macro, options, threshold=CONFIG.confidence_threshold, current_position=_current_position, spy_price=snapshot.spy_price, dry_run=dry_run)
    return snapshot, technical, macro, options, signal


def _refresh_trade_prices(trade) -> None:
    """Re-fetch live bid/ask for the selected contract and update the trade in place."""
    try:
        t = yf.Ticker(CONFIG.spy)
        chain = t.option_chain(trade.expiration)
        df = chain.calls if trade.direction == "call" else chain.puts
        row = df[df["strike"] == trade.strike]
        if row.empty:
            logger.warning("Price refresh: strike not found in chain")
            return
        ask = float(row.iloc[0].get("ask", 0) or 0)
        bid = float(row.iloc[0].get("bid", 0) or 0)
        if ask <= 0:
            logger.warning("Price refresh: ask is 0, keeping original prices")
            return
        trade.ask_price = round(ask, 2)
        trade.bid_price = round(bid, 2)
        trade.mid_price = round((ask + bid) / 2, 2)
        trade.cost_per_contract = round(ask * 100, 2)
        trade.total_cost = round(trade.contracts * trade.cost_per_contract, 2)
        trade.breakeven = round(trade.strike + ask, 2) if trade.direction == "call" else round(trade.strike - ask, 2)
        trade.profit_target = round(trade.breakeven + ask * 0.5, 2) if trade.direction == "call" else round(trade.breakeven - ask * 0.5, 2)
        trade.stop_loss_price = round(ask * 0.5, 2)
        trade.risk_pct_used = round(trade.total_cost / trade.account_size * 100, 2)
        logger.info(f"Price refresh: {CONFIG.spy} ${trade.strike:.0f} {trade.direction} → ask=${ask:.2f} bid=${bid:.2f}")
    except Exception as e:
        logger.warning(f"Price refresh failed (using snapshot prices): {e}")


def _fetch_current_option_price(ticker: str, expiration: str, strike: float, option_type: str) -> Optional[float]:
    """Current sellable price of an option: the bid, else bid/ask mid, else last."""
    try:
        t = yf.Ticker(ticker)
        chain = t.option_chain(expiration)
        df = chain.calls if option_type == "call" else chain.puts
        row = df[df["strike"] == strike]
        if row.empty:
            return None
        bid = float(row.iloc[0].get("bid", 0) or 0)
        ask = float(row.iloc[0].get("ask", 0) or 0)
        if bid > 0:                       # what you'd actually receive selling to close
            return round(bid, 2)
        if ask > 0:
            return round((bid + ask) / 2, 2)
        return float(row.iloc[0].get("lastPrice", 0) or 0)
    except Exception as e:
        logger.warning(f"Could not fetch option price: {e}")
        return None


# ─────────────────────────────────────────────
# STOP LOSS / TAKE PROFIT MONITOR — every 5 min
# ─────────────────────────────────────────────
def _exit_alert(kind: str, trade: dict, current: float) -> None:
    ticker, strike = trade["ticker"], trade["strike"]
    option_type = "Call" if trade["direction"] == "call" else "Put"
    pnl, pct = trade["pnl"], trade["pnl_pct"]
    now_et = datetime.now(ET).strftime("%I:%M %p ET")
    head = {
        "take_profit": f"🚨 **EXIT NOW — TAKE PROFIT!**\n**{ticker} ${strike:.0f} {option_type}** is up **{pct:.0f}%** 🎉",
        "stop_loss":   f"🛑 **STOP LOSS HIT — EXIT NOW**\n**{ticker} ${strike:.0f} {option_type}** is down **{abs(pct):.0f}%** 📉",
        "signal_exit": f"🔄 **EXIT SIGNAL — conditions reversed**\n**{ticker} ${strike:.0f} {option_type}** ({pct:+.0f}%)",
        "expiration":  f"⌛ **OPTION EXPIRING — CLOSE IT**\n**{ticker} ${strike:.0f} {option_type}** ({pct:+.0f}%)",
    }[kind]
    pnl_line = f"+${pnl:,.0f}" if pnl >= 0 else f"-${abs(pnl):,.0f}"
    send_discord_alert({
        "username": f"{ticker} Options Bot",
        "content": (
            f"{head}\n"
            f"Entry: **${trade['entry_price']:.2f}** → Now: **${current:.2f}**\n"
            f"💰 **P&L: {pnl_line}** on {trade['contracts']} contract(s) — **{trade['outcome'].upper()}**\n"
            f"📋 Go to Robinhood → find your {ticker} ${strike:.0f} {option_type} → **Sell to Close**\n"
            f"⏰ {now_et}"
        )
    })


def run_position_monitor() -> None:
    global _monitor_ticks

    if not is_market_open() or _active_paper_trade is None:
        return

    trade = _active_paper_trade
    current_price = _fetch_current_option_price(
        trade["ticker"], trade["expiration"], trade["strike"], trade["direction"]
    )
    if current_price is None or current_price <= 0:
        logger.warning("Position monitor: could not fetch a valid current option price")
        return

    entry = trade["entry_price"]
    pct_change = ((current_price - entry) / entry) * 100
    contracts = trade["contracts"]
    pnl = round((current_price - entry) * contracts * 100, 2)
    option_type = "Call" if trade["direction"] == "call" else "Put"
    strike, ticker = trade["strike"], trade["ticker"]
    now_et = datetime.now(ET).strftime("%I:%M %p ET")

    logger.info(f"Position monitor: {ticker} ${strike} {option_type} | entry=${entry:.2f} "
                f"current=${current_price:.2f} | {pct_change:+.1f}% | P&L ${pnl:+,.2f}")

    reason = None
    if pct_change >= TP_PCT:
        reason = "take_profit"
    elif pct_change <= SL_PCT:
        reason = "stop_loss"
    elif date.today().isoformat() >= trade["expiration"]:
        reason = "expiration"

    if reason:
        closed = _close_paper_trade(current_price, reason)
        _exit_alert(reason, closed, current_price)
        return

    # Regular status update every 6th check (30 min) — checks themselves run every 5 min
    _monitor_ticks += 1
    if _monitor_ticks % 6:
        return
    send_discord_alert({
        "username": f"{ticker} Options Bot",
        "content": (
            f"{'📈' if pct_change >= 0 else '📉'} **Open Trade Update** — {now_et}\n"
            f"**{ticker} ${strike:.0f} {option_type}** | Entry: ${entry:.2f} → Now: **${current_price:.2f}** ({pct_change:+.1f}%)\n"
            f"P&L: **{'+' if pnl >= 0 else '-'}${abs(pnl):,.0f}**  |  "
            f"Take profit at: **${entry * (1 + TP_PCT/100):.2f}** (+50%)  |  "
            f"Stop loss at: **${entry * (1 + SL_PCT/100):.2f}** (-50%)"
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
    _confidence_history.clear()
    try:
        snapshot = fetch_market_snapshot()
        _daily_stats["open_price"] = snapshot.spy_price
        send_discord_alert({
            "username": f"{CONFIG.spy} Options Bot",
            "content": (
                f"🔔 **Market is open! Let's get this money!** 💰\n"
                f"**{CONFIG.spy}:** ${snapshot.spy_price:.2f}  |  **VIX:** {snapshot.vix_current:.2f}\n"
                f"Bots are live and scanning every {CONFIG.scan_interval_minutes} minutes. "
                f"Signal fires at **{CONFIG.confidence_threshold}%+ confidence** 🚀"
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
        _confidence_history.append(signal.confidence)
        del _confidence_history[:-5]
        if _daily_stats["open_price"] is None:
            _daily_stats["open_price"] = snapshot.spy_price

        logger.info(
            f"Signal: {signal.signal} | Confidence: {signal.confidence}% | "
            f"Tech: {technical.technical_bias} | Macro: {macro.macro_bias}"
        )

        # ── Confidence milestone at 70% ──
        near = CONFIG.confidence_threshold - 5
        if near <= signal.confidence < CONFIG.confidence_threshold and not _daily_stats["milestone_70_sent"]:
            _daily_stats["milestone_70_sent"] = True
            filled = int(signal.confidence / 10)
            bar = "🟩" * filled + "⬜" * (10 - filled)
            send_discord_alert({
                "username": f"{CONFIG.spy} Options Bot",
                "content": (
                    f"⚠️ **Getting close! Confidence at {signal.confidence}%** — just {CONFIG.confidence_threshold - signal.confidence} point(s) from firing!\n"
                    f"**{CONFIG.spy}:** ${snapshot.spy_price:.2f}  |  **VIX:** {snapshot.vix_current:.2f}\n"
                    f"{bar}\n"
                    f"📌 Stay ready — next scan in {CONFIG.scan_interval_minutes} minutes could send a trade alert! 👀"
                )
            })
        if signal.confidence < near - 2:
            _daily_stats["milestone_70_sent"] = False

        alert_sent = False
        trade = None
        started_trade = False

        if signal.signal in (EXIT_LONG, EXIT_SHORT):
            # Reversal exit for the open paper trade (previously this fell into the entry path
            # and built a brand-new trade).
            if _active_paper_trade is not None:
                t = _active_paper_trade
                px = _fetch_current_option_price(t["ticker"], t["expiration"], t["strike"], t["direction"])
                if px and px > 0:
                    closed = _close_paper_trade(px, "signal_exit")
                    _exit_alert("signal_exit", closed, px)
                    alert_sent = True
                else:
                    logger.warning("Exit signal fired but option price unavailable — monitor will retry")
            else:
                _current_position = None

        elif signal.signal != HOLD:
            if _active_paper_trade is not None:
                logger.info(f"{signal.signal} suppressed — a paper trade is already open "
                            f"({_active_paper_trade['strike']:.0f} {_active_paper_trade['direction']})")
                reset_cooldown()
            else:
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
                if trade is None:
                    logger.warning(f"{signal.signal} fired but no tradable contract found — will retry next scan")
                    reset_cooldown()
                else:
                    _refresh_trade_prices(trade)
                    discord_payload = format_discord_embed(signal, trade, snapshot.spy_price)
                    whatsapp_msg = format_whatsapp_message(signal, trade, snapshot.spy_price)
                    for attempt in range(1, 4):
                        alert_sent = send_all_alerts(discord_payload, whatsapp_msg).get("discord", False)
                        if alert_sent:
                            break
                        logger.error(f"Alert delivery failed (attempt {attempt}/3)")
                        time.sleep(5)

                    if not alert_sent:
                        # Nothing reached the user, so don't count/track a trade they never saw;
                        # clear the cooldown so the next scan can fire it again.
                        logger.error("Signal NOT delivered — trade not tracked, cooldown reset")
                        reset_cooldown()
                        trade = None
                    else:
                        _daily_stats["trades_fired"] += 1
                        logger.info(f"Trade alert sent — {signal.signal} conf={signal.confidence}%")
                        _active_paper_trade = {
                            "ticker": CONFIG.spy,
                            "direction": trade.direction,
                            "strike": trade.strike,
                            "expiration": trade.expiration,
                            "entry_price": trade.ask_price,
                            "contracts": trade.contracts,
                            "total_cost": trade.total_cost,
                            "profit_target": round(trade.ask_price * (1 + TP_PCT / 100), 2),
                            "stop_loss_price": round(trade.ask_price * (1 + SL_PCT / 100), 2),
                            "entry_time": signal.timestamp,
                            "entry_date": datetime.now(ET).strftime("%Y-%m-%d"),
                            "confidence": signal.confidence,
                            "outcome": "open",
                        }
                        _current_position = _position_label(trade.direction)
                        started_trade = True

        signal_id = log_signal(signal, snapshot.spy_price, snapshot.vix_current, options.iv_rank, macro.breadth_pct, alert_sent=alert_sent)
        if started_trade:
            _active_paper_trade["db_trade_id"] = log_trade_entry(signal_id, trade)
            _persist_active_trade()

    except Exception as e:
        logger.exception(f"Scan cycle error: {e}")


# ─────────────────────────────────────────────
# HOURLY TREND UPDATE
# ─────────────────────────────────────────────
def run_trend_update() -> None:
    if not is_market_open():
        return
    try:
        snapshot, technical, macro, options, signal = _get_snapshot_and_signal(dry_run=True)
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

        if len(_confidence_history) >= 2:
            history_str = " → ".join(f"**{c}%**" for c in _confidence_history)
            d = _confidence_history[-1] - _confidence_history[-2]
            arrow = ("📈 Rising fast" if d >= 5 else "↗️ Rising" if d >= 1 else
                     "📉 Dropping fast" if d <= -5 else "↘️ Falling" if d <= -1 else "➡️ Holding steady")
            trend_line = f"Confidence: {history_str} — {arrow}"
        else:
            trend_line = f"Confidence: **{conf}%**"

        send_discord_alert({
            "username": f"{CONFIG.spy} Options Bot",
            "content": (
                f"{emoji} **{trend}** — {now_et}\n"
                f"**{CONFIG.spy}:** ${price:.2f}  |  **VIX:** {vix:.2f}\n"
                f"{trend_line}\n"
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
        todays_trades = [t for t in paper_trades if _exit_date(t) == today]
        wins = sum(1 for t in todays_trades if t.get("outcome") == "win")
        losses = sum(1 for t in todays_trades if t.get("outcome") == "loss")
        total_pnl = sum(t.get("pnl", 0) for t in todays_trades)
        paper_line = f"📊 Paper trades closed today: {wins}W / {losses}L | P&L: {'+' if total_pnl >= 0 else '-'}${abs(total_pnl):,.0f}" if todays_trades else "📊 No paper trades closed today"

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
def _exit_date(t: dict) -> str:
    """YYYY-MM-DD a trade closed on. Older ledger rows have no exit_date, so fall back to entry day."""
    return t.get("exit_date") or str(t.get("entry_time", ""))[:10]


def _build_money_report(period: str, trades: list, now_et: datetime) -> str:
    """Shared helper that builds a P&L report string for any period."""
    closed = [t for t in trades if t.get("outcome") in ("win", "loss", "breakeven")]
    wins   = [t for t in closed if t.get("outcome") == "win"]
    losses = [t for t in closed if t.get("outcome") == "loss"]
    open_t = [_active_paper_trade] if _active_paper_trade else []
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
        emoji = "✅" if t.get("outcome") == "win" else ("➖" if t.get("outcome") == "breakeven" else "❌")
        pnl   = t.get("pnl", 0)
        trade_lines += f"{emoji} ${t.get('strike',0):.0f} {t.get('direction','').title()} | {'+' if pnl >= 0 else '-'}${abs(pnl):,.0f} ({t.get('pnl_pct',0):+.0f}%)\n"
    if not trade_lines:
        trade_lines = "No closed trades this period\n"

    pnl_emoji = "💰" if total_pnl >= 0 else "📉"
    account   = CONFIG.account_size
    pnl_pct   = round(total_pnl / account * 100, 1) if account else 0

    return (
        f"{ticker_line}\n"
        f"━━━━━━━━━━━━━━━━━━━━━━\n"
        f"📊 **Trades:** {len(closed)} closed | {len(open_t)} open\n"
        f"✅ **Wins:** {len(wins)}  ❌ **Losses:** {len(losses)}  🎯 **Win Rate:** {win_rate}%\n"
        f"💵 **Avg Win:** +${avg_win:,.0f}  |  **Avg Loss:** -${abs(avg_loss):,.0f}\n"
        f"{pnl_emoji} **Total P&L:** {'+' if total_pnl >= 0 else '-'}${abs(total_pnl):,.0f} ({pnl_pct:+.1f}% of account)\n"
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
        week_trades = [t for t in all_trades if _exit_date(t) >= week_start_str]

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
        today_trades = [t for t in all_trades if _exit_date(t) == today]

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
    _restore_active_trade()
    scheduler = BlockingScheduler(timezone=ET)

    scheduler.add_job(run_scan_cycle, trigger=IntervalTrigger(minutes=CONFIG.scan_interval_minutes),
                      id="scan", max_instances=1, coalesce=True, misfire_grace_time=600)

    scheduler.add_job(run_position_monitor, trigger=IntervalTrigger(minutes=5),
                      id="position_monitor", max_instances=1, coalesce=True, misfire_grace_time=300)

    scheduler.add_job(run_trend_update, trigger=IntervalTrigger(minutes=30),
                      id="trend_update", max_instances=1, coalesce=True, misfire_grace_time=600)

    scheduler.add_job(run_premarket_alert, trigger=CronTrigger(hour=9, minute=0, day_of_week="mon-fri", timezone=ET),
                      id="premarket", max_instances=1, coalesce=True, misfire_grace_time=900)

    scheduler.add_job(run_opening_bell, trigger=CronTrigger(hour=9, minute=30, day_of_week="mon-fri", timezone=ET),
                      id="opening_bell", max_instances=1, coalesce=True, misfire_grace_time=900)

    scheduler.add_job(run_closing_bell, trigger=CronTrigger(hour=16, minute=0, day_of_week="mon-fri", timezone=ET),
                      id="closing_bell", max_instances=1, coalesce=True, misfire_grace_time=900)

    # Economic calendar — every weekday at 8:00 AM ET
    scheduler.add_job(run_economic_calendar_check, trigger=CronTrigger(hour=8, minute=0, day_of_week="mon-fri", timezone=ET),
                      id="econ_calendar", max_instances=1, coalesce=True, misfire_grace_time=900)

    # Weekly money report — every Friday at 4:00 PM ET
    scheduler.add_job(run_weekly_report, trigger=CronTrigger(hour=16, minute=0, day_of_week="fri", timezone=ET),
                      id="weekly_report", max_instances=1, coalesce=True, misfire_grace_time=900)

    # Daily money report — Mon-Thu at 4:01 PM ET
    scheduler.add_job(run_daily_money_report, trigger=CronTrigger(hour=16, minute=1, day_of_week="mon-thu", timezone=ET),
                      id="daily_money_report", max_instances=1, coalesce=True, misfire_grace_time=900)

    # NOTE: Flow scan, sentiment reports handled by SPY bot + sentiment bot only
    # QQQ bot only sends to QQQ channel

    logger.info(f"Scheduler started — scanning every {CONFIG.scan_interval_minutes} minutes")
    run_scan_cycle()
    scheduler.start()
