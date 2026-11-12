from datetime import datetime, timedelta
from typing import Optional

import pytz
from bot.config import CONFIG
from bot.signal_generator import TradingSignal, BUY_CALL, BUY_PUT, EXIT_LONG, EXIT_SHORT, HOLD
from bot.trade_constructor import SingleLegTrade

ET = pytz.timezone("America/New_York")
TICKER = CONFIG.spy  # "SPY" or "QQQ" depending on which bot

SIGNAL_COLORS = {
    BUY_CALL:   0x00C851,
    BUY_PUT:    0xFF4444,
    EXIT_LONG:  0xFF8800,
    EXIT_SHORT: 0xFF8800,
    HOLD:       0x9E9E9E,
}

SIGNAL_EMOJI = {
    BUY_CALL:   "🟢",
    BUY_PUT:    "🔴",
    EXIT_LONG:  "🟠",
    EXIT_SHORT: "🟠",
    HOLD:       "⚪",
}

RISK_DISCLAIMER = (
    "⚠️ Not financial advice. Options trading carries significant risk of loss. "
    "Only risk capital you can afford to lose. Paper trade first."
)


def format_discord_embed(
    signal: TradingSignal,
    trade: Optional[SingleLegTrade],
    spy_price: float,
) -> dict:
    """Entry alert. Read top-to-bottom: WHAT to buy -> HOW to exit -> WHY. Everything else is small print."""
    emoji = SIGNAL_EMOJI.get(signal.signal, "⚪")
    color = SIGNAL_COLORS.get(signal.signal, 0x9E9E9E)
    action = {BUY_CALL: "BUY CALL", BUY_PUT: "BUY PUT"}.get(signal.signal, signal.signal.replace("_", " "))
    title = f"{emoji} {TICKER} {action} — {signal.confidence}% ({_confidence_label(signal.confidence)})"

    bd = signal.confidence_breakdown
    scores = (f"Tech {bd.get('technical', 0)}/50 · Macro {bd.get('macro', 0)}/30 · "
              f"Options {bd.get('options', 0)}/20")
    fields = []

    if trade is not None:
        today = datetime.now(ET).date()
        exp = datetime.strptime(trade.expiration, "%Y-%m-%d").date()
        stall_by = today + timedelta(days=CONFIG.time_stop_days)
        close_by = exp - timedelta(days=CONFIG.expiry_buffer_days)
        tp = trade.ask_price * 1.5
        sl = trade.ask_price * 0.5
        description = (
            f"## BUY {trade.contracts}× {TICKER} ${trade.strike:.0f} {trade.option_type}\n"
            f"**Expires:** {trade.expiration_label}  ({trade.dte} days)\n"
            f"**Limit price:** ${trade.ask_price:.2f} per share  →  **${trade.total_cost:,.0f} total** "
            f"({trade.risk_pct_used:.1f}% of account)\n"
            f"**Most you can lose:** ${trade.total_cost:,.0f} (the premium)"
        )
        fields.append({
            "name": "🚪 Exit plan — set these up now",
            "value": (
                f"✅ **Take profit:** sell if the option hits **${tp:.2f}** (+50%)\n"
                f"🛑 **Stop loss:** sell if it falls to **${sl:.2f}** (−50%)\n"
                f"⏱️ **Stalled?** if it's still within ±{CONFIG.time_stop_move_pct:.0f}% on **{stall_by.strftime('%b %-d')}**, sell\n"
                f"⌛ **Last day to hold:** **{close_by.strftime('%b %-d')}** — time decay speeds up after that"
            ),
            "inline": False,
        })
    else:
        description = f"**{signal.direction.capitalize()}** setup on {TICKER} at ${spy_price:.2f}"

    reasons = "\n".join(f"• {r}" for r in signal.signal_reasons[:3])
    fields.append({"name": "📈 Why this signal", "value": reasons[:1024] or "N/A", "inline": False})

    small = [f"**{TICKER}:** ${spy_price:.2f}", f"**Risk:** {signal.risk_level.capitalize()}", f"**Score:** {scores}"]
    if trade is not None:
        small += [f"**Breakeven:** ${trade.breakeven:.2f}", f"**Delta:** {trade.delta:.2f}",
                  f"**IV:** {trade.implied_volatility:.0f}%"]
    fields.append({"name": "Details", "value": "  ·  ".join(small), "inline": False})

    return {
        "username": f"{TICKER} Options Bot",
        "embeds": [{
            "title": title,
            "description": description,
            "color": color,
            "fields": fields,
            "footer": {"text": f"{signal.timestamp} · Paper trade first · Not financial advice"},
        }],
    }


def format_whatsapp_message(
    signal: TradingSignal,
    trade: Optional[SingleLegTrade],
    spy_price: float,
) -> str:
    emoji = SIGNAL_EMOJI.get(signal.signal, "⚪")
    lines = [
        f"{emoji} *{TICKER} BOT — {signal.signal}*",
        f"Price: ${spy_price:.2f} | Conf: {signal.confidence}%",
        "",
    ]
    if trade:
        lines += [
            f"*Buy {trade.option_type}: ${trade.strike:.0f} Strike*",
            f"Exp: {trade.expiration_label} ({trade.dte} DTE)",
            f"Ask: ${trade.ask_price:.2f} | Cost: ${trade.total_cost:,.0f}",
            f"Breakeven: ${trade.breakeven:.2f}",
            f"Target: ${trade.profit_target:.2f} | Stop: ${trade.stop_loss_price:.2f}",
            "",
        ]
    lines.append("*Why:*")
    for r in signal.signal_reasons[:3]:
        lines.append(f"• {r}")
    lines.append("")
    lines.append("Not financial advice.")
    return "\n".join(lines)


def _confidence_label(conf: int) -> str:
    if conf >= 90: return "VERY HIGH"
    if conf >= 80: return "HIGH"
    if conf >= 70: return "MODERATE"
    return "LOW"


def _robinhood_steps(trade: SingleLegTrade) -> str:
    return (
        f"1️⃣ Open Robinhood → Search **{TICKER}**\n"
        f"2️⃣ Tap **Trade** → **{trade.option_type}**\n"
        f"3️⃣ Select expiration: **{trade.expiration_label}**\n"
        f"4️⃣ Find strike: **${trade.strike:.0f}**\n"
        f"5️⃣ Tap **Buy** → set quantity: **{trade.contracts} contract(s)**\n"
        f"6️⃣ Order type: **Limit** → enter **${trade.ask_price:.2f}**\n"
        f"7️⃣ Review cost: **${trade.total_cost:,.0f} total**\n"
        f"8️⃣ Tap **Review** → **Submit**\n"
        f"9️⃣ Set alert: notify if {TICKER} hits **${trade.breakeven:.2f}**"
    )[:1024]
