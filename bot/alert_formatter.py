from typing import Optional
from bot.signal_generator import TradingSignal, BUY_CALL, BUY_PUT, EXIT_LONG, EXIT_SHORT, HOLD
from bot.trade_constructor import SingleLegTrade

SIGNAL_COLORS = {
    BUY_CALL:   0x00C851,   # green
    BUY_PUT:    0xFF4444,   # red
    EXIT_LONG:  0xFF8800,   # orange
    EXIT_SHORT: 0xFF8800,
    HOLD:       0x9E9E9E,   # gray
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
    emoji = SIGNAL_EMOJI.get(signal.signal, "⚪")
    color = SIGNAL_COLORS.get(signal.signal, 0x9E9E9E)
    conf_label = _confidence_label(signal.confidence)

    title = f"{emoji} SPY {signal.signal.replace('_', ' ')} — {signal.confidence}% ({conf_label})"

    fields = []

    # Market snapshot
    fields.append({
        "name": "📊 Market Snapshot",
        "value": (
            f"**SPY Price:** ${spy_price:.2f}\n"
            f"**Signal:** {signal.signal}\n"
            f"**Direction:** {signal.direction.capitalize()}\n"
            f"**Risk Level:** {signal.risk_level.capitalize()}"
        ),
        "inline": False,
    })

    # Confidence breakdown
    bd = signal.confidence_breakdown
    fields.append({
        "name": "🎯 Confidence Breakdown",
        "value": (
            f"Technical: **{bd.get('technical', 0)}/50**\n"
            f"Macro: **{bd.get('macro', 0)}/30**\n"
            f"Options: **{bd.get('options', 0)}/20**\n"
            f"Total: **{signal.confidence}/100**"
        ),
        "inline": True,
    })

    # Research reasons
    reasons_text = "\n".join(f"• {r}" for r in signal.signal_reasons[:5])
    fields.append({
        "name": "🔍 Research Summary",
        "value": reasons_text[:1024] if reasons_text else "N/A",
        "inline": False,
    })

    if trade is not None:
        # Trade details
        fields.append({
            "name": f"💼 Trade: Buy {trade.option_type}",
            "value": (
                f"**Symbol:** SPY\n"
                f"**Action:** Buy {trade.option_type}\n"
                f"**Strike:** ${trade.strike:.0f}\n"
                f"**Expiration:** {trade.expiration_label} ({trade.dte} DTE)"
            ),
            "inline": True,
        })

        fields.append({
            "name": "💰 Pricing",
            "value": (
                f"**Ask Price:** ${trade.ask_price:.2f}/contract\n"
                f"**Bid Price:** ${trade.bid_price:.2f}/contract\n"
                f"**Cost:** ${trade.cost_per_contract:.0f} per contract\n"
                f"**IV:** {trade.implied_volatility:.0f}%  |  **Delta:** {trade.delta:.2f}"
            ),
            "inline": True,
        })

        fields.append({
            "name": "📐 Position Size",
            "value": (
                f"**Contracts:** {trade.contracts}\n"
                f"**Total Cost:** ${trade.total_cost:,.0f}\n"
                f"**% of Account:** {trade.risk_pct_used:.1f}%\n"
                f"**Max Loss:** ${trade.total_cost:,.0f} (full premium paid)"
            ),
            "inline": True,
        })

        fields.append({
            "name": "🎯 Key Price Levels",
            "value": (
                f"**Breakeven at expiry:** ${trade.breakeven:.2f}\n"
                f"**Profit target (SPY):** ${trade.profit_target:.2f}\n"
                f"**Stop loss:** Exit if option drops to **${trade.stop_loss_price:.2f}** (50% loss)"
            ),
            "inline": False,
        })

        fields.append({
            "name": "📋 How to Buy in Robinhood",
            "value": _robinhood_steps(trade),
            "inline": False,
        })

        fields.append({
            "name": "📌 Trade Management",
            "value": (
                f"• **Take profit:** Sell when option price reaches **${trade.ask_price * 1.5:.2f}** (+50%)\n"
                f"• **Stop loss:** Sell if option drops to **${trade.stop_loss_price:.2f}** (-50%)\n"
                f"• **Hard exit:** Sell with 3–5 days left before expiration\n"
                f"• **Do not hold** through expiration — time decay accelerates"
            ),
            "inline": False,
        })

    fields.append({
        "name": "⚠️ Risk Disclosure",
        "value": RISK_DISCLAIMER,
        "inline": False,
    })

    return {
        "username": "QQQ Options Bot",
        "embeds": [{
            "title": title,
            "color": color,
            "fields": fields[:25],
            "footer": {"text": f"Scanned at {signal.timestamp}"},
        }],
    }


def format_whatsapp_message(
    signal: TradingSignal,
    trade: Optional[SingleLegTrade],
    spy_price: float,
) -> str:
    emoji = SIGNAL_EMOJI.get(signal.signal, "⚪")
    lines = [
        f"{emoji} *SPY BOT — {signal.signal}*",
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
    if conf >= 90:
        return "VERY HIGH"
    if conf >= 80:
        return "HIGH"
    if conf >= 70:
        return "MODERATE"
    return "LOW"


def _robinhood_steps(trade: SingleLegTrade) -> str:
    return (
        f"1️⃣ Open Robinhood → Search **QQQ**\n"
        f"2️⃣ Tap **Trade** → **{trade.option_type}**\n"
        f"3️⃣ Select expiration: **{trade.expiration_label}**\n"
        f"4️⃣ Find strike: **${trade.strike:.0f}**\n"
        f"5️⃣ Tap **Buy** → set quantity: **{trade.contracts} contract(s)**\n"
        f"6️⃣ Order type: **Limit** → enter **${trade.ask_price:.2f}**\n"
        f"7️⃣ Review cost: **${trade.total_cost:,.0f} total**\n"
        f"8️⃣ Tap **Review** → **Submit**\n"
        f"9️⃣ Set alert: notify if SPY hits **${trade.breakeven:.2f}**"
    )[:1024]
