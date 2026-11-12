"""Entry guards that protect the account (not predictions): stress, monthly loss, and size limits."""
from datetime import datetime
from typing import List, Optional

import pytz

ET = pytz.timezone("America/New_York")


def month_realized_pnl(trades: List[dict], now: Optional[datetime] = None) -> float:
    now = now or datetime.now(ET)
    ym = now.strftime("%Y-%m")
    return sum(t.get("pnl", 0) for t in trades
               if t.get("outcome") in ("win", "loss", "breakeven")
               and str(t.get("exit_date") or t.get("entry_time", ""))[:7] == ym)


def entry_check(cfg, trades: List[dict], vix: float, cost_total: Optional[float] = None,
                now: Optional[datetime] = None) -> Optional[str]:
    """Return a plain-English reason to skip a new entry, or None if it's fine."""
    if vix is not None and cfg.risk_off_vix > 0 and vix > cfg.risk_off_vix:
        return f"risk-off: VIX {vix:.1f} is above the {cfg.risk_off_vix:.0f} limit"
    limit = cfg.account_size * cfg.max_monthly_loss_pct / 100
    pnl = month_realized_pnl(trades, now)
    if cfg.max_monthly_loss_pct > 0 and pnl <= -limit:
        return (f"monthly loss limit hit: realized ${pnl:,.0f} this month "
                f"(limit −${limit:,.0f}); new entries resume next month")
    if cost_total is not None and cfg.max_position_pct > 0:
        cap = cfg.account_size * cfg.max_position_pct / 100
        if cost_total > cap:
            return (f"position too big: costs ${cost_total:,.0f}, above your ${cap:,.0f} limit "
                    f"({cfg.max_position_pct:.0f}% of the ${cfg.account_size:,.0f} account)")
    return None
