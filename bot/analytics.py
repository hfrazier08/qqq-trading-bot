"""Win/loss breakdown by setup, so you can see which kinds of trades actually make money."""
from collections import defaultdict
from typing import Callable, Dict, List


def _bucket_conf(t: dict) -> str:
    c = t.get("confidence") or 0
    return "90+" if c >= 90 else "80-89" if c >= 80 else "75-79" if c >= 75 else "70-74" if c >= 70 else "<70"


def _bucket_hour(t: dict) -> str:
    h = t.get("entry_hour_et")
    if h is None:
        return "unknown"
    return "open (9:30-11)" if h < 11 else "midday (11-14)" if h < 14 else "late (14-16)"


def _bucket_iv(t: dict) -> str:
    v = t.get("iv_rank")
    if v is None:
        return "unknown"
    return "IV<20" if v < 20 else "IV 20-35" if v < 35 else "IV 35-50" if v < 50 else "IV 50+"


DIMENSIONS: Dict[str, Callable[[dict], str]] = {
    "confidence": _bucket_conf,
    "direction": lambda t: t.get("direction", "?"),
    "entry time": _bucket_hour,
    "IV rank": _bucket_iv,
    "exit reason": lambda t: t.get("exit_reason", "?"),
}


def breakdown(trades: List[dict], min_trades: int = 1) -> str:
    closed = [t for t in trades if t.get("outcome") in ("win", "loss", "breakeven")]
    if len(closed) < min_trades:
        return ""
    lines = [f"📊 **Setup breakdown** — {len(closed)} closed trades"
             + ("  ⚠️ small sample (<50), don't over-trust" if len(closed) < 50 else "")]
    for dim, fn in DIMENSIONS.items():
        groups = defaultdict(list)
        for t in closed:
            groups[fn(t)].append(t)
        parts = []
        for k in sorted(groups):
            g = groups[k]
            wr = sum(1 for t in g if t["outcome"] == "win") / len(g) * 100
            avg = sum(t.get("pnl_pct", 0) for t in g) / len(g)
            parts.append(f"{k}: {len(g)}T {wr:.0f}% win {avg:+.0f}% avg")
        lines.append(f"**{dim}** — " + " | ".join(parts))
    return "\n".join(lines)
