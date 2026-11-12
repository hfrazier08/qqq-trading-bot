#!/usr/bin/env python3
"""
Does an IV-rank cap rescue the signal?  Reuses backtest/cache/<SYM>_signals.csv from run_backtest.py.

For each cap (no cap, <=50, <=35, <=20) it simulates the bot's option trades on signal days that pass the
cap and compares against two random baselines:
  A) random days that pass the SAME IV cap  -> isolates the signal's edge
  B) random days with no cap                -> shows what the cap does to ANY strategy
Usage: python backtest/iv_filter_test.py QQQ
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_backtest as rb  # noqa: E402

sym = sys.argv[1].upper()
iv_mult = 1.25 if sym == "QQQ" else 1.0
sig = pd.read_csv(rb.CACHE / f"{sym}_signals.csv", index_col=0, parse_dates=True)
valid = sig.iloc[:-20]
fired = sig[sig["fire"]]
rng = np.random.default_rng(7)
N_SIMS = 100
long_frac = (fired["dir"] == 1).mean()


def stats(t):
    if t.empty:
        return dict(n=0, win=np.nan, avg=np.nan, med=np.nan)
    return dict(n=len(t), win=(t["pnl_pct"] > 0).mean() * 100, avg=t["pnl_pct"].mean() * 100, med=t["pnl_pct"].median() * 100)


def random_avg(pool: pd.DataFrame, n_target: int, **kw):
    avgs, wins = [], []
    for _ in range(N_SIMS):
        days = pool.sample(n=min(n_target * 3, len(pool)), random_state=int(rng.integers(1 << 30))).sort_index()
        days = days.assign(dir=np.where(rng.random(len(days)) < long_frac, 1, -1))
        tr = rb.option_trades(sig, iv_mult, days, **kw).head(n_target)
        if len(tr) >= max(5, n_target // 2):
            avgs.append(tr["pnl_pct"].mean() * 100)
            wins.append((tr["pnl_pct"] > 0).mean() * 100)
    return (np.mean(avgs), np.mean(wins), np.array(avgs)) if avgs else (np.nan, np.nan, np.array([]))


print(f"=== {sym} | IV rank on signal days: median {fired['iv_rank'].median():.0f}, "
      f"{(fired['iv_rank']<=50).mean()*100:.0f}% <=50, {(fired['iv_rank']<=35).mean()*100:.0f}% <=35, {(fired['iv_rank']<=20).mean()*100:.0f}% <=20")

# 1) every signal trade, bucketed by IV rank at entry (no one-at-a-time overlap so buckets are comparable)
allt = []
for d, row in fired.iterrows():
    i = sig.index.get_loc(d)
    t = rb.sim_option(sig, i, int(row["dir"]), iv_mult)
    if t:
        allt.append({"date": d, "iv_rank": row["iv_rank"], **t})
allt = pd.DataFrame(allt)
allt["bucket"] = pd.cut(allt["iv_rank"], [-1, 10, 20, 35, 50, 75, 101],
                        labels=["0-10", "10-20", "20-35", "35-50", "50-75", "75-100"])
b = allt.groupby("bucket", observed=True)["pnl_pct"].agg(n="count", win=lambda x: (x > 0).mean() * 100,
                                                         avg=lambda x: x.mean() * 100, med=lambda x: x.median() * 100).round(1)
print("\nAvg option P&L by IV rank at entry (all signal days, bot exit rules):")
print(b.to_string())

# 2) caps, bot rules and with a 5-day time stop
for label, kw in [("bot rules (hold to expiry)", {}), ("5-day time stop", {"max_hold": 5})]:
    print(f"\n[{label}]   avg/med/win are per-trade option P&L %")
    rows = []
    for cap in [None, 50, 35, 20]:
        passed = fired if cap is None else fired[fired["iv_rank"] <= cap]
        tr = rb.option_trades(sig, iv_mult, passed, **kw)
        s = stats(tr)
        pool_same = valid if cap is None else valid[valid["iv_rank"] <= cap]
        ra, rw, arr = random_avg(pool_same, max(s["n"], 10), **kw) if s["n"] else (np.nan, np.nan, np.array([]))
        p = float((arr >= s["avg"]).mean()) if len(arr) and s["n"] else np.nan
        rows.append({"IV cap": "none" if cap is None else f"<={cap}", "trades": s["n"], "win%": round(s["win"], 1),
                     "avg%": round(s["avg"], 1), "median%": round(s["med"], 1),
                     "random(same cap) avg%": round(ra, 1), "random win%": round(rw, 1), "p(random>=signal)": round(p, 3)})
    print(pd.DataFrame(rows).to_string(index=False))

# 3) what the cap does to purely random entries
print("\nRandom entries (no signal at all) by IV cap, bot rules:")
for cap in [None, 50, 35, 20]:
    pool = valid if cap is None else valid[valid["iv_rank"] <= cap]
    ra, rw, _ = random_avg(pool, 400)
    print(f"  cap {'none' if cap is None else '<=' + str(cap):>5}: avg {ra:+.1f}%  win {rw:.1f}%  ({len(pool)} eligible days)")
