#!/usr/bin/env python3
"""
Which option setup (if any) has a positive expectancy with this signal?  Grid over
  structure : ATM call | 3% ITM call | ATM/+4% call debit spread
  expiry    : 14 | 30 | 60 days
  exits     : see EXITS below
Each config is run on signal days and on random days; we report the difference and a t-stat.
Because ~54 configs are tested, some will look good by luck -> top configs are re-checked on two
separate halves of history (2007-2016 vs 2017-2026) and must be positive in BOTH to count.

Usage: python backtest/setups_test.py QQQ
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import norm

CACHE = Path(__file__).resolve().parent / "cache"
sym = sys.argv[1].upper()
IV_MULT = 1.25 if sym == "QQQ" else 1.0
H = 0.02            # half-spread on each leg, each way
R = 0.03
rng = np.random.default_rng(11)
N = 700             # trades sampled per cell

sig = pd.read_csv(CACHE / f"{sym}_signals.csv", index_col=0, parse_dates=True)
dates = sig.index.values.astype("datetime64[D]")
close = sig["close"].values
vix = sig["vix"].values
n_days = len(sig)


def bs_call(S, K, T, sigma):
    sigma = np.maximum(sigma, 0.05)
    Tp = np.maximum(T, 1e-6)
    d1 = (np.log(S / K) + (R + 0.5 * sigma ** 2) * Tp) / (sigma * np.sqrt(Tp))
    d2 = d1 - sigma * np.sqrt(Tp)
    val = S * norm.cdf(d1) - K * np.exp(-R * Tp) * norm.cdf(d2)
    return np.where(T <= 0, np.maximum(S - K, 0), val)


STRUCTS = {
    "ATM call":        [(+1, lambda S0: np.ceil(S0))],
    "3% ITM call":     [(+1, lambda S0: np.floor(S0 * 0.97))],
    "ATM/+4% spread":  [(+1, lambda S0: np.ceil(S0)), (-1, lambda S0: np.ceil(S0 * 1.04))],
}
# name: dict(tp, sl, hold (max days), stall=(days, band), buffer days before expiry to force-close)
EXITS = {
    "old bot: +50/-50, hold to exp":      dict(tp=.5, sl=-.5, hold=None, stall=None, buf=0),
    "current bot: +50/-50 +stall+buf3":   dict(tp=.5, sl=-.5, hold=None, stall=(5, .2), buf=3),
    "+100/-50, buf3":                     dict(tp=1.0, sl=-.5, hold=None, stall=None, buf=3),
    "+100 / no stop, buf3":               dict(tp=1.0, sl=None, hold=None, stall=None, buf=3),
    "no TP/SL, hold 10d":                 dict(tp=None, sl=None, hold=10, stall=None, buf=3),
    "no TP/SL, hold to buf3":             dict(tp=None, sl=None, hold=None, stall=None, buf=3),
}


def run_trade(i, dte, legs, ex):
    d0 = dates[i]
    expiry = d0 + np.timedelta64(dte, "D")
    jend = int(np.searchsorted(dates, expiry, side="right"))
    if jend >= n_days + 0 and dates[-1] < expiry:
        return None
    js = np.arange(i + 1, jend)
    if len(js) < 2:
        return None
    S0 = close[i]
    T0 = dte / 365
    s0 = vix[i] / 100 * IV_MULT
    T = (expiry - dates[js]).astype(int) / 365
    S = close[js]
    sg = vix[js] / 100 * IV_MULT
    entry = exitv = 0.0
    vals = np.zeros(len(js))
    for qty, kfn in legs:
        K = kfn(S0)
        p0 = float(bs_call(S0, K, T0, s0))
        pj = bs_call(S, K, T, sg)
        if qty > 0:     # long: buy ask, sell bid
            entry += p0 * (1 + H)
            vals += pj * (1 - H)
        else:           # short: sell bid, buy back ask
            entry -= p0 * (1 - H)
            vals -= pj * (1 + H)
    if entry < 0.05:
        return None
    pct = vals / entry - 1
    held = (dates[js] - d0).astype(int)
    left = (expiry - dates[js]).astype(int)
    stop = left <= ex["buf"]
    if ex["tp"] is not None:
        stop |= pct >= ex["tp"]
    if ex["sl"] is not None:
        stop |= pct <= ex["sl"]
    if ex["hold"] is not None:
        stop |= held >= ex["hold"]
    if ex["stall"] is not None:
        stop |= (held >= ex["stall"][0]) & (np.abs(pct) < ex["stall"][1])
    k = int(np.argmax(stop)) if stop.any() else len(js) - 1
    return float(pct[k])


def cell(idx, dte, legs, ex):
    out = []
    for i in idx:
        r = run_trade(int(i), dte, legs, ex)
        if r is not None:
            out.append(r)
    return np.array(out)


fired_idx = np.where(sig["fire"].values & (sig["dir"].values == 1))[0]
all_idx = np.arange(260, n_days - 70)
fired_idx = fired_idx[fired_idx < n_days - 70]
half = np.searchsorted(dates, np.datetime64("2017-01-01"))
sample_f = np.sort(rng.choice(fired_idx, min(N, len(fired_idx)), replace=False))
sample_r = np.sort(rng.choice(all_idx, N, replace=False))
first_f, second_f = sample_f[sample_f < half], sample_f[sample_f >= half]

print(f"=== {sym}: {len(fired_idx)} long signal days, sampling {len(sample_f)} signal / {N} random entries per config")
rows = []
for sname, legs in STRUCTS.items():
    for dte in (14, 30, 60):
        for ename, ex in EXITS.items():
            f = cell(sample_f, dte, legs, ex)
            r = cell(sample_r, dte, legs, ex)
            if len(f) < 50 or len(r) < 50:
                continue
            se = np.sqrt(f.var(ddof=1) / len(f) + r.var(ddof=1) / len(r))
            f1, f2 = cell(first_f, dte, legs, ex), cell(second_f, dte, legs, ex)
            rows.append({
                "setup": f"{sname} {dte}d", "exit": ename, "n": len(f),
                "win%": round((f > 0).mean() * 100, 1), "avg%": round(f.mean() * 100, 1),
                "med%": round(np.median(f) * 100, 1),
                "rand avg%": round(r.mean() * 100, 1),
                "signal-rand": round((f.mean() - r.mean()) * 100, 1),
                "t": round((f.mean() - r.mean()) / se, 2),
                "07-16 avg%": round(f1.mean() * 100, 1) if len(f1) > 20 else np.nan,
                "17-26 avg%": round(f2.mean() * 100, 1) if len(f2) > 20 else np.nan,
            })
df = pd.DataFrame(rows).sort_values("avg%", ascending=False)
df.to_csv(CACHE / f"{sym}_setups.csv", index=False)
pd.set_option("display.width", 250)
print("\nTOP 12 by average P&L per trade (signal days):")
print(df.head(12).to_string(index=False))
print("\nBOTTOM 4:")
print(df.tail(4).to_string(index=False))
pos = df[(df["avg%"] > 0)]
both = pos[(pos["07-16 avg%"] > 0) & (pos["17-26 avg%"] > 0)]
print(f"\n{len(df)} configs | {len(pos)} with positive avg | {len(both)} positive in BOTH halves | "
      f"{(df['t'] > 2).sum()} with signal beating random at t>2")
print("avg% by expiry:", df.assign(dte=df['setup'].str.extract(r'(\d+)d')[0]).groupby('dte')['avg%'].mean().round(1).to_dict())
print("avg% by structure:", df.assign(s=df['setup'].str.replace(r' \d+d', '', regex=True)).groupby('s')['avg%'].mean().round(1).to_dict())
print("avg% by exit:\n", df.groupby('exit')['avg%'].mean().round(1).to_string())
