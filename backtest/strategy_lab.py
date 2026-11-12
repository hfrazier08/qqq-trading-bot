#!/usr/bin/env python3
"""
Strategy lab round 2 — rules fixed BEFORE looking at results:

  A) Put credit spread (sell ~20-delta put, buy a put 3% of spot lower), 45 DTE,
     close at 50% of max profit, or loss of 2x credit, or 21 days left.
     Variants: ungated | only when close>200d SMA | >200d and VIX<30
  B) Trend-gated long call spread (ATM / +4%), 60 DTE, exit +100% / -50% / 3 days left,
     plus an extra exit if price closes below its 200d SMA. Variants: ungated | >200d gate

Judgement protocol (set in advance):
  * "design" period 2008-2016, "confirm" period 2017-2026 — a variant only counts if it is
    profitable in BOTH, on BOTH QQQ and SPY, and its worst year isn't a wipe-out
  * costs: 1bp of the underlying per leg per side; flat-skew pricing from VIX (QQQ IV = 1.25 x VIX)
  * one position at a time, sized so max loss = 1 unit; we report average return on risk, win rate,
    worst trade, max drawdown (in units of risk) and the worst calendar year

Usage: python backtest/strategy_lab.py
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import norm

sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_backtest as rb  # noqa: E402

R = 0.03
SPLIT = pd.Timestamp("2017-01-01")
COST = 1e-4      # of spot, per leg per side


def d1d2(S, K, T, sg):
    sg = np.maximum(sg, 0.05)
    Tp = np.maximum(T, 1e-6)
    d1 = (np.log(S / K) + (R + 0.5 * sg ** 2) * Tp) / (sg * np.sqrt(Tp))
    return d1, d1 - sg * np.sqrt(Tp)


def call(S, K, T, sg):
    a, b = d1d2(S, K, T, sg)
    v = S * norm.cdf(a) - K * np.exp(-R * np.maximum(T, 1e-6)) * norm.cdf(b)
    return np.where(T <= 0, np.maximum(S - K, 0), v)


def put(S, K, T, sg):
    a, b = d1d2(S, K, T, sg)
    v = K * np.exp(-R * np.maximum(T, 1e-6)) * norm.cdf(-b) - S * norm.cdf(-a)
    return np.where(T <= 0, np.maximum(K - S, 0), v)


class Market:
    def __init__(self, sym, start="2008-01-01", end=None):
        px = rb.load(sym)["Close"]
        vix = rb.load("^VIX")["Close"].reindex(px.index).ffill()
        self.mult = 1.25 if sym == "QQQ" else 1.0
        self.df = pd.DataFrame({"close": px, "vix": vix})
        self.df["sma200"] = px.rolling(200).mean()
        self.df = self.df.loc[start:end]
        self.dates = self.df.index.values.astype("datetime64[D]")
        self.close = self.df["close"].values
        self.vix = self.df["vix"].values
        self.sma = self.df["sma200"].values
        self.n = len(self.df)


def credit_spread(m: Market, i: int, dte=45):
    """Returns (return_on_risk, exit_idx) or None."""
    d0 = m.dates[i]
    expiry = d0 + np.timedelta64(dte, "D")
    if m.dates[-1] < expiry:
        return None
    S0, T0 = m.close[i], dte / 365
    s0 = m.vix[i] / 100 * m.mult
    Ks = S0 * np.exp(-0.84 * s0 * np.sqrt(T0))            # ~20 delta
    Kl = Ks - 0.03 * S0
    width = Ks - Kl
    c = COST * S0
    credit = float(put(S0, Ks, T0, s0)) - c - (float(put(S0, Kl, T0, s0)) + c)
    if credit <= 0.02 * width:
        return None
    js = np.arange(i + 1, int(np.searchsorted(m.dates, expiry, side="right")))
    if len(js) < 2:
        return None
    T = (expiry - m.dates[js]).astype(int) / 365
    S, sg = m.close[js], m.vix[js] / 100 * m.mult
    cost_to_close = (put(S, Ks, T, sg) + c) - (put(S, Kl, T, sg) - c)
    cost_to_close = np.minimum(cost_to_close, width)       # a spread can't cost more than its width
    pnl = credit - cost_to_close
    left = (expiry - m.dates[js]).astype(int)
    stop = (pnl >= 0.5 * credit) | (pnl <= -2 * credit) | (left <= 21)
    k = int(np.argmax(stop)) if stop.any() else len(js) - 1
    return pnl[k] / (width - credit), js[k]


def call_spread(m: Market, i: int, dte=60, regime_exit=False):
    d0 = m.dates[i]
    expiry = d0 + np.timedelta64(dte, "D")
    if m.dates[-1] < expiry:
        return None
    S0, T0 = m.close[i], dte / 365
    s0 = m.vix[i] / 100 * m.mult
    Kl, Ks = np.ceil(S0), np.ceil(S0 * 1.04)
    c = COST * S0
    debit = (float(call(S0, Kl, T0, s0)) + c) - (float(call(S0, Ks, T0, s0)) - c)
    if debit < 0.05:
        return None
    js = np.arange(i + 1, int(np.searchsorted(m.dates, expiry, side="right")))
    if len(js) < 2:
        return None
    T = (expiry - m.dates[js]).astype(int) / 365
    S, sg = m.close[js], m.vix[js] / 100 * m.mult
    val = (call(S, Kl, T, sg) - c) - (call(S, Ks, T, sg) + c)
    pct = val / debit - 1
    left = (expiry - m.dates[js]).astype(int)
    stop = (pct >= 1.0) | (pct <= -0.5) | (left <= 3)
    if regime_exit:
        stop |= S < m.sma[js]
    k = int(np.argmax(stop)) if stop.any() else len(js) - 1
    return float(np.clip(pct[k], -1, None)) * (debit / debit), js[k]    # return on risk (= debit)


def run(m: Market, fn, gate, entry_end=None) -> pd.DataFrame:
    rows, i = [], 252
    while i < m.n - 70:
        if entry_end is not None and m.df.index[i] >= entry_end:
            break
        if gate(m, i):
            r = fn(m, i)
            if r is not None:
                ret, j = r
                rows.append({"date": m.df.index[i], "ret": ret})
                i = max(j, i + 1) + 1
                continue
        i += 1
    return pd.DataFrame(rows)


def summarize(t: pd.DataFrame) -> dict:
    if len(t) < 10:
        return {"n": len(t)}
    x = t["ret"].values
    eq = np.cumsum(x)
    dd = float((eq - np.maximum.accumulate(eq)).min())
    yr = t.groupby(t["date"].dt.year)["ret"].sum()
    return {"n": len(t), "win%": round((x > 0).mean() * 100, 1), "avg%risk": round(x.mean() * 100, 1),
            "worst%risk": round(x.min() * 100), "maxDD(units)": round(dd, 1),
            "worst yr(units)": round(yr.min(), 1), "worst yr": int(yr.idxmin()), "total(units)": round(eq[-1], 1)}


GATES = {
    "ungated":            lambda m, i: True,
    ">200d":              lambda m, i: m.close[i] > m.sma[i],
    ">200d & VIX<30":     lambda m, i: m.close[i] > m.sma[i] and m.vix[i] < 30,
}


def main():
    early = len(sys.argv) > 1 and sys.argv[1] == "early"
    out = {}
    for sym in ("QQQ", "SPY"):
        # EARLY mode = true hold-out: 1997/99-2007, which includes the dot-com crash and the flat 2000-2007 market
        m = Market(sym, start=("1996-01-01" if sym == "SPY" else "1998-06-01"), end="2008-06-30") if early else Market(sym)
        print(f"\n################ {sym} ################")
        variants = {
            "A put credit spread, 45d, 20-delta": credit_spread,
            "B call spread 60d (gate=entry)":     lambda mm, ii: call_spread(mm, ii, regime_exit=False),
            "B call spread 60d + exit<200d":      lambda mm, ii: call_spread(mm, ii, regime_exit=True),
        }
        for vname, fn in variants.items():
            gates = GATES if vname.startswith("A") else {k: v for k, v in GATES.items() if k != ">200d & VIX<30"}
            for gname, g in gates.items():
                t = run(m, fn, g, entry_end=pd.Timestamp('2008-01-01') if early else None)
                if t.empty:
                    continue
                splits = (("ALL", t), ("08-16", t[t["date"] < pd.Timestamp("2002-01-01")]), ("17-26", t[t["date"] >= pd.Timestamp("2002-01-01")])) if early \
                    else (("ALL", t), ("08-16", t[t["date"] < SPLIT]), ("17-26", t[t["date"] >= SPLIT]))
                for label, sub in splits:
                    s = summarize(sub)
                    out[(sym, vname, gname, label)] = s
                print(f"\n{vname} | gate: {gname}")
                print(pd.DataFrame({lab: out[(sym, vname, gname, lab)] for lab in ("08-16", "17-26", "ALL")}).to_string())

    print("\n\n################ VERDICT: profitable in BOTH periods on BOTH tickers (avg%risk > 0) ################")
    for vname in {k[1] for k in out}:
        for gname in {k[2] for k in out if k[1] == vname}:
            cells = [out.get((s, vname, gname, p), {}) for s in ("QQQ", "SPY") for p in ("08-16", "17-26")]
            if any("avg%risk" not in c for c in cells):
                continue
            ok = all(c["avg%risk"] > 0 for c in cells)
            avgs = " / ".join(f"{c['avg%risk']:+.1f}" for c in cells)
            worst = min(c["worst yr(units)"] for c in cells)
            print(f"{'PASS' if ok else 'fail'}  {vname:38s} gate {gname:16s} avg%risk [QQQ 08-16 / QQQ 17-26 / SPY 08-16 / SPY 17-26]: {avgs}   worst year {worst} units")


if __name__ == "__main__":
    main()
