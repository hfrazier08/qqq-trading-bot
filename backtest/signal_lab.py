#!/usr/bin/env python3
"""
Signal lab: can ANY simple rule (bullish or bearish) beat the market's own drift?

Candidates are fixed up-front from textbook definitions (no parameter tuning on this data), so there are
only 9 of them -> little room for data-snooping.  Each is judged on:
  * edge = mean forward return after the signal  MINUS  mean forward return on ALL days (the drift)
  * for bearish signals the money-making condition is a NEGATIVE forward return (we'd hold puts)
  * a t-stat computed on thinned (non-overlapping) signal days
  * consistency: same sign of edge in 2007-2016 AND 2017-2026, on BOTH QQQ and SPY

Usage: python backtest/signal_lab.py
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_backtest as rb  # noqa: E402  (reuses cached price data)

HORIZONS = (5, 10, 20)
SPLIT = pd.Timestamp("2017-01-01")
START = pd.Timestamp("2008-01-01")   # need 252 days of warmup after 2007-07


def rsi(close: pd.Series, n: int) -> pd.Series:
    d = close.diff()
    up = d.clip(lower=0).ewm(alpha=1 / n, adjust=False).mean()
    dn = (-d.clip(upper=0)).ewm(alpha=1 / n, adjust=False).mean()
    return 100 - 100 / (1 + up / dn.replace(0, np.nan))


def features(sym: str) -> pd.DataFrame:
    px = rb.load(sym)["Close"]
    vix = rb.load("^VIX")["Close"].reindex(px.index).ffill()
    f = pd.DataFrame({"close": px, "vix": vix})
    f["sma50"] = px.rolling(50).mean()
    f["sma200"] = px.rolling(200).mean()
    f["rsi2"] = rsi(px, 2)
    f["mom"] = px.shift(21) / px.shift(252) - 1          # 12-1 month momentum
    f["vix_ma10"] = vix.rolling(10).mean()
    f["down3"] = (px.diff() < 0).rolling(3).sum() == 3
    for h in HORIZONS:
        f[f"r{h}"] = px.shift(-h) / px - 1
    return f.loc[START:]


def candidates(f: pd.DataFrame) -> dict:
    up = (f.close > f.sma200)
    dn = (f.close < f.sma200)
    vix_spike = f.vix > f.vix_ma10 * 1.15
    return {
        # name: (mask, side)   side +1 = want price UP after, -1 = want price DOWN after
        "L1 trend: >200d & 50d>200d":            (up & (f.sma50 > f.sma200), +1),
        "L2 pullback: >200d & RSI(2)<10":        (up & (f.rsi2 < 10), +1),
        "L3 dip: >200d & 3 down days":           (up & f.down3, +1),
        "L4 fear spike: VIX>1.15xMA10 & >200d":  (up & vix_spike, +1),
        "L5 momentum: 12-1m>0 & >200d":          (up & (f.mom > 0), +1),
        "S1 bear regime: <200d & 50d<200d":      (dn & (f.sma50 < f.sma200), -1),
        "S2 <200d & VIX rising":                 (dn & vix_spike, -1),
        "S3 bear rally: <200d & RSI(2)>90":      (dn & (f.rsi2 > 90), -1),
        "S4 momentum: 12-1m<0 & <200d":          (dn & (f.mom < 0), -1),
    }


def thin(idx: np.ndarray, gap: int) -> np.ndarray:
    keep, last = [], -10 ** 9
    for i in idx:
        if i - last >= gap:
            keep.append(i)
            last = i
    return np.array(keep, dtype=int)


def evaluate(f: pd.DataFrame, mask: pd.Series, side: int, h: int) -> dict:
    r = f[f"r{h}"].values
    ok = ~np.isnan(r)
    drift = r[ok].mean()
    idx = thin(np.where(mask.values & ok)[0], h)
    if len(idx) < 15:
        return {"n": len(idx)}
    x = r[idx]
    edge = (x.mean() - drift) * side            # positive = signal "worked" vs drift in its direction
    se = x.std(ddof=1) / np.sqrt(len(x))
    t = edge / se
    dates = f.index[idx]
    d1 = (f.index < SPLIT) & ok
    d2 = (f.index >= SPLIT) & ok
    e1 = (x[dates < SPLIT].mean() - r[d1].mean()) * side if (dates < SPLIT).sum() >= 8 else np.nan
    e2 = (x[dates >= SPLIT].mean() - r[d2].mean()) * side if (dates >= SPLIT).sum() >= 8 else np.nan
    return {"n": len(idx), "mean%": x.mean() * 100, "drift%": drift * 100, "edge%": edge * 100,
            "t": t, "hit%": ((x * side) > 0).mean() * 100,
            "edge 07-16": e1 * 100, "edge 17-26": e2 * 100,
            "abs P&L% (hold in direction)": x.mean() * side * 100}


def main():
    verdict = {}
    for sym in ("QQQ", "SPY"):
        f = features(sym)
        print(f"\n================ {sym}  ({f.index[0].date()} → {f.index[-1].date()}, {len(f)} days) ================")
        for h in HORIZONS:
            rows = []
            for name, (m, side) in candidates(f).items():
                e = evaluate(f, m, side, h)
                if "edge%" in e:
                    rows.append({"signal": name, **{k: (round(v, 2) if isinstance(v, float) else v) for k, v in e.items()}})
                    verdict.setdefault(name, {}).setdefault(h, []).append(e)
            df = pd.DataFrame(rows)
            print(f"\n-- {h}-day forward return, thinned to non-overlapping signals --")
            pd.set_option("display.width", 250)
            print(df.to_string(index=False))
        # reference: the bot's old signal on the same footing
        old = pd.read_csv(rb.CACHE / f"{sym}_signals.csv", index_col=0, parse_dates=True)
        fm = f.join(old[["fire"]], how="left")["fire"].fillna(False).astype(bool)
        e = evaluate(f, fm, +1, 10)
        print(f"\nreference — OLD bot signal (70%+, all long): n={e['n']} edge={e['edge%']:.2f}% t={e['t']:.2f} "
              f"halves {e['edge 07-16']:.2f}/{e['edge 17-26']:.2f}")

    print("\n\n################ VERDICT (10-day horizon, both tickers, edge must be >0 with t>2 and positive in both halves) ################")
    for name, byh in verdict.items():
        es = byh[10]
        if len(es) < 2:
            continue
        ok = all(e["t"] > 2 and e["edge 07-16"] > 0 and e["edge 17-26"] > 0 for e in es)
        print(f"{'PASS' if ok else 'fail'}  {name:42s}  edge% QQQ/SPY: {es[0]['edge%']:+.2f}/{es[1]['edge%']:+.2f}  "
              f"t: {es[0]['t']:+.1f}/{es[1]['t']:+.1f}  abs P&L%: {es[0]['abs P&L% (hold in direction)']:+.2f}/{es[1]['abs P&L% (hold in direction)']:+.2f}")


if __name__ == "__main__":
    main()
