#!/usr/bin/env python3
"""
Signal backtest: do the bot's 70%+ confidence signals beat random entries?

Replays the bot's REAL scoring code day by day on historical data:
  * technical layer  -> bot.technical_analysis.compute_technical_signals   (real code)
  * macro layer      -> bot.macro_analyzer.analyze_macro                   (real code)
  * options layer    -> bot.options_analyzer._score_options with iv_rank from VIX history;
                        skew / term-structure / liquidity are NOT available historically so
                        they are held at fixed neutral-ish values (skew=0, contango, liquid)
  * gating           -> same rules as bot.signal_generator (layers must agree, threshold, no panic)

Two tests:
  1. Underlying test: direction-adjusted forward return of the index after a signal vs random days
  2. Option test: simulate buying the ATM option (Black-Scholes, IV from VIX) with the bot's own
     exit rules (+50% / -50% on premium, hold to expiry), with spread costs, vs random entries

Usage:  python backtest/run_backtest.py QQQ [--start 2007-07-01] [--threshold 70]
"""
import argparse
import json
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf
from scipy.stats import norm

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from bot.technical_analysis import compute_technical_signals          # noqa: E402
from bot.macro_analyzer import analyze_macro                          # noqa: E402
from bot.options_analyzer import _score_options, compute_iv_rank_via_vix  # noqa: E402

SECTORS = ["XLK", "XLV", "XLF", "XLE", "XLI", "XLY", "XLP", "XLB", "XLU", "XLRE", "XLC"]
CACHE = Path(__file__).resolve().parent / "cache"
CACHE.mkdir(exist_ok=True)
RNG = np.random.default_rng(42)
HORIZONS = [1, 3, 5, 10]


# ───────────────────────── data ─────────────────────────
def load(sym: str) -> pd.DataFrame:
    f = CACHE / f"{sym.replace('^', '')}.pkl"
    if f.exists():
        return pd.read_pickle(f)
    df = yf.Ticker(sym).history(period="max", interval="1d")
    df.index = df.index.tz_localize(None).normalize()
    df = df[["Open", "High", "Low", "Close", "Volume"]].dropna()
    df.to_pickle(f)
    return df


def upto(df: pd.DataFrame, date, n: int) -> pd.DataFrame:
    if df.empty:
        return df
    i = df.index.searchsorted(date, side="right")
    return df.iloc[max(0, i - n):i]


# ───────────────────────── scoring replay ─────────────────────────
def replay(sym: str, start: str, threshold: int) -> pd.DataFrame:
    px = load(sym)
    vix = load("^VIX")
    sect = {s: load(s) for s in SECTORS}
    tlt, uup = load("TLT"), load("UUP")
    dummy_calls = pd.DataFrame({"x": range(5)})   # >=5 liquid contracts -> +2 liquidity pts

    dates = px.index[px.index >= pd.Timestamp(start)]
    rows = []
    for k, d in enumerate(dates):
        hist = upto(px, d, 420)
        if len(hist) < 260:
            continue
        price = float(hist["Close"].iloc[-1])
        tech = compute_technical_signals(pd.DataFrame(), hist, pd.DataFrame(), price)
        vix_h = upto(vix, d, 260)
        if vix_h.empty:
            continue
        vix_now = float(vix_h["Close"].iloc[-1])
        mac = analyze_macro(
            vix_now, vix_h, {s: upto(df, d, 80) for s, df in sect.items()},
            upto(tlt, d, 80), upto(uup, d, 80),
        )
        iv_rank, *_ = compute_iv_rank_via_vix(vix_h)
        opts, _ = _score_options(iv_rank, 0.0, "contango", dummy_calls)

        total = tech.technical_score + mac.macro_score + opts
        agree = tech.technical_bias == mac.macro_bias and tech.technical_bias in ("bullish", "bearish")
        fire = agree and total >= threshold and mac.vix_regime != "panic"
        rows.append({
            "date": d, "close": price, "vix": vix_now, "iv_rank": iv_rank,
            "tech": tech.technical_score, "macro": mac.macro_score, "opts": opts, "total": total,
            "tech_bias": tech.technical_bias, "macro_bias": mac.macro_bias, "agree": agree,
            "fire": fire, "dir": 1 if tech.technical_bias == "bullish" else -1,
        })
        if k % 500 == 0:
            print(f"  {sym}: replayed {k}/{len(dates)} days", flush=True)
    return pd.DataFrame(rows).set_index("date")


# ───────────────────────── test 1: underlying ─────────────────────────
def fwd_returns(sig: pd.DataFrame) -> pd.DataFrame:
    out = sig.copy()
    for h in HORIZONS:
        out[f"r{h}"] = out["close"].shift(-h) / out["close"] - 1
        out[f"adj{h}"] = out[f"r{h}"] * out["dir"]       # positive = signal direction was right
    return out


def underlying_test(sig: pd.DataFrame, n_sims: int = 5000) -> dict:
    res = {}
    fired = sig[sig["fire"]]
    for h in HORIZONS:
        col = f"adj{h}"
        pool = sig.dropna(subset=[col])
        f = fired.dropna(subset=[col])
        if len(f) < 5:
            res[h] = {"n": len(f)}
            continue
        obs = f[col].mean()
        long_frac = (f["dir"] == 1).mean()
        # random baseline: same number of entries, same long/short mix, random days
        raw = pool[f"r{h}"].values
        sims = np.empty(n_sims)
        for i in range(n_sims):
            idx = RNG.integers(0, len(raw), len(f))
            d = np.where(RNG.random(len(f)) < long_frac, 1, -1)
            sims[i] = (raw[idx] * d).mean()
        res[h] = {
            "n": int(len(f)),
            "signal_mean_%": round(obs * 100, 3),
            "signal_hit_rate_%": round((f[col] > 0).mean() * 100, 1),
            "random_mean_%": round(sims.mean() * 100, 3),
            "random_hit_rate_%": round(float(np.mean(
                (pool[f"r{h}"] > 0).astype(float)) * long_frac * 100 +
                (pool[f"r{h}"] < 0).mean() * (1 - long_frac) * 100), 1),
            "p_value(random>=signal)": round(float((sims >= obs).mean()), 3),
        }
    return res


# ───────────────────────── test 2: option simulation ─────────────────────────
def bs(S, K, T, r, sigma, call: bool) -> float:
    if T <= 0:
        return max(S - K, 0.0) if call else max(K - S, 0.0)
    sigma = max(sigma, 0.05)
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    if call:
        return S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    return K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)


def sim_option(sig: pd.DataFrame, i: int, direction: int, iv_mult: float, dte: int = 14,
               tp=0.5, sl=-0.5, max_hold=None, half_spread=0.02, r=0.03) -> dict:
    """Buy ATM option at close of day i. Check each following close. Returns outcome."""
    d0 = sig.index[i]
    S0 = sig["close"].iloc[i]
    call = direction == 1
    K = np.ceil(S0) if call else np.floor(S0)
    expiry = d0 + pd.Timedelta(days=dte)
    sigma0 = sig["vix"].iloc[i] / 100 * iv_mult
    entry = bs(S0, K, dte / 365, r, sigma0, call) * (1 + half_spread)   # pay the ask
    if entry < 0.05:
        return {}
    j = i
    while True:
        j += 1
        if j >= len(sig):
            return {}                       # ran out of data before the trade resolved
        d = sig.index[j]
        T = (expiry - d).days / 365
        S = sig["close"].iloc[j]
        sigma = sig["vix"].iloc[j] / 100 * iv_mult
        val = bs(S, K, max(T, 0), r, sigma, call) * (1 - half_spread)   # sell at the bid
        pct = val / entry - 1
        held = (d - d0).days
        if pct >= tp:
            reason = "take_profit"
        elif pct <= sl:
            reason = "stop_loss"
        elif d >= expiry:
            reason = "expiration"
        elif max_hold and held >= max_hold:
            reason = "time_stop"
        else:
            continue
        return {"pnl_pct": pct, "reason": reason, "held": held, "entry": entry}


def option_trades(sig: pd.DataFrame, iv_mult: float, entries: pd.DataFrame, **kw) -> pd.DataFrame:
    """One position at a time, like the bot."""
    trades, free_after = [], -1
    pos = {d: n for n, d in enumerate(sig.index)}
    for d, row in entries.iterrows():
        i = pos[d]
        if i <= free_after:
            continue
        t = sim_option(sig, i, int(row["dir"]), iv_mult, **kw)
        if not t:
            continue
        end = sig.index.searchsorted(d + pd.Timedelta(days=t["held"]))
        free_after = end
        trades.append({"date": d, "dir": int(row["dir"]), **t})
    return pd.DataFrame(trades)


def summarize(t: pd.DataFrame) -> dict:
    if t.empty:
        return {"n": 0}
    w = t["pnl_pct"] > 0
    return {
        "n": int(len(t)),
        "win_rate_%": round(w.mean() * 100, 1),
        "avg_pnl_%": round(t["pnl_pct"].mean() * 100, 1),
        "median_pnl_%": round(t["pnl_pct"].median() * 100, 1),
        "avg_win_%": round(t.loc[w, "pnl_pct"].mean() * 100, 1) if w.any() else 0,
        "avg_loss_%": round(t.loc[~w, "pnl_pct"].mean() * 100, 1) if (~w).any() else 0,
        "exits": t["reason"].value_counts().to_dict(),
    }


def option_test(sig: pd.DataFrame, iv_mult: float, n_sims: int = 300, **kw) -> dict:
    fired = sig[sig["fire"]]
    real = option_trades(sig, iv_mult, fired, **kw)
    s = summarize(real)
    if real.empty:
        return {"signal": s}
    # random baseline: same number of entries, same long/short mix, random days, same one-at-a-time rule
    long_frac = (fired["dir"] == 1).mean()
    valid = sig.iloc[:-20]
    sim_avgs, sim_wins = [], []
    for _ in range(n_sims):
        days = valid.sample(n=min(len(fired) * 3, len(valid)), random_state=int(RNG.integers(1 << 30))).sort_index()
        days = days.assign(dir=np.where(RNG.random(len(days)) < long_frac, 1, -1))
        tr = option_trades(sig, iv_mult, days, **kw).head(len(real))
        if len(tr) >= max(5, len(real) // 2):
            sim_avgs.append(tr["pnl_pct"].mean())
            sim_wins.append((tr["pnl_pct"] > 0).mean())
    if sim_avgs:
        sim_avgs = np.array(sim_avgs)
        s["random_avg_pnl_%"] = round(sim_avgs.mean() * 100, 1)
        s["random_win_rate_%"] = round(np.mean(sim_wins) * 100, 1)
        s["p_value(random>=signal)"] = round(float((sim_avgs >= real["pnl_pct"].mean()).mean()), 3)
    return {"signal": s, "trades": real}


# ───────────────────────── main ─────────────────────────
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("symbol")
    ap.add_argument("--start", default="2007-07-01")
    ap.add_argument("--threshold", type=int, default=70)
    a = ap.parse_args()
    sym = a.symbol.upper()
    iv_mult = 1.25 if sym == "QQQ" else 1.0     # QQQ implied vol runs ~25% above VIX

    print(f"== {sym}: replaying scoring from {a.start} (threshold {a.threshold}) ==", flush=True)
    sig = fwd_returns(replay(sym, a.start, a.threshold))
    sig.to_csv(CACHE / f"{sym}_signals.csv")
    n_days = len(sig)
    fired = sig[sig["fire"]]
    print(f"\n{n_days} trading days scored | {len(fired)} signals fired "
          f"({len(fired)/n_days*100:.1f}% of days) | long {int((fired['dir']==1).sum())} / short {int((fired['dir']==-1).sum())}")
    print("score distribution (total):", sig["total"].describe()[["mean", "50%", "max"]].round(1).to_dict())
    print("layers agree on", f"{sig['agree'].mean()*100:.1f}% of days")

    out = {"symbol": sym, "days": n_days, "signals": int(len(fired))}

    print("\n--- TEST 1: index move after signal vs random days (direction-adjusted) ---")
    u = underlying_test(sig)
    out["underlying"] = u
    print(pd.DataFrame(u).T.to_string())

    print("\n--- confidence buckets (5-day direction-adjusted return, all days where layers agree) ---")
    ag = sig[sig["agree"]].dropna(subset=["adj5"])
    ag = ag.assign(bucket=pd.cut(ag["total"], [0, 49, 59, 69, 79, 100]))
    bk = ag.groupby("bucket", observed=True)["adj5"].agg(n="count", mean_pct=lambda x: x.mean() * 100,
                                                         hit_rate=lambda x: (x > 0).mean() * 100).round(2)
    out["buckets_5d"] = bk.reset_index().astype({"bucket": str}).to_dict("records")
    print(bk.to_string())

    print("\n--- TEST 2: ATM option, +50%/-50% rules, 14 DTE, 2% half-spread each way ---")
    for label, kw in [("bot rules (hold to expiry)", {}), ("with 5-day time stop", {"max_hold": 5})]:
        o = option_test(sig, iv_mult, **kw)
        out[f"option_{label}"] = o["signal"]
        print(f"\n[{label}]")
        for k, v in o["signal"].items():
            print(f"  {k}: {v}")
        if "trades" in o:
            t = o["trades"]
            by_year = t.assign(year=t["date"].dt.year).groupby("year")["pnl_pct"].agg(n="count", avg=lambda x: x.mean()*100).round(1)
            out[f"option_{label}_by_year"] = by_year.reset_index().to_dict("records")

    (CACHE / f"{sym}_results.json").write_text(json.dumps(out, indent=2, default=str))
    print(f"\nsaved {CACHE / (sym + '_results.json')}")


if __name__ == "__main__":
    main()
