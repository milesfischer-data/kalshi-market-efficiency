"""
conditioners.py - Is there ANY observable that lifts P(aligned) above 66.24%?

Sizing cannot fix negative EV -- E[aX] = aE[X], so scaling never changes
the sign. But CONDITIONING might: if some observable identifies a subset
of windows where alignment runs above break-even, that subset is tradeable
even if the average is not.

This tests several candidate conditioners on the data alignment.py pulled.
Every one is strictly backward-looking.

    py conditioners.py

WARNING BUILT INTO THE OUTPUT: testing many conditioners means the best
one looks good by chance. The script reports how many were tried and what
significance level that demands.
"""

from pathlib import Path

import math

import numpy as np
import pandas as pd

def norm_cdf(x):
    """Standard normal CDF via the error function (stdlib only)."""
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def norm_ppf(p, lo=-10.0, hi=10.0, tol=1e-10):
    """Inverse normal CDF by bisection. No scipy needed."""
    for _ in range(200):
        mid = (lo + hi) / 2.0
        if norm_cdf(mid) < p:
            lo = mid
        else:
            hi = mid
        if hi - lo < tol:
            break
    return (lo + hi) / 2.0


DATA = Path("data")
BREAKEVEN = 5 / 18 + 5 / 13     # 0.6624, from the quoted parlay odds

# Net payoffs on a $10 double-parlay stake
R_YES = (5 / 18) / BREAKEVEN
NET_YES, NET_NO, NET_MIX = 8.0, 3.0, -10.0


def ev_at(p_align):
    py, pn, pm = p_align * R_YES, p_align * (1 - R_YES), 1 - p_align
    return py * NET_YES + pn * NET_NO + pm * NET_MIX


def report(name, mask, aligned, results):
    """Evaluate one conditioner. mask selects the windows it fires on."""
    n = int(mask.sum())
    if n < 100:
        results.append({"conditioner": name, "n": n, "rate": np.nan,
                        "vs_be": np.nan, "ev": np.nan, "t": np.nan})
        return
    rate = float(aligned[mask].mean())
    # one-sided test against the break-even threshold
    se = np.sqrt(rate * (1 - rate) / n)
    t = (rate - BREAKEVEN) / se if se > 0 else 0.0
    results.append({"conditioner": name, "n": n, "rate": rate,
                    "vs_be": rate - BREAKEVEN, "ev": ev_at(rate), "t": t})


def main():
    px = pd.read_parquet(DATA / "multi_spot.parquet")
    px.index = pd.to_datetime(px.index, unit="s", utc=True)
    px = px.sort_index()

    opens = px.resample("15min").first()
    closes = px.resample("15min").last()
    ok = opens.notna().all(axis=1) & closes.notna().all(axis=1)
    opens, closes = opens[ok], closes[ok]

    up = closes > opens
    n_up = up.sum(axis=1)
    aligned = ((n_up == len(px.columns)) | (n_up == 0)).astype(int)

    # ---- backward-looking features, all shifted so no window sees itself
    df = pd.DataFrame(index=aligned.index)
    df["aligned"] = aligned

    # magnitude of the previous window's BTC move
    btc_ret = (closes["BTC-USD"] / opens["BTC-USD"] - 1)
    df["prev_abs_move"] = btc_ret.abs().shift(1)

    # trailing realized vol of BTC over the last 8 windows (2 hours)
    df["trail_vol"] = btc_ret.rolling(8).std().shift(1)

    # trailing alignment rate over last 8 / 20 windows
    df["trail_align8"] = aligned.rolling(8).mean().shift(1)
    df["trail_align20"] = aligned.rolling(20).mean().shift(1)

    # consecutive aligned run length ending in the PREVIOUS window
    run = np.zeros(len(aligned), dtype=int)
    a = aligned.values
    for i in range(1, len(a)):
        run[i] = run[i - 1] + 1 if a[i - 1] == 1 else 0
    df["run_len"] = run

    # cross-sectional dispersion of the previous window's returns
    prev_rets = (closes / opens - 1).shift(1)
    df["prev_dispersion"] = prev_rets.std(axis=1)

    df = df.dropna()
    al = df.aligned.values.astype(bool)
    results = []

    report("ALL WINDOWS (baseline)", pd.Series(True, index=df.index).values,
           df.aligned, results)

    # run-length conditioners -- the entry rule under test, and extensions
    for k in [2, 3, 4, 5, 6]:
        report(f"run of {k}+ aligned", (df.run_len >= k).values,
               df.aligned, results)

    # volatility regime
    for q, lab in [(0.75, "top 25%"), (0.90, "top 10%")]:
        thr = df.trail_vol.quantile(q)
        report(f"trailing vol {lab}", (df.trail_vol >= thr).values,
               df.aligned, results)
    thr = df.trail_vol.quantile(0.25)
    report("trailing vol bottom 25%", (df.trail_vol <= thr).values,
           df.aligned, results)

    # previous move size
    for q, lab in [(0.75, "top 25%"), (0.90, "top 10%")]:
        thr = df.prev_abs_move.quantile(q)
        report(f"prev move {lab}", (df.prev_abs_move >= thr).values,
               df.aligned, results)

    # trailing alignment rate
    for col, lab in [("trail_align8", "8-window"), ("trail_align20", "20-window")]:
        thr = df[col].quantile(0.75)
        report(f"trailing align {lab} top 25%", (df[col] >= thr).values,
               df.aligned, results)

    # dispersion (low dispersion = moving together)
    thr = df.prev_dispersion.quantile(0.25)
    report("prev dispersion bottom 25%", (df.prev_dispersion <= thr).values,
           df.aligned, results)

    # combinations
    thr_v = df.trail_vol.quantile(0.75)
    report("run 3+ AND high vol",
           ((df.run_len >= 3) & (df.trail_vol >= thr_v)).values,
           df.aligned, results)
    thr_a = df.trail_align8.quantile(0.75)
    report("run 3+ AND high trailing align",
           ((df.run_len >= 3) & (df.trail_align8 >= thr_a)).values,
           df.aligned, results)

    res = pd.DataFrame(results)
    n_tested = len(res) - 1     # exclude the baseline row

    print("=" * 78)
    print(f" CONDITIONERS TESTED  (break-even = {BREAKEVEN:.4f})")
    print("=" * 78)
    print(f"{'conditioner':<32}{'n':>7}{'align':>9}{'vs BE':>9}"
          f"{'EV/$10':>9}{'t':>8}")
    print("-" * 78)
    for _, r in res.iterrows():
        if np.isnan(r["rate"]):
            print(f"{r.conditioner:<32}{r.n:>7}   too few observations")
            continue
        print(f"{r.conditioner:<32}{r.n:>7}{r['rate']:>9.4f}"
              f"{r.vs_be:>+9.4f}{r.ev:>+9.2f}{r.t:>8.2f}")

    print()
    print("=" * 78)
    print(" VERDICT")
    print("=" * 78)
    beat = res[(res.rate > BREAKEVEN)]
    # Sidak-corrected threshold for the number of conditioners tried
    alpha_adj = 1 - (1 - 0.05) ** (1 / max(n_tested, 1))
    req_t = norm_ppf(1 - alpha_adj)
    print(f"  conditioners tested          {n_tested}")
    print(f"  required t after correction  {req_t:.2f}")
    print(f"  any above break-even?        {len(beat)}")
    if len(beat):
        print()
        for _, r in beat.iterrows():
            ok = "SURVIVES" if r.t > req_t else "chance-level, does NOT survive"
            print(f"    {r.conditioner}: t={r.t:.2f} -> {ok}")
    else:
        print("\n  No conditioner reaches break-even. Sizing on 'confidence'")
        print("  cannot help, because no observable identifies a subset of")
        print("  windows where the bet is favourable.")
    print()


if __name__ == "__main__":
    main()
