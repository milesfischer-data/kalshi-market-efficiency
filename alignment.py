"""
alignment.py - Do BTC, ETH, SOL, XRP and DOGE finish 15m windows aligned
               more than 66.24% of the time?

That threshold comes from a set of quoted parlay odds ($5 -> $18 / $5 -> $13).
Above it the double-parlay is +EV. Below it, it is -EV. Nothing else about
the strategy matters.

    py -m pip install requests pandas pyarrow
    py alignment.py --days 60

Also tests the "it works at night" claim by bucketing on UTC hour, and
tests the "wait for 3 aligned cycles" rule directly.
"""

import argparse
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import requests

COINBASE = "https://api.exchange.coinbase.com"
PRODUCTS = ["BTC-USD", "ETH-USD", "SOL-USD", "XRP-USD", "DOGE-USD"]

# Break-even implied by $5->$18 (all yes) and $5->$13 (all no)
BREAKEVEN = 5 / 18 + 5 / 13

DATA = Path("data")
DATA.mkdir(exist_ok=True)


def _get(url, params=None, tries=5):
    for a in range(tries):
        try:
            r = requests.get(url, params=params, timeout=30)
        except requests.RequestException:
            time.sleep(2 ** a)
            continue
        if r.status_code == 200:
            return r.json()
        if r.status_code in (429, 500, 502, 503, 504):
            time.sleep(2 ** a)
            continue
        raise RuntimeError(f"HTTP {r.status_code}: {r.text[:200]}")
    raise RuntimeError("gave up: " + url)


def candles(product, start, end, granularity=60):
    rows, cur = [], start
    step = timedelta(seconds=granularity * 300)
    while cur < end:
        stop = min(cur + step, end)
        for t, lo, hi, op, cl, vol in _get(
                f"{COINBASE}/products/{product}/candles",
                {"start": cur.isoformat(), "end": stop.isoformat(),
                 "granularity": granularity}):
            rows.append({"ts": t, "close": cl})
        cur = stop
        time.sleep(0.22)
    df = pd.DataFrame(rows).drop_duplicates("ts").sort_values("ts")
    return df.set_index("ts")["close"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--days", type=int, default=60)
    args = ap.parse_args()

    end = datetime.now(timezone.utc)
    start = end - timedelta(days=args.days)

    print(f"\nPulling {args.days}d of 1m candles for {len(PRODUCTS)} assets")
    series = {}
    for p in PRODUCTS:
        print(f"  {p}...", end=" ", flush=True)
        series[p] = candles(p, start, end)
        print(f"{len(series[p]):,} bars")

    px = pd.DataFrame(series).dropna()
    px.to_parquet(DATA / "multi_spot.parquet")

    # ---- build 15-minute windows on :00 :15 :30 :45 like Kalshi ----
    px.index = pd.to_datetime(px.index, unit="s", utc=True)
    px = px.sort_index()

    # Window open price and close price for each aligned 15m block
    opens = px.resample("15min").first()
    closes = px.resample("15min").last()
    ok = opens.notna().all(axis=1) & closes.notna().all(axis=1)
    opens, closes = opens[ok], closes[ok]

    up = (closes > opens)          # True = finished UP for that asset
    n_up = up.sum(axis=1)
    aligned = (n_up == len(PRODUCTS)) | (n_up == 0)

    print("\n" + "=" * 64)
    print(" ALIGNMENT RATE")
    print("=" * 64)
    print(f"  windows measured        {len(aligned):,}")
    print(f"  all five UP             {(n_up == 5).mean():.4f}")
    print(f"  all five DOWN           {(n_up == 0).mean():.4f}")
    print(f"  ALIGNED (either way)    {aligned.mean():.4f}")
    print(f"  break-even needed       {BREAKEVEN:.4f}")
    gap = aligned.mean() - BREAKEVEN
    print(f"  gap                     {gap:+.4f}  "
          f"({'PROFITABLE' if gap > 0 else 'NOT PROFITABLE'})")
    print()
    print("  distribution of how many finished UP:")
    print(n_up.value_counts().sort_index().to_string())

    # ---- by hour of day: the "at night" claim ----
    print("\n" + "=" * 64)
    print(" BY UTC HOUR  (testing the overnight theory)")
    print("=" * 64)
    byhr = pd.DataFrame({"aligned": aligned.astype(float),
                         "hour": aligned.index.hour})
    h = byhr.groupby("hour").aligned.agg(["mean", "size"])
    h["vs_breakeven"] = h["mean"] - BREAKEVEN
    h = h.rename(columns={"mean": "align_rate", "size": "n"})
    print(h.to_string(float_format=lambda x: f"{x:.4f}"))
    best = h.align_rate.idxmax()
    print(f"\n  best hour: {best:02d}:00 UTC at {h.align_rate.max():.4f}")
    print(f"  hours above break-even: {(h.align_rate > BREAKEVEN).sum()} of 24")

    # ---- the "wait for 3 aligned cycles" rule ----
    print("\n" + "=" * 64)
    print(" DOES A RUN OF 3 ALIGNED CYCLES PREDICT THE NEXT ONE?")
    print("=" * 64)
    a = aligned.values
    run3 = np.array([a[i-3:i].all() for i in range(3, len(a))])
    nxt = a[3:]
    base = nxt.mean()
    after = nxt[run3].mean() if run3.sum() else float("nan")
    print(f"  base rate, any window            {base:.4f}")
    print(f"  after 3 aligned in a row         {after:.4f}   (n={run3.sum():,})")
    print(f"  difference                       {after - base:+.4f}")
    print(f"  break-even needed                {BREAKEVEN:.4f}")
    print()
    if after > BREAKEVEN:
        print("  -> The entry rule clears the bar. Worth a closer look.")
    else:
        print("  -> The entry rule does NOT clear the bar.")

    # ---- EV of the actual strategy ----
    print("\n" + "=" * 64)
    print(" EV OF THE STRATEGY AS DESCRIBED ($10 staked per cycle)")
    print("=" * 64)
    for label, p_al in [("every window", base), ("after 3 aligned", after)]:
        # split aligned into up/down in the same ratio the odds implied
        r_yes = (5 / 18) / BREAKEVEN
        py, pn, pm = p_al * r_yes, p_al * (1 - r_yes), 1 - p_al
        ev = py * 8 + pn * 3 + pm * (-10)
        print(f"  {label:<18} P(align)={p_al:.4f}  EV = ${ev:+.3f} per $10")

    print()


if __name__ == "__main__":
    main()
