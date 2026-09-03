"""
oos_runlength.py - Does the run-length gradient hold OUT OF SAMPLE?

The in-sample gradient was clean and monotonic:
    run 2+ 56.6% | 3+ 57.9% | 4+ 60.1% | 5+ 62.4% | 6+ 66.0%
terminating a hair under the 66.24% break-even.

The question is whether that is a real structural feature or a pattern
that happens to fit these particular 60 days. The only honest way to tell
is to split the data chronologically and check whether the SECOND half
reproduces what the FIRST half shows.

A real effect appears in both halves. An artefact appears in one.

    py oos_runlength.py

No scipy. No fitting. No parameter selection.
"""

import math
from pathlib import Path

import numpy as np
import pandas as pd

DATA = Path("data")
BREAKEVEN = 5 / 18 + 5 / 13          # 0.6624
R_YES = (5 / 18) / BREAKEVEN
NET_YES, NET_NO, NET_MIX = 8.0, 3.0, -10.0

# Kalshi parlay fees are charged on the combined contract. This is an
# ESTIMATE -- confirm the real number in the order ticket before trusting
# any of the net-of-fee figures below.
PARLAY_FEE_PCT = 0.02


def ev_at(p, fee_pct=0.0):
    py, pn, pm = p * R_YES, p * (1 - R_YES), 1 - p
    gross = py * NET_YES + pn * NET_NO + pm * NET_MIX
    return gross - 10.0 * fee_pct


def wilson(k, n, z=1.96):
    """Wilson score interval -- honest error bars for a proportion."""
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    d = 1 + z**2 / n
    c = (p + z**2 / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / d
    return (c - h, c + h)


def build(px):
    px = px.copy()
    px.index = pd.to_datetime(px.index, unit="s", utc=True)
    px = px.sort_index()
    opens = px.resample("15min").first()
    closes = px.resample("15min").last()
    ok = opens.notna().all(axis=1) & closes.notna().all(axis=1)
    opens, closes = opens[ok], closes[ok]
    n_up = (closes > opens).sum(axis=1)
    aligned = ((n_up == len(px.columns)) | (n_up == 0)).astype(int)

    a = aligned.values
    run = np.zeros(len(a), dtype=int)
    for i in range(1, len(a)):
        run[i] = run[i - 1] + 1 if a[i - 1] == 1 else 0
    return pd.DataFrame({"aligned": a, "run_len": run}, index=aligned.index)


def table(df, label, runs):
    print(f"\n  {label}   ({len(df):,} windows, "
          f"{df.index.min().date()} to {df.index.max().date()})")
    print(f"  {'run':>5} {'n':>7} {'align':>8} {'95% CI':>18} {'vs BE':>9}")
    print("  " + "-" * 52)
    out = {}
    for k in runs:
        m = df.run_len >= k
        n = int(m.sum())
        if n < 30:
            print(f"  {k:>4}+ {n:>7}   too few to say anything")
            out[k] = None
            continue
        hits = int(df.aligned[m].sum())
        p = hits / n
        lo, hi = wilson(hits, n)
        out[k] = p
        print(f"  {k:>4}+ {n:>7} {p:>8.4f}  [{lo:.4f}, {hi:.4f}] "
              f"{p - BREAKEVEN:>+9.4f}")
    return out


def main():
    px = pd.read_parquet(DATA / "multi_spot.parquet")
    df = build(px)
    runs = [2, 3, 4, 5, 6, 7, 8, 9, 10]

    mid = len(df) // 2
    first, second = df.iloc[:mid], df.iloc[mid:]

    print("=" * 60)
    print(" OUT-OF-SAMPLE TEST: run-length gradient")
    print("=" * 60)
    print(f" break-even = {BREAKEVEN:.4f}")
    print(" A real effect shows the SAME shape in both halves.")

    a = table(first, "FIRST HALF", runs)
    b = table(second, "SECOND HALF", runs)
    c = table(df, "FULL PERIOD (in-sample, for reference)", runs)

    print("\n" + "=" * 60)
    print(" DO THE TWO HALVES AGREE?")
    print("=" * 60)
    print(f"  {'run':>5} {'first':>9} {'second':>9} {'diff':>9} {'both>BE?':>10}")
    print("  " + "-" * 46)
    agree = 0
    checked = 0
    for k in runs:
        if a.get(k) is None or b.get(k) is None:
            continue
        checked += 1
        both = (a[k] > BREAKEVEN) and (b[k] > BREAKEVEN)
        agree += both
        print(f"  {k:>4}+ {a[k]:>9.4f} {b[k]:>9.4f} {a[k]-b[k]:>+9.4f} "
              f"{'YES' if both else 'no':>10}")

    print("\n" + "=" * 60)
    print(" MONOTONICITY CHECK")
    print("=" * 60)
    for lab, d in [("first half", a), ("second half", b), ("full", c)]:
        vals = [d[k] for k in runs if d.get(k) is not None]
        rises = sum(1 for i in range(1, len(vals)) if vals[i] > vals[i-1])
        print(f"  {lab:<12} {rises}/{len(vals)-1} steps increase"
              f"   values: {[f'{v:.3f}' for v in vals]}")
    print("\n  A gradient that is monotonic in ONE half and not the other")
    print("  is not a gradient. It is a coincidence you have named.")

    print("\n" + "=" * 60)
    print(" NET OF ESTIMATED PARLAY FEES  (full period)")
    print("=" * 60)
    print(f"  assuming {PARLAY_FEE_PCT*100:.1f}% on a $10 stake\n")
    print(f"  {'run':>5} {'n':>7} {'per day':>9} {'EV gross':>10} {'EV net':>9}")
    print("  " + "-" * 45)
    days = (df.index.max() - df.index.min()).days or 1
    for k in runs:
        if c.get(k) is None:
            continue
        n = int((df.run_len >= k).sum())
        print(f"  {k:>4}+ {n:>7} {n/days:>9.2f} "
              f"{ev_at(c[k]):>+10.3f} {ev_at(c[k], PARLAY_FEE_PCT):>+9.3f}")

    print("\n" + "=" * 60)
    print(" VERDICT")
    print("=" * 60)
    if checked and agree:
        print(f"  {agree} of {checked} run lengths clear break-even in BOTH halves.")
        print("  Worth a longer data pull and out-of-sample paper trading.")
    else:
        print("  NO run length clears break-even in both halves.")
        print("  The in-sample gradient does not reproduce out of sample.")
        print("  There is no edge here to size, condition, or scale.")
    print()


if __name__ == "__main__":
    main()
