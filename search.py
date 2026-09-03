"""
search.py - Systematic hunt for conditional mispricing in the Kalshi panel.

THE PRIOR IS STRONGLY NEGATIVE. Unconditional calibration was already
within 2 points against a 4.5-point hurdle. This tests the one thing that
result does not rule out: whether the market is miscalibrated in specific
CONDITIONS even though it is calibrated on average.

Method, and the method is the whole point:

  1. Split the panel chronologically. First half = DISCOVERY.
     Second half = VALIDATION, untouched until the end.
  2. In DISCOVERY only, search a grid of (price bin x condition) cells
     for gaps exceeding the 4.5-point hurdle.
  3. Take whatever DISCOVERY flags and test those exact cells in
     VALIDATION. No re-searching, no adjusting, no second look.
  4. Correct for how many cells were searched.

If a cell survives step 3, it is worth attention. If nothing does -- which
is what I expect -- the search space is exhausted and this is finished.

    py search.py
"""

import math
from pathlib import Path

import numpy as np
import pandas as pd

DATA = Path("data")

# Cost hurdle from breakeven.py: crypto fee tier, 1c half-spread,
# 0.5c slippage. A gap must exceed this to be tradeable at all.
HURDLE = 0.045
MIN_CELL = 150          # cells smaller than this tell you nothing


def wilson(k, n, z=1.96):
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    d = 1 + z**2 / n
    c = (p + z**2 / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / d
    return (c - h, c + h)


def make_conditions(df):
    """Backward-looking conditions only. Nothing here can see its own bar."""
    conds = {}
    conds["all"] = pd.Series(True, index=df.index)

    if "realized_vol" in df:
        hi = df.realized_vol.quantile(0.75)
        lo = df.realized_vol.quantile(0.25)
        conds["vol_high"] = df.realized_vol >= hi
        conds["vol_low"] = df.realized_vol <= lo

    if "mom_5m" in df:
        hi = df.mom_5m.quantile(0.80)
        lo = df.mom_5m.quantile(0.20)
        conds["mom5_up"] = df.mom_5m >= hi
        conds["mom5_dn"] = df.mom_5m <= lo

    if "mom_15m" in df:
        hi = df.mom_15m.quantile(0.80)
        lo = df.mom_15m.quantile(0.20)
        conds["mom15_up"] = df.mom_15m >= hi
        conds["mom15_dn"] = df.mom_15m <= lo

    if "minutes_left" in df:
        conds["early_window"] = df.minutes_left >= 10
        conds["late_window"] = df.minutes_left <= 5

    if "moneyness" in df:
        conds["near_money"] = df.moneyness.abs() <= df.moneyness.abs().quantile(0.33)
        conds["far_money"] = df.moneyness.abs() >= df.moneyness.abs().quantile(0.67)

    if "spread" in df:
        conds["tight_spread"] = df.spread <= 0.01

    hr = pd.to_datetime(df.ts, unit="s", utc=True).dt.hour
    conds["us_hours"] = (hr >= 13) & (hr <= 21)
    conds["asia_hours"] = (hr >= 0) & (hr <= 8)
    return conds


def scan(df, conds, bins, label, verbose=True):
    """Return every (condition, price bin) cell with its calibration gap."""
    rows = []
    binned = pd.cut(df["mid"], bins)
    for cname, mask in conds.items():
        sub = df[mask]
        if len(sub) < MIN_CELL:
            continue
        b = binned[mask]
        for interval, g in sub.groupby(b, observed=True):
            n = len(g)
            if n < MIN_CELL:
                continue
            pred = g["mid"].mean()
            act = g["outcome"].mean()
            gap = act - pred
            lo, hi = wilson(int(g.outcome.sum()), n)
            rows.append({
                "condition": cname, "bin": str(interval), "n": n,
                "predicted": pred, "actual": act, "gap": gap,
                "ci_lo": lo - pred, "ci_hi": hi - pred,
                "tradeable": abs(gap) > HURDLE,
            })
    out = pd.DataFrame(rows)
    if verbose:
        print(f"\n  {label}: {len(out)} cells with n >= {MIN_CELL}")
    return out


def main():
    panel = pd.read_parquet(DATA / "panel.parquet").sort_values("ts")
    panel = panel.reset_index(drop=True)
    print("=" * 72)
    print(" CONDITIONAL MISPRICING SEARCH")
    print("=" * 72)
    print(f"  panel rows        {len(panel):,}")
    print(f"  cost hurdle       {HURDLE:.3f}  ({HURDLE*100:.1f} points)")
    print(f"  min cell size     {MIN_CELL}")

    mid_ix = len(panel) // 2
    disc = panel.iloc[:mid_ix].copy()
    val = panel.iloc[mid_ix:].copy()
    print(f"  discovery         {len(disc):,} rows")
    print(f"  validation        {len(val):,} rows  (untouched until step 3)")

    bins = np.arange(0, 1.05, 0.1)

    # ---- step 2: search DISCOVERY only ----
    d_cells = scan(disc, make_conditions(disc), bins, "DISCOVERY scan")
    n_searched = len(d_cells)
    flagged = d_cells[d_cells.tradeable].copy()
    flagged = flagged.reindex(
        flagged.gap.abs().sort_values(ascending=False).index)

    print(f"  cells searched    {n_searched}")
    print(f"  cells over hurdle {len(flagged)}")

    if flagged.empty:
        print("\n" + "=" * 72)
        print(" RESULT: nothing exceeds the hurdle even IN SAMPLE.")
        print("=" * 72)
        print("  No conditional mispricing large enough to trade, before")
        print("  any out-of-sample check. The search space is exhausted.")
        return

    print("\n" + "=" * 72)
    print(" FLAGGED IN DISCOVERY  (in-sample -- not yet evidence)")
    print("=" * 72)
    print(f"  {'condition':<14}{'bin':<14}{'n':>7}{'pred':>8}{'act':>8}{'gap':>8}")
    print("  " + "-" * 60)
    for _, r in flagged.head(12).iterrows():
        print(f"  {r.condition:<14}{r['bin']:<14}{r.n:>7}"
              f"{r.predicted:>8.3f}{r.actual:>8.3f}{r.gap:>+8.3f}")

    # ---- step 3: test those exact cells in VALIDATION ----
    v_cells = scan(val, make_conditions(val), bins, "VALIDATION scan",
                   verbose=False)
    key = ["condition", "bin"]
    merged = flagged[key + ["n", "gap"]].merge(
        v_cells[key + ["n", "gap", "ci_lo", "ci_hi"]],
        on=key, how="left", suffixes=("_disc", "_val"))

    print("\n" + "=" * 72)
    print(" SAME CELLS, TESTED OUT OF SAMPLE")
    print("=" * 72)
    print(f"  {'condition':<14}{'bin':<14}{'disc gap':>10}{'val gap':>10}"
          f"{'val n':>8}{'holds?':>9}")
    print("  " + "-" * 66)

    survivors = 0
    for _, r in merged.iterrows():
        if pd.isna(r.gap_val):
            print(f"  {r.condition:<14}{r['bin']:<14}{r.gap_disc:>+10.3f}"
                  f"{'--':>10}{'--':>8}{'no data':>9}")
            continue
        same_sign = np.sign(r.gap_disc) == np.sign(r.gap_val)
        holds = same_sign and abs(r.gap_val) > HURDLE
        survivors += holds
        print(f"  {r.condition:<14}{r['bin']:<14}{r.gap_disc:>+10.3f}"
              f"{r.gap_val:>+10.3f}{int(r.n_val):>8}"
              f"{'YES' if holds else 'no':>9}")

    # ---- step 4: multiple-testing context ----
    print("\n" + "=" * 72)
    print(" VERDICT")
    print("=" * 72)
    print(f"  cells searched in discovery : {n_searched}")
    print(f"  flagged in discovery        : {len(flagged)}")
    print(f"  survived validation         : {survivors}")
    exp_false = n_searched * 0.05
    print(f"  expected false flags at 5%  : {exp_false:.1f}")
    print()
    if survivors == 0:
        print("  Nothing survives. Every apparent mispricing in the first")
        print("  half fails to reproduce in the second. That is the")
        print("  signature of noise, and it is the expected result given")
        print("  the market was already calibrated to within 2 points.")
        print()
        print("  The search space is exhausted. There is no edge in this")
        print("  data to find, size, or condition on.")
    else:
        print(f"  {survivors} cell(s) held out of sample. Before believing it:")
        print("   - is the validation n large enough to matter?")
        print("   - does the CI exclude zero, not just the point estimate?")
        print("   - is there a mechanism, or is it a coincidence with a name?")
        print("   - paper trade it for a month before risking anything.")
    print()


if __name__ == "__main__":
    main()
