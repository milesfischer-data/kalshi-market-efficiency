"""
scrutinize2.py - CORRECTED version. The previous script had look-ahead bias.

THE BUG: trade direction was chosen from the sign of the realized gap in
the very rows being evaluated. That picks the winning side after seeing
who won, so every cell returned positive EV by construction. All 13 of 13
"survived", which is itself the giveaway -- real edges do not go 13 for 13.

THE FIX: direction is locked in using the DISCOVERY half only, then
applied blind to the VALIDATION half. If a cell looked underpriced in
July, we buy YES in August and see what happens. No peeking.

    py scrutinize2.py
"""

import math
from pathlib import Path

import numpy as np
import pandas as pd

DATA = Path("data")
FEE_MULT = 0.10          # crypto tier; confirm in the order ticket
MIN_N = 100


def taker_fee(p):
    return math.ceil(round(FEE_MULT * p * (1 - p) * 100, 9)) / 100


def conditions(df):
    am = df.moneyness.abs()
    return {
        "near_money": am <= am.quantile(0.33),
        "far_money": am >= am.quantile(0.67),
        "vol_low": df.realized_vol <= df.realized_vol.quantile(0.25),
        "vol_high": df.realized_vol >= df.realized_vol.quantile(0.75),
    }


def trade_ev(g, side_yes):
    """
    Actually place the trade. side_yes is FIXED in advance -- it does not
    look at this data's outcomes.

    Buying YES costs the ask. Buying NO costs (1 - bid).
    """
    if side_yes:
        entry = g.ask_close.clip(0.01, 0.99)
        won = g.outcome == 1
    else:
        entry = (1.0 - g.bid_close).clip(0.01, 0.99)
        won = g.outcome == 0
    fees = entry.map(taker_fee)
    pnl = np.where(won, 1.0 - entry, -entry) - fees
    return pd.Series(pnl, index=g.index)


def main():
    p = pd.read_parquet(DATA / "panel.parquet").sort_values("ts").reset_index(drop=True)
    bins = np.arange(0, 1.05, 0.1)

    mid_ix = len(p) // 2
    disc = p.iloc[:mid_ix].copy()
    val = p.iloc[mid_ix:].copy()

    print("=" * 84)
    print(" CORRECTED TEST: direction locked from DISCOVERY, applied to VALIDATION")
    print("=" * 84)
    print(f"  discovery  {len(disc):,} rows  "
          f"({pd.to_datetime(disc.ts.min(), unit='s').date()} onward)")
    print(f"  validation {len(val):,} rows  "
          f"({pd.to_datetime(val.ts.min(), unit='s').date()} onward)")
    print()

    d_conds, v_conds = conditions(disc), conditions(val)
    disc["_bin"] = pd.cut(disc["mid"], bins).astype(str)
    val["_bin"] = pd.cut(val["mid"], bins).astype(str)

    print(f"{'condition':<12}{'bin':<12}{'disc gap':>10}{'side':>6}"
          f"{'val n':>7}{'val mkts':>9}{'val EV/$':>10}{'t':>8}{'':>9}")
    print("-" * 84)

    results = []
    for cname in d_conds:
        for cbin in sorted(disc._bin.unique()):
            dm = d_conds[cname] & (disc._bin == cbin)
            gd = disc[dm]
            if len(gd) < MIN_N:
                continue

            # ---- DIRECTION DECIDED HERE, ON DISCOVERY ONLY ----
            disc_gap = (gd.outcome - gd["mid"]).mean()
            side_yes = disc_gap > 0

            vm = v_conds[cname] & (val._bin == cbin)
            gv = val[vm]
            if len(gv) < MIN_N:
                continue

            pnl = trade_ev(gv, side_yes)
            ev = pnl.mean()

            # cluster by market
            per_mkt = pnl.groupby(gv.market_ticker).mean()
            se = per_mkt.std(ddof=1) / math.sqrt(len(per_mkt)) if len(per_mkt) > 1 else np.nan
            t = ev / se if se and se > 0 else 0.0

            results.append({"cond": cname, "bin": cbin, "disc_gap": disc_gap,
                            "side": "YES" if side_yes else "NO",
                            "n": len(gv), "mkts": gv.market_ticker.nunique(),
                            "ev": ev, "t": t})

            flag = "PROFITABLE" if (ev > 0 and t > 2.5) else ""
            print(f"{cname:<12}{cbin:<12}{disc_gap:>+10.3f}"
                  f"{('YES' if side_yes else 'NO'):>6}{len(gv):>7}"
                  f"{gv.market_ticker.nunique():>9}{ev:>+10.4f}{t:>8.2f}"
                  f"{flag:>12}")

    res = pd.DataFrame(results)
    alive = res[(res.ev > 0) & (res.t > 2.5)]

    print("\n" + "=" * 84)
    print(" VERDICT")
    print("=" * 84)
    print(f"  cells tested out of sample : {len(res)}")
    print(f"  profitable with t > 2.5    : {len(alive)}")
    print(f"  expected by chance at 5%   : {len(res) * 0.05:.1f}")
    if len(alive) == 0:
        print("\n  Nothing survives once direction is chosen honestly.")
        print("  The earlier 13-for-13 result was look-ahead bias.")
    elif len(alive) <= len(res) * 0.05:
        print("\n  Survivors are at or below the chance rate. Not evidence.")
    else:
        print("\n  More survivors than chance predicts. Worth a longer pull")
        print("  and paper trading. NOT worth funding an account.")
        print(f"\n  mean EV across survivors: ${alive.ev.mean():+.4f}/contract")
    print()


if __name__ == "__main__":
    main()
