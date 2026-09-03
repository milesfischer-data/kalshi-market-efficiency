"""
build_panel2.py - CORRECTED panel. Fixes a look-ahead bug in the spot join.

THE BUG
  Coinbase /candles returns buckets where `time` is the bucket START.
  A bar stamped ts=T therefore has close = the price at ~T+60s.
  Kalshi's end_period_ts is the bucket END.

  Joining them directly with merge_asof(direction="backward") matched a
  Kalshi bar ending at T to a Coinbase close from T+60. Every spot-derived
  feature -- moneyness, realized_vol, momentum -- was seeing 60 seconds
  into the future.

THE FIX
  Shift Coinbase timestamps forward by one bucket so ts marks when the
  bar CLOSED, then join backward. Now a Kalshi bar at T sees only spot
  that had already printed by T.

This script writes BOTH panels so you can compare directly:
    data/panel_lookahead.parquet   (the old, contaminated join)
    data/panel.parquet             (corrected)

    py build_panel2.py
"""

from pathlib import Path

import numpy as np
import pandas as pd

DATA = Path("data")
SKIP_OPENING_MIN = 1
SKIP_CLOSING_MIN = 1
GRANULARITY = 60


def spot_features(spot, shift_seconds):
    """
    shift_seconds=0   reproduces the old buggy behaviour
    shift_seconds=60  marks each bar by when it actually closed
    """
    s = spot.sort_values("ts").copy()
    s["ts"] = s["ts"] + shift_seconds
    s = s.rename(columns={"close": "spot"})
    s["logret"] = np.log(s.spot).diff()
    s["realized_vol"] = (s.logret.rolling(30).std().shift(1)
                         * np.sqrt(365 * 24 * 60))
    s["mom_5m"] = s.spot.pct_change(5).shift(1)
    s["mom_15m"] = s.spot.pct_change(15).shift(1)
    return s[["ts", "spot", "realized_vol", "mom_5m", "mom_15m"]]


def build(cand, meta, spot, shift_seconds):
    df = cand.dropna(subset=["bid_close", "ask_close"]).merge(
        meta[["market_ticker", "open_ts", "close_ts", "result", "floor_strike"]],
        on="market_ticker", how="inner")

    df["spread"] = df.ask_close - df.bid_close
    df["mid"] = (df.ask_close + df.bid_close) / 2.0
    df["minute_in_window"] = ((df.ts - df.open_ts) / 60.0).round().astype(int)
    df["minutes_left"] = (df.close_ts - df.ts) / 60.0
    df = df[(df.minute_in_window >= SKIP_OPENING_MIN)
            & (df.minutes_left >= SKIP_CLOSING_MIN)]
    df["outcome"] = (df.result == "yes").astype(int)

    feats = spot_features(spot, shift_seconds)
    df = pd.merge_asof(df.sort_values("ts"), feats.sort_values("ts"),
                       on="ts", direction="backward", tolerance=120)

    df["strike"] = df.floor_strike
    df["moneyness"] = np.log(df.spot / df.strike)
    df = df.dropna(subset=["spot", "realized_vol", "mid", "outcome", "strike"])
    return df.sort_values("ts").reset_index(drop=True)


def main():
    meta = pd.read_parquet(DATA / "market_meta.parquet")
    cand = pd.read_parquet(DATA / "markets.parquet")
    spot = pd.read_parquet(DATA / "spot.parquet")

    print("=" * 70)
    print(" REBUILDING PANEL WITH AND WITHOUT THE LOOK-AHEAD")
    print("=" * 70)

    old = build(cand, meta, spot, shift_seconds=0)
    new = build(cand, meta, spot, shift_seconds=GRANULARITY)

    old.to_parquet(DATA / "panel_lookahead.parquet")
    new.to_parquet(DATA / "panel.parquet")

    print(f"  contaminated panel : {len(old):,} rows -> panel_lookahead.parquet")
    print(f"  corrected panel    : {len(new):,} rows -> panel.parquet")

    print("\n" + "=" * 70)
    print(" HOW MUCH DID moneyness KNOW?")
    print("=" * 70)
    print("  Correlation between |moneyness| and the eventual outcome")
    print("  being 'obvious' (contract already at an extreme price):\n")

    for label, d in [("contaminated", old), ("corrected", new)]:
        # How well does moneyness predict outcome BEYOND what price says?
        resid = d.outcome - d["mid"]
        c = np.corrcoef(d.moneyness, resid)[0, 1]
        print(f"  {label:<14} corr(moneyness, outcome - price) = {c:+.4f}")

    print("\n  In an efficient market this correlation should be near ZERO:")
    print("  the price already contains everything moneyness tells you.")
    print("  A large value means moneyness holds information the price")
    print("  did not have yet -- i.e. information from the future.")
    print()


if __name__ == "__main__":
    main()
