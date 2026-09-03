"""
collect.py - Pull Kalshi KXBTC15M market data + matched Coinbase spot.

Schema confirmed against the live API 2026-09-03:
  price / yes_bid / yes_ask each contain *_dollars fields as STRINGS
  volume_fp and open_interest_fp are also strings

    py collect.py --series KXBTC15M --days 3 --max-markets 20   (test)
    py collect.py --series KXBTC15M --days 60                   (full)

Writes: data/market_meta.parquet, data/markets.parquet, data/spot.parquet
"""

import argparse
import json
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd
import requests

KALSHI = "https://api.elections.kalshi.com/trade-api/v2"
COINBASE = "https://api.exchange.coinbase.com"

DATA = Path("data")
DATA.mkdir(exist_ok=True)


def _f(d, key):
    """Pull a *_dollars string out of a nested block and float it."""
    if not d:
        return None
    v = d.get(key)
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def _get(url, params=None, tries=5):
    for attempt in range(tries):
        try:
            r = requests.get(url, params=params, timeout=30)
        except requests.RequestException as e:
            print(f"\n  network hiccup ({e}), retrying...")
            time.sleep(2 ** attempt)
            continue
        if r.status_code == 200:
            return r.json()
        if r.status_code in (429, 500, 502, 503, 504):
            time.sleep(2 ** attempt)
            continue
        raise RuntimeError(f"HTTP {r.status_code} from {url}\n{r.text[:400]}")
    raise RuntimeError(f"gave up after {tries} tries: {url}")


def list_markets(series_ticker, min_ts, max_ts):
    out, cursor, pages = [], None, 0
    while True:
        params = {"series_ticker": series_ticker, "status": "settled",
                  "limit": 1000, "min_close_ts": int(min_ts),
                  "max_close_ts": int(max_ts)}
        if cursor:
            params["cursor"] = cursor
        page = _get(f"{KALSHI}/markets", params)
        batch = page.get("markets", [])
        out.extend(batch)
        pages += 1
        print(f"  page {pages}: {len(out):,} markets so far", end="\r")
        cursor = page.get("cursor")
        if not cursor or not batch:
            break
        time.sleep(0.15)
    print()
    return out


def market_candles(series_ticker, ticker, start_ts, end_ts):
    """
    1-minute bars. We keep bid AND ask separately, not just the mid,
    because the spread is the thing that decides whether any of this is
    tradeable. At window open the book is nearly empty and the spread is
    enormous; it converges within the first minute.
    """
    url = f"{KALSHI}/series/{series_ticker}/markets/{ticker}/candlesticks"
    js = _get(url, {"start_ts": int(start_ts), "end_ts": int(end_ts),
                    "period_interval": 1})
    rows = []
    for c in js.get("candlesticks", []):
        px, bid, ask = c.get("price"), c.get("yes_bid"), c.get("yes_ask")
        try:
            vol = float(c.get("volume_fp", 0) or 0)
        except (TypeError, ValueError):
            vol = 0.0
        try:
            oi = float(c.get("open_interest_fp", 0) or 0)
        except (TypeError, ValueError):
            oi = 0.0
        rows.append({
            "market_ticker": ticker,
            "ts": c.get("end_period_ts"),
            "px_close": _f(px, "close_dollars"),
            "px_open": _f(px, "open_dollars"),
            "px_mean": _f(px, "mean_dollars"),
            "bid_close": _f(bid, "close_dollars"),
            "bid_open": _f(bid, "open_dollars"),
            "ask_close": _f(ask, "close_dollars"),
            "ask_open": _f(ask, "open_dollars"),
            "volume": vol,
            "open_interest": oi,
        })
    return rows


def coinbase_candles(product, start, end, granularity=60):
    rows, cur = [], start
    step = timedelta(seconds=granularity * 300)
    total = int((end - start) / step) + 1
    i = 0
    while cur < end:
        stop = min(cur + step, end)
        js = _get(f"{COINBASE}/products/{product}/candles",
                  {"start": cur.isoformat(), "end": stop.isoformat(),
                   "granularity": granularity})
        for t, lo, hi, op, cl, vol in js:
            rows.append({"ts": t, "low": lo, "high": hi,
                         "open": op, "close": cl, "volume": vol})
        i += 1
        print(f"  chunk {i}/{total}  ({len(rows):,} bars)", end="\r")
        cur = stop
        time.sleep(0.25)
    print()
    return pd.DataFrame(rows).drop_duplicates("ts").sort_values("ts")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--series", default="KXBTC15M")
    ap.add_argument("--product", default="BTC-USD")
    ap.add_argument("--days", type=int, default=60)
    ap.add_argument("--max-markets", type=int, default=0)
    args = ap.parse_args()

    end = datetime.now(timezone.utc)
    start = end - timedelta(days=args.days)

    print(f"\n[1/3] Markets for {args.series} ({args.days}d back)")
    markets = list_markets(args.series, start.timestamp(), end.timestamp())
    if not markets:
        print("\n  *** NOTHING RETURNED *** check the ticker")
        return
    (DATA / "sample_market.json").write_text(json.dumps(markets[0], indent=2))
    print(f"  {len(markets):,} settled markets")

    meta = pd.DataFrame([{
        "market_ticker": m.get("ticker"),
        "open_ts": pd.Timestamp(m["open_time"]).timestamp() if m.get("open_time") else None,
        "close_ts": pd.Timestamp(m["close_time"]).timestamp() if m.get("close_time") else None,
        "result": m.get("result"),
        "floor_strike": m.get("floor_strike"),
        "title": m.get("title"),
    } for m in markets]).dropna(subset=["open_ts", "close_ts"]).reset_index(drop=True)
    meta.to_parquet(DATA / "market_meta.parquet")
    print(f"  results: {meta.result.value_counts().to_dict()}")

    print(f"\n[2/3] Coinbase spot for {args.product}")
    spot = coinbase_candles(args.product, start, end)
    spot.to_parquet(DATA / "spot.parquet")
    print(f"  {len(spot):,} spot bars")

    subset = meta.head(args.max_markets) if args.max_markets else meta
    print(f"\n[3/3] Candlesticks for {len(subset):,} markets")
    all_rows, failures = [], 0
    for i, row in subset.iterrows():
        try:
            all_rows += market_candles(args.series, row.market_ticker,
                                       row.open_ts, row.close_ts)
        except Exception as e:
            failures += 1
            if failures <= 3:
                print(f"\n  skip {row.market_ticker}: {e}")
        if i % 20 == 0:
            print(f"  {i}/{len(subset)}  ({len(all_rows):,} bars)", end="\r")
        time.sleep(0.12)

    df = pd.DataFrame(all_rows)
    df.to_parquet(DATA / "markets.parquet")
    got = df.px_close.notna().sum() if len(df) else 0
    print(f"\n  {len(df):,} bars, {got:,} with prices, {failures} failures")
    print("\nDone. Next: py build_panel2.py")


if __name__ == "__main__":
    main()
