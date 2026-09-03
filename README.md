# Are Kalshi's 15-Minute Crypto Markets Efficient?

An empirical study of 5,681 settled prediction-market contracts, testing whether
short-horizon Bitcoin markets on Kalshi contain exploitable mispricing.

**Answer: no.** The market is calibrated to within 2 percentage points against a
4.5-point transaction cost hurdle. Four separate hypotheses were tested and
rejected. Two look-ahead bugs were found and corrected during the analysis —
both of which had produced apparently profitable results.

---

## Data

| | |
|---|---|
| Source | Kalshi public API (`KXBTC15M`), Coinbase Exchange API |
| Period | 2026-07-05 to 2026-09-03 (60 days) |
| Markets | 5,681 settled 15-minute contracts |
| Contract bars | 85,119 minute-level bid/ask observations |
| Analysis panel | 79,342 rows after filtering |
| Spot data | 86,399 1-minute BTC bars, plus ETH/SOL/XRP/DOGE |

## The cost hurdle

Kalshi's taker fee is `ceil(0.07 × contracts × P × (1−P))` per contract, peaking
at 1.75¢ on a 50¢ contract. Crypto series carry a higher multiplier. Adding a
measured 1¢ half-spread and 0.5¢ slippage:

| contract price | edge required to break even |
|---|---|
| $0.10 | 3.5 pts |
| $0.50 | **4.5 pts** |
| $0.90 | 2.5 pts |

A market quoting 50% must actually be 54.5% before a trade is profitable.
Professional market makers operate on 1–3 point edges and survive only because
they earn the spread rather than paying it.

## Result 1 — the market is calibrated

Binned all 79,342 observations by quoted price and compared to realised outcomes.

| quoted price | n | predicted | actual | gap (pts) |
|---|---|---|---|---|
| 0.0–0.1 | 11,859 | 0.033 | 0.032 | −0.08 |
| 0.2–0.3 | 6,353 | 0.251 | 0.242 | −0.88 |
| 0.4–0.5 | 8,504 | 0.451 | 0.447 | −0.44 |
| 0.6–0.7 | 7,355 | 0.648 | 0.663 | +1.47 |
| 0.8–0.9 | 5,366 | 0.850 | 0.870 | **+1.99** |

Largest deviation anywhere: 2.0 points. Required: 4.5. Base rate 0.5012 against
a mean quoted price of 0.4982.

## Result 2 — cross-asset alignment falls short

Tested a five-asset parlay strategy (BTC/ETH/SOL/XRP/DOGE finishing the same
direction), where quoted odds implied a **66.24%** break-even alignment rate.

Measured over 5,761 windows: **54.19%**. Expected value −$1.82 per $10 staked.
Zero of 24 hourly buckets cleared the threshold.

## Result 3 — a real effect, too small to trade

Alignment showed a clean monotonic gradient by consecutive-run length:

| run length | alignment | n |
|---|---|---|
| 2+ | 56.6% | 1,727 |
| 3+ | 57.9% | 978 |
| 4+ | 60.1% | 566 |
| 5+ | 62.4% | 340 |
| 6+ | 66.0% | 212 |

Volatility clustering is real and this is a genuine effect. But a chronological
split-half test showed the gradient present in the second half only — flat at
~55.6% through run 5 in the first half. Zero of seven run lengths cleared
break-even in **both** halves.

## Result 4 — no conditional mispricing

Searched 37 (price bin × condition) cells for calibration gaps exceeding the
hurdle, using a strict discovery/validation protocol: trade direction fixed on
the first half, applied blind to the second, standard errors clustered by market.

**Survivors: 0.** Expected by chance at α=0.05: 1.9.

## The bugs (the interesting part)

Both bugs produced *profitable-looking* results. Neither would have been caught
by a framework that only checked whether results looked good.

**Bug 1 — direction selected on realised outcomes.** The validation script chose
which side to trade from the sign of the realised gap *in the rows being
evaluated*, guaranteeing positive EV. It returned 13 profitable cells out of 13
tested — and that unanimity was the giveaway.

**Bug 2 — 60-second look-ahead in the spot join.** Coinbase candle timestamps
mark bucket *start*; Kalshi's mark bucket *end*. A backward `merge_asof` on raw
timestamps matched each contract bar to a spot close from 60 seconds in its
future, contaminating every spot-derived feature.

Diagnostic — correlation between `moneyness` and pricing error:

| | corr |
|---|---|
| contaminated | +0.0813 |
| corrected | **+0.0136** |

Apparent mispricings of +14.8 points collapsed to +0.2 once corrected. They were
never in the market.

## Method notes

- Chronological train/test splits; validation data untouched during discovery
- Standard errors clustered by market (bars within a market share one outcome)
- Šidák correction applied for the number of hypotheses tested
- Wilson score intervals rather than normal approximations on proportions
- All features strictly backward-looking and explicitly lagged
- Harness validated against synthetic data with planted edges of known size:
  correctly detected 10-point edges, correctly rejected 0- and 2-point edges,
  and correctly declined to certify a real 5-point edge at n=6,000

## Files

| file | purpose |
|---|---|
| `collect.py` | Pull Kalshi markets + candlesticks, Coinbase spot |
| `build_panel2.py` | Join into analysis panel (timestamp-corrected) |
| `breakeven_standalone.py` | Transaction cost hurdle, no dependencies |
| `alignment.py` | Five-asset alignment rate measurement |
| `conditioners.py` | Conditional alignment search |
| `oos_runlength.py` | Split-half test of the run-length gradient |
| `search.py` | Conditional mispricing grid search |
| `scrutinize2.py` | Out-of-sample validation with locked direction |

```
pip install requests pandas pyarrow numpy
python collect.py --series KXBTC15M --days 60
python build_panel2.py
python scrutinize2.py
```

## Takeaway

The most useful output of this project was not a strategy. It was a framework
that could say *no* — and that caught two errors which had produced convincing
false positives, including one that survived out-of-sample testing.

Errors producing disappointing results get caught quickly, because you go
looking for them. Errors producing exciting results feel like discoveries. That
asymmetry is why the second bug survived as long as it did.
