"""
breakeven_standalone.py - How much edge does a strategy need to be profitable?

Self-contained. No other files needed, no libraries beyond what ships with
Python. Just: py breakeven_standalone.py

This is pure arithmetic on Kalshi's published fee schedule plus a spread
assumption. Nothing is simulated, fitted, or assumed about markets.
"""

import math


def taker_fee(price, contracts=1, multiplier=0.07):
    """
    Kalshi taker fee: round_up(multiplier * C * P * (1-P)) per contract,
    rounded up to the cent. price in DOLLARS (0.50, not 50).
    """
    raw = multiplier * contracts * price * (1.0 - price)
    return math.ceil(round(raw * 100.0, 9)) / 100.0


def breakeven_edge(price, half_spread=0.01, slippage=0.005, multiplier=0.07):
    """
    You buy YES at `price`. Expected value is zero when:

        p*(1 - entry) - (1-p)*entry - fee = 0   =>   p = entry + fee

    So the true probability must exceed the quoted mid by exactly
    (half_spread + slippage + fee). That gap is your hurdle.
    """
    entry = price + half_spread + slippage
    fee = taker_fee(entry, 1, multiplier)
    return (entry + fee) - price


def main():
    print()
    print("=" * 64)
    print(" EDGE REQUIRED TO BREAK EVEN")
    print(" (percentage points of true probability, above market price)")
    print("=" * 64)
    print(" assumes 1c half-spread crossed on entry, 0.5c adverse slippage")
    print()
    print(f"  {'contract price':>16}  {'standard fees':>14}  {'crypto fees':>12}")
    print(f"  {'-'*16}  {'-'*14}  {'-'*12}")

    for price in [0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90]:
        std = breakeven_edge(price, multiplier=0.07) * 100
        cry = breakeven_edge(price, multiplier=0.10) * 100
        print(f"  {'$%.2f' % price:>16}  {std:>12.2f} pts  {cry:>10.2f} pts")

    e50 = breakeven_edge(0.50, multiplier=0.10)
    print()
    print("=" * 64)
    print(" WHAT THIS MEANS")
    print("=" * 64)
    print(f" At a 50c contract on crypto fees, you must judge the true")
    print(f" probability better than the market by {e50*100:.1f} points.")
    print()
    print(f" A market quoting 50% must ACTUALLY be {50 + e50*100:.1f}% for you")
    print(f" to break even. Not profit. Break even.")
    print()
    print(" For scale: professional prediction-market makers run on edges")
    print(" of 1-3 points, and survive only because they EARN the spread")
    print(" instead of paying it, at roughly a quarter the fee rate.")
    print()
    print("=" * 64)
    print(" THE COST OF A ZERO-EDGE STRATEGY")
    print("=" * 64)
    cost = breakeven_edge(0.50, multiplier=0.10)
    for trades_per_day in [10, 30, 100]:
        daily = cost * trades_per_day
        print(f"  {trades_per_day:>3} trades/day at $1/contract "
              f"-> -${daily:>5.2f}/day  (-${daily*30:>7.2f}/month)")
    print()
    print(" That is what a coin-flip strategy costs you. Not bad luck.")
    print(" Just friction, compounding every single day.")
    print()


if __name__ == "__main__":
    main()
