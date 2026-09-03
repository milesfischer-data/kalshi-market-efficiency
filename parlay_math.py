"""
parlay_math.py - What actually happens when you bet both sides of a parlay.

Self-contained. No data, no libraries beyond the standard library.
    py parlay_math.py

Two readings of the strategy, both analysed:

  A) Two parlays: one "all five YES", one "all five NO".
  B) Both sides of each individual market (a true hedge).
"""

import math
import random

random.seed(20260903)

ASSETS = ["BTC", "ETH", "SOL", "XRP", "DOGE"]
N = 5
TRIALS = 200_000


def simulate_alignment(rho, trials=TRIALS):
    """
    One-factor model of direction. Each asset's move is
        common * sqrt(rho) + idiosyncratic * sqrt(1-rho)
    and we take the sign. rho is the pairwise correlation of returns.
    Returns P(all five finish the same direction).
    """
    hits = 0
    for _ in range(trials):
        common = random.gauss(0, 1)
        signs = []
        for _ in range(N):
            idio = random.gauss(0, 1)
            val = common * math.sqrt(rho) + idio * math.sqrt(1 - rho)
            signs.append(val > 0)
        if all(signs) or not any(signs):
            hits += 1
    return hits / trials


def parlay_fair_odds(p_all_same):
    """
    A parlay paying only when all five align. If P(all yes) = P(all no)
    = p_all_same / 2, then a FAIRLY priced all-yes parlay costs that
    probability and pays $1.
    """
    return p_all_same / 2.0


def strategy_a(rho, fee_rate=0.02, trials=TRIALS):
    """
    Buy BOTH the all-YES parlay and the all-NO parlay, $1 each.

    You win on exactly 2 of the 2^5 = 32 possible outcomes. On the other
    30, BOTH tickets are worthless and you lose your entire stake.
    """
    p_same = simulate_alignment(rho, trials)
    p_yes = p_no = p_same / 2.0

    # Fair prices (no house edge at all -- generous to the strategy)
    cost_yes = p_yes
    cost_no = p_no
    total_cost = (cost_yes + cost_no) * (1 + fee_rate)

    # Payout: exactly one ticket pays $1 when all align, else nothing.
    expected_payout = p_same * 1.0

    ev = expected_payout - total_cost
    return {
        "rho": rho,
        "p_all_aligned": p_same,
        "p_lose_everything": 1 - p_same,
        "cost": total_cost,
        "expected_payout": expected_payout,
        "ev": ev,
        "ev_pct": ev / total_cost * 100 if total_cost else 0,
    }


def strategy_b(price=0.50, fee_mult=0.10):
    """
    Buy BOTH yes and no on the SAME market, 1 contract each.

    YES costs p, NO costs (1-p). Together they cost $1.00 and pay
    exactly $1.00 no matter what happens. Guaranteed outcome: you lose
    precisely the fees, every single time, with certainty.
    """
    def taker_fee(px):
        raw = fee_mult * px * (1 - px)
        return math.ceil(round(raw * 100, 9)) / 100

    cost = price + (1 - price) + taker_fee(price) + taker_fee(1 - price)
    payout = 1.00
    return {"cost": cost, "payout": payout, "pnl": payout - cost}


def main():
    print()
    print("=" * 66)
    print(" STRATEGY A: all-YES parlay + all-NO parlay, both sides")
    print("=" * 66)
    print(" Five assets means 2^5 = 32 possible combinations of outcomes.")
    print(" Your two tickets cover exactly 2 of them: all-up, and all-down.")
    print(" On the other 30, BOTH tickets expire worthless.")
    print()
    print(f" {'correlation':>12} {'P(all align)':>13} {'P(lose all)':>12} {'EV per $1':>11}")
    print(f" {'-'*12} {'-'*13} {'-'*12} {'-'*11}")

    for rho in [0.0, 0.3, 0.5, 0.7, 0.85, 0.95]:
        r = strategy_a(rho)
        print(f" {rho:>12.2f} {r['p_all_aligned']:>13.4f} "
              f"{r['p_lose_everything']:>12.4f} {r['ev_pct']:>10.2f}%")

    print()
    print(" Note the EV column. It is NEGATIVE at every correlation level,")
    print(" and that is with parlays priced with ZERO house edge, which is")
    print(" more generous than reality. The only cost here is the 2% fee.")
    print()
    print(" Higher correlation does NOT rescue it. It raises how often you")
    print(" win, but a fairly-priced parlay gets correspondingly cheaper to")
    print(" the point where the payout shrinks to match. You cannot buy both")
    print(" sides of anything and profit -- the price already contains the")
    print(" probability.")

    print()
    print("=" * 66)
    print(" STRATEGY B: yes + no on the SAME market (a true hedge)")
    print("=" * 66)
    for px in [0.20, 0.50, 0.80]:
        r = strategy_b(px)
        print(f"  contract at ${px:.2f}: pay ${r['cost']:.4f}, "
              f"receive ${r['payout']:.2f}  ->  {r['pnl']:+.4f}")
    print()
    print(" This one IS guaranteed. Guaranteed to LOSE, by exactly the fees,")
    print(" every time, with no variance at all. YES and NO together always")
    print(" cost $1.00 and always pay $1.00. The fees are pure subtraction.")

    print()
    print("=" * 66)
    print(" WHY IT FEELS LIKE IT WORKS")
    print("=" * 66)
    r = strategy_a(0.85)
    print(f" At 0.85 correlation, all five align {r['p_all_aligned']*100:.1f}% of the time.")
    print(" So most sessions you DO collect on one ticket and it feels like")
    print(" a system. The losses are rarer but much larger, and they are")
    print(" exactly big enough to more than cancel the wins.")
    print()
    print(" That shape -- win small often, lose big rarely -- is the same")
    print(" shape as an account going 200 -> 300 several times and then")
    print(" 800 -> 100 once.")
    print()


if __name__ == "__main__":
    main()
