"""
Sim option pricing model + exit ladder — pure functions, no I/O.

This is the single source of truth for how a simulated option position is
marked and when it exits. Both the orchestrator's monitor loop and the
/api/sim/prices dashboard endpoint use it, so the P&L the UI shows is always
the same P&L the exit engine acts on.

Pricing model (approximation for 1 long contract, ~0.25 delta at entry):
  • Gamma-adjusted delta: favorable moves push delta toward 0.80,
    adverse moves push it toward 0.05.
  • Vega: favorable moves compress IV (drag), adverse moves expand IV
    (cushion). Effect scales with IV rank at entry.
  • Theta: sqrt-of-time decay on the entry premium — decay accelerates
    as expiry approaches.
"""

import math

__all__ = [
    "entry_premium", "initial_stop_pct", "price_option", "mark_position",
    "update_stall_count", "compute_trail_floor", "exit_reason",
]


def entry_premium(stock_price: float, iv_rank: float, entry_dte: int = 35) -> float:
    """
    Modeled entry premium (per share) for a ~25-delta option:
        premium ≈ spot × (2% + 4% × IV/100) × sqrt(DTE/35)

    This keeps leverage uniform across symbols (~6× option P&L per 1% favorable
    stock move at IV 50). The old flat $1.00 premium made P&L scale with the
    stock's absolute dollar move: a $557 stock (AMD) carried ~140× leverage per
    1% move and blew through a -20% stop to -94.8% in 13 minutes, while an $18
    stock (SOFI) barely moved 5% on the same-size stock move.
    """
    pct = 0.02 + 0.04 * max(0.0, min(iv_rank, 100.0)) / 100.0
    premium = stock_price * pct * math.sqrt(max(entry_dte, 1) / 35.0)
    return round(max(premium, 0.05), 4)


def initial_stop_pct(iv_rank: float) -> float:
    """IV-aware hard stop: expensive options bleed faster on no movement."""
    if iv_rank >= 70:
        return -20.0   # very expensive — exit fast if wrong
    if iv_rank >= 50:
        return -28.0   # elevated IV
    if iv_rank >= 30:
        return -38.0   # moderate
    return -50.0       # cheap premium, give it room


def price_option(
    *,
    entry_stock: float,
    current_stock: float,
    entry_option: float,
    direction: str,
    delta: float,
    iv_rank: float,
    entry_dte: int,
    dte_left: int,
) -> float:
    """Modeled current option price (per share, floored at $0.01)."""
    if direction == "bullish":
        favorable_move = current_stock - entry_stock
    else:
        favorable_move = entry_stock - current_stock
    move_pct = favorable_move / max(entry_stock, 0.01)

    # Gamma-adjusted delta
    if move_pct >= 0:
        effective_delta = min(0.80, delta + move_pct * 0.35)
    else:
        effective_delta = max(0.05, delta + move_pct * 0.15)
    directional_pnl = favorable_move * effective_delta

    # Vega: ~1% stock move at IV rank 50 ≈ 4% effect on option value
    iv_vega_factor = (iv_rank / 100.0) * 0.08
    if move_pct >= 0:
        vega_pnl = -entry_option * iv_vega_factor * move_pct * 2.0
    else:
        vega_pnl = entry_option * iv_vega_factor * abs(move_pct) * 1.0

    # Sqrt-of-time theta: DTE 35→1.0 | 17→0.70 | 7→0.45 | 0→0
    time_factor = math.sqrt(max(0, dte_left) / max(entry_dte, 1))

    return round(max(0.01, entry_option * time_factor + directional_pnl + vega_pnl), 4)


def mark_position(pos: dict, current_stock: float, days_held: int) -> dict:
    """Mark a sim position to the model. Returns option price, P&L, DTE left."""
    entry_dte = int(pos.get("entry_dte", 35))
    entry_opt = float(pos.get("entry_option_price", 1.0))
    contracts = float(pos.get("contracts", 1))   # fractional: sized to $100 total cost
    dte_left  = max(0, entry_dte - max(0, days_held))

    current_opt = price_option(
        entry_stock=float(pos.get("entry_stock_price", current_stock)),
        current_stock=current_stock,
        entry_option=entry_opt,
        direction=pos.get("direction", "bullish"),
        delta=float(pos.get("delta", 0.25)),
        iv_rank=float(pos.get("iv_rank", 50.0)),
        entry_dte=entry_dte,
        dte_left=dte_left,
    )
    return {
        "option_price": current_opt,
        "pnl_pct":      round((current_opt - entry_opt) / entry_opt * 100, 2),
        "pnl_dollars":  round((current_opt - entry_opt) * contracts * 100, 2),
        "dte_left":     dte_left,
    }


def update_stall_count(stall_count: int, new_high: float, pnl_pct: float, prev_pnl: float) -> int:
    """Count consecutive declining checks after a profitable peak (≥40%)."""
    if new_high >= 40.0 and pnl_pct < prev_pnl - 2.0:
        return stall_count + 1   # declining from a profitable peak
    if pnl_pct >= new_high - 3.0:
        return 0                 # still near the high — reset
    return stall_count


def compute_trail_floor(
    *,
    new_high: float,
    pnl_pct: float,
    initial_stop: float,
    stall_count: int,
    dte_left: int,
    entry_confidence: float,
) -> float:
    """
    Trailing floor for the exit check. No fixed profit target — the floor
    tightens in tiers as the peak gain grows, then several modifiers lift it.
    """
    if new_high >= 150.0:
        trail_floor = new_high - 35.0            # give back 35pts max after 150%+
    elif new_high >= 100.0:
        trail_floor = new_high - 30.0            # give back 30pts max after 100%+
    elif new_high >= 50.0:
        trail_floor = max(0.0, new_high - 25.0)  # give back 25pts, floor ≥ 0%
    elif new_high >= 25.0:
        trail_floor = 0.0                        # protect breakeven after 25%
    else:
        trail_floor = initial_stop               # IV-adjusted hard stop while gain < 25%

    # Stall tightening: 3+ declining checks → reduce the allowance by 10pts
    if stall_count >= 3 and trail_floor > initial_stop:
        trail_floor = min(pnl_pct + 5.0, trail_floor + 10.0)

    # DTE-aware lift: theta accelerates in the final 2 weeks — raise a negative
    # floor toward breakeven so losers aren't held through rapid decay.
    if dte_left <= 14 and trail_floor < 0:
        dte_lift = max(0.0, (14 - dte_left) * 1.5)
        trail_floor = min(trail_floor + dte_lift, 0.0)

    # Mini-peak reversal: had a nice gain (10-25%) and given most of it back —
    # lock in the remaining small gain rather than riding to the stop loss.
    if 10.0 <= new_high < 25.0 and pnl_pct < 3.0 and pnl_pct > initial_stop:
        trail_floor = max(trail_floor, 2.0)

    # Low-conviction take-profit: low-conf trade hit 70%+ → don't give it back
    if entry_confidence <= 5 and new_high >= 70.0 and pnl_pct >= 50.0:
        trail_floor = max(trail_floor, 50.0)

    return trail_floor


def exit_reason(
    *,
    pnl_pct: float,
    new_high: float,
    trail_floor: float,
    initial_stop: float,
    iv_rank: float,
    days_held: int,
    dte_left: int,
) -> str | None:
    """Exit decision given the current mark and floor. None = keep holding."""
    if pnl_pct <= trail_floor:
        if trail_floor >= 50.0:
            return f"Low-conf take-profit: peak {new_high:+.0f}% → locking {pnl_pct:+.1f}%"
        if trail_floor >= 2.0 and new_high < 25.0:
            return f"Mini-peak reversal: peak {new_high:+.1f}% → back to {pnl_pct:+.1f}%"
        if new_high < 25.0:
            return (f"Stop loss {pnl_pct:.1f}% "
                    f"(IV {iv_rank:.0f} → floor {initial_stop:.0f}%)")
        return (f"Trailing stop — peak {new_high:+.0f}% | "
                f"floor {trail_floor:+.0f}% | now {pnl_pct:+.1f}%")

    if days_held >= 10 and pnl_pct < -15.0 and new_high < 5.0:
        # Thesis never materialized — cut and preserve capital for the next setup
        return (f"Stale-loser exit: {days_held}d held, "
                f"max gain {new_high:+.0f}%, now {pnl_pct:+.1f}%")

    if days_held >= 15 and abs(pnl_pct) < 10.0 and new_high < 15.0:
        # Burning theta with no directional move — redeploy the capital
        return (f"Dead-money exit: {days_held}d held, "
                f"max {new_high:+.0f}%, stuck at {pnl_pct:+.1f}% — theta decay")

    if dte_left <= 7 and pnl_pct < 20.0:
        # Final week: theta burns fast; close unless strongly profitable
        return f"Theta exit: {dte_left} DTE, P&L {pnl_pct:+.1f}% (final week)"

    if dte_left <= 2:
        return f"Expiry: {dte_left} DTE — forced close"

    return None
