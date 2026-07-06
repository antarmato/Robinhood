"""Tests for the sim option pricing model and exit ladder (backend/pricing.py)."""

import math

import pytest

from backend import pricing


def price(entry_stock=100.0, current_stock=100.0, entry_option=1.0,
          direction="bullish", delta=0.25, iv_rank=50.0,
          entry_dte=35, dte_left=35):
    return pricing.price_option(
        entry_stock=entry_stock, current_stock=current_stock,
        entry_option=entry_option, direction=direction, delta=delta,
        iv_rank=iv_rank, entry_dte=entry_dte, dte_left=dte_left,
    )


# ── entry_premium (leverage normalization) ──────────────────────────────────

def test_premium_scales_with_stock_price():
    # Same IV/DTE → premium proportional to spot
    p_small = pricing.entry_premium(18.0, 50.0)
    p_large = pricing.entry_premium(557.0, 50.0)
    assert p_large / p_small == pytest.approx(557.0 / 18.0, rel=0.01)


def test_premium_at_iv50_is_4pct_of_spot():
    assert pricing.entry_premium(100.0, 50.0, entry_dte=35) == pytest.approx(4.0, abs=0.01)


def test_premium_scales_with_iv():
    assert pricing.entry_premium(100.0, 90.0) > pricing.entry_premium(100.0, 10.0)


def test_leverage_uniform_across_symbols():
    # 1% favorable move should produce ~the same option P&L% for an $18 stock
    # and a $557 stock (this was the AMD -94.8%-in-13-minutes bug).
    def pnl_pct_for(spot):
        prem = pricing.entry_premium(spot, 50.0)
        pos = {"entry_stock_price": spot, "entry_option_price": prem,
               "direction": "bullish", "delta": 0.25, "iv_rank": 50.0,
               "entry_dte": 35, "contracts": round(1.0 / prem, 4)}
        return pricing.mark_position(pos, spot * 1.01, days_held=0)["pnl_pct"]

    assert pnl_pct_for(18.0) == pytest.approx(pnl_pct_for(557.0), abs=0.5)


def test_fractional_contracts_size_to_100_dollars():
    spot, iv = 557.0, 50.0
    prem = pricing.entry_premium(spot, iv)
    contracts = round(100.0 / (prem * 100.0), 4)
    pos = {"entry_stock_price": spot, "entry_option_price": prem,
           "direction": "bullish", "delta": 0.25, "iv_rank": iv,
           "entry_dte": 35, "contracts": contracts}
    # Total premium at entry ≈ $100
    assert prem * contracts * 100 == pytest.approx(100.0, abs=0.5)
    # A -0.68% stock move (the AMD trade) now marks around -4%, not -95%
    mark = pricing.mark_position(pos, spot * (1 - 0.0068), days_held=0)
    assert -8.0 < mark["pnl_pct"] < -2.0


# ── spread_fraction (bid-ask friction) ──────────────────────────────────────

@pytest.mark.parametrize("iv_rank,expected", [(0, 0.03), (50, 0.05), (100, 0.07)])
def test_spread_fraction_scales_with_iv(iv_rank, expected):
    assert pricing.spread_fraction(iv_rank) == pytest.approx(expected)


def test_mark_position_applies_spread_friction():
    pos = {"entry_stock_price": 100.0, "entry_option_price": 4.0,
           "direction": "bullish", "delta": 0.25, "iv_rank": 50.0,
           "entry_dte": 35, "contracts": 0.25, "spread_frac": 0.05}
    # Flat stock, no decay: liquidation = fair × 0.95 → P&L = -5%
    mark = pricing.mark_position(pos, 100.0, days_held=0)
    assert mark["pnl_pct"] == pytest.approx(-5.0, abs=0.01)
    assert mark["pnl_dollars"] == pytest.approx(-5.0, abs=0.05)  # $100 basis


def test_mark_position_no_friction_for_legacy_positions():
    pos = {"entry_stock_price": 100.0, "entry_option_price": 1.0,
           "direction": "bullish", "entry_dte": 35}
    assert pricing.mark_position(pos, 100.0, days_held=0)["pnl_pct"] == 0.0


# ── initial_stop_pct ────────────────────────────────────────────────────────

@pytest.mark.parametrize("iv_rank,expected", [
    (85, -20.0), (70, -20.0),
    (60, -28.0), (50, -28.0),
    (40, -38.0), (30, -38.0),
    (20, -50.0), (0, -50.0),
])
def test_initial_stop_tiers(iv_rank, expected):
    assert pricing.initial_stop_pct(iv_rank) == expected


# ── price_option ────────────────────────────────────────────────────────────

def test_no_move_no_decay_returns_entry():
    assert price() == 1.0


def test_theta_decay_no_move():
    # sqrt-of-time: dte 35 → 17 should decay price to ~sqrt(17/35)
    p = price(dte_left=17)
    assert p == pytest.approx(math.sqrt(17 / 35), abs=0.01)


def test_favorable_move_increases_price_bullish():
    assert price(current_stock=105) > 1.0


def test_adverse_move_decreases_price_bullish():
    assert price(current_stock=95) < 1.0


def test_bearish_mirrors_bullish():
    bull_up = price(current_stock=105, direction="bullish")
    bear_dn = price(current_stock=95, direction="bearish")
    assert bull_up == pytest.approx(bear_dn, abs=1e-6)


def test_price_floor():
    # Huge adverse move + full decay can never go below a penny
    assert price(current_stock=50, dte_left=0) == 0.01


def test_gamma_caps_effective_delta():
    # Massive favorable move: effective delta capped at 0.80
    p = price(current_stock=200)  # +100% move
    favorable = 100.0
    # directional pnl can't exceed favorable * 0.80
    assert p <= 1.0 + favorable * 0.80


def test_high_iv_vega_drag_on_favorable_move():
    # Same favorable move, higher IV rank → more IV compression → lower price
    assert price(current_stock=105, iv_rank=90) < price(current_stock=105, iv_rank=10)


# ── mark_position ───────────────────────────────────────────────────────────

def test_mark_position_pnl_math():
    pos = {
        "entry_stock_price": 100.0, "entry_option_price": 1.0,
        "direction": "bullish", "delta": 0.25, "iv_rank": 50.0,
        "entry_dte": 35, "contracts": 2,
    }
    mark = pricing.mark_position(pos, 100.0, days_held=0)
    assert mark["option_price"] == 1.0
    assert mark["pnl_pct"] == 0.0
    assert mark["pnl_dollars"] == 0.0
    assert mark["dte_left"] == 35

    mark = pricing.mark_position(pos, 105.0, days_held=0)
    # 2 contracts → dollars = (opt - 1.0) * 200
    assert mark["pnl_dollars"] == pytest.approx((mark["option_price"] - 1.0) * 200, abs=0.01)
    assert mark["pnl_pct"] > 0


def test_mark_position_dte_clamps_at_zero():
    pos = {"entry_stock_price": 100.0, "entry_option_price": 1.0,
           "direction": "bullish", "entry_dte": 35}
    mark = pricing.mark_position(pos, 100.0, days_held=50)
    assert mark["dte_left"] == 0


# ── virtual_trade_pnl_pct (counterfactual labeling) ─────────────────────────

def test_virtual_trade_flat_move_loses_spread_and_theta():
    # No stock move → small loss from 5 days of theta + crossing the spread
    pnl = pricing.virtual_trade_pnl_pct(100.0, 100.0, "bullish", 50.0)
    assert -25 < pnl < 0


def test_virtual_trade_favorable_move_wins():
    assert pricing.virtual_trade_pnl_pct(100.0, 106.0, "bullish", 30.0) > 0


def test_virtual_trade_adverse_move_loses():
    assert pricing.virtual_trade_pnl_pct(100.0, 94.0, "bullish", 30.0) < -20


def test_virtual_bearish_mirrors_bullish():
    b = pricing.virtual_trade_pnl_pct(100.0, 106.0, "bullish", 40.0)
    s = pricing.virtual_trade_pnl_pct(100.0, 94.0, "bearish", 40.0)
    assert b == pytest.approx(s, abs=0.01)


def test_virtual_trade_comparable_to_real_mark():
    # A virtual label and a real position mark with the same inputs should
    # agree — virtual outcomes must be directly comparable to actual ones.
    iv, spot, move = 50.0, 100.0, 1.03
    prem = pricing.entry_premium(spot, iv)
    pos = {"entry_stock_price": spot, "entry_option_price": prem,
           "direction": "bullish", "delta": 0.25, "iv_rank": iv,
           "entry_dte": 35, "contracts": 1, "spread_frac": pricing.spread_fraction(iv)}
    real = pricing.mark_position(pos, spot * move, days_held=5)["pnl_pct"]
    virt = pricing.virtual_trade_pnl_pct(spot, spot * move, "bullish", iv)
    assert virt == pytest.approx(real, abs=0.5)


# ── update_stall_count ──────────────────────────────────────────────────────

def test_stall_increments_when_declining_from_peak():
    assert pricing.update_stall_count(0, new_high=50.0, pnl_pct=40.0, prev_pnl=45.0) == 1


def test_stall_resets_near_high():
    assert pricing.update_stall_count(2, new_high=50.0, pnl_pct=48.0, prev_pnl=44.0) == 0


def test_stall_unchanged_below_threshold():
    # peak under 40% → no stall counting, and not near high → unchanged
    assert pricing.update_stall_count(1, new_high=30.0, pnl_pct=20.0, prev_pnl=25.0) == 1


# ── compute_trail_floor ─────────────────────────────────────────────────────

def floor(new_high, pnl_pct=0.0, initial_stop=-28.0, stall_count=0,
          dte_left=30, entry_confidence=7.0):
    return pricing.compute_trail_floor(
        new_high=new_high, pnl_pct=pnl_pct, initial_stop=initial_stop,
        stall_count=stall_count, dte_left=dte_left,
        entry_confidence=entry_confidence,
    )


@pytest.mark.parametrize("new_high,expected", [
    (160.0, 125.0),   # 150%+ tier: give back 35pts
    (120.0, 90.0),    # 100%+ tier: give back 30pts
    (60.0, 35.0),     # 50%+ tier: give back 25pts
    (30.0, 0.0),      # 25%+ tier: protect breakeven
    (10.0, -28.0),    # below 25%: initial stop
])
def test_trail_floor_tiers(new_high, expected):
    assert floor(new_high, pnl_pct=new_high) == expected


def test_stall_tightening_raises_floor():
    base = floor(60.0, pnl_pct=40.0)
    tightened = floor(60.0, pnl_pct=40.0, stall_count=3)
    assert tightened > base
    assert tightened == min(40.0 + 5.0, base + 10.0)


def test_dte_lift_raises_negative_floor_toward_breakeven():
    normal = floor(0.0, pnl_pct=-10.0, initial_stop=-38.0, dte_left=30)
    lifted = floor(0.0, pnl_pct=-10.0, initial_stop=-38.0, dte_left=7)
    assert normal == -38.0
    assert lifted == pytest.approx(-38.0 + (14 - 7) * 1.5)
    assert lifted <= 0.0


def test_mini_peak_reversal_locks_small_gain():
    assert floor(15.0, pnl_pct=1.0, initial_stop=-38.0) == 2.0


def test_low_confidence_take_profit():
    # Low-conf trade that peaked 90%: floor lifted to at least +50
    assert floor(90.0, pnl_pct=55.0, entry_confidence=5.0) == 65.0  # tier 90-25=65 > 50
    assert floor(72.0, pnl_pct=52.0, entry_confidence=4.0) == 50.0  # tier 47 → lifted to 50
    # High confidence → no low-conf lock, tier floor only
    assert floor(72.0, pnl_pct=52.0, entry_confidence=8.0) == 47.0


# ── exit_reason ─────────────────────────────────────────────────────────────

def reason(pnl_pct, new_high, trail_floor, initial_stop=-28.0, iv_rank=50.0,
           days_held=1, dte_left=30):
    return pricing.exit_reason(
        pnl_pct=pnl_pct, new_high=new_high, trail_floor=trail_floor,
        initial_stop=initial_stop, iv_rank=iv_rank,
        days_held=days_held, dte_left=dte_left,
    )


def test_hold_when_above_floor_and_healthy():
    assert reason(10.0, 15.0, -28.0) is None


def test_stop_loss_message():
    r = reason(-30.0, 5.0, -28.0)
    assert r is not None and r.startswith("Stop loss")


def test_trailing_stop_message():
    r = reason(34.0, 60.0, 35.0)
    assert r is not None and r.startswith("Trailing stop")


def test_stale_loser_exit():
    r = reason(-20.0, 2.0, -28.0, days_held=11)
    assert r is not None and r.startswith("Stale-loser")


def test_dead_money_exit():
    r = reason(3.0, 10.0, -28.0, days_held=16)
    assert r is not None and r.startswith("Dead-money")


def test_theta_exit_final_week():
    r = reason(5.0, 10.0, -28.0, dte_left=6)
    assert r is not None and r.startswith("Theta exit")


def test_no_theta_exit_if_strongly_profitable():
    # 25% gain in final week: theta exit skipped, expiry not reached yet
    assert reason(25.0, 30.0, 0.0, dte_left=6) is None


def test_expiry_forced_close():
    r = reason(50.0, 60.0, 35.0, dte_left=1)
    assert r is not None and r.startswith("Expiry")
