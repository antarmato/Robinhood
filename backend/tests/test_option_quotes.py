"""Tests for real-quote contract selection (backend/option_quotes.py)."""

from datetime import date

from backend import option_quotes as oq
from backend import pricing


TODAY = date(2026, 7, 14)


def occ(sym, yy, mm, dd, cp, strike):
    return f"{sym}{yy:02d}{mm:02d}{dd:02d}{cp}{int(strike * 1000):08d}"


def snap(bid, ask, delta):
    return {
        "latestQuote": {"bp": bid, "ap": ask},
        "greeks": {"delta": delta},
        "impliedVolatility": 0.55,
    }


# ── parse_occ ────────────────────────────────────────────────────────────────

def test_parse_occ_roundtrip():
    meta = oq.parse_occ(occ("COIN", 26, 8, 21, "C", 420.0))
    assert meta == {"underlying": "COIN", "expiry": date(2026, 8, 21),
                    "type": "call", "strike": 420.0}


def test_parse_occ_put_and_fractional_strike():
    meta = oq.parse_occ(occ("SOFI", 26, 8, 21, "P", 17.5))
    assert meta["type"] == "put" and meta["strike"] == 17.5


def test_parse_occ_garbage_returns_none():
    assert oq.parse_occ("not-an-occ") is None


def test_dte_left():
    assert oq.dte_left(occ("COIN", 26, 8, 21, "C", 420.0), today=TODAY) == 38


# ── select_contract ──────────────────────────────────────────────────────────

def test_select_picks_delta_closest_to_target_at_best_expiry():
    snaps = {
        occ("COIN", 26, 8, 21, "C", 400.0): snap(9.0, 9.6, 0.40),
        occ("COIN", 26, 8, 21, "C", 420.0): snap(5.0, 5.4, 0.27),  # ← winner
        occ("COIN", 26, 8, 21, "C", 440.0): snap(2.8, 3.1, 0.16),
    }
    pick = oq.select_contract(snaps, target_dte=35, today=TODAY)
    assert pick["strike"] == 420.0
    assert pick["ask"] == 5.4 and pick["bid"] == 5.0
    assert pick["dte"] == 38


def test_select_prefers_expiry_nearest_target_dte():
    near = occ("COIN", 26, 8, 14, "C", 420.0)   # 31 DTE
    far  = occ("COIN", 26, 9, 18, "C", 420.0)   # 66 DTE
    snaps = {near: snap(5.0, 5.4, 0.30), far: snap(7.0, 7.5, 0.25)}
    pick = oq.select_contract(snaps, target_dte=35, today=TODAY)
    assert pick["occ_symbol"] == near  # closer expiry wins even at worse delta


def test_select_rejects_one_sided_and_wide_quotes():
    snaps = {
        occ("COIN", 26, 8, 21, "C", 410.0): snap(0.0, 5.4, 0.30),   # no bid
        occ("COIN", 26, 8, 21, "C", 420.0): snap(2.0, 4.0, 0.26),   # 67% spread
        occ("COIN", 26, 8, 21, "C", 430.0): snap(4.0, 4.4, 0.22),   # ← only sane one
    }
    pick = oq.select_contract(snaps, target_dte=35, today=TODAY)
    assert pick["strike"] == 430.0


def test_select_requires_greeks():
    s = snap(5.0, 5.4, 0.25)
    del s["greeks"]
    assert oq.select_contract({occ("COIN", 26, 8, 21, "C", 420.0): s},
                              target_dte=35, today=TODAY) is None


def test_select_empty_chain():
    assert oq.select_contract({}, target_dte=35, today=TODAY) is None


# ── mark_position_quoted ─────────────────────────────────────────────────────

def test_quoted_mark_pnl_from_real_bid():
    pos = {"entry_option_price": 5.40, "contracts": 0.1852}
    mark = pricing.mark_position_quoted(pos, bid=6.48, dte_left=30)
    assert mark["option_price"] == 6.48
    assert mark["pnl_pct"] == 20.0
    assert mark["dte_left"] == 30
    assert abs(mark["pnl_dollars"] - (6.48 - 5.40) * 0.1852 * 100) < 0.01


def test_quoted_mark_floors_at_penny():
    pos = {"entry_option_price": 5.40, "contracts": 0.1852}
    mark = pricing.mark_position_quoted(pos, bid=0.0, dte_left=3)
    assert mark["option_price"] == 0.01
