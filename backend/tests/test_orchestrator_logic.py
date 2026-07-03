"""Tests for portfolio filters, thesis-review cooldown, and time utilities."""

from datetime import timedelta

import pytest

from backend.orchestrator import Orchestrator
from backend.timeutil import now_et, parse_iso_et, days_since


class FakeState:
    def __init__(self, open_positions=None, closed_positions=None, cycle=10):
        self._open = open_positions or []
        self._closed = closed_positions or []
        self.cycle_count = cycle
        self.events = []

    def get_sim_positions(self, status=None):
        if status == "open":
            return list(self._open)
        if status == "closed":
            return list(self._closed)
        return list(self._open) + list(self._closed)

    def log_event(self, event_type, data=None):
        self.events.append((event_type, data))


def make_orch(open_positions=None, closed_positions=None, cycle=10):
    orch = Orchestrator.__new__(Orchestrator)
    orch.state = FakeState(open_positions, closed_positions, cycle)
    orch.review_interval_h = 3.0
    return orch


def cand(symbol, direction="bullish"):
    return {"symbol": symbol, "direction": direction}


def open_pos(symbol, option_type="call", direction="bullish"):
    return {"symbol": symbol, "option_type": option_type, "direction": direction,
            "position_id": f"id-{symbol}"}


# ── _apply_portfolio_filters ────────────────────────────────────────────────

def test_passes_clean_candidate():
    orch = make_orch()
    out = orch._apply_portfolio_filters([cand("UBER")])
    assert [c["symbol"] for c in out] == ["UBER"]


def test_skips_already_open_symbol():
    orch = make_orch(open_positions=[open_pos("PLTR")])
    assert orch._apply_portfolio_filters([cand("PLTR")]) == []


def test_reentry_block_after_recent_loss():
    closed = [{"symbol": "SOFI", "status": "closed", "cycle": 9,
               "pnl_dollars": -40.0, "pnl_pct": -40.0}]
    orch = make_orch(closed_positions=closed, cycle=10)
    assert orch._apply_portfolio_filters([cand("SOFI")]) == []


def test_reentry_allowed_after_recent_win():
    closed = [{"symbol": "SOFI", "status": "closed", "cycle": 9,
               "pnl_dollars": 40.0, "pnl_pct": 40.0}]
    orch = make_orch(closed_positions=closed, cycle=10)
    assert [c["symbol"] for c in orch._apply_portfolio_filters([cand("SOFI")])] == ["SOFI"]


def test_reentry_allowed_after_old_loss():
    closed = [{"symbol": "SOFI", "status": "closed", "cycle": 3,
               "pnl_dollars": -40.0, "pnl_pct": -40.0}]
    orch = make_orch(closed_positions=closed, cycle=10)
    assert [c["symbol"] for c in orch._apply_portfolio_filters([cand("SOFI")])] == ["SOFI"]


def test_churn_block_after_two_same_direction_trades():
    # Two recent COIN puts closed (even winners) → third COIN put blocked
    closed = [
        {"symbol": "COIN", "direction": "bearish", "status": "closed",
         "cycle": 8, "pnl_dollars": 48.0, "pnl_pct": 48.0},
        {"symbol": "COIN", "direction": "bearish", "status": "closed",
         "cycle": 9, "pnl_dollars": 61.0, "pnl_pct": 61.0},
    ]
    orch = make_orch(closed_positions=closed, cycle=10)
    assert orch._apply_portfolio_filters([cand("COIN", "bearish")]) == []


def test_churn_block_allows_opposite_direction():
    closed = [
        {"symbol": "UBER", "direction": "bearish", "status": "closed",
         "cycle": 8, "pnl_dollars": 48.0, "pnl_pct": 48.0},
        {"symbol": "UBER", "direction": "bearish", "status": "closed",
         "cycle": 9, "pnl_dollars": 61.0, "pnl_pct": 61.0},
    ]
    orch = make_orch(closed_positions=closed, cycle=10)
    out = orch._apply_portfolio_filters([cand("UBER", "bullish")])
    assert [c["symbol"] for c in out] == ["UBER"]


def test_churn_block_expires_after_five_cycles():
    closed = [
        {"symbol": "UBER", "direction": "bullish", "status": "closed",
         "cycle": 2, "pnl_dollars": 48.0, "pnl_pct": 48.0},
        {"symbol": "UBER", "direction": "bullish", "status": "closed",
         "cycle": 3, "pnl_dollars": 61.0, "pnl_pct": 61.0},
    ]
    orch = make_orch(closed_positions=closed, cycle=10)
    out = orch._apply_portfolio_filters([cand("UBER", "bullish")])
    assert [c["symbol"] for c in out] == ["UBER"]


def test_correlation_group_blocks_crypto_pair():
    # MSTR open → COIN blocked (same correlation group)
    orch = make_orch(open_positions=[open_pos("MSTR")])
    assert orch._apply_portfolio_filters([cand("COIN")]) == []


def test_tech_sector_cap():
    orch = make_orch(open_positions=[open_pos("AAPL"), open_pos("META")])
    assert orch._apply_portfolio_filters([cand("CRM")]) == []


def test_portfolio_full_returns_empty():
    opens = [open_pos(s) for s in ("UBER", "PYPL", "ROKU", "TSLA")]
    orch = make_orch(open_positions=opens)
    assert orch._apply_portfolio_filters([cand("SOFI")]) == []


def test_last_slot_surcharge():
    opens = [open_pos(s) for s in ("UBER", "PYPL", "ROKU")]
    orch = make_orch(open_positions=opens)
    orch._apply_portfolio_filters([cand("SOFI")])
    assert orch._last_slot_surcharge == 5.0

    orch2 = make_orch(open_positions=[open_pos("UBER")])
    orch2._apply_portfolio_filters([cand("SOFI")])
    assert orch2._last_slot_surcharge == 0.0


def test_net_long_caution_flag():
    opens = [open_pos("UBER"), open_pos("PYPL")]
    orch = make_orch(open_positions=opens)
    out = orch._apply_portfolio_filters([cand("TSLA", "bullish")])
    assert out and "net long" in out[0]["caution"]


# ── _thesis_review_due (LLM call cooldown) ──────────────────────────────────

def test_review_due_when_never_reviewed():
    orch = make_orch()
    assert orch._thesis_review_due({}, pnl_pct=0.0) is True


def test_review_not_due_shortly_after_last_review():
    orch = make_orch()
    pos = {"last_thesis_review": (now_et() - timedelta(minutes=40)).isoformat(),
           "pnl_at_last_review": 5.0}
    assert orch._thesis_review_due(pos, pnl_pct=6.0) is False


def test_review_due_after_interval():
    orch = make_orch()
    pos = {"last_thesis_review": (now_et() - timedelta(hours=4)).isoformat(),
           "pnl_at_last_review": 5.0}
    assert orch._thesis_review_due(pos, pnl_pct=5.0) is True


def test_review_due_early_on_sharp_pnl_move():
    orch = make_orch()
    pos = {"last_thesis_review": (now_et() - timedelta(hours=1)).isoformat(),
           "pnl_at_last_review": 5.0}
    assert orch._thesis_review_due(pos, pnl_pct=-15.0) is True


def test_sharp_move_still_respects_30min_floor():
    orch = make_orch()
    pos = {"last_thesis_review": (now_et() - timedelta(minutes=10)).isoformat(),
           "pnl_at_last_review": 5.0}
    assert orch._thesis_review_due(pos, pnl_pct=-40.0) is False


# ── timeutil ────────────────────────────────────────────────────────────────

def test_parse_naive_timestamp_gets_et():
    dt = parse_iso_et("2026-07-01T10:00:00")
    assert dt.tzinfo is not None


def test_parse_aware_timestamp_preserved():
    dt = parse_iso_et("2026-07-01T10:00:00-04:00")
    assert dt.utcoffset() == timedelta(hours=-4)


def test_days_since_mixed_precision():
    three_days_ago = (now_et() - timedelta(days=3)).isoformat()
    assert days_since(three_days_ago) == 3
    # naive string (legacy positions) also works
    naive = (now_et() - timedelta(days=2)).replace(tzinfo=None).isoformat()
    assert days_since(naive) == 2
