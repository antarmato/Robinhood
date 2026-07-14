"""Tests for the adaptive score threshold (backend/strategy.py)."""

from backend import strategy
from backend import training_store


def thr(monkeypatch, stats, iv_rank=30.0, market_open=True, **kw):
    monkeypatch.setattr(training_store, "get_outcome_stats", lambda: stats)
    return strategy.score_threshold(iv_rank, market_open, **kw)


BASE = float(strategy.THRESHOLD_MARKET)


def test_threshold_neutral_without_history(monkeypatch):
    assert thr(monkeypatch, {}) == BASE


def test_threshold_unchanged_below_sample_minimum(monkeypatch):
    stats = {"total_trades": 8, "expectancy": -20.0, "win_rate": 0.2}
    assert thr(monkeypatch, stats) == BASE


def test_threshold_raised_hard_on_bad_expectancy(monkeypatch):
    stats = {"total_trades": 14, "expectancy": -12.0, "win_rate": 0.45}
    assert thr(monkeypatch, stats) == BASE + 4


def test_threshold_raised_mildly_on_negative_expectancy(monkeypatch):
    stats = {"total_trades": 14, "expectancy": -5.0, "win_rate": 0.45}
    assert thr(monkeypatch, stats) == BASE + 2


def test_threshold_lowered_on_strong_expectancy(monkeypatch):
    stats = {"total_trades": 20, "expectancy": 12.0, "win_rate": 0.6}
    assert thr(monkeypatch, stats) == BASE - 2


def test_low_win_rate_stacks_with_bad_expectancy(monkeypatch):
    stats = {"total_trades": 20, "expectancy": -12.0, "win_rate": 0.30}
    assert thr(monkeypatch, stats) == BASE + 4 + 2


def test_positive_expectancy_beats_mediocre_win_rate(monkeypatch):
    # 40% WR with big winners = healthy: expectancy rules, WR nudge absent
    stats = {"total_trades": 20, "expectancy": 6.0, "win_rate": 0.40}
    assert thr(monkeypatch, stats) == BASE - 1


# ── confidence_minimum (per-symbol floors) ──────────────────────────────────

def conf_min(monkeypatch, sym_stats):
    monkeypatch.setattr(training_store, "get_symbol_perf",
                        lambda min_trades=2: {"XYZ": sym_stats} if sym_stats else {})
    return strategy.confidence_minimum("XYZ")


def test_conf_floor_default_without_history(monkeypatch):
    assert conf_min(monkeypatch, None) == strategy.THRESHOLD_CONF


def test_conf_floor_not_raised_for_low_wr_positive_ev(monkeypatch):
    # COIN profile: 40% WR but +17%/trade expectancy — must stay tradeable
    stats = {"win_rate": 0.40, "avg_pnl": 17.4, "avg_win": 54.8, "avg_loss": -7.5}
    assert conf_min(monkeypatch, stats) == strategy.THRESHOLD_CONF


def test_conf_floor_raised_hard_for_bleeder(monkeypatch):
    # SOFI profile: 25% WR, tiny wins, big losses → ev ≈ -13.7
    stats = {"win_rate": 0.25, "avg_pnl": -13.7, "avg_win": 1.5, "avg_loss": -18.8}
    assert conf_min(monkeypatch, stats) == strategy.THRESHOLD_CONF + 2


def test_conf_floor_raised_mildly_for_negative_ev(monkeypatch):
    # ev = 0.5*4 - 0.5*16 = -6.0
    stats = {"win_rate": 0.50, "avg_pnl": -6.0, "avg_win": 4.0, "avg_loss": -16.0}
    assert conf_min(monkeypatch, stats) == strategy.THRESHOLD_CONF + 1


def test_conf_floor_raised_for_low_wr_when_ev_negative(monkeypatch):
    # ev = 0.3*10 - 0.7*6 = -1.2: mild, but 30% WR confirms the weakness
    stats = {"win_rate": 0.30, "avg_pnl": -1.2, "avg_win": 10.0, "avg_loss": -6.0}
    assert conf_min(monkeypatch, stats) == strategy.THRESHOLD_CONF + 1
