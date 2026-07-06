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
