"""Tests for the Judge's deterministic weighted score."""

import pytest

from backend.agents.judge import _compute_score


def score(tech=5, fund=5, sent=5, risk=8):
    return _compute_score(
        {"score": tech}, {"score": fund}, {"score": sent}, {"score": risk}
    )


def test_all_average_scores():
    # base = 5*3 + 5*1.5 + 5*1.5 + 8 = 38; 3 weak → consensus capped at -3
    assert score() == 35.0


def test_strong_consensus_bonus_capped():
    # base = 8*3 + 7*1.5 + 7*1.5 + 8 = 53; 3 strong → +4.5 capped at +3
    assert score(tech=8, fund=7, sent=7) == 56.0


def test_mid_tier_scores_get_small_bonus():
    # No 6.9 → 7.0 cliff: three mids get +0.5 each
    # base = 6.5*3 + 6*1.5 + 6*1.5 + 8 = 45.5; 3 mid → +1.5
    assert score(tech=6.5, fund=6, sent=6) == 47.0


def test_mixed_strong_and_weak():
    # base = 9*3 + 4*1.5 + 4*1.5 + 8 = 47; 1 strong (+1.5) + 2 weak (-3) = -1.5
    assert score(tech=9, fund=4, sent=4) == 45.5


def test_missing_scores_default_sensibly():
    assert _compute_score({}, {}, {}, {}) == 35.0  # defaults: 5/5/5/8
