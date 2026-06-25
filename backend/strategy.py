"""
Strategy parameters and IV rank thresholds.

Single-leg options only (Robinhood MCP constraint — no multi-leg spread orders).

IV rank rules:
  < 40  → cheap premium, normal score threshold
  40-60 → elevated premium, harder score threshold (+5)
  > 60  → skip entirely (don't buy expensive options)
"""

IV_RANK_HARD_SKIP   = 60   # above this: skip the symbol
IV_RANK_HARD_MODE   = 40   # above this: require higher score
THRESHOLD_MARKET    = 38   # base threshold during market hours
THRESHOLD_AFTERHOURS = 32  # base threshold pre/after market
THRESHOLD_HARD_MODE  = 43  # threshold when 40 <= iv_rank <= 60
THRESHOLD_CHEAP_IV   = 35  # threshold when iv_rank < 20 (cheap premium = lower bar)
THRESHOLD_CONF       = 5   # minimum confidence (1-10) — default

# High-beta symbols: require stronger conviction to enter at all
_SYMBOL_CONF_FLOOR: dict[str, int] = {
    "MSTR": 7, "COIN": 7,   # pure crypto proxies — need strong conviction
    "IONQ": 6, "SMCI": 6,   # small/mid speculative — moderate bar
    "RIVN": 6, "HOOD": 6,
}

# Trade parameters
DTE_MIN = 21
DTE_MAX = 45
DELTA_TARGET = 0.25    # buy ~25-delta option (achievable at $1/share on $25-100 stocks)
PROFIT_TARGET_PCT = 50 # close at 50% gain
STOP_LOSS_PCT     = 50 # close at 50% loss


def confidence_minimum(symbol: str) -> int:
    """
    Return the minimum confidence (1-10) required for this symbol.
    High-beta / crypto-proxy symbols require stronger conviction before entry.
    Also queries training DB — adjusts based on historical win rate AND expected value.
    """
    base = _SYMBOL_CONF_FLOOR.get(symbol, THRESHOLD_CONF)
    try:
        from .training_store import get_symbol_perf
        sym_stats = get_symbol_perf(min_trades=5).get(symbol)
        if sym_stats:
            wr      = sym_stats.get("win_rate", 0.5)
            avg_pnl = sym_stats.get("avg_pnl", 0.0)
            avg_win = sym_stats.get("avg_win", 0.0)
            avg_los = sym_stats.get("avg_loss", 0.0)
            # Expected value = WR * avg_win% - (1-WR) * |avg_loss%|
            ev = wr * avg_win - (1 - wr) * abs(avg_los) if avg_win or avg_los else avg_pnl

            if wr < 0.35 or ev < -10.0:
                # Consistently losing or terrible EV — require strong conviction
                base = max(base, 7)
            elif wr < 0.45 or ev < -5.0:
                # Below average — tighten slightly
                base = max(base, 6)
            elif wr > 0.65 and ev > 5.0:
                # Proven profitable winner — lower bar slightly
                base = max(THRESHOLD_CONF, base - 1)
    except Exception:
        pass
    return base


def score_threshold(iv_rank: float, market_open: bool, time_of_day=None,
                    regime_aligned: bool = None, regime_strength: int = 0,
                    streak_surcharge: float = 0.0) -> float:
    """
    Return the weighted_score threshold for this IV environment + time of day.

    IV rank:
      < 20  → 35 (cheap premium, easier bar)
      40-60 → 43 (elevated premium, harder bar)
      else  → 38 base

    Time of day (ET):
      9:30-10:30am  +3  opening — prices whip, fakeouts common
      12:00-2:00pm  +2  midday lull — low conviction
      10:30-12:00pm -1  best morning window
      2:30-4:00pm   -1  power hour — institutional follow-through

    Regime:
      Aligned + strength ≥ 7 → -2 (strong regime tailwind)
      Aligned + strength ≥ 5 → -1
      Misaligned             → +3 (counter-trend = much harder bar)

    Adaptive: self-adjusts ±5 based on rolling sim win rate
      < 35% → +5 (losing badly — raise bar)
      < 45% → +2 (underperforming — tighten)
      > 65% → -2 (winning well — can open up slightly)
      > 75% → -3 (strong edge — loosen further)
    """
    from datetime import time as dtime

    if iv_rank > IV_RANK_HARD_MODE:
        base = THRESHOLD_HARD_MODE
    elif iv_rank < 20:
        base = THRESHOLD_CHEAP_IV
    else:
        base = THRESHOLD_MARKET if market_open else THRESHOLD_AFTERHOURS

    if time_of_day:
        tod = time_of_day
        if   dtime(9, 30)  <= tod < dtime(10, 30): base += 3
        elif dtime(12, 0)  <= tod < dtime(14, 0):  base += 2
        elif dtime(10, 30) <= tod < dtime(12, 0):  base -= 1
        elif dtime(14, 30) <= tod <= dtime(16, 0): base -= 1

    # ── Regime alignment adjustment ───────────────────────────────────────────
    if regime_aligned is True:
        if regime_strength >= 7:
            base -= 2   # strong tailwind — lower bar
        elif regime_strength >= 5:
            base -= 1   # moderate tailwind
    elif regime_aligned is False:
        base += 3   # counter-trend is much harder — only exceptional setups qualify

    # ── Consecutive-loss surcharge ────────────────────────────────────────────
    base += streak_surcharge

    # ── Adaptive adjustment from rolling sim win rate ─────────────────────────
    try:
        from .outcome_tracker import get_outcome_tracker
        stats = get_outcome_tracker().get_stats()
        n = stats.get("total_trades", 0)
        if n >= 15:                     # need enough data before adapting
            wr = stats.get("win_rate", 0.5)
            if   wr < 0.35: base += 5  # losing badly — raise bar hard
            elif wr < 0.45: base += 2  # underperforming — tighten
            elif wr > 0.75: base -= 3  # strong edge — open up
            elif wr > 0.65: base -= 2  # winning well — loosen slightly
    except Exception:
        pass

    # ── Secondary adaptive signal from PostgreSQL training log ───────────────
    # Uses broader sample (persists across restarts) and regime-specific data.
    # Only active when regime context is provided (so we can be direction-specific).
    try:
        from .training_store import get_stats as ts_stats
        db_stats = ts_stats()
        db_n = int(db_stats.get("trades_entered") or 0)
        db_wins = int(db_stats.get("wins") or 0)
        db_losses = int(db_stats.get("losses") or 0)
        db_closed = db_wins + db_losses
        if db_closed >= 20:             # trust DB data after 20 closed trades
            db_wr = db_wins / db_closed
            # Only apply if it disagrees by more than 5% with outcome_tracker (avoid double-counting)
            try:
                from .outcome_tracker import get_outcome_tracker as _ot
                ot_wr = _ot().get_stats().get("win_rate", db_wr)
                wr_delta = abs(db_wr - ot_wr)
            except Exception:
                wr_delta = 0
            if wr_delta > 0.05:         # DB has meaningfully different read
                if   db_wr < 0.35: base += 3
                elif db_wr < 0.45: base += 1
                elif db_wr > 0.75: base -= 2
                elif db_wr > 0.65: base -= 1
    except Exception:
        pass

    return float(base)


def iv_edge_label(iv_rank: float) -> str:
    if iv_rank < 20:
        return f"very cheap (rank {iv_rank:.0f}) — favorable"
    if iv_rank < 40:
        return f"cheap (rank {iv_rank:.0f}) — good entry"
    if iv_rank < 60:
        return f"elevated (rank {iv_rank:.0f}) — harder threshold"
    return f"expensive (rank {iv_rank:.0f}) — skipped"


def trade_defaults() -> dict:
    return {
        "dte_min":          DTE_MIN,
        "dte_max":          DTE_MAX,
        "delta_target":     DELTA_TARGET,
        "profit_target_pct": PROFIT_TARGET_PCT,
        "stop_loss_pct":    STOP_LOSS_PCT,
    }
