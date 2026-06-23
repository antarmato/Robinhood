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
THRESHOLD_CONF       = 5   # minimum confidence (1-10)

# Trade parameters
DTE_MIN = 21
DTE_MAX = 45
DELTA_TARGET = 0.40    # buy ~40-delta option
PROFIT_TARGET_PCT = 50 # close at 50% gain
STOP_LOSS_PCT     = 50 # close at 50% loss


def score_threshold(iv_rank: float, market_open: bool, time_of_day=None) -> float:
    """
    Return the weighted_score threshold for this IV environment + time of day.

    IV rank:
      < 20  → 35 (cheap premium, easier bar)
      40-60 → 43 (elevated premium, harder bar)
      else  → 38 base

    Time of day adjustments (ET):
      9:30-10:30am  +3  opening chaos — prices whip, fakeouts common
      12:00-2:00pm  +2  midday lull — low conviction moves
      10:30-12:00pm -1  best morning window — momentum clear
      2:30-4:00pm   -1  power hour — institutional trend follow-through
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
