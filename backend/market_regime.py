"""
Market Regime Classifier

Classifies the broad market as bull / neutral / bear using pure Python.
Run once per day at pre-market prep; cached in state for all agents.

Signals:
  SPY 20-day EMA slope     — primary trend direction
  SPY vs EMA50             — medium-term trend
  SPY 20-day return        — recent momentum
  VIX level                — fear gauge
  VIXY 5-day return        — VIX direction (rising/falling volatility)
"""

import logging
from datetime import datetime

from . import market_data as md

logger = logging.getLogger(__name__)


def classify_regime() -> dict:
    try:
        spy_df = md.get_historicals("SPY", period="3mo")
        if spy_df.empty or len(spy_df) < 22:
            return _neutral("insufficient SPY data")

        close  = spy_df["close"]
        ema20  = close.ewm(span=20, adjust=False).mean()
        ema50  = close.ewm(span=50, adjust=False).mean()

        spy_slope_5d    = (float(ema20.iloc[-1]) - float(ema20.iloc[-5])) / float(ema20.iloc[-5]) * 100
        spy_above_ema20 = float(close.iloc[-1]) > float(ema20.iloc[-1])
        spy_above_ema50 = float(close.iloc[-1]) > float(ema50.iloc[-1])
        spy_ret_20d     = (float(close.iloc[-1]) - float(close.iloc[-21])) / float(close.iloc[-21]) * 100

        # EMA200 — major long-term structural regime signal
        spy_above_ema200 = False
        try:
            if len(close) >= 200:
                ema200 = close.ewm(span=200, adjust=False).mean()
                spy_above_ema200 = float(close.iloc[-1]) > float(ema200.iloc[-1])
        except Exception:
            pass

        vix_level = float(md.get_vix() or 20.0)
        vix_trend = _get_vix_trend()

        # Breadth: how many of SPY/QQQ/IWM are above their own EMA20?
        breadth_bull = 1 if spy_above_ema20 else 0
        for etf in ["QQQ", "IWM"]:
            try:
                df = md.get_historicals(etf, period="3mo")
                if not df.empty and len(df) >= 21:
                    c   = df["close"]
                    e20 = c.ewm(span=20, adjust=False).mean()
                    if float(c.iloc[-1]) > float(e20.iloc[-1]):
                        breadth_bull += 1
            except Exception:
                pass

        bull = 0
        bear = 0

        # SPY slope (strongest signal)
        if spy_slope_5d > 0.4:    bull += 3
        elif spy_slope_5d > 0.1:  bull += 1
        elif spy_slope_5d < -0.4: bear += 3
        elif spy_slope_5d < -0.1: bear += 1

        # EMA position
        if spy_above_ema50: bull += 1
        else:               bear += 1

        # Broad market breadth (SPY + QQQ + IWM above EMA20)
        if breadth_bull == 3:   bull += 2   # all three aligned bullish
        elif breadth_bull >= 2: bull += 1
        elif breadth_bull == 0: bear += 2   # all three bearish — real deterioration
        else:                   bear += 1

        # 20-day return
        if spy_ret_20d > 4:    bull += 2
        elif spy_ret_20d > 1:  bull += 1
        elif spy_ret_20d < -4: bear += 2
        elif spy_ret_20d < -1: bear += 1

        # VIX level
        if vix_level < 16:    bull += 1
        elif vix_level > 25:  bear += 2
        elif vix_level > 20:  bear += 1

        # VIX direction — matters as much as level
        if vix_trend == "rising":    bear += 2
        elif vix_trend == "falling": bull += 1

        # EMA200 — long-term structural bull/bear
        if spy_above_ema200:  bull += 1   # healthy long-term uptrend
        else:                 bear += 1   # below long-term MA — structural bear

        margin = bull - bear
        total  = bull + bear or 1

        if margin >= 4:    regime = "bull"
        elif margin <= -4: regime = "bear"
        else:              regime = "neutral"

        strength = min(10, int(abs(margin) / total * 10) + 3)

        return {
            "regime":          regime,
            "strength":        strength,
            "spy_slope_5d":    round(spy_slope_5d, 3),
            "spy_above_ema50": spy_above_ema50,
            "spy_above_ema200": spy_above_ema200,
            "spy_ret_20d":     round(spy_ret_20d, 2),
            "vix_level":       round(vix_level, 1),
            "vix_trend":       vix_trend,
            "breadth":         breadth_bull,   # 0-3: how many indexes above EMA20
            "bull_points":     bull,
            "bear_points":     bear,
            "summary": (
                f"SPY slope {spy_slope_5d:+.2f}%/5d | "
                f"{'above' if spy_above_ema50 else 'below'} EMA50 | "
                f"{'above' if spy_above_ema200 else 'below'} EMA200 | "
                f"breadth {breadth_bull}/3 | "
                f"20d ret {spy_ret_20d:+.1f}% | "
                f"VIX {vix_level:.0f} {vix_trend}"
            ),
            "computed_at": datetime.now().isoformat(),
        }

    except Exception as e:
        logger.error(f"Regime classification error: {e}")
        return _neutral(f"error: {e}")


def _get_vix_trend() -> str:
    """5-day VIX direction via VIXY ETF returns."""
    try:
        df = md.get_historicals("VIXY", period="1mo")
        if df.empty or len(df) < 6:
            return "flat"
        ret_5d = (float(df["close"].iloc[-1]) - float(df["close"].iloc[-6])) / float(df["close"].iloc[-6]) * 100
        if ret_5d > 5:    return "rising"
        if ret_5d < -5:   return "falling"
        return "flat"
    except Exception:
        return "flat"


def _neutral(reason: str = "") -> dict:
    return {
        "regime": "neutral", "strength": 5,
        "spy_slope_5d": 0.0, "spy_above_ema50": True, "spy_ret_20d": 0.0,
        "vix_level": 20.0, "vix_trend": "flat",
        "bull_points": 0, "bear_points": 0,
        "summary": f"Neutral (default) — {reason}",
        "computed_at": datetime.now().isoformat(),
    }
