"""
Technical Analysis Agent — evaluates momentum, trend, support/resistance.

Improvements from v1:
- Added ADX for trend strength (separates trending from ranging markets)
- Added Stochastic %K/%D for timing within the trend
- Added intraday context via hourly data (recent 5-day action)
- Better scoring: explicitly separated bull/bear setups
- 52-week context and key level proximity
"""

import logging
from typing import Optional

import anthropic
import numpy as np
import pandas as pd

from .base import BaseAgent, BroadcastFn
from .. import market_data as md

logger = logging.getLogger(__name__)


class TechnicalAgent(BaseAgent):
    def __init__(
        self,
        client: anthropic.AsyncAnthropic,
        broadcast: Optional[BroadcastFn] = None,
    ):
        super().__init__(client, "Technical", model="claude-sonnet-4-6", broadcast=broadcast)

    async def analyze(self, symbol: str, direction: str) -> dict:
        await self._emit("status", f"Running technical analysis on {symbol} ({direction})...")

        # Daily data for indicators
        df = md.get_historicals(symbol, period="1y")
        if df.empty or len(df) < 30:
            return {
                "score": 5, "signal": "neutral", "trend": "ranging",
                "rsi_reading": "N/A", "macd_reading": "N/A",
                "key_support": 0, "key_resistance": 0,
                "summary": "Insufficient historical data for technical analysis.",
            }

        indicators = self._compute_indicators(df)

        # Intraday hourly context (recent 5 days)
        hourly = md.get_intraday(symbol, period="5d", interval="1h")
        intraday_text = self._summarize_intraday(hourly)
        hv_data = md.get_hv(symbol)

        text = self._format_report(symbol, df, indicators, intraday_text, hv_data=hv_data)

        option_side = "bullish call" if direction == "bullish" else "bearish put"
        system = f"""You are a technical analysis expert evaluating whether the chart supports a {option_side} on {symbol}.

Your job: give an honest, quantitative assessment. Score 1-3 if the setup is against the proposed direction. Score 7-10 only if multiple indicators CONFIRM the direction.

Key rules:
- Score for calls: ADX > 20 + price above EMAs + RSI 40-70 + MACD bullish = 7-9
- Score for puts: ADX > 20 + price below EMAs + RSI 30-60 + MACD bearish = 7-9
- Low ADX (< 18) means the market is ranging — options lose value faster in ranging markets → score max 6
- RSI divergences (price makes new high but RSI doesn't) → mention and score down
- For calls: RSI > 75 or price far above BB upper = overbought, score 4-5
- For puts: RSI < 25 or price far below BB lower = oversold, score 4-5

Respond ONLY with JSON:
{{
  "score": <1-10>,
  "signal": "bullish" | "bearish" | "neutral",
  "trend": "strong_uptrend" | "uptrend" | "downtrend" | "strong_downtrend" | "ranging",
  "adx_reading": "<value and interpretation: trending or ranging>",
  "rsi_reading": "<value and interpretation>",
  "macd_reading": "bullish_cross" | "bearish_cross" | "bullish_above_zero" | "bearish_below_zero" | "neutral",
  "stochastic_reading": "<K/D values and interpretation>",
  "key_support": <price level>,
  "key_resistance": <price level>,
  "bb_position": "upper_band" | "upper_half" | "middle" | "lower_half" | "lower_band",
  "intraday_momentum": "strong_bullish" | "bullish" | "neutral" | "bearish" | "strong_bearish",
  "summary": "<3-4 sentences: trend assessment, key signals, and verdict on whether technicals support the {direction} trade>"
}}"""

        raw = await self._call(system, [{"role": "user", "content": text}], max_tokens=600)
        result = self._parse_json(raw)
        result.setdefault("score", 5)
        result.setdefault("signal", "neutral")
        result.setdefault("trend", "ranging")
        result.setdefault("summary", "Technical analysis complete.")
        result.setdefault("key_support", 0)
        result.setdefault("key_resistance", 0)
        result.setdefault("macd_reading", "neutral")
        result.setdefault("rsi_reading", "N/A")
        return result

    # ── Indicator computation ─────────────────────────────────────────────────

    def _compute_indicators(self, df: pd.DataFrame) -> dict:
        close  = df["close"]
        high   = df["high"]
        low    = df["low"]
        volume = df["volume"]
        ind    = {}

        # ── EMAs ─────────────────────────────────────────────────────────────
        ind["ema9"]   = close.ewm(span=9,   adjust=False).mean().iloc[-1]
        ind["ema20"]  = close.ewm(span=20,  adjust=False).mean().iloc[-1]
        ind["ema50"]  = close.ewm(span=50,  adjust=False).mean().iloc[-1]
        ind["ema200"] = close.ewm(span=200, adjust=False).mean().iloc[-1]

        # EMA slope (% change over 5 days)
        ema20_series = close.ewm(span=20, adjust=False).mean()
        ind["ema20_slope"] = (ema20_series.iloc[-1] - ema20_series.iloc[-5]) / ema20_series.iloc[-5] * 100

        # ── RSI(14) ───────────────────────────────────────────────────────────
        delta = close.diff()
        gain  = delta.where(delta > 0, 0.0).rolling(14).mean()
        loss  = (-delta.where(delta < 0, 0.0)).rolling(14).mean()
        rs    = gain / loss.replace(0, np.nan)
        rsi_series = 100 - 100 / (1 + rs)
        ind["rsi"]      = float(rsi_series.iloc[-1])
        ind["rsi_prev"] = float(rsi_series.iloc[-5])  # for divergence check

        # ── MACD ──────────────────────────────────────────────────────────────
        ema12  = close.ewm(span=12, adjust=False).mean()
        ema26  = close.ewm(span=26, adjust=False).mean()
        macd   = ema12 - ema26
        sig    = macd.ewm(span=9, adjust=False).mean()
        hist   = macd - sig
        ind["macd"]       = float(macd.iloc[-1])
        ind["macd_sig"]   = float(sig.iloc[-1])
        ind["macd_hist"]  = float(hist.iloc[-1])
        ind["macd_hist2"] = float(hist.iloc[-2])

        # ── Stochastic(14,3) ─────────────────────────────────────────────────
        low14  = low.rolling(14).min()
        high14 = high.rolling(14).max()
        k_raw  = (close - low14) / (high14 - low14).replace(0, np.nan) * 100
        ind["stoch_k"] = float(k_raw.rolling(3).mean().iloc[-1])
        ind["stoch_d"] = float(k_raw.rolling(3).mean().rolling(3).mean().iloc[-1])

        # ── ATR(14) ───────────────────────────────────────────────────────────
        tr  = pd.concat(
            [(high - low), (high - close.shift()).abs(), (low - close.shift()).abs()], axis=1
        ).max(axis=1)
        atr = tr.rolling(14).mean()
        ind["atr"]     = float(atr.iloc[-1])
        ind["atr_pct"] = ind["atr"] / float(close.iloc[-1]) * 100

        # ── ADX(14) — trend strength ───────────────────────────────────────
        ind["adx"] = self._compute_adx(high, low, close)

        # ── Bollinger Bands(20,2) ─────────────────────────────────────────────
        sma20 = close.rolling(20).mean()
        std20 = close.rolling(20).std()
        ind["bb_upper"]  = float((sma20 + 2 * std20).iloc[-1])
        ind["bb_lower"]  = float((sma20 - 2 * std20).iloc[-1])
        ind["bb_middle"] = float(sma20.iloc[-1])
        price = float(close.iloc[-1])
        bw = ind["bb_upper"] - ind["bb_lower"]
        ind["bb_pct"] = (price - ind["bb_lower"]) / bw if bw > 0 else 0.5

        # ── Key levels (swing highs/lows) ─────────────────────────────────────
        recent20 = df.tail(20)
        recent60 = df.tail(60)
        ind["resistance_20d"] = float(recent20["high"].max())
        ind["support_20d"]    = float(recent20["low"].min())
        ind["resistance_60d"] = float(recent60["high"].max())
        ind["support_60d"]    = float(recent60["low"].min())

        # ── 52-week context ───────────────────────────────────────────────────
        lookback = min(252, len(close))
        ind["high_52w"] = float(close.tail(lookback).max())
        ind["low_52w"]  = float(close.tail(lookback).min())

        # ── Volume trend ──────────────────────────────────────────────────────
        vol_avg20 = float(volume.iloc[-21:-1].mean()) if len(volume) >= 21 else float(volume.mean())
        ind["vol_ratio"] = float(volume.iloc[-1]) / vol_avg20 if vol_avg20 > 0 else 1.0

        ind["price"] = float(close.iloc[-1])
        return ind

    def _compute_adx(self, high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> float:
        """Average Directional Index — measures trend strength (0-100)."""
        try:
            up_move   = high.diff()
            down_move = -low.diff()

            plus_dm  = up_move.where((up_move > down_move) & (up_move > 0), 0.0)
            minus_dm = down_move.where((down_move > up_move) & (down_move > 0), 0.0)

            tr = pd.concat(
                [(high - low), (high - close.shift()).abs(), (low - close.shift()).abs()], axis=1
            ).max(axis=1)

            atr14   = tr.rolling(period).mean()
            plus_di = (plus_dm.rolling(period).mean() / atr14).replace([np.inf, -np.inf], 0) * 100
            minus_di = (minus_dm.rolling(period).mean() / atr14).replace([np.inf, -np.inf], 0) * 100

            dx = ((plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan) * 100)
            adx = dx.rolling(period).mean()
            val = float(adx.iloc[-1])
            return val if not np.isnan(val) else 20.0
        except Exception:
            return 20.0

    def _hv_line(self, hv: dict | None) -> str:
        if not hv or hv.get("hv20") is None:
            return "HV data unavailable"
        note = {
            "high":   "IV likely elevated — consider debit spread",
            "normal": "IV normal — reasonable premiums",
            "low":    "IV low — cheap to buy premium"
        }.get(hv.get("regime", ""), "")
        rank = hv.get("hv_rank") or 0
        return (f"HV20={hv.get('hv20')}%  HV60={hv.get('hv60')}%  "
                f"HV Rank={rank:.0f}/100 ({hv.get('regime','?').upper()}) — {note}")

    def _summarize_intraday(self, hourly: pd.DataFrame) -> str:
        if hourly.empty or len(hourly) < 6:
            return "Intraday data unavailable."
        try:
            # Today's action (last 8 bars ≈ today's session)
            today = hourly.tail(8)
            open_price  = float(today["open"].iloc[0])
            close_price = float(today["close"].iloc[-1])
            session_pct = (close_price - open_price) / open_price * 100
            session_high = float(today["high"].max())
            session_low  = float(today["low"].min())
            vol_today    = float(today["volume"].sum())

            # 5-day average session volume
            vol_all = float(hourly["volume"].sum())
            days = max(1, len(hourly) // 8)
            vol_day_avg = vol_all / days

            # Recent momentum (last 4 hours vs 4 hours before)
            recent  = float(hourly["close"].tail(4).mean())
            earlier = float(hourly["close"].tail(8).head(4).mean())
            intra_mom = (recent - earlier) / earlier * 100 if earlier else 0

            return (
                f"Today's session: open={open_price:.2f}, current={close_price:.2f}, "
                f"pct={session_pct:+.2f}%, H/L={session_high:.2f}/{session_low:.2f}\n"
                f"Intraday volume vs daily avg: {vol_today/vol_day_avg:.1f}x\n"
                f"Last 4h momentum vs prior 4h: {intra_mom:+.2f}%"
            )
        except Exception as e:
            logger.debug(f"Intraday summary error: {e}")
            return "Intraday data unavailable."

    def _format_report(self, symbol: str, df: pd.DataFrame, i: dict, intraday_text: str, hv_data: dict | None = None) -> str:
        p = i["price"]
        macd_cross = ("↑ Bullish cross" if i["macd_hist"] > 0 > i["macd_hist2"]
                      else "↓ Bearish cross" if i["macd_hist"] < 0 < i["macd_hist2"]
                      else f"{'Above' if i['macd_hist'] > 0 else 'Below'} zero")
        trend_str = (
            "Strong Uptrend" if i["adx"] > 25 and p > i["ema20"] and p > i["ema50"] else
            "Uptrend"        if p > i["ema20"] and p > i["ema50"] else
            "Strong Downtrend" if i["adx"] > 25 and p < i["ema20"] and p < i["ema50"] else
            "Downtrend"      if p < i["ema20"] and p < i["ema50"] else
            "Mixed / Ranging"
        )
        return f"""Technical Analysis: {symbol} — Price: ${p:.2f}

TREND ({trend_str}):
  EMA(9):   ${i['ema9']:.2f}   {'↑ above' if p > i['ema9'] else '↓ below'}
  EMA(20):  ${i['ema20']:.2f}  {'↑ above' if p > i['ema20'] else '↓ below'} (slope: {i['ema20_slope']:+.2f}%/5d)
  EMA(50):  ${i['ema50']:.2f}  {'↑ above' if p > i['ema50'] else '↓ below'}
  EMA(200): ${i['ema200']:.2f} {'↑ above' if p > i['ema200'] else '↓ below'}
  ADX(14):  {i['adx']:.1f}  ({'Strong trend' if i['adx'] > 25 else 'Moderate trend' if i['adx'] > 18 else 'Ranging/weak trend'})

MOMENTUM:
  RSI(14):  {i['rsi']:.1f}  ({'⚠ Overbought' if i['rsi'] > 70 else '⚠ Oversold' if i['rsi'] < 30 else 'Neutral zone'})
  RSI 5d ago: {i['rsi_prev']:.1f}  (trend: {'rising' if i['rsi'] > i['rsi_prev'] else 'falling'})
  MACD:     {i['macd']:.4f}  Signal: {i['macd_sig']:.4f}  Hist: {i['macd_hist']:.4f}
  MACD cross: {macd_cross}
  Stoch %K: {i['stoch_k']:.1f}  %D: {i['stoch_d']:.1f}  ({'Overbought' if i['stoch_k'] > 80 else 'Oversold' if i['stoch_k'] < 20 else 'Neutral'})

VOLATILITY:
  ATR(14):  ${i['atr']:.2f} ({i['atr_pct']:.1f}% of price)
  BB Upper: ${i['bb_upper']:.2f}  Mid: ${i['bb_middle']:.2f}  Lower: ${i['bb_lower']:.2f}
  BB Pct:   {i['bb_pct']:.2f}  (0=lower band, 0.5=mid, 1.0=upper band)

KEY LEVELS:
  52W High: ${i['high_52w']:.2f}  ({(p - i['high_52w']) / i['high_52w'] * 100:+.1f}% from high)
  52W Low:  ${i['low_52w']:.2f}  ({(p - i['low_52w']) / i['low_52w'] * 100:+.1f}% from low)
  Resistance (20d): ${i['resistance_20d']:.2f}  Resistance (60d): ${i['resistance_60d']:.2f}
  Support (20d):    ${i['support_20d']:.2f}  Support (60d):    ${i['support_60d']:.2f}
  Volume ratio: {i['vol_ratio']:.1f}x vs 20-day avg

INTRADAY (hourly):
  {intraday_text}

HISTORICAL VOLATILITY:
  {self._hv_line(hv_data)}

Last 10 closes: {', '.join(f'${v:.2f}' for v in df['close'].tail(10).tolist())}"""
