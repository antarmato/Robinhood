"""
Technical Analysis Agent — pure Python scoring, no LLM.

Computes EMAs, RSI, MACD, ADX, Stochastic, Bollinger Bands from Polygon OHLCV.
Scores 1-10 based on how many indicators confirm the proposed direction.
"""

import logging
from typing import Optional

import numpy as np
import pandas as pd

from .base import BaseAgent, BroadcastFn
from .. import market_data as md

logger = logging.getLogger(__name__)


class TechnicalAgent(BaseAgent):
    def __init__(self, client, broadcast: Optional[BroadcastFn] = None):
        super().__init__(client, "Technical", broadcast=broadcast)

    async def analyze(self, symbol: str, direction: str) -> dict:
        await self._emit("status", f"Technical: scoring {symbol} ({direction})...")

        df = md.get_historicals(symbol, period="1y")
        if df.empty or len(df) < 30:
            df = md.get_historicals(symbol, period="6mo")
        if df.empty or len(df) < 30:
            df = md.get_historicals(symbol, period="3mo")
        if df.empty or len(df) < 20:
            result = {
                "score": 5, "signal": "neutral", "trend": "ranging",
                "rsi_reading": "N/A", "macd_reading": "N/A",
                "key_support": 0, "key_resistance": 0,
                "summary": "Insufficient data.",
                "current_price": 0, "signals": [],
                "fatal_flaw": None,
            }
            await self._emit("score", result)
            return result

        ind = self._compute_indicators(df)
        score, trend, signals, fatal_flaw = self._score_direction(ind, direction)

        result = {
            "score":          score,
            "signal":         direction if score >= 6 else ("neutral" if score >= 4 else "opposite"),
            "trend":          trend,
            "rsi_reading":    f"{ind['rsi']:.1f}",
            "macd_reading":   self._macd_label(ind),
            "key_support":    round(ind["support_20d"], 2),
            "key_resistance": round(ind["resistance_20d"], 2),
            "current_price":  round(ind["price"], 2),
            "signals":        signals,
            "fatal_flaw":     fatal_flaw,
            "summary":        f"{trend} | {', '.join(signals[:4]) if signals else 'no clear signals'}",
            # pass raw indicators for Judge context
            "rsi":            round(ind["rsi"], 1),
            "adx":            round(ind["adx"], 1),
            "bb_pct":         round(ind["bb_pct"], 2),
            "vol_ratio":      round(ind["vol_ratio"], 2),
            "ema20_slope":    round(ind["ema20_slope"], 3),
        }
        await self._emit("score", {"symbol": symbol, "score": score, "trend": trend,
                                    "signals": signals, "fatal_flaw": fatal_flaw})
        return result

    # ── Indicator computation ─────────────────────────────────────────────────

    def _compute_indicators(self, df: pd.DataFrame) -> dict:
        close  = df["close"]
        high   = df["high"]
        low    = df["low"]
        volume = df["volume"]
        ind    = {}

        ind["ema9"]   = float(close.ewm(span=9,   adjust=False).mean().iloc[-1])
        ind["ema20"]  = float(close.ewm(span=20,  adjust=False).mean().iloc[-1])
        ind["ema50"]  = float(close.ewm(span=50,  adjust=False).mean().iloc[-1])
        ind["ema200"] = float(close.ewm(span=200, adjust=False).mean().iloc[-1])

        ema20_s = close.ewm(span=20, adjust=False).mean()
        ind["ema20_slope"] = ((float(ema20_s.iloc[-1]) - float(ema20_s.iloc[-5]))
                              / float(ema20_s.iloc[-5]) * 100) if len(ema20_s) >= 5 else 0.0

        delta = close.diff()
        gain  = delta.where(delta > 0, 0.0).rolling(14).mean()
        loss  = (-delta.where(delta < 0, 0.0)).rolling(14).mean()
        rs    = gain / loss.replace(0, np.nan)
        rsi_s = 100 - 100 / (1 + rs)
        ind["rsi"]      = float(rsi_s.iloc[-1])
        ind["rsi_prev"] = float(rsi_s.iloc[-5]) if len(rsi_s) >= 5 else ind["rsi"]
        if np.isnan(ind["rsi"]): ind["rsi"] = 50.0

        ema12 = close.ewm(span=12, adjust=False).mean()
        ema26 = close.ewm(span=26, adjust=False).mean()
        macd  = ema12 - ema26
        sig   = macd.ewm(span=9, adjust=False).mean()
        hist  = macd - sig
        ind["macd"]       = float(macd.iloc[-1])
        ind["macd_sig"]   = float(sig.iloc[-1])
        ind["macd_hist"]  = float(hist.iloc[-1])
        ind["macd_hist2"] = float(hist.iloc[-2]) if len(hist) >= 2 else float(hist.iloc[-1])

        low14  = low.rolling(14).min()
        high14 = high.rolling(14).max()
        k_raw  = (close - low14) / (high14 - low14).replace(0, np.nan) * 100
        ind["stoch_k"] = float(k_raw.rolling(3).mean().iloc[-1])
        ind["stoch_d"] = float(k_raw.rolling(3).mean().rolling(3).mean().iloc[-1])

        tr  = pd.concat(
            [(high - low), (high - close.shift()).abs(), (low - close.shift()).abs()], axis=1
        ).max(axis=1)
        atr = tr.rolling(14).mean()
        ind["atr"]     = float(atr.iloc[-1])
        ind["atr_pct"] = ind["atr"] / float(close.iloc[-1]) * 100

        ind["adx"] = self._compute_adx(high, low, close)

        sma20 = close.rolling(20).mean()
        std20 = close.rolling(20).std()
        ind["bb_upper"]  = float((sma20 + 2 * std20).iloc[-1])
        ind["bb_lower"]  = float((sma20 - 2 * std20).iloc[-1])
        ind["bb_middle"] = float(sma20.iloc[-1])
        price = float(close.iloc[-1])
        bw = ind["bb_upper"] - ind["bb_lower"]
        ind["bb_pct"] = (price - ind["bb_lower"]) / bw if bw > 0 else 0.5

        recent20 = df.tail(20)
        recent60 = df.tail(60)
        ind["resistance_20d"] = float(recent20["high"].max())
        ind["support_20d"]    = float(recent20["low"].min())
        ind["resistance_60d"] = float(recent60["high"].max())
        ind["support_60d"]    = float(recent60["low"].min())

        lookback = min(252, len(close))
        ind["high_52w"] = float(close.tail(lookback).max())
        ind["low_52w"]  = float(close.tail(lookback).min())

        vol_avg20 = float(volume.iloc[-21:-1].mean()) if len(volume) >= 21 else float(volume.mean())
        ind["vol_ratio"] = float(volume.iloc[-1]) / vol_avg20 if vol_avg20 > 0 else 1.0

        ind["price"] = price
        return ind

    def _compute_adx(self, high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> float:
        try:
            up_move   = high.diff()
            down_move = -low.diff()
            plus_dm   = up_move.where((up_move > down_move) & (up_move > 0), 0.0)
            minus_dm  = down_move.where((down_move > up_move) & (down_move > 0), 0.0)
            tr = pd.concat(
                [(high - low), (high - close.shift()).abs(), (low - close.shift()).abs()], axis=1
            ).max(axis=1)
            atr14    = tr.rolling(period).mean()
            plus_di  = (plus_dm.rolling(period).mean() / atr14).replace([np.inf, -np.inf], 0) * 100
            minus_di = (minus_dm.rolling(period).mean() / atr14).replace([np.inf, -np.inf], 0) * 100
            dx  = ((plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan) * 100)
            adx = dx.rolling(period).mean()
            val = float(adx.iloc[-1])
            return val if not np.isnan(val) else 20.0
        except Exception:
            return 20.0

    # ── Pure Python scoring ───────────────────────────────────────────────────

    def _score_direction(
        self, i: dict, direction: str
    ) -> tuple[float, str, list[str], Optional[str]]:
        """
        Returns (score 1-10, trend label, contributing signals list, fatal_flaw or None).

        Fatal flaws (hard rejection before any LLM call):
          - Calls: RSI > 78 (overbought — chase risk)
          - Puts:  RSI < 22 (oversold — squeeze risk)
        """
        price = i["price"]
        score = 5.0
        signals: list[str] = []
        fatal_flaw = None

        # ── Fatal flaw check ──────────────────────────────────────────────────
        rsi = i["rsi"]
        if direction == "bullish" and rsi > 78:
            fatal_flaw = f"RSI {rsi:.0f} — severely overbought, call buying chase risk"
        elif direction == "bearish" and rsi < 22:
            fatal_flaw = f"RSI {rsi:.0f} — severely oversold, put buying squeeze risk"

        above_ema20 = price > i["ema20"]
        above_ema50 = price > i["ema50"]

        if direction == "bullish":
            # EMA alignment
            if above_ema20 and above_ema50:
                score += 2.0;  signals.append("above both EMAs")
            elif above_ema20:
                score += 0.5;  signals.append("above EMA20")
            elif not above_ema20 and not above_ema50:
                score -= 2.0;  signals.append("below both EMAs")

            # EMA20 slope
            if i["ema20_slope"] > 0.2:   score += 0.5
            elif i["ema20_slope"] < -0.2: score -= 0.5

            # RSI
            if 45 <= rsi <= 68:
                score += 1.5;  signals.append(f"RSI {rsi:.0f} bullish zone")
            elif 35 <= rsi < 45 and above_ema50:
                score += 1.0;  signals.append(f"RSI {rsi:.0f} recovering in uptrend")
            elif rsi > 75:
                score -= 2.0;  signals.append(f"RSI {rsi:.0f} overbought")
            elif rsi < 35:
                score -= 1.0

            # MACD
            if i["macd_hist"] > 0 and i["macd_hist2"] <= 0:
                score += 2.0;  signals.append("MACD bullish cross")
            elif i["macd_hist"] > 0:
                score += 1.0;  signals.append("MACD above zero")
            elif i["macd_hist"] < 0:
                score -= 1.0

            # Volume confirmation
            if i["vol_ratio"] >= 1.5:
                score += 1.0;  signals.append(f"volume {i['vol_ratio']:.1f}x")
            elif i["vol_ratio"] < 0.7:
                score -= 0.5

            # ADX trend strength
            if i["adx"] > 25:    score += 0.5
            elif i["adx"] < 15:  score -= 0.5

            # BB — check for overextension
            if i["bb_pct"] > 0.92:
                score -= 1.5;  signals.append("near upper BB — stretched")
            elif i["bb_pct"] < 0.3 and above_ema50:
                score += 0.5;  signals.append("pullback in uptrend")

            trend = (
                "strong_uptrend"   if i["adx"] > 25 and above_ema50 else
                "uptrend"          if above_ema20 and above_ema50 else
                "ranging"          if abs(i["ema20_slope"]) < 0.1 else
                "downtrend"
            )

        else:  # bearish
            # EMA alignment
            if not above_ema20 and not above_ema50:
                score += 2.0;  signals.append("below both EMAs")
            elif not above_ema20:
                score += 0.5;  signals.append("below EMA20")
            elif above_ema20 and above_ema50:
                score -= 2.0;  signals.append("above both EMAs")

            if i["ema20_slope"] < -0.2:   score += 0.5
            elif i["ema20_slope"] > 0.2:  score -= 0.5

            # RSI
            if 30 <= rsi <= 55:
                score += 1.5;  signals.append(f"RSI {rsi:.0f} bearish zone")
            elif 55 < rsi <= 65 and not above_ema50:
                score += 1.0;  signals.append(f"RSI {rsi:.0f} overbought in downtrend")
            elif rsi < 25:
                score -= 2.0;  signals.append(f"RSI {rsi:.0f} oversold")
            elif rsi > 65:
                score -= 1.0

            # MACD
            if i["macd_hist"] < 0 and i["macd_hist2"] >= 0:
                score += 2.0;  signals.append("MACD bearish cross")
            elif i["macd_hist"] < 0:
                score += 1.0;  signals.append("MACD below zero")
            elif i["macd_hist"] > 0:
                score -= 1.0

            # Volume
            if i["vol_ratio"] >= 1.5:
                score += 1.0;  signals.append(f"volume {i['vol_ratio']:.1f}x")
            elif i["vol_ratio"] < 0.7:
                score -= 0.5

            # ADX
            if i["adx"] > 25:    score += 0.5
            elif i["adx"] < 15:  score -= 0.5

            # BB
            if i["bb_pct"] < 0.08:
                score -= 1.5;  signals.append("near lower BB — oversold")
            elif i["bb_pct"] > 0.7 and not above_ema50:
                score += 0.5;  signals.append("overbought in downtrend")

            trend = (
                "strong_downtrend" if i["adx"] > 25 and not above_ema50 else
                "downtrend"        if not above_ema20 and not above_ema50 else
                "ranging"          if abs(i["ema20_slope"]) < 0.1 else
                "uptrend"
            )

        score = round(max(1.0, min(10.0, score)), 1)
        return score, trend, signals, fatal_flaw

    def _macd_label(self, i: dict) -> str:
        if i["macd_hist"] > 0 and i["macd_hist2"] <= 0:
            return "bullish_cross"
        if i["macd_hist"] < 0 and i["macd_hist2"] >= 0:
            return "bearish_cross"
        if i["macd_hist"] > 0:
            return "bullish_above_zero"
        if i["macd_hist"] < 0:
            return "bearish_below_zero"
        return "neutral"
