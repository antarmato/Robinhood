"""
Technical Analysis Agent — pure Python scoring, no LLM.

Computes EMAs, RSI, MACD, ADX, Bollinger Bands, relative strength vs SPY,
momentum acceleration, and 52W range position from Alpaca OHLCV.
Scores 1-10; tuned so a genuinely strong setup earns 7-8, exceptional is 9-10.
"""

import logging
from typing import Optional

import numpy as np
import pandas as pd

from .base import BaseAgent, BroadcastFn
from .. import market_data as md

logger = logging.getLogger(__name__)

# SPY daily bars are fetched once and cached here for the scan cycle
_SPY_CACHE: dict = {}


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

        spy_df = self._get_spy_bars()
        ind = self._compute_indicators(df, spy_df)
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
            "rsi":            round(ind["rsi"], 1),
            "adx":            round(ind["adx"], 1),
            "bb_pct":         round(ind["bb_pct"], 2),
            "vol_ratio":      round(ind["vol_ratio"], 2),
            "ema20_slope":    round(ind["ema20_slope"], 3),
            "rs_5d":          round(ind.get("rs_5d", 0), 2),
            "momentum_5d":    round(ind.get("momentum_5d", 0), 2),
            "momentum_60d":   round(ind.get("momentum_60d", 0), 2),
            "w52_pct":        round(ind.get("w52_pct", 50), 1),
            "avg_vol_20d":    round(ind.get("avg_vol_20d", 0)),
            "stoch_k":        round(ind.get("stoch_k", 50), 1),
            "stoch_d":        round(ind.get("stoch_d", 50), 1),
            "vwap20":         round(ind.get("vwap20", 0), 2),
            "vwap20_pct":     round(ind.get("vwap20_pct", 0), 2),
            "ema200":         round(ind.get("ema200", 0), 2),
            "above_ema200":   ind.get("price", 0) > ind.get("ema200", 0),
        }
        await self._emit("score", {"symbol": symbol, "score": score, "trend": trend,
                                    "signals": signals, "fatal_flaw": fatal_flaw})
        return result

    def _get_spy_bars(self) -> Optional[pd.DataFrame]:
        global _SPY_CACHE
        try:
            df = md.get_historicals("SPY", period="3mo")
            if not df.empty and len(df) >= 20:
                return df
        except Exception:
            pass
        return None

    # ── Indicator computation ─────────────────────────────────────────────────

    def _compute_indicators(self, df: pd.DataFrame, spy_df: Optional[pd.DataFrame] = None) -> dict:
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
        rng = ind["high_52w"] - ind["low_52w"]
        ind["w52_pct"] = (price - ind["low_52w"]) / rng * 100 if rng > 0 else 50.0

        vol_avg20 = float(volume.iloc[-21:-1].mean()) if len(volume) >= 21 else float(volume.mean())
        ind["vol_ratio"]   = float(volume.iloc[-1]) / vol_avg20 if vol_avg20 > 0 else 1.0
        ind["avg_vol_20d"] = vol_avg20  # absolute avg daily volume (shares)

        # 20-day VWAP proxy: (H+L+C)/3 × volume, cumulated over 20 days
        typical = (high + low + close) / 3
        recent20 = df.tail(20)
        tp20   = (recent20["high"] + recent20["low"] + recent20["close"]) / 3
        vol20  = recent20["volume"]
        total_vol = float(vol20.sum())
        ind["vwap20"] = float((tp20 * vol20).sum() / total_vol) if total_vol > 0 else float(close.iloc[-1])
        ind["vwap20_pct"] = (price - ind["vwap20"]) / ind["vwap20"] * 100  # % above/below VWAP

        # ── Momentum (5d, 20d, 60d returns) ──────────────────────────────────
        ind["momentum_5d"]  = (float(close.iloc[-1]) - float(close.iloc[-5]))  / float(close.iloc[-5])  * 100 if len(close) >= 5  else 0.0
        ind["momentum_20d"] = (float(close.iloc[-1]) - float(close.iloc[-20])) / float(close.iloc[-20]) * 100 if len(close) >= 20 else 0.0
        ind["momentum_60d"] = (float(close.iloc[-1]) - float(close.iloc[-60])) / float(close.iloc[-60]) * 100 if len(close) >= 60 else 0.0

        # Acceleration: is recent 5d move stronger than expected slice of 20d?
        expected_5d = ind["momentum_20d"] / 4.0
        ind["momentum_accel"] = ind["momentum_5d"] - expected_5d

        # Multi-timeframe alignment: all 3 timeframes agree on direction
        ind["mtf_bull_aligned"] = (ind["momentum_5d"] > 0
                                   and ind["momentum_20d"] > 0
                                   and ind["momentum_60d"] > 0)
        ind["mtf_bear_aligned"] = (ind["momentum_5d"] < 0
                                   and ind["momentum_20d"] < 0
                                   and ind["momentum_60d"] < 0)

        # ── Relative strength vs SPY ───────────────────────────────────────────
        ind["rs_5d"]  = 0.0
        ind["rs_20d"] = 0.0
        if spy_df is not None and len(spy_df) >= 20:
            spy_close = spy_df["close"]
            try:
                spy_5d  = (float(spy_close.iloc[-1]) - float(spy_close.iloc[-5]))  / float(spy_close.iloc[-5])  * 100
                spy_20d = (float(spy_close.iloc[-1]) - float(spy_close.iloc[-20])) / float(spy_close.iloc[-20]) * 100
                if len(close) >= 5:
                    ind["rs_5d"]  = ind["momentum_5d"]  - spy_5d
                if len(close) >= 20:
                    ind["rs_20d"] = ind["momentum_20d"] - spy_20d
            except Exception:
                pass

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

        Calibration goal: strong trending stock with good RS = 7-8.
        Exceptional alignment (RS dominant, ADX strong, clean MACD cross) = 9-10.
        Ranging or conflicted = 5-6. Strong counter-signals = 3-4.
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

        above_ema20  = price > i["ema20"]
        above_ema50  = price > i["ema50"]
        above_ema200 = price > i.get("ema200", price * 0.8)  # default to always true if missing

        # ── EMA200 (long-term trend, both directions) ─────────────────────────
        if direction == "bullish":
            if above_ema200:
                score += 0.5; signals.append("above EMA200 — long-term bull structure")
            else:
                score -= 1.0; signals.append("below EMA200 — long-term bear structure")
        else:
            if not above_ema200:
                score += 0.5; signals.append("below EMA200 — long-term bear structure")
            else:
                score -= 1.0; signals.append("above EMA200 — long-term bull structure headwind")

        # ── Liquidity check (both directions) ────────────────────────────────
        avg_vol = i.get("avg_vol_20d", 1_000_000)
        if avg_vol < 500_000:
            score -= 1.5; signals.append(f"thin volume {avg_vol/1e6:.1f}M avg — wide option spreads")
        elif avg_vol < 1_000_000:
            score -= 0.5; signals.append(f"moderate volume {avg_vol/1e6:.1f}M avg")

        # ── 20-day VWAP (both directions) ─────────────────────────────────────
        vwap_pct = i.get("vwap20_pct", 0)  # % above (positive) or below (negative) 20d VWAP

        if direction == "bullish":
            # ── VWAP position (20d proxy) ─────────────────────────────────────
            if vwap_pct > 3.0:
                score += 0.75; signals.append(f"above 20d VWAP +{vwap_pct:.1f}% — buyers in control")
            elif vwap_pct > 0.5:
                score += 0.25
            elif vwap_pct < -3.0:
                score -= 0.75; signals.append(f"below 20d VWAP {vwap_pct:.1f}% — sellers in control")
            elif vwap_pct < -0.5:
                score -= 0.25

            # ── EMA alignment (max +1.5, not +2.0) ───────────────────────────
            if above_ema20 and above_ema50:
                score += 1.5;  signals.append("above both EMAs")
            elif above_ema20:
                score += 0.5;  signals.append("above EMA20")
            elif not above_ema20 and not above_ema50:
                score -= 1.5;  signals.append("below both EMAs")

            # ── EMA slope (scaled) ────────────────────────────────────────────
            slope = i["ema20_slope"]
            if slope > 0.4:    score += 0.75; signals.append(f"EMA20 rising +{slope:.2f}%/wk")
            elif slope > 0.15: score += 0.25
            elif slope < -0.4: score -= 0.75
            elif slope < -0.15:score -= 0.25

            # ── RSI (max +1.5) ────────────────────────────────────────────────
            if 45 <= rsi <= 65:
                score += 1.5;  signals.append(f"RSI {rsi:.0f} bullish zone")
            elif 35 <= rsi < 45 and above_ema50:
                score += 0.75; signals.append(f"RSI {rsi:.0f} recovering")
            elif rsi > 70:
                score -= 1.5;  signals.append(f"RSI {rsi:.0f} overbought")
            elif rsi < 35:
                score -= 0.75

            # ── MACD (max +1.5) ───────────────────────────────────────────────
            if i["macd_hist"] > 0 and i["macd_hist2"] <= 0:
                score += 1.5;  signals.append("MACD bullish cross")
            elif i["macd_hist"] > 0:
                score += 0.75; signals.append("MACD above zero")
            elif i["macd_hist"] < 0:
                score -= 0.75

            # ── Relative strength vs SPY (key differentiator) ─────────────────
            rs = i.get("rs_5d", 0)
            if rs > 4:
                score += 1.5;  signals.append(f"RS +{rs:.1f}% vs SPY")
            elif rs > 2:
                score += 0.75; signals.append(f"RS +{rs:.1f}% vs SPY")
            elif rs < -3:
                score -= 1.0;  signals.append(f"lagging SPY {rs:.1f}%")
            elif rs < -1:
                score -= 0.5

            # ── Momentum acceleration ─────────────────────────────────────────
            accel = i.get("momentum_accel", 0)
            if accel > 2:
                score += 0.5;  signals.append("momentum accelerating")
            elif accel < -2:
                score -= 0.5;  signals.append("momentum decelerating")

            # ── Volume ────────────────────────────────────────────────────────
            vr = i["vol_ratio"]
            if vr >= 2.0:
                score += 1.0;  signals.append(f"volume {vr:.1f}x surge")
            elif vr >= 1.4:
                score += 0.5;  signals.append(f"volume {vr:.1f}x above avg")
            elif vr < 0.6:
                score -= 0.5;  signals.append("low volume")

            # ── ADX trend strength (scaled) ───────────────────────────────────
            adx = i["adx"]
            if adx > 35:   score += 0.5
            elif adx > 25: score += 0.25
            elif adx < 15: score -= 0.5

            # ── Stochastic oscillator ─────────────────────────────────────────
            sk = i.get("stoch_k", 50)
            sd = i.get("stoch_d", 50)
            if sk < 25 and sd < 25:
                score += 0.75; signals.append(f"Stoch {sk:.0f}/{sd:.0f} oversold bounce zone")
            elif sk > 75 and sd > 75:
                score -= 0.5;  signals.append(f"Stoch {sk:.0f}/{sd:.0f} overbought")
            elif 30 <= sk <= 60 and sk > sd:
                score += 0.25  # stoch rising in healthy zone

            # ── 52W range position ────────────────────────────────────────────
            w52 = i.get("w52_pct", 50)
            if w52 > 92:
                score -= 0.75; signals.append("near 52W high — limited room")
            elif w52 < 30 and above_ema50:
                score += 0.5;  signals.append("low in 52W range, still in uptrend")

            # ── BB overextension ──────────────────────────────────────────────
            if i["bb_pct"] > 0.92:
                score -= 1.0;  signals.append("near upper BB — stretched")
            elif i["bb_pct"] < 0.3 and above_ema50:
                score += 0.5;  signals.append("pullback in uptrend")

            # ── Multi-timeframe alignment bonus ───────────────────────────────
            if i.get("mtf_bull_aligned"):
                score += 0.75; signals.append("MTF aligned bullish (5d/20d/60d)")

            trend = (
                "strong_uptrend" if i["adx"] > 25 and above_ema50 else
                "uptrend"        if above_ema20 and above_ema50 else
                "ranging"        if abs(i["ema20_slope"]) < 0.1 else
                "downtrend"
            )

        else:  # bearish
            # ── VWAP position ─────────────────────────────────────────────────
            if vwap_pct < -3.0:
                score += 0.75; signals.append(f"below 20d VWAP {vwap_pct:.1f}% — sellers in control")
            elif vwap_pct < -0.5:
                score += 0.25
            elif vwap_pct > 3.0:
                score -= 0.75; signals.append(f"above 20d VWAP +{vwap_pct:.1f}% — bull momentum headwind")
            elif vwap_pct > 0.5:
                score -= 0.25

            # ── EMA alignment ─────────────────────────────────────────────────
            if not above_ema20 and not above_ema50:
                score += 1.5;  signals.append("below both EMAs")
            elif not above_ema20:
                score += 0.5;  signals.append("below EMA20")
            elif above_ema20 and above_ema50:
                score -= 1.5;  signals.append("above both EMAs")

            slope = i["ema20_slope"]
            if slope < -0.4:   score += 0.75; signals.append(f"EMA20 falling {slope:.2f}%/wk")
            elif slope < -0.15:score += 0.25
            elif slope > 0.4:  score -= 0.75
            elif slope > 0.15: score -= 0.25

            # ── RSI ───────────────────────────────────────────────────────────
            if 32 <= rsi <= 55:
                score += 1.5;  signals.append(f"RSI {rsi:.0f} bearish zone")
            elif 55 < rsi <= 65 and not above_ema50:
                score += 0.75; signals.append(f"RSI {rsi:.0f} overbought in downtrend")
            elif rsi < 28:
                score -= 1.5;  signals.append(f"RSI {rsi:.0f} oversold")
            elif rsi > 65:
                score -= 0.75

            # ── MACD ──────────────────────────────────────────────────────────
            if i["macd_hist"] < 0 and i["macd_hist2"] >= 0:
                score += 1.5;  signals.append("MACD bearish cross")
            elif i["macd_hist"] < 0:
                score += 0.75; signals.append("MACD below zero")
            elif i["macd_hist"] > 0:
                score -= 0.75

            # ── Relative strength vs SPY (bearish: underperformance is good) ──
            rs = i.get("rs_5d", 0)
            if rs < -4:
                score += 1.5;  signals.append(f"lagging SPY {rs:.1f}% — bearish RS")
            elif rs < -2:
                score += 0.75; signals.append(f"lagging SPY {rs:.1f}%")
            elif rs > 3:
                score -= 1.0;  signals.append(f"outperforming SPY +{rs:.1f}% — bearish headwind")
            elif rs > 1:
                score -= 0.5

            # ── Momentum acceleration ─────────────────────────────────────────
            accel = i.get("momentum_accel", 0)
            if accel < -2:
                score += 0.5;  signals.append("downside momentum accelerating")
            elif accel > 2:
                score -= 0.5;  signals.append("downside momentum weakening")

            # ── Volume ────────────────────────────────────────────────────────
            vr = i["vol_ratio"]
            if vr >= 2.0:
                score += 1.0;  signals.append(f"volume {vr:.1f}x surge")
            elif vr >= 1.4:
                score += 0.5;  signals.append(f"volume {vr:.1f}x above avg")
            elif vr < 0.6:
                score -= 0.5;  signals.append("low volume")

            # ── ADX ───────────────────────────────────────────────────────────
            adx = i["adx"]
            if adx > 35:   score += 0.5
            elif adx > 25: score += 0.25
            elif adx < 15: score -= 0.5

            # ── Stochastic oscillator ─────────────────────────────────────────
            sk = i.get("stoch_k", 50)
            sd = i.get("stoch_d", 50)
            if sk > 75 and sd > 75:
                score += 0.75; signals.append(f"Stoch {sk:.0f}/{sd:.0f} overbought — put play")
            elif sk < 25 and sd < 25:
                score -= 0.5;  signals.append(f"Stoch {sk:.0f}/{sd:.0f} oversold — bounce risk")
            elif 40 <= sk <= 70 and sk < sd:
                score += 0.25  # stoch falling in sell zone

            # ── 52W range position ────────────────────────────────────────────
            w52 = i.get("w52_pct", 50)
            if w52 < 8:
                score -= 0.75; signals.append("near 52W low — bounce risk")
            elif w52 > 70 and not above_ema50:
                score += 0.5;  signals.append("high in 52W range, below EMAs")

            # ── BB ────────────────────────────────────────────────────────────
            if i["bb_pct"] < 0.08:
                score -= 1.0;  signals.append("near lower BB — oversold")
            elif i["bb_pct"] > 0.7 and not above_ema50:
                score += 0.5;  signals.append("overbought in downtrend")

            # ── Multi-timeframe alignment bonus ───────────────────────────────
            if i.get("mtf_bear_aligned"):
                score += 0.75; signals.append("MTF aligned bearish (5d/20d/60d)")

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
