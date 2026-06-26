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
            "momentum_1d":    round(ind.get("momentum_1d", 0), 2),
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

        # MACD histogram divergence — check 15-bar and 30-bar windows
        ind["macd_bull_div"] = False
        ind["macd_bear_div"] = False
        for window in [15, 30]:
            if len(close) >= window and len(hist) >= window:
                c_w = close.iloc[-window:]
                h_w = hist.iloc[-window:]
                mid = window // 2
                if float(c_w.iloc[-1]) < float(c_w.iloc[:mid].min()):
                    if float(h_w.iloc[-1]) > float(h_w.iloc[:mid].min()):
                        ind["macd_bull_div"] = True
                if float(c_w.iloc[-1]) > float(c_w.iloc[:mid].max()):
                    if float(h_w.iloc[-1]) < float(h_w.iloc[:mid].max()):
                        ind["macd_bear_div"] = True

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

        # ── Momentum (1d, 5d, 20d, 60d returns) ─────────────────────────────
        ind["momentum_1d"]  = (float(close.iloc[-1]) - float(close.iloc[-2]))  / float(close.iloc[-2])  * 100 if len(close) >= 2  else 0.0
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

        # Accumulation/distribution: last 5 bars — close position in daily range × volume
        acc_days = dist_days = 0
        recent5 = df.tail(5)
        vol_avg_5 = float(volume.iloc[-21:-1].mean()) if len(volume) >= 21 else float(volume.mean())
        for _, row in recent5.iterrows():
            bar_range = row["high"] - row["low"]
            if bar_range > 0 and row["volume"] > vol_avg_5 * 1.2:
                close_pos = (row["close"] - row["low"]) / bar_range
                if close_pos > 0.6:
                    acc_days += 1   # closed in upper 40% of range on high volume
                elif close_pos < 0.4:
                    dist_days += 1  # closed in lower 40% of range on high volume
        ind["acc_days"]  = acc_days
        ind["dist_days"] = dist_days

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
        adx = i.get("adx", 20.0)
        if direction == "bullish" and rsi > 78:
            fatal_flaw = f"RSI {rsi:.0f} — severely overbought, call buying chase risk"
        elif direction == "bearish" and rsi < 22:
            fatal_flaw = f"RSI {rsi:.0f} — severely oversold, put buying squeeze risk"
        elif adx < 13.0:
            fatal_flaw = f"ADX {adx:.0f} — extreme chop, no directional trend for options to profit"

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

            # ── RSI (max +1.5, non-overlapping zones) ────────────────────────
            if 50 <= rsi <= 65:
                score += 1.5;  signals.append(f"RSI {rsi:.0f} bullish momentum zone")
            elif 42 <= rsi < 50 and above_ema50:
                score += 0.75; signals.append(f"RSI {rsi:.0f} recovering in uptrend")
            elif 35 <= rsi < 42:
                score += 0.25  # oversold but not confirmed
            elif rsi > 70:
                score -= 1.5;  signals.append(f"RSI {rsi:.0f} overbought")
            elif rsi < 35 and not above_ema50:
                score -= 0.75  # oversold AND below trend — weak

            # ── MACD (max +1.5 + divergence bonus) ───────────────────────────
            if i["macd_hist"] > 0 and i["macd_hist2"] <= 0:
                score += 1.5;  signals.append("MACD bullish cross")
            elif i["macd_hist"] > 0:
                score += 0.75; signals.append("MACD above zero")
            elif i["macd_hist"] < 0:
                score -= 0.75
            if i.get("macd_bull_div"):
                score += 0.75; signals.append("MACD bullish divergence — momentum recovering")
            elif i.get("macd_bear_div"):
                score -= 0.75; signals.append("MACD bearish divergence — momentum weakening")

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

            # ── Volume + Accumulation/Distribution ────────────────────────────
            vr = i["vol_ratio"]
            acc = i.get("acc_days", 0)
            dist = i.get("dist_days", 0)
            if vr >= 2.0:
                score += 1.0;  signals.append(f"volume {vr:.1f}x surge")
            elif vr >= 1.4:
                score += 0.5;  signals.append(f"volume {vr:.1f}x above avg")
            elif vr < 0.6:
                score -= 0.5;  signals.append("low volume")
            if acc >= 2:
                score += 0.5;  signals.append(f"{acc} accumulation days — buyers absorbing supply")
            elif dist >= 2:
                score -= 0.5;  signals.append(f"{dist} distribution days — sellers in control")

            # ── ADX trend strength (scaled) ───────────────────────────────────
            adx = i["adx"]
            if adx > 35:   score += 0.5
            elif adx > 25: score += 0.25
            elif adx < 15: score -= 0.5

            # ── Stochastic oscillator (with K/D crossover detection) ──────────
            sk = i.get("stoch_k", 50)
            sd = i.get("stoch_d", 50)
            if sk < 25 and sd < 25 and sk > sd:
                score += 1.0;  signals.append(f"Stoch {sk:.0f} crossing up from oversold — bullish")
            elif sk < 25 and sd < 25:
                score += 0.75; signals.append(f"Stoch {sk:.0f}/{sd:.0f} oversold bounce zone")
            elif sk > 75 and sd > 75 and sk < sd:
                score -= 0.75; signals.append(f"Stoch {sk:.0f} crossing down from overbought")
            elif sk > 75 and sd > 75:
                score -= 0.5;  signals.append(f"Stoch {sk:.0f}/{sd:.0f} overbought")
            elif 30 <= sk <= 60 and sk > sd:
                score += 0.25  # stoch rising in healthy zone

            # ── Mean reversion: oversold in uptrend ───────────────────────────
            bb = i["bb_pct"]
            if rsi < 38 and above_ema50 and bb < 0.2:
                score += 0.75; signals.append(f"Mean reversion setup: RSI {rsi:.0f} oversold, above EMA50, near lower BB")

            # ── 52W range position ────────────────────────────────────────────
            w52 = i.get("w52_pct", 50)
            if w52 > 92 and vr >= 1.3:
                score += 0.5;  signals.append(f"52W high breakout with volume ({vr:.1f}x)")
            elif w52 > 92:
                score -= 0.75; signals.append("near 52W high — limited room, low volume")
            elif w52 < 30 and above_ema50:
                score += 0.5;  signals.append("low in 52W range, still in uptrend")

            # ── BB overextension / breakout (symmetric rewards) ───────────────
            if bb > 1.0 and vr >= 1.2:
                score += 0.75; signals.append("BB breakout with volume — momentum ignition")
            elif bb > 1.0:
                score += 0.25; signals.append("BB breakout, light volume")
            elif bb < 0.15 and above_ema50:
                score += 0.75; signals.append("lower BB tag in uptrend — pullback entry")
            elif bb < 0.3 and above_ema50:
                score += 0.25; signals.append("pullback toward lower BB in uptrend")

            # ── Multi-timeframe alignment bonus ───────────────────────────────
            if i.get("mtf_bull_aligned"):
                score += 0.75; signals.append("MTF aligned bullish (5d/20d/60d)")

            # ── Intraday timing (today's close vs yesterday's close) ──────────
            m1d = i.get("momentum_1d", 0)
            if m1d > 4.0 and rsi > 65:
                score -= 1.0; signals.append(f"overextended: +{m1d:.1f}% today, RSI {rsi:.0f}")
            elif m1d > 2.5:
                score -= 0.5; signals.append(f"up {m1d:.1f}% today — stretched entry")
            elif m1d < -3.0 and above_ema50:
                score += 0.75; signals.append(f"pullback {m1d:.1f}% in uptrend — better entry")
            elif m1d < -1.5 and above_ema50:
                score += 0.25; signals.append(f"mild pullback {m1d:.1f}% in uptrend")

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

            # ── RSI (non-overlapping bearish zones) ──────────────────────────
            if 35 <= rsi <= 50:
                score += 1.5;  signals.append(f"RSI {rsi:.0f} bearish momentum zone")
            elif 50 < rsi <= 58 and not above_ema50:
                score += 0.75; signals.append(f"RSI {rsi:.0f} overbought in downtrend")
            elif 58 < rsi <= 65 and not above_ema50:
                score += 0.25  # mildly extended
            elif rsi < 28:
                score -= 1.5;  signals.append(f"RSI {rsi:.0f} oversold — bounce risk")
            elif rsi > 65 and above_ema50:
                score -= 0.75  # strong bull momentum against short

            # ── MACD ──────────────────────────────────────────────────────────
            if i["macd_hist"] < 0 and i["macd_hist2"] >= 0:
                score += 1.5;  signals.append("MACD bearish cross")
            elif i["macd_hist"] < 0:
                score += 0.75; signals.append("MACD below zero")
            elif i["macd_hist"] > 0:
                score -= 0.75
            if i.get("macd_bear_div"):
                score += 0.75; signals.append("MACD bearish divergence — momentum weakening")
            elif i.get("macd_bull_div"):
                score -= 0.75; signals.append("MACD bullish divergence — bounce risk")

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

            # ── Volume + Accumulation/Distribution ────────────────────────────
            vr = i["vol_ratio"]
            acc = i.get("acc_days", 0)
            dist = i.get("dist_days", 0)
            if vr >= 2.0:
                score += 1.0;  signals.append(f"volume {vr:.1f}x surge")
            elif vr >= 1.4:
                score += 0.5;  signals.append(f"volume {vr:.1f}x above avg")
            elif vr < 0.6:
                score -= 0.5;  signals.append("low volume")
            if dist >= 2:
                score += 0.5;  signals.append(f"{dist} distribution days — sellers dumping")
            elif acc >= 2:
                score -= 0.5;  signals.append(f"{acc} accumulation days — buyers absorbing, short harder")

            # ── ADX ───────────────────────────────────────────────────────────
            adx = i["adx"]
            if adx > 35:   score += 0.5
            elif adx > 25: score += 0.25
            elif adx < 15: score -= 0.5

            # ── Stochastic oscillator (with K/D crossover detection) ──────────
            sk = i.get("stoch_k", 50)
            sd = i.get("stoch_d", 50)
            if sk > 75 and sd > 75 and sk < sd:
                score += 1.0;  signals.append(f"Stoch {sk:.0f} crossing down from overbought — bearish")
            elif sk > 75 and sd > 75:
                score += 0.75; signals.append(f"Stoch {sk:.0f}/{sd:.0f} overbought — put play")
            elif sk < 25 and sd < 25 and sk > sd:
                score -= 0.75; signals.append(f"Stoch {sk:.0f} crossing up from oversold — bounce risk")
            elif sk < 25 and sd < 25:
                score -= 0.5;  signals.append(f"Stoch {sk:.0f}/{sd:.0f} oversold — bounce risk")
            elif 40 <= sk <= 70 and sk < sd:
                score += 0.25  # stoch falling in sell zone

            # ── Mean reversion: overbought in downtrend ───────────────────────
            bb_b = i["bb_pct"]
            if rsi > 62 and not above_ema50 and bb_b > 0.8:
                score += 0.75; signals.append(f"Mean reversion setup: RSI {rsi:.0f} overbought, below EMA50, near upper BB")

            # ── 52W range position ────────────────────────────────────────────
            w52  = i.get("w52_pct", 50)
            vr_b = i["vol_ratio"]
            if w52 < 8 and vr_b >= 1.3:
                score += 0.5;  signals.append(f"52W low breakdown with volume ({vr_b:.1f}x) — momentum short")
            elif w52 < 8:
                score -= 0.75; signals.append("near 52W low — bounce risk, low volume")
            elif w52 > 70 and not above_ema50:
                score += 0.5;  signals.append("high in 52W range, below EMAs")

            # ── BB breakdown / overextension (symmetric) ──────────────────────
            if bb_b < 0.0 and vr_b >= 1.2:
                score += 0.75; signals.append("BB breakdown with volume — momentum short")
            elif bb_b < 0.0:
                score += 0.25; signals.append("BB breakdown, light volume")
            elif bb_b > 0.85 and not above_ema50:
                score += 0.75; signals.append("upper BB tag in downtrend — short entry")
            elif bb_b > 0.7 and not above_ema50:
                score += 0.25; signals.append("overbought in downtrend")

            # ── Intraday timing (bearish) ─────────────────────────────────────
            m1d_b = i.get("momentum_1d", 0)
            if m1d_b < -4.0 and rsi < 35:
                score -= 1.0; signals.append(f"oversold: {m1d_b:.1f}% today, RSI {rsi:.0f} — risk of bounce")
            elif m1d_b < -2.5:
                score -= 0.5; signals.append(f"down {m1d_b:.1f}% today — chasing the drop")
            elif m1d_b > 3.0 and not above_ema50:
                score += 0.75; signals.append(f"dead-cat bounce {m1d_b:+.1f}% in downtrend — better put entry")
            elif m1d_b > 1.5 and not above_ema50:
                score += 0.25; signals.append(f"mild bounce {m1d_b:+.1f}% in downtrend")

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
