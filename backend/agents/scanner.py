"""
Scanner Agent — identifies the best 1-3 option trade candidates from the watchlist.

Architecture:
  1. Fetch each symbol individually via yf.Ticker.history() (no MultiIndex issues)
  2. Run all fetches concurrently via asyncio.gather + run_in_executor
  3. Compute 12 quantitative signals per symbol from 3-month daily data
  4. Score each symbol independently for BULL (calls) and BEAR (puts)
  5. Claude Haiku gets the pre-scored table and selects top 1-3 with direction
  6. Auto-select fallback if LLM returns bad JSON
"""

import asyncio
import logging
from typing import Optional

import anthropic
import numpy as np
import pandas as pd

from .base import BaseAgent, BroadcastFn
from .. import market_data as md

logger = logging.getLogger(__name__)


class ScannerAgent(BaseAgent):
    def __init__(
        self,
        client: anthropic.AsyncAnthropic,
        watchlist: list[str],
        broadcast: Optional[BroadcastFn] = None,
    ):
        super().__init__(client, "Scanner", model="claude-haiku-4-5-20251001", broadcast=broadcast)
        self.watchlist = watchlist

    # ── Public entry point ─────────────────────────────────────────────────────

    async def scan(self) -> list[dict]:
        await self._emit("status", f"Fetching market data for {len(self.watchlist)} symbols...")

        # Fetch each symbol concurrently (individual Ticker.history() calls — no MultiIndex)
        loop = asyncio.get_event_loop()
        tasks = [
            loop.run_in_executor(None, self._fetch_one, sym)
            for sym in self.watchlist
        ]
        raw_results = await asyncio.gather(*tasks, return_exceptions=True)

        scored: dict = {}
        failed: list[str] = []
        for sym, r in zip(self.watchlist, raw_results):
            if isinstance(r, dict) and r:
                scored[sym] = r
            else:
                failed.append(sym)
                if isinstance(r, Exception):
                    logger.error(f"Fetch error for {sym}: {r}")

        if failed:
            await self._emit("status", f"No data for {len(failed)} symbol(s): {failed}. Got data for {len(scored)}.")

        if not scored:
            await self._emit("status", "No market data returned — yfinance may be blocked or rate-limited on Railway.")
            return []

        await self._emit("status", f"Scored {len(scored)}/{len(self.watchlist)} symbols. Selecting best setups...")
        summary = self._build_summary(scored)

        # LLM selects top 1-3 candidates
        raw = await self._call(
            self._system_prompt(),
            [{"role": "user", "content": summary + "\n\nAlways return at least 1 candidate."}],
            max_tokens=700,
            stream=False,
        )
        candidates = self._parse_json(raw)
        if not isinstance(candidates, list) or len(candidates) == 0:
            await self._emit("status", "LLM returned no candidates — using auto-select fallback.")
            candidates = self._auto_select(scored)

        # Enrich with computed data
        for c in candidates:
            sym = c.get("symbol", "")
            if sym in scored:
                d = scored[sym]
                c["current_price"] = d["price"]
                c["bull_score"]    = d["bull_score"]
                c["bear_score"]    = d["bear_score"]
                c["volume_ratio"]  = d["volume_ratio"]
                c["rsi"]           = d["rsi"]
                c["pct_change"]    = d["pct_change"]
                c["iv_rank"]       = None

        cand_list = [c.get("symbol", "") + " " + c.get("direction", "") for c in candidates]
        await self._emit("status", f"Found {len(candidates)} candidate(s): {cand_list}")
        return candidates

    # ── Data fetch — one symbol at a time, no MultiIndex ──────────────────────

    def _fetch_one(self, sym: str) -> Optional[dict]:
        """
        Fetch 3 months of daily OHLCV for one symbol using yf.Ticker.history().
        Returns computed signal dict or None on failure.
        Called from run_in_executor — must be synchronous.
        """
        import yfinance as yf
        try:
            t = yf.Ticker(sym)
            # history() returns flat columns: Open, High, Low, Close, Volume, Dividends, Stock Splits
            df = t.history(period="3mo", interval="1d", auto_adjust=True)
            if df is None or df.empty:
                logger.warning(f"{sym}: empty history from yfinance")
                return None
            if len(df) < 20:
                logger.warning(f"{sym}: only {len(df)} rows — need 20+")
                return None

            # Normalize columns to lowercase
            df.columns = [c.lower() for c in df.columns]
            needed = [c for c in ["open", "close", "high", "low", "volume"] if c in df.columns]
            if "close" not in needed:
                logger.warning(f"{sym}: no close column in {list(df.columns)}")
                return None
            df = df[needed].dropna(subset=["close"])
            if len(df) < 20:
                return None

            result = self._compute_signals(sym, df)
            if result:
                logger.debug(f"{sym}: bull={result['bull_score']} bear={result['bear_score']} rsi={result['rsi']:.1f}")
            return result

        except Exception as e:
            logger.error(f"_fetch_one({sym}): {type(e).__name__}: {e}")
            return None

    # ── Signal computation ─────────────────────────────────────────────────────

    def _compute_signals(self, sym: str, df: pd.DataFrame) -> Optional[dict]:
        """Compute all quantitative signals for one symbol from OHLCV DataFrame."""
        try:
            close  = df["close"]
            volume = df["volume"] if "volume" in df.columns else pd.Series([1] * len(df), index=df.index)
            high   = df["high"]   if "high"   in df.columns else close
            low    = df["low"]    if "low"    in df.columns else close

            price = float(close.iloc[-1])
            if price <= 0:
                return None

            # ── EMAs ─────────────────────────────────────────────────────────
            ema20 = close.ewm(span=20, adjust=False).mean()
            ema50 = close.ewm(span=50, adjust=False).mean()
            ema20_curr = float(ema20.iloc[-1])
            ema50_curr = float(ema50.iloc[-1])
            ema20_slope = ((ema20_curr - float(ema20.iloc[-5])) / float(ema20.iloc[-5]) * 100
                          if len(ema20) >= 5 else 0.0)
            above_ema20 = price > ema20_curr
            above_ema50 = price > ema50_curr

            # ── RSI(14) ──────────────────────────────────────────────────────
            delta = close.diff()
            gain  = delta.where(delta > 0, 0.0).rolling(14).mean()
            loss  = (-delta.where(delta < 0, 0.0)).rolling(14).mean()
            rs    = gain / loss.replace(0, np.nan)
            rsi_s = 100 - 100 / (1 + rs)
            rsi   = float(rsi_s.iloc[-1])
            rsi   = 50.0 if np.isnan(rsi) else max(0.0, min(100.0, rsi))

            # ── MACD ─────────────────────────────────────────────────────────
            ema12  = close.ewm(span=12, adjust=False).mean()
            ema26  = close.ewm(span=26, adjust=False).mean()
            macd   = ema12 - ema26
            sig    = macd.ewm(span=9, adjust=False).mean()
            hist   = macd - sig
            h_curr = float(hist.iloc[-1])
            h_prev = float(hist.iloc[-2]) if len(hist) >= 2 else h_curr
            macd_above_zero    = h_curr > 0
            macd_turning_bull  = h_curr > h_prev and h_prev <= 0
            macd_turning_bear  = h_curr < h_prev and h_prev >= 0
            macd_accel_bull    = h_curr > h_prev and h_curr > 0
            macd_accel_bear    = h_curr < h_prev and h_curr < 0

            # ── Volume ratio ──────────────────────────────────────────────────
            vol_today = float(volume.iloc[-1])
            vol_avg   = float(volume.iloc[-21:-1].mean()) if len(volume) >= 21 else float(volume.mean())
            vol_ratio = round(vol_today / vol_avg, 2) if vol_avg > 0 else 1.0

            # ── Momentum ─────────────────────────────────────────────────────
            def ret(n: int) -> float:
                if len(close) <= n:
                    return 0.0
                base = float(close.iloc[-n - 1])
                return (price - base) / base * 100 if base > 0 else 0.0

            ret_1d  = round(ret(1), 2)
            ret_5d  = round(ret(5), 2)
            ret_10d = round(ret(10), 2)
            ret_20d = round(ret(20), 2)

            # ── 52-week context ───────────────────────────────────────────────
            look = min(252, len(close))
            high52 = float(close.tail(look).max())
            low52  = float(close.tail(look).min())
            pct_from_high = round((price - high52) / high52 * 100, 1)
            pct_from_low  = round((price - low52)  / low52  * 100, 1)
            near_52w_high = pct_from_high > -8
            near_52w_low  = pct_from_low  < 15

            # ── Bollinger Band position ───────────────────────────────────────
            sma20 = close.rolling(20).mean()
            std20 = close.rolling(20).std()
            bb_up  = float((sma20 + 2 * std20).iloc[-1])
            bb_low_val = float((sma20 - 2 * std20).iloc[-1])
            bw = bb_up - bb_low_val
            bb_pos = (price - bb_low_val) / bw if bw > 0 else 0.5

            # ═══════════════════════════════════════════════════════════════
            # BULL SCORING (0-12 points) — how good a CALL setup is this?
            # ═══════════════════════════════════════════════════════════════
            bull = 0
            if above_ema20 and above_ema50:   bull += 3
            elif above_ema20:                  bull += 1
            if ema20_slope > 0.15:             bull += 1
            if 50 <= rsi <= 70:                bull += 2
            elif 38 <= rsi < 50 and above_ema50: bull += 2   # oversold in uptrend
            if macd_turning_bull:              bull += 2
            elif macd_accel_bull:              bull += 1
            if vol_ratio >= 1.5 and ret_1d > 0:  bull += 2
            elif vol_ratio >= 1.1:             bull += 1
            if ret_5d > 1.5 and ret_10d > 2:  bull += 2
            elif ret_5d > 0.3:                 bull += 1
            if near_52w_high:                  bull += 1
            if 0.2 <= bb_pos <= 0.55 and above_ema20: bull += 1

            # ═══════════════════════════════════════════════════════════════
            # BEAR SCORING (0-12 points) — how good a PUT setup is this?
            # ═══════════════════════════════════════════════════════════════
            bear = 0
            if not above_ema20 and not above_ema50: bear += 3
            elif not above_ema20:              bear += 1
            if ema20_slope < -0.15:            bear += 1
            if 30 <= rsi <= 52:                bear += 2
            elif 52 < rsi <= 65 and not above_ema50: bear += 2   # overbought in downtrend
            if macd_turning_bear:              bear += 2
            elif macd_accel_bear:              bear += 1
            if vol_ratio >= 1.5 and ret_1d < 0:  bear += 2
            elif vol_ratio >= 1.1 and ret_1d < 0: bear += 1
            if ret_5d < -1.5 and ret_10d < -2: bear += 2
            elif ret_5d < -0.3:                bear += 1
            if near_52w_low:                   bear += 1
            if 0.45 <= bb_pos <= 0.8 and not above_ema20: bear += 1

            best_dir   = "bullish" if bull >= bear else "bearish"
            best_score = max(bull, bear)

            return {
                "symbol":        sym,
                "price":         round(price, 2),
                "pct_change":    ret_1d,
                "volume_ratio":  vol_ratio,
                "rsi":           round(rsi, 1),
                "ema20":         round(ema20_curr, 2),
                "ema50":         round(ema50_curr, 2),
                "ema20_slope":   round(ema20_slope, 3),
                "above_ema20":   above_ema20,
                "above_ema50":   above_ema50,
                "ret_5d":        ret_5d,
                "ret_10d":       ret_10d,
                "ret_20d":       ret_20d,
                "pct_from_high": pct_from_high,
                "pct_from_low":  pct_from_low,
                "macd_above_zero":   macd_above_zero,
                "macd_turning_bull": macd_turning_bull,
                "macd_turning_bear": macd_turning_bear,
                "near_52w_high": near_52w_high,
                "near_52w_low":  near_52w_low,
                "bull_score":    bull,
                "bear_score":    bear,
                "best_direction": best_dir,
                "best_score":    best_score,
            }
        except Exception as e:
            logger.error(f"_compute_signals({sym}): {e}")
            return None

    # ── Summary and prompting ─────────────────────────────────────────────────

    def _build_summary(self, data: dict) -> str:
        rows = sorted(data.values(), key=lambda x: x["best_score"], reverse=True)

        lines = [
            "WATCHLIST SIGNAL SCORES — sorted by best setup quality (BullPts or BearPts)",
            "",
            f"{'Sym':6} | {'Price':>8} | {'1d%':>6} | {'VolRatio':>8} | {'RSI':>5} | "
            f"{'5d%':>6} | {'10d%':>6} | {'52wH%':>6} | {'Trend':>7} | "
            f"{'MACD':>5} | {'BullPts':>7} | {'BearPts':>7}",
            "-" * 96,
        ]
        for d in rows:
            trend = ("↑↑" if d["above_ema20"] and d["above_ema50"]
                     else "↑"  if d["above_ema20"]
                     else "↓↓" if not d["above_ema20"] and not d["above_ema50"]
                     else "↓")
            macd_s = ("+TURN" if d["macd_turning_bull"] else
                      "-TURN" if d["macd_turning_bear"] else
                      "+" if d["macd_above_zero"] else "-")
            lines.append(
                f"{d['symbol']:6} | ${d['price']:7.2f} | {d['pct_change']:+5.1f}% | "
                f"{d['volume_ratio']:7.1f}x | {d['rsi']:5.1f} | "
                f"{d['ret_5d']:+5.1f}% | {d['ret_10d']:+5.1f}% | "
                f"{d['pct_from_high']:+5.1f}% | {trend:>7} | {macd_s:>5} | "
                f"{d['bull_score']:>7} | {d['bear_score']:>7}"
            )
        lines.append("")
        lines.append("BullPts/BearPts: 0-12 quantitative score (higher = stronger setup)")
        lines.append("Trend: ↑↑=strong uptrend, ↓↓=strong downtrend, ↑/↓=mixed")
        lines.append("MACD: +TURN/-TURN = fresh bullish/bearish crossover (strong signal)")
        return "\n".join(lines)

    def _system_prompt(self) -> str:
        return """You are a professional options trader selecting the best 1-3 directional trades.

Pre-scored data: BullPts (0-12) = call setup quality. BearPts (0-12) = put setup quality.

SELECTION RULES:
1. Choose CALLS when BullPts > BearPts and BullPts >= 4
2. Choose PUTS when BearPts > BullPts and BearPts >= 4
3. Fresh MACD crossovers (+TURN/-TURN) are high-conviction — prioritize these
4. Volume ratio > 1.3x adds conviction
5. RSI 40-72 = good call entry. RSI 28-58 = good put entry
6. Near 52w high (52wH% between -8% and 0%) = strong bullish momentum
7. ALWAYS return at least 1 candidate — pick the BEST available even in quiet markets
8. You may pick the SAME symbol twice if both a bullish AND bearish case are strong

Return JSON array of 1-3 candidates:
[
  {
    "symbol": "NVDA",
    "direction": "bullish",
    "option_type": "call",
    "signal_strength": 8,
    "key_reason": "Strong uptrend, +TURN MACD, volume 1.6x — momentum continuation",
    "priority": 1
  }
]

Only valid JSON. No text outside the array."""

    def _auto_select(self, scored: dict) -> list[dict]:
        """Fallback: pick top 1-2 by score when LLM fails."""
        ranked = sorted(scored.values(), key=lambda x: x["best_score"], reverse=True)
        result = []
        for d in ranked[:2]:
            direction = d["best_direction"]
            result.append({
                "symbol":         d["symbol"],
                "direction":      direction,
                "option_type":    "call" if direction == "bullish" else "put",
                "signal_strength": d["best_score"],
                "key_reason":     f"Auto-selected: bull={d['bull_score']}, bear={d['bear_score']}, RSI={d['rsi']}",
                "priority":       len(result) + 1,
            })
        return result
