"""
Scanner Agent — identifies the best 1-3 option trade candidates from the watchlist.

Architecture:
  1. Fetch 3mo of daily data per symbol in a thread pool (parallel, fast)
  2. Compute 12 quantitative signals per symbol (no LLM needed)
  3. Score each symbol independently for both BULL (calls) and BEAR (puts)
  4. Claude Haiku gets the pre-scored table and picks the top 1-3 setups with direction
"""

import asyncio
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Optional

import anthropic
import pandas as pd
import numpy as np

from .base import BaseAgent, BroadcastFn

logger = logging.getLogger(__name__)

# How many worker threads for parallel data fetching
_FETCH_WORKERS = 5


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
        await self._emit("status", f"Scanning {len(self.watchlist)} symbols for opportunities...")

        # Step 1: Fetch and score data in parallel (runs in thread pool, non-blocking)
        loop = asyncio.get_event_loop()
        scored = await loop.run_in_executor(None, self._fetch_and_score_all)

        if not scored:
            await self._emit("status", "No market data available — yfinance may be rate-limited.")
            return []

        await self._emit("status", f"Computed signals for {len(scored)} symbols. Selecting best setups...")

        # Step 2: Build summary table for LLM
        summary = self._build_summary(scored)
        await self._emit("thinking_chunk", f"\n{summary}\n")

        # Step 3: LLM picks top 1-3 candidates with direction
        raw = await self._call(
            self._system_prompt(),
            [{"role": "user", "content": f"{summary}\n\nSelect the best 1-3 option trade setups. "
              "You may choose the same symbol twice if both a bull and bear thesis exist. "
              "Always return at least 1 candidate — every market day has a best opportunity."}],
            max_tokens=700,
            stream=False,
        )
        candidates = self._parse_json(raw)
        if not isinstance(candidates, list):
            # Fallback: auto-select the highest-scored symbol
            candidates = self._auto_select(scored)

        # Step 4: Enrich candidates with computed data
        for c in candidates:
            sym = c.get("symbol", "")
            if sym in scored:
                d = scored[sym]
                c["current_price"]    = d["price"]
                c["bull_score"]       = d["bull_score"]
                c["bear_score"]       = d["bear_score"]
                c["volume_ratio"]     = d["volume_ratio"]
                c["rsi"]              = d["rsi"]
                c["pct_change"]       = d["pct_change"]
                c["iv_rank"]          = None  # computed later by Options Analyst

        await self._emit("status",
            f"Found {len(candidates)} candidate(s): "
            f"{[f\"{c.get('symbol')} {c.get('direction')}\" for c in candidates]}")
        return candidates

    # ── Parallel data fetch and scoring ────────────────────────────────────────

    def _fetch_and_score_all(self) -> dict:
        """Fetch data for all watchlist symbols in a thread pool."""
        results = {}
        with ThreadPoolExecutor(max_workers=_FETCH_WORKERS) as pool:
            future_to_sym = {pool.submit(self._fetch_and_score_symbol, sym): sym
                             for sym in self.watchlist}
            for future in as_completed(future_to_sym):
                sym = future_to_sym[future]
                try:
                    data = future.result(timeout=20)
                    if data:
                        results[sym] = data
                except Exception as e:
                    logger.warning(f"Scanner: fetch failed for {sym}: {e}")
        return results

    def _fetch_and_score_symbol(self, sym: str) -> Optional[dict]:
        """
        Fetch 3 months of daily OHLCV for one symbol and compute all signals.
        Runs in a thread pool — must be thread-safe (no async).
        """
        import yfinance as yf
        try:
            t = yf.Ticker(sym)
            df = t.history(period="3mo", interval="1d", auto_adjust=True)
            if df.empty or len(df) < 20:
                return None
            df.columns = [c.lower() for c in df.columns]
            df = df[["open", "close", "high", "low", "volume"]].dropna()

            close  = df["close"]
            volume = df["volume"]
            price  = float(close.iloc[-1])

            if price <= 0:
                return None

            # ── Moving averages ──────────────────────────────────────────────
            ema20  = close.ewm(span=20, adjust=False).mean()
            ema50  = close.ewm(span=50, adjust=False).mean()
            ema20_curr = float(ema20.iloc[-1])
            ema50_curr = float(ema50.iloc[-1])
            # EMA slope: % change over last 5 sessions
            ema20_slope = (ema20_curr - float(ema20.iloc[-5])) / float(ema20.iloc[-5]) * 100 if len(ema20) >= 5 else 0.0

            above_ema20 = price > ema20_curr
            above_ema50 = price > ema50_curr
            both_emas_aligned_bull = above_ema20 and above_ema50
            both_emas_aligned_bear = (not above_ema20) and (not above_ema50)

            # ── RSI(14) ──────────────────────────────────────────────────────
            delta = close.diff()
            gain  = delta.where(delta > 0, 0.0).rolling(14).mean()
            loss  = (-delta.where(delta < 0, 0.0)).rolling(14).mean()
            rs    = gain / loss.replace(0, np.nan)
            rsi   = float((100 - 100 / (1 + rs)).iloc[-1])
            rsi   = max(0.0, min(100.0, rsi)) if not np.isnan(rsi) else 50.0

            # ── MACD ─────────────────────────────────────────────────────────
            ema12   = close.ewm(span=12, adjust=False).mean()
            ema26   = close.ewm(span=26, adjust=False).mean()
            macd    = ema12 - ema26
            sig     = macd.ewm(span=9, adjust=False).mean()
            hist    = macd - sig
            macd_hist_curr = float(hist.iloc[-1])
            macd_hist_prev = float(hist.iloc[-2]) if len(hist) >= 2 else macd_hist_curr
            macd_above_zero    = macd_hist_curr > 0
            macd_turning_bull  = macd_hist_curr > macd_hist_prev and macd_hist_prev <= 0
            macd_turning_bear  = macd_hist_curr < macd_hist_prev and macd_hist_prev >= 0
            macd_accelerating_bull = macd_hist_curr > macd_hist_prev and macd_hist_curr > 0
            macd_accelerating_bear = macd_hist_curr < macd_hist_prev and macd_hist_curr < 0

            # ── Volume ───────────────────────────────────────────────────────
            vol_today = float(volume.iloc[-1])
            vol_avg20 = float(volume.iloc[-21:-1].mean()) if len(volume) >= 21 else float(volume.mean())
            vol_ratio = round(vol_today / vol_avg20, 2) if vol_avg20 > 0 else 1.0

            # ── Momentum (price returns) ──────────────────────────────────────
            def safe_ret(n: int) -> float:
                if len(close) <= n:
                    return 0.0
                return (price - float(close.iloc[-n-1])) / float(close.iloc[-n-1]) * 100

            ret_1d  = round(safe_ret(1), 2)
            ret_5d  = round(safe_ret(5), 2)
            ret_10d = round(safe_ret(10), 2)
            ret_20d = round(safe_ret(20), 2)

            # ── 52-week context ───────────────────────────────────────────────
            lookback = min(252, len(close))
            high_52w = float(close.tail(lookback).max())
            low_52w  = float(close.tail(lookback).min())
            pct_from_high = round((price - high_52w) / high_52w * 100, 1)
            pct_from_low  = round((price - low_52w)  / low_52w  * 100, 1)
            near_52w_high = pct_from_high > -8    # within 8% of 52w high
            near_52w_low  = pct_from_low  < 15    # within 15% of 52w low

            # ── ATR for context ───────────────────────────────────────────────
            hi, lo = df["high"], df["low"]
            tr = pd.concat(
                [(hi - lo), (hi - close.shift()).abs(), (lo - close.shift()).abs()], axis=1
            ).max(axis=1)
            atr = float(tr.rolling(14).mean().iloc[-1])
            atr_pct = round(atr / price * 100, 2)

            # ── Bollinger Band position ───────────────────────────────────────
            sma20  = close.rolling(20).mean()
            std20  = close.rolling(20).std()
            bb_up  = float((sma20 + 2 * std20).iloc[-1])
            bb_low = float((sma20 - 2 * std20).iloc[-1])
            bb_pos = (price - bb_low) / (bb_up - bb_low) if (bb_up - bb_low) > 0 else 0.5
            # 0 = at lower band, 0.5 = mid, 1.0 = upper band

            # ═══════════════════════════════════════════════════════════════
            # BULL SCORING (0-12 points)
            # Good call setups: uptrend, not overbought, volume, momentum
            # ═══════════════════════════════════════════════════════════════
            bull = 0

            # Trend alignment (max 3)
            if both_emas_aligned_bull:
                bull += 3
            elif above_ema20:
                bull += 1
            if ema20_slope > 0.15:
                bull += 1

            # RSI zone for calls (45-72 = ideal; avoid overbought >75)
            if 50 <= rsi <= 70:
                bull += 2
            elif 42 <= rsi < 50:
                bull += 1   # slight oversold in uptrend = great call entry
            elif rsi < 35 and above_ema50:
                bull += 2   # deeply oversold mean-reversion in uptrend

            # MACD (max 2)
            if macd_turning_bull:
                bull += 2
            elif macd_accelerating_bull:
                bull += 1

            # Volume on up day (max 2)
            if vol_ratio >= 1.5 and ret_1d > 0:
                bull += 2
            elif vol_ratio >= 1.2:
                bull += 1

            # Momentum (max 2)
            if ret_5d > 1.5 and ret_10d > 2:
                bull += 2
            elif ret_5d > 0.5:
                bull += 1

            # 52-week strength (max 1)
            if near_52w_high:
                bull += 1

            # Bollinger band position — below midpoint in uptrend = good entry
            if 0.2 <= bb_pos <= 0.55 and above_ema20:
                bull += 1

            # ═══════════════════════════════════════════════════════════════
            # BEAR SCORING (0-12 points)
            # Good put setups: downtrend, not oversold, volume, negative momentum
            # ═══════════════════════════════════════════════════════════════
            bear = 0

            # Trend alignment (max 3)
            if both_emas_aligned_bear:
                bear += 3
            elif not above_ema20:
                bear += 1
            if ema20_slope < -0.15:
                bear += 1

            # RSI zone for puts (28-55 = ideal; avoid oversold <25)
            if 30 <= rsi <= 52:
                bear += 2
            elif 52 < rsi <= 62 and not above_ema50:
                bear += 1   # mildly overbought in downtrend = great put entry
            elif rsi > 65 and not above_ema50:
                bear += 2   # overbought mean-reversion in downtrend

            # MACD (max 2)
            if macd_turning_bear:
                bear += 2
            elif macd_accelerating_bear:
                bear += 1

            # Volume on down day (max 2)
            if vol_ratio >= 1.5 and ret_1d < 0:
                bear += 2
            elif vol_ratio >= 1.2 and ret_1d < 0:
                bear += 1

            # Momentum (max 2)
            if ret_5d < -1.5 and ret_10d < -2:
                bear += 2
            elif ret_5d < -0.5:
                bear += 1

            # 52-week weakness (max 1)
            if near_52w_low:
                bear += 1

            # Bollinger band position — above midpoint in downtrend = good put entry
            if 0.45 <= bb_pos <= 0.8 and not above_ema20:
                bear += 1

            best_direction = "bullish" if bull >= bear else "bearish"
            best_score     = max(bull, bear)

            return {
                "symbol":          sym,
                "price":           round(price, 2),
                "pct_change":      ret_1d,
                "volume_ratio":    vol_ratio,
                "rsi":             round(rsi, 1),
                "ema20":           round(ema20_curr, 2),
                "ema50":           round(ema50_curr, 2),
                "ema20_slope":     round(ema20_slope, 3),
                "above_ema20":     above_ema20,
                "above_ema50":     above_ema50,
                "ret_5d":          ret_5d,
                "ret_10d":         ret_10d,
                "ret_20d":         ret_20d,
                "pct_from_high":   pct_from_high,
                "pct_from_low":    pct_from_low,
                "atr_pct":         atr_pct,
                "bb_pos":          round(bb_pos, 2),
                "macd_hist":       round(macd_hist_curr, 4),
                "macd_above_zero": macd_above_zero,
                "macd_turning_bull": macd_turning_bull,
                "macd_turning_bear": macd_turning_bear,
                "near_52w_high":   near_52w_high,
                "near_52w_low":    near_52w_low,
                "bull_score":      bull,
                "bear_score":      bear,
                "best_direction":  best_direction,
                "best_score":      best_score,
            }

        except Exception as e:
            logger.error(f"_fetch_and_score_symbol({sym}): {e}")
            return None

    # ── Summary and prompting ─────────────────────────────────────────────────

    def _build_summary(self, data: dict) -> str:
        """Build a rich, pre-scored table for the LLM."""
        # Sort by best_score descending so LLM sees best first
        rows = sorted(data.values(), key=lambda x: x["best_score"], reverse=True)

        header = (
            f"{'Sym':6} | {'Price':>8} | {'1d%':>6} | {'VolRatio':>8} | {'RSI':>5} | "
            f"{'5d%':>6} | {'10d%':>6} | {'52wH%':>6} | {'52wL%':>6} | "
            f"{'EMATrend':>9} | {'MACD':>5} | {'BullPts':>7} | {'BearPts':>7} | {'BestDir':>8}"
        )
        sep = "-" * len(header)
        lines = ["WATCHLIST SIGNAL SCORES", sep, header, sep]

        for d in rows:
            trend = ("↑↑" if d["above_ema20"] and d["above_ema50"]
                     else "↑" if d["above_ema20"]
                     else "↓↓" if not d["above_ema20"] and not d["above_ema50"]
                     else "↓")
            macd_str = ("+turn" if d["macd_turning_bull"] else
                        "-turn" if d["macd_turning_bear"] else
                        "+" if d["macd_above_zero"] else "-")
            lines.append(
                f"{d['symbol']:6} | ${d['price']:7.2f} | {d['pct_change']:+5.1f}% | "
                f"{d['volume_ratio']:7.1f}x | {d['rsi']:5.1f} | "
                f"{d['ret_5d']:+5.1f}% | {d['ret_10d']:+5.1f}% | "
                f"{d['pct_from_high']:+5.1f}% | {d['pct_from_low']:+5.1f}% | "
                f"{trend:>9} | {macd_str:>5} | {d['bull_score']:>7} | {d['bear_score']:>7} | "
                f"{d['best_direction']:>8}"
            )

        lines.append(sep)
        lines.append("\nColumn guide:")
        lines.append("  BullPts/BearPts: 0-12 quantitative score (higher = stronger setup)")
        lines.append("  52wH%: % distance from 52-week high (negative = below high)")
        lines.append("  52wL%: % above 52-week low (small number = near the lows)")
        lines.append("  MACD: +turn/−turn = fresh crossover; +/− = above/below zero")
        lines.append("  EMATrend: ↑↑=above both EMAs, ↓↓=below both, ↑/↓=mixed")
        return "\n".join(lines)

    def _system_prompt(self) -> str:
        return """You are a professional options trader selecting the best 1-3 directional trades.

You have pre-scored quantitative data (BullPts and BearPts, each 0-12). Higher = stronger setup.

SELECT THE BEST 1-3 CANDIDATES following these rules:
1. Prefer setups where BullPts OR BearPts >= 5 (good signal strength)
2. A CALL setup: BullPts > BearPts, RSI in 40-72, uptrend preferred
3. A PUT setup: BearPts > BullPts, RSI in 28-58, downtrend preferred
4. Avoid RSI > 78 for calls (overbought) or RSI < 22 for puts (oversold) — bad entries
5. Volume ratio > 1.2 adds conviction to any directional move
6. Fresh MACD crossovers (+turn / -turn) are high-conviction signals
7. ALWAYS return at least 1 candidate — pick the BEST available setup even in quiet markets

Return a JSON array of 1-3 candidates:
[
  {
    "symbol": "NVDA",
    "direction": "bullish",
    "option_type": "call",
    "signal_strength": 8,
    "key_reason": "Strong uptrend, MACD turning bullish, volume 1.8x — momentum continuation setup",
    "priority": 1
  }
]

Respond ONLY with valid JSON. No text outside the array."""

    def _auto_select(self, scored: dict) -> list[dict]:
        """Fallback if LLM returns bad JSON: pick top 1-2 by score."""
        ranked = sorted(scored.values(), key=lambda x: x["best_score"], reverse=True)
        result = []
        for d in ranked[:2]:
            direction = d["best_direction"]
            result.append({
                "symbol":         d["symbol"],
                "direction":      direction,
                "option_type":    "call" if direction == "bullish" else "put",
                "signal_strength": d["best_score"],
                "key_reason":     f"Auto-selected: bull={d['bull_score']}, bear={d['bear_score']}",
                "priority":       len(result) + 1,
            })
        return result
