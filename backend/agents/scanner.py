"""
Scanner Agent — pure Python, IV-rank-first symbol selection.

Pipeline:
  1. Fetch OHLCV for all watchlist symbols in parallel (Polygon)
  2. Compute IV rank per symbol (Tradier → HV rank fallback)
  3. FILTER: drop symbols with IV rank > 60 (expensive premium — skip)
  4. Compute 12 directional signals per symbol (pure Python)
  5. Apply relative-strength bonus across the group
  6. Return top 5 candidates sorted by combined score + IV edge bonus
  No LLM call.
"""

import asyncio
import logging
import os
from typing import Optional

import numpy as np
import pandas as pd

from .base import BaseAgent, BroadcastFn
from .. import market_data as md
from ..strategy import IV_RANK_HARD_SKIP

logger = logging.getLogger(__name__)


class ScannerAgent(BaseAgent):
    def __init__(self, client, watchlist: list[str], broadcast: Optional[BroadcastFn] = None):
        super().__init__(client, "Scanner", broadcast=broadcast)
        self.watchlist = watchlist

    # ── Public entry point ─────────────────────────────────────────────────────

    async def scan(
        self,
        symbol_performance: dict = None,
        market_regime: dict = None,
        premarket_context: dict = None,
    ) -> list[dict]:
        """
        symbol_performance : {symbol: {win_rate, trade_count, avg_pnl}} from OutcomeTracker
        market_regime      : classify_regime() result — biases bull/bear scores
        premarket_context  : {symbol: premarket_snapshot} — gap confirms direction
        """
        import os
        alpaca_key = os.getenv("ALPACA_API_KEY", "")
        poly_key   = os.getenv("POLYGON_API_KEY", "")

        if alpaca_key:
            await self._emit("status",
                f"✅ Alpaca key set ({alpaca_key[:6]}…). Using Alpaca as primary data source.")
        else:
            await self._emit("status",
                "⚠️ ALPACA_API_KEY not set — falling back to Polygon (rate-limited). "
                "Add ALPACA_API_KEY + ALPACA_API_SECRET in Railway → Variables.")

        if not poly_key and not alpaca_key:
            await self._emit("status", "❌ No data source configured. Set ALPACA_API_KEY in Railway.")
            return []

        test_df = md.get_historicals("SPY", period="3mo")
        if test_df.empty:
            await self._emit("status",
                "❌ All data sources failed for SPY — check API keys in Railway Variables.")
            return []

        source = "Alpaca" if alpaca_key else "Polygon"
        await self._emit("status", f"✅ {source} OK. Fetching {len(self.watchlist)} symbols...")

        # ── Fetch OHLCV in batches ─────────────────────────────────────────────
        loop = asyncio.get_event_loop()
        raw_results = []
        for batch_start in range(0, len(self.watchlist), 5):
            batch = self.watchlist[batch_start:batch_start + 5]
            tasks = [loop.run_in_executor(None, self._fetch_one, sym) for sym in batch]
            raw_results.extend(await asyncio.gather(*tasks, return_exceptions=True))
            if batch_start + 5 < len(self.watchlist):
                await asyncio.sleep(1)

        scored: dict = {}
        errors: dict = {}
        for sym, r in zip(self.watchlist, raw_results):
            if isinstance(r, dict) and r:
                scored[sym] = r
            elif isinstance(r, Exception):
                errors[sym] = str(r)
            else:
                errors[sym] = "None returned"

        if errors:
            sample = next(iter(errors.values()))
            await self._emit("status",
                f"⚠️ {len(errors)}/{len(self.watchlist)} symbols failed "
                f"(e.g. {sample[:80]}). Got data for {len(scored)}.")

        if not scored:
            await self._emit("status", "❌ ZERO symbols returned usable data.")
            return []

        # ── IV rank filter ─────────────────────────────────────────────────────
        # Hard-skip (IV > 60) only applies when TRADIER_TOKEN is set and we have
        # real options IV data. Without Tradier, get_iv_rank_best() returns HV rank
        # (historical vol proxy) — high HV ≠ expensive options, so don't filter on it.
        tradier_available = bool(os.getenv("TRADIER_TOKEN", ""))

        iv_passed = {}
        iv_skipped = []
        for sym, d in scored.items():
            iv_rank = md.get_iv_rank_best(sym)
            d["iv_rank"] = round(iv_rank, 1)
            if tradier_available and iv_rank > IV_RANK_HARD_SKIP:
                iv_skipped.append(f"{sym}({iv_rank:.0f})")
            else:
                d["iv_bonus"] = round(max(0.0, (40.0 - iv_rank) / 40.0 * 3.0), 2)
                iv_passed[sym] = d

        source = "real IV" if tradier_available else "HV proxy (no Tradier)"
        await self._emit("status",
            f"IV filter ({source}): {len(iv_passed)}/{len(scored)} passed "
            f"(skipped: {', '.join(iv_skipped) if iv_skipped else 'none'})")

        if not iv_passed:
            await self._emit("status",
                "⚠️ All symbols filtered out by IV rank (premium too expensive). "
                "Waiting for IV to compress. No trades this cycle.")
            return []

        # ── Apply historical performance modifier ─────────────────────────────
        if symbol_performance:
            for sym, d in iv_passed.items():
                perf = symbol_performance.get(sym)
                if perf and perf.get("trade_count", 0) >= 3:
                    win_rate = perf.get("win_rate", 0.5)
                    if win_rate >= 0.65:
                        # Proven winner on this system — boost both scores
                        bonus = 2 if win_rate >= 0.75 else 1
                        d["bull_score"] = min(16, d["bull_score"] + bonus)
                        d["bear_score"] = min(16, d["bear_score"] + bonus)
                        logger.debug(f"{sym}: +{bonus} performance bonus (win_rate={win_rate:.0%})")
                    elif win_rate <= 0.30:
                        # Consistently losing — penalize
                        d["bull_score"] = max(0, d["bull_score"] - 1)
                        d["bear_score"] = max(0, d["bear_score"] - 1)
                        logger.debug(f"{sym}: -1 performance penalty (win_rate={win_rate:.0%})")
                d["best_score"]     = max(d["bull_score"], d["bear_score"])
                d["best_direction"] = "bullish" if d["bull_score"] >= d["bear_score"] else "bearish"

        # ── Market regime bias ────────────────────────────────────────────────
        if market_regime:
            regime   = market_regime.get("regime", "neutral")
            strength = market_regime.get("strength", 5)
            boost    = 2 if strength >= 8 else 1
            if regime == "bull":
                for d in iv_passed.values():
                    d["bull_score"] = min(16, d["bull_score"] + boost)
            elif regime == "bear":
                for d in iv_passed.values():
                    d["bear_score"] = min(16, d["bear_score"] + boost)
            if regime != "neutral":
                await self._emit("status",
                    f"Regime {regime.upper()} (strength {strength}/10): "
                    f"{'+' if regime == 'bull' else '-'}{boost} applied to "
                    f"{'bull' if regime == 'bull' else 'bear'} scores")
            for d in iv_passed.values():
                d["best_score"]     = max(d["bull_score"], d["bear_score"])
                d["best_direction"] = "bullish" if d["bull_score"] >= d["bear_score"] else "bearish"

        # ── Pre-market gap bonus ───────────────────────────────────────────────
        if premarket_context:
            gap_notes = []
            for sym, d in iv_passed.items():
                pm = premarket_context.get(sym, {})
                if not pm.get("significant"):
                    continue
                gap_pct = pm.get("gap_pct", 0.0)
                gap_dir = pm.get("gap_direction", "flat")
                d["premarket_gap"] = gap_pct
                # Strong gap confirming direction = +2 bonus
                if gap_dir == "up":
                    d["bull_score"] = min(16, d["bull_score"] + 2)
                    gap_notes.append(f"{sym}↑{gap_pct:+.1f}%")
                elif gap_dir == "down":
                    d["bear_score"] = min(16, d["bear_score"] + 2)
                    gap_notes.append(f"{sym}↓{gap_pct:+.1f}%")
                d["best_score"]     = max(d["bull_score"], d["bear_score"])
                d["best_direction"] = "bullish" if d["bull_score"] >= d["bear_score"] else "bearish"
            if gap_notes:
                await self._emit("status", f"Pre-market gap signals: {', '.join(gap_notes)}")

        # ── Relative strength across filtered group ────────────────────────────
        if len(iv_passed) >= 3:
            ret_vals = sorted(d["ret_20d"] for d in iv_passed.values())
            median_ret = ret_vals[len(ret_vals) // 2]
            for d in iv_passed.values():
                rs = round(d["ret_20d"] - median_ret, 2)
                d["rs_vs_group"] = rs
                if rs > 3:
                    d["bull_score"] = min(16, d["bull_score"] + 2)
                elif rs < -3:
                    d["bear_score"] = min(16, d["bear_score"] + 2)
                elif rs > 1:
                    d["bull_score"] = min(16, d["bull_score"] + 1)
                elif rs < -1:
                    d["bear_score"] = min(16, d["bear_score"] + 1)
                d["best_score"]     = max(d["bull_score"], d["bear_score"])
                d["best_direction"] = "bullish" if d["bull_score"] >= d["bear_score"] else "bearish"
        else:
            for d in iv_passed.values():
                d["rs_vs_group"] = 0.0

        # ── Live prices (snapshot) — one batch call replaces yesterday's close ──
        live_prices = md.get_batch_quotes(list(iv_passed.keys()))
        for sym, d in iv_passed.items():
            if sym in live_prices and live_prices[sym] > 0:
                d["live_price"] = live_prices[sym]

        # ── Select top 5 by (best_score + iv_bonus) ───────────────────────────
        candidates = self._select_candidates(iv_passed)
        cand_list  = [f"{c['symbol']} {c['direction']}(IV={c.get('iv_rank','?')})" for c in candidates]
        await self._emit("status", f"Selected {len(candidates)} candidate(s): {cand_list}")
        return candidates

    # ── Data fetch ─────────────────────────────────────────────────────────────

    def _fetch_one(self, sym: str) -> Optional[dict]:
        try:
            df = md.get_historicals(sym, period="3mo")
            if df.empty:
                return Exception(f"{sym}: Polygon returned no data")
            if len(df) < 20:
                return Exception(f"{sym}: only {len(df)} rows")
            return self._compute_signals(sym, df)
        except Exception as e:
            return Exception(f"{type(e).__name__}: {str(e)[:80]}")

    # ── Signal computation ─────────────────────────────────────────────────────

    def _compute_signals(self, sym: str, df: pd.DataFrame) -> Optional[dict]:
        try:
            close  = df["close"]
            volume = df["volume"] if "volume" in df.columns else pd.Series([1] * len(df), index=df.index)
            high   = df["high"]   if "high"   in df.columns else close
            low    = df["low"]    if "low"    in df.columns else close

            price = float(close.iloc[-1])
            if price <= 0:
                return None

            ema20 = close.ewm(span=20, adjust=False).mean()
            ema50 = close.ewm(span=50, adjust=False).mean()
            ema20_curr  = float(ema20.iloc[-1])
            ema50_curr  = float(ema50.iloc[-1])
            ema20_slope = ((ema20_curr - float(ema20.iloc[-5])) / float(ema20.iloc[-5]) * 100
                           if len(ema20) >= 5 else 0.0)
            above_ema20 = price > ema20_curr
            above_ema50 = price > ema50_curr

            delta = close.diff()
            gain  = delta.where(delta > 0, 0.0).rolling(14).mean()
            loss  = (-delta.where(delta < 0, 0.0)).rolling(14).mean()
            rs    = gain / loss.replace(0, np.nan)
            rsi_s = 100 - 100 / (1 + rs)
            rsi   = float(rsi_s.iloc[-1])
            rsi   = 50.0 if np.isnan(rsi) else max(0.0, min(100.0, rsi))

            ema12  = close.ewm(span=12, adjust=False).mean()
            ema26  = close.ewm(span=26, adjust=False).mean()
            macd   = ema12 - ema26
            sig    = macd.ewm(span=9, adjust=False).mean()
            hist   = macd - sig
            h_curr = float(hist.iloc[-1])
            h_prev = float(hist.iloc[-2]) if len(hist) >= 2 else h_curr
            macd_above_zero   = h_curr > 0
            macd_turning_bull = h_curr > h_prev and h_prev <= 0
            macd_turning_bear = h_curr < h_prev and h_prev >= 0
            macd_accel_bull   = h_curr > h_prev and h_curr > 0
            macd_accel_bear   = h_curr < h_prev and h_curr < 0

            vol_today = float(volume.iloc[-1])
            vol_avg   = float(volume.iloc[-21:-1].mean()) if len(volume) >= 21 else float(volume.mean())
            vol_ratio = round(vol_today / vol_avg, 2) if vol_avg > 0 else 1.0

            def ret(n: int) -> float:
                if len(close) <= n:
                    return 0.0
                base = float(close.iloc[-n - 1])
                return (price - base) / base * 100 if base > 0 else 0.0

            ret_1d  = round(ret(1), 2)
            ret_5d  = round(ret(5), 2)
            ret_10d = round(ret(10), 2)
            ret_20d = round(ret(20), 2)

            look  = min(252, len(close))
            high52 = float(close.tail(look).max())
            low52  = float(close.tail(look).min())
            near_52w_high = (price - high52) / high52 * 100 > -8
            near_52w_low  = (price - low52)  / low52  * 100 < 15

            sma20 = close.rolling(20).mean()
            std20 = close.rolling(20).std()
            bb_up      = float((sma20 + 2 * std20).iloc[-1])
            bb_low_val = float((sma20 - 2 * std20).iloc[-1])
            bw = bb_up - bb_low_val
            bb_pos = (price - bb_low_val) / bw if bw > 0 else 0.5

            try:
                bw_hist = (close.rolling(20).std() * 4 / close.rolling(20).mean()).tail(60).dropna()
                bw_avg  = float(bw_hist.mean()) if len(bw_hist) > 5 else bb_pos * 2
                bb_bw_pct = bw / float(sma20.iloc[-1]) if float(sma20.iloc[-1]) > 0 else 0.05
                squeeze = bb_bw_pct < bw_avg * 0.75
            except Exception:
                squeeze = False

            # Bull scoring (0-16)
            bull = 0
            if above_ema20 and above_ema50: bull += 3
            elif above_ema20:               bull += 1
            if ema20_slope > 0.15:          bull += 1
            if 50 <= rsi <= 70:             bull += 2
            elif 38 <= rsi < 50 and above_ema50: bull += 2
            if macd_turning_bull:           bull += 2
            elif macd_accel_bull:           bull += 1
            if vol_ratio >= 1.5 and ret_1d > 0: bull += 2
            elif vol_ratio >= 1.1:          bull += 1
            if ret_5d > 1.5 and ret_10d > 2: bull += 2
            elif ret_5d > 0.3:              bull += 1
            if near_52w_high:               bull += 1
            if 0.2 <= bb_pos <= 0.55 and above_ema20: bull += 1
            if squeeze and above_ema20:     bull += 1

            # Bear scoring (0-16)
            bear = 0
            if not above_ema20 and not above_ema50: bear += 3
            elif not above_ema20:           bear += 1
            if ema20_slope < -0.15:         bear += 1
            if 30 <= rsi <= 52:             bear += 2
            elif 52 < rsi <= 65 and not above_ema50: bear += 2
            if macd_turning_bear:           bear += 2
            elif macd_accel_bear:           bear += 1
            if vol_ratio >= 1.5 and ret_1d < 0: bear += 2
            elif vol_ratio >= 1.1 and ret_1d < 0: bear += 1
            if ret_5d < -1.5 and ret_10d < -2: bear += 2
            elif ret_5d < -0.3:             bear += 1
            if near_52w_low:                bear += 1
            if 0.45 <= bb_pos <= 0.8 and not above_ema20: bear += 1
            if squeeze and not above_ema20: bear += 1

            best_dir   = "bullish" if bull >= bear else "bearish"
            best_score = max(bull, bear)

            return {
                "symbol":           sym,
                "price":            round(price, 2),
                "pct_change":       ret_1d,
                "volume_ratio":     vol_ratio,
                "rsi":              round(rsi, 1),
                "ema20":            round(ema20_curr, 2),
                "ema50":            round(ema50_curr, 2),
                "ema20_slope":      round(ema20_slope, 3),
                "above_ema20":      above_ema20,
                "above_ema50":      above_ema50,
                "ret_5d":           ret_5d,
                "ret_10d":          ret_10d,
                "ret_20d":          ret_20d,
                "macd_above_zero":  macd_above_zero,
                "macd_turning_bull": macd_turning_bull,
                "macd_turning_bear": macd_turning_bear,
                "near_52w_high":    near_52w_high,
                "near_52w_low":     near_52w_low,
                "squeeze":          squeeze,
                "bull_score":       bull,
                "bear_score":       bear,
                "best_direction":   best_dir,
                "best_score":       best_score,
                "rs_vs_group":      0.0,
                "iv_rank":          50.0,
                "iv_bonus":         0.0,
            }
        except Exception as e:
            logger.error(f"_compute_signals({sym}): {e}")
            return None

    # ── Candidate selection ────────────────────────────────────────────────────

    def _select_candidates(self, scored: dict) -> list[dict]:
        """
        Select top 5 candidates by (best_score + iv_bonus).
        Tiebreaker (same combined score):
          1. Lower IV rank preferred (cheaper premium)
          2. Better RS vs group (stronger relative momentum)
        Enforce diversity: at least one bull and one bear if scores support it.
        """
        def combined(d):
            primary = d["best_score"] + d.get("iv_bonus", 0.0)
            # Tiebreaker: lower IV rank = cheaper premium = preferred
            iv_tiebreak = -d.get("iv_rank", 50.0) / 1000.0
            # Secondary: relative strength vs group
            rs_tiebreak = d.get("rs_vs_group", 0.0) / 10000.0
            return primary + iv_tiebreak + rs_tiebreak

        ranked = sorted(scored.values(), key=combined, reverse=True)

        # Separate bull / bear pools
        bulls = [d for d in ranked if d["best_direction"] == "bullish" and d["bull_score"] >= 4]
        bears = [d for d in ranked if d["best_direction"] == "bearish" and d["bear_score"] >= 4]

        selected = []
        used = set()

        def add(d, direction):
            sym = d["symbol"]
            if sym in used:
                return
            used.add(sym)
            option_type = "call" if direction == "bullish" else "put"
            score = d["bull_score"] if direction == "bullish" else d["bear_score"]
            sqz = " BB-squeeze" if d.get("squeeze") else ""
            selected.append({
                "symbol":          sym,
                "direction":       direction,
                "option_type":     option_type,
                "signal_strength": score,
                "iv_rank":         d.get("iv_rank", 50.0),
                "iv_bonus":        d.get("iv_bonus", 0.0),
                "current_price":   d["price"],
                "bull_score":      d["bull_score"],
                "bear_score":      d["bear_score"],
                "volume_ratio":    d["volume_ratio"],
                "rsi":             d["rsi"],
                "pct_change":      d["pct_change"],
                "key_reason": (
                    f"{'bull' if direction=='bullish' else 'bear'}={score}, "
                    f"RSI={d['rsi']}, RS={d.get('rs_vs_group',0):+.1f}%"
                    f"{sqz}"
                    + (f", gap{d['premarket_gap']:+.1f}%" if d.get("premarket_gap") else "")
                ),
                "premarket_gap": d.get("premarket_gap", 0.0),
                "priority": len(selected) + 1,
            })

        # Always lead with top bull and top bear for diversity
        if bulls:
            add(bulls[0], "bullish")
        if bears:
            add(bears[0], "bearish")

        # Fill up to 5 from the overall ranked list
        for d in ranked:
            if len(selected) >= 5:
                break
            direction = d["best_direction"]
            if d["symbol"] not in used:
                score = d["bull_score"] if direction == "bullish" else d["bear_score"]
                if score >= 4:
                    add(d, direction)

        return selected
