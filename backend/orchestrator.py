"""
Orchestrator — analysis-only deliberation loop.

Pipeline (v3 — regime-aware, pre-market-informed, continuous):
  Pre-market (9:00am ET)   → classify market regime + fetch overnight gaps
  After-hours (4:15pm ET)  → warm data cache for next day
  Every scan cycle          → Scanner (IV-first, regime-biased, gap-informed)
                           → [Technical + Fundamental + Sentiment + Risk] parallel
                           → Judge (single LLM call with full context)
  Monitor cycle            → trailing stops, theta exits, high-water mark
"""

import asyncio
import logging
import os
import time as _wtime
import uuid
from datetime import datetime, time as dtime
from typing import Any, Callable, Awaitable

import anthropic

from .agents import (
    ScannerAgent, TechnicalAgent,
    FundamentalAgent, SentimentAgent, RiskAgent,
    JudgeAgent, PositionReviewer,
)
from .state import get_state
from .outcome_tracker import get_outcome_tracker
from .market_regime import classify_regime
from . import market_data as md
from . import pricing
from . import training_store as ts
from .timeutil import now_et, parse_iso_et

logger = logging.getLogger(__name__)

BroadcastFn = Callable[[str, str, Any], Awaitable[None]]

_TECH    = {"NVDA", "AAPL", "MSFT", "META", "GOOGL", "AMD", "CRM", "PLTR", "SMCI", "IONQ", "ROKU"}
_CRYPTO  = {"COIN", "MSTR"}
_FINTECH = {"SOFI", "SQ", "PYPL", "HOOD"}

# Correlation groups: at most 1 open position per group (high same-factor correlation)
_CORR_GROUPS: list[frozenset] = [
    frozenset({"COIN", "MSTR"}),          # pure crypto proxies
    frozenset({"HOOD", "SOFI", "SQ"}),    # retail fintech
    frozenset({"NVDA", "AMD", "SMCI"}),   # semiconductor / AI hardware
    frozenset({"IONQ"}),                  # standalone (quantum, low liquidity)
]


class Orchestrator:
    def __init__(self):
        self.state = get_state()
        self._broadcast: BroadcastFn | None = None
        self._task: asyncio.Task | None = None
        self._claude: anthropic.AsyncAnthropic | None = None

        watchlist_raw = os.getenv(
            "WATCHLIST",
            "PLTR,HOOD,SOFI,RIVN,IONQ,AMD,SMCI,MSTR,TSLA,NVDA,COIN,UBER,SQ,PYPL,ROKU"
        )
        self.watchlist         = [s.strip() for s in watchlist_raw.split(",")]
        self.max_loss          = float(os.getenv("MAX_LOSS_PER_TRADE", "100"))
        self.scan_interval     = int(os.getenv("SCAN_INTERVAL_MINUTES", "30")) * 60
        # 5-min monitor: trailing floors were being gapped through at 15 min
        # (COIN peaked +103% but wasn't caught until +48%). One batch quote
        # call per tick, and the thesis-review cooldown caps LLM cost.
        self.monitor_interval  = int(os.getenv("MONITOR_INTERVAL_MINUTES", "5")) * 60
        self.review_interval_h = float(os.getenv("THESIS_REVIEW_HOURS", "3"))
        self._force_scan       = False   # set True to trigger immediate scan
        self._wake_event: asyncio.Event | None = None  # wakes the loop immediately

    # ── Lifecycle ──────────────────────────────────────────────────────────────

    @property
    def claude(self) -> anthropic.AsyncAnthropic:
        if self._claude is None:
            self._claude = anthropic.AsyncAnthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
        return self._claude

    def set_broadcast(self, fn: BroadcastFn):
        self._broadcast = fn

    async def start(self):
        if self._task and not self._task.done():
            raise ValueError("Orchestrator already running")
        if "ANTHROPIC_API_KEY" not in os.environ:
            raise ValueError("ANTHROPIC_API_KEY not set")
        self.state.system_status = "running"
        self._task = asyncio.create_task(self._main_loop())
        await self._emit("system", "status", {"status": "running"})

    async def stop(self):
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
        self.state.system_status = "stopped"
        await self._emit("system", "status", {"status": "stopped"})

    # ── Main loop ──────────────────────────────────────────────────────────────

    async def _interruptible_sleep(self, seconds: float):
        """Sleep up to `seconds` but return immediately when _wake_event fires."""
        if self._wake_event is None:
            await asyncio.sleep(seconds)
            return
        try:
            await asyncio.wait_for(self._wake_event.wait(), timeout=seconds)
        except asyncio.TimeoutError:
            pass
        finally:
            self._wake_event.clear()

    async def _main_loop(self):
        logger.info("Orchestrator starting.")
        # Create event inside the running loop so it binds to the correct loop.
        self._wake_event = asyncio.Event()
        # Use wall-clock time (time.time()) so backend and frontend stay in sync.
        last_scan      = 0.0
        last_monitor   = 0.0
        last_heartbeat = 0.0

        while True:
            try:
                now_et = self._now_et()
                tod    = now_et.time()
                today  = now_et.strftime("%Y-%m-%d")

                # ── After-hours cache warm (4:15-4:45pm ET, once per day) ─────
                if (dtime(16, 15) <= tod <= dtime(16, 45)
                        and self._is_trading_day()
                        and self.state.get_last_afterhours_date() != today):
                    await self._run_afterhours_capture()
                    await self._interruptible_sleep(60)
                    continue

                # ── Off-hours idle ────────────────────────────────────────────
                if not self._is_market_hours():
                    if not self._is_trading_day():
                        wait_msg = f"Weekend — idle until Monday 9:00am ET. ({now_et.strftime('%a %I:%M %p ET')})"
                        sleep_s  = 3600
                    elif tod < dtime(9, 0):
                        secs = ((now_et.replace(hour=9, minute=0, second=0, microsecond=0)) - now_et).total_seconds()
                        wait_msg = f"Pre-market — warm-up at 9:00am ET ({int(secs/60)} min)."
                        sleep_s  = min(secs, 1800)
                    else:
                        wait_msg = "Market closed — idle until 9:00am ET tomorrow."
                        sleep_s  = 3600
                    await self._emit("system", "info", {"message": wait_msg})
                    # Interruptible so the loop wakes quickly on restart/force.
                    await self._interruptible_sleep(max(sleep_s, 60))
                    continue

                # ── Pre-market prep (9:00-9:30am, once per day) ──────────────
                if (dtime(9, 0) <= tod < dtime(9, 30)
                        and self.state.get_last_premarket_date() != today):
                    await self._run_premarket_prep()

                now = _wtime.time()   # wall-clock seconds — matches state.last_scan

                # ── Monitor positions ─────────────────────────────────────────
                if now - last_monitor >= self.monitor_interval:
                    await self._run_monitor()
                    last_monitor = now

                # ── Expire stale proposals ────────────────────────────────────
                self._expire_stale_proposals()

                # ── Heartbeat every 5 min so UI can confirm backend is alive ──
                if now - last_heartbeat >= 300:
                    last_heartbeat = now
                    open_count = len(self.state.get_sim_positions(status="open"))
                    secs_to_scan = max(0, int(self.scan_interval - (now - last_scan)))
                    mm, ss = divmod(secs_to_scan, 60)
                    await self._emit("system", "heartbeat", {
                        "message": (
                            f"⟳ System alive | Cycle {self.state.cycle_count} | "
                            f"{open_count} open | Next scan in {mm}m{ss:02d}s"
                        ),
                        "next_scan_secs": secs_to_scan,
                    })

                # ── Scan ─────────────────────────────────────────────────────
                scan_due = (now - last_scan >= self.scan_interval) or self._force_scan
                if scan_due:
                    self._force_scan = False
                    last_scan = now
                    open_count = len(self.state.get_sim_positions(status="open"))
                    if open_count >= self.MAX_OPEN_POSITIONS:
                        await self._emit("system", "info", {
                            "message": (
                                f"Portfolio full ({open_count}/{self.MAX_OPEN_POSITIONS}) — "
                                f"skipping scan cycle. Monitor still active."
                            )
                        })
                    else:
                        if open_count:
                            await self._emit("system", "info",
                                {"message": f"📊 {open_count}/{self.MAX_OPEN_POSITIONS} open — scanning for new entries..."})
                        await self._run_scan_cycle()

                # Wait up to 30s — wakes instantly if trigger_scan() is called.
                await self._interruptible_sleep(30)

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.exception(f"Orchestrator error: {e}")
                err_str = str(e)
                await self._emit("system", "error", {"message": err_str})
                if "credit balance" in err_str.lower() or "billing" in err_str.lower():
                    await self._emit("system", "error",
                        {"message": "API credits exhausted — pausing 30 min."})
                    await self._interruptible_sleep(1800)
                else:
                    await self._interruptible_sleep(60)

    # ── Pre-market prep ────────────────────────────────────────────────────────

    async def _run_premarket_prep(self):
        await self._emit("system", "info",
            {"message": "Pre-market prep: classifying market regime + fetching gap data..."})
        loop = asyncio.get_event_loop()

        regime = await loop.run_in_executor(None, classify_regime)
        self.state.market_regime = regime
        await self._emit("system", "info", {
            "message": (
                f"Market regime: {regime['regime'].upper()} "
                f"(strength {regime['strength']}/10) | {regime['summary']}"
            )
        })

        snapshots = {}
        for sym in self.watchlist:
            try:
                snap = await loop.run_in_executor(None, md.get_premarket_snapshot, sym)
                snapshots[sym] = snap
                if snap.get("significant"):
                    dirn = "up" if snap["gap_pct"] > 0 else "down"
                    await self._emit("system", "info", {
                        "message": f"Gap: {sym} {dirn} {snap['gap_pct']:+.1f}% on {snap['vol_ratio']:.1f}x vol"
                    })
                await asyncio.sleep(0.1)
            except Exception as e:
                logger.debug(f"Premarket snapshot {sym}: {e}")

        self.state.premarket_context = snapshots
        self.state.mark_premarket_done()
        sig = sum(1 for s in snapshots.values() if s.get("significant"))
        await self._emit("system", "info",
            {"message": f"Pre-market prep complete: {sig}/{len(snapshots)} significant gaps."})

    # ── After-hours data capture ───────────────────────────────────────────────

    async def _run_afterhours_capture(self):
        await self._emit("system", "info",
            {"message": "After-hours: caching today's prices for tomorrow..."})
        loop = asyncio.get_event_loop()
        for i, sym in enumerate(self.watchlist):
            try:
                await loop.run_in_executor(None, lambda s=sym: md.get_historicals(s, period="3mo"))
                if i % 5 == 4:
                    await asyncio.sleep(1)
            except Exception:
                pass
        self.state.mark_afterhours_done()
        await self._emit("system", "info",
            {"message": "After-hours capture complete."})

    # ── Scan cycle ─────────────────────────────────────────────────────────────

    async def _run_scan_cycle(self):
        self.state.increment_cycle()
        cycle   = self.state.cycle_count
        session = {"pre_open": "PRE-OPEN", "market": "LIVE MARKET",
                   "after_hours": "AFTER-HOURS", "closed": "CLOSED"
                   }.get(self._session_phase(), "LIVE")
        await self._emit("system", "cycle_start", {"cycle": cycle, "session": session})

        # Reclassify regime each cycle (catches intraday market shifts)
        loop = asyncio.get_event_loop()
        prior_regime = self.state.market_regime
        try:
            regime = await loop.run_in_executor(None, classify_regime)
            prior_label = prior_regime.get("regime", "") if prior_regime else ""
            # Alert on regime change
            if prior_label and prior_label != regime["regime"]:
                await self._emit("system", "info", {
                    "message": (
                        f"⚠️ REGIME CHANGE: {prior_label.upper()} → {regime['regime'].upper()} "
                        f"(strength {regime['strength']}/10) — reviewing open positions..."
                    )
                })
            elif not prior_label:
                await self._emit("system", "info", {
                    "message": (
                        f"Regime: {regime['regime'].upper()} "
                        f"(strength {regime['strength']}/10) | {regime.get('summary', '')}"
                    )
                })
            self.state.market_regime = regime
            # Tighten stops on counter-trend positions when regime shifts
            if prior_label and prior_label != regime["regime"] and regime["regime"] in ("bull", "bear"):
                await self._tighten_counter_trend_positions(regime["regime"])
        except Exception as e:
            logger.warning(f"Regime classification failed: {e}")
            regime = prior_regime or {}

        premarket = self.state.premarket_context
        sym_perf  = await asyncio.to_thread(ts.get_symbol_perf)

        scanner    = ScannerAgent(self.claude, self.watchlist, self._make_broadcast())
        candidates = await scanner.scan(
            symbol_performance=sym_perf,
            market_regime=regime,
            premarket_context=premarket,
        )
        # All 15 scanner scores (for complete scan board display)
        all_scored  = getattr(scanner, "_all_scored", {})

        if not candidates:
            await self._emit("system", "info",
                {"message": f"Cycle {cycle}: no candidates."})
            return

        candidates = self._apply_portfolio_filters(candidates)

        # ── Consecutive-loss circuit breaker (tiered) ────────────────────────
        recent_closed = self.state.get_sim_positions(status="closed")[-7:]
        surcharge = 0.0
        if len(recent_closed) >= 3:
            # Count trailing losses from the end
            trailing_losses = 0
            for p in reversed(recent_closed):
                if float(p.get("pnl_dollars", 0)) < 0:
                    trailing_losses += 1
                else:
                    break
            if trailing_losses >= 5:
                surcharge = 6.0   # severe: skip all but exceptional setups
                await self._emit("system", "info", {
                    "message": (
                        f"🚨 {trailing_losses} consecutive losses — circuit breaker: "
                        "+6 threshold surcharge. Only exceptional setups qualify."
                    )
                })
            elif trailing_losses >= 3:
                surcharge = 3.0   # moderate: tighten but don't stop
                await self._emit("system", "info", {
                    "message": (
                        f"⚠️ {trailing_losses} consecutive losses — circuit breaker: "
                        "+3 threshold surcharge this cycle."
                    )
                })
        self._streak_surcharge = surcharge

        await self._emit("system", "info",
            {"message": f"Cycle {cycle}: {len(candidates)} candidate(s) — analyzing..."})

        best_result = None
        best_score  = -1
        rejections  = []
        scan_summary = []

        for i, candidate in enumerate(candidates[:5]):
            symbol    = candidate.get("symbol", "")
            direction = candidate.get("direction", "bullish")
            price     = candidate.get("live_price") or candidate.get("current_price", 0)
            iv_rank   = candidate.get("iv_rank", 50.0)

            await self._emit("system", "analyzing", {
                "symbol": symbol, "direction": direction, "priority": i + 1,
                "iv_rank": iv_rank, "reason": candidate.get("key_reason", ""),
                "premarket_gap": candidate.get("premarket_gap", 0.0),
                "caution": candidate.get("caution", ""),
            })

            result = await self._analyze_candidate(symbol, direction, price, iv_rank, regime)
            if result is None:
                rejections.append(f"{symbol}: error")
                scan_summary.append({
                    "symbol": symbol, "direction": direction, "price": price,
                    "iv_rank": iv_rank, "rsi": candidate.get("rsi", 0),
                    "decision": "error", "pass_reason": "Analysis failed",
                    "proposal_generated": False,
                })
                continue

            judge    = result.get("judge", {})
            decision = judge.get("decision", "pass")
            score    = judge.get("weighted_score", 0)
            conf     = judge.get("confidence", 0)
            reason   = judge.get("pass_reason") or judge.get("reasoning", "")

            self.state.record_symbol_analysis(symbol, direction, result, decision, score)
            news_data = result["sentiment"].get("news", {})
            news_headline = (news_data.get("headlines") or [None])[0] if news_data else None
            tech = result["technical"]
            scan_summary.append({
                "symbol":          symbol,
                "direction":       direction,
                "price":           price,
                "iv_rank":         iv_rank,
                "rsi":             candidate.get("rsi", 0),
                "bull_score":      candidate.get("bull_score", 0),
                "bear_score":      candidate.get("bear_score", 0),
                "tech_score":      tech.get("score"),
                "fund_score":      result["fundamental"].get("score"),
                "sent_score":      result["sentiment"].get("score"),
                "weighted_score":  score,
                "confidence":      conf,
                "decision":        decision,
                "pass_reason":     (reason or "")[:120],
                "tech_fatal_flaw": tech.get("fatal_flaw"),
                "news_score":      news_data.get("score") if news_data and news_data.get("available") else None,
                "news_headline":   news_headline,
                "bull_case":       judge.get("bull_case", ""),
                "bear_case":       judge.get("bear_case", ""),
                "reasoning":       judge.get("reasoning", ""),
                "vwap20_pct":      tech.get("vwap20_pct"),
                "stoch_k":         tech.get("stoch_k"),
                "momentum_60d":    tech.get("momentum_60d"),
                "above_ema200":    tech.get("above_ema200"),
                "above_ema20":     candidate.get("above_ema20"),
                "above_ema50":     candidate.get("above_ema50"),
                "adx":             tech.get("adx") or candidate.get("adx"),
                "vol_ratio":       tech.get("vol_ratio") or candidate.get("vol_ratio"),
                "acc_days":        tech.get("acc_days"),
                "dist_days":       tech.get("dist_days"),
                "macd_bull_div":   tech.get("macd_bull_div"),
                "macd_bear_div":   tech.get("macd_bear_div"),
                "proposal_generated": False,
            })

            if decision == "trade":
                logger.info(f"Cycle {cycle}: {symbol} APPROVED score={score} conf={conf} IV={iv_rank:.0f}")
                if score > best_score:
                    best_score  = score
                    best_result = result
            else:
                rejections.append(f"{symbol}: {reason[:100]}")
                await self._emit("system", "info",
                    {"message": f"{symbol} {direction}: PASS — {reason[:120]}"})

        if best_result:
            best_sym = best_result.get("symbol")
            for e in scan_summary:
                if e["symbol"] == best_sym:
                    e["proposal_generated"] = True
            await self._store_proposal(best_result)
        else:
            await self._emit("system", "info", {
                "message": f"Cycle {cycle} done — no trades. "
                           + (" | ".join(rejections) or "All passed threshold")
            })

        # Add remaining watchlist symbols (scorer-only, not fully analyzed) to scan board
        analyzed_syms = {e["symbol"] for e in scan_summary}
        for sym, d in all_scored.items():
            if sym in analyzed_syms:
                continue
            direction = d.get("best_direction", "bullish")
            scan_summary.append({
                "symbol":    sym,
                "direction": direction,
                "price":     d.get("live_price") or d.get("price", 0),
                "iv_rank":   d.get("iv_rank", 50),
                "rsi":       d.get("rsi", 0),
                "bull_score": d.get("bull_score", 0),
                "bear_score": d.get("bear_score", 0),
                "above_ema200": d.get("above_ema200"),
                "above_ema20":  d.get("above_ema20"),
                "above_ema50":  d.get("above_ema50"),
                "adx":          d.get("adx"),
                "vol_ratio":    d.get("volume_ratio") or d.get("vol_ratio"),
                "decision":  "pass",
                "pass_reason": f"Not in top-5 (scanner bull={d.get('bull_score',0)} bear={d.get('bear_score',0)})",
                "proposal_generated": False,
                "_scanner_only": True,
            })

        self.state.store_scan_results(scan_summary, cycle)

        # Log full scan to training DB — maps symbol → position_id for entered trades
        try:
            position_id_map = {}
            if best_result:
                sym = best_result.get("symbol")
                # Find the position we just opened
                for p in self.state.get_sim_positions(status="open"):
                    if p.get("symbol") == sym and p.get("cycle") == cycle:
                        position_id_map[sym] = p["position_id"]
                        break
            await asyncio.to_thread(
                ts.log_scan_results, cycle, scan_summary, self.state.market_regime, position_id_map)
        except Exception as e:
            logger.warning(f"Training store log failed: {e}")

    # ── Candidate analysis ─────────────────────────────────────────────────────

    async def _analyze_candidate(
        self, symbol: str, direction: str, price: float, iv_rank: float,
        market_regime: dict = None,
    ) -> dict | None:
        try:
            market_open = self._is_market_hours()

            technical, fundamental, sentiment, risk = await asyncio.gather(
                TechnicalAgent(self.claude, self._make_broadcast()).analyze(symbol, direction),
                FundamentalAgent(self.claude, self._make_broadcast()).analyze(symbol, direction=direction),
                SentimentAgent(self.claude, self._make_broadcast()).analyze(
                    symbol, direction, market_regime=market_regime),
                RiskAgent(self.claude, self.max_loss, self._make_broadcast()).evaluate(
                    symbol, {}, self.state.get_sim_positions(status="open")),
            )

            if not risk.get("approved", True):
                return {
                    "symbol": symbol, "direction": direction, "price": price,
                    "technical": technical, "fundamental": fundamental,
                    "sentiment": sentiment, "risk": risk, "iv_rank": iv_rank,
                    "market_regime": market_regime,
                    "judge": {
                        "decision": "pass", "weighted_score": 0, "confidence": 0,
                        "pass_reason": risk.get("rejection_reason", "Risk rejected"),
                        "trade_proposal": None, "bull_case": "", "bear_case": "",
                        "reasoning": risk.get("rejection_reason", ""),
                    },
                }

            combined_surcharge = (
                getattr(self, "_streak_surcharge", 0.0) +
                getattr(self, "_last_slot_surcharge", 0.0)
            )
            judge = await JudgeAgent(self.claude, self._make_broadcast()).decide(
                symbol, direction, technical, fundamental, sentiment, risk,
                self.state.cycle_count,
                market_open=market_open,
                symbol_history=self.state.get_symbol_history(symbol),
                iv_rank=iv_rank,
                market_regime=market_regime,
                streak_surcharge=combined_surcharge,
            )

            return {
                "symbol": symbol, "direction": direction, "price": price,
                "technical": technical, "fundamental": fundamental,
                "sentiment": sentiment, "risk": risk, "judge": judge,
                "iv_rank": iv_rank, "market_regime": market_regime,
            }

        except Exception as e:
            logger.exception(f"Error analyzing {symbol}: {e}")
            await self._emit("system", "error", {"message": f"{symbol} error: {e}"})
            return None

    # ── Portfolio filters ──────────────────────────────────────────────────────

    MAX_OPEN_POSITIONS = 4  # hard cap on total sim positions at once

    def _apply_portfolio_filters(self, candidates: list[dict]) -> list[dict]:
        open_sims   = self.state.get_sim_positions(status="open")
        tech_open   = sum(1 for p in open_sims if p.get("symbol") in _TECH)
        crypto_open = sum(1 for p in open_sims if p.get("symbol") in _CRYPTO)
        bull_open   = sum(1 for p in open_sims if p.get("option_type") == "call")
        bear_open   = sum(1 for p in open_sims if p.get("option_type") == "put")
        open_syms   = {p["symbol"] for p in open_sims}

        # Re-entry prevention: skip symbols closed in the last 3 cycles at a loss
        current_cycle = self.state.cycle_count
        all_closed = self.state.get_sim_positions(status="closed")
        recently_closed = {
            p["symbol"]: p
            for p in all_closed
            if (current_cycle - int(p.get("cycle", 0))) <= 3
               and float(p.get("pnl_dollars", 0)) < 0
        }
        if recently_closed:
            logger.debug(f"Re-entry prevention: {list(recently_closed.keys())} recently closed at a loss")

        # Churn prevention: 2+ closed trades on the same symbol+direction in the
        # last 5 cycles means the move is likely exhausted — even after wins.
        # (Live evidence: COIN puts #1-2 won, #3-4 re-entered the same fading
        # move and gave the profits back.)
        same_dir_counts: dict[tuple, int] = {}
        for p in all_closed:
            if (current_cycle - int(p.get("cycle", 0))) <= 5:
                key = (p.get("symbol"), p.get("direction"))
                same_dir_counts[key] = same_dir_counts.get(key, 0) + 1

        # Pre-compute which correlation groups already have a position
        occupied_groups = set()
        for p in open_sims:
            sym = p.get("symbol", "")
            for i, grp in enumerate(_CORR_GROUPS):
                if sym in grp:
                    occupied_groups.add(i)

        # Set last-slot surcharge for the judge (stored on self for this cycle)
        slots_remaining = self.MAX_OPEN_POSITIONS - len(open_sims)
        self._last_slot_surcharge = 5.0 if slots_remaining == 1 else 0.0
        if slots_remaining == 1 and candidates:
            self.state.log_event("info", {"message":
                "⚠️ Last portfolio slot — applying +5 quality gate this cycle."})

        if len(open_sims) >= self.MAX_OPEN_POSITIONS:
            self.state.log_event("info", {
                "message": f"Portfolio full ({len(open_sims)}/{self.MAX_OPEN_POSITIONS} open) — no new entries"
            })
            return []

        filtered = []
        for c in candidates:
            sym  = c["symbol"]
            dirn = c["direction"]
            if sym in open_syms:
                self.state.log_event("info", {"message": f"Portfolio filter: {sym} already open — skipping"})
                continue
            if sym in recently_closed:
                closed_pos = recently_closed[sym]
                pnl = float(closed_pos.get("pnl_pct", 0))
                self.state.log_event("info", {"message":
                    f"Re-entry block: {sym} closed {pnl:+.0f}% in last 3 cycles — skipping"})
                continue
            if same_dir_counts.get((sym, dirn), 0) >= 2:
                self.state.log_event("info", {"message":
                    f"Churn block: {sym} {dirn} traded {same_dir_counts[(sym, dirn)]}× "
                    "in last 5 cycles — move likely exhausted"})
                continue
            if sym in _TECH and tech_open >= 2:
                self.state.log_event("info", {"message": f"Sector filter: {sym} skipped (2 tech open)"})
                continue
            if sym in _CRYPTO and crypto_open >= 1:
                self.state.log_event("info", {"message": f"Crypto limit: {sym} skipped (1 crypto max)"})
                continue
            # Correlation group guard
            corr_blocked = False
            for i, grp in enumerate(_CORR_GROUPS):
                if sym in grp and i in occupied_groups:
                    self.state.log_event("info", {"message": f"Corr filter: {sym} skipped (correlated position open)"})
                    corr_blocked = True
                    break
            if corr_blocked:
                continue
            if dirn == "bullish" and bull_open >= 2 and bear_open == 0:
                c["caution"] = f"{bull_open} long positions open — net long exposure"
            elif dirn == "bearish" and bear_open >= 2 and bull_open == 0:
                c["caution"] = f"{bear_open} short positions open — net short exposure"
            filtered.append(c)
        return filtered

    # ── Sim auto-execution ─────────────────────────────────────────────────────

    async def _store_proposal(self, analysis: dict):
        """In sim mode: auto-execute the best trade as a simulated position."""
        judge    = analysis["judge"]
        proposal = judge.get("trade_proposal")
        if not proposal:
            return

        symbol    = analysis.get("symbol", proposal.get("symbol", ""))
        direction = analysis.get("direction", "bullish")
        price     = float(analysis.get("price", 0))
        iv_rank   = float(analysis.get("iv_rank", 50.0))
        opt_type  = "call" if direction == "bullish" else "put"

        # Opening 15-minute guard — first candles are volatile/wide spread
        now_et = self._now_et()
        tod    = now_et.time()
        if dtime(9, 30) <= tod < dtime(9, 45):
            await self._emit("system", "info",
                {"message": f"SIM: Skipping {symbol} — opening 15 min (volatile spreads). Will try next cycle."})
            return

        # Closing 30-minute guard — a fill this late holds overnight gap risk
        # with no chance to manage it (late-day entries lost 2 of 3 so far)
        if tod >= dtime(15, 30):
            await self._emit("system", "info",
                {"message": f"SIM: Skipping {symbol} — last 30 min (overnight gap risk). Will reconsider tomorrow."})
            return

        # Don't open a second position in the same symbol
        open_syms = {p["symbol"] for p in self.state.get_sim_positions(status="open")}
        if symbol in open_syms:
            await self._emit("system", "info",
                {"message": f"SIM: {symbol} already open — skipping duplicate."})
            return

        if price <= 0:
            await self._emit("system", "info",
                {"message": f"SIM: {symbol} skipped — no valid live price."})
            return

        # Collect scores from sub-agents to store with position (for learning)
        technical   = analysis.get("technical", {})
        fundamental = analysis.get("fundamental", {})
        sentiment   = analysis.get("sentiment", {})

        # Dynamic DTE: shorter in strong trends (higher leverage), longer in mixed conditions
        # Bull/bear regime + high confidence → 28 days; neutral or low confidence → 42 days
        regime     = analysis.get("market_regime", {})
        reg_name   = regime.get("regime", "neutral") if regime else "neutral"
        reg_str    = regime.get("strength", 5) if regime else 5
        conf       = judge.get("confidence", 5)
        wt_score   = judge.get("weighted_score", 0)
        if (reg_name != "neutral") and reg_str >= 7 and conf >= 7 and wt_score >= 50:
            entry_dte = 28   # high conviction, strong regime → shorter DTE for leverage
        elif conf <= 5 or wt_score < 42:
            entry_dte = 42   # marginal setup → more time to be right
        else:
            entry_dte = 35   # standard

        # Premium scales with spot price and IV so leverage is uniform across
        # symbols; fractional contracts size every position to $100 total cost.
        entry_opt   = pricing.entry_premium(price, iv_rank, entry_dte)
        contracts   = round(100.0 / (entry_opt * 100.0), 4)
        spread_frac = pricing.spread_fraction(iv_rank)

        pos = {
            "position_id":       str(uuid.uuid4()),
            "symbol":            symbol,
            "direction":         direction,
            "option_type":       opt_type,
            "entry_stock_price": round(price, 2),
            "entry_option_price": entry_opt,
            "contracts":         contracts,
            "spread_frac":       spread_frac,
            "total_cost":        100.00,
            "entry_dte":         entry_dte,
            "delta":             0.25,
            "iv_rank":           iv_rank,
            "weighted_score":    judge.get("weighted_score", 0),
            "confidence":        judge.get("confidence", 0),
            "tech_score":        technical.get("score", 5),
            "fund_score":        fundamental.get("score", 5),
            "sent_score":        sentiment.get("score", 5),
            "bull_case":         judge.get("bull_case", ""),
            "bear_case":         judge.get("bear_case", ""),
            "reasoning":         judge.get("reasoning", ""),
            # Regime context at entry — useful for post-trade analysis
            "entry_regime":      regime.get("regime", "neutral") if regime else "neutral",
            "entry_regime_str":  regime.get("strength", 5) if regime else 5,
            "entry_vix":         round(float(regime.get("vix_level", 20)), 1) if regime else None,
            # Technical signals at entry
            "entry_adx":         technical.get("adx"),
            "entry_above_ema200": technical.get("above_ema200"),
            "entry_momentum_1d": technical.get("momentum_1d"),
            "entry_rsi":         technical.get("rsi"),
            "opened_at":         now_et().isoformat(),
            "cycle":             self.state.cycle_count,
            "status":            "open",
            "high_water_pnl_pct": 0.0,
            "last_stock_price":  round(price, 2),
            "last_option_price": entry_opt,
            "last_pnl_pct":      0.0,
            "last_pnl_dollars":  0.0,
        }
        self.state.add_sim_position(pos)

        await self._emit("system", "sim_opened", {
            "symbol": symbol, "direction": direction, "option_type": opt_type,
            "entry_price": price, "score": judge.get("weighted_score"),
            "confidence": judge.get("confidence"),
            "entry_dte": entry_dte,
            "message": (
                f"SIM OPENED: {symbol} {opt_type.upper()} @ ${price:.2f} | "
                f"Score {judge.get('weighted_score'):.0f} | Conf {judge.get('confidence')}/10 | "
                f"DTE {entry_dte} | $100 max loss"
            ),
        })
        logger.info(
            f"SIM: Opened {symbol} {opt_type.upper()} @ ${price:.2f} "
            f"score={judge.get('weighted_score')} conf={judge.get('confidence')}"
        )

    def trigger_scan(self):
        """Force an immediate scan — wakes the loop within milliseconds."""
        self._force_scan = True
        if self._wake_event is not None:
            self._wake_event.set()

    # ── Monitor ────────────────────────────────────────────────────────────────

    async def _run_monitor(self):
        await self._monitor_sim_positions()
        self.state.update_last_monitor()

    async def _monitor_sim_positions(self):
        # Only act on regular-session prices (9:30-16:00). Pre-open quotes are
        # thin and the model marks are unreliable — positions were being exited
        # at 9:00-9:30 AM on premarket prices.
        if self._session_phase() != "market":
            logger.debug("Outside regular session — skipping position monitor")
            return

        open_positions = self.state.get_sim_positions(status="open")
        if not open_positions:
            return

        loop = asyncio.get_event_loop()

        # Batch-fetch all quotes in a single API call
        symbols = [p["symbol"] for p in open_positions]
        try:
            batch_quotes = await loop.run_in_executor(
                None, lambda: md.get_batch_quotes(symbols))
        except Exception as e:
            logger.warning(f"Batch quote fetch failed: {e}")
            batch_quotes = {}

        for pos in open_positions:
            symbol      = pos["symbol"]
            direction   = pos["direction"]
            high_water  = float(pos.get("high_water_pnl_pct", 0.0))
            pos_id      = pos["position_id"]

            opened_at  = parse_iso_et(pos["opened_at"])
            now        = now_et()
            days_held  = max(0, (now - opened_at).days)
            hours_held = (now - opened_at).total_seconds() / 3600

            # Use batch price, fall back to individual call if missing
            current_stock = float(batch_quotes.get(symbol, 0))
            if not current_stock:
                try:
                    quote = await loop.run_in_executor(None, lambda s=symbol: md.get_quote(s))
                    current_stock = float(quote.get("price", 0))
                except Exception:
                    current_stock = 0.0
            if not current_stock:
                continue

            iv_rank_pos  = float(pos.get("iv_rank", 50.0))
            initial_stop = pricing.initial_stop_pct(iv_rank_pos)

            mark        = pricing.mark_position(pos, current_stock, days_held)
            current_opt = mark["option_price"]
            pnl_pct     = mark["pnl_pct"]
            pnl_dollars = mark["pnl_dollars"]
            dte_left    = mark["dte_left"]
            new_high    = max(high_water, pnl_pct)
            prev_pnl    = float(pos.get("last_pnl_pct", 0.0))

            stall_count = pricing.update_stall_count(
                int(pos.get("stall_count", 0)), new_high, pnl_pct, prev_pnl)

            updates = {
                "high_water_pnl_pct": new_high,
                "last_stock_price":   round(current_stock, 2),
                "last_option_price":  current_opt,
                "last_pnl_pct":       pnl_pct,
                "last_pnl_dollars":   pnl_dollars,
                "days_held":          days_held,
                "dte_left":           dte_left,
                "stall_count":        stall_count,
            }

            # ── Exit logic ────────────────────────────────────────────────────
            # No fixed profit target — trailing floor tiers, stall tightening,
            # DTE lift, mini-peak lock, low-conf take-profit (see pricing.py).
            trail_floor = pricing.compute_trail_floor(
                new_high=new_high, pnl_pct=pnl_pct, initial_stop=initial_stop,
                stall_count=stall_count, dte_left=dte_left,
                entry_confidence=float(pos.get("confidence", 7)),
            )

            # ── LLM thesis review ──────────────────────────────────────────────
            # Re-evaluate whether the original entry thesis still holds.
            # First review after 2 hours; then cooldown-gated (see
            # _thesis_review_due) so we don't burn an LLM call every monitor tick.
            # Can exit early (thesis broken) or tighten the stop floor.
            ai_exit_reason = None
            if hours_held >= 2 and self._thesis_review_due(pos, pnl_pct):
                updates["last_thesis_review"] = now.isoformat()
                updates["pnl_at_last_review"] = pnl_pct
                try:
                    fresh_pos = {
                        **pos,
                        "last_pnl_pct":       pnl_pct,
                        "high_water_pnl_pct": new_high,
                        "last_stock_price":   current_stock,
                        "days_held":          days_held,
                        "dte_left":           dte_left,
                    }
                    reviewer = PositionReviewer(self.claude, self._make_broadcast())
                    verdict  = await reviewer.review(fresh_pos, self.state.market_regime or {})
                    v_action = verdict.get("action", "hold")
                    v_reason = verdict.get("reason", "")

                    emoji = {"hold": "✅", "exit": "🚪", "tighten_stop": "⚠️"}.get(v_action, "📋")
                    await self._emit("system", "info", {
                        "message": (
                            f"{emoji} {symbol} thesis review → {v_action.upper()}: {v_reason}"
                        )
                    })

                    if v_action == "exit":
                        ai_exit_reason = f"Thesis review: {v_reason}"
                    elif v_action == "tighten_stop":
                        tighter = verdict.get("tighter_floor")
                        if tighter is not None:
                            new_floor = float(tighter)
                            if new_floor > trail_floor:   # only ever tighten, never loosen
                                logger.info(
                                    f"{symbol}: stop tightened by AI review "
                                    f"{trail_floor:+.0f}% → {new_floor:+.0f}%"
                                )
                                trail_floor = new_floor
                except Exception as e:
                    logger.warning(f"Position review failed for {symbol}: {e}")

            exit_reason = ai_exit_reason or pricing.exit_reason(
                pnl_pct=pnl_pct, new_high=new_high, trail_floor=trail_floor,
                initial_stop=initial_stop, iv_rank=iv_rank_pos,
                days_held=days_held, dte_left=dte_left,
            )

            if exit_reason:
                exit_data = {
                    "exit_stock_price":  round(current_stock, 2),
                    "exit_option_price": current_opt,
                    "pnl_dollars":       pnl_dollars,
                    "pnl_pct":           pnl_pct,
                    "exit_reason":       exit_reason,
                    "days_held":         days_held,
                }
                self.state.close_sim_position(pos_id, exit_data)

                # Feed result into outcome tracker so the learning loop accumulates data
                closed_pos = {**pos, **exit_data}
                try:
                    await asyncio.to_thread(get_outcome_tracker().record_sim_close, closed_pos)
                except Exception as e:
                    logger.warning(f"Outcome tracker record failed: {e}")

                # Write outcome back to training DB
                try:
                    await asyncio.to_thread(
                        ts.update_outcome, pos_id, pnl_pct, pnl_dollars,
                        days_held=days_held, exit_reason=exit_reason,
                    )
                except Exception as e:
                    logger.warning(f"Training store outcome update failed: {e}")

                cumulative = self.state.cumulative_sim_pnl()
                await self._emit("system", "sim_closed", {
                    "symbol": symbol, "direction": direction,
                    "pnl_pct": pnl_pct, "pnl_dollars": pnl_dollars,
                    "exit_reason": exit_reason, "cumulative_pnl": cumulative,
                    "message": (
                        f"SIM CLOSED: {symbol} {pos['option_type'].upper()} | "
                        f"{exit_reason} | P&L: ${pnl_dollars:+.2f} ({pnl_pct:+.1f}%) | "
                        f"Running total: ${cumulative:+.2f}"
                    ),
                })
                logger.info(
                    f"SIM EXIT: {symbol} | {exit_reason} | "
                    f"P&L ${pnl_dollars:+.2f} ({pnl_pct:+.1f}%) | Total ${cumulative:+.2f}"
                )
            else:
                self.state.update_sim_position(pos_id, updates)

    def _thesis_review_due(self, pos: dict, pnl_pct: float) -> bool:
        """
        Cooldown gate for the LLM thesis review: at most one review every
        THESIS_REVIEW_HOURS per position. Exception: if P&L moved ≥15pts since
        the last review (thesis may have broken fast), allow after 30 min.
        """
        last = pos.get("last_thesis_review")
        if not last:
            return True
        try:
            hours_since = (now_et() - parse_iso_et(last)).total_seconds() / 3600
        except ValueError:
            return True
        if hours_since >= self.review_interval_h:
            return True
        moved = abs(pnl_pct - float(pos.get("pnl_at_last_review", pnl_pct)))
        return moved >= 15.0 and hours_since >= 0.5

    async def _tighten_counter_trend_positions(self, new_regime: str):
        """
        When market regime flips (e.g. bull → bear), tighten trailing stops on
        any open positions that are now trading against the new regime.
        Counter-trend positions get stall_count bumped to trigger faster tightening.
        """
        open_positions = self.state.get_sim_positions(status="open")
        tightened = []
        for pos in open_positions:
            direction = pos.get("direction", "bullish")
            pos_id    = pos["position_id"]
            symbol    = pos.get("symbol", "")
            is_counter = (new_regime == "bear" and direction == "bullish") or \
                         (new_regime == "bull" and direction == "bearish")
            if is_counter:
                old_stall = int(pos.get("stall_count", 0))
                new_stall = max(3, old_stall + 2)  # jump to tightening threshold
                self.state.update_sim_position(pos_id, {"stall_count": new_stall})
                tightened.append(symbol)

        if tightened:
            await self._emit("system", "info", {
                "message": (
                    f"⚠️ Regime flip → {new_regime.upper()}: "
                    f"tightened trailing stops on counter-trend positions: {', '.join(tightened)}"
                )
            })

    def _expire_stale_proposals(self):
        pass  # No manual proposals in sim mode

    @staticmethod
    def _proposal_age_minutes(p: dict) -> float:
        try:
            return (datetime.now() - datetime.fromisoformat(p.get("proposed_at", ""))).total_seconds() / 60
        except Exception:
            return 0.0

    # ── Utils ──────────────────────────────────────────────────────────────────

    def _make_broadcast(self) -> BroadcastFn:
        async def _fn(agent: str, etype: str, data: Any):
            await self._emit(agent, etype, data)
        return _fn

    async def _emit(self, agent: str, event_type: str, data: Any):
        self.state.log_event(event_type, {"agent": agent, "data": data})
        if self._broadcast:
            try:
                await self._broadcast(agent, event_type, data)
            except Exception as e:
                logger.debug(f"Broadcast error: {e}")

    @staticmethod
    def _now_et() -> datetime:
        return now_et()

    @classmethod
    def _is_market_hours(cls) -> bool:
        now = cls._now_et()
        return now.weekday() < 5 and dtime(9, 0) <= now.time() <= dtime(16, 0)

    @classmethod
    def _is_trading_day(cls) -> bool:
        return cls._now_et().weekday() < 5

    @classmethod
    def _session_phase(cls) -> str:
        now = cls._now_et()
        t   = now.time()
        if now.weekday() >= 5: return "closed"
        if t < dtime(9, 30):   return "pre_open"
        if t <= dtime(16, 0):  return "market"
        return "after_hours"


_orchestrator = Orchestrator()

def get_orchestrator() -> Orchestrator:
    return _orchestrator
