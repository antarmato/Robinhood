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
import uuid
from datetime import datetime, time as dtime
try:
    from zoneinfo import ZoneInfo
    _ET = ZoneInfo("America/New_York")
except ImportError:
    _ET = None
from typing import Any, Callable, Awaitable

import anthropic

from .agents import (
    ScannerAgent, TechnicalAgent,
    FundamentalAgent, SentimentAgent, RiskAgent,
    JudgeAgent, MonitorAgent,
)
from .state import get_state
from .outcome_tracker import get_outcome_tracker
from .market_regime import classify_regime
from . import market_data as md

logger = logging.getLogger(__name__)

BroadcastFn = Callable[[str, str, Any], Awaitable[None]]

_TECH   = {"NVDA", "AAPL", "MSFT", "META", "GOOGL", "AMD", "CRM", "PLTR", "SMCI"}
_CRYPTO = {"COIN", "MSTR", "HOOD"}


class Orchestrator:
    def __init__(self):
        self.state = get_state()
        self._broadcast: BroadcastFn | None = None
        self._task: asyncio.Task | None = None
        self._claude: anthropic.AsyncAnthropic | None = None

        watchlist_raw = os.getenv(
            "WATCHLIST",
            "PLTR,HOOD,SOFI,RIVN,IONQ,AMD,SMCI,MSTR"
        )
        self.watchlist         = [s.strip() for s in watchlist_raw.split(",")]
        self.max_loss          = float(os.getenv("MAX_LOSS_PER_TRADE", "100"))
        self.scan_interval     = int(os.getenv("SCAN_INTERVAL_MINUTES", "30")) * 60
        self.monitor_interval  = int(os.getenv("MONITOR_INTERVAL_MINUTES", "15")) * 60

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

    async def _main_loop(self):
        logger.info("Orchestrator starting.")
        last_scan    = 0.0
        last_monitor = 0.0

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
                    await asyncio.sleep(60)
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
                    await asyncio.sleep(max(sleep_s, 60))
                    continue

                # ── Pre-market prep (9:00-9:30am, once per day) ──────────────
                if (dtime(9, 0) <= tod < dtime(9, 30)
                        and self.state.get_last_premarket_date() != today):
                    await self._run_premarket_prep()

                now = asyncio.get_event_loop().time()

                # ── Monitor positions ─────────────────────────────────────────
                if now - last_monitor >= self.monitor_interval:
                    await self._run_monitor()
                    last_monitor = now

                # ── Expire stale proposals ────────────────────────────────────
                self._expire_stale_proposals()

                # ── Scan (always runs — proposals never block scanning) ───────
                if now - last_scan >= self.scan_interval:
                    last_scan = now
                    pending = self.state.get_pending_proposals()
                    if pending:
                        ages = ", ".join(
                            f"{p.get('symbol')}({self._proposal_age_minutes(p):.0f}min)"
                            for p in pending
                        )
                        await self._emit("system", "info", {
                            "message": f"ℹ️ {len(pending)} proposal(s) pending: {ages}. Scanning for better opportunities..."
                        })
                    await self._run_scan_cycle()

                await asyncio.sleep(30)

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.exception(f"Orchestrator error: {e}")
                err_str = str(e)
                await self._emit("system", "error", {"message": err_str})
                if "credit balance" in err_str.lower() or "billing" in err_str.lower():
                    await self._emit("system", "error",
                        {"message": "API credits exhausted — pausing 30 min."})
                    await asyncio.sleep(1800)
                else:
                    await asyncio.sleep(60)

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

        regime    = self.state.market_regime
        premarket = self.state.premarket_context
        sym_perf  = get_outcome_tracker().get_all_symbol_stats()

        scanner    = ScannerAgent(self.claude, self.watchlist, self._make_broadcast())
        candidates = await scanner.scan(
            symbol_performance=sym_perf,
            market_regime=regime,
            premarket_context=premarket,
        )

        if not candidates:
            await self._emit("system", "info",
                {"message": f"Cycle {cycle}: no candidates."})
            return

        candidates = self._apply_portfolio_filters(candidates)
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
            scan_summary.append({
                "symbol":          symbol,
                "direction":       direction,
                "price":           price,
                "iv_rank":         iv_rank,
                "rsi":             candidate.get("rsi", 0),
                "bull_score":      candidate.get("bull_score", 0),
                "bear_score":      candidate.get("bear_score", 0),
                "tech_score":      result["technical"].get("score"),
                "fund_score":      result["fundamental"].get("score"),
                "sent_score":      result["sentiment"].get("score"),
                "weighted_score":  score,
                "confidence":      conf,
                "decision":        decision,
                "pass_reason":     (reason or "")[:120],
                "tech_fatal_flaw": result["technical"].get("fatal_flaw"),
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
        self.state.store_scan_results(scan_summary, cycle)

    # ── Candidate analysis ─────────────────────────────────────────────────────

    async def _analyze_candidate(
        self, symbol: str, direction: str, price: float, iv_rank: float,
        market_regime: dict = None,
    ) -> dict | None:
        try:
            market_open = self._is_market_hours()

            technical, fundamental, sentiment, risk = await asyncio.gather(
                TechnicalAgent(self.claude, self._make_broadcast()).analyze(symbol, direction),
                FundamentalAgent(self.claude, self._make_broadcast()).analyze(symbol),
                SentimentAgent(self.claude, self._make_broadcast()).analyze(
                    symbol, direction, market_regime=market_regime),
                RiskAgent(self.claude, self.max_loss, self._make_broadcast()).evaluate(
                    symbol, {}, self.state.active_trades),
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

            judge = await JudgeAgent(self.claude, self._make_broadcast()).decide(
                symbol, direction, technical, fundamental, sentiment, risk,
                self.state.cycle_count,
                market_open=market_open,
                symbol_history=self.state.get_symbol_history(symbol),
                iv_rank=iv_rank,
                market_regime=market_regime,
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

    def _apply_portfolio_filters(self, candidates: list[dict]) -> list[dict]:
        active      = self.state.active_trades
        tech_open   = sum(1 for t in active if t.get("symbol") in _TECH)
        crypto_open = sum(1 for t in active if t.get("symbol") in _CRYPTO)
        bull_open   = sum(1 for t in active if t.get("option_type") == "call")
        bear_open   = sum(1 for t in active if t.get("option_type") == "put")

        filtered = []
        for c in candidates:
            sym  = c["symbol"]
            dirn = c["direction"]
            if sym in _TECH and tech_open >= 2:
                self.state.log_event("info", {"message": f"Sector filter: {sym} skipped (2 tech positions open)"})
                continue
            if sym in _CRYPTO and crypto_open >= 2:
                self.state.log_event("info", {"message": f"Sector filter: {sym} skipped (2 crypto positions open)"})
                continue
            if dirn == "bullish" and bull_open >= 2 and bear_open == 0:
                c["caution"] = f"All {bull_open} open positions are long — check net exposure"
            elif dirn == "bearish" and bear_open >= 2 and bull_open == 0:
                c["caution"] = f"All {bear_open} open positions are short — check net exposure"
            filtered.append(c)
        return filtered if filtered else candidates

    # ── Proposal storage ───────────────────────────────────────────────────────

    async def _store_proposal(self, analysis: dict):
        judge    = analysis["judge"]
        proposal = judge.get("trade_proposal")
        if not proposal:
            return

        # Skip exact duplicate symbol+direction
        for p in self.state.get_pending_proposals():
            if p.get("symbol") == proposal.get("symbol") and \
               p.get("direction") == proposal.get("direction"):
                age = self._proposal_age_minutes(p)
                await self._emit("system", "info", {
                    "message": f"{proposal['symbol']} {proposal['direction']} already pending ({age:.0f}min) — keeping."
                })
                return

        proposal["proposal_id"]   = str(uuid.uuid4())
        proposal["proposed_at"]   = datetime.now().isoformat()
        proposal["status"]        = "pending"
        proposal["current_price"] = analysis.get("price", 0)
        regime = analysis.get("market_regime") or self.state.market_regime
        proposal["analysis_summary"] = {
            "direction":      analysis.get("direction"),
            "iv_rank":        analysis.get("iv_rank"),
            "regime":         (regime or {}).get("regime", "neutral"),
            "bull_case":      judge.get("bull_case", ""),
            "bear_case":      judge.get("bear_case", ""),
            "reasoning":      judge.get("reasoning", ""),
            "confidence":     judge.get("confidence"),
            "weighted_score": judge.get("weighted_score"),
            "threshold":      judge.get("threshold"),
            "agent_scores": {
                "technical":   analysis["technical"].get("score"),
                "fundamental": analysis["fundamental"].get("score"),
                "sentiment":   analysis["sentiment"].get("score"),
                "risk":        analysis["risk"].get("score"),
            },
        }

        self.state.add_proposal(proposal)
        get_outcome_tracker().record_entry(proposal["proposal_id"], proposal, {
            "iv_rank":        analysis.get("iv_rank"),
            "tech_score":     analysis["technical"].get("score"),
            "sent_score":     analysis["sentiment"].get("score"),
            "fund_score":     analysis["fundamental"].get("score"),
            "weighted_score": judge.get("weighted_score"),
            "confidence":     judge.get("confidence"),
        })

        regime_label = f" | Regime {regime.get('regime','?').upper()}" if regime else ""
        await self._emit("system", "trade_proposal", proposal)
        await self._emit("system", "info", {
            "message": (
                f"PROPOSAL: {proposal['symbol']} {proposal.get('option_type','').upper()} "
                f"| IV {analysis.get('iv_rank', 50):.0f}/100{regime_label} "
                f"| Score {judge.get('weighted_score'):.0f}/{judge.get('threshold'):.0f} "
                f"| Conf {judge.get('confidence')}/10 "
                f"| Max ${proposal.get('max_premium', 0):.2f}/share"
                f" — Open Cowork to approve."
            )
        })

    # ── Monitor ────────────────────────────────────────────────────────────────

    async def _run_monitor(self):
        active = self.state.active_trades
        if not active:
            return
        signals = await MonitorAgent(self.claude, self._make_broadcast()).check_positions(active)
        for sig in signals:
            new_hw = sig.get("new_high_water")
            if new_hw is not None:
                tid = sig.get("trade_id")
                current_hw = next(
                    (float(t.get("high_water_pct", 0)) for t in active if t.get("trade_id") == tid), 0.0
                )
                if new_hw > current_hw:
                    self.state.update_trade(tid, {"high_water_pct": new_hw})
            if sig.get("action") == "exit":
                await self._emit("system", "exit_signal", sig)
                self.state.add_exit_signal(sig)
        self.state.update_last_monitor()

    # ── Proposal expiry ────────────────────────────────────────────────────────

    PROPOSAL_TIMEOUT_MINUTES = 30

    def _expire_stale_proposals(self):
        for p in self.state.get_pending_proposals():
            age = self._proposal_age_minutes(p)
            if age >= self.PROPOSAL_TIMEOUT_MINUTES:
                pid = p.get("proposal_id", "")
                self.state.resolve_proposal(pid, "rejected", {"auto_expired": True})
                self.state.log_event("proposal_expired", {
                    "symbol": p.get("symbol"), "age_minutes": round(age, 1),
                    "message": f"{p.get('symbol')} proposal expired after {age:.0f}min — resuming scan.",
                })

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
        return datetime.now(_ET) if _ET else datetime.now()

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
