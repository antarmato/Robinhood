"""
Orchestrator — analysis-only deliberation loop.
Runs agents, stores trade proposals in state for Cowork to execute.

Pipeline (v2):
  Scanner (pure Python, IV-first) → candidates
  For each: [Technical + Fundamental + Sentiment + Risk] in parallel (all pure Python)
  Judge (single LLM call) → decision
  OutcomeTracker.record_entry() on approved proposal
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

logger = logging.getLogger(__name__)

BroadcastFn = Callable[[str, str, Any], Awaitable[None]]


class Orchestrator:
    def __init__(self):
        self.state = get_state()
        self._broadcast: BroadcastFn | None = None
        self._task: asyncio.Task | None = None
        self._claude: anthropic.AsyncAnthropic | None = None

        watchlist_raw = os.getenv(
            "WATCHLIST",
            "NVDA,AAPL,MSFT,TSLA,AMZN,META,GOOGL,AMD,NFLX,CRM,COIN,MSTR,PLTR,SMCI,CRWD,HOOD,UBER,SOFI,RIVN,IONQ"
        )
        self.watchlist              = [s.strip() for s in watchlist_raw.split(",")]
        self.max_loss               = float(os.getenv("MAX_LOSS_PER_TRADE", "200"))
        self._scan_interval_market  = int(os.getenv("SCAN_INTERVAL_MINUTES", "20")) * 60
        self._scan_interval_after   = int(os.getenv("SCAN_INTERVAL_AFTER_HOURS_MINUTES", "120")) * 60
        self.scan_interval          = self._scan_interval_market
        self.monitor_interval       = int(os.getenv("MONITOR_INTERVAL_MINUTES", "15")) * 60
        self.max_dte                = int(os.getenv("MAX_DTE", "45"))
        self.min_dte                = int(os.getenv("MIN_DTE", "7"))

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
                if not self._is_market_hours():
                    now_dt = datetime.now()
                    if not self._is_trading_day():
                        wait_msg = f"Weekend — idle until Monday 9:00am ET. ({self._now_et().strftime('%a %I:%M %p ET')})"
                        sleep_s  = 3600
                    else:
                        market_open_dt = now_dt.replace(hour=9, minute=0, second=0, microsecond=0)
                        if now_dt < market_open_dt:
                            secs = (market_open_dt - now_dt).total_seconds()
                            wait_msg = f"Pre-market — warm-up at 9:00am ET ({int(secs/60)} min). Now {self._now_et().strftime('%I:%M %p ET')}."
                            sleep_s  = min(secs, 1800)
                        else:
                            wait_msg = "Market closed — idle until 9:00am ET tomorrow."
                            sleep_s  = 3600

                    await self._emit("system", "info", {"message": wait_msg})
                    await asyncio.sleep(max(sleep_s, 60))
                    continue

                now = asyncio.get_event_loop().time()

                if now - last_monitor >= self.monitor_interval:
                    await self._run_monitor()
                    last_monitor = now

                self.scan_interval = self._scan_interval_market

                if now - last_scan >= self.scan_interval:
                    last_scan = now
                    self._expire_stale_proposals()
                    # Always scan — never block on pending proposals.
                    # _store_proposal skips duplicates (same symbol+direction already pending).
                    pending = self.state.get_pending_proposals()
                    if pending:
                        ages = ", ".join(
                            f"{p.get('symbol')}({self._proposal_age_minutes(p):.0f}min)"
                            for p in pending
                        )
                        await self._emit("system", "info", {
                            "message": f"ℹ️ {len(pending)} proposal(s) still pending: {ages}. Scanning for better opportunities..."
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
                        {"message": "⚠️ API credits exhausted — pausing 30 min."})
                    await asyncio.sleep(1800)
                else:
                    await asyncio.sleep(60)

    # ── Scan cycle ─────────────────────────────────────────────────────────────

    async def _run_scan_cycle(self):
        self.state.increment_cycle()
        cycle = self.state.cycle_count
        phase = self._session_phase()
        session_label = {
            "pre_open": "PRE-OPEN WARM-UP", "market": "LIVE MARKET",
            "after_hours": "AFTER-HOURS", "closed": "CLOSED"
        }.get(phase, "LIVE")
        await self._emit("system", "cycle_start", {"cycle": cycle, "session": session_label})
        logger.info(f"Starting cycle {cycle} [{session_label}]")

        symbol_performance = get_outcome_tracker().get_all_symbol_stats()
        scanner    = ScannerAgent(self.claude, self.watchlist, self._make_broadcast())
        candidates = await scanner.scan(symbol_performance=symbol_performance)

        if not candidates:
            await self._emit("system", "info",
                {"message": f"Cycle {cycle}: Scanner returned no candidates."})
            return

        await self._emit("system", "info",
            {"message": f"Cycle {cycle}: {len(candidates)} candidate(s) — beginning analysis..."})

        best_result = None
        best_score  = -1
        rejection_log = []

        for i, candidate in enumerate(candidates[:5]):
            symbol    = candidate.get("symbol", "")
            direction = candidate.get("direction", "bullish")
            price     = candidate.get("current_price", 0)
            iv_rank   = candidate.get("iv_rank", 50.0)

            await self._emit("system", "analyzing", {
                "symbol": symbol, "direction": direction, "priority": i + 1,
                "signal_strength": candidate.get("signal_strength", 0),
                "iv_rank": iv_rank,
                "reason": candidate.get("key_reason", ""),
            })

            result = await self._analyze_candidate(symbol, direction, price, iv_rank)

            if result is None:
                rejection_log.append(f"{symbol}: analysis error")
                continue

            judge    = result.get("judge", {})
            decision = judge.get("decision", "pass")
            reason   = judge.get("pass_reason") or judge.get("reasoning", "")
            score    = judge.get("weighted_score", 0)
            conf     = judge.get("confidence", 0)

            self.state.record_symbol_analysis(symbol, direction, result, decision, score)

            if decision == "trade":
                logger.info(f"Cycle {cycle}: {symbol} APPROVED — score={score}, conf={conf}, IV={iv_rank:.0f}")
                if score > best_score:
                    best_score  = score
                    best_result = result
            else:
                rejection_log.append(f"{symbol}: PASS — {reason[:120]}")
                logger.info(f"Cycle {cycle}: {symbol} passed — {reason[:120]}")
                await self._emit("system", "info",
                    {"message": f"{symbol} {direction}: PASS — {reason[:120]}"})

        if best_result:
            await self._store_proposal(best_result)
        else:
            summary = " | ".join(rejection_log) if rejection_log else "All candidates failed deliberation"
            await self._emit("system", "info",
                {"message": f"Cycle {cycle} complete — no trades. {summary}"})

    # ── Candidate analysis ─────────────────────────────────────────────────────

    async def _analyze_candidate(
        self, symbol: str, direction: str, price: float, iv_rank: float
    ) -> dict | None:
        try:
            market_open = self._is_market_hours()

            # All pure Python agents run in parallel
            tech_agent  = TechnicalAgent(self.claude, self._make_broadcast())
            fund_agent  = FundamentalAgent(self.claude, self._make_broadcast())
            sent_agent  = SentimentAgent(self.claude, self._make_broadcast())
            risk_agent  = RiskAgent(self.claude, self.max_loss, self._make_broadcast())

            technical, fundamental, sentiment, risk = await asyncio.gather(
                tech_agent.analyze(symbol, direction),
                fund_agent.analyze(symbol),
                sent_agent.analyze(symbol, direction),
                risk_agent.evaluate(symbol, {}, self.state.active_trades),
            )

            # Short-circuit if risk hard-rejected
            if not risk.get("approved", True):
                return {
                    "symbol": symbol, "direction": direction, "price": price,
                    "technical": technical, "fundamental": fundamental,
                    "sentiment": sentiment, "risk": risk,
                    "judge": {
                        "decision": "pass", "weighted_score": 0, "confidence": 0,
                        "pass_reason": risk.get("rejection_reason", "Risk rejected"),
                        "trade_proposal": None, "bull_case": "", "bear_case": "",
                        "reasoning": risk.get("rejection_reason", ""),
                    },
                    "iv_rank": iv_rank,
                }

            # Single LLM call: Judge
            symbol_history = self.state.get_symbol_history(symbol)
            judge_agent    = JudgeAgent(self.claude, self._make_broadcast())
            judge = await judge_agent.decide(
                symbol, direction, technical, fundamental, sentiment, risk,
                self.state.cycle_count,
                market_open=market_open,
                symbol_history=symbol_history,
                iv_rank=iv_rank,
            )

            return {
                "symbol": symbol, "direction": direction, "price": price,
                "technical": technical, "fundamental": fundamental,
                "sentiment": sentiment, "risk": risk, "judge": judge,
                "iv_rank": iv_rank,
            }

        except Exception as e:
            logger.exception(f"Error analyzing {symbol}: {e}")
            await self._emit("system", "error", {"message": f"{symbol} analysis error: {e}"})
            return None

    # ── Proposal storage ───────────────────────────────────────────────────────

    async def _store_proposal(self, analysis: dict):
        judge    = analysis["judge"]
        proposal = judge.get("trade_proposal")
        if not proposal:
            logger.warning("Judge returned trade decision but no proposal")
            return

        # Skip if this exact symbol+direction already has a pending proposal
        # (avoids spamming Cowork with the same trade every 20 min)
        pending = self.state.get_pending_proposals()
        for p in pending:
            if p.get("symbol") == proposal.get("symbol") and \
               p.get("direction") == proposal.get("direction"):
                age = self._proposal_age_minutes(p)
                await self._emit("system", "info", {
                    "message": (
                        f"↩ {proposal['symbol']} {proposal['direction']} already pending "
                        f"({age:.0f} min old) — keeping existing proposal."
                    )
                })
                return

        proposal["proposal_id"]   = str(uuid.uuid4())
        proposal["proposed_at"]   = datetime.now().isoformat()
        proposal["status"]        = "pending"
        proposal["current_price"] = analysis.get("price", 0)
        proposal["analysis_summary"] = {
            "direction":      analysis.get("direction"),
            "iv_rank":        analysis.get("iv_rank"),
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

        # Record entry in outcome tracker so we can measure results later
        tracker = get_outcome_tracker()
        tracker.record_entry(proposal["proposal_id"], proposal, {
            "iv_rank":        analysis.get("iv_rank"),
            "tech_score":     analysis["technical"].get("score"),
            "sent_score":     analysis["sentiment"].get("score"),
            "fund_score":     analysis["fundamental"].get("score"),
            "weighted_score": judge.get("weighted_score"),
            "confidence":     judge.get("confidence"),
        })

        await self._emit("system", "trade_proposal", proposal)
        await self._emit("system", "info", {
            "message": (
                f"📋 PROPOSAL: {proposal['symbol']} {proposal.get('option_type','').upper()} "
                f"| DTE {proposal.get('dte_min')}-{proposal.get('dte_max')} "
                f"| Max ${proposal.get('max_premium', 0):.2f}/share "
                f"| IV rank {analysis.get('iv_rank', 50):.0f}/100 "
                f"| Conf={judge.get('confidence')}/10 Score={judge.get('weighted_score'):.0f}/{judge.get('threshold'):.0f} "
                f"| Open Cowork to approve."
            )
        })

    # ── Monitor ────────────────────────────────────────────────────────────────

    async def _run_monitor(self):
        active = self.state.active_trades
        if not active:
            return
        monitor = MonitorAgent(self.claude, self._make_broadcast())
        signals = await monitor.check_positions(active)
        for sig in signals:
            if sig.get("action") == "exit":
                await self._emit("system", "exit_signal", sig)
                self.state.add_exit_signal(sig)
        self.state.update_last_monitor()

    # ── Helpers ────────────────────────────────────────────────────────────────

    def _make_broadcast(self) -> BroadcastFn:
        async def _fn(agent_name: str, event_type: str, content: Any):
            await self._emit(agent_name, event_type, content)
        return _fn

    async def _emit(self, agent: str, event_type: str, data: Any):
        self.state.log_event(event_type, {"agent": agent, "data": data})
        if self._broadcast:
            try:
                await self._broadcast(agent, event_type, data)
            except Exception as e:
                logger.debug(f"Broadcast error: {e}")

    PROPOSAL_TIMEOUT_MINUTES = 30

    def _expire_stale_proposals(self):
        """Auto-reject proposals that have been pending > PROPOSAL_TIMEOUT_MINUTES."""
        pending = self.state.get_pending_proposals()
        for p in pending:
            age = self._proposal_age_minutes(p)
            if age >= self.PROPOSAL_TIMEOUT_MINUTES:
                pid = p.get("proposal_id", "")
                self.state.resolve_proposal(pid, "rejected", {"auto_expired": True})
                logger.info(f"Auto-expired proposal {pid} for {p.get('symbol')} after {age:.0f} min")
                self.state.log_event("proposal_expired", {
                    "proposal_id": pid,
                    "symbol":      p.get("symbol"),
                    "age_minutes": round(age, 1),
                    "message": (
                        f"Proposal {p.get('symbol')} expired after {age:.0f} min "
                        f"— Cowork did not process it. Resuming scan."
                    ),
                })

    @staticmethod
    def _proposal_age_minutes(proposal: dict) -> float:
        try:
            proposed_at = datetime.fromisoformat(proposal.get("proposed_at", ""))
            return (datetime.now() - proposed_at).total_seconds() / 60
        except Exception:
            return 0.0

    @staticmethod
    def _now_et() -> datetime:
        if _ET:
            return datetime.now(_ET)
        return datetime.now()

    @classmethod
    def _is_market_hours(cls) -> bool:
        now = cls._now_et()
        if now.weekday() >= 5:
            return False
        return dtime(9, 0) <= now.time() <= dtime(16, 0)

    @classmethod
    def _is_trading_day(cls) -> bool:
        return cls._now_et().weekday() < 5

    @classmethod
    def _session_phase(cls) -> str:
        now = cls._now_et()
        t   = now.time()
        if now.weekday() >= 5:        return "closed"
        if t < dtime(9, 30):          return "pre_open"
        if t <= dtime(16, 0):         return "market"
        return "after_hours"


_orchestrator = Orchestrator()

def get_orchestrator() -> Orchestrator:
    return _orchestrator
