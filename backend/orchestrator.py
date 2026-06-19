"""
Orchestrator — analysis-only deliberation loop.
Runs agents, stores trade proposals in state for the Cowork artifact to execute.
No Robinhood credentials required here.
"""

import asyncio
import logging
import os
import uuid
from datetime import datetime, time as dtime
from typing import Any, Callable, Awaitable

import anthropic

from .agents import (
    ScannerAgent, TechnicalAgent, OptionsAnalystAgent,
    FundamentalAgent, SentimentAgent, RiskAgent,
    DevilsAdvocateAgent, JudgeAgent, MonitorAgent,
)
from .state import get_state

logger = logging.getLogger(__name__)

BroadcastFn = Callable[[str, str, Any], Awaitable[None]]


class Orchestrator:
    def __init__(self):
        self.state = get_state()
        self._broadcast: BroadcastFn | None = None
        self._task: asyncio.Task | None = None
        self._claude: anthropic.AsyncAnthropic | None = None

        watchlist_raw = os.getenv("WATCHLIST", "SPY,QQQ,NVDA,AAPL,MSFT,TSLA,AMZN")
        self.watchlist = [s.strip() for s in watchlist_raw.split(",")]
        self.max_loss    = float(os.getenv("MAX_LOSS_PER_TRADE", "200"))
        self.scan_interval    = int(os.getenv("SCAN_INTERVAL_MINUTES", "30")) * 60
        self.monitor_interval = int(os.getenv("MONITOR_INTERVAL_MINUTES", "15")) * 60
        self.max_dte = int(os.getenv("MAX_DTE", "21"))
        self.min_dte = int(os.getenv("MIN_DTE", "5"))

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
            return
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
        last_scan = 0.0
        last_monitor = 0.0

        while True:
            try:
                now = asyncio.get_event_loop().time()

                if now - last_monitor >= self.monitor_interval:
                    await self._run_monitor()
                    last_monitor = now

                if now - last_scan >= self.scan_interval:
                    if self._is_market_hours():
                        # Don't scan if there's already a pending proposal
                        if not self.state.has_pending_proposal():
                            await self._run_scan_cycle()
                        else:
                            await self._emit("system", "info",
                                {"message": "Pending proposal awaiting execution — skipping scan."})
                    else:
                        await self._emit("system", "info", {"message": "Market closed — standing by."})
                    last_scan = now

                await asyncio.sleep(30)

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.exception(f"Orchestrator error: {e}")
                await self._emit("system", "error", {"message": str(e)})
                await asyncio.sleep(60)

    # ── Scan cycle ─────────────────────────────────────────────────────────────

    async def _run_scan_cycle(self):
        self.state.increment_cycle()
        cycle = self.state.cycle_count
        await self._emit("system", "cycle_start", {"cycle": cycle})

        scanner = ScannerAgent(self.claude, self.watchlist, self._make_broadcast())
        candidates = await scanner.scan()

        if not candidates:
            await self._emit("system", "info", {"message": "No candidates found this cycle."})
            return

        best_result = None
        best_score  = -1

        for candidate in candidates[:3]:
            symbol      = candidate.get("symbol")
            direction   = candidate.get("direction", "bullish")
            price       = candidate.get("current_price", 0)

            await self._emit("system", "analyzing", {"symbol": symbol, "direction": direction})

            result = await self._analyze_candidate(symbol, direction, price)
            if result and result.get("judge", {}).get("decision") == "trade":
                score = result["judge"].get("confidence", 0)
                if score > best_score:
                    best_score  = score
                    best_result = result

        if best_result:
            await self._store_proposal(best_result)
        else:
            await self._emit("system", "info",
                {"message": f"Cycle {cycle} complete — no trades passed deliberation."})

    async def _analyze_candidate(self, symbol: str, direction: str, price: float) -> dict | None:
        try:
            options_agent = OptionsAnalystAgent(
                self.claude, self.max_dte, self.min_dte, self._make_broadcast()
            )
            options = await options_agent.analyze(symbol, direction, price)

            if not options.get("expiration_date") or not options.get("strike"):
                await self._emit("system", "info", {"message": f"{symbol}: No viable strike found."})
                return None

            expiry = options["expiration_date"]

            tech_agent  = TechnicalAgent(self.claude, self._make_broadcast())
            fund_agent  = FundamentalAgent(self.claude, self._make_broadcast())
            sent_agent  = SentimentAgent(self.claude, self._make_broadcast())

            technical, fundamental, sentiment = await asyncio.gather(
                tech_agent.analyze(symbol, direction),
                fund_agent.analyze(symbol, expiry),
                sent_agent.analyze(symbol, direction, expiry),
            )

            risk_agent = RiskAgent(self.claude, self.max_loss, self._make_broadcast())
            risk = await risk_agent.evaluate(symbol, options, self.state.active_trades)

            advocate_agent = DevilsAdvocateAgent(self.claude, self._make_broadcast())
            advocate = await advocate_agent.challenge(
                symbol, direction, technical, options, fundamental, sentiment, risk
            )

            judge_agent = JudgeAgent(self.claude, self._make_broadcast())
            judge = await judge_agent.decide(
                symbol, direction, technical, options, fundamental, sentiment,
                risk, advocate, self.state.cycle_count
            )

            return {
                "symbol": symbol, "direction": direction,
                "technical": technical, "options": options,
                "fundamental": fundamental, "sentiment": sentiment,
                "risk": risk, "advocate": advocate, "judge": judge,
            }

        except Exception as e:
            logger.exception(f"Error analyzing {symbol}: {e}")
            await self._emit("system", "error", {"message": f"{symbol} analysis failed: {e}"})
            return None

    # ── Proposal storage ───────────────────────────────────────────────────────

    async def _store_proposal(self, analysis: dict):
        judge    = analysis["judge"]
        proposal = judge.get("trade_proposal")
        if not proposal:
            return

        proposal["proposal_id"]    = str(uuid.uuid4())
        proposal["proposed_at"]    = datetime.utcnow().isoformat()
        proposal["status"]         = "pending"
        proposal["analysis_summary"] = {
            "bull_case":      judge.get("bull_case", ""),
            "bear_case":      judge.get("bear_case", ""),
            "reasoning":      judge.get("reasoning", ""),
            "confidence":     judge.get("confidence"),
            "weighted_score": judge.get("weighted_score"),
            "agent_scores": {
                "technical":          analysis["technical"].get("score"),
                "options":            analysis["options"].get("score"),
                "fundamental":        analysis["fundamental"].get("score"),
                "sentiment":          analysis["sentiment"].get("score"),
                "risk":               analysis["risk"].get("score"),
                "objection_strength": analysis["advocate"].get("objection_strength"),
            }
        }

        self.state.add_proposal(proposal)
        await self._emit("system", "trade_proposal", proposal)
        await self._emit("system", "info",
            {"message": f"📋 Trade proposal stored: {proposal['symbol']} {proposal['option_type']} "
                        f"${proposal['strike']} — open Cowork dashboard to review."})

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
                # Create an exit proposal for the Cowork artifact to action
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

    @staticmethod
    def _is_market_hours() -> bool:
        now = datetime.utcnow()
        if now.weekday() >= 5:
            return False
        return dtime(13, 30) <= now.time() <= dtime(20, 0)


_orchestrator = Orchestrator()

def get_orchestrator() -> Orchestrator:
    return _orchestrator
