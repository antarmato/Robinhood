"""
Judge Agent — final go/no-go decision. Uses Opus for best reasoning.
Outputs a structured single-leg option trade proposal or PASS.
"""

import logging
from typing import Optional

import anthropic

from .base import BaseAgent, BroadcastFn

logger = logging.getLogger(__name__)


class JudgeAgent(BaseAgent):
    def __init__(self, client: anthropic.AsyncAnthropic, broadcast: Optional[BroadcastFn] = None):
        super().__init__(client, "Judge", model="claude-opus-4-8", broadcast=broadcast)

    async def decide(
        self,
        symbol: str,
        direction: str,
        technical: dict,
        options_analysis: dict,
        fundamental: dict,
        sentiment: dict,
        risk: dict,
        advocate: dict,
        cycle_number: int = 1,
    ) -> dict:
        await self._emit("status", f"Judge deliberating on {symbol} (cycle {cycle_number})...")

        context = f"""=== TRADING COMMITTEE — CYCLE {cycle_number} ===
Symbol: {symbol} | Direction: {direction}
Proposed: Buy {options_analysis.get('option_type', '')} ${options_analysis.get('strike', '?')} | Exp {options_analysis.get('expiration_date', '?')} ({options_analysis.get('dte', '?')} DTE)
Premium: ${options_analysis.get('estimated_premium', 0):.2f}/share | IV: {options_analysis.get('iv', 0):.1%}

─── TECHNICAL (25%) ───
Score: {technical.get('score', 5)}/10 | Trend: {technical.get('trend', '?')} | Signal: {technical.get('signal', '?')}
RSI: {technical.get('rsi_reading', 'N/A')} | MACD: {technical.get('macd_reading', 'N/A')}
{technical.get('summary', '')}

─── OPTIONS (20%) ───
Score: {options_analysis.get('score', 5)}/10 | Vol: {options_analysis.get('volume', 0)} | OI: {options_analysis.get('open_interest', 0)}
{options_analysis.get('summary', '')}

─── FUNDAMENTAL (20%) ───
Score: {fundamental.get('score', 5)}/10 | Earnings in window: {fundamental.get('earnings_before_expiry', False)}
{fundamental.get('summary', '')}

─── SENTIMENT (15%) ───
Score: {sentiment.get('score', 5)}/10 | PCR: {sentiment.get('pcr', 'N/A')} | Skew: {sentiment.get('skew', '?')}
{sentiment.get('summary', '')}

─── RISK (10%) ───
Score: {risk.get('score', 5)}/10 | Approved: {risk.get('approved', False)} | Contracts: {risk.get('contracts', 0)} | Cost: ${risk.get('total_cost', 0):.2f}
{risk.get('rejection_reason', '') or risk.get('summary', '')}

─── DEVIL'S ADVOCATE (up to -30 pts) ───
Objection strength: {advocate.get('objection_strength', 5)}/10
Fatal flaw: {advocate.get('fatal_flaw') or 'None'}
{chr(10).join('  - ' + o for o in advocate.get('key_objections', []))}
{advocate.get('summary', '')}"""

        system = """You are the head trader making the final call on a single-leg option trade.

PASS immediately if ANY of these are true:
1. Risk agent rejected (approved: false)
2. Devil's Advocate found a fatal flaw
3. Earnings before expiry
4. Weighted score < 58
5. Confidence < 6

Scoring: Tech(25%) + Options(20%) + Fundamental(20%) + Sentiment(15%) + Risk(10%) - Advocate_penalty(0-30)

trade_proposal must have all fields needed for execution via Robinhood API.
profit_target_pct: recommend 50 (exit when option up 50%)
stop_loss_pct: recommend 50 (exit when option down 50% from purchase price)

Respond ONLY with JSON:
{
  "decision": "trade" | "pass",
  "confidence": <1-10>,
  "weighted_score": <0-100>,
  "pass_reason": "<null or reason>",
  "trade_proposal": {
    "symbol": "<string>",
    "option_type": "<call | put>",
    "strike": <float>,
    "expiration_date": "<YYYY-MM-DD>",
    "contracts": <int>,
    "limit_price": <float per share>,
    "total_max_loss": <contracts * limit_price * 100>,
    "profit_target_pct": 50,
    "stop_loss_pct": 50
  },
  "bull_case": "<one sentence — strongest reason to trade>",
  "bear_case": "<one sentence — strongest reason to avoid>",
  "reasoning": "<3-5 sentences explaining the decision>"
}"""

        raw = await self._call(system, [{"role": "user", "content": context}], max_tokens=900)
        result = self._parse_json(raw)
        result.setdefault("decision", "pass")
        result.setdefault("confidence", 5)
        result.setdefault("reasoning", "Analysis complete.")

        # Hard overrides
        if not risk.get("approved", True):
            result["decision"] = "pass"
            result["pass_reason"] = risk.get("rejection_reason", "Risk agent rejected")
            result["trade_proposal"] = None
        if fundamental.get("earnings_before_expiry"):
            result["decision"] = "pass"
            result["pass_reason"] = "Earnings before expiry — too risky"
            result["trade_proposal"] = None
        if advocate.get("fatal_flaw"):
            result["decision"] = "pass"
            result["pass_reason"] = f"Fatal flaw: {advocate['fatal_flaw']}"
            result["trade_proposal"] = None

        await self._emit("decision", {
            "decision": result["decision"],
            "symbol": symbol,
            "confidence": result.get("confidence"),
            "reasoning": result.get("reasoning"),
        })
        return result
