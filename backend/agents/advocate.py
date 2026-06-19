"""
Devil's Advocate Agent — argues the strongest case AGAINST every proposed trade.
"""

import logging
from typing import Optional

import anthropic

from .base import BaseAgent, BroadcastFn

logger = logging.getLogger(__name__)


class DevilsAdvocateAgent(BaseAgent):
    def __init__(self, client: anthropic.AsyncAnthropic, broadcast: Optional[BroadcastFn] = None):
        super().__init__(client, "DevilsAdvocate", model="claude-sonnet-4-6", broadcast=broadcast)

    async def challenge(
        self,
        symbol: str,
        direction: str,
        technical: dict,
        options_analysis: dict,
        fundamental: dict,
        sentiment: dict,
        risk: dict,
    ) -> dict:
        await self._emit("status", f"Challenging the {symbol} trade thesis...")

        context = f"""Proposed trade to kill:
{symbol} — Buy {options_analysis.get('option_type', '')} | Strike ${options_analysis.get('strike', '?')} | Exp {options_analysis.get('expiration_date', '?')} ({options_analysis.get('dte', '?')} DTE)
Premium: ${options_analysis.get('estimated_premium', 0):.2f}/share | Contracts: {risk.get('contracts', 0)} | Total cost: ${risk.get('total_cost', 0):.2f}

Agent scores (the bull case):
- Technical:   {technical.get('score', 5)}/10 — {technical.get('signal', '?')} | {technical.get('summary', '')}
- Options:     {options_analysis.get('score', 5)}/10 | {options_analysis.get('summary', '')}
- Fundamental: {fundamental.get('score', 5)}/10 | Earnings risk: {fundamental.get('earnings_before_expiry', False)} | {fundamental.get('summary', '')}
- Sentiment:   {sentiment.get('score', 5)}/10 | PCR: {sentiment.get('pcr', 'N/A')} | {sentiment.get('summary', '')}
- Risk:        {risk.get('score', 5)}/10 | {risk.get('summary', '')}

Find every reason NOT to make this trade. Be specific, use the numbers. Look for:
- Premium decay (theta): how fast does this option lose value?
- IV crush risk: will IV drop after a catalyst?
- Timing: is the signal already priced in? Wrong part of the cycle?
- Liquidity: can we actually exit at a fair price?
- False signal: is this just noise, not a real breakout?"""

        system = """You are a professional skeptic. Your ONLY job is to find reasons not to trade.
You are NOT trying to be balanced. Find the holes.

A fatal flaw = do not trade regardless (earnings in window, premium too expensive, terrible liquidity).

Respond ONLY with JSON:
{
  "objection_strength": <1-10 where 10=definitely do not trade>,
  "key_objections": ["<specific objection 1>", "<specific objection 2>", "<specific objection 3>"],
  "fatal_flaw": "<null or show-stopping issue>",
  "summary": "<2 sentences — strongest bear case>"
}"""

        raw = await self._call(system, [{"role": "user", "content": context}], max_tokens=500)
        result = self._parse_json(raw)
        result.setdefault("objection_strength", 5)
        result.setdefault("key_objections", [])
        result.setdefault("fatal_flaw", None)
        result.setdefault("summary", "No major objections.")
        return result
