"""
Devil's Advocate Agent — argues the strongest case AGAINST every proposed trade.

Design philosophy:
  The advocate should find REAL problems, not kill everything.
  A fatal_flaw must be a genuine show-stopper — not vague concerns about premium.
  The judge reads this output and weighs it, so the advocate should be specific and honest.

Fatal flaw criteria (STRICT — must meet at least one to qualify):
  1. Confirmed earnings within the option's DTE window
  2. Bid-ask spread > 30% of mid (truly illiquid — can't exit)
  3. Zero open interest AND zero volume on the proposed strike
  4. Premium > 8% of stock price (extreme cost)

Everything else = an objection (0-9 scale), not a fatal flaw.
"""

import logging
from typing import Optional

import anthropic

from .base import BaseAgent, BroadcastFn

logger = logging.getLogger(__name__)


class DevilsAdvocateAgent(BaseAgent):
    def __init__(self, client: anthropic.AsyncAnthropic, broadcast: Optional[BroadcastFn] = None):
        super().__init__(client, "Advocate", model="claude-sonnet-4-6", broadcast=broadcast)

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
        await self._emit("status", f"Devil's Advocate: stress-testing {symbol} {direction} thesis...")

        premium     = options_analysis.get("estimated_premium", 0) or 0
        ba_pct      = options_analysis.get("bid_ask_spread_pct", 0) or 0
        vol         = options_analysis.get("volume", 0) or 0
        oi          = options_analysis.get("open_interest", 0) or 0
        price       = options_analysis.get("strike", 100) or 100
        stock_price = price  # close enough for the ratio check

        # Pre-compute hard fatal flaw checks (definitive, no LLM subjectivity needed)
        pre_fatal = None
        if fundamental.get("earnings_before_expiry") is True:
            pre_fatal = f"Earnings confirmed before expiry {options_analysis.get('expiration_date')} — gap risk disqualifies this trade"
        elif ba_pct > 0.30:
            pre_fatal = f"Bid-ask spread is {ba_pct:.0%} of mid — cannot exit at a fair price"
        elif vol == 0 and oi == 0:
            pre_fatal = f"Zero volume and zero open interest on the {options_analysis.get('strike')} strike — no market"

        context = f"""Proposed trade to stress-test:
{symbol} — Buy {options_analysis.get('option_type', '')} | Strike ${options_analysis.get('strike', '?')} | Exp {options_analysis.get('expiration_date', '?')} ({options_analysis.get('dte', '?')} DTE)
Premium: ${premium:.2f}/share | Contracts: {risk.get('contracts', 0)} | Total risk: ${risk.get('total_cost', 0):.2f}
Bid-Ask Spread: {ba_pct:.0%} of mid | Volume: {vol} | OI: {oi}

Bull case (what supporters say):
- Technical: {technical.get('score', 5)}/10 — {technical.get('trend', '?')} | {technical.get('summary', '')}
- Options:   {options_analysis.get('score', 5)}/10 | {options_analysis.get('summary', '')}
- Fundamental: {fundamental.get('score', 5)}/10 | {fundamental.get('summary', '')}
- Sentiment: {sentiment.get('score', 5)}/10 | {sentiment.get('summary', '')}
- Risk: {risk.get('score', 5)}/10 | {risk.get('summary', '')}
- Earnings before expiry: {fundamental.get('earnings_before_expiry', False)}

Your job: find the 2-4 strongest SPECIFIC reasons this trade could fail.
Use actual numbers from the data. Be honest — if the setup is actually solid, say so (objection_strength 1-3).

IMPORTANT — fatal_flaw definition (must meet at least one):
- Confirmed earnings within the DTE window
- Bid-ask spread > 30% (exit is impossible at fair price)
- Zero liquidity (volume=0 AND OI=0 on chosen strike)
- Premium > 8% of stock price
For anything else: use objection_strength (1-9) — NOT a fatal flaw."""

        system = """You are a professional risk manager finding specific reasons a trade could fail.
You are honest — if the setup is clean, say so and give a LOW objection_strength (1-4).
You are NOT automatically trying to kill every trade. Many trades ARE worth taking.

fatal_flaw: ONLY set this (non-null) for the four specific criteria listed. Otherwise null.

Respond ONLY with JSON:
{
  "objection_strength": <1-9, where 9=very strong case against, 1-3=minor concerns only>,
  "key_objections": [
    "<specific objection with numbers>",
    "<specific objection with numbers>"
  ],
  "fatal_flaw": null,
  "summary": "<2 sentences — specific risks and net verdict on this trade>"
}"""

        raw = await self._call(system, [{"role": "user", "content": context}], max_tokens=500)
        result = self._parse_json(raw)
        result.setdefault("objection_strength", 4)
        result.setdefault("key_objections", [])
        result.setdefault("fatal_flaw", None)
        result.setdefault("summary", "No major objections found.")

        # Override with pre-computed fatal flaws (these are definitive)
        if pre_fatal:
            result["fatal_flaw"] = pre_fatal
            result["objection_strength"] = 9

        # Normalize: if objection_strength is a string, cast to int
        try:
            result["objection_strength"] = int(result["objection_strength"])
        except (ValueError, TypeError):
            result["objection_strength"] = 5

        await self._emit("challenge", {
            "objection_strength": result["objection_strength"],
            "fatal_flaw": result.get("fatal_flaw"),
            "summary": result.get("summary"),
        })
        return result
