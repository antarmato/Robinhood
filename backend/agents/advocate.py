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
        options_analysis: dict,  # kept for compatibility, not used for fatal flaws
        fundamental: dict,
        sentiment: dict,
        risk: dict,
    ) -> dict:
        await self._emit("status", f"Devil's Advocate: stress-testing {symbol} {direction} thesis...")

        # Pre-compute only definitive fatal flaw: confirmed earnings within 45 days
        pre_fatal = None
        if fundamental.get("earnings_before_expiry") is True:
            earn_date = fundamental.get("earnings_date", "upcoming")
            pre_fatal = f"Earnings confirmed {earn_date} — binary gap risk before we can exit"

        vix_regime = sentiment.get("vix_regime", "normal")
        vix_level  = sentiment.get("vix_level", 20)

        context = f"""Proposed trade to stress-test:
{symbol} — {direction.upper()} | Budget: {risk.get('contracts', 1)} contract(s), max ${risk.get('max_premium', 0):.2f}/share

NOTE: Options chain data (OI, volume, IV, bid-ask spreads) is NOT available at this stage — it is fetched at execution time. Do NOT raise options liquidity or premium cost as objections here.

Bull case:
- Technical ({technical.get('score', 5)}/10): {technical.get('trend', '?')} | {technical.get('summary', '')}
- Fundamental ({fundamental.get('score', 5)}/10): {fundamental.get('summary', '')}
- Sentiment ({sentiment.get('score', 5)}/10): VIX={vix_regime} ({vix_level}) | {sentiment.get('summary', '')}
- Earnings in 45-day window: {fundamental.get('earnings_before_expiry', False)}

Your job: find 2-4 SPECIFIC, data-backed reasons this DIRECTIONAL trade could fail:
  * Trend or momentum concerns (use the technical numbers)
  * Poor timing (RSI extreme, trend exhaustion, price at resistance)
  * Macro or sector headwinds
  * Calendar risks (earnings, Fed, macro events)
  * VIX={vix_regime}: {'elevated volatility means the stock can chop violently' if vix_regime in ('elevated', 'extreme') else 'normal/low VIX'}

Be calibrated — if the setup is genuinely solid, objection_strength 1-3 is correct.

FATAL FLAW (only these 3 qualify — everything else is objection_strength):
1. Confirmed earnings within the next 45 days
2. RSI > 80 on a bullish call play
3. RSI < 20 on a bearish put play"""

        system = """You are a risk manager stress-testing directional option trade theses. Be calibrated.
Strong setups deserve objection_strength 1-3. Only cry wolf on real problems.
Do NOT mention options data (OI, volume, IV, bid-ask) — not your concern at this stage.
Focus on: directional timing, macro headwinds, momentum quality, calendar risk.

Respond ONLY with JSON:
{
  "objection_strength": <1-9>,
  "key_objections": ["<specific objection with data>", "<specific objection with data>"],
  "fatal_flaw": null,
  "summary": "<2 sentences: key risks and net verdict>"
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
