"""
Judge Agent — final go/no-go decision using Claude Opus.

New architecture: no options data at research time.
Judge recommends option parameters; Cowork selects the actual strike/expiry at execution.

Scoring model (0-100):
  Technical     35%  (trend, momentum, indicators)
  Fundamental   25%  (earnings risk, company health)
  Sentiment     20%  (macro, VIX, PCR)
  Risk          20%  (position count, budget feasibility)
  Advocate      penalty: 0-10 pts based on objection_strength
               ────
               100%

Pass thresholds:
  weighted_score < 52 → auto-pass
  confidence < 5 → auto-pass
  advocate fatal_flaw → auto-pass
  risk.approved == False → auto-pass
"""

import logging
from typing import Optional

import anthropic

from .base import BaseAgent, BroadcastFn

logger = logging.getLogger(__name__)

PASS_THRESHOLD_SCORE = 52
PASS_THRESHOLD_CONF  = 5


class JudgeAgent(BaseAgent):
    def __init__(self, client: anthropic.AsyncAnthropic, broadcast: Optional[BroadcastFn] = None):
        super().__init__(client, "Judge", model="claude-opus-4-8", broadcast=broadcast)

    async def decide(
        self,
        symbol: str,
        direction: str,
        technical: dict,
        fundamental: dict,
        sentiment: dict,
        risk: dict,
        advocate: dict,
        cycle_number: int = 1,
    ) -> dict:
        await self._emit("status", f"Judge: final deliberation on {symbol} {direction} (cycle {cycle_number})...")

        tech_score  = float(technical.get("score", 5))
        fund_score  = float(fundamental.get("score", 5))
        sent_score  = float(sentiment.get("score", 5))
        risk_score  = float(risk.get("score", 5))
        obj_str     = float(advocate.get("objection_strength", 4))

        # Advocate penalty: 0 pts at obj_str=1, 10 pts at obj_str=9
        advocate_penalty = round((obj_str - 1) / 8 * 10, 1)

        raw_weighted = (
            tech_score * 0.35 +
            fund_score * 0.25 +
            sent_score * 0.20 +
            risk_score * 0.20
        ) * 10  # scale to 0-100
        weighted_score = round(raw_weighted - advocate_penalty, 1)

        # Hard rejection checks
        hard_rejection = None
        if not risk.get("approved", True):
            hard_rejection = risk.get("rejection_reason", "Risk manager rejected")
        elif advocate.get("fatal_flaw"):
            hard_rejection = f"Fatal flaw: {advocate['fatal_flaw']}"
        elif weighted_score < PASS_THRESHOLD_SCORE:
            hard_rejection = f"Weighted score {weighted_score:.0f} below threshold {PASS_THRESHOLD_SCORE}"

        # Infer option type from direction
        option_type = "call" if direction == "bullish" else "put"
        max_premium = risk.get("max_premium", 2.00)
        contracts   = risk.get("contracts", 1)

        context = f"""=== TRADING COMMITTEE — CYCLE {cycle_number} ===

PROPOSED TRADE:
  Symbol:    {symbol} | Direction: {direction}
  Option:    Buy {option_type}s | Budget: ${max_premium:.2f}/share max premium, {contracts} contract(s)
  Max loss:  ${risk.get('total_max_loss', max_premium * contracts * 100):.0f}

PRE-COMPUTED WEIGHTED SCORE: {weighted_score:.1f}/100
  Technical  (35%): {tech_score}/10 → {tech_score * 3.5:.1f} pts
  Fundamental(25%): {fund_score}/10 → {fund_score * 2.5:.1f} pts
  Sentiment  (20%): {sent_score}/10 → {sent_score * 2.0:.1f} pts
  Risk       (20%): {risk_score}/10 → {risk_score * 2.0:.1f} pts
  Raw score: {raw_weighted:.1f} − Advocate penalty: {advocate_penalty:.1f} = {weighted_score:.1f}

{'⛔ HARD REJECTION: ' + hard_rejection if hard_rejection else '✓ Passes hard checks — needs your judgment'}

─── TECHNICAL ({tech_score}/10) ───
Trend: {technical.get('trend', '?')} | Signal: {technical.get('signal', '?')} | ADX: {technical.get('adx_reading', 'N/A')}
RSI: {technical.get('rsi_reading', 'N/A')} | MACD: {technical.get('macd_reading', 'N/A')}
Intraday: {technical.get('intraday_momentum', 'N/A')}
{technical.get('summary', '')}

─── FUNDAMENTAL ({fund_score}/10) ───
Earnings in 45-day window: {fundamental.get('earnings_before_expiry', False)}
Catalyst risk: {fundamental.get('catalyst_risk', 'N/A')} | Consensus: {fundamental.get('analyst_consensus', 'N/A')}
{fundamental.get('summary', '')}

─── SENTIMENT ({sent_score}/10) ───
VIX: {sentiment.get('vix_regime', 'N/A')} | Skew: {sentiment.get('skew', 'N/A')} | Macro: {sentiment.get('macro_sentiment', 'N/A')}
{sentiment.get('summary', '')}

─── RISK ({risk_score}/10) ───
Approved: {risk.get('approved', False)} | Budget: ${max_premium:.2f}/share | {contracts} contract(s)
{risk.get('summary', '')}

─── ADVOCATE (penalty: {advocate_penalty:.0f}pts) ───
Objection strength: {obj_str}/9 | Fatal flaw: {advocate.get('fatal_flaw') or 'None'}
{chr(10).join('  • ' + o for o in advocate.get('key_objections', []))}
{advocate.get('summary', '')}"""

        system = f"""You are the head trader. The committee has scored this {direction} trade on {symbol}.

PASS CONDITIONS (decision must be "pass" if ANY are true):
1. Hard rejection is present
2. You independently assess confidence < {PASS_THRESHOLD_CONF}
3. You identify a critical undiscovered risk

TRADE CONDITIONS (all must be true):
1. No hard rejection
2. Confidence >= {PASS_THRESHOLD_CONF}
3. The {direction} thesis is coherent and risk/reward is favorable

If decision is "trade", also recommend option parameters for execution.
These are GUIDELINES for Cowork to find the best contract at execution time.

dte_min/dte_max: days to expiry range (recommend 21-45 for most trades; shorter 14-21 for very high-conviction, longer 45-60 for uncertain timing)
delta_target: target delta 0.0-1.0 (0.45-0.55 = ATM; 0.35-0.45 = slightly OTM — prefer OTM when IV is high)
max_premium: maximum to pay per share in dollars (keep within ${max_premium:.2f} budget; lower = defined risk)

profit_target_pct: 50 (exit when option up 50%)
stop_loss_pct: 50 (exit when option down 50%)

Respond ONLY with JSON:
{{
  "decision": "trade" | "pass",
  "confidence": <1-10>,
  "weighted_score": {weighted_score},
  "pass_reason": null | "<specific reason>",
  "trade_proposal": {{
    "symbol": "{symbol}",
    "option_type": "{option_type}",
    "contracts": {contracts},
    "dte_min": <int, e.g. 21>,
    "dte_max": <int, e.g. 45>,
    "delta_target": <float, e.g. 0.40>,
    "max_premium": {max_premium},
    "total_max_loss": {risk.get('total_max_loss', max_premium * contracts * 100):.0f},
    "profit_target_pct": 50,
    "stop_loss_pct": 50
  }},
  "bull_case": "<one sentence — strongest reason TO trade>",
  "bear_case": "<one sentence — strongest reason NOT to trade>",
  "reasoning": "<3-5 sentences synthesizing the committee's views and explaining the decision>"
}}

If decision is "pass", set trade_proposal to null."""

        raw = await self._call(system, [{"role": "user", "content": context}], max_tokens=900)
        result = self._parse_json(raw)
        result.setdefault("decision", "pass")
        result.setdefault("confidence", 5)
        result.setdefault("reasoning", "Analysis complete.")
        result["weighted_score"] = weighted_score  # use computed value

        # Enforce hard rejections
        if hard_rejection:
            result["decision"] = "pass"
            result["pass_reason"] = hard_rejection
            result["trade_proposal"] = None
        elif result["decision"] == "trade":
            if result.get("confidence", 0) < PASS_THRESHOLD_CONF:
                result["decision"] = "pass"
                result["pass_reason"] = f"Confidence {result['confidence']} below minimum {PASS_THRESHOLD_CONF}"
                result["trade_proposal"] = None
            elif result.get("trade_proposal"):
                tp = result["trade_proposal"]
                tp.setdefault("symbol", symbol)
                tp.setdefault("option_type", option_type)
                tp.setdefault("contracts", contracts)
                tp.setdefault("dte_min", 21)
                tp.setdefault("dte_max", 45)
                tp.setdefault("delta_target", 0.40)
                tp.setdefault("max_premium", max_premium)
                tp.setdefault("total_max_loss", max_premium * contracts * 100)
                tp.setdefault("profit_target_pct", 50)
                tp.setdefault("stop_loss_pct", 50)
            else:
                result["decision"] = "pass"
                result["pass_reason"] = "Trade decision but no proposal generated"
                result["trade_proposal"] = None

        await self._emit("decision", {
            "decision":       result["decision"],
            "symbol":         symbol,
            "direction":      direction,
            "confidence":     result.get("confidence"),
            "weighted_score": weighted_score,
            "pass_reason":    result.get("pass_reason"),
            "reasoning":      result.get("reasoning"),
        })
        return result
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                  