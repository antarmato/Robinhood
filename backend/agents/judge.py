"""
Judge Agent — final go/no-go decision using Claude Opus.

Scoring model (0-100):
  Technical     25%  (trend, momentum, indicators)
  Options       20%  (IV environment, strike quality, liquidity)
  Risk          20%  (sizing, cost, position count) ← increased from 10%
  Fundamental   15%  (earnings risk, company health)  ← reduced from 20%
  Sentiment     10%  (macro, PCR, VIX)  ← reduced from 15%
  Advocate      10%  (penalty: 0-10 pts based on objection_strength)
               ----
               100%

Pass thresholds:
  weighted_score < 52 → auto-pass (below average across all dimensions)
  confidence < 5 → auto-pass (not enough conviction)
  advocate fatal_flaw → auto-pass (show-stopper identified)
  risk.approved == False → auto-pass (can't size the trade)
  earnings_before_expiry → auto-pass (gap risk)

The threshold of 52 means: if all agents score 6+/10 (slightly above average),
the system will trade. Average setups (5/10 across all) yield ~50 — just below threshold.
Good setups (7/10) yield ~70 — comfortably above threshold.
"""

import logging
from typing import Optional

import anthropic

from .base import BaseAgent, BroadcastFn

logger = logging.getLogger(__name__)

PASS_THRESHOLD_SCORE = 52      # minimum weighted score (0-100)
PASS_THRESHOLD_CONF  = 5       # minimum confidence (1-10)


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
        await self._emit("status", f"Judge: final deliberation on {symbol} {direction} (cycle {cycle_number})...")

        # Pre-compute weighted score for transparency
        tech_score   = float(technical.get("score", 5))
        opts_score   = float(options_analysis.get("score", 5))
        risk_score   = float(risk.get("score", 5))
        fund_score   = float(fundamental.get("score", 5))
        sent_score   = float(sentiment.get("score", 5))
        obj_str      = float(advocate.get("objection_strength", 4))

        # Advocate penalty: 0 pts at obj_str=1, 10 pts at obj_str=9
        advocate_penalty = round((obj_str - 1) / 8 * 10, 1)

        # Weighted score: each dimension already on 1-10 scale
        raw_weighted = (
            tech_score * 0.25 +
            opts_score * 0.20 +
            risk_score * 0.20 +
            fund_score * 0.15 +
            sent_score * 0.10
        ) * 10  # scale to 0-100
        weighted_score = round(raw_weighted - advocate_penalty, 1)

        # Hard rejection checks (no LLM needed)
        hard_rejection = None
        if not risk.get("approved", True):
            hard_rejection = risk.get("rejection_reason", "Risk manager rejected")
        elif fundamental.get("earnings_before_expiry") is True:
            hard_rejection = f"Earnings before expiry {options_analysis.get('expiration_date')} — gap risk"
        elif advocate.get("fatal_flaw"):
            hard_rejection = f"Fatal flaw: {advocate['fatal_flaw']}"
        elif weighted_score < PASS_THRESHOLD_SCORE:
            hard_rejection = f"Weighted score {weighted_score:.0f} below threshold {PASS_THRESHOLD_SCORE}"

        context = f"""=== TRADING COMMITTEE — CYCLE {cycle_number} ===

PROPOSED TRADE:
  Symbol:    {symbol} | Direction: {direction}
  Option:    Buy {options_analysis.get('option_type', '')} ${options_analysis.get('strike', '?')} exp {options_analysis.get('expiration_date', '?')} ({options_analysis.get('dte', '?')} DTE)
  Premium:   ${options_analysis.get('estimated_premium', 0):.2f}/share
  IV:        {options_analysis.get('iv', 0):.1%}
  Liquidity: Vol={options_analysis.get('volume', 0)}, OI={options_analysis.get('open_interest', 0)}, Spread={options_analysis.get('bid_ask_spread_pct', 0):.0%}
  Sizing:    {risk.get('contracts', 0)} contract(s) @ ${options_analysis.get('estimated_premium', 0):.2f} = ${risk.get('total_cost', 0):.2f} max loss

PRE-COMPUTED WEIGHTED SCORE: {weighted_score:.1f}/100
  Technical  (25%): {tech_score}/10 → {tech_score * 2.5:.1f} pts
  Options    (20%): {opts_score}/10 → {opts_score * 2.0:.1f} pts
  Risk       (20%): {risk_score}/10 → {risk_score * 2.0:.1f} pts
  Fundamental(15%): {fund_score}/10 → {fund_score * 1.5:.1f} pts
  Sentiment  (10%): {sent_score}/10 → {sent_score * 1.0:.1f} pts
  Raw score: {raw_weighted:.1f} − Advocate penalty: {advocate_penalty:.1f} = {weighted_score:.1f}

{'⛔ HARD REJECTION: ' + hard_rejection if hard_rejection else '✓ Passes hard checks — needs your judgment'}

─── TECHNICAL ({tech_score}/10) ───
Trend: {technical.get('trend', '?')} | Signal: {technical.get('signal', '?')} | ADX: {technical.get('adx_reading', 'N/A')}
RSI: {technical.get('rsi_reading', 'N/A')} | MACD: {technical.get('macd_reading', 'N/A')}
Intraday: {technical.get('intraday_momentum', 'N/A')}
{technical.get('summary', '')}

─── OPTIONS ({opts_score}/10) ───
Strike ${options_analysis.get('strike', '?')} | {options_analysis.get('dte', '?')} DTE | IV {options_analysis.get('iv', 0):.1%}
{options_analysis.get('summary', '')}

─── RISK ({risk_score}/10) ───
Approved: {risk.get('approved', False)} | {risk.get('contracts', 0)} contracts | ${risk.get('total_cost', 0):.2f} max loss
{risk.get('rejection_reason', '') or risk.get('summary', '')}

─── FUNDAMENTAL ({fund_score}/10) ───
Earnings in window: {fundamental.get('earnings_before_expiry', False)}
{fundamental.get('summary', '')}

─── SENTIMENT ({sent_score}/10) ───
VIX: {sentiment.get('vix_regime', 'N/A')} | Skew: {sentiment.get('skew', 'N/A')} | Macro: {sentiment.get('macro_sentiment', 'N/A')}
{sentiment.get('summary', '')}

─── ADVOCATE (penalty: {advocate_penalty:.0f}pts) ───
Objection strength: {obj_str}/9 | Fatal flaw: {advocate.get('fatal_flaw') or 'None'}
{chr(10).join('  • ' + o for o in advocate.get('key_objections', []))}
{advocate.get('summary', '')}"""

        system = f"""You are the head trader. The committee has scored this trade.

PASS CONDITIONS (if ANY are true, decision must be "pass"):
1. Hard rejection is present (already marked in the context)
2. You independently assess confidence < {PASS_THRESHOLD_CONF}
3. You identify a critical risk not caught by the other agents

TRADE CONDITIONS (if ALL are true):
1. No hard rejection
2. Confidence >= {PASS_THRESHOLD_CONF}
3. The trade thesis is coherent and the risk/reward is favorable

profit_target_pct: 50 (exit when option up 50% — take the money)
stop_loss_pct: 50 (exit when option down 50% from purchase — cut losses)

trade_proposal MUST include all fields below. Use the risk agent's contracts and limit_price.

Respond ONLY with JSON:
{{
  "decision": "trade" | "pass",
  "confidence": <1-10>,
  "weighted_score": {weighted_score},
  "pass_reason": null | "<specific reason>",
  "trade_proposal": {{
    "symbol": "{symbol}",
    "option_type": "{options_analysis.get('option_type', '')}",
    "strike": {options_analysis.get('strike', 0)},
    "expiration_date": "{options_analysis.get('expiration_date', '')}",
    "contracts": {risk.get('contracts', 1)},
    "limit_price": {risk.get('limit_price', options_analysis.get('estimated_premium', 0))},
    "total_max_loss": {risk.get('total_cost', 0)},
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
        result["weighted_score"] = weighted_score  # use our computed value, not LLM's guess

        # Enforce hard rejections regardless of LLM output
        if hard_rejection:
            result["decision"] = "pass"
            result["pass_reason"] = hard_rejection
            result["trade_proposal"] = None
        elif result["decision"] == "trade":
            # Enforce confidence threshold
            if result.get("confidence", 0) < PASS_THRESHOLD_CONF:
                result["decision"] = "pass"
                result["pass_reason"] = f"Confidence {result['confidence']} below minimum {PASS_THRESHOLD_CONF}"
                result["trade_proposal"] = None
            # Ensure trade_proposal is complete
            elif result.get("trade_proposal"):
                tp = result["trade_proposal"]
                tp.setdefault("symbol", symbol)
                tp.setdefault("option_type", options_analysis.get("option_type", ""))
                tp.setdefault("strike", options_analysis.get("strike", 0))
                tp.setdefault("expiration_date", options_analysis.get("expiration_date", ""))
                tp.setdefault("contracts", risk.get("contracts", 1))
                tp.setdefault("limit_price", risk.get("limit_price", options_analysis.get("estimated_premium", 0)))
                tp.setdefault("total_max_loss", risk.get("total_cost", 0))
                tp.setdefault("profit_target_pct", 50)
                tp.setdefault("stop_loss_pct", 50)
            else:
                # LLM said "trade" but no proposal — treat as pass
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
