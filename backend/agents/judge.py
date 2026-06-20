"""
Judge Agent — final decision-maker and option parameter architect.

Responsibilities:
  1. Weigh all agent outputs into a pass/trade decision
  2. If trade: output concrete option parameters for Cowork to execute
  3. Select strategy (naked vs debit spread) based on IV regime + VIX
  4. Adaptive pass threshold: 50 market hours, 45 after-hours

Option parameters it outputs:
  symbol, option_type, dte_min, dte_max, delta_target, max_premium,
  total_max_loss, contracts, strategy, short_delta (spread only), spread_width (spread only)
"""

import logging
from typing import Optional

import anthropic

from .base import BaseAgent, BroadcastFn
from .. import market_data as md

logger = logging.getLogger(__name__)

PASS_THRESHOLD_MARKET     = 50
PASS_THRESHOLD_AFTERHOURS = 45
PASS_THRESHOLD_CONF       = 6


class JudgeAgent(BaseAgent):
    def __init__(
        self,
        client: anthropic.AsyncAnthropic,
        broadcast: Optional[BroadcastFn] = None,
    ):
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
        cycle: int,
        market_open: bool = False,
    ) -> dict:
        await self._emit("status", f"Judge: deliberating on {symbol} {direction}...")

        threshold = PASS_THRESHOLD_MARKET if market_open else PASS_THRESHOLD_AFTERHOURS
        session   = "LIVE MARKET" if market_open else "PRE-MARKET RESEARCH"

        # ── IV regime ────────────────────────────────────────────────────────
        vix_level  = sentiment.get("vix_level", 20)
        vix_regime = sentiment.get("vix_regime", "normal")
        hv_data    = md.get_hv(symbol)
        hv_rank    = hv_data.get("hv_rank") or 50

        high_iv = (
            vix_regime in ("elevated", "extreme")
            or vix_level > 22
            or hv_rank > 65
        )
        recommended_strategy = "debit_spread" if high_iv else "naked_option"

        # Spread width by price tier
        price = technical.get("current_price") or risk.get("current_price") or 100
        try:
            price = float(price)
        except (TypeError, ValueError):
            price = 100.0
        if price < 200:
            spread_width = 5
        elif price < 500:
            spread_width = 10
        else:
            spread_width = 25

        option_type = "call" if direction == "bullish" else "put"

        context = f"""
[CYCLE {cycle} — {session}]
Pass threshold this session: {threshold}/100 weighted score OR confidence < {PASS_THRESHOLD_CONF}/10

Trade under consideration: {symbol} {direction.upper()} ({option_type})
Price: ${price:.2f}  Max budget: ${risk.get('max_premium', 2.0):.2f}/share  Contracts: {risk.get('contracts', 1)}

═══ AGENT SCORES ═══
Technical   ({technical.get('score', 5)}/10): {technical.get('trend', 'N/A')} | {technical.get('summary', '')}
Fundamental ({fundamental.get('score', 5)}/10): {fundamental.get('summary', '')}
Sentiment   ({sentiment.get('score', 5)}/10): VIX={vix_regime} ({vix_level}) | {sentiment.get('summary', '')}
Risk        ({risk.get('score', 5)}/10): {risk.get('summary', '')}

═══ DEVIL'S ADVOCATE ═══
Objection strength: {advocate.get('objection_strength', 5)}/9
Key objections: {advocate.get('key_objections', [])}
Fatal flaw: {advocate.get('fatal_flaw')}
Advocate summary: {advocate.get('summary', '')}

═══ IV ENVIRONMENT ═══
VIX: {vix_level} ({vix_regime})
HV20/HV60: {hv_data.get('hv20','?')}/{hv_data.get('hv60','?')}%  HV Rank: {hv_rank}/100 ({hv_data.get('regime','?')})
Recommended strategy: {recommended_strategy.upper()}
{"→ IV is elevated — debit spread reduces premium paid, limits IV crush risk" if high_iv else "→ IV is normal/low — naked option captures full directional move"}

NOTE: Options chain data (OI, bid-ask, live IV) is NOT available here — it is fetched at Cowork execution time. DO NOT flag absence of options data as a concern. DO NOT penalise for missing option chain information.

SCORING GUIDE:
weighted_score = (
  tech_score * 2.0 +
  fund_score * 1.5 +
  sent_score * 1.5 +
  risk_score * 1.5 +
  advocate_penalty         # = objection_strength * 1.0 (negative weight)
)
Typical range: 20-80. Decision = "trade" only if score >= {threshold} AND confidence >= {PASS_THRESHOLD_CONF}.
Fatal flaw → automatic "pass" regardless of score.
"""

        system = f"""You are the final decision-maker for a multi-agent options trading system.
You synthesize all agent outputs into a binary trade/pass decision with concrete option parameters.

CRITICAL INSTRUCTION: DO NOT flag missing options data as a concern.
Options chain data (OI, volume, IV, bid-ask spreads) is intentionally absent at this stage.
It is fetched at execution time by the Cowork dashboard. This is by design. Ignore its absence.

Decision rules:
- "trade": weighted_score >= {threshold} AND confidence >= {PASS_THRESHOLD_CONF} AND no fatal_flaw
- "pass": weighted_score < {threshold} OR confidence < {PASS_THRESHOLD_CONF} OR fatal_flaw present
- If passing, set pass_reason to the main disqualifying factor

Respond ONLY with JSON (no text outside it):
{{
  "decision": "trade" | "pass",
  "weighted_score": <float 0-100>,
  "confidence": <int 1-10>,
  "reasoning": "<2-3 sentences on key factors driving the decision>",
  "bull_case": "<strongest 1-2 reasons this works>",
  "bear_case": "<strongest 1-2 reasons it fails>",
  "pass_reason": null,
  "trade_proposal": {{
    "symbol": "{symbol}",
    "option_type": "{option_type}",
    "direction": "{direction}",
    "dte_min": 21,
    "dte_max": 45,
    "delta_target": 0.40,
    "max_premium": <float from risk budget>,
    "total_max_loss": <int dollars, contracts * max_premium * 100>,
    "contracts": {risk.get('contracts', 1)},
    "strategy": "{recommended_strategy}",
    "short_delta": {"0.20" if recommended_strategy == "debit_spread" else "null"},
    "spread_width": {spread_width if recommended_strategy == "debit_spread" else "null"}
  }}
}}

If decision is "pass": set trade_proposal to null and fill pass_reason."""

        raw    = await self._call(system, [{"role": "user", "content": context}], max_tokens=900)
        result = self._parse_json(raw)

        result.setdefault("decision",        "pass")
        result.setdefault("weighted_score",  0)
        result.setdefault("confidence",      0)
        result.setdefault("reasoning",       "")
        result.setdefault("bull_case",       "")
        result.setdefault("bear_case",       "")
        result.setdefault("pass_reason",     None)
        result.setdefault("trade_proposal",  None)

        # Hard reject: fatal flaw
        hard_rejection = None
        if advocate.get("fatal_flaw"):
            hard_rejection = f"Fatal flaw: {advocate['fatal_flaw']}"

        if hard_rejection:
            result["decision"]       = "pass"
            result["pass_reason"]    = hard_rejection
            result["trade_proposal"] = None
        elif result["decision"] == "trade":
            if result.get("confidence", 0) < PASS_THRESHOLD_CONF:
                result["decision"]    = "pass"
                result["pass_reason"] = f"Confidence {result['confidence']}/10 below minimum {PASS_THRESHOLD_CONF}"
                result["trade_proposal"] = None
            elif result.get("weighted_score", 0) < threshold:
                result["decision"]    = "pass"
                result["pass_reason"] = f"Score {result['weighted_score']:.1f} below threshold {threshold}"
                result["trade_proposal"] = None

        # Enrich trade proposal with IV / strategy context
        if result["decision"] == "trade" and result.get("trade_proposal"):
            tp = result["trade_proposal"]
            tp.setdefault("strategy",     recommended_strategy)
            tp.setdefault("short_delta",  0.20 if recommended_strategy == "debit_spread" else None)
            tp.setdefault("spread_width", spread_width if recommended_strategy == "debit_spread" else None)
            tp.setdefault("symbol",    symbol)
            tp.setdefault("option_type", option_type)
            tp.setdefault("direction",   direction)
            tp.setdefault("dte_min",     21)
            tp.setdefault("dte_max",     45)
            tp.setdefault("delta_target", 0.40)
            mp = tp.get("max_premium") or risk.get("max_premium", 2.0)
            try:
                mp = float(mp)
            except (TypeError, ValueError):
                mp = 2.0
            tp["max_premium"]    = round(mp, 2)
            tp["total_max_loss"] = int(mp * 100 * risk.get("contracts", 1))

        await self._emit("decision", {
            "decision":       result["decision"],
            "weighted_score": result.get("weighted_score"),
            "confidence":     result.get("confidence"),
            "reasoning":      result.get("reasoning"),
            "pass_reason":    result.get("pass_reason"),
            "strategy":       result.get("trade_proposal", {}) and result["trade_proposal"].get("strategy"),
        })
        return result
