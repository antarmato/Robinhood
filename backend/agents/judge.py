"""
Judge Agent — final decision-maker and option parameter architect.

Architecture (v3 — deterministic scoring):
  - Python computes weighted_score mathematically from actual agent outputs
  - Claude ONLY provides: confidence (1-10), reasoning, bull_case, bear_case
  - This removes the "Claude lands at 44 to safely fail" bias
  - Decision is made by Python, not Claude

Thresholds:
  - weighted_score >= 38 (market hours) or 32 (after-hours)
  - confidence >= 5/10
  - No fatal flaw from advocate
"""

import logging
from datetime import datetime, time as dtime
from typing import Optional

import anthropic
try:
    from zoneinfo import ZoneInfo
    _ET = ZoneInfo("America/New_York")
except ImportError:
    _ET = None

from .base import BaseAgent, BroadcastFn
from .. import market_data as md

logger = logging.getLogger(__name__)

THRESHOLD_MARKET     = 38   # Python-computed score threshold (market hours)
THRESHOLD_AFTERHOURS = 32
THRESHOLD_CONF       = 5    # minimum Claude confidence to trade


def _compute_score(technical: dict, fundamental: dict, sentiment: dict,
                   risk: dict, advocate: dict) -> float:
    """
    Deterministic weighted score — not computed by Claude.

    Weights:
      tech  * 3.0   (max 30) — primary signal
      fund  * 1.5   (max 15) — context/risk
      sent  * 1.5   (max 15) — macro environment
      risk  * 1.0   (max 10) — always ~8, small contribution
      adv   * -1.5  (max -7.5, hard-capped) — skepticism discount

    Threshold 38 means: need at least tech=6 OR strong fund+sent with tech=5.
    Pure ranging (tech=5, neutral everything): ~36 → correct PASS.
    Decent setup (tech=7, normal fund/sent): ~42 → TRADE.
    """
    tech_s = float(technical.get("score", 5))
    fund_s = float(fundamental.get("score", 5))
    sent_s = float(sentiment.get("score", 5))
    risk_s = float(risk.get("score", 8))

    # Hard-cap advocate at 5 — prevents single agent killing a 45-point setup
    adv_s  = min(5.0, float(advocate.get("objection_strength", 3)))

    score = (
        tech_s * 3.0
        + fund_s * 1.5
        + sent_s * 1.5
        + risk_s * 1.0
        - adv_s  * 1.5
    )
    return round(score, 1)


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
        symbol_history: list | None = None,
    ) -> dict:
        await self._emit("status", f"Judge: evaluating {symbol} {direction}...")

        threshold = THRESHOLD_MARKET if market_open else THRESHOLD_AFTERHOURS
        session   = "LIVE MARKET" if market_open else "PRE-MARKET RESEARCH"

        # ── Hard reject first (before any API call) ───────────────────────────
        if advocate.get("fatal_flaw"):
            msg = f"Fatal flaw: {advocate['fatal_flaw']}"
            await self._emit("decision", {
                "symbol": symbol, "decision": "pass",
                "weighted_score": 0, "confidence": 0,
                "reasoning": msg, "pass_reason": msg, "strategy": "—",
            })
            return {"decision": "pass", "weighted_score": 0, "confidence": 0,
                    "reasoning": msg, "pass_reason": msg, "trade_proposal": None,
                    "bull_case": "", "bear_case": ""}

        # ── Deterministic score (Python, not Claude) ──────────────────────────
        weighted_score = _compute_score(technical, fundamental, sentiment, risk, advocate)
        score_failed   = weighted_score < threshold

        # ── IV / strategy ─────────────────────────────────────────────────────
        vix_level  = sentiment.get("vix_level", 20)
        vix_regime = sentiment.get("vix_regime", "normal")
        hv_data    = md.get_hv(symbol)
        hv_rank    = hv_data.get("hv_rank") or 50

        high_iv = (vix_regime in ("elevated", "extreme") or vix_level > 22 or hv_rank > 65)
        recommended_strategy = "debit_spread" if high_iv else "naked_option"

        price = technical.get("current_price") or risk.get("current_price") or 100
        try:
            price = float(price)
        except (TypeError, ValueError):
            price = 100.0
        spread_width = 5 if price < 200 else (10 if price < 500 else 25)
        option_type  = "call" if direction == "bullish" else "put"

        # ── History block ─────────────────────────────────────────────────────
        hist_lines = []
        consistency_note = ""
        if symbol_history:
            for h in symbol_history[-5:]:
                hist_lines.append(
                    f"  Cycle {h['cycle']}: {h['direction']} decision={h['decision']} "
                    f"score={h['score']} tech={h['tech_score']} sent={h['sent_score']} "
                    f"adv={h['adv_strength']}"
                )
            recent = symbol_history[-3:]
            if all(h.get('tech_score') and h['tech_score'] >= 6 and h['direction'] == 'bullish' for h in recent):
                consistency_note = "⚡ CONSISTENCY: Bullish 6+ tech for 3 straight cycles — elevated conviction."
            elif all(h.get('tech_score') and h['tech_score'] >= 6 and h['direction'] == 'bearish' for h in recent):
                consistency_note = "⚡ CONSISTENCY: Bearish 6+ tech for 3 straight cycles — elevated conviction."

        # ── Time of day ───────────────────────────────────────────────────────
        now_et = datetime.now(_ET) if _ET else datetime.now()
        tod    = now_et.time()
        if   tod < dtime(9, 30):  tod_note = "Pre-open warm-up."
        elif tod < dtime(10, 30): tod_note = "Opening hour — momentum setups preferred."
        elif tod < dtime(12, 0):  tod_note = "Mid-morning — trends establishing."
        elif tod < dtime(14, 0):  tod_note = "Midday lull — raise bar for low-conviction setups."
        elif tod < dtime(15, 30): tod_note = "Afternoon — institutional activity picks up."
        else:                     tod_note = "Power hour — directional moves tend to follow through."
        if now_et.weekday() == 0:
            tod_note += " Monday open — check for overnight gap vs scanner price."

        # ── Build Claude context ──────────────────────────────────────────────
        hist_block = ""
        if hist_lines:
            hist_block = "Prior cycles:\n" + "\n".join(hist_lines)
            if consistency_note:
                hist_block += f"\n{consistency_note}"

        tech_s = technical.get('score', 5)
        fund_s = fundamental.get('score', 5)
        sent_s = sentiment.get('score', 5)
        adv_s  = min(5, advocate.get('objection_strength', 3))

        context = f"""
[{session} | Cycle {cycle} | {tod_note}]

TRADE: {symbol} {direction.upper()} {option_type} @ ${price:.2f}
Strategy: {recommended_strategy.upper()} | VIX {vix_level} ({vix_regime}) | HV Rank {hv_rank}/100

AGENT SCORES:
  Technical   {tech_s}/10: {technical.get('trend','?')} — {technical.get('summary','')}
  Fundamental {fund_s}/10: {fundamental.get('summary','')}
  Sentiment   {sent_s}/10: {sentiment.get('summary','')}
  Advocate    strength={adv_s}/9 (capped at 5): {advocate.get('summary','')}
  Key objections: {advocate.get('key_objections', [])}

COMPUTED SCORE: {weighted_score} / threshold {threshold}
Score breakdown: tech({tech_s}×3={tech_s*3}) + fund({fund_s}×1.5={fund_s*1.5}) + sent({sent_s}×1.5={sent_s*1.5}) + risk(8×1.0=8.0) - adv({adv_s}×1.5={adv_s*1.5}) = {weighted_score}
{"✅ SCORE PASSES — provide confidence and reasoning" if not score_failed else f"❌ SCORE FAILS ({weighted_score} < {threshold}) — provide pass_reason"}

{hist_block}
"""

        # ── Claude prompt: reasoning + confidence ONLY ─────────────────────────
        system = f"""You are the final judge for an options trading system.
The weighted score has already been computed mathematically: {weighted_score} (threshold: {threshold}).

{"The score PASSES. Your job: provide confidence (1-10) and reasoning. If you have genuine conviction concerns, reflect them in confidence. Trade happens if confidence >= {THRESHOLD_CONF}." if not score_failed else f"The score FAILS. Confirm pass with clear reason."}

RULES:
- Confidence 1-4: serious reservations (risky timing, real headwinds)
- Confidence 5-6: reasonable setup, normal uncertainty
- Confidence 7-8: clear directional setup, good risk/reward
- Confidence 9-10: exceptional setup, multiple confirming signals
- Do NOT factor in missing options data (OI, IV, bid-ask) — handled at execution
- Ranging market alone is NOT a confidence killer if technicals already show that in their score

Respond ONLY with JSON:
{{
  "confidence": <int 1-10>,
  "reasoning": "<2-3 sentences: what drives your conviction or lack thereof>",
  "bull_case": "<strongest reason this works>",
  "bear_case": "<strongest reason it fails>",
  "pass_reason": "<only if score failed: one clear reason>"
}}"""

        raw    = await self._call(system, [{"role": "user", "content": context}], max_tokens=500)
        result = self._parse_json(raw)

        confidence  = int(result.get("confidence", 5))
        reasoning   = result.get("reasoning", "")
        bull_case   = result.get("bull_case", "")
        bear_case   = result.get("bear_case", "")
        pass_reason = result.get("pass_reason", "")

        # ── Final decision (Python, not Claude) ───────────────────────────────
        if score_failed:
            decision    = "pass"
            pass_reason = pass_reason or f"Score {weighted_score} below threshold {threshold}"
            trade_proposal = None
        elif confidence < THRESHOLD_CONF:
            decision    = "pass"
            pass_reason = f"Confidence {confidence}/10 below minimum {THRESHOLD_CONF}"
            trade_proposal = None
        else:
            decision = "trade"
            pass_reason = None

            mp = risk.get("max_premium", 2.0)
            try:
                mp = float(mp)
            except (TypeError, ValueError):
                mp = 2.0

            trade_proposal = {
                "symbol":         symbol,
                "option_type":    option_type,
                "direction":      direction,
                "dte_min":        21,
                "dte_max":        45,
                "delta_target":   0.40,
                "max_premium":    round(mp, 2),
                "total_max_loss": int(mp * 100 * risk.get("contracts", 1)),
                "contracts":      risk.get("contracts", 1),
                "strategy":       recommended_strategy,
                "short_delta":    0.20 if recommended_strategy == "debit_spread" else None,
                "spread_width":   spread_width if recommended_strategy == "debit_spread" else None,
            }

        await self._emit("decision", {
            "symbol":         symbol,
            "decision":       decision,
            "weighted_score": weighted_score,
            "confidence":     confidence,
            "reasoning":      reasoning,
            "pass_reason":    pass_reason,
            "strategy":       trade_proposal.get("strategy", "—") if trade_proposal else "—",
        })

        return {
            "decision":       decision,
            "weighted_score": weighted_score,
            "confidence":     confidence,
            "reasoning":      reasoning,
            "bull_case":      bull_case,
            "bear_case":      bear_case,
            "pass_reason":    pass_reason,
            "trade_proposal": trade_proposal,
        }
