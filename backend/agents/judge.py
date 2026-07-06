"""
Judge Agent — final decision maker.

Architecture (v5 — self-learning loop):
  - Python computes weighted_score from agent outputs
  - Python applies IV-aware threshold (cheap IV = lower bar, expensive = harder bar)
  - Claude provides: confidence (1-10), reasoning, bull_case, bear_case
  - Context includes: IV rank label, OutcomeTracker stats, risk flags,
    AND self-learned calibration from PostgreSQL scan_log (win rates by regime,
    direction, score bucket, symbol, confidence — built from every past cycle)
  - Decision: Python only (weighted_score >= threshold AND confidence >= THRESHOLD_CONF)
  - Fatal flaws from Technical/Fundamental short-circuit before any LLM call
"""

import asyncio
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
from ..strategy import (
    score_threshold, iv_edge_label, trade_defaults, confidence_minimum,
    THRESHOLD_CONF, MARGINAL_CONF_SCORE_CUSHION,
)
from .. import training_store as ts

logger = logging.getLogger(__name__)


def _compute_score(technical: dict, fundamental: dict, sentiment: dict, risk: dict) -> float:
    """
    Deterministic weighted score.
    Weights:
      tech  × 3.0  (max 30) — primary directional signal
      fund  × 1.5  (max 15) — catalyst/earnings safety
      sent  × 1.5  (max 15) — macro environment
      risk  × 1.0  (max 10) — always ~8, small contribution
    Signal consensus bonus/penalty (graduated to remove 6.9→7.0 cliff):
      strong (≥7.0): +1.5 each  mid (6.0–6.9): +0.5 each  weak (<6.0): −1.5 each
      Capped ±3.0. Example: 3 mids → +1.5 (old: −3.0)
    """
    tech_s = float(technical.get("score", 5))
    fund_s = float(fundamental.get("score", 5))
    sent_s = float(sentiment.get("score", 5))
    risk_s = float(risk.get("score", 8))
    base = tech_s * 3.0 + fund_s * 1.5 + sent_s * 1.5 + risk_s * 1.0

    # Graduated consensus: strong/mid/weak tiers instead of a hard ≥7 cliff
    strong_count = sum([tech_s >= 7.0, fund_s >= 7.0, sent_s >= 7.0])
    mid_count    = sum([6.0 <= tech_s < 7.0, 6.0 <= fund_s < 7.0, 6.0 <= sent_s < 7.0])
    weak_count   = 3 - strong_count - mid_count
    consensus = strong_count * 1.5 + mid_count * 0.5 - weak_count * 1.5
    consensus = max(-3.0, min(3.0, consensus))

    return round(base + consensus, 1)


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
        cycle: int,
        market_open: bool = False,
        symbol_history: list | None = None,
        iv_rank: float = 50.0,
        market_regime: dict = None,
        streak_surcharge: float = 0.0,
    ) -> dict:
        await self._emit("status", f"Judge: evaluating {symbol} {direction}...")

        # ── Fatal flaws (pre-LLM hard reject) ────────────────────────────────
        for source, label in [
            (technical.get("fatal_flaw"),    "Technical"),
            (fundamental.get("fatal_flaw"),  "Fundamental"),
        ]:
            if source:
                msg = f"{label} fatal flaw: {source}"
                await self._emit("decision", {
                    "symbol": symbol, "decision": "pass", "weighted_score": 0,
                    "confidence": 0, "reasoning": msg, "pass_reason": msg, "strategy": "—",
                })
                return {"decision": "pass", "weighted_score": 0, "confidence": 0,
                        "reasoning": msg, "pass_reason": msg, "trade_proposal": None,
                        "bull_case": "", "bear_case": ""}

        # ── Deterministic score + IV-aware threshold ──────────────────────────
        now_et = datetime.now(_ET) if _ET else datetime.now()
        tod    = now_et.time()
        weighted_score = _compute_score(technical, fundamental, sentiment, risk)

        # ── Python-level feature adjustment (self-learned from scan_log) ──────
        # Queries historical win rates for the specific feature conditions present
        # in this setup (EMA200 structure, ADX zone, regime, agent consensus).
        # Adjusts the Python score by up to ±5 points — fully data-driven, no LLM.
        tech_s = float(technical.get("score", 5))
        fund_s = float(fundamental.get("score", 5))
        sent_s = float(sentiment.get("score", 5))
        consensus_n = sum([tech_s >= 7.0, fund_s >= 7.0, sent_s >= 7.0])
        feat_adj, feat_adj_reason = await asyncio.to_thread(
            ts.get_feature_score_adjustment,
            direction=direction,
            above_ema200=technical.get("above_ema200"),
            above_ema50=(technical.get("current_price", 0) > technical.get("ema50", 0))
                        if technical.get("ema50") else None,
            adx=technical.get("adx"),
            regime=(market_regime or {}).get("regime"),
            consensus_score=consensus_n,
        )
        adjusted_score = round(weighted_score + feat_adj, 1)

        # ── Overextension guard (deterministic) ───────────────────────────────
        # Live evidence: buying calls after a +151% 60-day run at RSI 67 (AMD)
        # entered a parabola top and hit the stop in 13 minutes. Chasing extreme
        # extension is penalized in the score itself, not just flagged.
        overext_penalty = 0.0
        m60   = technical.get("momentum_60d")
        rsi_j = technical.get("rsi", 50) or 50
        if direction == "bullish" and m60 is not None and m60 >= 80 and rsi_j >= 65:
            overext_penalty = round(3.0 + min(2.0, (m60 - 80) / 40.0), 1)
            adjusted_score  = round(adjusted_score - overext_penalty, 1)

        # Regime alignment for threshold
        _r_aligned = None
        _r_strength = 0
        if market_regime:
            _reg = market_regime.get("regime", "neutral")
            _r_strength = market_regime.get("strength", 0)
            if _reg == "bull" and direction == "bullish":
                _r_aligned = True
            elif _reg == "bear" and direction == "bearish":
                _r_aligned = True
            elif _reg in ("bull", "bear"):
                _r_aligned = False   # counter-trend
        threshold = score_threshold(iv_rank, market_open, time_of_day=tod,
                                    regime_aligned=_r_aligned, regime_strength=_r_strength,
                                    streak_surcharge=streak_surcharge)
        score_failed = adjusted_score < threshold

        # ── Strategy / HV context ─────────────────────────────────────────────
        vix_level  = sentiment.get("vix_level", 20)
        vix_regime = sentiment.get("vix_regime", "normal")
        hv_data    = await asyncio.to_thread(md.get_hv, symbol)
        hv_rank    = hv_data.get("hv_rank") or 50

        price = technical.get("current_price") or risk.get("current_price") or 100
        try:
            price = float(price)
        except (TypeError, ValueError):
            price = 100.0
        option_type = "call" if direction == "bullish" else "put"
        defaults    = trade_defaults()

        # ── Outcome context from training DB (PostgreSQL — survives restarts) ──
        similar, stats, sym_perf_map = await asyncio.gather(
            asyncio.to_thread(ts.get_similar_iv_stats, iv_rank, direction),
            asyncio.to_thread(ts.get_outcome_stats),
            asyncio.to_thread(ts.get_symbol_perf, min_trades=2),
        )
        sym_stats = sym_perf_map.get(symbol)

        # DB-level targeted pattern match for this specific setup
        try:
            tech_score_val = float(technical.get("score", 5))
            above_ema200_val = technical.get("above_ema200")
            adx_val = technical.get("adx")
            regime_str = (market_regime or {}).get("regime")
            db_similar = await asyncio.to_thread(
                ts.get_similar_trade_stats,
                symbol=symbol, direction=direction,
                tech_score=tech_score_val, above_ema200=above_ema200_val,
                adx=adx_val, regime=regime_str, min_samples=3,
            )
        except Exception:
            db_similar = None

        similar_block = ""
        if db_similar:
            similar_block += (
                f"\nDB MATCH — This exact setup: {db_similar['n']} trades | "
                f"win rate {db_similar['win_rate']:.0%} | avg P&L {db_similar['avg_pnl']:+.1f}%"
            )
        if sym_stats:
            similar_block += (
                f"\n{symbol} HISTORY ({sym_stats['trade_count']} trades): "
                f"win rate {sym_stats['win_rate']:.0%} | avg P&L {sym_stats['avg_pnl']:+.1f}%"
            )
        if similar:
            similar_block += (
                f"\nSIMILAR SETUPS (IV ±20, {direction}): "
                f"{similar['count']} trades | win rate {similar['win_rate']:.0%} "
                f"| avg P&L {similar['avg_pnl']:+.1f}%"
            )
        if not similar_block and stats.get("total_trades", 0) > 0:
            similar_block = (
                f"\nOVERALL HISTORY: {stats['total_trades']} closed trades | "
                f"win rate {stats.get('win_rate', 0):.0%} | "
                f"expectancy {stats.get('expectancy', 0):+.1f}%/trade"
            )

        # ── Symbol history block ──────────────────────────────────────────────
        hist_lines = []
        consistency_note = ""
        if symbol_history:
            for h in symbol_history[-5:]:
                hist_lines.append(
                    f"  Cycle {h['cycle']}: {h['direction']} "
                    f"decision={h['decision']} score={h['score']} tech={h['tech_score']}"
                )
            recent = symbol_history[-3:]
            if all(h.get("tech_score") and h["tech_score"] >= 6 and h["direction"] == direction
                   for h in recent):
                consistency_note = f"⚡ {direction.upper()} tech≥6 for 3 consecutive cycles — elevated conviction."

        # ── Risk flags (replaces advocate) ────────────────────────────────────
        risk_flags = []
        rsi = technical.get("rsi", 50)
        if direction == "bullish" and rsi > 70:
            risk_flags.append(f"RSI {rsi:.0f} — approaching overbought")
        if direction == "bearish" and rsi < 30:
            risk_flags.append(f"RSI {rsi:.0f} — approaching oversold")
        if vix_regime in ("elevated", "extreme"):
            risk_flags.append(f"VIX {vix_level:.0f} {vix_regime} — premium is more expensive, wider expected moves")
        sent_comp = sentiment.get("components", {})
        if not sent_comp.get("sector_aligned", True):
            risk_flags.append("Sector ETFs misaligned with trade direction")
        if fundamental.get("earnings_before_expiry") is False and fundamental.get("earnings_date"):
            pass  # clean — no flag needed
        if technical.get("vol_ratio", 1.0) < 0.7:
            risk_flags.append("Low volume — weak conviction")
        if overext_penalty:
            risk_flags.append(
                f"Overextended: +{m60:.0f}% in 60d at RSI {rsi_j:.0f} — "
                f"parabola-chase penalty −{overext_penalty} applied to score"
            )
        m1d = technical.get("momentum_1d", 0)
        if direction == "bullish" and m1d > 4.0:
            risk_flags.append(f"Stock up {m1d:+.1f}% today — overextended entry, options expensive")
        elif direction == "bullish" and m1d < -3.0:
            risk_flags.append(f"Stock down {m1d:.1f}% today — potential pullback entry opportunity")
        elif direction == "bearish" and m1d < -4.0:
            risk_flags.append(f"Stock down {m1d:.1f}% today — chasing drop, bounce risk")
        elif direction == "bearish" and m1d > 3.0:
            risk_flags.append(f"Stock up {m1d:+.1f}% today — potential reversal entry for puts")
        bb_pct = technical.get("bb_pct", 0.5)
        if direction == "bullish":
            if bb_pct > 1.0:
                pass   # BB breakout — positive signal, no flag
            elif 0.85 < bb_pct <= 1.0:
                risk_flags.append(f"Price extended near upper BB ({bb_pct:.2f}) — possible pullback")
            elif bb_pct < 0:
                risk_flags.append("Price below lower BB — momentum breakdown, consider put")
        elif direction == "bearish":
            if bb_pct < 0:
                pass   # BB breakdown — positive bear signal, no flag
            elif 0 <= bb_pct < 0.15:
                risk_flags.append(f"Price extended near lower BB ({bb_pct:.2f}) — bounce risk")
            elif bb_pct > 1.0:
                risk_flags.append("Price above upper BB — overbought, could aid put play")

        # ── Time of day ───────────────────────────────────────────────────────
        if   tod < dtime(9, 30):  tod_note = "Pre-open warm-up."
        elif tod < dtime(10, 30): tod_note = "Opening hour — momentum setups preferred."
        elif tod < dtime(12, 0):  tod_note = "Mid-morning."
        elif tod < dtime(14, 0):  tod_note = "Midday — raise bar for marginal setups."
        elif tod < dtime(15, 30): tod_note = "Afternoon — institutional activity picking up."
        else:                     tod_note = "Power hour — directional follow-through likely."
        if now_et.weekday() == 0:
            tod_note += " Monday — verify no gap vs scanner price."

        session = "LIVE MARKET" if market_open else "PRE-MARKET RESEARCH"

        # ── Build Judge context ───────────────────────────────────────────────
        # tech_s / fund_s / sent_s already defined above in feature adjustment section

        hist_block = ""
        if hist_lines:
            hist_block = "Prior cycles for this symbol:\n" + "\n".join(hist_lines)
            if consistency_note:
                hist_block += f"\n{consistency_note}"

        # ── Market regime block ───────────────────────────────────────────────
        regime_block = ""
        if market_regime:
            regime    = market_regime.get("regime", "neutral")
            reg_str   = market_regime.get("summary", "")
            vix_trend = market_regime.get("vix_trend", "flat")
            mismatch  = (regime == "bear" and direction == "bullish") or \
                        (regime == "bull" and direction == "bearish")
            align     = (regime == "bull" and direction == "bullish") or \
                        (regime == "bear" and direction == "bearish")
            regime_block = (
                f"\nMARKET REGIME: {regime.upper()} "
                f"({'aligned with trade ✅' if align else 'misaligned with trade ⚠️' if mismatch else 'neutral'})"
                f"\n  {reg_str}"
                f"\n  VIX trend: {vix_trend}"
                + (f"\n  ⚠️ Regime mismatch — directional trade against {regime} market" if mismatch else "")
            )

        # ── Self-learned calibration context ─────────────────────────────────
        learned_context = await asyncio.to_thread(ts.get_learned_context, min_samples=3)
        learned_block = ("SELF-LEARNED CALIBRATION (from PostgreSQL training log):\n" + learned_context) if learned_context else ""

        # ── Episodic memory: actual conditions of recent similar trades ────────
        episodic = await asyncio.to_thread(
            ts.get_episodic_context, direction=direction, symbol=symbol, limit=5)
        # Fall back to direction-only if symbol-specific is thin
        if not episodic:
            episodic = await asyncio.to_thread(
                ts.get_episodic_context, direction=direction, limit=5)
        episodic_block = episodic if episodic else ""

        strong_count = sum([tech_s >= 7.0, fund_s >= 7.0, sent_s >= 7.0])
        mid_count_j  = sum([6.0 <= tech_s < 7.0, 6.0 <= fund_s < 7.0, 6.0 <= sent_s < 7.0])
        weak_count_j = 3 - strong_count - mid_count_j
        raw_consensus = strong_count * 1.5 + mid_count_j * 0.5 - weak_count_j * 1.5
        capped = max(-3.0, min(3.0, raw_consensus))
        consensus_label = f"{capped:+.1f} ({strong_count}s/{mid_count_j}m/{weak_count_j}w)"
        feat_adj_line = (
            f"  + feature adjustment {feat_adj:+.1f} ({feat_adj_reason})\n  "
            if feat_adj != 0.0 and feat_adj_reason else "  "
        )
        if overext_penalty:
            feat_adj_line += f"− overextension penalty {overext_penalty} (60d momentum {m60:+.0f}%, RSI {rsi_j:.0f})\n  "
        context = f"""[{session} | Cycle {cycle} | {tod_note}]

TRADE: {symbol} {direction.upper()} {option_type.upper()} @ ${price:.2f}
IV Rank: {iv_rank:.0f}/100 — {iv_edge_label(iv_rank)}
VIX: {vix_level:.1f} ({vix_regime}) | HV Rank: {hv_rank:.0f}/100
{regime_block}

AGENT SCORES (Python-computed):
  Technical   {tech_s}/10: {technical.get('trend','?')} | {', '.join(technical.get('signals', [])[:3]) or technical.get('summary','')}
  Fundamental {fund_s}/10: {fundamental.get('summary','')}
  Sentiment   {sent_s}/10: {sentiment.get('summary','')}

COMPUTED SCORE: {adjusted_score} / threshold {threshold}
  tech({tech_s}×3={tech_s*3.0:.0f}) + fund({fund_s}×1.5={fund_s*1.5:.0f}) + sent({sent_s}×1.5={sent_s*1.5:.0f}) + risk(8×1=8)
  + consensus ({strong_count} strong/≥7, {mid_count_j} mid/6-7, {weak_count_j} weak) = {consensus_label}
  = base {weighted_score}
{feat_adj_line}= adjusted {adjusted_score}  |  IV threshold reason: {iv_edge_label(iv_rank)}
{"✅ SCORE PASSES" if not score_failed else f"❌ SCORE FAILS ({adjusted_score} < {threshold})"}

RISK FLAGS:
{chr(10).join(f'  • {f}' for f in risk_flags) if risk_flags else '  None identified'}
{similar_block}
{hist_block}
{episodic_block}
{learned_block}"""

        regime_mismatch = False
        regime_aligned  = False
        if market_regime:
            r = market_regime.get("regime", "neutral")
            regime_mismatch = (r == "bear" and direction == "bullish") or \
                              (r == "bull" and direction == "bearish")
            regime_aligned  = (r == "bull" and direction == "bullish") or \
                              (r == "bear" and direction == "bearish")
        regime_cap_note = (
            "\n⚠️ REGIME MISMATCH: Market regime opposes this trade. "
            "Cap your confidence at 7 maximum — only exceptional setups trade against regime."
            if regime_mismatch else
            "\n✅ REGIME ALIGNED: This trade aligns with overall market regime. "
            "Can go up to 10 if all other signals confirm."
            if regime_aligned else ""
        )

        conf_min_for_prompt = confidence_minimum(symbol)
        system = f"""You are the final judge for a self-improving Robinhood options trading system.
The Python score is {weighted_score} (threshold {threshold}).

{"SCORE PASSES. Give confidence 1-10. If you have genuine reservations, reflect in confidence. Trade happens only if confidence >= " + str(conf_min_for_prompt) + (" (raised for " + symbol + " due to high beta or poor history)" if conf_min_for_prompt > THRESHOLD_CONF else "") + "." if not score_failed else "SCORE FAILS. Confirm pass with a clear one-line reason."}
{regime_cap_note}

Confidence calibration:
  1-3: serious flaws — multiple agents disagree, regime strongly opposed, or historical edge clearly negative
  4:   notable reservations — use sparingly, only when you have a specific concrete reason this setup will fail
  5-6: reasonable setup, normal uncertainty — this is the DEFAULT for a passing score with no glaring issues
  7-8: clear directional setup, good risk/reward, agents aligned
  9-10: multiple strong signals confirming, IV environment ideal (only when regime aligned)

Anchoring rule: if the Python score PASSES and you have no specific, articulable reason this will fail, your baseline is 5-6 — not 4. "Market uncertainty" is not a reason to go to 4.

Rules:
  - Do NOT penalize for missing options data (IV, OI, bid-ask) — handled at execution
  - RSI alone in a ranging market is NOT a confidence killer if technical score reflects it
  - Low IV rank is a POSITIVE for confidence — cheaper premium means better risk/reward
  - Consider the risk flags but don't double-count what's already in the scores
  - Regime mismatch: cap at 7 — if you'd give 8+ you must justify why this stock bucks the trend
  - Regime aligned: strong setups can reach 9-10

IMPORTANT — HOW TO USE THE SELF-LEARNED CALIBRATION:
  The context below contains actual historical win rates from this system's past trades.
  Use these win rates to nudge (not override) your confidence — the current setup's technicals are the primary signal:
  - If this symbol/regime/direction historically wins < 35%: reduce confidence by at most 1 point
  - If this symbol historically wins > 65%: you may increase confidence by 1 point
  - If current conditions match a 'LOW WIN pattern': treat as a mild negative signal
  - If current conditions match a 'TOP WIN pattern': treat as a positive signal
  - History adjusts your view by ±1 max — do NOT let poor historical data alone push a technically strong setup (tech score ≥ 8) below confidence 5
  - If < 10 historical trades exist for this setup, treat the history as unreliable and weight it minimally

Respond ONLY with JSON:
{{
  "confidence": <int 1-10>,
  "reasoning": "<2-3 sentences explaining your conviction, referencing historical patterns if available>",
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

        # ── Final decision (Python only) ──────────────────────────────────────
        conf_min = confidence_minimum(symbol)

        if score_failed:
            decision    = "pass"
            pass_reason = pass_reason or f"Score {adjusted_score} below threshold {threshold}"
            trade_proposal = None
        elif confidence < conf_min:
            decision    = "pass"
            pass_reason = f"Confidence {confidence}/10 below minimum {conf_min} for {symbol}"
            trade_proposal = None
        elif confidence <= THRESHOLD_CONF and adjusted_score < threshold + MARGINAL_CONF_SCORE_CUSHION:
            # Bare-minimum confidence needs a score cushion — conf-5 trades with
            # scores hugging the threshold were the biggest live losers.
            decision    = "pass"
            pass_reason = (
                f"Marginal: confidence {confidence}/10 needs score ≥ "
                f"{threshold + MARGINAL_CONF_SCORE_CUSHION:.0f} (got {adjusted_score})"
            )
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
                "symbol":           symbol,
                "option_type":      option_type,
                "direction":        direction,
                "dte_min":          defaults["dte_min"],
                "dte_max":          defaults["dte_max"],
                "delta_target":     defaults["delta_target"],
                "profit_target_pct": defaults["profit_target_pct"],
                "stop_loss_pct":    defaults["stop_loss_pct"],
                "max_premium":      round(mp, 2),
                "total_max_loss":   int(mp * 100 * risk.get("contracts", 1)),
                "contracts":        risk.get("contracts", 1),
                "iv_rank":          iv_rank,
                "strategy":         "naked_option",
            }

        await self._emit("decision", {
            "symbol":         symbol,
            "decision":       decision,
            "weighted_score": adjusted_score,
            "threshold":      threshold,
            "iv_rank":        iv_rank,
            "confidence":     confidence,
            "reasoning":      reasoning,
            "pass_reason":    pass_reason,
            "strategy":       "naked_option" if decision == "trade" else "—",
        })

        return {
            "decision":       decision,
            "weighted_score": adjusted_score,
            "threshold":      threshold,
            "confidence":     confidence,
            "reasoning":      reasoning,
            "bull_case":      bull_case,
            "bear_case":      bear_case,
            "pass_reason":    pass_reason,
            "trade_proposal": trade_proposal,
        }
