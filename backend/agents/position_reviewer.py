"""
PositionReviewer — LLM thesis re-evaluation for open positions.

Runs once per monitor cycle (every 15 min) per open position (after day 1).
Gathers fresh technical data + recent news, then asks Claude:
  "Does the original entry thesis still hold?"

Returns one of:
  hold         — thesis intact, keep the position
  exit         — thesis broken, close now (don't wait for the hard stop)
  tighten_stop — thesis weakening, raise the stop floor to lock in more

Uses claude-haiku for speed and cost (recurring task, structured output).
"""

import asyncio
import logging

import anthropic

from .base import BaseAgent, BroadcastFn
from .technical import TechnicalAgent
from .. import market_data as md

logger = logging.getLogger(__name__)


class PositionReviewer(BaseAgent):
    def __init__(
        self,
        client: anthropic.AsyncAnthropic,
        broadcast: BroadcastFn | None = None,
    ):
        super().__init__(client, "Monitor", model="claude-haiku-4-5-20251001", broadcast=broadcast)

    async def review(self, position: dict, current_regime: dict) -> dict:
        """
        Re-evaluate an open position against current conditions.

        Returns:
            {
              "action": "hold" | "exit" | "tighten_stop",
              "reason": str,
              "tighter_floor": float | None   # new stop floor % if tighten_stop
            }
        """
        symbol    = position.get("symbol", "")
        direction = position.get("direction", "bullish")

        # Fresh technical snapshot — pure Python, no LLM cost
        try:
            tech = await TechnicalAgent(self.client, broadcast=None).analyze(symbol, direction)
        except Exception as e:
            logger.warning(f"PositionReviewer tech failed for {symbol}: {e}")
            tech = {}

        # Recent news headlines — API call only
        loop = asyncio.get_event_loop()
        try:
            news_data  = await loop.run_in_executor(None, lambda: md.get_news_sentiment(symbol))
            headlines  = (news_data.get("headlines") or [])[:5]
            news_score = float(news_data.get("score", 0.0))
        except Exception:
            headlines  = []
            news_score = 0.0

        return await self._llm_review(position, tech, headlines, news_score, current_regime)

    async def _llm_review(
        self,
        pos: dict,
        tech: dict,
        headlines: list,
        news_score: float,
        current_regime: dict,
    ) -> dict:
        symbol        = pos.get("symbol", "")
        direction     = pos.get("direction", "bullish")
        opt_type      = pos.get("option_type", "call").upper()
        pnl_pct       = float(pos.get("last_pnl_pct", 0))
        high_water    = float(pos.get("high_water_pnl_pct", 0))
        days_held     = int(pos.get("days_held", 0))
        dte_left      = int(pos.get("dte_left", 35))
        entry_price   = float(pos.get("entry_stock_price", 0))
        current_price = float(pos.get("last_stock_price", 0))
        price_move    = (current_price - entry_price) / max(entry_price, 0.01) * 100

        entry_regime  = pos.get("entry_regime", "neutral")
        current_reg   = (current_regime or {}).get("regime", "unknown")
        regime_note   = "  ← REGIME CHANGED" if (entry_regime != current_reg and current_reg not in ("unknown", "")) else ""

        news_lines = "\n".join(f"  • {h}" for h in headlines) if headlines else "  (no recent news)"
        tech_signals = "\n".join(f"  • {s}" for s in (tech.get("signals") or [])[:6]) or "  (no signals)"
        fatal_flaw = tech.get("fatal_flaw")
        flaw_line  = f"\n  ⚠ FATAL FLAW: {fatal_flaw}" if fatal_flaw else ""

        system = (
            "You are a disciplined options trader reviewing an open position to decide "
            "whether to hold, exit early, or tighten the stop. Focus on whether the "
            "ORIGINAL ENTRY THESIS still holds — not just the P&L number. "
            "Be concise and decisive. Respond ONLY with valid JSON."
        )

        user = f"""OPEN POSITION REVIEW: {symbol} {direction.upper()} {opt_type}

ENTRY CONTEXT:
  Entry price: ${entry_price:.2f}  |  Current: ${current_price:.2f}  (move: {price_move:+.1f}%)
  Days held: {days_held}d  |  DTE remaining: {dte_left}
  P&L: {pnl_pct:+.1f}%  (peak: {high_water:+.1f}%)

ORIGINAL THESIS (why this trade was entered):
  Bull case: {pos.get("bull_case") or "N/A"}
  Bear case: {pos.get("bear_case") or "N/A"}
  Reasoning: {pos.get("reasoning") or "N/A"}
  Entry scores — Tech: {pos.get("tech_score", "?")}/10  Fund: {pos.get("fund_score", "?")}/10  Sent: {pos.get("sent_score", "?")}/10
  Entry regime: {entry_regime}  |  RSI at entry: {pos.get("entry_rsi") or "?"}  |  ADX at entry: {pos.get("entry_adx") or "?"}
  Above EMA200 at entry: {pos.get("entry_above_ema200", "?")}

CURRENT TECHNICAL PICTURE:
  Trend: {tech.get("trend", "unknown")}  |  RSI: {tech.get("rsi", "?")}  |  ADX: {tech.get("adx", "?")}
  Above EMA200: {tech.get("above_ema200", "?")}  |  EMA20 slope: {tech.get("ema20_slope", "?")}%/wk
  MACD: {tech.get("macd_reading", "?")}  |  Tech score now: {tech.get("score", "?")}/10{flaw_line}
  Signals:
{tech_signals}

REGIME:
  At entry: {entry_regime}  |  Now: {current_reg}{regime_note}

RECENT NEWS:
{news_lines}
  News sentiment: {news_score:+.2f}

DECISION CRITERIA:
  EXIT if any of these are true:
    - Stock has moved decisively against the trade direction (e.g. bullish entry, now in confirmed downtrend)
    - News/earnings fundamentally changed the fundamental picture
    - Regime flipped against position direction and tech confirms the change
    - Position has drifted the wrong way for 5+ days with zero recovery
    - A fatal technical flaw now exists that didn't at entry

  TIGHTEN_STOP if:
    - Thesis is weakening but not broken (e.g. mixed signals, one bad signal appeared)
    - Want to protect remaining capital without closing yet
    - Suggest a "tighter_floor" value (e.g. -15.0 means stop at -15%)

  HOLD if:
    - Thesis intact, just normal volatility
    - News is neutral or supportive
    - Technical picture consistent with entry analysis

Respond ONLY with this JSON (no explanation outside the JSON):
{{
  "action": "hold" | "exit" | "tighten_stop",
  "reason": "one sentence explaining the decision",
  "tighter_floor": null
}}"""

        try:
            raw    = await self._call(system, [{"role": "user", "content": user}],
                                      max_tokens=256, stream=False)
            result = self._parse_json(raw)
            action = result.get("action", "hold")
            if action not in ("hold", "exit", "tighten_stop"):
                action = "hold"
            tighter = result.get("tighter_floor")
            if tighter is not None:
                try:
                    tighter = float(tighter)
                except (TypeError, ValueError):
                    tighter = None
            return {
                "action":        action,
                "reason":        str(result.get("reason", "No reason provided")),
                "tighter_floor": tighter,
            }
        except Exception as e:
            logger.warning(f"PositionReviewer LLM call failed for {symbol}: {e}")
            return {"action": "hold", "reason": f"Review unavailable: {e}", "tighter_floor": None}
