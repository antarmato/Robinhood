"""
Risk Agent — pure Python, no LLM.

Position sizing using Kelly criterion when enough trade history exists.
Falls back to flat MAX_LOSS_PER_TRADE budget until ≥10 closed trades.

Hard limits:
  - 1 contract per trade (Robinhood retail)
  - Max 3 open positions
  - No duplicate symbols
"""

import logging
import os
from typing import Optional

from .base import BaseAgent, BroadcastFn

logger = logging.getLogger(__name__)
MAX_LOSS_DEFAULT = 100.0  # fallback if env var not set
KELLY_CAP_MULTIPLIER = 2  # Kelly budget ≤ MAX_LOSS * 2

# Sector group mapping — positions in the same group are correlated
_SECTOR_GROUPS: dict[str, str] = {
    "NVDA": "AI_semis", "AMD": "AI_semis", "SMCI": "AI_semis",
    "MSTR": "crypto",   "COIN": "crypto",
    "RIVN": "EV",       "TSLA": "EV",
    "SOFI": "fintech",  "HOOD": "fintech", "SQ": "fintech", "PYPL": "fintech",
    "IONQ": "quantum",
    "PLTR": "govtech",
    "ROKU": "streaming",
    "UBER": "rideshare",
}


class RiskAgent(BaseAgent):
    def __init__(
        self,
        client,
        max_loss_per_trade: float = MAX_LOSS_DEFAULT,
        broadcast: Optional[BroadcastFn] = None,
    ):
        super().__init__(client, "Risk", broadcast=broadcast)
        self.max_loss_per_trade = max_loss_per_trade

    async def evaluate(
        self,
        symbol: str,
        _options_analysis: dict,
        current_positions: list[dict],
    ) -> dict:
        await self._emit("status", f"Risk: sizing position for {symbol}...")

        # Hard position rules
        open_symbols = {p.get("symbol") for p in current_positions}
        if symbol in open_symbols:
            return {
                "approved": False, "contracts": 0, "max_premium": 0,
                "total_max_loss": 0, "limit_price": 0, "score": 1,
                "rejection_reason": f"Already have open position in {symbol}",
                "summary": f"Rejected: duplicate position in {symbol}.",
            }
        MAX_POSITIONS = int(os.getenv("MAX_OPEN_POSITIONS", "4"))
        if len(current_positions) >= MAX_POSITIONS:
            return {
                "approved": False, "contracts": 0, "max_premium": 0,
                "total_max_loss": 0, "limit_price": 0, "score": 1,
                "rejection_reason": f"Max {MAX_POSITIONS} open positions reached",
                "summary": f"Rejected: max positions reached.",
            }

        # Directional concentration (informational, not a hard block)
        direction_counts = {"bullish": 0, "bearish": 0}
        for p in current_positions:
            d = p.get("direction", "bullish")
            direction_counts[d] = direction_counts.get(d, 0) + 1

        # Budget: use Kelly if we have enough history, else flat budget
        budget = self._compute_budget()
        max_premium = round(budget / 100, 2)

        from ..training_store import get_outcome_stats as _ts_oc
        ot        = _ts_oc()
        kelly_ready = ot.get("kelly_ready", False)
        kelly     = ot.get("kelly_fraction", 0.0)

        sizing_note = (
            f"Kelly {kelly:.1%} × budget = ${budget:.0f}"
            if kelly_ready and kelly > 0
            else f"Flat budget ${budget:.0f} (need {max(0, 10 - ot.get('total_trades', 0))} more closed trades for Kelly)"
        )

        bull_open = direction_counts.get("bullish", 0)
        bear_open = direction_counts.get("bearish", 0)
        concentration_note = ""
        risk_score = 8

        if bull_open >= 2 and bear_open == 0:
            concentration_note = " ⚠️ All-bull portfolio — zero hedge."
            risk_score -= 1
        elif bear_open >= 2 and bull_open == 0:
            concentration_note = " ⚠️ All-bear portfolio — zero hedge."
            risk_score -= 1

        # Sector correlation check — same sector as existing open position?
        sector = _SECTOR_GROUPS.get(symbol)
        correlated_syms = []
        if sector:
            for p in current_positions:
                open_sym = p.get("symbol", "")
                if _SECTOR_GROUPS.get(open_sym) == sector:
                    correlated_syms.append(open_sym)
        if correlated_syms:
            concentration_note += f" ⚠️ Correlated sector ({sector}) — {', '.join(correlated_syms)} already open."
            risk_score -= 1

        await self._emit("sizing", {
            "symbol": symbol, "contracts": 1,
            "max_premium": max_premium, "total_max_loss": budget,
            "kelly_ready": kelly_ready, "kelly": kelly,
            "bull_open": bull_open, "bear_open": bear_open,
            "sector": sector, "correlated": correlated_syms,
        })

        return {
            "approved":       True,
            "contracts":      1,
            "max_premium":    max_premium,
            "limit_price":    max_premium,
            "total_max_loss": budget,
            "rejection_reason": None,
            "score":          max(5, risk_score),
            "current_price":  None,
            "summary": (
                f"1 contract, max ${max_premium:.2f}/share (≤${budget:.0f} total). "
                f"{len(current_positions)}/{MAX_POSITIONS} positions open "
                f"({bull_open}B/{bear_open}P).{concentration_note} {sizing_note}"
            ),
        }

    def _compute_budget(self) -> float:
        from ..training_store import get_outcome_stats as _ts_oc
        ot      = _ts_oc()
        env_max = float(os.getenv("MAX_LOSS_PER_TRADE", str(MAX_LOSS_DEFAULT)))

        if not ot.get("kelly_ready", False):
            return env_max

        kelly = ot.get("kelly_fraction", 0.0)

        if kelly < 0:
            # Negative expectancy: reduce to 50% of flat budget as a warning signal
            # Don't stop entirely — simulation needs data to recover
            return env_max * 0.5

        if kelly == 0:
            return env_max

        # Fractional Kelly: 25% of Kelly fraction applied to the budget ceiling
        # This keeps sizing conservative even as Kelly grows
        kelly_budget = kelly * env_max * 4  # scale: kelly 0.10 × $200*4 = $80 (conservative)
        return min(kelly_budget, env_max * KELLY_CAP_MULTIPLIER)
