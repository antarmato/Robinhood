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
from ..outcome_tracker import get_outcome_tracker

logger = logging.getLogger(__name__)
MAX_LOSS_DEFAULT = 200.0  # fallback if env var not set
KELLY_CAP_MULTIPLIER = 2  # Kelly budget ≤ MAX_LOSS * 2


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
        if len(current_positions) >= 3:
            return {
                "approved": False, "contracts": 0, "max_premium": 0,
                "total_max_loss": 0, "limit_price": 0, "score": 1,
                "rejection_reason": "Max 3 open positions reached",
                "summary": "Rejected: max positions reached.",
            }

        # Budget: use Kelly if we have enough history, else flat budget
        budget = self._compute_budget()
        max_premium = round(budget / 100, 2)

        tracker = get_outcome_tracker()
        kelly_ready = tracker.is_kelly_ready()
        kelly = tracker.get_kelly_fraction()
        stats = tracker.get_stats()

        sizing_note = (
            f"Kelly {kelly:.1%} × budget = ${budget:.0f}"
            if kelly_ready and kelly > 0
            else f"Flat budget ${budget:.0f} (need {10 - stats.get('total_trades', 0)} more closed trades for Kelly)"
        )

        await self._emit("sizing", {
            "symbol": symbol, "contracts": 1,
            "max_premium": max_premium, "total_max_loss": budget,
            "kelly_ready": kelly_ready, "kelly": kelly,
        })

        return {
            "approved":       True,
            "contracts":      1,
            "max_premium":    max_premium,
            "limit_price":    max_premium,
            "total_max_loss": budget,
            "rejection_reason": None,
            "score":          8,
            "summary": (
                f"1 contract, max ${max_premium:.2f}/share (≤${budget:.0f} total). "
                f"{len(current_positions)}/3 positions open. {sizing_note}"
            ),
        }

    def _compute_budget(self) -> float:
        tracker = get_outcome_tracker()
        env_max = float(os.getenv("MAX_LOSS_PER_TRADE", str(MAX_LOSS_DEFAULT)))

        if not tracker.is_kelly_ready():
            return env_max

        kelly = tracker.get_kelly_fraction()
        if kelly <= 0:
            return env_max

        # Fractional Kelly: 25% of Kelly fraction applied to the budget ceiling
        # This keeps sizing conservative even as Kelly grows
        kelly_budget = kelly * env_max * 4  # scale: kelly 0.10 × $200*4 = $80 (conservative)
        return min(kelly_budget, env_max * KELLY_CAP_MULTIPLIER)
