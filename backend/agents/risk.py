"""
Risk Agent — position sizing and hard limits.

Simplified: no longer needs live options premium.
Budget logic: 1 contract, max_premium = max_loss / 100.
Cowork selects the actual strike/expiry at execution time.
"""

import logging
from typing import Optional

import anthropic

from .base import BaseAgent, BroadcastFn

logger = logging.getLogger(__name__)


class RiskAgent(BaseAgent):
    def __init__(
        self,
        client: anthropic.AsyncAnthropic,
        max_loss_per_trade: float = 200.0,
        broadcast: Optional[BroadcastFn] = None,
    ):
        super().__init__(client, "Risk", model="claude-haiku-4-5-20251001", broadcast=broadcast)
        self.max_loss_per_trade = max_loss_per_trade

    async def evaluate(
        self,
        symbol: str,
        _options_analysis: dict,   # kept for signature compatibility, not used
        current_positions: list[dict],
    ) -> dict:
        await self._emit("status", f"Evaluating trade risk for {symbol}...")

        # Hard rules — no LLM needed
        open_symbols = {p.get("symbol") for p in current_positions}
        if symbol in open_symbols:
            return {
                "approved": False, "contracts": 0, "max_premium": 0,
                "total_max_loss": 0, "limit_price": 0,
                "rejection_reason": f"Already have an open position in {symbol}",
                "score": 1, "summary": f"Rejected: duplicate position in {symbol}.",
            }
        if len(current_positions) >= 3:
            return {
                "approved": False, "contracts": 0, "max_premium": 0,
                "total_max_loss": 0, "limit_price": 0,
                "rejection_reason": "Max 3 open positions reached",
                "score": 1, "summary": "Rejected: max positions reached.",
            }

        # Budget: 1 contract, max_premium is how much per share we can pay
        max_premium = round(self.max_loss_per_trade / 100, 2)

        await self._emit("sizing", {
            "symbol": symbol, "contracts": 1,
            "max_premium": max_premium,
            "total_max_loss": self.max_loss_per_trade,
        })

        return {
            "approved": True,
            "contracts": 1,
            "max_premium": max_premium,
            "limit_price": max_premium,         # Cowork will use this as order limit
            "total_max_loss": self.max_loss_per_trade,
            "rejection_reason": None,
            "score": 8,
            "summary": (
                f"1 contract, max ${max_premium:.2f}/share premium "
                f"(≤${self.max_loss_per_trade:.0f} total risk). "
                f"{len(current_positions)}/3 positions open."
            ),
        }
