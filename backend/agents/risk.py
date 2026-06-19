"""
Risk Agent — evaluates position sizing and enforces hard limits.
For single-leg options: max loss = premium paid. Simple, defined risk.
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
        options_analysis: dict,
        current_positions: list[dict],
    ) -> dict:
        await self._emit("status", f"Evaluating trade risk for {symbol}...")

        premium = options_analysis.get("estimated_premium", 0) or 0
        max_contracts = int(self.max_loss_per_trade / (premium * 100)) if premium > 0 else 0
        contracts = max(1, min(max_contracts, 5))
        total_cost = contracts * premium * 100

        context = f"""Risk evaluation for {symbol} single-leg option

Proposed trade:
  Option type:    {options_analysis.get('option_type', 'N/A')}
  Strike:         {options_analysis.get('strike', 'N/A')}
  Expiry:         {options_analysis.get('expiration_date', 'N/A')} ({options_analysis.get('dte', '?')} DTE)
  Premium (mid):  ${premium:.2f}/share
  Contracts:      {contracts}
  Total cost:     ${total_cost:.2f} (max loss)

Risk limits:
  Max loss per trade: ${self.max_loss_per_trade:.0f}
  Max open positions: 3

Current open positions: {len(current_positions)}
Open symbols: {[p.get('symbol') for p in current_positions]}"""

        # Hard rules — no LLM needed
        open_symbols = {p.get("symbol") for p in current_positions}
        if symbol in open_symbols:
            return {
                "approved": False, "contracts": 0,
                "rejection_reason": f"Already have an open position in {symbol}",
                "score": 1, "summary": f"Rejected: duplicate position in {symbol}.",
                "total_cost": 0,
            }
        if len(current_positions) >= 3:
            return {
                "approved": False, "contracts": 0,
                "rejection_reason": "Max 3 open positions reached",
                "score": 1, "summary": "Rejected: max positions reached.",
                "total_cost": 0,
            }
        if premium <= 0:
            return {
                "approved": False, "contracts": 0,
                "rejection_reason": "Could not determine option premium",
                "score": 1, "summary": "Rejected: no valid premium.",
                "total_cost": 0,
            }
        if total_cost > self.max_loss_per_trade:
            return {
                "approved": False, "contracts": 0,
                "rejection_reason": f"Cost ${total_cost:.0f} exceeds max loss ${self.max_loss_per_trade:.0f}",
                "score": 2, "summary": f"Rejected: premium too high for {contracts} contracts.",
                "total_cost": 0,
            }

        return {
            "approved": True,
            "contracts": contracts,
            "limit_price": round(premium * 1.02, 2),  # 2% above mid for fill
            "total_cost": total_cost,
            "rejection_reason": None,
            "score": 8,
            "summary": f"{contracts} contract(s) at ${premium:.2f} = ${total_cost:.2f} max loss (within ${self.max_loss_per_trade:.0f} limit).",
        }
