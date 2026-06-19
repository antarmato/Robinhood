"""
Monitor Agent — checks open positions against profit target and stop loss.
Uses yfinance to get current option prices.
"""

import logging
from datetime import date, datetime
from typing import Optional

import anthropic

from .base import BaseAgent, BroadcastFn
from .. import market_data as md

logger = logging.getLogger(__name__)


class MonitorAgent(BaseAgent):
    def __init__(self, client: anthropic.AsyncAnthropic, broadcast: Optional[BroadcastFn] = None):
        super().__init__(client, "Monitor", model="claude-haiku-4-5-20251001", broadcast=broadcast)

    async def check_positions(self, active_trades: list[dict]) -> list[dict]:
        if not active_trades:
            await self._emit("status", "No open positions to monitor.")
            return []
        await self._emit("status", f"Monitoring {len(active_trades)} open position(s)...")
        return [await self._evaluate(t) for t in active_trades]

    async def _evaluate(self, trade: dict) -> dict:
        symbol        = trade.get("symbol", "")
        trade_id      = trade.get("trade_id", "")
        entry_price   = trade.get("limit_price", 0)          # what we paid per share
        contracts     = trade.get("contracts", 1)
        profit_target = trade.get("profit_target_pct", 50)
        stop_loss     = trade.get("stop_loss_pct", 50)
        expiry        = trade.get("expiration_date", "")
        strike        = trade.get("strike")
        option_type   = trade.get("option_type", "call")

        # Get current option mid price from yfinance
        current_mid = self._get_current_price(symbol, expiry, strike, option_type)

        # DTE
        try:
            dte = (datetime.strptime(expiry, "%Y-%m-%d").date() - date.today()).days
        except Exception:
            dte = 999

        if current_mid is None:
            return {"trade_id": trade_id, "symbol": symbol, "action": "hold",
                    "reason": "Could not fetch current option price"}

        pnl_pct   = (current_mid - entry_price) / entry_price * 100 if entry_price else 0
        pnl_total = (current_mid - entry_price) * contracts * 100

        # Hard rules — no Claude needed
        if pnl_pct >= profit_target:
            return {"trade_id": trade_id, "symbol": symbol, "action": "exit",
                    "reason": f"Profit target hit: +{pnl_pct:.1f}%",
                    "pnl_pct": pnl_pct, "pnl_total": pnl_total,
                    "current_price": current_mid}
        if pnl_pct <= -stop_loss:
            return {"trade_id": trade_id, "symbol": symbol, "action": "exit",
                    "reason": f"Stop loss hit: {pnl_pct:.1f}%",
                    "pnl_pct": pnl_pct, "pnl_total": pnl_total,
                    "current_price": current_mid}
        if dte <= 2:
            return {"trade_id": trade_id, "symbol": symbol, "action": "exit",
                    "reason": f"Only {dte} DTE — closing to avoid expiry risk",
                    "pnl_pct": pnl_pct, "pnl_total": pnl_total,
                    "current_price": current_mid}

        await self._emit("status",
            f"{symbol} {option_type} ${strike} | P&L: {pnl_pct:+.1f}% | DTE: {dte} | HOLD")
        return {"trade_id": trade_id, "symbol": symbol, "action": "hold",
                "reason": f"P&L: {pnl_pct:+.1f}% — within targets",
                "pnl_pct": pnl_pct, "pnl_total": pnl_total,
                "current_price": current_mid}

    def _get_current_price(
        self, symbol: str, expiry: str, strike: float, option_type: str
    ) -> Optional[float]:
        try:
            chain = md.get_options_chain(symbol, expiry, option_type)
            match = min(chain, key=lambda x: abs(x["strike_price"] - strike))
            if abs(match["strike_price"] - strike) > 1.0:
                return None
            bid, ask = match.get("bid", 0), match.get("ask", 0)
            if bid and ask:
                return (bid + ask) / 2
            return match.get("last") or None
        except Exception as e:
            logger.error(f"_get_current_price error: {e}")
            return None
