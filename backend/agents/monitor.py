"""
Monitor Agent — pure Python position management.

Exit rules (in priority order):
  1. Profit target hit (default 50%) → exit
  2. Trailing stop: once gain hits +25%, floor moves to breakeven (0%)
     Once gain hits +40%, trail at high_water - 20% (lock in partial gains)
  3. Hard stop loss hit (default -50%) → exit
  4. Time decay: DTE ≤ 14 AND pnl < -10% → exit (theta accelerates sharply)
  5. DTE ≤ 2 → always exit (avoid expiry risk)
  6. Otherwise: hold

No LLM call — pure Python rules.
"""

import logging
from datetime import date, datetime
from typing import Optional

import anthropic

from .base import BaseAgent, BroadcastFn
from .. import market_data as md

logger = logging.getLogger(__name__)

TRAILING_ACTIVATION_1 = 25.0   # once gain hits this %, floor → breakeven
TRAILING_ACTIVATION_2 = 40.0   # once gain hits this %, trail at peak - 20%
TRAILING_TRAIL_2      = 20.0   # max give-back after +40% peak
THETA_EXIT_DTE        = 14     # time-based exit threshold
THETA_EXIT_LOSS       = -10.0  # only cut on theta if also losing this %


class MonitorAgent(BaseAgent):
    def __init__(self, client: anthropic.AsyncAnthropic, broadcast: Optional[BroadcastFn] = None):
        super().__init__(client, "Monitor", broadcast=broadcast)

    async def check_positions(self, active_trades: list[dict]) -> list[dict]:
        if not active_trades:
            await self._emit("status", "No open positions to monitor.")
            return []
        await self._emit("status", f"Monitoring {len(active_trades)} open position(s)...")
        return [await self._evaluate(t) for t in active_trades]

    async def _evaluate(self, trade: dict) -> dict:
        symbol        = trade.get("symbol", "")
        trade_id      = trade.get("trade_id", "")
        entry_price   = float(trade.get("limit_price") or trade.get("max_premium") or 0)
        contracts     = int(trade.get("contracts", 1))
        profit_target = float(trade.get("profit_target_pct", 50))
        stop_loss     = float(trade.get("stop_loss_pct", 50))
        expiry        = trade.get("expiration_date", "")
        strike        = trade.get("strike")
        option_type   = trade.get("option_type", "call")
        high_water    = float(trade.get("high_water_pct", 0.0))

        current_mid = self._get_current_price(symbol, expiry, strike, option_type)
        if current_mid is None:
            return self._hold(trade_id, symbol, "Could not fetch option price — holding",
                              0.0, 0.0, None, high_water)

        pnl_pct   = (current_mid - entry_price) / entry_price * 100 if entry_price else 0.0
        pnl_total = (current_mid - entry_price) * contracts * 100
        new_high  = max(high_water, pnl_pct)

        try:
            dte = (datetime.strptime(expiry, "%Y-%m-%d").date() - date.today()).days
        except Exception:
            dte = 999

        # ── 1. Profit target ─────────────────────────────────────────────────
        if pnl_pct >= profit_target:
            return self._exit(trade_id, symbol,
                f"Profit target +{pnl_pct:.1f}% (target {profit_target:.0f}%)",
                pnl_pct, pnl_total, current_mid, new_high)

        # ── 2. Trailing stop ──────────────────────────────────────────────────
        effective_floor = -stop_loss
        trail_note      = ""
        if new_high >= TRAILING_ACTIVATION_2:
            effective_floor = new_high - TRAILING_TRAIL_2
            trail_note = f"trailing stop (peak {new_high:.0f}% − {TRAILING_TRAIL_2:.0f}% = floor {effective_floor:+.0f}%)"
        elif new_high >= TRAILING_ACTIVATION_1:
            effective_floor = 0.0
            trail_note = f"trailing stop (peak {new_high:.0f}% → floor at breakeven)"

        if pnl_pct <= effective_floor:
            prefix = "Trailing stop" if trail_note else "Stop loss"
            return self._exit(trade_id, symbol,
                f"{prefix}: P&L {pnl_pct:+.1f}% hit floor {effective_floor:+.0f}%"
                + (f" | {trail_note}" if trail_note else ""),
                pnl_pct, pnl_total, current_mid, new_high)

        # ── 3. Theta time-based exit ──────────────────────────────────────────
        if dte <= THETA_EXIT_DTE and pnl_pct < THETA_EXIT_LOSS:
            return self._exit(trade_id, symbol,
                f"Theta exit: {dte} DTE remaining & P&L {pnl_pct:+.1f}% (accelerating decay)",
                pnl_pct, pnl_total, current_mid, new_high)

        # ── 4. Expiry floor ───────────────────────────────────────────────────
        if dte <= 2:
            return self._exit(trade_id, symbol,
                f"{dte} DTE — closing to avoid expiry risk",
                pnl_pct, pnl_total, current_mid, new_high)

        # ── Hold ──────────────────────────────────────────────────────────────
        trail_str = f" | {trail_note}" if trail_note else ""
        await self._emit("status",
            f"{symbol} {option_type.upper()} ${strike} "
            f"| P&L: {pnl_pct:+.1f}% (peak {new_high:+.1f}%) "
            f"| DTE: {dte}{trail_str} | HOLD")
        return self._hold(trade_id, symbol,
            f"P&L {pnl_pct:+.1f}% (peak {new_high:+.1f}%) | DTE {dte} | within targets",
            pnl_pct, pnl_total, current_mid, new_high)

    # ── Helpers ───────────────────────────────────────────────────────────────

    @staticmethod
    def _exit(trade_id, symbol, reason, pnl_pct, pnl_total, price, new_high):
        return {"trade_id": trade_id, "symbol": symbol, "action": "exit",
                "reason": reason, "pnl_pct": round(pnl_pct, 2),
                "pnl_total": round(pnl_total, 2), "current_price": price,
                "new_high_water": round(new_high, 2)}

    @staticmethod
    def _hold(trade_id, symbol, reason, pnl_pct, pnl_total, price, new_high):
        return {"trade_id": trade_id, "symbol": symbol, "action": "hold",
                "reason": reason, "pnl_pct": round(pnl_pct, 2),
                "pnl_total": round(pnl_total, 2), "current_price": price,
                "new_high_water": round(new_high, 2)}

    def _get_current_price(
        self, symbol: str, expiry: str, strike: float, option_type: str
    ) -> Optional[float]:
        try:
            chain = md.get_options_chain(symbol, expiry, option_type)
            if not chain:
                return None
            match = min(chain, key=lambda x: abs(float(x.get("strike_price", 0)) - float(strike or 0)))
            if abs(float(match.get("strike_price", 0)) - float(strike or 0)) > 1.5:
                return None
            bid = float(match.get("bid", 0))
            ask = float(match.get("ask", 0))
            if bid and ask:
                return round((bid + ask) / 2, 2)
            last = match.get("last") or match.get("last_trade_price")
            return float(last) if last else None
        except Exception as e:
            logger.error(f"_get_current_price error for {symbol}: {e}")
            return None
