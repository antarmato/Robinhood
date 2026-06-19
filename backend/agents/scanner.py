"""
Scanner Agent — finds the best 3-5 option trade candidates from the watchlist.
Uses yfinance market data (no credentials required).
"""

import logging
from typing import Optional

import anthropic

from .base import BaseAgent, BroadcastFn
from .. import market_data as md

logger = logging.getLogger(__name__)


class ScannerAgent(BaseAgent):
    def __init__(
        self,
        client: anthropic.AsyncAnthropic,
        watchlist: list[str],
        broadcast: Optional[BroadcastFn] = None,
    ):
        super().__init__(client, "Scanner", model="claude-haiku-4-5-20251001", broadcast=broadcast)
        self.watchlist = watchlist

    async def scan(self) -> list[dict]:
        await self._emit("status", "Scanning watchlist for opportunities...")
        market_snapshot = self._gather_data()
        if not market_snapshot:
            await self._emit("status", "No market data available.")
            return []

        summary = self._build_summary(market_snapshot)

        system = """You are a quantitative options trader scanning for the best single-leg option trade candidates.
You are looking for stocks with strong directional setups where buying a call or put makes sense.

Ideal candidates:
- Clear directional momentum (up or down)
- IV rank 20-60 (not too expensive to buy premium)
- Above-average volume (confirms the move)
- Clear near-term catalyst or technical breakout

Return a JSON array of the top 3-5 candidates, ranked best to worst:
[
  {
    "symbol": "SPY",
    "direction": "bullish" or "bearish",
    "option_type": "call" or "put",
    "signal_strength": 1-10,
    "key_reason": "one concise sentence",
    "priority": 1
  }
]

Only respond with valid JSON. No text outside the JSON array."""

        messages = [{"role": "user", "content": f"Market data snapshot:\n\n{summary}"}]
        raw = await self._call(system, messages, max_tokens=800, stream=False)
        candidates = self._parse_json(raw)
        if not isinstance(candidates, list):
            candidates = []

        for c in candidates:
            sym = c.get("symbol", "")
            if sym in market_snapshot:
                c["current_price"] = market_snapshot[sym].get("price", 0)
                c["iv_rank"] = market_snapshot[sym].get("iv_rank")

        await self._emit("status", f"Found {len(candidates)} candidates: {[c.get('symbol') for c in candidates]}")
        return candidates

    def _gather_data(self) -> dict:
        result = {}
        for sym in self.watchlist:
            try:
                q = md.get_quote(sym)
                price = q.get("price", 0)
                if not price:
                    continue
                result[sym] = {
                    "price": price,
                    "pct_change": q.get("pct_change", 0),
                    "volume_ratio": md.get_volume_ratio(sym),
                    "iv_rank": md.get_iv_rank(sym),
                }
            except Exception as e:
                logger.debug(f"Data error for {sym}: {e}")
        return result

    def _build_summary(self, data: dict) -> str:
        lines = ["Symbol | Price   | Day%   | Vol Ratio | IV Rank"]
        lines.append("-" * 55)
        for sym, d in data.items():
            iv = f"{d['iv_rank']:.0f}" if d.get("iv_rank") is not None else "N/A"
            lines.append(
                f"{sym:6} | ${d['price']:7.2f} | {d['pct_change']:+5.1f}% "
                f"| {d['volume_ratio']:.1f}x      | {iv}"
            )
        return "\n".join(lines)
