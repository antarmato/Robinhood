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

        await self._emit("status", f"Data gathered for {len(market_snapshot)} symbols.")
        summary = self._build_summary(market_snapshot)

        system = """You are a quantitative options trader scanning for directional option trade setups.
Your job is to ALWAYS return the best 1-3 opportunities from the watchlist, even in quiet markets.
Every market day has at least one symbol with a better setup than the others — find it.

Look for:
- Relative strength or weakness vs the market
- Price movement with volume confirmation (vol ratio > 1.0 is notable)
- Stocks near 52-week highs (bullish momentum) or selling off (bearish)
- Any symbol moving more than the rest is worth a call or put

Return a JSON array of the TOP 1-3 candidates. ALWAYS return at least 1:
[
  {
    "symbol": "NVDA",
    "direction": "bullish",
    "option_type": "call",
    "signal_strength": 7,
    "key_reason": "one concise sentence explaining the setup",
    "priority": 1
  }
]

Only respond with valid JSON. No text outside the JSON array."""

        messages = [{"role": "user", "content": f"Market data snapshot:\n\n{summary}\n\nPick the best 1-3 setups. Always return at least 1."}]
        raw = await self._call(system, messages, max_tokens=600, stream=False)
        candidates = self._parse_json(raw)
        if not isinstance(candidates, list):
            candidates = []

        for c in candidates:
            sym = c.get("symbol", "")
            if sym in market_snapshot:
                c["current_price"] = market_snapshot[sym].get("price", 0)
                c["iv_rank"] = None  # IV rank is computed later by Options Analyst

        await self._emit("status", f"Found {len(candidates)} candidates: {[c.get('symbol') for c in candidates]}")
        return candidates

    def _gather_data(self) -> dict:
        """Batch-fetch quotes — fast, single yfinance call."""
        import yfinance as yf
        result = {}
        try:
            # Single batched download for all symbols — much faster than per-symbol calls
            raw = yf.download(
                self.watchlist, period="5d", interval="1d",
                progress=False, auto_adjust=True, group_by="ticker"
            )
            for sym in self.watchlist:
                try:
                    if len(self.watchlist) == 1:
                        df = raw
                    else:
                        df = raw[sym] if sym in raw.columns.get_level_values(0) else None
                    if df is None or df.empty:
                        continue
                    df = df.dropna(subset=["Close"])
                    if len(df) < 2:
                        continue
                    price = float(df["Close"].iloc[-1])
                    prev  = float(df["Close"].iloc[-2])
                    vol_today = float(df["Volume"].iloc[-1]) if "Volume" in df.columns else 0
                    vol_avg   = float(df["Volume"].iloc[:-1].mean()) if "Volume" in df.columns else 1
                    pct = round((price - prev) / prev * 100, 2) if prev else 0
                    vol_ratio = round(vol_today / vol_avg, 2) if vol_avg else 1.0
                    result[sym] = {
                        "price": price,
                        "pct_change": pct,
                        "volume_ratio": vol_ratio,
                    }
                except Exception as e:
                    logger.debug(f"Parse error for {sym}: {e}")
        except Exception as e:
            logger.error(f"Batch download error: {e}")
            # Fallback: individual quotes
            for sym in self.watchlist:
                try:
                    q = md.get_quote(sym)
                    if q.get("price"):
                        result[sym] = {
                            "price": q["price"],
                            "pct_change": q.get("pct_change", 0),
                            "volume_ratio": 1.0,
                        }
                except Exception:
                    pass
        return result

    def _build_summary(self, data: dict) -> str:
        lines = ["Symbol | Price    | Day %  | Vol Ratio"]
        lines.append("-" * 42)
        for sym, d in sorted(data.items(), key=lambda x: abs(x[1].get("pct_change", 0)), reverse=True):
            lines.append(
                f"{sym:6} | ${d['price']:8.2f} | {d['pct_change']:+5.1f}% | {d['volume_ratio']:.1f}x"
            )
        return "\n".join(lines)
