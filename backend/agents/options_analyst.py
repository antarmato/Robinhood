"""
Options Analyst Agent — evaluates IV environment, selects the best single-leg
call or put strike and expiry, and prices the trade.
"""

import logging
from datetime import date, datetime
from typing import Optional

import anthropic

from .base import BaseAgent, BroadcastFn
from .. import market_data as md

logger = logging.getLogger(__name__)


class OptionsAnalystAgent(BaseAgent):
    def __init__(
        self,
        client: anthropic.AsyncAnthropic,
        max_dte: int = 21,
        min_dte: int = 5,
        broadcast: Optional[BroadcastFn] = None,
    ):
        super().__init__(client, "Options", model="claude-sonnet-4-6", broadcast=broadcast)
        self.max_dte = max_dte
        self.min_dte = min_dte

    async def analyze(self, symbol: str, direction: str, current_price: float) -> dict:
        """
        Evaluate the options chain and propose the best single-leg call or put.
        Returns the recommended trade details + score.
        """
        await self._emit("status", f"Analyzing options chain for {symbol}...")

        option_type = "call" if direction == "bullish" else "put"
        expiry, dte = self._pick_expiry(symbol)
        if not expiry:
            return {"score": 1, "summary": "No suitable expiration found.", "strike": None}

        chain = md.get_options_chain(symbol, expiry, option_type)
        if not chain:
            return {"score": 1, "summary": "Could not fetch options chain.", "strike": None}

        chain_text = self._format_chain(symbol, chain, current_price, expiry, dte, direction)

        system = f"""You are an expert options trader selecting the best single-leg {option_type} to buy on {symbol}.

Selection criteria:
- Strike: slightly OTM (for calls: 1-5% above current price; for puts: 1-5% below)
- Target delta: 0.35-0.45 (estimated from moneyness since Greeks not always available)
- Premium: ideally $0.50-$3.00 per share — not too cheap (lottery), not too expensive
- Liquidity: volume > 50, open interest > 200 preferred
- Avoid buying when IV rank > 70 (premium too expensive)

Respond ONLY with JSON:
{{
  "strike": <float>,
  "expiration_date": "{expiry}",
  "option_type": "{option_type}",
  "estimated_premium": <mid price per share>,
  "estimated_max_loss": <premium per share — that's it, defined risk>,
  "iv": <implied volatility of recommended strike>,
  "volume": <volume of recommended strike>,
  "open_interest": <OI of recommended strike>,
  "score": <1-10>,
  "summary": "<2-3 sentence assessment of the options setup>"
}}"""

        messages = [{"role": "user", "content": chain_text}]
        raw = await self._call(system, messages, max_tokens=512)
        result = self._parse_json(raw)
        result.setdefault("score", 5)
        result.setdefault("summary", "Options analysis complete.")
        result["dte"] = dte
        result["expiration_date"] = expiry
        result["option_type"] = option_type
        return result

    def _pick_expiry(self, symbol: str) -> tuple[Optional[str], int]:
        expirations = md.get_options_expiration_dates(symbol)
        today = date.today()
        for exp in expirations:
            dte = (datetime.strptime(exp, "%Y-%m-%d").date() - today).days
            if self.min_dte <= dte <= self.max_dte:
                return exp, dte
        return None, 0

    def _format_chain(
        self, symbol: str, chain: list[dict], price: float,
        expiry: str, dte: int, direction: str
    ) -> str:
        lo, hi = price * 0.85, price * 1.15
        filtered = sorted(
            [o for o in chain if lo <= o["strike_price"] <= hi],
            key=lambda x: x["strike_price"]
        )
        lines = [
            f"{symbol} {direction} options | {expiry} ({dte} DTE) | Current: ${price:.2f}",
            "",
            f"{'Strike':>8} | {'Bid':>6} | {'Ask':>6} | {'Mid':>6} | {'Vol':>6} | {'OI':>7} | {'IV':>6} | {'ITM':>4}",
            "-" * 65,
        ]
        for o in filtered:
            mid = (o["bid"] + o["ask"]) / 2 if o["bid"] and o["ask"] else o["last"]
            atm = " ◄" if abs(o["strike_price"] - price) < price * 0.015 else ""
            lines.append(
                f"{o['strike_price']:>8.1f} | {o['bid']:>6.2f} | {o['ask']:>6.2f} | {mid:>6.2f} | "
                f"{o['volume']:>6} | {o['open_interest']:>7} | {o['implied_volatility']:>5.1%} | "
                f"{'Y' if o['in_the_money'] else 'N':>4}{atm}"
            )
        return "\n".join(lines)
