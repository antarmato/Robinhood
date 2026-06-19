"""
Options Analyst Agent — evaluates IV environment, selects the best single-leg
call or put strike and expiry, and prices the trade.

Key fix from v1: DTE range expanded to 7-45 (was 5-21), with a fallback
to the nearest available expiry so this never returns None on a tradable stock.
"""

import logging
from datetime import date, datetime
from typing import Optional, Tuple

import anthropic

from .base import BaseAgent, BroadcastFn
from .. import market_data as md

logger = logging.getLogger(__name__)

# DTE preference tiers (checked in order, first match wins)
_DTE_TIERS = [
    (10, 35),   # Sweet spot: enough time, not too much theta
    (7,  42),   # Acceptable range
    (5,  56),   # Last resort fallback
]


class OptionsAnalystAgent(BaseAgent):
    def __init__(
        self,
        client: anthropic.AsyncAnthropic,
        max_dte: int = 45,
        min_dte: int = 7,
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
            await self._emit("status", f"{symbol}: No valid expiration found in any DTE range.")
            return {
                "score": 1, "summary": "No suitable expiration found.",
                "strike": None, "expiration_date": None, "option_type": option_type,
            }

        chain = md.get_options_chain(symbol, expiry, option_type)
        if not chain:
            await self._emit("status", f"{symbol}: Empty options chain for {expiry}.")
            return {
                "score": 1, "summary": f"Empty options chain for {expiry}.",
                "strike": None, "expiration_date": expiry, "option_type": option_type,
            }

        # Filter and score the chain
        chain_metrics = self._compute_chain_metrics(chain, current_price, option_type)
        chain_text = self._format_chain(symbol, chain, current_price, expiry, dte, direction, chain_metrics)

        system = f"""You are an expert options trader selecting the best single-leg {option_type} to buy on {symbol}.

Direction: {direction} | Expiry: {expiry} ({dte} DTE) | Stock price: ${current_price:.2f}

Strike selection criteria for {option_type}s:
- Target delta zone: slightly OTM ({"1-5% above current price" if option_type == "call" else "1-5% below current price"})
- Premium sweet spot: $0.30–$5.00 per share (defined risk)
- Minimum liquidity: volume > 10 OR open_interest > 50 (we need to be able to exit)
- Avoid: bid-ask spread > 25% of mid price (too illiquid)
- Avoid: IV > 90% (premium too expensive relative to historical)

Liquidity tier in the table: A=best (vol>100, oi>500), B=good, C=marginal, D=avoid

Respond ONLY with JSON:
{{
  "strike": <float>,
  "expiration_date": "{expiry}",
  "option_type": "{option_type}",
  "estimated_premium": <mid price per share>,
  "estimated_max_loss": <premium per share — total per share risk>,
  "iv": <implied volatility of recommended strike as decimal, e.g. 0.35>,
  "volume": <volume of recommended strike>,
  "open_interest": <OI of recommended strike>,
  "bid_ask_spread_pct": <(ask-bid)/mid as decimal>,
  "score": <1-10 where 10=excellent setup>,
  "score_reason": "<one line: why this score>",
  "summary": "<2-3 sentence assessment of IV environment and the chosen strike>"
}}"""

        messages = [{"role": "user", "content": chain_text}]
        raw = await self._call(system, messages, max_tokens=600)
        result = self._parse_json(raw)

        # Validate required fields
        result.setdefault("score", 5)
        result.setdefault("summary", "Options analysis complete.")
        result["dte"]             = dte
        result["expiration_date"] = expiry
        result["option_type"]     = option_type

        # Hard checks: reject if strike is missing or premium is 0
        if not result.get("strike"):
            result["score"] = 1
            result["summary"] = "Could not identify a viable strike."
        elif result.get("estimated_premium", 0) <= 0:
            result["score"] = 2
            result["summary"] = "Could not price the recommended option."

        # Flag liquidity issues
        ba_pct = result.get("bid_ask_spread_pct", 0) or 0
        if ba_pct > 0.25:
            result["score"] = min(result["score"], 4)
            result["summary"] += f" WARNING: wide bid-ask spread ({ba_pct:.0%})."

        return result

    # ── Expiry selection ──────────────────────────────────────────────────────

    def _pick_expiry(self, symbol: str) -> Tuple[Optional[str], int]:
        """
        Find the best expiration date. Tries DTE tiers in order:
        1. 10-35 DTE (sweet spot)
        2. 7-42 DTE (acceptable)
        3. 5-56 DTE (last resort)
        4. Any future expiry ≥ 4 DTE (absolute fallback)
        """
        expirations = md.get_options_expiration_dates(symbol)
        if not expirations:
            return None, 0

        today = date.today()
        parsed = []
        for exp in expirations:
            try:
                exp_date = datetime.strptime(exp, "%Y-%m-%d").date()
                dte = (exp_date - today).days
                if dte >= 4:
                    parsed.append((exp, dte))
            except ValueError:
                continue

        if not parsed:
            return None, 0

        # Try preferred DTE tiers first
        for lo, hi in _DTE_TIERS:
            for exp, dte in parsed:
                if lo <= dte <= hi:
                    return exp, dte

        # Absolute fallback: nearest future expiry
        parsed.sort(key=lambda x: x[1])
        return parsed[0]

    # ── Chain analysis ────────────────────────────────────────────────────────

    def _compute_chain_metrics(self, chain: list[dict], price: float, option_type: str) -> dict:
        """Compute aggregate chain metrics for context."""
        total_oi  = sum(o.get("open_interest", 0) for o in chain)
        total_vol = sum(o.get("volume", 0) for o in chain)
        ivs = [o["implied_volatility"] for o in chain if o.get("implied_volatility", 0) > 0]
        atm = min(chain, key=lambda x: abs(x["strike_price"] - price)) if chain else None
        atm_iv = atm.get("implied_volatility", 0) if atm else 0
        return {
            "total_oi":  total_oi,
            "total_vol": total_vol,
            "avg_iv":    round(sum(ivs) / len(ivs), 4) if ivs else 0,
            "atm_iv":    atm_iv,
            "atm_strike": atm.get("strike_price") if atm else price,
        }

    def _format_chain(
        self, symbol: str, chain: list[dict], price: float,
        expiry: str, dte: int, direction: str, metrics: dict,
    ) -> str:
        """Format the chain for LLM consumption. Filter to ±12% moneyness."""
        lo, hi = price * 0.88, price * 1.12
        filtered = sorted(
            [o for o in chain if lo <= o["strike_price"] <= hi],
            key=lambda x: x["strike_price"]
        )
        if not filtered:
            filtered = chain[:20]  # fallback: show all

        lines = [
            f"{symbol} {direction.upper()} | {expiry} ({dte} DTE) | Stock: ${price:.2f}",
            f"Chain stats: ATM IV={metrics['atm_iv']:.1%} | Avg IV={metrics['avg_iv']:.1%} "
            f"| Total OI={metrics['total_oi']:,} | Total Vol={metrics['total_vol']:,}",
            "",
            f"{'Strike':>8} | {'Bid':>6} | {'Ask':>6} | {'Mid':>6} | {'Sprd%':>6} | "
            f"{'Vol':>6} | {'OI':>7} | {'IV':>6} | {'ITM':>4} | {'Liq':>3}",
            "-" * 73,
        ]
        for o in filtered:
            mid  = o.get("mid") or (o["bid"] + o["ask"]) / 2 if o["bid"] and o["ask"] else o["last"]
            sprd = (o["ask"] - o["bid"]) / mid if mid > 0 else 0
            # Liquidity tier
            vol, oi = o.get("volume", 0), o.get("open_interest", 0)
            if vol > 100 and oi > 500:   liq = "A"
            elif vol > 30 and oi > 150:  liq = "B"
            elif vol > 10 or oi > 50:    liq = "C"
            else:                         liq = "D"

            atm_marker = " ◄ATM" if abs(o["strike_price"] - price) < price * 0.015 else ""
            lines.append(
                f"{o['strike_price']:>8.1f} | {o['bid']:>6.2f} | {o['ask']:>6.2f} | {mid:>6.2f} | "
                f"{sprd:>5.0%} | {vol:>6} | {oi:>7} | {o['implied_volatility']:>5.1%} | "
                f"{'Y' if o['in_the_money'] else 'N':>4} | {liq:>3}{atm_marker}"
            )
        return "\n".join(lines)
