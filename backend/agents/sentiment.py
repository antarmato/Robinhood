"""
Sentiment Agent — evaluates put/call ratio, options skew, and macro context.
Uses yfinance options chain data.
"""

import logging
from typing import Optional

import anthropic

from .base import BaseAgent, BroadcastFn
from .. import market_data as md

logger = logging.getLogger(__name__)


class SentimentAgent(BaseAgent):
    def __init__(
        self,
        client: anthropic.AsyncAnthropic,
        broadcast: Optional[BroadcastFn] = None,
    ):
        super().__init__(client, "Sentiment", model="claude-haiku-4-5-20251001", broadcast=broadcast)

    async def analyze(self, symbol: str, direction: str, expiration_date: str) -> dict:
        await self._emit("status", f"Analyzing sentiment for {symbol}...")

        spy_ctx = self._get_spy_context()
        pcr = self._compute_pcr(symbol, expiration_date)
        context = self._build_context(symbol, direction, spy_ctx, pcr)

        system = f"""You are a market sentiment analyst evaluating whether sentiment supports a {direction} options trade on {symbol}.

Assess:
- Put/Call ratio (OI): < 0.7 bullish, > 1.0 bearish/fearful
- IV skew: puts more expensive than calls = bearish skew
- Macro backdrop: SPY trend

Does the sentiment SUPPORT or CONTRADICT the {direction} trade?

Respond ONLY with JSON:
{{
  "score": <1-10 where 10=strongly supports direction>,
  "pcr": <put_call_ratio or null>,
  "skew": "bullish" | "neutral" | "bearish",
  "macro_sentiment": "risk_on" | "neutral" | "risk_off",
  "summary": "<2 sentence sentiment assessment>"
}}"""

        raw = await self._call(system, [{"role": "user", "content": context}], max_tokens=300, stream=False)
        result = self._parse_json(raw)
        result.setdefault("score", 5)
        result.setdefault("summary", "Sentiment analysis complete.")
        return result

    def _get_spy_context(self) -> dict:
        spy = md.get_quote("SPY")
        qqq = md.get_quote("QQQ")
        return {
            "spy_price": spy.get("price", 0),
            "spy_change": spy.get("pct_change", 0),
            "qqq_price": qqq.get("price", 0),
            "qqq_change": qqq.get("pct_change", 0),
        }

    def _compute_pcr(self, symbol: str, expiration_date: str) -> dict:
        try:
            calls = md.get_options_chain(symbol, expiration_date, "call")
            puts  = md.get_options_chain(symbol, expiration_date, "put")
            if not calls or not puts:
                return {}

            call_oi  = sum(c.get("open_interest", 0) for c in calls)
            put_oi   = sum(p.get("open_interest", 0) for p in puts)
            call_vol = sum(c.get("volume", 0) for c in calls)
            put_vol  = sum(p.get("volume", 0) for p in puts)
            call_ivs = [c["implied_volatility"] for c in calls if c.get("implied_volatility")]
            put_ivs  = [p["implied_volatility"] for p in puts  if p.get("implied_volatility")]

            return {
                "pcr_oi":      round(put_oi  / call_oi,  3) if call_oi  else None,
                "pcr_vol":     round(put_vol / call_vol, 3) if call_vol else None,
                "call_oi":     call_oi,   "put_oi":  put_oi,
                "call_vol":    call_vol,  "put_vol": put_vol,
                "avg_call_iv": round(sum(call_ivs)/len(call_ivs), 4) if call_ivs else 0,
                "avg_put_iv":  round(sum(put_ivs) /len(put_ivs),  4) if put_ivs  else 0,
            }
        except Exception as e:
            logger.error(f"PCR error: {e}")
            return {}

    def _build_context(self, symbol: str, direction: str, spy: dict, pcr: dict) -> str:
        skew = round((pcr.get("avg_put_iv", 0) - pcr.get("avg_call_iv", 0)), 4)
        return f"""Sentiment for {symbol} | Direction: {direction}

Macro:
  SPY: ${spy.get('spy_price', 0):.2f} ({spy.get('spy_change', 0):+.2f}%)
  QQQ: ${spy.get('qqq_price', 0):.2f} ({spy.get('qqq_change', 0):+.2f}%)

{symbol} Options Sentiment:
  Put/Call Ratio (OI):  {pcr.get('pcr_oi', 'N/A')}
  Put/Call Ratio (Vol): {pcr.get('pcr_vol', 'N/A')}
  Call OI: {pcr.get('call_oi', 0):,}  |  Put OI: {pcr.get('put_oi', 0):,}
  Call Vol: {pcr.get('call_vol', 0):,}  |  Put Vol: {pcr.get('put_vol', 0):,}
  Avg Call IV: {pcr.get('avg_call_iv', 0):.1%}  |  Avg Put IV: {pcr.get('avg_put_iv', 0):.1%}
  IV Skew (P-C): {skew:.4f}  ({'Bearish skew' if skew > 0.02 else 'Bullish skew' if skew < -0.02 else 'Neutral'})"""
