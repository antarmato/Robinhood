"""
Sentiment Agent — evaluates put/call ratio, options skew, VIX, and macro context.

Improvements from v1:
- VIX level context (fear/greed regime)
- Sector ETF performance (is the stock's sector cooperating?)
- More nuanced PCR interpretation
- Macro breadth: SPY + QQQ + sector
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

    async def analyze(self, symbol: str, direction: str, expiration_date: str = None) -> dict:
        await self._emit("status", f"Analyzing sentiment and macro for {symbol}...")

        # Gather all context (these are fast yfinance calls)
        spy_ctx   = self._get_macro_context()
        pcr       = self._compute_pcr(symbol, expiration_date)
        vix       = md.get_vix()
        sectors   = md.get_sector_etf_performance()
        context   = self._build_context(symbol, direction, spy_ctx, pcr, vix, sectors)

        system = f"""You are a market sentiment analyst evaluating whether macro conditions and options flow support a {direction} trade on {symbol}.

VIX context:
- VIX < 15: low fear, good for buying calls (market complacent, trending up)
- VIX 15-20: normal, neutral for either direction
- VIX 20-30: elevated fear, good for buying puts OR contrarian calls (if oversold)
- VIX > 30: high fear, puts are expensive, better for OTM calls as mean-reversion plays

PCR interpretation (OI-based):
- PCR < 0.6: very bullish sentiment (market positioned long)
- PCR 0.6-0.8: mildly bullish
- PCR 0.8-1.1: neutral
- PCR 1.1-1.5: bearish/hedging
- PCR > 1.5: extreme fear or crowded put positioning (often contrarian bullish)

Does the overall sentiment SUPPORT or OPPOSE the {direction} trade on {symbol}?

Respond ONLY with JSON:
{{
  "score": <1-10 where 10=strongly supports the {direction} direction>,
  "pcr": <pcr_oi value or null>,
  "skew": "bullish" | "neutral" | "bearish",
  "vix_regime": "low_fear" | "normal" | "elevated" | "high_fear",
  "vix_impact": "bullish_for_calls" | "neutral" | "bullish_for_puts" | "expensive_options",
  "macro_sentiment": "risk_on" | "neutral" | "risk_off",
  "sector_aligned": true | false,
  "summary": "<2-3 sentences: VIX context, PCR signal, sector backdrop, and net impact on the proposed {direction} trade>"
}}"""

        raw = await self._call(system, [{"role": "user", "content": context}], max_tokens=400, stream=False)
        result = self._parse_json(raw)
        result.setdefault("score", 5)
        result.setdefault("summary", "Sentiment analysis complete.")
        result.setdefault("pcr", None)
        result.setdefault("skew", "neutral")
        return result

    # ── Data gathering ─────────────────────────────────────────────────────────

    def _get_macro_context(self) -> dict:
        spy = md.get_quote("SPY")
        qqq = md.get_quote("QQQ")
        iwm = md.get_quote("IWM")  # Small caps for breadth
        return {
            "spy_price":  spy.get("price", 0),
            "spy_change": spy.get("pct_change", 0),
            "qqq_price":  qqq.get("price", 0),
            "qqq_change": qqq.get("pct_change", 0),
            "iwm_price":  iwm.get("price", 0),
            "iwm_change": iwm.get("pct_change", 0),
        }

    def _compute_pcr(self, symbol: str, expiration_date: str) -> dict:
        try:
            calls = md.get_options_chain(symbol, expiration_date, "call")
            puts  = md.get_options_chain(symbol, expiration_date, "put")
            if not calls or not puts:
                return {}

            call_oi   = sum(c.get("open_interest", 0) for c in calls)
            put_oi    = sum(p.get("open_interest", 0) for p in puts)
            call_vol  = sum(c.get("volume", 0) for c in calls)
            put_vol   = sum(p.get("volume", 0) for p in puts)

            call_ivs  = [c["implied_volatility"] for c in calls if c.get("implied_volatility", 0) > 0]
            put_ivs   = [p["implied_volatility"] for p in puts  if p.get("implied_volatility", 0) > 0]
            avg_call_iv = sum(call_ivs) / len(call_ivs) if call_ivs else 0
            avg_put_iv  = sum(put_ivs)  / len(put_ivs)  if put_ivs  else 0

            # OTM skew: compare 10-delta puts vs 10-delta calls
            # Use options at ~10% OTM as proxy for skew
            return {
                "pcr_oi":      round(put_oi  / call_oi,  3) if call_oi  else None,
                "pcr_vol":     round(put_vol / call_vol, 3) if call_vol else None,
                "call_oi":     call_oi,
                "put_oi":      put_oi,
                "call_vol":    call_vol,
                "put_vol":     put_vol,
                "avg_call_iv": round(avg_call_iv, 4),
                "avg_put_iv":  round(avg_put_iv, 4),
                "iv_skew":     round(avg_put_iv - avg_call_iv, 4),
            }
        except Exception as e:
            logger.error(f"PCR error for {symbol}: {e}")
            return {}

    def _build_context(
        self,
        symbol: str,
        direction: str,
        macro: dict,
        pcr: dict,
        vix: float,
        sectors: dict,
    ) -> str:
        # Macro breadth: count how many indexes are green today
        changes = [macro.get("spy_change", 0), macro.get("qqq_change", 0), macro.get("iwm_change", 0)]
        green_count = sum(1 for c in changes if c > 0)
        breadth_str = f"{green_count}/3 major indexes green today"

        # Sector context (top 3 movers)
        sector_lines = ""
        if sectors:
            sorted_sectors = sorted(sectors.items(), key=lambda x: abs(x[1]), reverse=True)[:4]
            sector_lines = "\n".join(f"  {s}: {v:+.2f}%" for s, v in sorted_sectors)

        skew_val = pcr.get("iv_skew", 0)
        skew_desc = ("Bearish skew (puts more expensive)" if skew_val > 0.03
                     else "Bullish skew (calls more expensive)" if skew_val < -0.03
                     else "Neutral skew")

        return f"""Sentiment Analysis for {symbol} | Direction: {direction}

═══ MACRO ═══
VIX: {vix:.1f}  ({'Low fear' if vix < 15 else 'Normal' if vix < 20 else 'Elevated fear' if vix < 30 else 'HIGH FEAR'})
SPY: ${macro.get('spy_price', 0):.2f}  ({macro.get('spy_change', 0):+.2f}%)
QQQ: ${macro.get('qqq_price', 0):.2f}  ({macro.get('qqq_change', 0):+.2f}%)
IWM: ${macro.get('iwm_price', 0):.2f}  ({macro.get('iwm_change', 0):+.2f}%)
Market breadth: {breadth_str}

═══ SECTOR ETF PERFORMANCE ═══
{sector_lines if sector_lines else '  No sector data available'}

═══ {symbol} OPTIONS FLOW ═══
Put/Call Ratio (OI):  {pcr.get('pcr_oi', 'N/A')}
Put/Call Ratio (Vol): {pcr.get('pcr_vol', 'N/A')}
Call OI:  {pcr.get('call_oi', 0):,}  |  Put OI:  {pcr.get('put_oi', 0):,}
Call Vol: {pcr.get('call_vol', 0):,}  |  Put Vol: {pcr.get('put_vol', 0):,}
Avg Call IV: {pcr.get('avg_call_iv', 0):.1%}  |  Avg Put IV: {pcr.get('avg_put_iv', 0):.1%}
IV Skew (Put IV - Call IV): {skew_val:.4f}  → {skew_desc}"""
