"""
Fundamental Agent — checks earnings risk, company health, and catalyst backdrop.

Improvements from v1:
- Multiple earnings timestamp fields for better detection
- Analyst consensus (target price vs current, recommendation key)
- Short interest context (high short ratio = squeeze risk for puts, pain for calls)
- Cleaner earnings window check using both earningsTimestampStart/End
"""

import logging
from datetime import date, datetime
from typing import Optional

import anthropic

from .base import BaseAgent, BroadcastFn
from .. import market_data as md

logger = logging.getLogger(__name__)


class FundamentalAgent(BaseAgent):
    def __init__(
        self,
        client: anthropic.AsyncAnthropic,
        broadcast: Optional[BroadcastFn] = None,
    ):
        super().__init__(client, "Fundamental", model="claude-haiku-4-5-20251001", broadcast=broadcast)

    async def analyze(self, symbol: str, expiration_date: Optional[str] = None) -> dict:
        await self._emit("status", f"Checking fundamentals for {symbol}...")

        if not expiration_date:
            from datetime import timedelta
            expiration_date = (date.today() + timedelta(days=45)).strftime("%Y-%m-%d")

        fund = md.get_fundamentals(symbol)
        earnings_risk, earnings_date_str = self._check_earnings_risk(fund, expiration_date)
        context = self._build_context(symbol, fund, earnings_risk, earnings_date_str, expiration_date)

        system = f"""You are a fundamental analyst evaluating option trade risk for {symbol}.

Scoring guide:
  9-10: Clean backdrop (no earnings risk, solid growth, positive analyst consensus)
  7-8:  Decent (minor concerns, no near-term catalysts, stable business)
  5-6:  Neutral (some uncertainty, watch earnings date)
  3-4:  Risky (slowing growth, negative analyst view, catalyst risk present)
  1-2:  Avoid (confirmed earnings in window, high short interest, or major red flags)

Score 1-2 ONLY if earnings_before_expiry is confirmed true.
Short ratio > 10: significant squeeze risk (good for unexpected calls, risky for puts).

Respond ONLY with JSON:
{{
  "score": <1-10>,
  "earnings_before_expiry": true | false,
  "earnings_date": "<YYYY-MM-DD or null>",
  "catalyst_risk": "high" | "medium" | "low",
  "analyst_consensus": "buy" | "hold" | "sell" | "unknown",
  "upside_to_target": <% upside from current price to analyst target, or null>,
  "summary": "<2-3 sentences: earnings situation, company health, and analyst backdrop>"
}}"""

        raw = await self._call(system, [{"role": "user", "content": context}], max_tokens=400, stream=False)
        result = self._parse_json(raw)
        result.setdefault("score", 5)
        result.setdefault("earnings_before_expiry", earnings_risk)
        result.setdefault("summary", "Fundamental check complete.")

        # Override earnings_before_expiry with our computed value (more reliable than LLM guess)
        if earnings_risk:
            result["earnings_before_expiry"] = True
            result["score"] = min(result.get("score", 5), 2)  # cap at 2 if earnings in window
        return result

    # ── Earnings risk detection ───────────────────────────────────────────────

    def _check_earnings_risk(self, fund: dict, expiration_date: str) -> tuple[bool, Optional[str]]:
        """
        Check if any earnings date falls within the option's DTE window.
        Uses multiple yfinance fields since they're inconsistently populated.
        Returns (is_risky, earnings_date_str).
        """
        try:
            exp = datetime.strptime(expiration_date, "%Y-%m-%d").date()
            today = date.today()

            # Try all timestamp fields yfinance might populate
            for key in ("earnings_ts_start", "earningsTimestamp", "earningsTimestampStart"):
                ts = fund.get(key)
                if ts and isinstance(ts, (int, float)) and ts > 0:
                    try:
                        earn_date = datetime.utcfromtimestamp(ts).date()
                        if today <= earn_date <= exp:
                            return True, earn_date.strftime("%Y-%m-%d")
                    except Exception:
                        pass

            # Try string date field
            earn_str = fund.get("earnings_date")
            if earn_str:
                earn_date = datetime.strptime(earn_str, "%Y-%m-%d").date()
                if today <= earn_date <= exp:
                    return True, earn_str

        except Exception as e:
            logger.debug(f"Earnings check error: {e}")
        return False, None

    def _build_context(
        self,
        symbol: str,
        fund: dict,
        earnings_risk: bool,
        earnings_date_str: Optional[str],
        expiration_date: str,
    ) -> str:
        today = date.today()
        exp   = datetime.strptime(expiration_date, "%Y-%m-%d").date()
        dte   = (exp - today).days

        # Analyst upside
        target  = fund.get("analyst_target")
        price_q = md.get_quote(symbol)
        current = price_q.get("price", 0)
        upside_str = "N/A"
        if target and current > 0:
            upside = (target - current) / current * 100
            upside_str = f"${target:.2f} ({upside:+.1f}% from current)"

        return f"""Fundamental snapshot for {symbol}

Trade window: {today} → {expiration_date} ({dte} DTE)

=== EARNINGS RISK ===
Next earnings:    {earnings_date_str or fund.get('earnings_date', 'Unknown')}
In trade window:  {'⚠ YES — HIGH RISK' if earnings_risk else 'No'}

=== ANALYST CONSENSUS ===
Recommendation:   {fund.get('analyst_rating', 'N/A').upper()}
Price target:     {upside_str}
Analysts:         {fund.get('analyst_count', 'N/A')} covering

=== COMPANY DATA ===
Sector:           {fund.get('sector', 'N/A')}
Industry:         {fund.get('industry', 'N/A')}
Market Cap:       {fund.get('market_cap', 'N/A')}
P/E (trailing):   {fund.get('pe_ratio', 'N/A')}
P/E (forward):    {fund.get('forward_pe', 'N/A')}
Revenue growth:   {fund.get('revenue_growth', 'N/A')}
Earnings growth:  {fund.get('earnings_growth', 'N/A')}
Beta:             {fund.get('beta', 'N/A')}
Short ratio:      {fund.get('short_ratio', 'N/A')} {'⚠ HIGH — squeeze risk' if (fund.get('short_ratio') or 0) > 8 else ''}
52W High:         {fund.get('52w_high', 'N/A')}
52W Low:          {fund.get('52w_low', 'N/A')}

Description: {fund.get('description', 'N/A')}"""
