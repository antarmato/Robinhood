"""
Fundamental Agent — checks earnings risk, company health, and catalyst backdrop.
Uses yfinance for fundamentals and earnings dates.
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

    async def analyze(self, symbol: str, expiration_date: str) -> dict:
        await self._emit("status", f"Checking fundamentals for {symbol}...")

        fund = md.get_fundamentals(symbol)
        earnings_risk = self._check_earnings_risk(fund, expiration_date)
        context = self._build_context(symbol, fund, earnings_risk, expiration_date)

        system = f"""You are a fundamental analyst evaluating option trade risk for {symbol}.

Key concerns:
1. Earnings — is there an earnings report before the option expires? Creates massive gap risk. Score 1-3 if yes.
2. Company health — any red flags (high short ratio, negative growth)?
3. Sector/macro — any known headwinds?

Respond ONLY with JSON:
{{
  "score": <1-10, where 10=clean fundamental backdrop>,
  "earnings_before_expiry": true | false,
  "earnings_date": "<YYYY-MM-DD or null>",
  "catalyst_risk": "high" | "medium" | "low",
  "summary": "<2-3 sentence fundamental assessment>"
}}"""

        raw = await self._call(system, [{"role": "user", "content": context}], max_tokens=350, stream=False)
        result = self._parse_json(raw)
        result.setdefault("score", 5)
        result.setdefault("earnings_before_expiry", earnings_risk)
        result.setdefault("summary", "Fundamental check complete.")
        return result

    def _check_earnings_risk(self, fund: dict, expiration_date: str) -> bool:
        """Check if known earnings date falls before option expiry."""
        try:
            earnings = fund.get("earnings_date")
            if not earnings:
                return False
            exp = datetime.strptime(expiration_date, "%Y-%m-%d").date()
            earn = datetime.strptime(earnings, "%Y-%m-%d").date()
            today = date.today()
            return today <= earn <= exp
        except Exception:
            return False

    def _build_context(self, symbol: str, fund: dict, earnings_risk: bool, expiration_date: str) -> str:
        today = date.today()
        exp = datetime.strptime(expiration_date, "%Y-%m-%d").date()
        dte = (exp - today).days
        return f"""Fundamental snapshot for {symbol}

Trade window: {today} → {expiration_date} ({dte} days)

Company Data:
  Sector:         {fund.get('sector', 'N/A')}
  Market Cap:     {fund.get('market_cap', 'N/A')}
  P/E (trailing): {fund.get('pe_ratio', 'N/A')}
  P/E (forward):  {fund.get('forward_pe', 'N/A')}
  Revenue growth: {fund.get('revenue_growth', 'N/A')}
  Earnings growth:{fund.get('earnings_growth', 'N/A')}
  Short ratio:    {fund.get('short_ratio', 'N/A')}
  Beta:           {fund.get('beta', 'N/A')}
  52W High:       {fund.get('52w_high', 'N/A')}
  52W Low:        {fund.get('52w_low', 'N/A')}

Next Earnings:    {fund.get('earnings_date', 'Unknown')}
Earnings in window: {'YES — HIGH RISK' if earnings_risk else 'No'}

Description: {fund.get('description', 'N/A')}"""
