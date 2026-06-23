"""
Fundamental Agent — pure Python, no LLM.

Checks earnings risk within option DTE window, market cap quality, and analyst data.
Scoring rules:
  Earnings in 45-day window → score 2 (near-fatal for directional plays)
  Clean, large cap, no catalyst risk → score 7-8
  Small cap / no analyst data → score 5-6
"""

import logging
from datetime import date, datetime, timedelta
from typing import Optional

from .base import BaseAgent, BroadcastFn
from .. import market_data as md

logger = logging.getLogger(__name__)


class FundamentalAgent(BaseAgent):
    def __init__(self, client, broadcast: Optional[BroadcastFn] = None):
        super().__init__(client, "Fundamental", broadcast=broadcast)

    async def analyze(self, symbol: str, expiration_date: Optional[str] = None) -> dict:
        await self._emit("status", f"Fundamental: checking {symbol}...")

        if not expiration_date:
            expiration_date = (date.today() + timedelta(days=45)).strftime("%Y-%m-%d")

        fund = md.get_fundamentals(symbol)
        earnings_risk, earnings_date_str = self._check_earnings_risk(fund, expiration_date)

        score, summary, catalyst_risk = self._score(fund, earnings_risk, earnings_date_str, symbol)

        result = {
            "score":                score,
            "earnings_before_expiry": earnings_risk,
            "earnings_date":        earnings_date_str,
            "catalyst_risk":        catalyst_risk,
            "analyst_consensus":    "unknown",
            "upside_to_target":     None,
            "market_cap":           fund.get("market_cap"),
            "summary":              summary,
            "fatal_flaw": (
                f"Earnings {earnings_date_str} — binary gap risk before expiration"
                if earnings_risk else None
            ),
        }
        await self._emit("score", {"symbol": symbol, "score": score,
                                    "earnings_risk": earnings_risk, "catalyst": catalyst_risk})
        return result

    # ── Earnings check ────────────────────────────────────────────────────────

    def _check_earnings_risk(self, fund: dict, expiration_date: str) -> tuple[bool, Optional[str]]:
        try:
            exp   = datetime.strptime(expiration_date, "%Y-%m-%d").date()
            today = date.today()
            for key in ("earnings_ts_start", "earningsTimestamp", "earningsTimestampStart"):
                ts = fund.get(key)
                if ts and isinstance(ts, (int, float)) and ts > 0:
                    try:
                        earn_date = datetime.utcfromtimestamp(ts).date()
                        if today <= earn_date <= exp:
                            return True, earn_date.strftime("%Y-%m-%d")
                    except Exception:
                        pass
            earn_str = fund.get("earnings_date")
            if earn_str:
                earn_date = datetime.strptime(earn_str, "%Y-%m-%d").date()
                if today <= earn_date <= exp:
                    return True, earn_str
        except Exception as e:
            logger.debug(f"Earnings check error: {e}")
        return False, None

    # ── Pure Python scoring ───────────────────────────────────────────────────

    def _score(
        self, fund: dict, earnings_risk: bool, earnings_date_str: Optional[str], symbol: str
    ) -> tuple[float, str, str]:
        score = 7.0  # default: clean backdrop

        # Earnings risk is the only hard penalty
        if earnings_risk:
            score = 2.0
            return score, f"⚠ Earnings {earnings_date_str} within option window — binary risk", "high"

        # Market cap quality bonus
        mcap = fund.get("market_cap")
        if mcap:
            if mcap > 100_000_000_000:    # > $100B mega-cap
                score += 0.5
            elif mcap < 2_000_000_000:    # < $2B small-cap
                score -= 1.0

        # Short squeeze risk flag (doesn't change score much — info only)
        short_ratio = fund.get("short_ratio") or 0
        squeeze_note = f" | Short ratio {short_ratio:.1f} — squeeze risk" if short_ratio > 8 else ""

        # Beta — extreme beta means higher risk
        beta = fund.get("beta") or 1.0
        if beta and beta > 2.5:
            score -= 0.5

        score = round(max(1.0, min(10.0, score)), 1)
        catalyst_risk = "low" if score >= 7 else "medium"

        cap_str = ""
        if mcap:
            if mcap > 1e12:    cap_str = f"${mcap/1e12:.1f}T"
            elif mcap > 1e9:   cap_str = f"${mcap/1e9:.1f}B"
            else:              cap_str = f"${mcap/1e6:.0f}M"

        summary = (
            f"No earnings risk in window. "
            f"Market cap {cap_str or 'N/A'}, beta {beta:.1f}"
            f"{squeeze_note}"
        )
        return score, summary, catalyst_risk
