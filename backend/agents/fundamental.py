"""
Fundamental Agent — pure Python, no LLM.

Checks earnings risk, market cap, beta, short ratio, and liquidity.
Scoring is deliberately differentiated: 5 = risky setup, 7 = neutral/clean, 9 = ideal.
"""

import logging
from datetime import date, datetime, timedelta
from typing import Optional

from .base import BaseAgent, BroadcastFn
from .. import market_data as md

logger = logging.getLogger(__name__)

# Known high-volatility / meme / crypto-adjacent stocks — extra beta penalty
_HIGH_RISK = {"MSTR", "IONQ", "RIVN", "SMCI", "HOOD", "COIN", "AMC", "GME", "BBBY", "PLTR"}
# Known mega/large cap with good liquidity — bonus
_QUALITY_CAP = {"AAPL", "MSFT", "NVDA", "GOOGL", "AMZN", "META", "AMD", "SPY", "QQQ"}


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
        score = 6.5  # neutral starting point (lower than before to allow upside differentiation)
        notes: list[str] = []

        # ── Earnings: hard penalty (fatal flaw handled above) ─────────────────
        if earnings_risk:
            return 2.0, f"⚠ Earnings {earnings_date_str} within option window — binary risk", "high"

        # ── Market cap quality ─────────────────────────────────────────────────
        mcap = fund.get("market_cap")
        if mcap:
            if mcap > 200_000_000_000:    # >$200B — mega cap, liquid
                score += 1.0; notes.append(f"mega-cap ${mcap/1e9:.0f}B")
            elif mcap > 50_000_000_000:   # >$50B — large cap
                score += 0.5; notes.append(f"large-cap ${mcap/1e9:.0f}B")
            elif mcap > 10_000_000_000:   # >$10B — mid cap
                score += 0.0; notes.append(f"mid-cap ${mcap/1e9:.0f}B")
            elif mcap > 2_000_000_000:    # >$2B — small-mid
                score -= 0.5; notes.append(f"small-mid ${mcap/1e9:.1f}B")
            else:                          # micro cap
                score -= 1.5; notes.append(f"micro-cap ${mcap/1e6:.0f}M")

        # ── Beta — directional option risk ────────────────────────────────────
        beta = fund.get("beta") or 1.0
        try:
            beta = float(beta)
        except (TypeError, ValueError):
            beta = 1.0

        if beta > 3.0:
            score -= 1.5; notes.append(f"beta {beta:.1f} — extreme vol, wide stops needed")
        elif beta > 2.0:
            score -= 0.75; notes.append(f"beta {beta:.1f} — high vol")
        elif beta > 1.5:
            score -= 0.25; notes.append(f"beta {beta:.1f}")
        elif beta < 0.5 and beta > 0:
            score -= 0.5;  notes.append(f"beta {beta:.1f} — low movement, poor for options")

        # ── Short ratio — squeeze risk for puts ───────────────────────────────
        short_ratio = fund.get("short_ratio") or 0
        try:
            short_ratio = float(short_ratio)
        except (TypeError, ValueError):
            short_ratio = 0

        if short_ratio > 12:
            score -= 1.5; notes.append(f"short ratio {short_ratio:.1f} — extreme squeeze risk")
        elif short_ratio > 8:
            score -= 0.75; notes.append(f"short ratio {short_ratio:.1f} — squeeze risk")
        elif short_ratio > 5:
            score -= 0.25; notes.append(f"short ratio {short_ratio:.1f}")

        # ── Known high-risk symbols ───────────────────────────────────────────
        if symbol in _HIGH_RISK:
            score -= 0.5; notes.append("high-volatility / speculative")

        # ── Quality bonus ─────────────────────────────────────────────────────
        if symbol in _QUALITY_CAP:
            score += 0.5; notes.append("quality large cap")

        score = round(max(1.0, min(10.0, score)), 1)
        catalyst_risk = "low" if score >= 7 else "medium" if score >= 5 else "high"
        summary = ", ".join(notes) if notes else "No earnings risk, neutral fundamentals"
        return score, summary, catalyst_risk
