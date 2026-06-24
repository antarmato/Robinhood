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

# Polygon free tier doesn't return beta — use hard-coded estimates for our watchlist
# Updated periodically; these are 1Y beta vs SPY (rough market consensus)
_BETA_MAP: dict[str, float] = {
    "MSTR":  3.5,  # near-pure Bitcoin proxy
    "COIN":  3.0,  # crypto exchange
    "IONQ":  2.5,  # small-cap quantum, highly speculative
    "RIVN":  2.2,  # pre-profit EV startup
    "SMCI":  2.2,  # high growth hardware, volatile
    "HOOD":  1.9,  # retail fintech, high correlation to sentiment
    "ROKU":  1.8,  # streaming growth stock
    "TSLA":  2.0,  # megacap but highly volatile
    "AMD":   1.7,  # semiconductor cycle amp
    "NVDA":  1.6,  # AI darling, above-market but liquid
    "PLTR":  1.5,  # data/gov tech
    "SOFI":  1.5,  # fintech small-mid cap
    "SQ":    1.6,  # fintech mid-cap
    "UBER":  1.3,  # gig economy, moderate beta
    "PYPL":  1.2,  # mature fintech, lower vol
}


class FundamentalAgent(BaseAgent):
    def __init__(self, client, broadcast: Optional[BroadcastFn] = None):
        super().__init__(client, "Fundamental", broadcast=broadcast)

    async def analyze(self, symbol: str, expiration_date: Optional[str] = None) -> dict:
        await self._emit("status", f"Fundamental: checking {symbol}...")

        if not expiration_date:
            expiration_date = (date.today() + timedelta(days=45)).strftime("%Y-%m-%d")

        fund = md.get_fundamentals(symbol)
        # Inject current price so _score() can compute 52w range position
        price_data = md.get_quote(symbol)
        if price_data:
            fund["current_price"] = price_data.get("price", 0)

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
            "52w_high":             fund.get("52w_high"),
            "52w_low":              fund.get("52w_low"),
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
        # Polygon free tier doesn't return beta; use our hard-coded table, then
        # fall back to 1.0 (neutral) if symbol is unknown.
        beta_raw = fund.get("beta")
        if beta_raw is not None:
            try:
                beta = float(beta_raw)
            except (TypeError, ValueError):
                beta = _BETA_MAP.get(symbol, 1.0)
        else:
            beta = _BETA_MAP.get(symbol, 1.0)

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

        # ── 52-week price position ────────────────────────────────────────────
        price = fund.get("current_price") or fund.get("price")
        high52 = fund.get("52w_high")
        low52  = fund.get("52w_low")
        if price and high52 and low52 and high52 > low52:
            try:
                price, high52, low52 = float(price), float(high52), float(low52)
                pct_from_high = (price - high52) / high52 * 100   # negative = below high
                pct_from_low  = (price - low52)  / low52  * 100   # positive = above low
                rng = high52 - low52
                # Where in 52w range is price? (0=low, 1=high)
                range_pos = (price - low52) / rng
                if range_pos > 0.90:
                    score += 0.5; notes.append(f"52w high breakout range ({range_pos:.0%})")
                elif range_pos > 0.75:
                    score += 0.25; notes.append(f"52w upper range ({range_pos:.0%})")
                elif range_pos < 0.10:
                    score -= 0.5; notes.append(f"52w low breakdown range ({range_pos:.0%})")
                elif range_pos < 0.25:
                    score -= 0.25; notes.append(f"52w lower range ({range_pos:.0%})")
            except (TypeError, ValueError):
                pass

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
