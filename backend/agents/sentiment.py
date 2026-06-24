"""
Sentiment Agent — pure Python scoring, no LLM.

Scores 1-10 based on VIX level, sector ETF alignment, and market breadth.
Uses symbol-specific sector ETF mapping for precise alignment scoring.
"""

import logging
from typing import Optional

from .base import BaseAgent, BroadcastFn
from .. import market_data as md

logger = logging.getLogger(__name__)

# Hard-coded sector ETF for each watchlist symbol (covers default + common additions)
_SECTOR_ETF = {
    # Tech
    "NVDA": "XLK", "AMD": "XLK", "SMCI": "XLK", "PLTR": "XLK",
    "IONQ": "XLK", "ROKU": "XLK", "CRM": "XLK",
    # Crypto/Fintech adjacent (use XLK as best proxy)
    "MSTR": "XLK", "COIN": "XLK",
    # Financials
    "SOFI": "XLF", "HOOD": "XLF", "SQ": "XLF", "PYPL": "XLF",
    # Consumer Discretionary
    "TSLA": "XLY", "RIVN": "XLY", "UBER": "XLY",
    # Broad market (no sector tilt)
    "SPY": "SPY", "QQQ": "XLK",
}


class SentimentAgent(BaseAgent):
    def __init__(self, client, broadcast: Optional[BroadcastFn] = None):
        super().__init__(client, "Sentiment", broadcast=broadcast)

    async def analyze(
        self, symbol: str, direction: str,
        expiration_date: str = None,
        market_regime: dict = None,
    ) -> dict:
        await self._emit("status", f"Sentiment: scoring macro for {symbol} ({direction})...")

        vix          = md.get_vix()
        macro        = self._get_macro_context()
        sectors      = md.get_sector_etf_performance()
        vix_trend    = (market_regime or {}).get("vix_trend", "flat")
        primary_etf  = _SECTOR_ETF.get(symbol)     # specific ETF for this symbol
        news         = md.get_news_sentiment(symbol)

        score, components = self._score(direction, vix, macro, sectors, vix_trend, primary_etf, news)

        # Compute VIX regime (same logic as before — still used by Judge)
        if vix > 30:   vix_regime = "extreme"
        elif vix > 22: vix_regime = "elevated"
        elif vix > 15: vix_regime = "normal"
        else:          vix_regime = "low"

        sector_aligned = components.get("sector_aligned", True)

        news_note = ""
        if news and news.get("available") and components.get("news"):
            news_note = f", news {components['news']}"

        result = {
            "score":          score,
            "skew":           "bullish" if score >= 6 else ("bearish" if score <= 4 else "neutral"),
            "vix_level":      round(float(vix), 1),
            "vix_regime":     vix_regime,
            "macro_sentiment": "risk_on" if score >= 6 else ("risk_off" if score <= 4 else "neutral"),
            "sector_aligned": sector_aligned,
            "components":     components,
            "news":           news,
            "summary": (
                f"VIX {vix:.1f} ({vix_regime}) {vix_trend if vix_trend != 'flat' else ''}, "
                f"breadth {components.get('breadth_green', 0)}/3 green, "
                f"sector {'aligned' if sector_aligned else 'misaligned'}"
                f"{news_note} — score {score}/10"
            ),
        }
        await self._emit("score", {"symbol": symbol, "score": score, "vix": vix,
                                    "vix_regime": vix_regime, "sector_aligned": sector_aligned})
        return result

    # ── Data gathering ─────────────────────────────────────────────────────────

    def _get_macro_context(self) -> dict:
        spy = md.get_quote("SPY")
        qqq = md.get_quote("QQQ")
        iwm = md.get_quote("IWM")
        spy_chg = spy.get("pct_change", 0)
        qqq_chg = qqq.get("pct_change", 0)
        iwm_chg = iwm.get("pct_change", 0)
        return {
            "spy_change":  spy_chg,
            "qqq_change":  qqq_chg,
            "iwm_change":  iwm_chg,
            "green_count": sum([spy_chg > 0, qqq_chg > 0, iwm_chg > 0]),
            "data_ok":     spy.get("price", 0) > 0,
        }

    # ── Pure Python scoring ───────────────────────────────────────────────────

    def _score(self, direction: str, vix: float, macro: dict, sectors: dict,
               vix_trend: str = "flat", primary_etf: str = None,
               news: dict = None) -> tuple[float, dict]:
        score = 5.0
        components: dict = {}

        # ── VIX component ──────────────────────────────────────────────────────
        if vix < 15:
            vix_adj = 1.5;   components["vix"] = f"low ({vix:.1f}) — calm, favor premium buyers"
        elif vix <= 22:
            vix_adj = 0.0;   components["vix"] = f"normal ({vix:.1f}) — no headwind"
        elif vix <= 30:
            vix_adj = -1.0;  components["vix"] = f"elevated ({vix:.1f}) — volatile, watch sizing"
        else:
            vix_adj = -2.5;  components["vix"] = f"extreme ({vix:.1f}) — high fear, skip"

        # For put plays, high VIX is somewhat aligned (fear = bearish)
        if direction == "bearish" and vix > 22:
            vix_adj = max(0.0, vix_adj + 1.0)  # partial credit for bears in fearful market

        score += vix_adj

        # ── Breadth component ──────────────────────────────────────────────────
        green_count = macro.get("green_count", 0) if macro.get("data_ok") else None
        if green_count is not None:
            if direction == "bullish":
                breadth_adj = {0: -1.0, 1: -0.5, 2: 0.5, 3: 1.0}.get(green_count, 0)
            else:
                breadth_adj = {0: 1.0, 1: 0.5, 2: -0.5, 3: -1.0}.get(green_count, 0)
            score += breadth_adj
            components["breadth_green"] = green_count
            components["breadth"] = f"{green_count}/3 indexes green"

        # ── Sector component ──────────────────────────────────────────────────
        sector_aligned = True
        if sectors:
            # If we know the symbol's specific sector ETF, use it directly (±2.0 pts)
            # Otherwise fall back to majority vote across all sectors (±1.5 pts)
            primary_chg = sectors.get(primary_etf) if primary_etf else None
            if primary_chg is not None:
                if direction == "bullish":
                    if   primary_chg > 1.0:  score += 2.0; sector_aligned = True
                    elif primary_chg > 0.0:  score += 1.0; sector_aligned = True
                    elif primary_chg < -1.0: score -= 1.5; sector_aligned = False
                    elif primary_chg < 0.0:  score -= 0.5; sector_aligned = False
                else:  # bearish
                    if   primary_chg < -1.0: score += 2.0; sector_aligned = True
                    elif primary_chg < 0.0:  score += 1.0; sector_aligned = True
                    elif primary_chg > 1.0:  score -= 1.5; sector_aligned = False
                    elif primary_chg > 0.0:  score -= 0.5; sector_aligned = False
                components["sector"] = f"{primary_etf} {primary_chg:+.2f}%"
            else:
                # Fallback: majority of all sector ETFs
                up_sectors   = sum(1 for v in sectors.values() if v > 0)
                down_sectors = sum(1 for v in sectors.values() if v < 0)
                total = up_sectors + down_sectors
                if total > 0:
                    if direction == "bullish":
                        if up_sectors / total >= 0.6:
                            score += 1.5; sector_aligned = True
                            components["sector"] = f"{up_sectors}/{total} sectors up"
                        elif down_sectors / total >= 0.6:
                            score -= 1.0; sector_aligned = False
                            components["sector"] = f"{down_sectors}/{total} sectors down"
                        else:
                            components["sector"] = "mixed sectors"
                    else:  # bearish
                        if down_sectors / total >= 0.6:
                            score += 1.5; sector_aligned = True
                            components["sector"] = f"{down_sectors}/{total} sectors down"
                        elif up_sectors / total >= 0.6:
                            score -= 1.0; sector_aligned = False
                            components["sector"] = f"{up_sectors}/{total} sectors up"
                        else:
                            components["sector"] = "mixed sectors"

        components["sector_aligned"] = sector_aligned

        # ── News sentiment ────────────────────────────────────────────────────
        if news and news.get("available") and news.get("total", 0) >= 3:
            ns = news["score"]   # -1.0 (fully negative) to +1.0 (fully positive)
            if direction == "bullish":
                if   ns > 0.5:  score += 1.5; components["news"] = f"positive ({news['positive']}/{news['total']} articles)"
                elif ns > 0.1:  score += 0.5; components["news"] = f"slightly positive"
                elif ns < -0.5: score -= 1.5; components["news"] = f"negative ({news['negative']}/{news['total']} articles)"
                elif ns < -0.1: score -= 0.5; components["news"] = f"slightly negative"
            else:  # bearish
                if   ns < -0.5: score += 1.5; components["news"] = f"negative ({news['negative']}/{news['total']} articles)"
                elif ns < -0.1: score += 0.5; components["news"] = f"slightly negative"
                elif ns > 0.5:  score -= 1.5; components["news"] = f"positive ({news['positive']}/{news['total']} articles)"
                elif ns > 0.1:  score -= 0.5; components["news"] = f"slightly positive"

        # ── VIX trend ─────────────────────────────────────────────────────────
        if vix_trend == "rising":
            if direction == "bullish":
                score -= 0.5
                components["vix_trend"] = "rising — expanding fear, headwind for calls"
            else:
                score += 0.5
                components["vix_trend"] = "rising — expanding fear, tailwind for puts"
        elif vix_trend == "falling":
            if direction == "bullish":
                score += 0.5
                components["vix_trend"] = "falling — fear contracting, tailwind for calls"
            else:
                score -= 0.5
                components["vix_trend"] = "falling — fear contracting, headwind for puts"

        score = round(max(1.0, min(10.0, score)), 1)
        return score, components
