"""
Outcome tracker — records every trade's signal values and realized PnL.

Enables:
  1. Kelly criterion position sizing (based on actual win rate / avg win / avg loss)
  2. Similar-setup stats fed into Judge context so Claude can see historical hit rate
  3. Future: weight recalibration based on which signals actually predict wins
"""

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)
OUTCOMES_FILE = Path("/app/data/outcomes.json")
MIN_TRADES_FOR_KELLY = 10   # need at least this many closed trades before using Kelly


def _default() -> dict:
    return {"trades": [], "stats": {}}


class OutcomeTracker:
    def __init__(self):
        self._data = self._load()

    def _load(self) -> dict:
        try:
            if OUTCOMES_FILE.exists():
                with open(OUTCOMES_FILE) as f:
                    d = json.load(f)
                    d.setdefault("trades", [])
                    d.setdefault("stats", {})
                    return d
        except Exception as e:
            logger.warning(f"OutcomeTracker load failed: {e}")
        return _default()

    def save(self):
        try:
            OUTCOMES_FILE.parent.mkdir(parents=True, exist_ok=True)
            with open(OUTCOMES_FILE, "w") as f:
                json.dump(self._data, f, indent=2, default=str)
        except Exception as e:
            logger.error(f"OutcomeTracker save failed: {e}")

    # ── Entry ──────────────────────────────────────────────────────────────────

    def record_entry(self, trade_id: str, proposal: dict, signals: dict):
        """Record signal snapshot at trade entry."""
        entry = {
            "trade_id":       trade_id,
            "symbol":         proposal.get("symbol"),
            "direction":      proposal.get("direction"),
            "option_type":    proposal.get("option_type"),
            "iv_rank":        signals.get("iv_rank"),
            "tech_score":     signals.get("tech_score"),
            "sent_score":     signals.get("sent_score"),
            "fund_score":     signals.get("fund_score"),
            "weighted_score": signals.get("weighted_score"),
            "confidence":     signals.get("confidence"),
            "entered_at":     datetime.now().isoformat(),
            "outcome":        None,
            "pnl_pct":        None,
            "pnl_total":      None,
            "closed_at":      None,
        }
        self._data["trades"].append(entry)
        if len(self._data["trades"]) > 500:
            self._data["trades"] = self._data["trades"][-500:]
        self.save()

    # ── Outcome ────────────────────────────────────────────────────────────────

    def record_outcome(self, trade_id: str, pnl_pct: float, pnl_total: float):
        """Record realized PnL when a trade closes."""
        for t in self._data["trades"]:
            if t["trade_id"] == trade_id:
                t["outcome"]   = "win" if pnl_pct > 0 else "loss"
                t["pnl_pct"]   = round(pnl_pct, 2)
                t["pnl_total"] = round(pnl_total, 2)
                t["closed_at"] = datetime.now().isoformat()
                break
        self._refresh_stats()
        self.save()

    # ── Stats ──────────────────────────────────────────────────────────────────

    def _refresh_stats(self):
        closed = [t for t in self._data["trades"] if t.get("outcome")]
        if len(closed) < MIN_TRADES_FOR_KELLY:
            self._data["stats"] = {"total_trades": len(closed), "kelly_ready": False}
            self.save()
            return

        wins   = [t for t in closed if t["outcome"] == "win"]
        losses = [t for t in closed if t["outcome"] == "loss"]

        win_rate = len(wins) / len(closed)
        avg_win  = (sum(t["pnl_pct"] for t in wins) / len(wins)) if wins else 0.0
        avg_loss = abs((sum(t["pnl_pct"] for t in losses) / len(losses))) if losses else 1.0

        # Kelly: f* = (b*p - q) / b   where b=avg_win/avg_loss, p=win_rate, q=1-p
        b = avg_win / avg_loss if avg_loss > 0 else 1.0
        q = 1.0 - win_rate
        kelly_raw = (b * win_rate - q) / b if b > 0 else 0.0
        kelly = max(0.0, min(0.25, kelly_raw))  # fractional Kelly, cap at 25%

        self._data["stats"] = {
            "total_trades":   len(closed),
            "win_rate":       round(win_rate, 3),
            "avg_win_pct":    round(avg_win, 2),
            "avg_loss_pct":   round(avg_loss, 2),
            "kelly_fraction": round(kelly, 4),
            "expectancy":     round(win_rate * avg_win - (1 - win_rate) * avg_loss, 2),
            "kelly_ready":    True,
            "updated_at":     datetime.now().isoformat(),
        }

    def get_stats(self) -> dict:
        return self._data.get("stats", {})

    def get_kelly_fraction(self) -> float:
        """Returns Kelly fraction 0.0-0.25. Returns 0.0 if not enough data yet."""
        return float(self._data.get("stats", {}).get("kelly_fraction", 0.0))

    def is_kelly_ready(self) -> bool:
        return bool(self._data.get("stats", {}).get("kelly_ready", False))

    # ── Similar setups ─────────────────────────────────────────────────────────

    def get_symbol_stats(self, symbol: str) -> Optional[dict]:
        """
        Win rate + avg PnL for a specific symbol.
        Returns None if fewer than 3 closed trades for this symbol.
        """
        trades = [
            t for t in self._data["trades"]
            if t.get("outcome") and t.get("symbol") == symbol
        ]
        if len(trades) < 3:
            return None
        wins = [t for t in trades if t["outcome"] == "win"]
        return {
            "symbol":      symbol,
            "trade_count": len(trades),
            "win_rate":    round(len(wins) / len(trades), 3),
            "avg_pnl":     round(sum(t["pnl_pct"] for t in trades) / len(trades), 2),
        }

    def get_all_symbol_stats(self) -> dict:
        """Return symbol stats for all symbols with >= 3 closed trades."""
        symbols = {t.get("symbol") for t in self._data["trades"] if t.get("outcome") and t.get("symbol")}
        result = {}
        for sym in symbols:
            s = self.get_symbol_stats(sym)
            if s:
                result[sym] = s
        return result

    def record_sim_close(self, pos: dict):
        """
        Record a closed sim position in the outcome tracker so the learning
        loop accumulates real win-rate data from simulated trades.
        """
        pnl_dollars = float(pos.get("pnl_dollars", 0))
        pnl_pct     = float(pos.get("pnl_pct", 0))
        outcome     = "win" if pnl_dollars > 0 else "loss"
        entry = {
            "trade_id":       pos.get("position_id", ""),
            "symbol":         pos.get("symbol", ""),
            "direction":      pos.get("direction", ""),
            "option_type":    pos.get("option_type", ""),
            "iv_rank":        float(pos.get("iv_rank", 50)),
            "tech_score":     float(pos.get("tech_score", 5)),
            "sent_score":     float(pos.get("sent_score", 5)),
            "fund_score":     float(pos.get("fund_score", 5)),
            "weighted_score": float(pos.get("weighted_score", 0)),
            "confidence":     float(pos.get("confidence", 5)),
            "entered_at":     pos.get("opened_at", datetime.now().isoformat()),
            "closed_at":      pos.get("closed_at", datetime.now().isoformat()),
            "outcome":        outcome,
            "pnl_pct":        round(pnl_pct, 2),
            "pnl_total":      round(pnl_dollars, 2),
            "exit_reason":    pos.get("exit_reason", ""),
            "days_held":      pos.get("days_held", 0),
            "source":         "sim",
        }
        # Avoid duplicates
        existing_ids = {t.get("trade_id") for t in self._data["trades"]}
        if entry["trade_id"] and entry["trade_id"] in existing_ids:
            return
        self._data["trades"].append(entry)
        if len(self._data["trades"]) > 500:
            self._data["trades"] = self._data["trades"][-500:]
        self._refresh_stats()
        self.save()

    def get_similar_setups(
        self, iv_rank: float, direction: str, min_count: int = 3
    ) -> Optional[dict]:
        """
        Find closed trades with similar IV rank (±20) and same direction.
        Returns win rate + avg PnL for Judge context.
        """
        similar = [
            t for t in self._data["trades"]
            if t.get("outcome")
            and t.get("direction") == direction
            and t.get("iv_rank") is not None
            and abs(t["iv_rank"] - iv_rank) <= 20
        ]
        if len(similar) < min_count:
            return None

        wins = [t for t in similar if t["outcome"] == "win"]
        return {
            "count":    len(similar),
            "win_rate": round(len(wins) / len(similar), 3),
            "avg_pnl":  round(sum(t["pnl_pct"] for t in similar) / len(similar), 2),
        }


_tracker = OutcomeTracker()


def get_outcome_tracker() -> OutcomeTracker:
    return _tracker
