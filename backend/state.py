"""
State manager — persists system state, proposals, active trades, and exit signals.
"""

import json
import logging
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger(__name__)
STATE_FILE = Path("/app/data/state.json")


def _default() -> dict:
    return {
        "system_status": "stopped",
        "active_trades":  [],
        "proposals":      [],
        "exit_signals":   [],
        "cycle_count":    0,
        "last_scan":      None,
        "last_monitor":   None,
        "event_log":      [],
        "symbol_history": {},   # symbol -> list of last 10 analysis snapshots
    }


class StateManager:
    def __init__(self):
        self._s = self._load()

    def _load(self) -> dict:
        try:
            if STATE_FILE.exists():
                with open(STATE_FILE) as f:
                    data = json.load(f)
                d = _default()
                d.update(data)
                return d
        except Exception as e:
            logger.warning(f"State load failed: {e}")
        return _default()

    def save(self):
        try:
            STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
            with open(STATE_FILE, "w") as f:
                json.dump(self._s, f, indent=2, default=str)
        except Exception as e:
            logger.error(f"State save failed: {e}")

    # ── System ─────────────────────────────────────────────────────────────────

    @property
    def system_status(self) -> str:
        return self._s["system_status"]

    @system_status.setter
    def system_status(self, v: str):
        self._s["system_status"] = v
        self.save()

    @property
    def cycle_count(self) -> int:
        return self._s["cycle_count"]

    def increment_cycle(self):
        self._s["cycle_count"] += 1
        self._s["last_scan"] = datetime.now().isoformat()
        self.save()

    def update_last_monitor(self):
        self._s["last_monitor"] = datetime.now().isoformat()
        self.save()

    def record_symbol_analysis(self, symbol: str, direction: str, analysis: dict, decision: str, score: float):
        """Store per-symbol analysis snapshot so future cycles can see history."""
        if "symbol_history" not in self._s:
            self._s["symbol_history"] = {}
        hist = self._s["symbol_history"].setdefault(symbol, [])
        tech = analysis.get("technical", {})
        sent = analysis.get("sentiment", {})
        fund = analysis.get("fundamental", {})
        snap = {
            "cycle":       self._s["cycle_count"],
            "timestamp":   datetime.now().isoformat(),
            "direction":   direction,
            "decision":    decision,
            "score":       round(score, 1),
            "tech_score":  tech.get("score"),
            "tech_trend":  tech.get("trend"),
            "sent_score":  sent.get("score"),
            "fund_score":  fund.get("score"),
            "vix_regime":  sent.get("vix_regime"),
            "rsi":         tech.get("rsi_reading", "")[:20] if tech.get("rsi_reading") else None,
            "macd":        tech.get("macd_reading"),
            "adv_strength":analysis.get("advocate", {}).get("objection_strength"),
        }
        hist.append(snap)
        self._s["symbol_history"][symbol] = hist[-10:]  # keep last 10 per symbol
        self.save()

    def get_symbol_history(self, symbol: str) -> list:
        return self._s.get("symbol_history", {}).get(symbol, [])

    def get_all_symbol_history(self) -> dict:
        return self._s.get("symbol_history", {})

    def get_full_state(self) -> dict:
        return self._s.copy()

    # ── Proposals ──────────────────────────────────────────────────────────────

    @property
    def proposals(self) -> list[dict]:
        return self._s["proposals"]

    def add_proposal(self, proposal: dict):
        proposal.setdefault("proposal_id", str(uuid.uuid4()))
        proposal["status"] = "pending"
        proposal["proposed_at"] = datetime.now().isoformat()
        self._s["proposals"].append(proposal)
        if len(self._s["proposals"]) > 50:
            self._s["proposals"] = self._s["proposals"][-50:]
        self.save()

    def get_pending_proposals(self) -> list[dict]:
        return [p for p in self._s["proposals"] if p.get("status") == "pending"]

    def has_pending_proposal(self) -> bool:
        return bool(self.get_pending_proposals())

    def resolve_proposal(self, proposal_id: str, action: str, order_info: dict = None):
        """Mark a proposal as executed or rejected."""
        for p in self._s["proposals"]:
            if p.get("proposal_id") == proposal_id:
                p["status"] = action  # "executed" | "rejected"
                p["resolved_at"] = datetime.now().isoformat()
                if order_info:
                    p["order_info"] = order_info
                break
        self.save()

    # ── Active trades (placed by Cowork artifact) ──────────────────────────────

    @property
    def active_trades(self) -> list[dict]:
        return self._s["active_trades"]

    def add_active_trade(self, trade: dict):
        trade["opened_at"] = datetime.now().isoformat()
        self._s["active_trades"].append(trade)
        self.save()

    def close_trade(self, trade_id: str, pnl: float):
        self._s["active_trades"] = [
            t for t in self._s["active_trades"] if t.get("trade_id") != trade_id
        ]
        self.log_event("trade_closed", {"trade_id": trade_id, "pnl": pnl})
        self.save()

    # ── Exit signals ───────────────────────────────────────────────────────────

    @property
    def exit_signals(self) -> list[dict]:
        return self._s["exit_signals"]

    def add_exit_signal(self, signal: dict):
        signal["created_at"] = datetime.now().isoformat()
        signal["status"] = "pending"
        self._s["exit_signals"].append(signal)
        self.save()

    def resolve_exit_signal(self, trade_id: str):
        for s in self._s["exit_signals"]:
            if s.get("trade_id") == trade_id:
                s["status"] = "resolved"
                s["resolved_at"] = datetime.now().isoformat()
        self.save()

    def get_pending_exit_signals(self) -> list[dict]:
        return [s for s in self._s["exit_signals"] if s.get("status") == "pending"]

    # ── Event log ──────────────────────────────────────────────────────────────

    def log_event(self, event_type: str, data: Any = None):
        self._s["event_log"].append({
            "id":        str(uuid.uuid4())[:8],
            "type":      event_type,
            "data":      data,
            "timestamp": datetime.now().isoformat(),
        })
        if len(self._s["event_log"]) > 200:
            self._s["event_log"] = self._s["event_log"][-200:]
        self.save()


_state = StateManager()

def get_state() -> StateManager:
    return _state
