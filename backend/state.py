"""
State manager — persists system state to PostgreSQL (DATABASE_URL) with
a local JSON file fallback for development environments.
"""

import json
import logging
import os
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger(__name__)

# ── Storage backends ──────────────────────────────────────────────────────────

DATABASE_URL = os.environ.get("DATABASE_URL", "")
STATE_FILE   = Path("/app/data/state.json")

# psycopg2 is optional — only needed when DATABASE_URL is set
_pg_conn = None

def _get_conn():
    global _pg_conn
    try:
        if _pg_conn is None or _pg_conn.closed:
            import psycopg2
            import psycopg2.extras
            _pg_conn = psycopg2.connect(DATABASE_URL)
            _pg_conn.autocommit = True
            _ensure_table(_pg_conn)
        return _pg_conn
    except Exception as e:
        logger.error(f"DB connect failed: {e}")
        _pg_conn = None
        return None

def _ensure_table(conn):
    with conn.cursor() as cur:
        cur.execute("""
            CREATE TABLE IF NOT EXISTS state_store (
                id      INTEGER PRIMARY KEY DEFAULT 1,
                data    JSONB NOT NULL,
                updated_at TIMESTAMPTZ DEFAULT NOW()
            )
        """)

def _db_load() -> Optional[dict]:
    conn = _get_conn()
    if not conn:
        return None
    try:
        with conn.cursor() as cur:
            cur.execute("SELECT data FROM state_store WHERE id = 1")
            row = cur.fetchone()
            return row[0] if row else None
    except Exception as e:
        logger.error(f"DB load failed: {e}")
        return None

def _db_save(data: dict):
    conn = _get_conn()
    if not conn:
        return False
    try:
        import psycopg2.extras
        with conn.cursor() as cur:
            cur.execute("""
                INSERT INTO state_store (id, data, updated_at)
                VALUES (1, %s, NOW())
                ON CONFLICT (id) DO UPDATE
                    SET data = EXCLUDED.data,
                        updated_at = NOW()
            """, (json.dumps(data, default=str),))
        return True
    except Exception as e:
        logger.error(f"DB save failed: {e}")
        # Reset connection so next call reconnects
        global _pg_conn
        _pg_conn = None
        return False

def _file_load() -> Optional[dict]:
    try:
        if STATE_FILE.exists():
            with open(STATE_FILE) as f:
                return json.load(f)
    except Exception as e:
        logger.warning(f"File load failed: {e}")
    return None

def _file_save(data: dict):
    try:
        STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
        with open(STATE_FILE, "w") as f:
            json.dump(data, f, indent=2, default=str)
    except Exception as e:
        logger.error(f"File save failed: {e}")

# ── Default state ─────────────────────────────────────────────────────────────

def _default() -> dict:
    return {
        "system_status":           "stopped",
        "active_trades":           [],
        "proposals":               [],
        "exit_signals":            [],
        "cycle_count":             0,
        "last_scan":               None,
        "last_monitor":            None,
        "event_log":               [],
        "symbol_history":          {},
        "market_regime":           {},
        "premarket_context":       {},
        "last_premarket_prep":     None,
        "last_afterhours_capture": None,
        "last_scan_results":       [],
        "last_scan_cycle":         0,
        "sim_positions":           [],
        "pnl_history":             [],
    }

# ── State manager ─────────────────────────────────────────────────────────────

class StateManager:
    def __init__(self):
        self._use_db = bool(DATABASE_URL)
        self._s = self._load()
        logger.info(f"StateManager init: backend={'postgres' if self._use_db else 'file'}")

    def _load(self) -> dict:
        raw = _db_load() if self._use_db else _file_load()
        if raw:
            d = _default()
            d.update(raw)
            return d
        return _default()

    def save(self):
        if self._use_db:
            ok = _db_save(self._s)
            if not ok:
                # fallback write to file so we don't lose data
                _file_save(self._s)
        else:
            _file_save(self._s)

    # ── System ────────────────────────────────────────────────────────────────

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
        }
        hist.append(snap)
        self._s["symbol_history"][symbol] = hist[-10:]
        self.save()

    def get_symbol_history(self, symbol: str) -> list:
        return self._s.get("symbol_history", {}).get(symbol, [])

    def get_all_symbol_history(self) -> dict:
        return self._s.get("symbol_history", {})

    def get_full_state(self) -> dict:
        return self._s.copy()

    # ── Proposals ─────────────────────────────────────────────────────────────

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
        for p in self._s["proposals"]:
            if p.get("proposal_id") == proposal_id:
                p["status"] = action
                p["resolved_at"] = datetime.now().isoformat()
                if order_info:
                    p["order_info"] = order_info
                break
        self.save()

    # ── Active trades ──────────────────────────────────────────────────────────

    @property
    def active_trades(self) -> list[dict]:
        return self._s["active_trades"]

    def add_active_trade(self, trade: dict):
        trade["opened_at"] = datetime.now().isoformat()
        self._s["active_trades"].append(trade)
        self.save()

    def update_trade(self, trade_id: str, updates: dict):
        for t in self._s["active_trades"]:
            if t.get("trade_id") == trade_id:
                t.update(updates)
                break
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

    # ── Market regime ──────────────────────────────────────────────────────────

    @property
    def market_regime(self) -> dict:
        return self._s.get("market_regime", {})

    @market_regime.setter
    def market_regime(self, v: dict):
        self._s["market_regime"] = v
        self.save()

    @property
    def premarket_context(self) -> dict:
        return self._s.get("premarket_context", {})

    @premarket_context.setter
    def premarket_context(self, v: dict):
        self._s["premarket_context"] = v
        self.save()

    def mark_premarket_done(self):
        self._s["last_premarket_prep"] = datetime.now().isoformat()
        self.save()

    def mark_afterhours_done(self):
        self._s["last_afterhours_capture"] = datetime.now().isoformat()
        self.save()

    def get_last_premarket_date(self) -> Optional[str]:
        ts = self._s.get("last_premarket_prep")
        return ts[:10] if ts else None

    def get_last_afterhours_date(self) -> Optional[str]:
        ts = self._s.get("last_afterhours_capture")
        return ts[:10] if ts else None

    def store_scan_results(self, results: list, cycle: int):
        self._s["last_scan_results"] = results
        self._s["last_scan_cycle"] = cycle
        self.save()

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

    # ── Sim positions ──────────────────────────────────────────────────────────

    def add_sim_position(self, pos: dict):
        pos.setdefault("position_id", str(uuid.uuid4()))
        pos["status"] = "open"
        self._s.setdefault("sim_positions", []).append(pos)
        if len(self._s["sim_positions"]) > 200:
            self._s["sim_positions"] = self._s["sim_positions"][-200:]
        self.save()

    def get_sim_positions(self, status: str = None) -> list:
        positions = self._s.get("sim_positions", [])
        if status:
            return [p for p in positions if p.get("status") == status]
        return list(positions)

    def update_sim_position(self, pos_id: str, updates: dict):
        for p in self._s.get("sim_positions", []):
            if p.get("position_id") == pos_id:
                p.update(updates)
                break
        self.save()

    def close_sim_position(self, pos_id: str, exit_data: dict):
        pnl_dollars = exit_data.get("pnl_dollars", 0.0)
        symbol, direction, pnl_pct = "", "", 0.0
        for p in self._s.get("sim_positions", []):
            if p.get("position_id") == pos_id:
                symbol    = p.get("symbol", "")
                direction = p.get("direction", "")
                p.update(exit_data)
                p["status"]    = "closed"
                p["closed_at"] = datetime.now().isoformat()
                pnl_pct = float(p.get("pnl_pct", 0.0))
                break

        cumulative = self.cumulative_sim_pnl()
        self._s.setdefault("pnl_history", []).append({
            "timestamp":      datetime.now().isoformat(),
            "cumulative_pnl": round(cumulative, 2),
            "trade_pnl":      round(pnl_dollars, 2),
            "trade_pnl_pct":  round(pnl_pct, 2),
            "outcome":        "win" if pnl_dollars > 0 else "loss",
            "position_id":    pos_id,
            "symbol":         symbol,
            "direction":      direction,
        })
        if len(self._s["pnl_history"]) > 500:
            self._s["pnl_history"] = self._s["pnl_history"][-500:]
        self.save()

    def cumulative_sim_pnl(self) -> float:
        closed = [p for p in self._s.get("sim_positions", []) if p.get("status") == "closed"]
        return round(sum(float(p.get("pnl_dollars", 0)) for p in closed), 2)

    def get_pnl_history(self) -> list:
        return list(self._s.get("pnl_history", []))

    def reset_sim(self) -> dict:
        """Clear sim positions and P&L history. Training DB is untouched."""
        cleared_open   = len([p for p in self._s.get("sim_positions", []) if p.get("status") == "open"])
        cleared_closed = len([p for p in self._s.get("sim_positions", []) if p.get("status") == "closed"])
        self._s["sim_positions"] = []
        self._s["pnl_history"]   = []
        self.save()
        return {"cleared_open": cleared_open, "cleared_closed": cleared_closed}


_state = StateManager()

def get_state() -> StateManager:
    return _state
