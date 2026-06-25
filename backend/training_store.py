"""
Training data store — writes every scan decision + outcome to PostgreSQL
so the dataset grows with each cycle and can be used to tune the model.

Schema: scan_log
  One row per symbol per cycle. 'trade' rows get outcome filled in
  when the position closes. 'pass' rows stay as-is (negative examples).
"""

import json
import logging
import os
from datetime import datetime

logger = logging.getLogger(__name__)

DATABASE_URL = os.environ.get("DATABASE_URL", "")
_conn = None


def _get_conn():
    global _conn
    if not DATABASE_URL:
        return None
    try:
        if _conn is None or _conn.closed:
            import psycopg2
            _conn = psycopg2.connect(DATABASE_URL)
            _conn.autocommit = True
            _ensure_schema(_conn)
        return _conn
    except Exception as e:
        logger.error(f"TrainingStore connect failed: {e}")
        _conn = None
        return None


def _ensure_schema(conn):
    with conn.cursor() as cur:
        cur.execute("""
            CREATE TABLE IF NOT EXISTS scan_log (
                id                   SERIAL PRIMARY KEY,
                logged_at            TIMESTAMPTZ DEFAULT NOW(),
                cycle_id             INTEGER,
                symbol               VARCHAR(20),
                direction            VARCHAR(20),
                decision             VARCHAR(20),

                -- Agent scores
                weighted_score       FLOAT,
                confidence           INTEGER,
                tech_score           FLOAT,
                fund_score           FLOAT,
                sent_score           FLOAT,

                -- Market snapshot at decision time
                price                FLOAT,
                iv_rank              FLOAT,
                rsi                  FLOAT,
                regime               VARCHAR(20),
                regime_strength      INTEGER,
                vix                  FLOAT,
                breadth              INTEGER,

                -- Technical indicators
                above_ema200         BOOLEAN,
                momentum_60d         FLOAT,
                stoch_k              FLOAT,
                vwap20_pct           FLOAT,
                tech_fatal_flaw      TEXT,

                -- LLM reasoning (for audit / fine-tuning)
                pass_reason          TEXT,
                reasoning            TEXT,
                bull_case            TEXT,
                bear_case            TEXT,

                -- Links to position if a trade was entered
                position_id          VARCHAR(50),

                -- Outcome (filled when position closes)
                outcome              VARCHAR(20),
                outcome_pnl_pct      FLOAT,
                outcome_pnl_dollars  FLOAT,
                outcome_days_held    INTEGER,
                outcome_exit_reason  TEXT,
                outcome_closed_at    TIMESTAMPTZ
            )
        """)
        cur.execute("""
            CREATE INDEX IF NOT EXISTS scan_log_symbol_idx   ON scan_log (symbol);
            CREATE INDEX IF NOT EXISTS scan_log_cycle_idx    ON scan_log (cycle_id);
            CREATE INDEX IF NOT EXISTS scan_log_position_idx ON scan_log (position_id);
        """)


def log_scan_results(cycle: int, scan_summary: list, regime: dict, position_id_map: dict = None):
    """
    Insert one row per symbol in this cycle's scan.
    position_id_map: {symbol -> position_id} for symbols that got entered.
    """
    conn = _get_conn()
    if not conn:
        return

    position_id_map = position_id_map or {}
    reg = regime or {}
    vix = reg.get("vix_level") or reg.get("vix")

    rows = []
    for r in scan_summary:
        sym    = r.get("symbol", "")
        pos_id = position_id_map.get(sym)
        rows.append((
            cycle,
            sym,
            r.get("direction"),
            r.get("decision", "pass"),
            r.get("weighted_score"),
            r.get("confidence"),
            r.get("tech_score"),
            r.get("fund_score"),
            r.get("sent_score"),
            r.get("price"),
            r.get("iv_rank"),
            r.get("rsi"),
            reg.get("regime"),
            reg.get("strength"),
            float(vix) if vix else None,
            reg.get("breadth"),
            r.get("above_ema200"),
            r.get("momentum_60d"),
            r.get("stoch_k"),
            r.get("vwap20_pct"),
            r.get("tech_fatal_flaw"),
            (r.get("pass_reason") or "")[:500],
            (r.get("reasoning") or "")[:1000],
            (r.get("bull_case") or "")[:500],
            (r.get("bear_case") or "")[:500],
            pos_id,
        ))

    try:
        with conn.cursor() as cur:
            cur.executemany("""
                INSERT INTO scan_log (
                    cycle_id, symbol, direction, decision,
                    weighted_score, confidence, tech_score, fund_score, sent_score,
                    price, iv_rank, rsi,
                    regime, regime_strength, vix, breadth,
                    above_ema200, momentum_60d, stoch_k, vwap20_pct, tech_fatal_flaw,
                    pass_reason, reasoning, bull_case, bear_case,
                    position_id
                ) VALUES (%s,%s,%s,%s, %s,%s,%s,%s,%s, %s,%s,%s, %s,%s,%s,%s,
                          %s,%s,%s,%s,%s, %s,%s,%s,%s, %s)
            """, rows)
        logger.info(f"TrainingStore: logged {len(rows)} rows for cycle {cycle}")
    except Exception as e:
        logger.error(f"TrainingStore log_scan_results failed: {e}")


def update_outcome(position_id: str, pnl_pct: float, pnl_dollars: float,
                   days_held: int = None, exit_reason: str = None):
    """Fill in the outcome columns on the row that entered this position."""
    conn = _get_conn()
    if not conn or not position_id:
        return
    outcome = "win" if pnl_dollars > 0 else "loss"
    try:
        with conn.cursor() as cur:
            cur.execute("""
                UPDATE scan_log
                SET outcome             = %s,
                    outcome_pnl_pct     = %s,
                    outcome_pnl_dollars = %s,
                    outcome_days_held   = %s,
                    outcome_exit_reason = %s,
                    outcome_closed_at   = NOW()
                WHERE position_id = %s
            """, (outcome, pnl_pct, pnl_dollars, days_held,
                  (exit_reason or "")[:300], position_id))
        logger.info(f"TrainingStore: outcome updated for position {position_id[:8]} → {outcome} {pnl_pct:+.1f}%")
    except Exception as e:
        logger.error(f"TrainingStore update_outcome failed: {e}")


def get_recent(limit: int = 200) -> list:
    """Return the most recent scan_log rows as dicts (for the API)."""
    conn = _get_conn()
    if not conn:
        return []
    try:
        with conn.cursor() as cur:
            cur.execute("""
                SELECT id, logged_at, cycle_id, symbol, direction, decision,
                       weighted_score, confidence, tech_score, fund_score, sent_score,
                       price, iv_rank, rsi, regime, regime_strength, vix, breadth,
                       above_ema200, momentum_60d, stoch_k, vwap20_pct,
                       pass_reason, reasoning, bull_case, bear_case,
                       position_id, outcome, outcome_pnl_pct, outcome_pnl_dollars,
                       outcome_days_held, outcome_exit_reason, outcome_closed_at
                FROM scan_log
                ORDER BY logged_at DESC
                LIMIT %s
            """, (limit,))
            cols = [d[0] for d in cur.description]
            rows = []
            for row in cur.fetchall():
                d = dict(zip(cols, row))
                # Convert datetime to ISO string for JSON
                for k, v in d.items():
                    if isinstance(v, datetime):
                        d[k] = v.isoformat()
                rows.append(d)
            return rows
    except Exception as e:
        logger.error(f"TrainingStore get_recent failed: {e}")
        return []


def get_stats() -> dict:
    """Aggregate stats over all training data (for dashboard display)."""
    conn = _get_conn()
    if not conn:
        return {}
    try:
        with conn.cursor() as cur:
            cur.execute("""
                SELECT
                    COUNT(*)                                         AS total_decisions,
                    COUNT(*) FILTER (WHERE decision='trade')        AS trades_entered,
                    COUNT(*) FILTER (WHERE outcome='win')           AS wins,
                    COUNT(*) FILTER (WHERE outcome='loss')          AS losses,
                    AVG(outcome_pnl_pct) FILTER (WHERE outcome='win')  AS avg_win_pct,
                    AVG(outcome_pnl_pct) FILTER (WHERE outcome='loss') AS avg_loss_pct,
                    COUNT(DISTINCT cycle_id)                        AS cycles_logged,
                    COUNT(DISTINCT symbol)                          AS symbols_tracked
                FROM scan_log
            """)
            row = cur.fetchone()
            if not row:
                return {}
            cols = [d[0] for d in cur.description]
            return {k: (float(v) if v is not None else None)
                    for k, v in zip(cols, row)}
    except Exception as e:
        logger.error(f"TrainingStore get_stats failed: {e}")
        return {}
