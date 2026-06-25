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
                above_ema20          BOOLEAN,
                above_ema50          BOOLEAN,
                adx                  FLOAT,
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
        # Migrate: add new columns if they don't exist yet (idempotent ALTER TABLE)
        for col_ddl in [
            "ALTER TABLE scan_log ADD COLUMN IF NOT EXISTS above_ema20 BOOLEAN",
            "ALTER TABLE scan_log ADD COLUMN IF NOT EXISTS above_ema50 BOOLEAN",
            "ALTER TABLE scan_log ADD COLUMN IF NOT EXISTS adx FLOAT",
        ]:
            try:
                cur.execute(col_ddl)
            except Exception:
                pass


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
            r.get("above_ema20"),
            r.get("above_ema50"),
            r.get("adx"),
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
                    above_ema200, above_ema20, above_ema50, adx,
                    momentum_60d, stoch_k, vwap20_pct, tech_fatal_flaw,
                    pass_reason, reasoning, bull_case, bear_case,
                    position_id
                ) VALUES (%s,%s,%s,%s, %s,%s,%s,%s,%s, %s,%s,%s, %s,%s,%s,%s,
                          %s,%s,%s,%s, %s,%s,%s,%s, %s,%s,%s,%s, %s)
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


def get_best_patterns(min_samples: int = 3) -> list:
    """
    Find feature-combination patterns that historically predict winning trades.
    Returns a list of {description, win_rate, n} dicts sorted by win_rate desc.
    Used to inject high-confidence pattern context into the Judge.
    """
    conn = _get_conn()
    if not conn:
        return []
    patterns = []
    try:
        with conn.cursor() as cur:
            # Pattern 1: tech_score + regime
            cur.execute("""
                SELECT CONCAT('tech≥',
                    CASE WHEN tech_score >= 8 THEN '8' WHEN tech_score >= 7 THEN '7' ELSE '<7' END,
                    ' + ', COALESCE(regime,'unknown')) AS pat,
                    COUNT(*) AS n,
                    ROUND(100.0 * COUNT(*) FILTER (WHERE outcome='win') / COUNT(*), 0) AS wr
                FROM scan_log
                WHERE decision='trade' AND outcome IS NOT NULL AND tech_score IS NOT NULL
                GROUP BY 1 HAVING COUNT(*) >= %s
                ORDER BY wr DESC LIMIT 10
            """, (min_samples,))
            for row in cur.fetchall():
                patterns.append({"desc": row[0], "n": int(row[1]), "win_rate": float(row[2])})

            # Pattern 2: iv_rank bucket + above_ema200
            cur.execute("""
                SELECT CONCAT(
                    CASE WHEN iv_rank < 25 THEN 'IV<25' WHEN iv_rank < 40 THEN 'IV<40' ELSE 'IV≥40' END,
                    ' + ', CASE WHEN above_ema200 THEN 'above EMA200' ELSE 'below EMA200' END
                ) AS pat,
                    COUNT(*) AS n,
                    ROUND(100.0 * COUNT(*) FILTER (WHERE outcome='win') / COUNT(*), 0) AS wr
                FROM scan_log
                WHERE decision='trade' AND outcome IS NOT NULL AND iv_rank IS NOT NULL AND above_ema200 IS NOT NULL
                GROUP BY 1 HAVING COUNT(*) >= %s
                ORDER BY wr DESC LIMIT 10
            """, (min_samples,))
            for row in cur.fetchall():
                patterns.append({"desc": row[0], "n": int(row[1]), "win_rate": float(row[2])})

            # Pattern 3: confidence bucket
            cur.execute("""
                SELECT CONCAT('conf=',
                    CASE WHEN confidence >= 8 THEN '8+' WHEN confidence >= 6 THEN '6-7' ELSE '≤5' END,
                    ' + ', direction) AS pat,
                    COUNT(*) AS n,
                    ROUND(100.0 * COUNT(*) FILTER (WHERE outcome='win') / COUNT(*), 0) AS wr
                FROM scan_log
                WHERE decision='trade' AND outcome IS NOT NULL AND confidence IS NOT NULL
                GROUP BY 1 HAVING COUNT(*) >= %s
                ORDER BY wr DESC LIMIT 10
            """, (min_samples,))
            for row in cur.fetchall():
                patterns.append({"desc": row[0], "n": int(row[1]), "win_rate": float(row[2])})

            # Pattern 4: ADX trending vs choppy
            cur.execute("""
                SELECT CONCAT(
                    CASE WHEN adx >= 25 THEN 'ADX≥25(trending)'
                         WHEN adx < 18  THEN 'ADX<18(choppy)'
                         ELSE 'ADX neutral' END,
                    ' + ', direction) AS pat,
                    COUNT(*) AS n,
                    ROUND(100.0 * COUNT(*) FILTER (WHERE outcome='win') / COUNT(*), 0) AS wr
                FROM scan_log
                WHERE decision='trade' AND outcome IS NOT NULL AND adx IS NOT NULL
                GROUP BY 1 HAVING COUNT(*) >= %s
                ORDER BY wr DESC LIMIT 8
            """, (min_samples,))
            for row in cur.fetchall():
                patterns.append({"desc": row[0], "n": int(row[1]), "win_rate": float(row[2])})

            # Pattern 5: EMA20/50 alignment
            cur.execute("""
                SELECT CONCAT(
                    CASE WHEN above_ema20 AND above_ema50 THEN 'above both EMAs'
                         WHEN NOT above_ema20 AND NOT above_ema50 THEN 'below both EMAs'
                         WHEN above_ema20 THEN 'above EMA20 only'
                         ELSE 'below EMA20 only' END,
                    ' + ', direction) AS pat,
                    COUNT(*) AS n,
                    ROUND(100.0 * COUNT(*) FILTER (WHERE outcome='win') / COUNT(*), 0) AS wr
                FROM scan_log
                WHERE decision='trade' AND outcome IS NOT NULL
                  AND above_ema20 IS NOT NULL AND above_ema50 IS NOT NULL
                GROUP BY 1 HAVING COUNT(*) >= %s
                ORDER BY wr DESC LIMIT 8
            """, (min_samples,))
            for row in cur.fetchall():
                patterns.append({"desc": row[0], "n": int(row[1]), "win_rate": float(row[2])})

    except Exception as e:
        logger.warning(f"TrainingStore get_best_patterns failed: {e}")

    # Deduplicate and return top patterns sorted by win_rate
    seen = set()
    result = []
    for p in sorted(patterns, key=lambda x: x["win_rate"], reverse=True):
        k = p["desc"]
        if k not in seen:
            seen.add(k)
            result.append(p)
    return result[:10]


def get_learned_context(min_samples: int = 5) -> str:
    """
    Query historical outcomes and return a compact calibration summary
    for the Judge agent's system prompt. Returns empty string if insufficient data.
    """
    conn = _get_conn()
    if not conn:
        return ""
    try:
        with conn.cursor() as cur:
            # Overall win rate on entered trades
            cur.execute("""
                SELECT COUNT(*) AS n, COUNT(*) FILTER (WHERE outcome='win') AS wins,
                       AVG(outcome_pnl_pct) FILTER (WHERE outcome='win') AS avg_win,
                       AVG(outcome_pnl_pct) FILTER (WHERE outcome='loss') AS avg_loss
                FROM scan_log WHERE decision='trade' AND outcome IS NOT NULL
            """)
            row = cur.fetchone()
            if not row or not row[0] or row[0] < min_samples:
                return ""
            n, wins, avg_win, avg_loss = row
            wr = wins / n * 100 if n else 0

            lines = [
                f"HISTORICAL PERFORMANCE ({n} closed trades):",
                f"  Win rate: {wr:.0f}%  |  Avg win: {avg_win:+.1f}%  |  Avg loss: {avg_loss:+.1f}%",
            ]

            # Win rate by regime
            cur.execute("""
                SELECT regime,
                       COUNT(*) AS n,
                       ROUND(100.0 * COUNT(*) FILTER (WHERE outcome='win') / COUNT(*), 0) AS wr
                FROM scan_log
                WHERE decision='trade' AND outcome IS NOT NULL AND regime IS NOT NULL
                GROUP BY regime HAVING COUNT(*) >= 3
                ORDER BY wr DESC
            """)
            regime_rows = cur.fetchall()
            if regime_rows:
                lines.append("  Win rate by regime: " +
                    " | ".join(f"{r[0]} {int(r[2])}% ({r[1]}T)" for r in regime_rows))

            # Win rate by direction
            cur.execute("""
                SELECT direction,
                       COUNT(*) AS n,
                       ROUND(100.0 * COUNT(*) FILTER (WHERE outcome='win') / COUNT(*), 0) AS wr
                FROM scan_log
                WHERE decision='trade' AND outcome IS NOT NULL
                GROUP BY direction HAVING COUNT(*) >= 3
            """)
            dir_rows = cur.fetchall()
            if dir_rows:
                lines.append("  Win rate by direction: " +
                    " | ".join(f"{r[0]} {int(r[2])}% ({r[1]}T)" for r in dir_rows))

            # Win rate by score bucket
            cur.execute("""
                SELECT
                    CASE WHEN weighted_score >= 55 THEN 'score≥55'
                         WHEN weighted_score >= 50 THEN 'score 50-55'
                         ELSE 'score<50' END AS bucket,
                    COUNT(*) AS n,
                    ROUND(100.0 * COUNT(*) FILTER (WHERE outcome='win') / COUNT(*), 0) AS wr
                FROM scan_log
                WHERE decision='trade' AND outcome IS NOT NULL AND weighted_score IS NOT NULL
                GROUP BY 1 HAVING COUNT(*) >= 3
                ORDER BY MIN(weighted_score) DESC
            """)
            score_rows = cur.fetchall()
            if score_rows:
                lines.append("  Win rate by score: " +
                    " | ".join(f"{r[0]}: {int(r[2])}% ({r[1]}T)" for r in score_rows))

            # Per-symbol performance
            cur.execute("""
                SELECT symbol,
                       COUNT(*) AS n,
                       ROUND(100.0 * COUNT(*) FILTER (WHERE outcome='win') / COUNT(*), 0) AS wr,
                       ROUND(AVG(outcome_pnl_pct)::NUMERIC, 1) AS avg_pnl
                FROM scan_log
                WHERE decision='trade' AND outcome IS NOT NULL
                GROUP BY symbol HAVING COUNT(*) >= 2
                ORDER BY wr DESC
            """)
            sym_rows = cur.fetchall()
            if sym_rows:
                lines.append("  Per-symbol history: " +
                    " | ".join(f"{r[0]} {int(r[2])}%WR avg{r[3]:+.0f}% ({r[1]}T)" for r in sym_rows))

            # Confidence calibration
            cur.execute("""
                SELECT
                    CASE WHEN confidence >= 7 THEN 'conf≥7'
                         WHEN confidence >= 5 THEN 'conf 5-6'
                         ELSE 'conf<5' END AS bucket,
                    COUNT(*) AS n,
                    ROUND(100.0 * COUNT(*) FILTER (WHERE outcome='win') / COUNT(*), 0) AS wr
                FROM scan_log
                WHERE decision='trade' AND outcome IS NOT NULL AND confidence IS NOT NULL
                GROUP BY 1 HAVING COUNT(*) >= 2
            """)
            conf_rows = cur.fetchall()
            if conf_rows:
                lines.append("  Win rate by confidence: " +
                    " | ".join(f"{r[0]}: {int(r[2])}% ({r[1]}T)" for r in conf_rows))

            # Days held by outcome — tells us optimal hold duration
            cur.execute("""
                SELECT outcome,
                       ROUND(AVG(outcome_days_held), 1) AS avg_days,
                       ROUND(MIN(outcome_days_held), 0) AS min_days,
                       ROUND(MAX(outcome_days_held), 0) AS max_days,
                       COUNT(*) AS n
                FROM scan_log
                WHERE decision='trade' AND outcome IS NOT NULL
                  AND outcome_days_held IS NOT NULL AND outcome_days_held > 0
                GROUP BY outcome
            """)
            hold_rows = cur.fetchall()
            if hold_rows:
                hold_parts = [f"{r[0]}: avg {r[1]}d (range {r[2]}-{r[3]}d, {r[4]}T)"
                              for r in hold_rows]
                lines.append("  Hold duration: " + " | ".join(hold_parts))

            # Recent trend: last 10 closed trades vs overall
            cur.execute("""
                SELECT COUNT(*) AS n,
                       COUNT(*) FILTER (WHERE outcome='win') AS wins
                FROM (
                    SELECT outcome FROM scan_log
                    WHERE decision='trade' AND outcome IS NOT NULL
                    ORDER BY logged_at DESC LIMIT 10
                ) sub
            """)
            rec = cur.fetchone()
            if rec and rec[0] >= 5:
                rec_wr = rec[1] / rec[0] * 100
                trend_note = "↑improving" if rec_wr > wr else ("↓declining" if rec_wr < wr - 10 else "→stable")
                lines.append(f"  Recent (last {rec[0]}T): {rec_wr:.0f}% WR {trend_note} vs {wr:.0f}% overall")

            # Regime + direction combos
            cur.execute("""
                SELECT CONCAT(COALESCE(regime,'?'), '+', direction) AS combo,
                       COUNT(*) AS n,
                       ROUND(100.0 * COUNT(*) FILTER (WHERE outcome='win') / COUNT(*), 0) AS wr
                FROM scan_log
                WHERE decision='trade' AND outcome IS NOT NULL AND regime IS NOT NULL
                GROUP BY 1 HAVING COUNT(*) >= 2
                ORDER BY wr DESC LIMIT 8
            """)
            combo_rows = cur.fetchall()
            if combo_rows:
                lines.append("  Regime+direction combos: " +
                    " | ".join(f"{r[0]} {int(r[2])}% ({r[1]}T)" for r in combo_rows))

            # Top predictive patterns
            patterns = get_best_patterns(min_samples=3)
            good = [p for p in patterns if p["win_rate"] >= 60 and p["n"] >= 3]
            bad  = [p for p in patterns if p["win_rate"] <= 35 and p["n"] >= 3]
            if good:
                lines.append("  TOP WIN patterns: " +
                    " | ".join(f"{p['desc']} {int(p['win_rate'])}%WR ({p['n']}T)" for p in good[:4]))
            if bad:
                lines.append("  LOW WIN patterns: " +
                    " | ".join(f"{p['desc']} {int(p['win_rate'])}%WR ({p['n']}T)" for p in bad[:4]))

            lines.append("CALIBRATION RULE: Reduce confidence by 1-2 for any regime/symbol/pattern matching LOW WIN conditions. Increase by 1 for TOP WIN matches.")
            return "\n".join(lines)

    except Exception as e:
        logger.warning(f"TrainingStore get_learned_context failed: {e}")
        return ""


def get_symbol_perf(min_trades: int = 2) -> dict:
    """
    Return per-symbol performance stats in the same shape as OutcomeTracker.get_all_symbol_stats().
    Used to seed the Scanner's symbol_performance with persistent DB data.
    """
    conn = _get_conn()
    if not conn:
        return {}
    try:
        with conn.cursor() as cur:
            cur.execute("""
                SELECT symbol,
                       COUNT(*) AS n,
                       COUNT(*) FILTER (WHERE outcome='win') AS wins,
                       AVG(outcome_pnl_pct) AS avg_pnl,
                       AVG(outcome_pnl_pct) FILTER (WHERE outcome='win')  AS avg_win,
                       AVG(outcome_pnl_pct) FILTER (WHERE outcome='loss') AS avg_loss
                FROM scan_log
                WHERE decision='trade' AND outcome IS NOT NULL
                GROUP BY symbol
                HAVING COUNT(*) >= %s
            """, (min_trades,))
            result = {}
            for row in cur.fetchall():
                sym, n, wins, avg_pnl, avg_win, avg_loss = row
                result[sym] = {
                    "trade_count": int(n),
                    "win_rate":    float(wins) / int(n) if n else 0.0,
                    "avg_pnl":     float(avg_pnl) if avg_pnl else 0.0,
                    "avg_win":     float(avg_win) if avg_win else 0.0,
                    "avg_loss":    float(avg_loss) if avg_loss else 0.0,
                }
            return result
    except Exception as e:
        logger.warning(f"TrainingStore get_symbol_perf failed: {e}")
        return {}


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
