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

                -- Multi-agent consensus (0-3: how many of tech/fund/sent scored >= 7)
                consensus_score      INTEGER,

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
            "ALTER TABLE scan_log ADD COLUMN IF NOT EXISTS consensus_score INTEGER",
            "ALTER TABLE scan_log ADD COLUMN IF NOT EXISTS vol_ratio FLOAT",
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
        # Consensus: how many of tech/fund/sent scored >= 7
        t_s = r.get("tech_score") or 0
        f_s = r.get("fund_score") or 0
        s_s = r.get("sent_score") or 0
        consensus = sum([t_s >= 7, f_s >= 7, s_s >= 7]) if any([t_s, f_s, s_s]) else None
        rows.append((
            cycle,
            sym,
            r.get("direction"),
            r.get("decision", "pass"),
            r.get("weighted_score"),
            r.get("confidence"),
            t_s or None,
            f_s or None,
            s_s or None,
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
            consensus,
            r.get("vol_ratio"),
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
                    position_id, consensus_score, vol_ratio
                ) VALUES (%s,%s,%s,%s, %s,%s,%s,%s,%s, %s,%s,%s, %s,%s,%s,%s,
                          %s,%s,%s,%s, %s,%s,%s,%s, %s,%s,%s,%s, %s,%s,%s)
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

            # Pattern 6: Multi-agent consensus score
            cur.execute("""
                SELECT CONCAT(
                    CASE WHEN consensus_score = 3 THEN '3/3 agents≥7'
                         WHEN consensus_score = 2 THEN '2/3 agents≥7'
                         WHEN consensus_score = 1 THEN '1/3 agents≥7'
                         ELSE '0/3 agents≥7' END,
                    ' + ', direction) AS pat,
                    COUNT(*) AS n,
                    ROUND(100.0 * COUNT(*) FILTER (WHERE outcome='win') / COUNT(*), 0) AS wr
                FROM scan_log
                WHERE decision='trade' AND outcome IS NOT NULL AND consensus_score IS NOT NULL
                GROUP BY 1 HAVING COUNT(*) >= %s
                ORDER BY wr DESC LIMIT 8
            """, (min_samples,))
            for row in cur.fetchall():
                patterns.append({"desc": row[0], "n": int(row[1]), "win_rate": float(row[2])})

            # Pattern 7: VIX level bucket
            cur.execute("""
                SELECT CONCAT(
                    CASE WHEN vix < 16 THEN 'VIX<16(calm)'
                         WHEN vix < 20 THEN 'VIX 16-20'
                         WHEN vix < 25 THEN 'VIX 20-25'
                         ELSE 'VIX≥25(fear)' END,
                    ' + ', direction) AS pat,
                    COUNT(*) AS n,
                    ROUND(100.0 * COUNT(*) FILTER (WHERE outcome='win') / COUNT(*), 0) AS wr
                FROM scan_log
                WHERE decision='trade' AND outcome IS NOT NULL AND vix IS NOT NULL
                GROUP BY 1 HAVING COUNT(*) >= %s
                ORDER BY wr DESC LIMIT 8
            """, (min_samples,))
            for row in cur.fetchall():
                patterns.append({"desc": row[0], "n": int(row[1]), "win_rate": float(row[2])})

            # Pattern 8: Volume ratio (breakout vs quiet)
            cur.execute("""
                SELECT CONCAT(
                    CASE WHEN vol_ratio >= 1.5 THEN 'vol≥1.5x(breakout)'
                         WHEN vol_ratio >= 1.1 THEN 'vol 1.1-1.5x'
                         WHEN vol_ratio >= 0.8 THEN 'vol normal'
                         ELSE 'vol<0.8x(dry)' END,
                    ' + ', direction) AS pat,
                    COUNT(*) AS n,
                    ROUND(100.0 * COUNT(*) FILTER (WHERE outcome='win') / COUNT(*), 0) AS wr
                FROM scan_log
                WHERE decision='trade' AND outcome IS NOT NULL AND vol_ratio IS NOT NULL
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
    return result[:12]


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

            # Most common pass reasons (top 5 — helps identify systematic filters)
            cur.execute("""
                SELECT
                    CASE
                        WHEN pass_reason ILIKE '%fatal flaw%'        THEN 'fatal-flaw'
                        WHEN pass_reason ILIKE '%score%fails%'       THEN 'score-below-threshold'
                        WHEN pass_reason ILIKE '%confidence%'        THEN 'low-confidence'
                        WHEN pass_reason ILIKE '%earnings%'          THEN 'earnings-risk'
                        WHEN pass_reason ILIKE '%iv%high%'           THEN 'IV-too-high'
                        WHEN pass_reason ILIKE '%max position%'      THEN 'max-positions'
                        WHEN pass_reason ILIKE '%duplicate%'         THEN 'duplicate-symbol'
                        WHEN pass_reason ILIKE '%regime%'            THEN 'regime-mismatch'
                        ELSE 'other' END AS reason_type,
                    COUNT(*) AS n
                FROM scan_log
                WHERE decision='pass' AND pass_reason IS NOT NULL AND pass_reason != ''
                GROUP BY 1 HAVING COUNT(*) >= 2
                ORDER BY n DESC LIMIT 5
            """)
            pass_rows = cur.fetchall()
            if pass_rows:
                lines.append("  Top pass reasons: " +
                    " | ".join(f"{r[0]} ({r[1]}×)" for r in pass_rows))

            # Time-of-day win rates (using logged_at → ET hour)
            cur.execute("""
                SELECT
                    CASE
                        WHEN EXTRACT(HOUR FROM logged_at AT TIME ZONE 'America/New_York') < 10 THEN 'pre-open(<10am)'
                        WHEN EXTRACT(HOUR FROM logged_at AT TIME ZONE 'America/New_York') < 12 THEN 'morning(10-12)'
                        WHEN EXTRACT(HOUR FROM logged_at AT TIME ZONE 'America/New_York') < 14 THEN 'midday(12-2)'
                        ELSE 'afternoon(2pm+)' END AS tod,
                    COUNT(*) AS n,
                    ROUND(100.0 * COUNT(*) FILTER (WHERE outcome='win') / COUNT(*), 0) AS wr
                FROM scan_log
                WHERE decision='trade' AND outcome IS NOT NULL
                GROUP BY 1 HAVING COUNT(*) >= 3
                ORDER BY wr DESC
            """)
            tod_rows = cur.fetchall()
            if tod_rows:
                lines.append("  Win rate by time of day: " +
                    " | ".join(f"{r[0]}: {int(r[2])}% ({r[1]}T)" for r in tod_rows))

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

            # Exit reason analysis — which exits tend to preserve more value?
            cur.execute("""
                SELECT
                    CASE
                        WHEN outcome_exit_reason ILIKE '%trailing stop%' THEN 'trailing-stop'
                        WHEN outcome_exit_reason ILIKE '%stop loss%'     THEN 'stop-loss'
                        WHEN outcome_exit_reason ILIKE '%theta%'         THEN 'theta-exit'
                        WHEN outcome_exit_reason ILIKE '%mini-peak%'     THEN 'mini-peak'
                        WHEN outcome_exit_reason ILIKE '%stale%'         THEN 'stale-loser'
                        WHEN outcome_exit_reason ILIKE '%expiry%'        THEN 'expiry'
                        WHEN outcome_exit_reason ILIKE '%low-conf%'      THEN 'low-conf-tp'
                        ELSE 'other' END AS exit_type,
                    COUNT(*) AS n,
                    ROUND(AVG(outcome_pnl_pct)::NUMERIC, 1) AS avg_pnl,
                    ROUND(100.0 * COUNT(*) FILTER (WHERE outcome='win') / COUNT(*), 0) AS wr
                FROM scan_log
                WHERE decision='trade' AND outcome IS NOT NULL AND outcome_exit_reason IS NOT NULL
                GROUP BY 1 HAVING COUNT(*) >= 2
                ORDER BY avg_pnl DESC
            """)
            exit_rows = cur.fetchall()
            if exit_rows:
                lines.append("  Exit reason outcomes: " +
                    " | ".join(f"{r[0]} avg{r[2]:+.0f}% {int(r[3])}%WR ({r[1]}T)" for r in exit_rows))

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


def get_similar_trade_stats(
    symbol: str,
    direction: str,
    tech_score: float = None,
    above_ema200: bool = None,
    adx: float = None,
    regime: str = None,
    min_samples: int = 3,
) -> dict | None:
    """
    Find historically similar setups to the current candidate and return their win rate.
    Matches on the most specific combination available, falling back to broader filters.
    Used to inject hyper-specific historical context into the Judge.
    """
    conn = _get_conn()
    if not conn:
        return None
    try:
        with conn.cursor() as cur:
            # Try progressively broader queries until we get enough samples
            filters_list = [
                # Most specific: all 4 features
                (
                    "symbol=%s AND direction=%s AND tech_score>=%s AND above_ema200=%s AND adx>=%s AND regime=%s",
                    [symbol, direction, (tech_score or 0) - 1, above_ema200, (adx or 0) - 5, regime]
                ) if all(x is not None for x in [tech_score, above_ema200, adx, regime]) else None,
                # Without adx constraint
                (
                    "symbol=%s AND direction=%s AND tech_score>=%s AND above_ema200=%s AND regime=%s",
                    [symbol, direction, (tech_score or 0) - 1, above_ema200, regime]
                ) if all(x is not None for x in [tech_score, above_ema200, regime]) else None,
                # Symbol + direction + tech tier
                (
                    "symbol=%s AND direction=%s AND tech_score>=%s",
                    [symbol, direction, (tech_score or 0) - 1]
                ) if tech_score is not None else None,
                # Symbol + direction only
                (
                    "symbol=%s AND direction=%s",
                    [symbol, direction]
                ),
            ]
            for filter_entry in filters_list:
                if filter_entry is None:
                    continue
                where, params = filter_entry
                cur.execute(f"""
                    SELECT COUNT(*) AS n,
                           COUNT(*) FILTER (WHERE outcome='win') AS wins,
                           ROUND(AVG(outcome_pnl_pct)::NUMERIC, 1) AS avg_pnl
                    FROM scan_log
                    WHERE decision='trade' AND outcome IS NOT NULL AND {where}
                """, params)
                row = cur.fetchone()
                if row and row[0] >= min_samples:
                    n, wins, avg_pnl = row
                    return {
                        "n":        int(n),
                        "win_rate": round(float(wins) / int(n), 3),
                        "avg_pnl":  float(avg_pnl) if avg_pnl else 0.0,
                        "filter":   where.replace("%s", "?"),
                    }
    except Exception as e:
        logger.debug(f"get_similar_trade_stats failed: {e}")
    return None


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


def get_outcome_stats() -> dict:
    """
    Compute Kelly fraction, win rate, and expectancy from the training DB.
    PostgreSQL-backed drop-in replacement for OutcomeTracker.get_stats().
    """
    conn = _get_conn()
    if not conn:
        return {}
    try:
        with conn.cursor() as cur:
            cur.execute("""
                SELECT
                    COUNT(*)                                                  AS n,
                    COUNT(*) FILTER (WHERE outcome='win')                     AS wins,
                    AVG(outcome_pnl_pct) FILTER (WHERE outcome='win')         AS avg_win,
                    ABS(AVG(outcome_pnl_pct) FILTER (WHERE outcome='loss'))   AS avg_loss
                FROM scan_log
                WHERE decision='trade' AND outcome IS NOT NULL
            """)
            row = cur.fetchone()
            if not row or not row[0]:
                return {"total_trades": 0, "kelly_ready": False}
            n, wins, avg_win, avg_loss = row
            n        = int(n)
            avg_win  = float(avg_win  or 0)
            avg_loss = float(avg_loss or 1)
            win_rate = float(wins) / n
            b        = avg_win / avg_loss if avg_loss > 0 else 1.0
            kelly_raw = (b * win_rate - (1 - win_rate)) / b if b > 0 else 0.0
            kelly    = max(0.0, min(0.25, kelly_raw))
            return {
                "total_trades":   n,
                "win_rate":       round(win_rate, 3),
                "avg_win_pct":    round(avg_win, 2),
                "avg_loss_pct":   round(avg_loss, 2),
                "kelly_fraction": round(kelly, 4),
                "expectancy":     round(win_rate * avg_win - (1 - win_rate) * avg_loss, 2),
                "kelly_ready":    n >= 10,
            }
    except Exception as e:
        logger.error(f"TrainingStore get_outcome_stats failed: {e}")
        return {}


def _wr_to_adj(win_rate: float, n: int) -> float:
    """Convert a feature win rate into a score delta, scaled by sample confidence."""
    # Confidence: full weight at 30+ trades, half at 10, quarter at 5
    conf = min(1.0, n / 30.0) * 0.6 + min(1.0, n / 10.0) * 0.4
    if   win_rate >= 0.70: base =  3.0
    elif win_rate >= 0.60: base =  1.5
    elif win_rate >= 0.55: base =  0.5
    elif win_rate >= 0.45: base =  0.0
    elif win_rate >= 0.40: base = -0.5
    elif win_rate >= 0.30: base = -1.5
    else:                  base = -3.0
    return round(base * conf, 2)


def get_feature_score_adjustment(
    direction: str,
    above_ema200: bool | None = None,
    above_ema50: bool | None = None,
    adx: float = None,
    regime: str | None = None,
    consensus_score: int | None = None,
    min_samples: int = 5,
) -> tuple[float, str]:
    """
    Query historical feature-level win rates from scan_log and return a Python-level
    score adjustment (bounded to [-5, +5]) plus a short reason label.

    This makes the Python scoring self-learning: the same indicators used in each
    trade are tracked against outcomes, and future scores shift accordingly.
    Only activates when >= min_samples matching trades exist for a feature.
    """
    conn = _get_conn()
    if not conn:
        return 0.0, ""

    components: list[tuple[float, str]] = []  # (weighted_adj, label)

    try:
        with conn.cursor() as cur:

            # ── Feature 1: EMA200 structure + direction (40% weight) ──────────
            if above_ema200 is not None:
                cur.execute("""
                    SELECT COUNT(*) AS n,
                           COUNT(*) FILTER (WHERE outcome='win') AS wins
                    FROM scan_log
                    WHERE decision='trade' AND outcome IS NOT NULL
                      AND direction=%s AND above_ema200=%s
                """, (direction, above_ema200))
                row = cur.fetchone()
                if row and row[0] >= min_samples:
                    n, wins = int(row[0]), int(row[1])
                    wr = wins / n
                    adj = _wr_to_adj(wr, n)
                    if abs(adj) > 0.1:
                        label = (
                            f"{'above' if above_ema200 else 'below'} EMA200 + {direction}: "
                            f"{wr:.0%}WR ({n}T)"
                        )
                        components.append((adj * 0.40, label))

            # ── Feature 2: ADX trend zone + direction (25% weight) ────────────
            if adx is not None:
                adx_zone = 'trending' if adx >= 25 else ('choppy' if adx < 18 else 'neutral')
                cur.execute("""
                    SELECT COUNT(*) AS n,
                           COUNT(*) FILTER (WHERE outcome='win') AS wins
                    FROM scan_log
                    WHERE decision='trade' AND outcome IS NOT NULL
                      AND direction=%s
                      AND CASE WHEN adx >= 25 THEN 'trending'
                               WHEN adx < 18  THEN 'choppy'
                               ELSE 'neutral' END = %s
                """, (direction, adx_zone))
                row = cur.fetchone()
                if row and row[0] >= min_samples:
                    n, wins = int(row[0]), int(row[1])
                    wr = wins / n
                    adj = _wr_to_adj(wr, n)
                    if abs(adj) > 0.1:
                        components.append((adj * 0.25, f"ADX {adx_zone} + {direction}: {wr:.0%}WR ({n}T)"))

            # ── Feature 3: Regime + direction (25% weight) ───────────────────
            if regime is not None:
                cur.execute("""
                    SELECT COUNT(*) AS n,
                           COUNT(*) FILTER (WHERE outcome='win') AS wins
                    FROM scan_log
                    WHERE decision='trade' AND outcome IS NOT NULL
                      AND direction=%s AND regime=%s
                """, (direction, regime))
                row = cur.fetchone()
                if row and row[0] >= min_samples:
                    n, wins = int(row[0]), int(row[1])
                    wr = wins / n
                    adj = _wr_to_adj(wr, n)
                    if abs(adj) > 0.1:
                        components.append((adj * 0.25, f"{regime}+{direction}: {wr:.0%}WR ({n}T)"))

            # ── Feature 4: Agent consensus + direction (10% weight) ──────────
            if consensus_score is not None:
                cur.execute("""
                    SELECT COUNT(*) AS n,
                           COUNT(*) FILTER (WHERE outcome='win') AS wins
                    FROM scan_log
                    WHERE decision='trade' AND outcome IS NOT NULL
                      AND direction=%s AND consensus_score=%s
                """, (direction, consensus_score))
                row = cur.fetchone()
                if row and row[0] >= min_samples:
                    n, wins = int(row[0]), int(row[1])
                    wr = wins / n
                    adj = _wr_to_adj(wr, n)
                    if abs(adj) > 0.1:
                        components.append((adj * 0.10, f"consensus {consensus_score}/3 + {direction}: {wr:.0%}WR ({n}T)"))

    except Exception as e:
        logger.debug(f"get_feature_score_adjustment failed: {e}")
        return 0.0, ""

    if not components:
        return 0.0, ""

    total = round(sum(c[0] for c in components), 1)
    total = max(-5.0, min(5.0, total))
    reason = " | ".join(c[1] for c in components if abs(c[0]) > 0.05)
    return total, reason


def get_episodic_context(direction: str, symbol: str = None, limit: int = 5) -> str:
    """
    Return a narrative of recent closed trades with matching direction (and optionally
    symbol), including their key technical conditions at entry and actual outcome.

    This gives the Judge concrete episodic memory: "last time conditions were X, outcome was Y."
    More useful than summary stats alone because the Judge can compare current indicators
    directly against past winning and losing setups.
    """
    conn = _get_conn()
    if not conn:
        return ""
    try:
        with conn.cursor() as cur:
            params: list = [direction]
            sym_filter = ""
            if symbol:
                sym_filter = "AND symbol=%s "
                params.append(symbol)
            params.append(limit * 2)  # fetch more for diversity
            cur.execute(f"""
                SELECT symbol, direction, outcome, outcome_pnl_pct, outcome_days_held,
                       tech_score, fund_score, sent_score, rsi, adx,
                       above_ema200, above_ema50, regime, iv_rank,
                       consensus_score, outcome_exit_reason, logged_at
                FROM scan_log
                WHERE decision='trade' AND outcome IS NOT NULL
                  AND direction=%s {sym_filter}
                ORDER BY logged_at DESC
                LIMIT %s
            """, params)
            rows = cur.fetchall()
            if not rows:
                return ""

            lines = ["RECENT SIMILAR TRADES (same direction — learn from these):"]
            for row in rows[:limit]:
                (sym, dirn, outcome, pnl, days, tech_s, fund_s, sent_s,
                 rsi, adx, ema200, ema50, reg, iv, cons, exit_r, logged) = row
                outcome_str = f"{'✅ WON' if outcome == 'win' else '❌ LOST'} {float(pnl or 0):+.1f}% in {days or '?'}d"
                ema_str = ("above EMA200" if ema200 else "below EMA200") if ema200 is not None else "EMA200 N/A"
                adx_str = f"ADX {float(adx or 0):.0f}" if adx else ""
                rsi_str = f"RSI {float(rsi or 0):.0f}" if rsi else ""
                reg_str = f"{reg} regime" if reg else ""
                iv_str  = f"IV rank {float(iv or 0):.0f}" if iv else ""
                cons_str = f"consensus {cons}/3" if cons is not None else ""
                exit_str = f"exit: {exit_r[:50]}" if exit_r else ""
                conditions = ", ".join(filter(None, [ema_str, adx_str, rsi_str, reg_str, iv_str, cons_str]))
                score_str  = f"tech {tech_s}/{fund_s}/{sent_s}" if tech_s else ""
                lines.append(
                    f"  {sym} {dirn}: {conditions} | {score_str} | {outcome_str}"
                    + (f" ({exit_str})" if exit_str else "")
                )

            lines.append(
                "Judge: compare current setup's RSI, ADX, EMA alignment, and regime to "
                "the winning vs losing trades above. Adjust confidence accordingly."
            )
            return "\n".join(lines)
    except Exception as e:
        logger.debug(f"get_episodic_context failed: {e}")
        return ""


def get_similar_iv_stats(iv_rank: float, direction: str, min_samples: int = 3) -> dict | None:
    """
    Find closed trades with similar IV rank (±20) and same direction.
    PostgreSQL-backed replacement for OutcomeTracker.get_similar_setups().
    """
    conn = _get_conn()
    if not conn:
        return None
    try:
        with conn.cursor() as cur:
            cur.execute("""
                SELECT COUNT(*) AS n,
                       COUNT(*) FILTER (WHERE outcome='win') AS wins,
                       ROUND(AVG(outcome_pnl_pct)::NUMERIC, 1) AS avg_pnl
                FROM scan_log
                WHERE decision='trade' AND outcome IS NOT NULL
                  AND direction = %s
                  AND iv_rank BETWEEN %s AND %s
            """, (direction, iv_rank - 20, iv_rank + 20))
            row = cur.fetchone()
            if not row or row[0] < min_samples:
                return None
            n, wins, avg_pnl = row
            return {
                "count":    int(n),
                "win_rate": round(float(wins) / int(n), 3),
                "avg_pnl":  float(avg_pnl) if avg_pnl else 0.0,
            }
    except Exception as e:
        logger.debug(f"get_similar_iv_stats failed: {e}")
        return None


def get_all_closed_trades(exclude_position_ids: set = None) -> list:
    """
    Return all closed trade rows from scan_log formatted like sim_positions.
    Used to show historical trades in the UI that survive sim resets.
    Optionally exclude position_ids already present in sim_positions.
    """
    conn = _get_conn()
    if not conn:
        return []
    try:
        with conn.cursor() as cur:
            cur.execute("""
                SELECT position_id, symbol, direction, decision,
                       weighted_score, confidence, tech_score, fund_score, sent_score,
                       outcome, outcome_pnl_pct, outcome_pnl_dollars,
                       outcome_days_held, outcome_exit_reason, outcome_closed_at, logged_at
                FROM scan_log
                WHERE decision = 'trade' AND outcome IS NOT NULL AND position_id IS NOT NULL
                ORDER BY outcome_closed_at ASC NULLS LAST
            """)
            rows = []
            exclude = exclude_position_ids or set()
            for r in cur.fetchall():
                pid = r[0]
                if pid in exclude:
                    continue
                pnl_d = float(r[11] or 0)
                pnl_p = float(r[10] or 0)
                rows.append({
                    "position_id":       pid,
                    "symbol":            r[1] or "—",
                    "direction":         r[2] or "bullish",
                    "option_type":       "call" if r[2] == "bullish" else "put",
                    "status":            "closed",
                    "source":            "db",
                    "pnl_dollars":       round(pnl_d, 2),
                    "pnl_pct":           round(pnl_p, 2),
                    "weighted_score":    float(r[4]) if r[4] is not None else None,
                    "confidence":        int(r[5]) if r[5] is not None else None,
                    "tech_score":        float(r[6]) if r[6] is not None else None,
                    "fund_score":        float(r[7]) if r[7] is not None else None,
                    "sent_score":        float(r[8]) if r[8] is not None else None,
                    "days_held":         int(r[12]) if r[12] is not None else None,
                    "exit_reason":       r[13] or "",
                    "closed_at":         (r[14] or r[15]).isoformat() if (r[14] or r[15]) else None,
                    "opened_at":         r[15].isoformat() if r[15] else None,
                })
            return rows
    except Exception as e:
        logger.error(f"get_all_closed_trades failed: {e}")
        return []


def delete_scan_log_trades(position_ids: list) -> int:
    """Hard-delete scan_log rows by position_id. Returns number removed."""
    conn = _get_conn()
    if not conn or not position_ids:
        return 0
    try:
        with conn.cursor() as cur:
            cur.execute(
                "DELETE FROM scan_log WHERE position_id = ANY(%s)",
                (list(position_ids),)
            )
            removed = cur.rowcount
        logger.info(f"delete_scan_log_trades: removed {removed} rows")
        return removed
    except Exception as e:
        logger.error(f"delete_scan_log_trades failed: {e}")
        return 0
