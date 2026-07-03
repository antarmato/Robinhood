# Options Trading System

A multi-agent, self-learning options trading **simulator**. A FastAPI backend runs a
continuous scan/monitor loop during market hours, uses Claude agents to analyze
candidates, auto-executes the best setups as simulated single-leg option positions,
and learns from every outcome via a PostgreSQL training log. A single-page dashboard
(`frontend/index.html`) visualizes everything live over WebSocket.

> ⚠️ All positions are **simulated** ($100 max risk per trade, modeled option pricing).
> Nothing places real orders.

## Architecture

```
                          ┌──────────────────────────────┐
 pre-market (9:00 ET) ───▶│  Market regime classifier    │
                          │  + overnight gap snapshots   │
                          └──────────────┬───────────────┘
                                         ▼
 every SCAN_INTERVAL ────▶ Scanner (IV-first, regime-biased)
                                         ▼  top candidates
                     ┌───────────┬───────────┬───────────┐
                     │ Technical │Fundamental│ Sentiment │  + Risk   (parallel)
                     └─────┬─────┴─────┬─────┴─────┬─────┘
                           ▼           ▼           ▼
                          Judge (claude-opus-4-8, one LLM call)
                           │  deterministic score + threshold + confidence
                           ▼
                  best "trade" decision → simulated position
                                         ▼
 every MONITOR_INTERVAL ─▶ Monitor: mark-to-model (backend/pricing.py),
                           trailing stops, theta/stale exits,
                           cooldown-gated LLM thesis review
                                         ▼
                           Outcome tracker + scan_log (Postgres)
                           → feeds win rates back into future Judge calls
```

### Key modules

| Module | Role |
|---|---|
| `backend/orchestrator.py` | Main loop: pre-market prep, scan cycles, position monitor, circuit breakers, portfolio filters |
| `backend/agents/` | Scanner, Technical, Fundamental, Sentiment, Risk, Judge, PositionReviewer |
| `backend/pricing.py` | **Single source of truth** for entry premium sizing, the option pricing model (gamma/vega/theta), and the exit ladder — used by both the monitor loop and `/api/sim/prices` |
| `backend/strategy.py` | IV-rank thresholds, confidence minimums, trade defaults |
| `backend/market_data.py` | Alpaca (primary) / Polygon quotes, historicals, news, with caching |
| `backend/market_regime.py` | Bull/bear/neutral classification (SPY trend + VIX) |
| `backend/state.py` | State persistence — Postgres (`DATABASE_URL`) with JSON-file fallback; writes are debounced on a background thread |
| `backend/training_store.py` | `scan_log` table: every decision + outcome; win-rate queries feed the Judge's self-learned calibration |
| `backend/outcome_tracker.py` | Win rate / expectancy / Kelly fraction |
| `backend/timeutil.py` | Eastern-time helpers (naive-timestamp compatible) |
| `backend/main.py` | FastAPI app: dashboard, REST API, WebSocket |
| `frontend/index.html` | Self-contained dashboard SPA |

## Sim position model

- Entry premium is modeled as `spot × (2% + 4% × IV/100) × sqrt(DTE/35)` (~25-delta
  pricing), and fractional contracts size every position to **$100 total cost** — so
  leverage per 1% stock move is uniform (~6× at IV 50) regardless of share price.
- Round-trip bid-ask friction (3%–7%, wider at high IV) is deducted from
  liquidation value, so realized P&L reflects actually crossing the spread.
- Exits: IV-aware initial stop, tiered trailing floor, stall tightening, DTE lift,
  mini-peak lock, low-confidence take-profit, stale-loser / dead-money / theta /
  expiry exits — all in `backend/pricing.py` with unit tests. Positions are only
  marked/exited during the regular session (9:30–16:00 ET).
- Entry gates: score threshold (IV/time/regime adjusted), per-symbol confidence
  minimums, a score cushion for bare-minimum-confidence setups, IV rank hard cap
  (>75 never enters), sector/correlation caps, re-entry and churn blocks, and
  no fills in the first 15 or last 30 minutes of the session.

## Running locally

```sh
python -m venv .venv
.venv/Scripts/pip install -r backend/requirements.txt   # (bin/pip on macOS/Linux)
cp .env.example .env                                     # fill in keys
uvicorn backend.main:app --reload
```

Open http://localhost:8000 and press **Start**. Without `DATABASE_URL` the state
persists to a local JSON file and the training log is disabled.

### Tests

```sh
.venv/Scripts/pip install pytest
.venv/Scripts/python -m pytest backend/tests -q
```

## Environment variables

| Variable | Required | Purpose |
|---|---|---|
| `ANTHROPIC_API_KEY` | ✅ | Claude agent calls |
| `ALPACA_API_KEY` / `ALPACA_SECRET_KEY` | ✅ | Market data (quotes, bars, news) |
| `POLYGON_API_KEY` | — | Fallback data source |
| `TRADIER_TOKEN` | — | Optional real options IV (otherwise HV proxy) |
| `DATABASE_URL` | — | Postgres for state + training log (falls back to file) |
| `API_KEY` | recommended | Auth for control endpoints (start/stop/reset/trades). **Unset = open access** — a warning is logged at startup |
| `WATCHLIST` | — | Comma-separated symbols (default: 15-name high-beta list) |
| `SCAN_INTERVAL_MINUTES` | — | Default 30 |
| `MONITOR_INTERVAL_MINUTES` | — | Default 5 |
| `THESIS_REVIEW_HOURS` | — | Min hours between LLM thesis reviews per position (default 3) |
| `MAX_LOSS_PER_TRADE` | — | Default 100 |
| `STATE_FILE` | — | File-fallback state path (default `/app/data/state.json`) |
| `TZ` | recommended | Set to `America/New_York` in deployment |

## API surface (selected)

| Endpoint | Purpose |
|---|---|
| `GET /` | Dashboard |
| `GET /api/status`, `/api/health`, `/api/diagnostics` | Health/monitoring |
| `GET /api/system/start` · `stop` · `scan` | Control loop (requires `api_key` when `API_KEY` is set) |
| `GET /api/sim` · `/api/sim/prices` | Positions, stats, live P&L refresh |
| `GET /api/scan-results` | Latest scan board |
| `GET /api/trades/all` | Full trade history incl. training-DB records |
| `GET /api/sim/reset` · `clear-closed` | Destructive sim maintenance (keyed) |
| `GET /api/training-data` · `/api/model-insights` | Learning-loop introspection |
| `WS /ws` | Live event stream for the dashboard |

Mutating GET endpoints exist because the Cowork artifact client can only issue GETs;
POST equivalents exist for the dashboard.

## Deployment

Deployed on Railway via the included `Dockerfile` + `railway.toml`. Attach a Postgres
plugin (sets `DATABASE_URL`) so state and the training log survive restarts. `push.ps1`
commits and pushes (Railway auto-deploys from main).
