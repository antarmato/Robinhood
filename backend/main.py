"""
FastAPI app — serves the Railway monitoring dashboard and exposes the proposal API
that the Cowork artifact polls and updates.
"""

import logging
import os
from pathlib import Path
from typing import Set

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

from .orchestrator import get_orchestrator
from .state import get_state
from .outcome_tracker import get_outcome_tracker
from . import pricing
from . import training_store as ts
from .timeutil import days_since, parse_iso_et, now_et

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger(__name__)

app = FastAPI(title="Options Trading System", version="2.0.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

FRONTEND_DIR = Path(__file__).parent.parent / "frontend"

# ── WebSocket manager ──────────────────────────────────────────────────────────

class ConnectionManager:
    def __init__(self):
        self.active: Set[WebSocket] = set()

    async def connect(self, ws: WebSocket):
        await ws.accept()
        self.active.add(ws)

    def disconnect(self, ws: WebSocket):
        self.active.discard(ws)

    async def broadcast(self, message: dict):
        dead = set()
        for ws in self.active:
            try:
                await ws.send_json(message)
            except Exception:
                dead.add(ws)
        for ws in dead:
            self.active.discard(ws)

manager = ConnectionManager()

async def _broadcast(agent: str, event_type: str, data):
    import datetime
    await manager.broadcast({
        "agent": agent, "type": event_type, "data": data,
        "timestamp": datetime.datetime.now().isoformat(),
    })

orchestrator = get_orchestrator()
orchestrator.set_broadcast(_broadcast)

# ── Auth ───────────────────────────────────────────────────────────────────────

API_KEY = os.getenv("API_KEY", "")  # Optional key for Cowork artifact to authenticate
if not API_KEY:
    logger.warning(
        "API_KEY not set — control endpoints (start/stop/reset/trades) are UNAUTHENTICATED. "
        "Set API_KEY in the environment before exposing this service publicly."
    )

def _check_key(key: str = ""):
    if API_KEY and key != API_KEY:
        raise HTTPException(status_code=401, detail="Invalid API key")

# ── Pydantic models ────────────────────────────────────────────────────────────

class StartRequest(BaseModel):
    api_key: str = ""

class ResolveProposalRequest(BaseModel):
    action: str           # "executed" | "rejected"
    order_info: dict = {}
    api_key: str = ""

class AddTradeRequest(BaseModel):
    trade: dict
    api_key: str = ""

class CloseTradeRequest(BaseModel):
    trade_id: str
    pnl: float = 0.0
    api_key: str = ""

class ResolveExitRequest(BaseModel):
    trade_id: str
    api_key: str = ""

# ── Routes ─────────────────────────────────────────────────────────────────────

@app.get("/", response_class=HTMLResponse)
async def root():
    index = FRONTEND_DIR / "index.html"
    return HTMLResponse(index.read_text(encoding="utf-8") if index.exists() else "<h1>Dashboard loading...</h1>")

@app.get("/api/health")
async def health():
    """Diagnostic: check env vars and Alpaca/Polygon connectivity."""
    import os
    from . import market_data as mdata
    poly_key    = os.getenv("POLYGON_API_KEY", "")
    alpaca_key  = os.getenv("ALPACA_API_KEY", "")
    anthropic_key = os.getenv("ANTHROPIC_API_KEY", "")
    result = {
        "status": "ok",
        "alpaca_key_set":    bool(alpaca_key),
        "polygon_key_set":   bool(poly_key),
        "anthropic_key_set": bool(anthropic_key),
        "tz": os.getenv("TZ", "NOT SET"),
        "tradier_set": bool(os.getenv("TRADIER_TOKEN", "")),
    }
    try:
        df = mdata.get_historicals("SPY", period="3mo")
        result["data_test"] = f"OK — {len(df)} bars for SPY, last ${df['close'].iloc[-1]:.2f}" if not df.empty else "FAIL — empty"
    except Exception as e:
        result["data_test"] = f"ERROR — {e}"
    return result

@app.get("/api/debug/network")
async def debug_network():
    """Test connectivity to Polygon.io and Yahoo Finance from Railway."""
    import asyncio, requests as req, os
    from datetime import date, timedelta
    loop = asyncio.get_event_loop()

    async def _test(name: str, url: str, params=None):
        def _fetch():
            try:
                r = req.get(url, params=params, timeout=8,
                            headers={"User-Agent": "Mozilla/5.0"})
                return {"ok": r.status_code == 200, "status": r.status_code, "bytes": len(r.content)}
            except Exception as e:
                return {"ok": False, "error": f"{type(e).__name__}: {str(e)[:120]}"}
        return name, await loop.run_in_executor(None, _fetch)

    api_key = os.getenv("POLYGON_API_KEY", "")
    end   = date.today().strftime("%Y-%m-%d")
    start = (date.today() - timedelta(days=5)).strftime("%Y-%m-%d")

    tests = await asyncio.gather(
        _test("httpbin", "https://httpbin.org/get"),
        _test("polygon_SPY",
              f"https://api.polygon.io/v2/aggs/ticker/SPY/range/1/day/{start}/{end}",
              {"adjusted": "true", "sort": "asc", "limit": 10, "apiKey": api_key or "NO_KEY"}),
        _test("yahoo_SPY",
              "https://query1.finance.yahoo.com/v8/finance/chart/SPY"),
    )
    result = {name: data for name, data in tests}
    result["polygon_key_set"] = bool(api_key)
    result["all_ok"] = result.get("polygon_SPY", {}).get("ok", False)
    return result

@app.get("/api/status")
async def status():
    s = get_state()
    from .orchestrator import Orchestrator
    session = Orchestrator._session_phase()
    session_label = {
        "pre_open": "PRE-MKT",
        "market": "OPEN",
        "after_hours": "AH",
        "closed": "CLOSED",
    }.get(session, "CLOSED")
    return {
        "system_status":  s.system_status,
        "cycle_count":    s.cycle_count,
        "last_scan":      s.get_full_state().get("last_scan"),
        "last_monitor":   s.get_full_state().get("last_monitor"),
        "active_trades":  len(s.active_trades),
        "pending_proposals": len(s.get_pending_proposals()),
        "pending_exits":  len(s.get_pending_exit_signals()),
        "session":        session_label,
        "market_open":    Orchestrator._is_market_hours(),
    }

# ── Proposal API (polled by Cowork artifact) ───────────────────────────────────


@app.get("/api/history")
async def get_history(api_key: str = ""):
    _check_key(api_key)
    return get_state().get_all_symbol_history()

@app.get("/api/proposals")
async def get_proposals(api_key: str = ""):
    _check_key(api_key)
    s = get_state()
    return {
        "pending":  s.get_pending_proposals(),
        "recent":   [p for p in s.proposals[-10:] if p.get("status") != "pending"],
        "exit_signals": s.get_pending_exit_signals(),
        "active_trades": s.active_trades,
    }

@app.post("/api/proposals/{proposal_id}/resolve")
async def resolve_proposal(proposal_id: str, req: ResolveProposalRequest):
    _check_key(req.api_key)
    if req.action not in ("executed", "rejected"):
        raise HTTPException(status_code=400, detail="action must be 'executed' or 'rejected'")
    get_state().resolve_proposal(proposal_id, req.action, req.order_info)
    await _broadcast("system", "proposal_resolved",
        {"proposal_id": proposal_id, "action": req.action})
    return {"status": "ok", "proposal_id": proposal_id, "action": req.action}

# ── GET-based mutation endpoints (used by Cowork artifact via web_fetch, GET only) ──

@app.get("/api/proposals/{proposal_id}/resolve")
async def resolve_proposal_get(proposal_id: str, action: str, api_key: str = "",
                                order_id: str = "", ref_id: str = ""):
    _check_key(api_key)
    if action not in ("executed", "rejected"):
        raise HTTPException(status_code=400, detail="action must be 'executed' or 'rejected'")
    order_info = {}
    if order_id:
        order_info = {"order_id": order_id, "ref_id": ref_id,
                      "placed_at": __import__("datetime").datetime.now().isoformat()}
    get_state().resolve_proposal(proposal_id, action, order_info)
    await _broadcast("system", "proposal_resolved", {"proposal_id": proposal_id, "action": action})
    return {"status": "ok", "proposal_id": proposal_id, "action": action}

@app.get("/api/trades/register")
async def register_trade_get(api_key: str = "", symbol: str = "", option_type: str = "",
                              strike: str = "", expiration_date: str = "", contracts: int = 1,
                              limit_price: str = "", total_max_loss: str = "",
                              option_id: str = "", order_id: str = "", trade_id: str = ""):
    _check_key(api_key)
    trade = {
        "trade_id": trade_id or order_id,
        "symbol": symbol, "option_type": option_type, "strike": strike,
        "expiration_date": expiration_date, "contracts": contracts,
        "limit_price": limit_price, "total_max_loss": total_max_loss,
        "option_id": option_id, "order_id": order_id,
    }
    get_state().add_active_trade(trade)
    await _broadcast("system", "trade_opened", trade)
    return {"status": "ok", "trade_id": trade["trade_id"]}

@app.get("/api/trades/{trade_id}/close")
async def close_trade_get(trade_id: str, api_key: str = "", pnl: float = 0.0):
    _check_key(api_key)
    s = get_state()
    # Find the trade to compute total PnL before removing it
    trade = next((t for t in s.active_trades if t.get("trade_id") == trade_id), {})
    contracts     = trade.get("contracts", 1)
    entry_premium = float(trade.get("limit_price") or trade.get("max_premium") or 0)
    pnl_pct = (pnl / (entry_premium * contracts * 100) * 100) if entry_premium and contracts else 0.0
    s.close_trade(trade_id, pnl)
    s.resolve_exit_signal(trade_id)
    get_outcome_tracker().record_outcome(trade_id, pnl_pct, pnl)
    await _broadcast("system", "trade_closed", {"trade_id": trade_id, "pnl": pnl, "pnl_pct": round(pnl_pct, 2)})
    return {"status": "ok"}

@app.get("/api/exits/{trade_id}/resolve")
async def resolve_exit_get(trade_id: str, api_key: str = ""):
    _check_key(api_key)
    get_state().resolve_exit_signal(trade_id)
    return {"status": "ok"}

@app.get("/api/system/start")
async def start_get(api_key: str = ""):
    _check_key(api_key)
    try:
        await orchestrator.start()
        return {"status": "started"}
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.get("/api/system/stop")
async def stop_get(api_key: str = ""):
    _check_key(api_key)
    await orchestrator.stop()
    return {"status": "stopped"}


@app.get("/api/system/scan")
async def force_scan(api_key: str = ""):
    _check_key(api_key)
    task_alive = orchestrator._task is not None and not orchestrator._task.done()
    if not task_alive:
        return {"ok": False, "message": "System not running — start it first"}
    orchestrator.trigger_scan()
    return {"ok": True, "message": "Scan queued — will fire within 30 seconds"}

# ── POST versions (kept for Railway web dashboard) ─────────────────────────────

@app.post("/api/trades")
async def add_trade(req: AddTradeRequest):
    _check_key(req.api_key)
    get_state().add_active_trade(req.trade)
    await _broadcast("system", "trade_opened", req.trade)
    return {"status": "ok"}

@app.post("/api/trades/close")
async def close_trade(req: CloseTradeRequest):
    _check_key(req.api_key)
    s = get_state()
    trade = next((t for t in s.active_trades if t.get("trade_id") == req.trade_id), {})
    contracts     = trade.get("contracts", 1)
    entry_premium = float(trade.get("limit_price") or trade.get("max_premium") or 0)
    pnl_pct = (req.pnl / (entry_premium * contracts * 100) * 100) if entry_premium and contracts else 0.0
    s.close_trade(req.trade_id, req.pnl)
    s.resolve_exit_signal(req.trade_id)
    get_outcome_tracker().record_outcome(req.trade_id, pnl_pct, req.pnl)
    await _broadcast("system", "trade_closed", {"trade_id": req.trade_id, "pnl": req.pnl, "pnl_pct": round(pnl_pct, 2)})
    return {"status": "ok"}

@app.post("/api/exits/{trade_id}/resolve")
async def resolve_exit(trade_id: str, req: ResolveExitRequest):
    _check_key(req.api_key)
    get_state().resolve_exit_signal(trade_id)
    return {"status": "ok"}

# ── System control (POST) ──────────────────────────────────────────────────────

@app.post("/api/system/start")
async def start(req: StartRequest):
    _check_key(req.api_key)
    try:
        await orchestrator.start()
        return {"status": "started"}
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/api/system/stop")
async def stop(req: StartRequest):
    _check_key(req.api_key)
    await orchestrator.stop()
    return {"status": "stopped"}

@app.get("/api/events")
async def events():
    return {"events": get_state().get_full_state().get("event_log", [])[-50:]}

@app.get("/api/sim")
async def get_sim():
    """Sim mode: all positions, P&L history, and stats."""
    s = get_state()
    positions = s.get_sim_positions()
    history   = s.get_pnl_history()

    closed = [p for p in positions if p.get("status") == "closed"]
    open_  = [p for p in positions if p.get("status") == "open"]
    wins   = [p for p in closed if float(p.get("pnl_dollars", 0)) > 0]
    losses = [p for p in closed if float(p.get("pnl_dollars", 0)) < 0]

    total_pnl = s.cumulative_sim_pnl()
    ot_stats = ts.get_outcome_stats()

    # Max drawdown from equity curve (worst peak-to-trough)
    max_dd = 0.0
    if history:
        peak = history[0]["cumulative_pnl"] if history else 0.0
        for h in history:
            v = h["cumulative_pnl"]
            if v > peak:
                peak = v
            dd = peak - v
            if dd > max_dd:
                max_dd = dd

    # Expectancy: E = WR * avg_win% - (1-WR) * avg_loss%
    avg_win_pct  = ot_stats.get("avg_win_pct", 0)
    avg_loss_pct = ot_stats.get("avg_loss_pct", 0)
    win_rate_frac = ot_stats.get("win_rate", 0)
    expectancy_pct = round(win_rate_frac * avg_win_pct - (1 - win_rate_frac) * avg_loss_pct, 1) if ot_stats else 0

    # Rolling win rates for trend visualization
    def _rolling_wr(n: int) -> float | None:
        recent = closed[-n:] if len(closed) >= n else None
        if not recent:
            return None
        w = sum(1 for p in recent if float(p.get("pnl_dollars", 0)) > 0)
        return round(w / len(recent) * 100, 1)

    overall_wr = round(len(wins) / len(closed) * 100, 1) if closed else 0

    today = now_et().strftime("%Y-%m-%d")
    today_pnl = round(sum(
        float(p.get("pnl_dollars", 0)) for p in closed
        if str(p.get("closed_at") or "")[:10] == today
    ), 2)

    return {
        "open_positions":  open_,
        "closed_positions": closed,
        "pnl_history":     history,
        "stats": {
            "total_pnl":       total_pnl,
            "today_pnl":       today_pnl,
            "total_trades":    len(closed),
            "open_count":      len(open_),
            "win_rate":        overall_wr,
            "avg_win":         round(sum(float(p.get("pnl_dollars", 0)) for p in wins)  / len(wins),   2) if wins   else 0,
            "avg_loss":        round(sum(float(p.get("pnl_dollars", 0)) for p in losses) / len(losses), 2) if losses else 0,
            "best_trade":      round(max((float(p.get("pnl_dollars", 0)) for p in closed), default=0), 2),
            "worst_trade":     round(min((float(p.get("pnl_dollars", 0)) for p in closed), default=0), 2),
            "avg_win_pct":     avg_win_pct,
            "avg_loss_pct":    avg_loss_pct,
            "expectancy_pct":  expectancy_pct,
            "kelly_fraction":  ot_stats.get("kelly_fraction", 0),
            "max_drawdown":    round(max_dd, 2),
            "recent_5_wr":     _rolling_wr(5),
            "recent_10_wr":    _rolling_wr(10),
            "recent_20_wr":    _rolling_wr(20),
        },
    }


@app.get("/api/sim/prices")
async def get_sim_prices():
    """
    Live price refresh for open positions — called by frontend every 30s.
    Uses the same pricing model as the monitor loop (backend/pricing.py) so the
    dashboard P&L always matches what the exit engine acts on.
    """
    import asyncio as _aio
    from . import market_data as _md
    s = get_state()
    open_positions = [p for p in s.get_sim_positions() if p.get("status") == "open"]
    if not open_positions:
        return {"updates": [], "total_unrealized_pnl": 0.0}

    symbols = list(set(p["symbol"] for p in open_positions))
    try:
        batch = await _aio.to_thread(_md.get_batch_quotes, symbols)
    except Exception:
        return {"updates": [], "error": "price fetch failed"}

    updates = []
    total_unrealized = 0.0
    for pos in open_positions:
        symbol  = pos["symbol"]
        current = float(batch.get(symbol, 0))
        if not current:
            continue
        try:
            days_held = days_since(pos["opened_at"])
        except (KeyError, ValueError):
            days_held = 0
        mark = pricing.mark_position(pos, current, days_held)

        total_unrealized += mark["pnl_dollars"]
        updates.append({
            "position_id":   pos["position_id"],
            "symbol":        symbol,
            "current_price": round(current, 2),
            "pnl_pct":       round(mark["pnl_pct"], 1),
            "pnl_dollars":   mark["pnl_dollars"],
        })

    return {"updates": updates, "total_unrealized_pnl": round(total_unrealized, 2)}


@app.get("/api/test/price")
async def test_price(symbol: str = "PYPL"):
    """Test endpoint: fetch live price for any symbol (default PYPL)."""
    import asyncio as _aio
    from . import market_data as _md
    loop = _aio.get_event_loop()
    try:
        quote = await loop.run_in_executor(None, lambda: _md.get_quote(symbol.upper()))
        return {"ok": True, "symbol": symbol.upper(), "price": quote.get("price"), "pct_change": quote.get("pct_change"), "source": "alpaca/iex"}
    except Exception as e:
        return {"ok": False, "symbol": symbol.upper(), "error": str(e)}


@app.get("/api/symbol-stats")
async def symbol_stats():
    """Per-symbol win rate and avg P&L from training DB."""
    return {"symbol_stats": ts.get_symbol_perf(min_trades=1)}


@app.get("/api/diagnostics")
async def diagnostics():
    """Run all API connectivity tests — Alpaca, Anthropic, market status, env vars."""
    import asyncio as _aio
    import time
    from . import market_data as mdata
    from .orchestrator import Orchestrator

    loop = _aio.get_event_loop()

    # ── Env vars ───────────────────────────────────────────────────────────────
    alpaca_key    = os.getenv("ALPACA_API_KEY", "")
    anthropic_key = os.getenv("ANTHROPIC_API_KEY", "")
    tradier_key   = os.getenv("TRADIER_TOKEN", "")
    results: dict = {
        "env_vars": {
            "ok":       bool(alpaca_key and anthropic_key),
            "alpaca":   "✓ set" if alpaca_key else "✗ NOT SET",
            "anthropic": "✓ set" if anthropic_key else "✗ NOT SET",
            "tradier":  f"✓ set" if tradier_key else "— optional (not set)",
            "tz":       os.getenv("TZ", "not set"),
            "latency_ms": 0,
        },
        "market": {
            "ok":          True,
            "phase":       Orchestrator._session_phase(),
            "market_open": Orchestrator._is_market_hours(),
            "trading_day": Orchestrator._is_trading_day(),
            "latency_ms":  0,
        },
    }

    # ── Async tests ────────────────────────────────────────────────────────────
    async def _test_alpaca_history():
        t0 = time.time()
        try:
            df = await loop.run_in_executor(None, lambda: mdata.get_historicals("SOFI", period="3mo"))
            ms = round((time.time() - t0) * 1000)
            if df.empty:
                return {"ok": False, "detail": "Empty — no bars returned", "latency_ms": ms}
            return {"ok": True,
                    "detail": f"SOFI: {len(df)} bars | last close ${df['close'].iloc[-1]:.2f}",
                    "latency_ms": ms}
        except Exception as e:
            return {"ok": False, "detail": str(e)[:150], "latency_ms": round((time.time()-t0)*1000)}

    async def _test_alpaca_quotes():
        t0 = time.time()
        try:
            quotes = await loop.run_in_executor(
                None, lambda: mdata.get_batch_quotes(["SOFI", "PLTR", "MSTR"]))
            ms = round((time.time() - t0) * 1000)
            if not quotes:
                return {"ok": False, "detail": "No quotes returned (IEX feed)", "latency_ms": ms}
            items = " | ".join(f"{s} ${p:.2f}" for s, p in quotes.items())
            return {"ok": True, "detail": items, "latency_ms": ms}
        except Exception as e:
            return {"ok": False, "detail": str(e)[:150], "latency_ms": round((time.time()-t0)*1000)}

    async def _test_alpaca_macro():
        t0 = time.time()
        try:
            vix     = await loop.run_in_executor(None, mdata.get_vix)
            spy     = await loop.run_in_executor(None, lambda: mdata.get_quote("SPY"))
            sectors = await loop.run_in_executor(None, mdata.get_sector_etf_performance)
            ms = round((time.time() - t0) * 1000)
            spy_p, spy_c = spy.get("price", 0), spy.get("pct_change", 0)
            return {
                "ok":    vix > 0 or spy_p > 0,
                "detail": (f"VIXY ${vix:.2f} (VIX proxy) | SPY ${spy_p:.2f} "
                           f"({spy_c:+.2f}%) | {len(sectors)} sector ETFs"),
                "latency_ms": ms,
            }
        except Exception as e:
            return {"ok": False, "detail": str(e)[:150], "latency_ms": round((time.time()-t0)*1000)}

    async def _test_anthropic():
        t0 = time.time()
        if not anthropic_key:
            return {"ok": False, "detail": "ANTHROPIC_API_KEY not set", "latency_ms": 0}
        try:
            import anthropic as _ant
            client = _ant.AsyncAnthropic(api_key=anthropic_key)
            resp = await client.messages.create(
                model="claude-haiku-4-5-20251001", max_tokens=10,
                messages=[{"role": "user", "content": "ping"}])
            ms = round((time.time() - t0) * 1000)
            return {"ok": True,
                    "detail": f"Haiku OK | reply: '{resp.content[0].text.strip()[:30]}'",
                    "latency_ms": ms}
        except Exception as e:
            return {"ok": False, "detail": str(e)[:150], "latency_ms": round((time.time()-t0)*1000)}

    async def _test_alpaca_news():
        t0 = time.time()
        try:
            news = await loop.run_in_executor(
                None, lambda: mdata.get_news_sentiment("NVDA", limit=5))
            ms = round((time.time() - t0) * 1000)
            if not news.get("available"):
                return {"ok": False, "detail": "Alpaca news unavailable or key not set", "latency_ms": ms}
            return {
                "ok": True,
                "detail": (
                    f"NVDA: {news.get('total', 0)} articles | "
                    f"score {news.get('score', 0):+.2f} "
                    f"({news.get('positive', 0)}+ / {news.get('negative', 0)}-)"
                ),
                "latency_ms": ms,
            }
        except Exception as e:
            return {"ok": False, "detail": str(e)[:150], "latency_ms": round((time.time()-t0)*1000)}

    async def _test_database():
        t0 = time.time()
        from . import state as _state_mod
        try:
            result = await _aio.to_thread(_state_mod.db_health)
        except Exception as e:
            result = {"ok": False, "detail": str(e)[:150]}
        result["latency_ms"] = round((time.time() - t0) * 1000)
        return result

    hist_r, quotes_r, macro_r, ant_r, news_r, db_r = await _aio.gather(
        _test_alpaca_history(),
        _test_alpaca_quotes(),
        _test_alpaca_macro(),
        _test_anthropic(),
        _test_alpaca_news(),
        _test_database(),
    )
    results["alpaca_history"]   = hist_r
    results["alpaca_quotes"]    = quotes_r
    results["alpaca_macro"]     = macro_r
    results["anthropic"]        = ant_r
    results["alpaca_news"]      = news_r
    results["database"]         = db_r
    results["all_ok"] = all(
        v.get("ok", False)
        for v in results.values()
        if isinstance(v, dict) and "ok" in v
    )
    return results


@app.get("/api/scan-results")
async def get_scan_results():
    s = get_state()
    full = s.get_full_state()
    sym_perf_map = ts.get_symbol_perf(min_trades=1)
    results = list(full.get("last_scan_results", []))
    for r in results:
        sym_stats = sym_perf_map.get(r.get("symbol", ""))
        if sym_stats:
            r["sym_perf"] = sym_stats
    return {
        "cycle":      full.get("last_scan_cycle", 0),
        "scanned_at": full.get("last_scan"),
        "results":    results,
        "regime":     s.market_regime,
    }

@app.get("/api/stats")
async def stats():
    """Trade statistics: win rate, Kelly fraction, expectancy."""
    ot = ts.get_outcome_stats()
    return {
        "performance":    ot,
        "kelly_ready":    ot.get("kelly_ready", False),
        "kelly_fraction": ot.get("kelly_fraction", 0),
    }

@app.get("/api/training-data")
async def training_data(limit: int = 200):
    """Full scan_log table — every decision with context and outcome."""
    from . import training_store as ts
    return {
        "rows":  ts.get_recent(limit=limit),
        "stats": ts.get_stats(),
    }


@app.get("/api/model-insights")
async def model_insights():
    """Structured learning insights: patterns, symbol stats, calibration context."""
    from . import training_store as ts
    return {
        "learned_context":    ts.get_learned_context(min_samples=3),
        "patterns":           ts.get_best_patterns(min_samples=2),
        "symbol_perf":        ts.get_symbol_perf(min_trades=1),
        "stats":              ts.get_stats(),
        "filter_calibration": ts.get_filter_calibration(min_samples=5),
    }


@app.get("/api/sim/reset")
async def sim_reset(api_key: str = ""):
    """
    Clear all sim positions and P&L history. Useful for starting a fresh sim.
    Requires API key (or open if no DASHBOARD_KEY set).
    Does NOT clear the outcome tracker — historical win rate is preserved.
    """
    _check_key(api_key)
    s = get_state()
    result = s.reset_sim()
    cleared_open   = result["cleared_open"]
    cleared_closed = result["cleared_closed"]
    await _broadcast("system", "sim_reset", {
        "message": f"Sim reset: cleared {cleared_open} open + {cleared_closed} closed positions"
    })
    return {
        "status": "reset",
        "cleared_open":   cleared_open,
        "cleared_closed": cleared_closed,
    }

@app.get("/api/sim/clear-closed")
async def sim_clear_closed(api_key: str = ""):
    """Remove all closed positions while keeping open ones."""
    _check_key(api_key)
    s = get_state()
    cleared = s.clear_closed_positions()
    await _broadcast("system", "sim_clear_closed", {
        "message": f"Cleared {cleared} closed position{'s' if cleared != 1 else ''}"
    })
    return {"status": "ok", "cleared_closed": cleared}


class DeletePositionsRequest(BaseModel):
    position_ids: list[str]
    api_key: str = ""

@app.post("/api/sim/delete-positions")
async def sim_delete_positions(req: DeletePositionsRequest):
    """Delete specific positions from both sim_positions and scan_log."""
    _check_key(req.api_key)
    if not req.position_ids:
        return {"status": "ok", "removed": 0}
    s = get_state()
    removed_sim = s.delete_sim_positions(req.position_ids)
    removed_db  = ts.delete_scan_log_trades(req.position_ids)
    removed = removed_sim + removed_db
    await _broadcast("system", "positions_deleted", {
        "message": f"Deleted {removed} position{'s' if removed != 1 else ''}"
    })
    return {"status": "ok", "removed": removed, "sim": removed_sim, "db": removed_db}


def _close_epoch(iso: str) -> int:
    """Epoch seconds for a close timestamp. Handles both naive-ET sim strings
    and tz-aware UTC strings from Postgres; 0 if unparseable."""
    try:
        return int(parse_iso_et(iso).timestamp())
    except (ValueError, TypeError):
        return 0


def _combined_closed_history() -> tuple[list, list, list]:
    """(sim_positions, db_trades, combined_history) — closes across sessions,
    ordered by real chronology (parsed epochs, not string comparison)."""
    s = get_state()
    sim_positions = s.get_sim_positions()
    sim_ids = {p.get("position_id") for p in sim_positions if p.get("position_id")}

    # Historical trades from scan_log (excluding any already in sim_positions)
    db_trades = ts.get_all_closed_trades(exclude_position_ids=sim_ids)

    all_closed = []
    for t in db_trades:
        all_closed.append({
            "closed_at":  t.get("closed_at") or "",
            "pnl_dollars": t.get("pnl_dollars", 0),
            "pnl_pct":    t.get("pnl_pct", 0),
            "symbol":     t.get("symbol", ""),
            "direction":  t.get("direction", ""),
            "position_id": t.get("position_id", ""),
            "source":     "db",
        })
    for p in sim_positions:
        if p.get("status") == "closed":
            all_closed.append({
                "closed_at":  p.get("closed_at") or "",
                "pnl_dollars": float(p.get("pnl_dollars", 0)),
                "pnl_pct":    float(p.get("pnl_pct", 0)),
                "symbol":     p.get("symbol", ""),
                "direction":  p.get("direction", ""),
                "position_id": p.get("position_id", ""),
                "source":     "sim",
            })
    for t in all_closed:
        t["ts_epoch"] = _close_epoch(t["closed_at"])
    all_closed.sort(key=lambda x: x["ts_epoch"])

    cumulative = 0.0
    combined_history = []
    for t in all_closed:
        cumulative += float(t["pnl_dollars"])
        combined_history.append({
            "timestamp":      t["closed_at"],
            "ts_epoch":       t["ts_epoch"],
            "cumulative_pnl": round(cumulative, 2),
            "trade_pnl":      round(float(t["pnl_dollars"]), 2),
            "trade_pnl_pct":  round(float(t["pnl_pct"]), 2),
            "outcome":        "win" if float(t["pnl_dollars"]) > 0 else "loss",
            "symbol":         t["symbol"],
            "direction":      t["direction"],
        })
    return sim_positions, db_trades, combined_history


@app.get("/api/trades/all")
async def get_all_trades():
    """
    Return ALL trades across sessions: current sim_positions + historical scan_log records.
    The scan_log survives sim resets, so this shows the complete trade history.
    """
    sim_positions, db_trades, combined_history = _combined_closed_history()
    return {
        "open_positions":    [p for p in sim_positions if p.get("status") == "open"],
        "closed_positions":  [p for p in sim_positions if p.get("status") == "closed"] + db_trades,
        "db_trades":         db_trades,
        "combined_history":  combined_history,
    }


@app.get("/api/equity")
async def get_equity():
    """
    Equity-curve data for the Trades tab chart:
      snapshots — server-persisted intraday points (5-min cadence, market hours),
                  so the intraday view survives page reloads and restarts
      trades    — per-close cumulative P&L across all sessions (epoch-ordered)
    """
    s = get_state()
    today = now_et().strftime("%Y-%m-%d")
    snapshots = []
    for h in s.get_equity_history():
        t = _close_epoch(h.get("ts", ""))
        if not t:
            continue
        snapshots.append({
            "t":          t,
            "total":      h.get("total", 0.0),
            "realized":   h.get("realized", 0.0),
            "unrealized": h.get("unrealized", 0.0),
            "day":        str(h.get("ts", ""))[:10],
        })
    _, _, combined_history = _combined_closed_history()
    return {
        "snapshots": snapshots,
        "today":     today,
        "trades":    combined_history,
    }


# ── WebSocket ──────────────────────────────────────────────────────────────────

@app.websocket("/ws")
async def websocket_endpoint(ws: WebSocket):
    await manager.connect(ws)
    s = get_state()
    import datetime
    open_positions = s.get_sim_positions(status="open")
    closed_positions = [p for p in s.get_sim_positions() if p.get("status") == "closed"]
    total_pnl = s.cumulative_sim_pnl()
    await ws.send_json({
        "agent": "system", "type": "init",
        "data": {
            "status":        s.system_status,
            "cycle_count":   s.cycle_count,
            "regime":        s.market_regime,
            "open_count":    len(open_positions),
            "total_pnl":     total_pnl,
            "closed_count":  len(closed_positions),
            "recent_events": s.get_full_state().get("event_log", [])[-20:],
        },
        "timestamp": datetime.datetime.now().isoformat(),
    })
    try:
        while True:
            await ws.receive_text()
    except WebSocketDisconnect:
        manager.disconnect(ws)
