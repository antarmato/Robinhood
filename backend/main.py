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
    return HTMLResponse(index.read_text() if index.exists() else "<h1>Dashboard loading...</h1>")

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
        "alpaca_key_prefix": alpaca_key[:6] + "..." if alpaca_key else "NOT SET",
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
    return {
        "system_status":  s.system_status,
        "cycle_count":    s.cycle_count,
        "last_scan":      s.get_full_state().get("last_scan"),
        "last_monitor":   s.get_full_state().get("last_monitor"),
        "active_trades":  len(s.active_trades),
        "pending_proposals": len(s.get_pending_proposals()),
        "pending_exits":  len(s.get_pending_exit_signals()),
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
    losses = [p for p in closed if float(p.get("pnl_dollars", 0)) <= 0]

    total_pnl = s.cumulative_sim_pnl()
    from .outcome_tracker import get_outcome_tracker
    ot_stats = get_outcome_tracker().get_stats()

    # Expectancy: E = WR * avg_win% - (1-WR) * avg_loss%
    avg_win_pct  = ot_stats.get("avg_win_pct", 0)
    avg_loss_pct = ot_stats.get("avg_loss_pct", 0)
    win_rate_frac = ot_stats.get("win_rate", 0)
    expectancy_pct = round(win_rate_frac * avg_win_pct - (1 - win_rate_frac) * avg_loss_pct, 1) if ot_stats else 0

    return {
        "open_positions":  open_,
        "closed_positions": closed[-20:],  # last 20
        "pnl_history":     history,
        "stats": {
            "total_pnl":       total_pnl,
            "total_trades":    len(closed),
            "open_count":      len(open_),
            "win_rate":        round(len(wins) / len(closed) * 100, 1) if closed else 0,
            "avg_win":         round(sum(float(p.get("pnl_dollars", 0)) for p in wins)  / len(wins),   2) if wins   else 0,
            "avg_loss":        round(sum(float(p.get("pnl_dollars", 0)) for p in losses) / len(losses), 2) if losses else 0,
            "best_trade":      round(max((float(p.get("pnl_dollars", 0)) for p in closed), default=0), 2),
            "worst_trade":     round(min((float(p.get("pnl_dollars", 0)) for p in closed), default=0), 2),
            "avg_win_pct":     avg_win_pct,
            "avg_loss_pct":    avg_loss_pct,
            "expectancy_pct":  expectancy_pct,
            "kelly_fraction":  ot_stats.get("kelly_fraction", 0),
        },
    }


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
            "alpaca":   f"✓ {alpaca_key[:6]}..." if alpaca_key else "✗ NOT SET",
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

    hist_r, quotes_r, macro_r, ant_r, news_r = await _aio.gather(
        _test_alpaca_history(),
        _test_alpaca_quotes(),
        _test_alpaca_macro(),
        _test_anthropic(),
        _test_alpaca_news(),
    )
    results["alpaca_history"]   = hist_r
    results["alpaca_quotes"]    = quotes_r
    results["alpaca_macro"]     = macro_r
    results["anthropic"]        = ant_r
    results["alpaca_news"]      = news_r
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
    tracker = get_outcome_tracker()
    results = list(full.get("last_scan_results", []))
    for r in results:
        sym_stats = tracker.get_symbol_stats(r.get("symbol", ""))
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
    tracker = get_outcome_tracker()
    return {
        "performance": tracker.get_stats(),
        "kelly_ready": tracker.is_kelly_ready(),
        "kelly_fraction": tracker.get_kelly_fraction(),
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
    d = s.get_full_state()
    cleared_open   = len([p for p in d.get("sim_positions", []) if p.get("status") == "open"])
    cleared_closed = len([p for p in d.get("sim_positions", []) if p.get("status") == "closed"])
    d["sim_positions"] = []
    d["pnl_history"]   = []
    await _broadcast("system", "sim_reset", {
        "message": f"Sim reset: cleared {cleared_open} open + {cleared_closed} closed positions"
    })
    return {
        "status": "reset",
        "cleared_open":   cleared_open,
        "cleared_closed": cleared_closed,
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
