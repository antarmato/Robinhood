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
    return {
        "open_positions":  open_,
        "closed_positions": closed[-20:],  # last 20
        "pnl_history":     history,
        "stats": {
            "total_pnl":    total_pnl,
            "total_trades": len(closed),
            "open_count":   len(open_),
            "win_rate":     round(len(wins) / len(closed) * 100, 1) if closed else 0,
            "avg_win":      round(sum(float(p.get("pnl_dollars", 0)) for p in wins)  / len(wins),   2) if wins   else 0,
            "avg_loss":     round(sum(float(p.get("pnl_dollars", 0)) for p in losses) / len(losses), 2) if losses else 0,
            "best_trade":   round(max((float(p.get("pnl_dollars", 0)) for p in closed), default=0), 2),
            "worst_trade":  round(min((float(p.get("pnl_dollars", 0)) for p in closed), default=0), 2),
        },
    }


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

# ── WebSocket ──────────────────────────────────────────────────────────────────

@app.websocket("/ws")
async def websocket_endpoint(ws: WebSocket):
    await manager.connect(ws)
    s = get_state()
    import datetime
    await ws.send_json({
        "agent": "system", "type": "init",
        "data": {
            "status":           s.system_status,
            "cycle_count":      s.cycle_count,
            "active_trades":    s.active_trades,
            "pending_proposals": s.get_pending_proposals(),
            "recent_events":    s.get_full_state().get("event_log", [])[-20:],
        },
        "timestamp": datetime.datetime.now().isoformat(),
    })
    try:
        while True:
            await ws.receive_text()
    except WebSocketDisconnect:
        manager.disconnect(ws)
