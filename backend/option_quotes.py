"""
Real option quotes via Alpaca's options data API (indicative feed, free tier).

Used to (a) pick an actual contract at entry — nearest ~25-delta strike in the
21-45 DTE window — and (b) mark open positions to the contract's real bid.
Every caller must fall back to the synthetic model in pricing.py when quotes
are unavailable (missing keys, unknown symbol, feed outage): a position with
an `occ_symbol` marks real when possible and falls back to the model, while
positions without one behave exactly as before.
"""

import logging
import os
from datetime import date, datetime, timedelta

import requests

logger = logging.getLogger(__name__)

_BASE    = "https://data.alpaca.markets"
_TIMEOUT = 12
_FEED    = os.getenv("ALPACA_OPTIONS_FEED", "indicative")  # opra needs a paid sub

# Selection sanity bounds
_MAX_REL_SPREAD = 0.35   # reject quotes where (ask-bid)/mid exceeds this
_TARGET_DELTA   = 0.25


def _headers() -> dict:
    return {
        "APCA-API-KEY-ID":     os.getenv("ALPACA_API_KEY", ""),
        "APCA-API-SECRET-KEY": os.getenv("ALPACA_API_SECRET", ""),
        "Accept": "application/json",
    }


def _available() -> bool:
    return bool(os.getenv("ALPACA_API_KEY", "")) and bool(os.getenv("ALPACA_API_SECRET", ""))


def parse_occ(occ: str) -> dict | None:
    """AAPL240621C00100000 → {underlying, expiry(date), type, strike}."""
    try:
        # Strike = last 8 digits, type = 1 char before, expiry = 6 digits before that
        strike = int(occ[-8:]) / 1000.0
        opt_type = occ[-9].upper()
        yy, mm, dd = int(occ[-15:-13]), int(occ[-13:-11]), int(occ[-11:-9])
        return {
            "underlying": occ[:-15],
            "expiry":     date(2000 + yy, mm, dd),
            "type":       "call" if opt_type == "C" else "put",
            "strike":     strike,
        }
    except Exception:
        return None


def dte_left(occ: str, today: date | None = None) -> int | None:
    meta = parse_occ(occ)
    if not meta:
        return None
    return max(0, (meta["expiry"] - (today or date.today())).days)


def get_chain_snapshot(underlying: str, opt_type: str, spot: float,
                       dte_min: int = 21, dte_max: int = 45) -> dict:
    """
    Chain snapshots (quotes + greeks) for one underlying/type in the DTE window,
    strike-windowed around spot so one page covers the ~25-delta region.
    Returns {occ_symbol: snapshot} ({} on any failure).
    """
    if not _available():
        return {}
    today = date.today()
    if opt_type == "call":
        lo, hi = spot * 0.97, spot * 1.35
    else:
        lo, hi = spot * 0.65, spot * 1.03
    params = {
        "feed": _FEED,
        "type": opt_type,
        "limit": 1000,
        "strike_price_gte": round(lo, 2),
        "strike_price_lte": round(hi, 2),
        "expiration_date_gte": (today + timedelta(days=dte_min)).isoformat(),
        "expiration_date_lte": (today + timedelta(days=dte_max)).isoformat(),
    }
    try:
        r = requests.get(f"{_BASE}/v1beta1/options/snapshots/{underlying}",
                         params=params, headers=_headers(), timeout=_TIMEOUT)
        if r.status_code != 200:
            logger.warning(f"Alpaca options snapshots {r.status_code} for {underlying}: {r.text[:200]}")
            return {}
        return r.json().get("snapshots") or {}
    except Exception as e:
        logger.warning(f"Alpaca options snapshots failed for {underlying}: {e}")
        return {}


def _quote_ok(bid: float, ask: float) -> bool:
    if bid <= 0 or ask <= 0 or ask < bid:
        return False
    mid = (bid + ask) / 2.0
    return (ask - bid) / mid <= _MAX_REL_SPREAD


def select_contract(snapshots: dict, target_dte: int, today: date | None = None,
                    target_delta: float = _TARGET_DELTA) -> dict | None:
    """
    Pure selection: from {occ: snapshot}, pick the contract at the expiry
    closest to target_dte whose |delta| is closest to target_delta, requiring
    a sane two-sided quote. Returns
    {occ_symbol, bid, ask, delta, iv, strike, expiry, dte} or None.
    """
    today = today or date.today()
    candidates = []
    for occ, snap in snapshots.items():
        meta = parse_occ(occ)
        if not meta:
            continue
        q = snap.get("latestQuote") or {}
        bid, ask = float(q.get("bp") or 0), float(q.get("ap") or 0)
        if not _quote_ok(bid, ask):
            continue
        greeks = snap.get("greeks") or {}
        delta = greeks.get("delta")
        if delta is None:
            continue
        candidates.append({
            "occ_symbol": occ,
            "bid": bid, "ask": ask,
            "delta": abs(float(delta)),
            "iv": snap.get("impliedVolatility"),
            "strike": meta["strike"],
            "expiry": meta["expiry"].isoformat(),
            "dte": (meta["expiry"] - today).days,
        })
    if not candidates:
        return None
    best_dte = min({c["dte"] for c in candidates}, key=lambda d: abs(d - target_dte))
    at_expiry = [c for c in candidates if c["dte"] == best_dte]
    return min(at_expiry, key=lambda c: abs(c["delta"] - target_delta))


def find_entry_contract(underlying: str, opt_type: str, spot: float,
                        target_dte: int, dte_min: int = 21, dte_max: int = 45) -> dict | None:
    """Chain fetch + selection. None means: use the synthetic model."""
    snaps = get_chain_snapshot(underlying, opt_type, spot, dte_min, dte_max)
    if not snaps:
        return None
    pick = select_contract(snaps, target_dte)
    if pick:
        logger.info(f"Real contract for {underlying}: {pick['occ_symbol']} "
                    f"delta={pick['delta']:.2f} bid={pick['bid']} ask={pick['ask']}")
    return pick


def get_latest_quotes(occ_symbols: list[str]) -> dict:
    """Batch latest quotes: {occ_symbol: {bid, ask}}. {} / missing on failure."""
    if not occ_symbols or not _available():
        return {}
    try:
        r = requests.get(f"{_BASE}/v1beta1/options/quotes/latest",
                         params={"symbols": ",".join(occ_symbols), "feed": _FEED},
                         headers=_headers(), timeout=_TIMEOUT)
        if r.status_code != 200:
            logger.warning(f"Alpaca options quotes {r.status_code}: {r.text[:200]}")
            return {}
        out = {}
        for occ, q in (r.json().get("quotes") or {}).items():
            bid, ask = float(q.get("bp") or 0), float(q.get("ap") or 0)
            if bid > 0:
                out[occ] = {"bid": bid, "ask": ask}
        return out
    except Exception as e:
        logger.warning(f"Alpaca options quotes failed: {e}")
        return {}
