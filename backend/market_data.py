"""
Market data layer using Polygon.io free tier.
Yahoo Finance returns 429 on cloud IPs — Polygon.io does not.

Required: POLYGON_API_KEY env var (free at polygon.io — sign up takes 2 min).
"""

import logging
import os
from datetime import date, datetime, timedelta
from typing import Optional

import pandas as pd
import requests

logger = logging.getLogger(__name__)

_BASE    = "https://api.polygon.io"
_TIMEOUT = 12

# ── In-memory cache — prevents redundant Polygon calls within a cycle ─────────
# Scanner fetches symbol data, then technical/HV agents re-fetch the same symbol.
# Cache with 10-min TTL eliminates rate-limit 429s and speeds up each cycle.
_HIST_CACHE: dict = {}   # (symbol, period) -> (timestamp, DataFrame)
_INTRA_CACHE: dict = {}  # (symbol, period, interval) -> (timestamp, DataFrame)
_CACHE_TTL = 600         # seconds


def _key() -> str:
    return os.getenv("POLYGON_API_KEY", "")


def _get(path: str, params: dict = None) -> Optional[dict]:
    """GET wrapper with error handling."""
    k = _key()
    if not k:
        logger.error("POLYGON_API_KEY not set — set it in Railway environment variables")
        return None
    p = dict(params or {})
    p["apiKey"] = k
    try:
        r = requests.get(f"{_BASE}{path}", params=p, timeout=_TIMEOUT)
        if r.status_code == 403:
            logger.error("Polygon 403 — API key invalid or subscription limit hit")
            return None
        if r.status_code == 429:
            logger.warning("Polygon 429 — rate limited (free tier: 5 calls/min)")
            return None
        if r.status_code != 200:
            logger.warning(f"Polygon {r.status_code} for {path}")
            return None
        return r.json()
    except Exception as e:
        logger.error(f"Polygon request {path}: {e}")
        return None


# ── Historical OHLCV ──────────────────────────────────────────────────────────

def get_historicals(symbol: str, period: str = "1y") -> pd.DataFrame:
    """
    Daily OHLCV via Polygon aggregates endpoint.
    period: '3mo', '6mo', '1y', '2y' — mapped to calendar days.
    Caches results for 10 minutes — prevents rate-limit 429s when multiple
    agents fetch the same symbol within a single cycle.
    """
    import time as _time
    cache_key = (symbol, period)
    now_ts = _time.time()
    if cache_key in _HIST_CACHE:
        ts, cached_df = _HIST_CACHE[cache_key]
        if now_ts - ts < _CACHE_TTL:
            logger.debug(f"get_historicals({symbol},{period}): cache hit")
            return cached_df.copy()

    days = {"3mo": 95, "6mo": 185, "1y": 370, "2y": 740}.get(period, 370)
    end   = date.today().strftime("%Y-%m-%d")
    start = (date.today() - timedelta(days=days)).strftime("%Y-%m-%d")

    data = _get(f"/v2/aggs/ticker/{symbol}/range/1/day/{start}/{end}",
                {"adjusted": "true", "sort": "asc", "limit": 500})
    if not data or not data.get("results"):
        logger.warning(f"get_historicals({symbol}): no results from Polygon")
        return pd.DataFrame()

    rows = data["results"]
    df = pd.DataFrame({
        "open":   [r.get("o") for r in rows],
        "high":   [r.get("h") for r in rows],
        "low":    [r.get("l") for r in rows],
        "close":  [r.get("c") for r in rows],
        "volume": [r.get("v") for r in rows],
    }, index=pd.to_datetime([r["t"] for r in rows], unit="ms", utc=True).tz_localize(None))

    df = df.dropna(subset=["close"])
    _HIST_CACHE[cache_key] = (_time.time(), df)
    return df.copy()


def get_intraday(symbol: str, period: str = "5d", interval: str = "1h") -> pd.DataFrame:
    """
    Intraday OHLCV via Polygon aggregates.
    interval: '1m', '5m', '15m', '30m', '1h'
    Cached for 10 min — prevents redundant calls within a cycle.
    """
    import time as _time
    cache_key = (symbol, period, interval)
    if cache_key in _INTRA_CACHE:
        ts, cached_df = _INTRA_CACHE[cache_key]
        if _time.time() - ts < _CACHE_TTL:
            return cached_df.copy()

    span_map = {"1m": 1, "5m": 5, "15m": 15, "30m": 30, "1h": 60}
    mult = span_map.get(interval, 60)
    span = "minute" if mult < 60 else "hour"

    days = {"1d": 1, "3d": 3, "5d": 5, "10d": 10}.get(period, 5)
    end   = date.today().strftime("%Y-%m-%d")
    start = (date.today() - timedelta(days=days + 1)).strftime("%Y-%m-%d")

    data = _get(f"/v2/aggs/ticker/{symbol}/range/{mult}/{span}/{start}/{end}",
                {"adjusted": "true", "sort": "asc", "limit": 200})
    if not data or not data.get("results"):
        return pd.DataFrame()

    rows = data["results"]
    df = pd.DataFrame({
        "open":   [r.get("o") for r in rows],
        "high":   [r.get("h") for r in rows],
        "low":    [r.get("l") for r in rows],
        "close":  [r.get("c") for r in rows],
        "volume": [r.get("v") for r in rows],
    }, index=pd.to_datetime([r["t"] for r in rows], unit="ms", utc=True).tz_localize(None))
    result = df.dropna(subset=["close"])
    _INTRA_CACHE[cache_key] = (_time.time(), result)
    return result.copy()


# ── Quotes ────────────────────────────────────────────────────────────────────

def get_batch_quotes(symbols: list) -> dict:
    """Live prices for multiple symbols in one Polygon snapshot call. Returns {symbol: price}."""
    tickers = ",".join(symbols)
    data = _get("/v2/snapshot/locale/us/markets/stocks/tickers", {"tickers": tickers})
    result = {}
    if data and data.get("tickers"):
        for t in data["tickers"]:
            sym = t.get("ticker", "")
            last = (t.get("lastTrade") or {}).get("p", 0)
            day  = t.get("day", {})
            prev = t.get("prevDay", {})
            price = last or day.get("c") or prev.get("c") or 0
            if sym and price:
                result[sym] = float(price)
    return result


def get_quote(symbol: str) -> dict:
    """Current price via Polygon snapshot."""
    data = _get(f"/v2/snapshot/locale/us/markets/stocks/tickers/{symbol}")
    if data and data.get("ticker"):
        t = data["ticker"]
        day = t.get("day", {})
        prev = t.get("prevDay", {})
        price = day.get("c") or prev.get("c") or 0
        prev_close = prev.get("c") or price
        pct = round((price - prev_close) / prev_close * 100, 2) if prev_close else 0
        return {
            "symbol": symbol, "price": price, "prev_close": prev_close,
            "pct_change": pct, "volume": int(day.get("v", 0)),
        }
    return {"symbol": symbol, "price": 0, "prev_close": 0, "pct_change": 0, "volume": 0}


def get_premarket_snapshot(symbol: str) -> dict:
    """
    Overnight gap and pre-market volume via Polygon snapshot.
    'todaysChange' reflects pre-market price vs previous close before 9:30am ET.
    """
    data = _get(f"/v2/snapshot/locale/us/markets/stocks/tickers/{symbol}")
    if not data or not data.get("ticker"):
        return {"symbol": symbol, "gap_pct": 0.0, "gap_direction": "flat",
                "vol_ratio": 1.0, "significant": False}
    t        = data["ticker"]
    day      = t.get("day", {})
    prev     = t.get("prevDay", {})
    current  = day.get("o") or day.get("c") or prev.get("c") or 0
    prev_c   = prev.get("c") or current
    vol_today = float(day.get("v") or 0)
    vol_prev  = float(prev.get("v") or 0)
    gap_pct  = round((current - prev_c) / prev_c * 100, 2) if prev_c else 0.0
    vol_ratio = round(vol_today / vol_prev, 2) if vol_prev > 0 else 1.0
    return {
        "symbol":       symbol,
        "gap_pct":      gap_pct,
        "gap_direction": "up" if gap_pct > 0.5 else ("down" if gap_pct < -0.5 else "flat"),
        "vol_ratio":    vol_ratio,
        "current":      current,
        "prev_close":   prev_c,
        "significant":  abs(gap_pct) >= 1.5 and vol_ratio >= 1.1,
    }


def get_vix() -> float:
    """VIX via Polygon (falls back to 20 if unavailable on free tier)."""
    try:
        # Polygon free tier may not have VIX index — use SPY historicals as proxy
        data = _get("/v2/snapshot/locale/us/markets/stocks/tickers/VIXY")
        if data and data.get("ticker"):
            day = data["ticker"].get("day", {})
            return float(day.get("c", 20.0))
    except Exception:
        pass
    return 20.0


def get_sector_etf_performance() -> dict:
    """Today's % change for major sector ETFs via Polygon snapshot."""
    etf_map = {
        "XLK": "Tech", "XLF": "Financials", "XLE": "Energy",
        "XLV": "Healthcare", "XLY": "Consumer Disc", "XLC": "Comm Services",
        "XLI": "Industrials", "XLB": "Materials",
    }
    tickers = ",".join(etf_map.keys())
    data = _get("/v2/snapshot/locale/us/markets/stocks/tickers",
                {"tickers": tickers})
    result = {}
    if data and data.get("tickers"):
        for t in data["tickers"]:
            sym = t.get("ticker", "")
            if sym in etf_map:
                day  = t.get("day", {})
                prev = t.get("prevDay", {})
                c, p = day.get("c", 0), prev.get("c", 0)
                if c and p:
                    result[etf_map[sym]] = round((c - p) / p * 100, 2)
    return result


# ── Fundamentals ──────────────────────────────────────────────────────────────

def get_fundamentals(symbol: str) -> dict:
    """
    Basic fundamentals from Polygon reference endpoint.
    Note: P/E, earnings dates, analyst data are limited on free tier —
    fields will be None if unavailable.
    """
    base = {
        "symbol": symbol, "sector": "", "industry": "", "market_cap": None,
        "pe_ratio": None, "forward_pe": None, "revenue_growth": None,
        "earnings_growth": None, "short_ratio": None, "beta": None,
        "52w_high": None, "52w_low": None, "avg_volume_10d": None,
        "analyst_target": None, "analyst_rating": "", "analyst_count": None,
        "earnings_date": None, "earnings_ts_start": None, "earnings_ts_end": None,
        "description": "",
    }

    data = _get(f"/v3/reference/tickers/{symbol}")
    if data and data.get("results"):
        r = data["results"]
        base.update({
            "sector":      r.get("sic_description", ""),
            "market_cap":  r.get("market_cap"),
            "description": (r.get("description") or "")[:400],
        })

    # 52-week high/low from daily aggregates
    hist = get_historicals(symbol, period="1y")
    if not hist.empty:
        base["52w_high"] = round(float(hist["close"].max()), 2)
        base["52w_low"]  = round(float(hist["close"].min()), 2)

    return base


# ── Options ───────────────────────────────────────────────────────────────────

_TRADIER_BASE    = "https://sandbox.tradier.com/v1"
_TRADIER_HEADERS = {"Accept": "application/json"}


def _tradier_token() -> str:
    return os.getenv("TRADIER_TOKEN", "")


def _tradier_get(path: str, params: dict = None) -> Optional[dict]:
    token = _tradier_token()
    if not token:
        logger.warning("TRADIER_TOKEN not set — options data unavailable")
        return None
    headers = {**_TRADIER_HEADERS, "Authorization": f"Bearer {token}"}
    try:
        r = requests.get(f"{_TRADIER_BASE}{path}", params=params or {}, headers=headers, timeout=12)
        if r.status_code != 200:
            logger.warning(f"Tradier {r.status_code} for {path}: {r.text[:80]}")
            return None
        return r.json()
    except Exception as e:
        logger.error(f"Tradier request {path}: {e}")
        return None


def get_options_expiration_dates(symbol: str) -> list[str]:
    """Get available option expiration dates via Tradier sandbox."""
    data = _tradier_get("/markets/options/expirations",
                        {"symbol": symbol, "includeAllRoots": "true"})
    if not data:
        return []
    exps = (data.get("expirations") or {}).get("date", [])
    if isinstance(exps, str):
        exps = [exps]
    today = date.today().strftime("%Y-%m-%d")
    return sorted(e for e in exps if e >= today)


def get_options_chain(symbol: str, expiration_date: str, option_type: str = "call") -> list[dict]:
    """
    Options chain for a specific expiry via Tradier sandbox.
    option_type: 'call' or 'put'
    """
    data = _tradier_get("/markets/options/chains",
                        {"symbol": symbol, "expiration": expiration_date, "greeks": "true"})
    if not data:
        return []

    raw_options = (data.get("options") or {}).get("option", [])
    if not raw_options:
        return []
    if isinstance(raw_options, dict):
        raw_options = [raw_options]

    result = []
    for o in raw_options:
        if o.get("option_type", "").lower() != option_type[0].lower():
            continue
        bid  = float(o.get("bid",  0) or 0)
        ask  = float(o.get("ask",  0) or 0)
        last = float(o.get("last", 0) or 0)
        iv   = float(o.get("greeks", {}).get("smv_vol", 0) or o.get("iv", 0) or 0)
        mid  = round((bid + ask) / 2, 2) if bid and ask else last
        result.append({
            "symbol":             symbol,
            "expiration_date":    expiration_date,
            "option_type":        option_type,
            "strike_price":       float(o.get("strike", 0)),
            "bid":                bid,
            "ask":                ask,
            "last":               last,
            "mid":                mid,
            "volume":             int(o.get("volume", 0) or 0),
            "open_interest":      int(o.get("open_interest", 0) or 0),
            "implied_volatility": iv,
            "in_the_money":       o.get("in_the_money", False) == "true" or bool(o.get("in_the_money")),
        })

    return sorted(result, key=lambda x: x["strike_price"])


def get_iv_rank(symbol: str) -> Optional[float]:
    """
    IV rank 0-100: where current ATM IV sits relative to 1-year HV range.
    Computes from historical data — slow but works without extra endpoints.
    """
    try:
        expirations = get_options_expiration_dates(symbol)
        if not expirations:
            return None

        today = date.today()
        target_expiry = None
        for exp in expirations:
            dte = (datetime.strptime(exp, "%Y-%m-%d").date() - today).days
            if 20 <= dte <= 45:
                target_expiry = exp
                break
        if not target_expiry:
            target_expiry = expirations[0]

        q = get_quote(symbol)
        price = q.get("price", 0)
        if not price:
            return None

        chain = get_options_chain(symbol, target_expiry, "call")
        if not chain:
            return None

        atm = min(chain, key=lambda x: abs(x["strike_price"] - price))
        current_iv = atm.get("implied_volatility", 0)
        if not current_iv:
            return None

        hist = get_historicals(symbol, period="1y")
        if hist.empty:
            return None
        hist["returns"] = hist["close"].pct_change()
        hist["hv_21"] = hist["returns"].rolling(21).std() * (252 ** 0.5)
        hv_min = hist["hv_21"].min()
        hv_max = hist["hv_21"].max()
        if hv_max <= hv_min:
            return 50.0
        iv_rank = (current_iv - hv_min) / (hv_max - hv_min) * 100
        return round(max(0.0, min(100.0, iv_rank)), 1)
    except Exception as e:
        logger.error(f"get_iv_rank({symbol}): {e}")
        return None


def get_volume_ratio(symbol: str) -> float:
    """Today's volume vs 30-day average."""
    try:
        hist = get_historicals(symbol, period="3mo")
        if hist.empty or len(hist) < 5:
            return 1.0
        avg = hist["volume"].iloc[:-1].mean()
        today_vol = hist["volume"].iloc[-1]
        return round(today_vol / avg, 2) if avg else 1.0
    except Exception:
        return 1.0


# ── Put/Call Ratio (sentiment) ─────────────────────────────────────────────────

def get_pcr(symbol: str = "SPY") -> dict:
    """
    Put/Call ratio from options snapshot — aggregate volume comparison.
    Falls back to neutral values if options data unavailable.
    """
    neutral = {"pcr_volume": 1.0, "pcr_oi": 1.0, "skew": "neutral"}
    try:
        exps = get_options_expiration_dates(symbol)
        if not exps:
            return neutral

        today = date.today()
        # Pick expiry 20-40 DTE
        target = None
        for e in exps:
            dte = (datetime.strptime(e, "%Y-%m-%d").date() - today).days
            if 20 <= dte <= 40:
                target = e
                break
        if not target:
            target = exps[0]

        calls = get_options_chain(symbol, target, "call")
        puts  = get_options_chain(symbol, target, "put")
        if not calls or not puts:
            return neutral

        call_vol = sum(o["volume"] for o in calls)
        put_vol  = sum(o["volume"] for o in puts)
        call_oi  = sum(o["open_interest"] for o in calls)
        put_oi   = sum(o["open_interest"] for o in puts)

        pcr_v = round(put_vol / call_vol, 2) if call_vol else 1.0
        pcr_o = round(put_oi  / call_oi,  2) if call_oi  else 1.0

        if pcr_v > 1.2:   skew = "bearish"
        elif pcr_v < 0.8: skew = "bullish"
        else:              skew = "neutral"

        return {"pcr_volume": pcr_v, "pcr_oi": pcr_o, "skew": skew}
    except Exception as e:
        logger.error(f"get_pcr({symbol}): {e}")
        return neutral


def get_iv_rank_best(symbol: str) -> float:
    """
    Best available IV rank 0-100.
    Tries Tradier (real options IV) first; falls back to HV rank from Polygon.
    HV rank is a solid proxy — IV and HV are highly correlated.
    Returns 50.0 if no data available (neutral — don't penalize or reward).
    """
    try:
        iv = get_iv_rank(symbol)
        if iv is not None:
            return float(iv)
    except Exception:
        pass
    hv = get_hv(symbol)
    rank = hv.get("hv_rank")
    return float(rank) if rank is not None else 50.0


def get_hv(symbol: str) -> dict:
    """Historical volatility proxy: HV20, HV60, HV rank (0-100), regime."""
    import numpy as np
    try:
        hist = get_historicals(symbol, period="1y")
        if hist.empty or len(hist) < 25:
            return {"hv20": None, "hv60": None, "hv_rank": None, "regime": "unknown"}
        lr = np.log(hist["close"] / hist["close"].shift(1))
        hv20_series = lr.rolling(20).std() * np.sqrt(252) * 100
        hv60_series = lr.rolling(60).std() * np.sqrt(252) * 100
        curr20 = float(hv20_series.iloc[-1])
        curr60_raw = float(hv60_series.iloc[-1])
        curr60 = curr60_raw if not np.isnan(curr60_raw) else curr20
        valid = hv20_series.dropna()
        lo, hi = float(valid.min()), float(valid.max())
        rank = (curr20 - lo) / (hi - lo) * 100 if hi > lo else 50.0
        regime = "high" if rank > 65 else "normal" if rank > 35 else "low"
        return {
            "hv20":    round(curr20, 1),
            "hv60":    round(curr60, 1),
            "hv_rank": round(rank, 1),
            "regime":  regime,
        }
    except Exception as e:
        logger.warning(f"get_hv({symbol}): {e}")
        return {"hv20": None, "hv60": None, "hv_rank": None, "regime": "unknown"}

