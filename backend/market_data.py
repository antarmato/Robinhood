"""
Market data layer using yfinance — free, no credentials required.
Provides quotes, historical OHLCV, intraday data, options chains, and IV rank.
"""

import logging
from datetime import date, datetime, timedelta
from typing import Optional

import yfinance as yf
import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)


# ── Quotes ────────────────────────────────────────────────────────────────────

def get_quote(symbol: str) -> dict:
    """Get a single symbol's current price info."""
    try:
        t = yf.Ticker(symbol)
        info = t.fast_info
        price = float(info.last_price) if hasattr(info, "last_price") else 0
        prev  = float(info.previous_close) if hasattr(info, "previous_close") else price
        return {
            "symbol":     symbol,
            "price":      price,
            "prev_close": prev,
            "pct_change": round((price - prev) / prev * 100, 2) if prev else 0,
            "volume":     int(info.three_month_average_volume) if hasattr(info, "three_month_average_volume") else 0,
        }
    except Exception as e:
        logger.error(f"get_quote({symbol}) error: {e}")
        return {"symbol": symbol, "price": 0, "prev_close": 0, "pct_change": 0, "volume": 0}


def get_vix() -> float:
    """Current VIX level. Returns 20.0 on failure."""
    try:
        t = yf.Ticker("^VIX")
        info = t.fast_info
        return float(info.last_price) if hasattr(info, "last_price") else 20.0
    except Exception:
        return 20.0


def get_sector_etf_performance() -> dict:
    """Today's % change for major sector ETFs."""
    etfs = {
        "XLK": "Tech", "XLF": "Financials", "XLE": "Energy",
        "XLV": "Healthcare", "XLY": "Consumer Disc", "XLC": "Comm Services",
        "XLI": "Industrials", "XLB": "Materials",
    }
    result = {}
    try:
        for etf, sector in etfs.items():
            try:
                t = yf.Ticker(etf)
                hist = t.history(period="2d", interval="1d", auto_adjust=True)
                if len(hist) >= 2:
                    pct = (hist["Close"].iloc[-1] - hist["Close"].iloc[-2]) / hist["Close"].iloc[-2] * 100
                    result[sector] = round(float(pct), 2)
            except Exception:
                pass
    except Exception:
        pass
    return result


# ── Historical OHLCV ─────────────────────────────────────────────────────────

def get_historicals(symbol: str, period: str = "1y") -> pd.DataFrame:
    """Daily OHLCV. period: 1mo, 3mo, 6mo, 1y, 2y."""
    try:
        t = yf.Ticker(symbol)
        df = t.history(period=period, auto_adjust=True)
        if df.empty:
            return pd.DataFrame()
        df.columns = [c.lower() for c in df.columns]
        return df[["open", "close", "high", "low", "volume"]].dropna()
    except Exception as e:
        logger.error(f"get_historicals({symbol}) error: {e}")
        return pd.DataFrame()


def get_intraday(symbol: str, period: str = "5d", interval: str = "1h") -> pd.DataFrame:
    """
    Intraday OHLCV. interval: 1m, 5m, 15m, 30m, 1h.
    period must be ≤ 60d for sub-hour intervals, ≤ 730d for 1h.
    """
    try:
        t = yf.Ticker(symbol)
        df = t.history(period=period, interval=interval, auto_adjust=True)
        if df.empty:
            return pd.DataFrame()
        df.columns = [c.lower() for c in df.columns]
        return df[["open", "close", "high", "low", "volume"]].dropna()
    except Exception as e:
        logger.error(f"get_intraday({symbol}, {interval}) error: {e}")
        return pd.DataFrame()


# ── Fundamentals ──────────────────────────────────────────────────────────────

def get_fundamentals(symbol: str) -> dict:
    """Fundamental info from yfinance."""
    try:
        t = yf.Ticker(symbol)
        info = t.info
        return {
            "symbol":             symbol,
            "sector":             info.get("sector", ""),
            "industry":           info.get("industry", ""),
            "market_cap":         info.get("marketCap"),
            "pe_ratio":           info.get("trailingPE"),
            "forward_pe":         info.get("forwardPE"),
            "revenue_growth":     info.get("revenueGrowth"),
            "earnings_growth":    info.get("earningsGrowth"),
            "short_ratio":        info.get("shortRatio"),
            "beta":               info.get("beta"),
            "52w_high":           info.get("fiftyTwoWeekHigh"),
            "52w_low":            info.get("fiftyTwoWeekLow"),
            "avg_volume_10d":     info.get("averageVolume10days"),
            "analyst_target":     info.get("targetMeanPrice"),
            "analyst_rating":     info.get("recommendationKey", ""),
            "analyst_count":      info.get("numberOfAnalystOpinions"),
            "earnings_date":      _get_next_earnings(info),
            "earnings_ts_start":  info.get("earningsTimestampStart"),
            "earnings_ts_end":    info.get("earningsTimestampEnd"),
            "description":        (info.get("longBusinessSummary") or "")[:400],
        }
    except Exception as e:
        logger.error(f"get_fundamentals({symbol}) error: {e}")
        return {}


def _get_next_earnings(info: dict) -> Optional[str]:
    """Extract next earnings date from yfinance info."""
    try:
        # Try multiple fields — yfinance is inconsistent
        for key in ("earningsTimestamp", "earningsTimestampStart"):
            ts = info.get(key)
            if ts and ts > 0:
                dt = datetime.utcfromtimestamp(ts)
                # Only return if it's a future date
                if dt.date() >= date.today():
                    return dt.strftime("%Y-%m-%d")
    except Exception:
        pass
    return None


# ── Options ───────────────────────────────────────────────────────────────────

def get_options_expiration_dates(symbol: str) -> list[str]:
    """Get available option expiration dates."""
    try:
        t = yf.Ticker(symbol)
        return list(t.options)
    except Exception as e:
        logger.error(f"get_options_expiration_dates({symbol}) error: {e}")
        return []


def get_options_chain(symbol: str, expiration_date: str, option_type: str = "call") -> list[dict]:
    """
    Options chain for a specific expiry.
    option_type: 'call' or 'put'
    Returns list of dicts with strike, bid, ask, iv, volume, oi.
    """
    try:
        t = yf.Ticker(symbol)
        chain = t.option_chain(expiration_date)
        df = chain.calls if option_type == "call" else chain.puts
        if df.empty:
            return []
        result = []
        for _, row in df.iterrows():
            bid = float(row.get("bid", 0) or 0)
            ask = float(row.get("ask", 0) or 0)
            last = float(row.get("lastPrice", 0) or 0)
            iv   = float(row.get("impliedVolatility", 0) or 0)
            result.append({
                "symbol":           symbol,
                "expiration_date":  expiration_date,
                "option_type":      option_type,
                "strike_price":     float(row.get("strike", 0)),
                "bid":              bid,
                "ask":              ask,
                "last":             last,
                "mid":              round((bid + ask) / 2, 2) if bid and ask else last,
                "volume":           int(row.get("volume", 0) or 0),
                "open_interest":    int(row.get("openInterest", 0) or 0),
                "implied_volatility": iv,
                "in_the_money":     bool(row.get("inTheMoney", False)),
            })
        return result
    except Exception as e:
        logger.error(f"get_options_chain({symbol}, {expiration_date}) error: {e}")
        return []


def get_iv_rank(symbol: str) -> Optional[float]:
    """
    IV rank 0-100: where current ATM IV sits relative to its 1-year HV range.
    NOTE: This is slow (4 yfinance calls). Use sparingly — only on finalists.
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
        logger.error(f"get_iv_rank({symbol}) error: {e}")
        return None


def get_volume_ratio(symbol: str) -> float:
    """Today's volume vs 30-day average."""
    try:
        hist = get_historicals(symbol, period="3mo")
        if hist.empty or len(hist) < 5:
            return 1.0
        avg = hist["volume"].iloc[:-1].mean()
        today = hist["volume"].iloc[-1]
        return round(today / avg, 2) if avg else 1.0
    except Exception:
        return 1.0
