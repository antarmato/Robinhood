"""
Market data layer using yfinance — free, no credentials required.
Provides quotes, historical OHLCV, options chains, and IV rank.
"""

import logging
from datetime import date, datetime, timedelta
from typing import Optional

import yfinance as yf
import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)


def get_quotes(symbols: list[str]) -> dict:
    """Get current quotes for multiple symbols."""
    try:
        tickers = yf.download(symbols, period="2d", interval="1d", progress=False, auto_adjust=True)
        result = {}
        for sym in symbols:
            try:
                t = yf.Ticker(sym)
                info = t.fast_info
                close = float(info.last_price) if hasattr(info, 'last_price') else 0
                prev_close = float(info.previous_close) if hasattr(info, 'previous_close') else close
                result[sym] = {
                    "symbol": sym,
                    "price": close,
                    "prev_close": prev_close,
                    "volume": int(info.three_month_average_volume) if hasattr(info, 'three_month_average_volume') else 0,
                }
            except Exception as e:
                logger.debug(f"Quote error for {sym}: {e}")
        return result
    except Exception as e:
        logger.error(f"get_quotes error: {e}")
        return {}


def get_quote(symbol: str) -> dict:
    """Get a single symbol's current price info."""
    try:
        t = yf.Ticker(symbol)
        info = t.fast_info
        price = float(info.last_price) if hasattr(info, 'last_price') else 0
        prev = float(info.previous_close) if hasattr(info, 'previous_close') else price
        return {
            "symbol": symbol,
            "price": price,
            "prev_close": prev,
            "pct_change": round((price - prev) / prev * 100, 2) if prev else 0,
            "volume": int(info.three_month_average_volume) if hasattr(info, 'three_month_average_volume') else 0,
        }
    except Exception as e:
        logger.error(f"get_quote({symbol}) error: {e}")
        return {"symbol": symbol, "price": 0, "prev_close": 0, "pct_change": 0, "volume": 0}


def get_historicals(symbol: str, period: str = "1y") -> pd.DataFrame:
    """
    Get OHLCV daily history.
    period: 1mo, 3mo, 6mo, 1y, 2y
    """
    try:
        t = yf.Ticker(symbol)
        df = t.history(period=period, auto_adjust=True)
        if df.empty:
            return pd.DataFrame()
        df.index = pd.to_datetime(df.index)
        df.columns = [c.lower() for c in df.columns]
        return df[["open", "close", "high", "low", "volume"]].dropna()
    except Exception as e:
        logger.error(f"get_historicals({symbol}) error: {e}")
        return pd.DataFrame()


def get_fundamentals(symbol: str) -> dict:
    """Get fundamental info from yfinance."""
    try:
        t = yf.Ticker(symbol)
        info = t.info
        return {
            "symbol": symbol,
            "sector": info.get("sector", ""),
            "industry": info.get("industry", ""),
            "market_cap": info.get("marketCap"),
            "pe_ratio": info.get("trailingPE"),
            "forward_pe": info.get("forwardPE"),
            "revenue_growth": info.get("revenueGrowth"),
            "earnings_growth": info.get("earningsGrowth"),
            "short_ratio": info.get("shortRatio"),
            "beta": info.get("beta"),
            "52w_high": info.get("fiftyTwoWeekHigh"),
            "52w_low": info.get("fiftyTwoWeekLow"),
            "avg_volume_10d": info.get("averageVolume10days"),
            "earnings_date": _get_next_earnings(info),
            "description": (info.get("longBusinessSummary") or "")[:400],
        }
    except Exception as e:
        logger.error(f"get_fundamentals({symbol}) error: {e}")
        return {}


def _get_next_earnings(info: dict) -> Optional[str]:
    """Extract next earnings date from yfinance info."""
    try:
        ts = info.get("earningsTimestamp") or info.get("earningsTimestampStart")
        if ts:
            return datetime.utcfromtimestamp(ts).strftime("%Y-%m-%d")
    except Exception:
        pass
    return None


def get_options_expiration_dates(symbol: str) -> list[str]:
    """Get available expiration dates."""
    try:
        t = yf.Ticker(symbol)
        return list(t.options)
    except Exception as e:
        logger.error(f"get_options_expiration_dates({symbol}) error: {e}")
        return []


def get_options_chain(symbol: str, expiration_date: str, option_type: str = "call") -> list[dict]:
    """
    Get options chain for a specific expiry.
    option_type: 'call' or 'put'
    Returns list of option dicts with strike, bid, ask, iv, volume, oi.
    """
    try:
        t = yf.Ticker(symbol)
        chain = t.option_chain(expiration_date)
        df = chain.calls if option_type == "call" else chain.puts
        if df.empty:
            return []
        result = []
        for _, row in df.iterrows():
            result.append({
                "symbol": symbol,
                "expiration_date": expiration_date,
                "option_type": option_type,
                "strike_price": float(row.get("strike", 0)),
                "bid": float(row.get("bid", 0) or 0),
                "ask": float(row.get("ask", 0) or 0),
                "last": float(row.get("lastPrice", 0) or 0),
                "volume": int(row.get("volume", 0) or 0),
                "open_interest": int(row.get("openInterest", 0) or 0),
                "implied_volatility": float(row.get("impliedVolatility", 0) or 0),
                "in_the_money": bool(row.get("inTheMoney", False)),
            })
        return result
    except Exception as e:
        logger.error(f"get_options_chain({symbol}, {expiration_date}) error: {e}")
        return []


def get_iv_rank(symbol: str) -> Optional[float]:
    """
    Estimate IV rank (0-100) using ATM implied volatility vs 1-year HV range.
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

        # Historical vol range
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
