"""
Robinhood API wrapper using robin_stocks.
Handles auth, quotes, options chains, order placement, and positions.
"""

import os
import json
import asyncio
import logging
from datetime import datetime, date
from typing import Optional
from pathlib import Path

import robin_stocks.robinhood as rh
import pandas as pd

logger = logging.getLogger(__name__)

AUTH_FILE = Path("/app/data/rh_auth.json")


class RobinhoodClient:
    def __init__(self):
        self._logged_in = False

    # ── Authentication ─────────────────────────────────────────────────────────

    def login(self, username: str, password: str, mfa_code: Optional[str] = None) -> bool:
        """Login to Robinhood. Returns True on success."""
        try:
            kwargs = {
                "username": username,
                "password": password,
                "store_session": True,
                "pickle_name": str(AUTH_FILE),
            }
            if mfa_code:
                kwargs["mfa_code"] = mfa_code

            result = rh.login(**kwargs)
            self._logged_in = bool(result)
            if self._logged_in:
                logger.info("Robinhood login successful")
            return self._logged_in
        except Exception as e:
            logger.error(f"Robinhood login failed: {e}")
            raise

    def logout(self):
        rh.logout()
        self._logged_in = False

    def is_logged_in(self) -> bool:
        return self._logged_in

    # ── Market data ────────────────────────────────────────────────────────────

    def get_quotes(self, symbols: list[str]) -> dict:
        """Get current quotes for a list of symbols."""
        try:
            data = rh.stocks.get_quotes(symbols)
            result = {}
            for i, sym in enumerate(symbols):
                q = data[i] if data and i < len(data) else {}
                if q:
                    result[sym] = {
                        "symbol": sym,
                        "price": float(q.get("last_trade_price") or q.get("last_extended_hours_trade_price") or 0),
                        "bid": float(q.get("bid_price") or 0),
                        "ask": float(q.get("ask_price") or 0),
                        "volume": int(float(q.get("volume") or 0)),
                        "high": float(q.get("high_price") or 0),
                        "low": float(q.get("low_price") or 0),
                        "open": float(q.get("open") or q.get("adjusted_previous_close") or 0),
                        "prev_close": float(q.get("previous_close") or 0),
                    }
            return result
        except Exception as e:
            logger.error(f"get_quotes error: {e}")
            return {}

    def get_historicals(self, symbol: str, interval: str = "day", span: str = "3month") -> pd.DataFrame:
        """
        Get OHLCV historical data.
        interval: 5minute, 10minute, hour, day, week
        span: day, week, month, 3month, year, 5year
        """
        try:
            data = rh.stocks.get_stock_historicals(symbol, interval=interval, span=span)
            if not data:
                return pd.DataFrame()
            df = pd.DataFrame(data)
            df["begins_at"] = pd.to_datetime(df["begins_at"])
            df.set_index("begins_at", inplace=True)
            for col in ["open_price", "close_price", "high_price", "low_price", "volume"]:
                df[col] = pd.to_numeric(df[col], errors="coerce")
            df.rename(columns={
                "open_price": "open",
                "close_price": "close",
                "high_price": "high",
                "low_price": "low",
            }, inplace=True)
            return df.dropna()
        except Exception as e:
            logger.error(f"get_historicals({symbol}) error: {e}")
            return pd.DataFrame()

    def get_fundamentals(self, symbol: str) -> dict:
        """Get fundamental data for a stock."""
        try:
            data = rh.stocks.get_fundamentals(symbol)
            if data and data[0]:
                d = data[0]
                return {
                    "symbol": symbol,
                    "pe_ratio": d.get("pe_ratio"),
                    "market_cap": d.get("market_cap"),
                    "volume": d.get("volume"),
                    "average_volume": d.get("average_volume"),
                    "high_52_weeks": d.get("high_52_weeks"),
                    "low_52_weeks": d.get("low_52_weeks"),
                    "description": d.get("description", "")[:500],
                }
            return {}
        except Exception as e:
            logger.error(f"get_fundamentals({symbol}) error: {e}")
            return {}

    # ── Options ────────────────────────────────────────────────────────────────

    def get_options_expiration_dates(self, symbol: str) -> list[str]:
        """Get available expiration dates for options on a symbol."""
        try:
            chains = rh.options.get_chains(symbol)
            if chains and "expiration_dates" in chains:
                return chains["expiration_dates"]
            return []
        except Exception as e:
            logger.error(f"get_options_expiration_dates({symbol}) error: {e}")
            return []

    def get_options_chain(self, symbol: str, expiration_date: str, option_type: str = "call") -> list[dict]:
        """
        Get options chain for a symbol/expiry/type.
        option_type: 'call' or 'put'
        """
        try:
            data = rh.options.find_options_by_expiration(
                symbol,
                expirationDate=expiration_date,
                optionType=option_type,
                info=None
            )
            if not data:
                return []
            result = []
            for o in data:
                try:
                    result.append({
                        "symbol": symbol,
                        "expiration_date": expiration_date,
                        "option_type": option_type,
                        "strike_price": float(o.get("strike_price") or 0),
                        "bid": float(o.get("bid_price") or 0),
                        "ask": float(o.get("ask_price") or 0),
                        "last": float(o.get("last_trade_price") or 0),
                        "volume": int(float(o.get("volume") or 0)),
                        "open_interest": int(float(o.get("open_interest") or 0)),
                        "implied_volatility": float(o.get("implied_volatility") or 0),
                        "delta": float(o.get("delta") or 0),
                        "gamma": float(o.get("gamma") or 0),
                        "theta": float(o.get("theta") or 0),
                        "vega": float(o.get("vega") or 0),
                        "rho": float(o.get("rho") or 0),
                        "instrument_url": o.get("url", ""),
                        "id": o.get("id", ""),
                    })
                except (TypeError, ValueError):
                    continue
            return result
        except Exception as e:
            logger.error(f"get_options_chain({symbol}, {expiration_date}) error: {e}")
            return []

    def get_iv_rank(self, symbol: str) -> Optional[float]:
        """
        Approximate IV rank: current ATM IV vs 52-week range of historical volatility.
        Returns 0-100 float.
        """
        try:
            expirations = self.get_options_expiration_dates(symbol)
            if not expirations:
                return None
            # Find expiry ~30 DTE
            today = date.today()
            target_expiry = None
            for exp in expirations:
                exp_date = datetime.strptime(exp, "%Y-%m-%d").date()
                dte = (exp_date - today).days
                if 20 <= dte <= 45:
                    target_expiry = exp
                    break
            if not target_expiry:
                target_expiry = expirations[0]

            # Get ATM call IV
            quotes = self.get_quotes([symbol])
            if symbol not in quotes:
                return None
            current_price = quotes[symbol]["price"]
            chain = self.get_options_chain(symbol, target_expiry, "call")
            if not chain:
                return None
            # Find ATM
            atm = min(chain, key=lambda x: abs(x["strike_price"] - current_price))
            current_iv = atm["implied_volatility"]
            if not current_iv:
                return None

            # Estimate HV from historical data to compute rank
            hist = self.get_historicals(symbol, interval="day", span="year")
            if hist.empty:
                return None
            hist["returns"] = hist["close"].pct_change()
            hist["hv_30"] = hist["returns"].rolling(21).std() * (252 ** 0.5)
            hv_min = hist["hv_30"].min()
            hv_max = hist["hv_30"].max()
            if hv_max == hv_min:
                return 50.0
            iv_rank = ((current_iv - hv_min) / (hv_max - hv_min)) * 100
            return max(0.0, min(100.0, iv_rank))
        except Exception as e:
            logger.error(f"get_iv_rank({symbol}) error: {e}")
            return None

    # ── Orders ─────────────────────────────────────────────────────────────────

    def place_debit_spread(
        self,
        symbol: str,
        expiration_date: str,
        long_strike: float,
        short_strike: float,
        option_type: str,  # 'call' or 'put'
        quantity: int,
        debit_limit: float,  # max debit willing to pay per spread
    ) -> dict:
        """
        Place a vertical debit spread.
        Bull call: long lower strike, short higher strike.
        Bear put: long higher strike, short lower strike.
        Returns order result dict.
        """
        try:
            result = rh.options.order_option_spread(
                direction="debit",
                netAmount=str(round(debit_limit, 2)),
                symbol=symbol,
                quantity=quantity,
                spread_type="vertical" if option_type == "call" else "vertical",
                expiration_date=expiration_date,
                long_strike_price=str(long_strike),
                short_strike_price=str(short_strike),
                option_type=option_type,
            )
            return result or {}
        except Exception as e:
            logger.error(f"place_debit_spread error: {e}")
            raise

    def get_open_option_positions(self) -> list[dict]:
        """Get all open option positions."""
        try:
            positions = rh.options.get_open_option_positions()
            if not positions:
                return []
            result = []
            for p in positions:
                try:
                    result.append({
                        "symbol": p.get("chain_symbol", ""),
                        "type": p.get("type", ""),
                        "quantity": float(p.get("quantity") or 0),
                        "average_price": float(p.get("average_price") or 0),
                        "option_type": p.get("option_type", ""),
                        "strike_price": float(p.get("strike_price") or 0),
                        "expiration_date": p.get("expiration_date", ""),
                        "id": p.get("option_id", ""),
                        "instrument_url": p.get("option", ""),
                    })
                except (TypeError, ValueError):
                    continue
            return result
        except Exception as e:
            logger.error(f"get_open_option_positions error: {e}")
            return []

    def get_option_market_data(self, option_id: str) -> dict:
        """Get current market data for an option by ID."""
        try:
            data = rh.options.get_option_market_data_by_id(option_id)
            if data:
                return {
                    "bid": float(data.get("bid_price") or 0),
                    "ask": float(data.get("ask_price") or 0),
                    "last": float(data.get("last_trade_price") or 0),
                    "delta": float(data.get("delta") or 0),
                    "implied_volatility": float(data.get("implied_volatility") or 0),
                }
            return {}
        except Exception as e:
            logger.error(f"get_option_market_data({option_id}) error: {e}")
            return {}

    def cancel_option_order(self, order_id: str) -> bool:
        """Cancel a pending option order."""
        try:
            rh.options.cancel_option_order(order_id)
            return True
        except Exception as e:
            logger.error(f"cancel_option_order({order_id}) error: {e}")
            return False

    def get_account_info(self) -> dict:
        """Get account buying power and portfolio value."""
        try:
            profile = rh.profiles.load_account_profile()
            portfolio = rh.profiles.load_portfolio_profile()
            return {
                "buying_power": float(profile.get("buying_power") or 0) if profile else 0,
                "portfolio_value": float(portfolio.get("equity") or 0) if portfolio else 0,
                "cash": float(profile.get("cash") or 0) if profile else 0,
            }
        except Exception as e:
            logger.error(f"get_account_info error: {e}")
            return {"buying_power": 0, "portfolio_value": 0, "cash": 0}


# Singleton
_client = RobinhoodClient()


def get_rh_client() -> RobinhoodClient:
    return _client
