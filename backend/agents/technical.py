"""
Technical Analysis Agent — evaluates momentum, trend, support/resistance.
Uses yfinance OHLCV + pandas for indicator computation.
"""

import logging
from typing import Optional

import anthropic
import pandas as pd

from .base import BaseAgent, BroadcastFn
from .. import market_data as md

logger = logging.getLogger(__name__)


class TechnicalAgent(BaseAgent):
    def __init__(
        self,
        client: anthropic.AsyncAnthropic,
        broadcast: Optional[BroadcastFn] = None,
    ):
        super().__init__(client, "Technical", model="claude-sonnet-4-6", broadcast=broadcast)

    async def analyze(self, symbol: str, direction: str) -> dict:
        await self._emit("status", f"Running technical analysis on {symbol}...")

        df = md.get_historicals(symbol, period="1y")
        if df.empty:
            return {"score": 5, "signal": "neutral", "summary": "No historical data available."}

        indicators = self._compute_indicators(df)
        text = self._format(symbol, df, indicators)

        system = f"""You are a technical analysis expert evaluating a {'bullish call' if direction == 'bullish' else 'bearish put'} option setup on {symbol}.

Assess whether the technical picture supports buying a {direction} option.
Consider: trend direction, momentum, RSI extremes, MACD signal, Bollinger Band position.

Respond ONLY with JSON:
{{
  "score": <1-10, where 10=strong confirmation of {direction} direction>,
  "signal": "bullish" | "bearish" | "neutral",
  "trend": "uptrend" | "downtrend" | "ranging",
  "rsi_reading": "<value and interpretation>",
  "macd_reading": "<bullish cross/bearish cross/neutral>",
  "key_support": <price>,
  "key_resistance": <price>,
  "summary": "<2-3 sentence technical assessment>"
}}"""

        raw = await self._call(system, [{"role": "user", "content": text}], max_tokens=512)
        result = self._parse_json(raw)
        result.setdefault("score", 5)
        result.setdefault("signal", "neutral")
        result.setdefault("summary", "Technical analysis complete.")
        return result

    def _compute_indicators(self, df: pd.DataFrame) -> dict:
        close = df["close"]
        ind = {}
        ind["ema_20"]  = close.ewm(span=20).mean().iloc[-1]
        ind["ema_50"]  = close.ewm(span=50).mean().iloc[-1]
        ind["ema_200"] = close.ewm(span=200).mean().iloc[-1]

        delta = close.diff()
        gain = delta.where(delta > 0, 0).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
        ind["rsi"] = (100 - 100 / (1 + gain / loss)).iloc[-1]

        ema12 = close.ewm(span=12).mean()
        ema26 = close.ewm(span=26).mean()
        macd = ema12 - ema26
        sig  = macd.ewm(span=9).mean()
        ind["macd"] = macd.iloc[-1]
        ind["macd_signal"] = sig.iloc[-1]
        ind["macd_hist"] = (macd - sig).iloc[-1]

        sma20 = close.rolling(20).mean()
        std20 = close.rolling(20).std()
        ind["bb_upper"] = (sma20 + 2 * std20).iloc[-1]
        ind["bb_lower"] = (sma20 - 2 * std20).iloc[-1]

        hi, lo = df["high"], df["low"]
        tr = pd.concat([(hi - lo), (hi - close.shift()).abs(), (lo - close.shift()).abs()], axis=1).max(axis=1)
        ind["atr"] = tr.rolling(14).mean().iloc[-1]

        recent = df.tail(30)
        ind["recent_high"] = recent["high"].max()
        ind["recent_low"]  = recent["low"].min()
        ind["price"] = close.iloc[-1]
        return ind

    def _format(self, symbol: str, df: pd.DataFrame, i: dict) -> str:
        p = i["price"]
        return f"""Technical data for {symbol} — Current: ${p:.2f}

Moving Averages:
  EMA(20):  ${i['ema_20']:.2f}  {'↑ above' if p > i['ema_20'] else '↓ below'}
  EMA(50):  ${i['ema_50']:.2f}  {'↑ above' if p > i['ema_50'] else '↓ below'}
  EMA(200): ${i['ema_200']:.2f}  {'↑ above' if p > i['ema_200'] else '↓ below'}

Momentum:
  RSI(14): {i['rsi']:.1f}  ({'Overbought' if i['rsi'] > 70 else 'Oversold' if i['rsi'] < 30 else 'Neutral'})
  MACD: {i['macd']:.4f}  Signal: {i['macd_signal']:.4f}  Hist: {i['macd_hist']:.4f}
  MACD cross: {'Bullish' if i['macd'] > i['macd_signal'] else 'Bearish'}

Volatility:
  ATR(14): ${i['atr']:.2f}
  BB Upper: ${i['bb_upper']:.2f}  BB Lower: ${i['bb_lower']:.2f}
  BB Position: {'Upper zone' if p > i['bb_upper'] else 'Lower zone' if p < i['bb_lower'] else 'Mid zone'}

Key Levels (30-day):
  High: ${i['recent_high']:.2f}  Low: ${i['recent_low']:.2f}

Last 5 closes: {', '.join(f'${v:.2f}' for v in df['close'].tail(5).tolist())}"""
