"""Market-data providers. All return a UTC-indexed OHLCV DataFrame, oldest first."""
from __future__ import annotations

import logging
import os
import zlib
from abc import ABC, abstractmethod

import numpy as np
import pandas as pd

from ..http import get_json, session, TIMEOUT

log = logging.getLogger(__name__)
COLUMNS = ["open", "high", "low", "close", "volume"]


class MarketData(ABC):
    @abstractmethod
    def candles(self, symbol: str, timeframe: str = "1h", limit: int = 300) -> pd.DataFrame:
        ...

    @staticmethod
    def _frame(rows, unit: str = "ms") -> pd.DataFrame:
        df = pd.DataFrame(rows, columns=["ts", *COLUMNS])
        df["ts"] = pd.to_datetime(df["ts"].astype("int64"), unit=unit, utc=True)
        df = df.set_index("ts").astype(float).sort_index()
        df = df[~df.index.duplicated(keep="last")]
        return df.dropna(subset=["open", "high", "low", "close"])


class YahooData(MarketData):
    """Yahoo Finance chart API. Good for gold (GC=F) and forex (EURUSD=X)."""

    URL = "https://query1.finance.yahoo.com/v8/finance/chart/{symbol}"
    _RANGES = {"5m": "30d", "15m": "30d", "30m": "30d", "1h": "180d", "4h": "720d", "1d": "5y"}

    def candles(self, symbol, timeframe="1h", limit=300):
        interval = "1h" if timeframe == "4h" else timeframe
        data = get_json(
            self.URL.format(symbol=symbol),
            params={"interval": interval, "range": self._RANGES.get(timeframe, "180d")},
        )
        res = data["chart"]["result"][0]
        q = res["indicators"]["quote"][0]
        rows = zip(res["timestamp"], q["open"], q["high"], q["low"], q["close"], q["volume"])
        rows = [[t, o, h, lo, c, v or 0] for t, o, h, lo, c, v in rows]
        df = self._frame(rows, unit="s")
        if timeframe == "4h":
            df = resample(df, "4h")
        return df.tail(limit)


class OKXData(MarketData):
    """OKX public candles (crypto). Symbols like BTC-USDT."""

    URL = "https://www.okx.com/api/v5/market/history-candles"
    _BARS = {"1m": "1m", "5m": "5m", "15m": "15m", "30m": "30m", "1h": "1H", "4h": "4H", "1d": "1Dutc"}

    def candles(self, symbol, timeframe="1h", limit=300):
        rows, after = [], None
        while len(rows) < limit:
            params = {"instId": symbol, "bar": self._BARS[timeframe], "limit": 100}
            if after:
                params["after"] = after
            data = get_json(self.URL, params=params)["data"]
            if not data:
                break
            rows.extend(r[:6] for r in data)
            after = data[-1][0]
        return self._frame(rows).tail(limit)


class OandaData(MarketData):
    """OANDA v20 candles. Needs OANDA_API_KEY; instruments like XAU_USD."""

    _GRAN = {"1m": "M1", "5m": "M5", "15m": "M15", "30m": "M30", "1h": "H1", "4h": "H4", "1d": "D"}

    def __init__(self) -> None:
        env = os.getenv("OANDA_ENV", "practice")
        self.base = f"https://api-fx{'trade' if env == 'live' else 'practice'}.oanda.com/v3"
        self.token = os.environ["OANDA_API_KEY"]

    def candles(self, symbol, timeframe="1h", limit=300):
        r = session().get(
            f"{self.base}/instruments/{symbol}/candles",
            params={"granularity": self._GRAN[timeframe], "count": min(limit, 5000), "price": "M"},
            headers={"Authorization": f"Bearer {self.token}"},
            timeout=TIMEOUT,
        )
        r.raise_for_status()
        rows = [
            [pd.Timestamp(c["time"]).value // 10**6, c["mid"]["o"], c["mid"]["h"],
             c["mid"]["l"], c["mid"]["c"], c["volume"]]
            for c in r.json()["candles"] if c["complete"]
        ]
        return self._frame(rows)


class SyntheticData(MarketData):
    """Deterministic random-walk data with regimes. For tests and offline demos."""

    _START = {"XAU": 2400.0, "BTC": 60000.0, "ETH": 3000.0, "USD": 150.0}

    def __init__(self, seed: int = 7) -> None:
        self.seed = seed

    def candles(self, symbol, timeframe="1h", limit=300):
        rng = np.random.default_rng(zlib.crc32(f"{symbol}:{self.seed}".encode()))
        start = next((v for k, v in self._START.items() if symbol.upper().startswith(k)), 1.1)
        drift = np.repeat(rng.normal(0, 0.0008, limit // 50 + 1), 50)[:limit]
        rets = drift + rng.normal(0, 0.004, limit)
        close = start * np.exp(np.cumsum(rets))
        open_ = np.concatenate([[start], close[:-1]])
        spread = np.abs(rng.normal(0, 0.002, limit)) * close
        idx = pd.date_range(end=pd.Timestamp.now(tz="UTC").floor("h"), periods=limit, freq="h")
        return pd.DataFrame(
            {"open": open_, "high": np.maximum(open_, close) + spread,
             "low": np.minimum(open_, close) - spread, "close": close,
             "volume": rng.uniform(100, 1000, limit)},
            index=idx,
        )


def resample(df: pd.DataFrame, rule: str) -> pd.DataFrame:
    return df.resample(rule).agg(
        {"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"}
    ).dropna()


_PROVIDERS = {"yahoo": YahooData, "okx": OKXData, "oanda": OandaData, "synthetic": SyntheticData}
_cache: dict[str, MarketData] = {}


def get_provider(name: str) -> MarketData:
    if name not in _cache:
        _cache[name] = _PROVIDERS[name]()
    return _cache[name]
