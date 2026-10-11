"""Technical strategies plus the ensemble that blends them with news and positioning.

Every strategy scores *every* bar at once (score in [-1, 1] plus a confidence), using
only data up to that bar. Live trading reads the last row; the backtester and the
research tools read the whole history, so all three run exactly the same logic.
The ensemble detects the market regime with ADX and re-weights: trend-following
gets more say in trending markets, mean reversion more in ranging ones.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from . import indicators as ind
from .models import Decision, Signal


class Strategy(ABC):
    name: str = "strategy"
    style: str = "trend"     # "trend" or "reversion"
    min_bars: int = 60

    @abstractmethod
    def frame(self, df: pd.DataFrame) -> pd.DataFrame:
        """Columns ``score`` and ``confidence`` for every bar (NaN while warming up)."""

    def describe(self, df: pd.DataFrame) -> str:
        return ""

    def evaluate(self, df: pd.DataFrame) -> Signal:
        last = self.frame(df).iloc[-1]
        score = 0.0 if np.isnan(last["score"]) else float(last["score"])
        return Signal(self.name, score, float(last["confidence"]), self.describe(df))


def _frame(score, confidence, index) -> pd.DataFrame:
    return pd.DataFrame({"score": np.clip(np.asarray(score, dtype=float), -1, 1),
                         "confidence": np.asarray(confidence, dtype=float)}, index=index)


class TrendFollowing(Strategy):
    """EMA fast/slow alignment with a long-term filter, scaled by trend strength."""

    name, style = "trend", "trend"

    def __init__(self, fast=20, slow=50, filt=200):
        self.fast, self.slow, self.filt = fast, slow, filt
        self.min_bars = filt + 5

    def _gap(self, df):
        c = df["close"]
        a = ind.atr(df).replace(0, np.nan)
        return (ind.ema(c, self.fast) - ind.ema(c, self.slow)) / a

    def frame(self, df):
        c = df["close"]
        score = np.tanh(self._gap(df).fillna(0) / 2)
        against = (score > 0) != (c > ind.ema(c, self.filt))   # against the long-term trend -> halve it
        return _frame(score.where(~against, score * 0.5), 1.0, df.index)

    def describe(self, df):
        return f"EMA{self.fast}-EMA{self.slow} = {self._gap(df).iloc[-1]:+.2f} ATR"


class Momentum(Strategy):
    """MACD histogram direction plus rate of change."""

    name, style, min_bars = "momentum", "trend", 60

    def frame(self, df):
        c = df["close"]
        _, _, hist = ind.macd(c)
        a = ind.atr(df).replace(0, np.nan)
        h_norm = np.tanh((hist / a).fillna(0) * 4)
        accel = np.where(hist > hist.shift(), 0.3, -0.3)
        roc = np.tanh((c / c.shift(19) - 1) * 20)
        return _frame(0.5 * h_norm + 0.2 * accel + 0.3 * roc, 1.0, df.index)

    def describe(self, df):
        c = df["close"]
        return f"MACD hist {ind.macd(c)[2].iloc[-1]:+.4g}, 20-bar ROC {c.iloc[-1] / c.iloc[-20] - 1:+.2%}"


class Breakout(Strategy):
    """Donchian channel breakout confirmed by expanding volatility."""

    name, style = "breakout", "trend"

    def __init__(self, n=20):
        self.n = n
        self.min_bars = n + 30

    def frame(self, df):
        lo, hi = ind.donchian(df, self.n)
        prev_lo, prev_hi = lo.shift(), hi.shift()
        c = df["close"]
        atr = ind.atr(df)
        vol_up = atr > atr.rolling(20).mean()
        up, down = c > prev_hi, c < prev_lo
        span = (prev_hi - prev_lo).where(prev_hi > prev_lo)
        inside = (((c - prev_lo) / span).fillna(0.5) - 0.5) * 0.4
        score = np.select([up, down], [np.where(vol_up, 0.9, 0.5), np.where(vol_up, -0.9, -0.5)], inside)
        score = np.where(prev_hi.isna(), np.nan, score)
        return _frame(score, np.where(up | down, 1.0, 0.5), df.index)

    def describe(self, df):
        lo, hi = ind.donchian(df, self.n)
        c = df["close"].iloc[-1]
        if c > hi.iloc[-2]:
            return f"broke {self.n}-bar high {hi.iloc[-2]:.5g}"
        if c < lo.iloc[-2]:
            return f"broke {self.n}-bar low {lo.iloc[-2]:.5g}"
        span = hi.iloc[-2] - lo.iloc[-2]
        return f"inside channel at {(c - lo.iloc[-2]) / span if span else 0.5:.0%}"


class MeanReversion(Strategy):
    """RSI extremes + Bollinger band stretch: fade overextended moves."""

    name, style, min_bars = "mean_reversion", "reversion", 40

    def _parts(self, df):
        c = df["close"]
        r = ind.rsi(c)
        lower, mid, upper = ind.bollinger(c)
        width = ((upper - lower) / 2).replace(0, 1.0)
        return r, (c - mid) / width

    def frame(self, df):
        r, z = self._parts(df)
        rsi_part = np.where(r < 30, (30 - r) / 30, np.where(r > 70, -(r - 70) / 30, 0.0))
        score = 0.6 * rsi_part - 0.4 * np.tanh(z)
        return _frame(score, 1.0, df.index)

    def describe(self, df):
        r, z = self._parts(df)
        return f"RSI {r.iloc[-1]:.0f}, band z {z.iloc[-1]:+.2f}"


STRATEGIES = {"trend": TrendFollowing, "momentum": Momentum, "breakout": Breakout,
              "mean_reversion": MeanReversion}


def default_strategies() -> list[Strategy]:
    return [TrendFollowing(), Momentum(), Breakout(), MeanReversion()]


def build_strategies(names: list[str]) -> list[Strategy]:
    unknown = set(names) - set(STRATEGIES)
    if unknown:
        raise ValueError(f"unknown strategies {sorted(unknown)}; choose from {sorted(STRATEGIES)}")
    return [STRATEGIES[n]() for n in names]


@dataclass
class EnsembleConfig:
    # Defaults follow `tradingbot research`: on daily bars trend-following is the only
    # technical style that stayed positive out of sample; mean reversion is available
    # but its small intraday edge did not survive trading costs.
    strategies: list[str] = field(default_factory=lambda: ["trend", "momentum", "breakout"])
    weights: dict[str, float] = field(default_factory=lambda: {
        "technical": 0.5, "positioning": 0.3, "news": 0.2})
    entry_threshold: float = 0.35
    exit_threshold: float = 0.10     # close when the blended score fades/flips past this
    adx_trend: float = 25.0
    adx_range: float = 20.0


REGIME_STYLE_WEIGHTS = {"trending": {"trend": 1.0, "reversion": 0.2},
                        "ranging": {"trend": 0.3, "reversion": 1.0},
                        "neutral": {"trend": 0.7, "reversion": 0.5}}


class Ensemble:
    def __init__(self, strategies: list[Strategy] | None = None, config: EnsembleConfig | None = None):
        self.cfg = config or EnsembleConfig()
        self.strategies = strategies or build_strategies(self.cfg.strategies)

    @property
    def min_bars(self) -> int:
        return max(s.min_bars for s in self.strategies)

    def regimes(self, df: pd.DataFrame) -> tuple[pd.Series, pd.Series]:
        a = ind.adx(df)
        regime = np.where(a >= self.cfg.adx_trend, "trending",
                          np.where(a <= self.cfg.adx_range, "ranging", "neutral"))
        return pd.Series(regime, index=df.index), a

    def regime(self, df: pd.DataFrame) -> tuple[str, float]:
        regime, a = self.regimes(df)
        return regime.iloc[-1], float(a.iloc[-1])

    def technical_frame(self, df: pd.DataFrame) -> pd.DataFrame:
        """Per-bar score of every strategy plus the regime-weighted ``technical`` blend."""
        regime, _ = self.regimes(df)
        out = pd.DataFrame(index=df.index)
        num = pd.Series(0.0, index=df.index)
        den = pd.Series(0.0, index=df.index)
        for strat in self.strategies:
            f = strat.frame(df)
            w = regime.map(lambda r, st=strat.style: REGIME_STYLE_WEIGHTS[r][st]) * f["confidence"]
            num += (w * f["score"]).fillna(0)
            den += w.where(f["score"].notna(), 0).fillna(0)
            out[strat.name] = f["score"]
        out["technical"] = (num / den.replace(0, np.nan)).fillna(0.0)
        out.iloc[: self.min_bars - 1] = np.nan          # not enough history yet
        return out

    def technical(self, df: pd.DataFrame) -> tuple[Signal, list[Signal]]:
        regime, adx_val = self.regime(df)
        score = float(self.technical_frame(df)["technical"].iloc[-1])
        parts = [s.evaluate(df) for s in self.strategies]
        return Signal("technical", 0.0 if np.isnan(score) else score, 1.0,
                      f"{regime} market (ADX {adx_val:.0f})"), parts

    def blend(self, signals: list[Signal]) -> float:
        num = den = 0.0
        for s in signals:
            w = self.cfg.weights.get(s.source, 0.0) * (s.confidence if s.source != "technical" else 1.0)
            num += w * s.score
            den += w
        return num / den if den else 0.0

    def action(self, score: float, position_side: int = 0) -> str:
        """position_side: +1 long, -1 short, 0 flat."""
        if position_side == 0:
            if score >= self.cfg.entry_threshold:
                return "buy"
            if score <= -self.cfg.entry_threshold:
                return "sell"
            return "hold"
        return "close" if position_side * score < -self.cfg.exit_threshold else "hold"

    def decide(self, name: str, df: pd.DataFrame, extra: list[Signal] = (),
               position_side: int = 0) -> Decision:
        tech, parts = self.technical(df)
        signals = [tech, *extra]
        score = self.blend(signals)
        return Decision(name, score, self.action(score, position_side), signals + parts)
