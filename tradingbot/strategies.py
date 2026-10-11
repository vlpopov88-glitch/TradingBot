"""Technical strategies plus the ensemble that blends them with news and positioning.

Every strategy maps a candle DataFrame to a ``Signal`` (score in [-1, 1]). The
ensemble detects the market regime with ADX and re-weights: trend-following gets
more say in trending markets, mean reversion more in ranging ones.
"""
from __future__ import annotations

import math
from abc import ABC, abstractmethod
from dataclasses import dataclass, field

import pandas as pd

from . import indicators as ind
from .models import Decision, Signal


class Strategy(ABC):
    name: str = "strategy"
    style: str = "trend"     # "trend" or "reversion"
    min_bars: int = 60

    @abstractmethod
    def evaluate(self, df: pd.DataFrame) -> Signal:
        ...


class TrendFollowing(Strategy):
    """EMA fast/slow alignment with a long-term filter, scaled by trend strength."""

    name, style = "trend", "trend"

    def __init__(self, fast=20, slow=50, filt=200):
        self.fast, self.slow, self.filt = fast, slow, filt
        self.min_bars = filt + 5

    def evaluate(self, df):
        c = df["close"]
        f, s, l = ind.ema(c, self.fast).iloc[-1], ind.ema(c, self.slow).iloc[-1], ind.ema(c, self.filt).iloc[-1]
        a = ind.atr(df).iloc[-1]
        gap = (f - s) / a if a else 0.0           # spread in ATRs
        score = math.tanh(gap / 2)
        if (score > 0) != (c.iloc[-1] > l):       # against the long-term trend -> halve it
            score *= 0.5
        return Signal(self.name, score, 1.0, f"EMA{self.fast}-EMA{self.slow} = {gap:+.2f} ATR")


class Momentum(Strategy):
    """MACD histogram direction plus rate of change."""

    name, style, min_bars = "momentum", "trend", 60

    def evaluate(self, df):
        c = df["close"]
        _, _, hist = ind.macd(c)
        a = ind.atr(df).iloc[-1] or 1.0
        h_norm = math.tanh(hist.iloc[-1] / a * 4)
        accel = 0.3 if hist.iloc[-1] > hist.iloc[-2] else -0.3
        roc = math.tanh((c.iloc[-1] / c.iloc[-20] - 1) * 20)
        return Signal(self.name, 0.5 * h_norm + 0.2 * accel + 0.3 * roc, 1.0,
                      f"MACD hist {hist.iloc[-1]:+.4g}, 20-bar ROC {c.iloc[-1] / c.iloc[-20] - 1:+.2%}")


class Breakout(Strategy):
    """Donchian channel breakout confirmed by expanding volatility."""

    name, style = "breakout", "trend"

    def __init__(self, n=20):
        self.n = n
        self.min_bars = n + 30

    def evaluate(self, df):
        lo, hi = ind.donchian(df, self.n)
        prev_lo, prev_hi = lo.iloc[-2], hi.iloc[-2]
        c = df["close"].iloc[-1]
        atr = ind.atr(df)
        vol_up = atr.iloc[-1] > atr.iloc[-20:].mean()
        if c > prev_hi:
            return Signal(self.name, 0.9 if vol_up else 0.5, 1.0, f"broke {self.n}-bar high {prev_hi:.5g}")
        if c < prev_lo:
            return Signal(self.name, -0.9 if vol_up else -0.5, 1.0, f"broke {self.n}-bar low {prev_lo:.5g}")
        pos = (c - prev_lo) / (prev_hi - prev_lo) if prev_hi > prev_lo else 0.5
        return Signal(self.name, (pos - 0.5) * 0.4, 0.5, f"inside channel at {pos:.0%}")


class MeanReversion(Strategy):
    """RSI extremes + Bollinger band stretch: fade overextended moves."""

    name, style, min_bars = "mean_reversion", "reversion", 40

    def evaluate(self, df):
        c = df["close"]
        r = ind.rsi(c).iloc[-1]
        lower, mid, upper = (b.iloc[-1] for b in ind.bollinger(c))
        width = (upper - lower) / 2 or 1.0
        z = (c.iloc[-1] - mid) / width          # ~ -1 at lower band, +1 at upper
        rsi_part = 0.0
        if r < 30:
            rsi_part = (30 - r) / 30
        elif r > 70:
            rsi_part = -(r - 70) / 30
        score = 0.6 * rsi_part - 0.4 * math.tanh(z)
        return Signal(self.name, score, 1.0, f"RSI {r:.0f}, band z {z:+.2f}")


def default_strategies() -> list[Strategy]:
    return [TrendFollowing(), Momentum(), Breakout(), MeanReversion()]


@dataclass
class EnsembleConfig:
    weights: dict[str, float] = field(default_factory=lambda: {
        "technical": 0.6, "news": 0.25, "positioning": 0.15})
    entry_threshold: float = 0.35
    exit_threshold: float = 0.10     # close when the blended score fades/flips past this
    adx_trend: float = 25.0
    adx_range: float = 20.0


class Ensemble:
    def __init__(self, strategies: list[Strategy] | None = None, config: EnsembleConfig | None = None):
        self.strategies = strategies or default_strategies()
        self.cfg = config or EnsembleConfig()

    @property
    def min_bars(self) -> int:
        return max(s.min_bars for s in self.strategies)

    def regime(self, df: pd.DataFrame) -> tuple[str, float]:
        a = ind.adx(df).iloc[-1]
        if a >= self.cfg.adx_trend:
            return "trending", a
        if a <= self.cfg.adx_range:
            return "ranging", a
        return "neutral", a

    def technical(self, df: pd.DataFrame) -> tuple[Signal, list[Signal]]:
        regime, adx_val = self.regime(df)
        style_w = {"trending": {"trend": 1.0, "reversion": 0.2},
                   "ranging": {"trend": 0.3, "reversion": 1.0},
                   "neutral": {"trend": 0.7, "reversion": 0.5}}[regime]
        sigs = [s.evaluate(df) for s in self.strategies]
        num = den = 0.0
        for strat, sig in zip(self.strategies, sigs):
            w = style_w[strat.style] * sig.confidence
            num += w * sig.score
            den += w
        score = num / den if den else 0.0
        return Signal("technical", score, 1.0, f"{regime} market (ADX {adx_val:.0f})"), sigs

    def decide(self, name: str, df: pd.DataFrame, extra: list[Signal] = (),
               position_side: int = 0) -> Decision:
        """position_side: +1 long, -1 short, 0 flat."""
        tech, parts = self.technical(df)
        signals = [tech, *extra]
        num = den = 0.0
        for s in signals:
            w = self.cfg.weights.get(s.source, 0.0) * (s.confidence if s.source != "technical" else 1.0)
            num += w * s.score
            den += w
        score = num / den if den else 0.0

        action = "hold"
        if position_side == 0:
            if score >= self.cfg.entry_threshold:
                action = "buy"
            elif score <= -self.cfg.entry_threshold:
                action = "sell"
        elif position_side * score < -self.cfg.exit_threshold:
            action = "close"
        return Decision(name, score, action, signals + parts)
