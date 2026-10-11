"""Load config.yaml into typed objects."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import yaml

from .models import AssetClass, Instrument
from .risk import RiskConfig
from .strategies import EnsembleConfig


@dataclass
class Config:
    mode: str = "paper"                       # paper | live
    timeframe: str = "1h"
    loop_seconds: int = 300
    history_bars: int = 400
    starting_cash: float = 10_000.0
    state_dir: str = "state"
    instruments: list[Instrument] = field(default_factory=list)
    risk: RiskConfig = field(default_factory=RiskConfig)
    ensemble: EnsembleConfig = field(default_factory=EnsembleConfig)
    news_enabled: bool = True
    news_feeds: list[str] | None = None
    news_half_life_hours: float = 6.0
    positioning_enabled: bool = True
    brokers: dict[str, str] = field(default_factory=lambda: {
        "metal": "oanda", "forex": "oanda", "crypto": "ccxt"})

    @property
    def by_name(self) -> dict[str, Instrument]:
        return {i.name: i for i in self.instruments}


def _instrument(d: dict) -> Instrument:
    d = dict(d)
    d["asset_class"] = AssetClass(d["asset_class"])
    d["news_keywords"] = tuple(d.get("news_keywords", ()))
    return Instrument(**d)


def load_config(path: str | Path = "config.yaml") -> Config:
    raw = yaml.safe_load(Path(path).read_text()) or {}
    cfg = Config()
    for key in ("mode", "timeframe", "loop_seconds", "history_bars", "starting_cash", "state_dir"):
        if key in raw:
            setattr(cfg, key, raw[key])
    if cfg.mode not in ("paper", "live"):
        raise ValueError("mode must be 'paper' or 'live'")
    cfg.instruments = [_instrument(i) for i in raw.get("instruments", [])]
    if "risk" in raw:
        cfg.risk = RiskConfig(**{**RiskConfig().__dict__, **raw["risk"]})
    if "ensemble" in raw:
        cfg.ensemble = EnsembleConfig(**{**EnsembleConfig().__dict__, **raw["ensemble"]})
    news = raw.get("news", {})
    cfg.news_enabled = news.get("enabled", True)
    cfg.news_feeds = news.get("feeds")
    cfg.news_half_life_hours = news.get("half_life_hours", 6.0)
    cfg.positioning_enabled = raw.get("positioning", {}).get("enabled", True)
    cfg.brokers = {**cfg.brokers, **raw.get("brokers", {})}
    return cfg
