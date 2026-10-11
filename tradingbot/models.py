"""Core data types shared across the bot."""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum


class Side(str, Enum):
    BUY = "buy"
    SELL = "sell"

    @property
    def sign(self) -> int:
        return 1 if self is Side.BUY else -1

    def opposite(self) -> "Side":
        return Side.SELL if self is Side.BUY else Side.BUY


class AssetClass(str, Enum):
    METAL = "metal"
    FOREX = "forex"
    CRYPTO = "crypto"


@dataclass(frozen=True)
class Instrument:
    """A tradable market, e.g. XAU/USD.

    ``data_symbol`` is the ticker used by the market-data provider and
    ``broker_symbol`` the one used by the execution broker; they rarely match.
    """

    name: str
    asset_class: AssetClass
    data_source: str
    data_symbol: str
    broker_symbol: str
    cot_code: str | None = None          # CFTC contract code, for COT positioning
    okx_swap: str | None = None          # OKX perpetual id, for crypto positioning
    news_keywords: tuple[str, ...] = ()
    min_qty: float = 0.0
    qty_step: float = 0.0

    @property
    def base(self) -> str:
        return self.name.split("/")[0]

    @property
    def quote(self) -> str:
        return self.name.split("/")[1]


@dataclass
class Signal:
    """Output of one analysis component.

    ``score`` is directional conviction in [-1, 1] (positive = bullish) and
    ``confidence`` in [0, 1] says how much that component trusts itself now.
    """

    source: str
    score: float
    confidence: float = 1.0
    reason: str = ""

    def __post_init__(self) -> None:
        self.score = max(-1.0, min(1.0, float(self.score)))
        self.confidence = max(0.0, min(1.0, float(self.confidence)))


@dataclass
class Decision:
    instrument: str
    score: float
    action: str                 # "buy", "sell", "close", "hold"
    signals: list[Signal] = field(default_factory=list)


@dataclass
class Position:
    instrument: str
    side: Side
    qty: float
    entry_price: float
    stop_loss: float
    take_profit: float
    opened_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    broker_id: str | None = None
    highest: float = 0.0        # best price seen, for trailing stops
    lowest: float = 0.0

    def __post_init__(self) -> None:
        self.highest = self.highest or self.entry_price
        self.lowest = self.lowest or self.entry_price

    def unrealized(self, price: float) -> float:
        """PnL in quote currency."""
        return (price - self.entry_price) * self.qty * self.side.sign


@dataclass
class Trade:
    instrument: str
    side: Side
    qty: float
    entry_price: float
    exit_price: float
    pnl_usd: float
    opened_at: datetime
    closed_at: datetime
    reason: str
