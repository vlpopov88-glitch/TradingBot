from __future__ import annotations

from abc import ABC, abstractmethod

from ..models import Instrument, Position, Side, Trade


class Broker(ABC):
    """Execution venue. Prices are in the instrument's quote currency; equity in USD."""

    live: bool = False

    @abstractmethod
    def equity(self, prices: dict[str, float]) -> float: ...

    @abstractmethod
    def positions(self) -> dict[str, Position]: ...

    @abstractmethod
    def open(self, inst: Instrument, side: Side, qty: float, price: float,
             stop_loss: float, take_profit: float) -> Position | None: ...

    @abstractmethod
    def close(self, inst: Instrument, price: float, reason: str) -> Trade | None: ...

    def can_short(self, inst: Instrument) -> bool:
        return True

    def update_stop(self, inst: Instrument, pos: Position) -> None:
        """Push a moved (trailing) stop to the venue, if it holds stops server-side."""

    def save_state(self) -> None:
        """Persist locally tracked position state (stops, check progress), if any."""

    def sync(self, instruments: dict[str, Instrument]) -> list[Trade]:
        """Reconcile with the venue; return positions it closed on its own (e.g. hard stops)."""
        return []
