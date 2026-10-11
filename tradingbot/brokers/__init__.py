"""Brokers and a router that sends each asset class to its own venue."""
from __future__ import annotations

from ..models import Instrument
from .base import Broker
from .paper import PaperBroker


class RoutingBroker(Broker):
    """Send orders for each asset class to a different broker (e.g. OANDA + Kraken)."""

    def __init__(self, routes: dict[str, Broker]):
        self.routes = routes
        self.live = any(b.live for b in routes.values())
        self._unique = list({id(b): b for b in routes.values()}.values())

    def _for(self, inst: Instrument) -> Broker:
        return self.routes[inst.asset_class.value]

    def equity(self, prices):
        return sum(b.equity(prices) for b in self._unique)

    def positions(self):
        out = {}
        for b in self._unique:
            out.update(b.positions())
        return out

    def can_short(self, inst):
        return self._for(inst).can_short(inst)

    def open(self, inst, *args, **kw):
        return self._for(inst).open(inst, *args, **kw)

    def close(self, inst, *args, **kw):
        return self._for(inst).close(inst, *args, **kw)

    def update_stop(self, inst, pos):
        self._for(inst).update_stop(inst, pos)

    def save_state(self):
        for b in self._unique:
            b.save_state()

    def sync(self, instruments):
        return [t for b in self._unique for t in b.sync(instruments)]


def build_broker(name: str, paper_kwargs: dict | None = None) -> Broker:
    if name == "paper":
        return PaperBroker(**(paper_kwargs or {}))
    if name == "oanda":
        from .oanda import OandaBroker
        return OandaBroker()
    if name == "ccxt":
        from .crypto import CCXTBroker
        return CCXTBroker()
    raise ValueError(f"unknown broker {name!r}")


__all__ = ["Broker", "PaperBroker", "RoutingBroker", "build_broker"]
