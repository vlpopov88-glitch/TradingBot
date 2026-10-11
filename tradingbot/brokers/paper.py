"""Simulated broker with spread/slippage/commission, persisted to a JSON file."""
from __future__ import annotations

import json
import logging
import os
from dataclasses import asdict
from datetime import datetime, timezone

from ..models import Instrument, Position, Side, Trade
from ..risk import usd_per_quote
from .base import Broker

log = logging.getLogger(__name__)

# Half-spread + slippage per fill, in basis points of price.
DEFAULT_COST_BPS = {"forex": 1.0, "metal": 2.0, "crypto": 5.0}


class PaperBroker(Broker):
    def __init__(self, starting_cash: float = 10_000.0, state_file: str | None = None,
                 cost_bps: dict[str, float] | None = None, commission_bps: float = 0.0):
        self.cash = starting_cash
        self.state_file = state_file
        self.cost_bps = cost_bps or DEFAULT_COST_BPS
        self.commission_bps = commission_bps
        self._positions: dict[str, Position] = {}
        self._fx: dict[str, float] = {}   # last known usd_per_quote per instrument
        if state_file and os.path.exists(state_file):
            self._load()

    # ---- state ------------------------------------------------------------
    def _load(self) -> None:
        with open(self.state_file) as f:
            st = json.load(f)
        self.cash = st["cash"]
        for p in st["positions"]:
            p["side"] = Side(p["side"])
            p["opened_at"] = datetime.fromisoformat(p["opened_at"])
            self._positions[p["instrument"]] = Position(**p)
        self._fx = st.get("fx", {})
        log.info("paper broker restored: cash %.2f, %d positions", self.cash, len(self._positions))

    def _save(self) -> None:
        if not self.state_file:
            return
        pos = []
        for p in self._positions.values():
            d = asdict(p)
            d["side"] = p.side.value
            d["opened_at"] = p.opened_at.isoformat()
            pos.append(d)
        tmp = self.state_file + ".tmp"
        with open(tmp, "w") as f:
            json.dump({"cash": self.cash, "positions": pos, "fx": self._fx}, f, indent=2)
        os.replace(tmp, self.state_file)

    # ---- Broker API -------------------------------------------------------
    def positions(self):
        return self._positions

    def equity(self, prices):
        eq = self.cash
        for name, p in self._positions.items():
            px = prices.get(name, p.entry_price)
            eq += p.unrealized(px) * self._fx.get(name, 1.0)
        return eq

    def _fill(self, inst: Instrument, side: Side, price: float) -> float:
        bps = self.cost_bps.get(inst.asset_class.value, 2.0)
        return price * (1 + side.sign * bps / 10_000)

    def _commission(self, inst: Instrument, qty: float, price: float) -> float:
        return qty * price * usd_per_quote(inst, price) * self.commission_bps / 10_000

    def open(self, inst, side, qty, price, stop_loss, take_profit, when=None):
        fill = self._fill(inst, side, price)
        self.cash -= self._commission(inst, qty, fill)
        pos = Position(inst.name, side, qty, fill, stop_loss, take_profit,
                       opened_at=when or datetime.now(timezone.utc))
        self._positions[inst.name] = pos
        self._fx[inst.name] = usd_per_quote(inst, fill)
        self._save()
        return pos

    def close(self, inst, price, reason, when=None):
        pos = self._positions.pop(inst.name, None)
        if pos is None:
            return None
        fill = self._fill(inst, pos.side.opposite(), price)
        pnl = pos.unrealized(fill) * usd_per_quote(inst, fill) - self._commission(inst, pos.qty, fill)
        self.cash += pnl
        self._fx.pop(inst.name, None)
        self._save()
        return Trade(inst.name, pos.side, pos.qty, pos.entry_price, fill, pnl,
                     pos.opened_at, when or datetime.now(timezone.utc), reason)

    def mark(self, inst: Instrument, price: float) -> None:
        if inst.name in self._positions:
            self._fx[inst.name] = usd_per_quote(inst, price)

    def update_stop(self, inst, pos):
        self._save()

    def save_state(self):
        self._save()
