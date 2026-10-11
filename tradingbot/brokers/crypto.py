"""Crypto exchange broker via CCXT (Binance, Kraken, Coinbase, OKX, Bybit, ...).

Env: CCXT_EXCHANGE (e.g. "kraken"), CCXT_API_KEY, CCXT_SECRET, optional CCXT_PASSWORD,
CCXT_SANDBOX=1 to use the exchange testnet where supported.
On spot markets the bot only goes long (no shorting). Stops are managed by the bot.
Positions are tracked in a local JSON file so restarts don't forget them.
"""
from __future__ import annotations

import json
import logging
import os
from datetime import datetime, timezone

from ..models import Instrument, Position, Side, Trade
from .base import Broker

log = logging.getLogger(__name__)


class CCXTBroker(Broker):
    def __init__(self, state_file: str = "state/ccxt_positions.json"):
        try:
            import ccxt  # optional dependency
        except ImportError as exc:
            raise RuntimeError("pip install ccxt to trade crypto live") from exc
        name = os.environ["CCXT_EXCHANGE"]
        self.ex = getattr(ccxt, name)({
            "apiKey": os.environ["CCXT_API_KEY"], "secret": os.environ["CCXT_SECRET"],
            "password": os.getenv("CCXT_PASSWORD"), "enableRateLimit": True,
        })
        sandbox = os.getenv("CCXT_SANDBOX", "1") == "1"
        if sandbox:
            self.ex.set_sandbox_mode(True)
        self.live = not sandbox
        self.state_file = state_file
        self._positions: dict[str, Position] = {}
        if os.path.exists(state_file):
            with open(state_file) as f:
                for p in json.load(f):
                    p["side"] = Side(p["side"])
                    p["opened_at"] = datetime.fromisoformat(p["opened_at"])
                    self._positions[p["instrument"]] = Position(**p)

    def _save(self):
        os.makedirs(os.path.dirname(self.state_file) or ".", exist_ok=True)
        with open(self.state_file, "w") as f:
            json.dump([{**p.__dict__, "side": p.side.value, "opened_at": p.opened_at.isoformat()}
                       for p in self._positions.values()], f, indent=2)

    def can_short(self, inst: Instrument) -> bool:
        return False

    def equity(self, prices):
        bal = self.ex.fetch_balance()["total"]
        eq = float(bal.get("USDT", 0)) + float(bal.get("USD", 0)) + float(bal.get("USDC", 0))
        for name, p in self._positions.items():
            eq += p.qty * prices.get(name, p.entry_price)
        return eq

    def positions(self):
        return self._positions

    def open(self, inst, side, qty, price, stop_loss, take_profit):
        if side is Side.SELL:
            log.info("%s: spot account, skipping short", inst.name)
            return None
        qty = float(self.ex.amount_to_precision(inst.broker_symbol, qty))
        order = self.ex.create_market_order(inst.broker_symbol, "buy", qty)
        fill = float(order.get("average") or order.get("price") or price)
        filled = float(order.get("filled") or qty)
        pos = Position(inst.name, side, filled, fill, stop_loss, take_profit, broker_id=str(order["id"]))
        self._positions[inst.name] = pos
        self._save()
        return pos

    def close(self, inst, price, reason):
        pos = self._positions.get(inst.name)
        if pos is None:
            return None
        qty = float(self.ex.amount_to_precision(inst.broker_symbol, pos.qty))
        order = self.ex.create_market_order(inst.broker_symbol, "sell", qty)
        self._positions.pop(inst.name)
        self._save()
        fill = float(order.get("average") or order.get("price") or price)
        return Trade(inst.name, pos.side, pos.qty, pos.entry_price, fill,
                     (fill - pos.entry_price) * pos.qty, pos.opened_at, datetime.now(timezone.utc), reason)

    def update_stop(self, inst, pos):
        self._save()

    def save_state(self):
        self._save()
