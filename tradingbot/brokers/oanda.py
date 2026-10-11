"""OANDA v20 REST broker for XAU/USD and forex.

Env: OANDA_API_KEY, OANDA_ACCOUNT_ID, OANDA_ENV=practice|live (default practice).
Stops/targets are attached server-side, so they protect you even if the bot dies.
"""
from __future__ import annotations

import logging
import os
from datetime import datetime, timezone

import pandas as pd

from ..http import session, TIMEOUT
from ..models import Instrument, Position, Side, Trade
from .base import Broker

log = logging.getLogger(__name__)


def _price_str(inst: Instrument, px: float) -> str:
    decimals = 3 if (inst.quote == "JPY" or inst.base == "XAU") else 5
    return f"{px:.{decimals}f}"


class OandaBroker(Broker):
    def __init__(self):
        env = os.getenv("OANDA_ENV", "practice")
        self.live = env == "live"
        self.base = f"https://api-fx{'trade' if self.live else 'practice'}.oanda.com/v3"
        self.account = os.environ["OANDA_ACCOUNT_ID"]
        self.headers = {"Authorization": f"Bearer {os.environ['OANDA_API_KEY']}",
                        "Content-Type": "application/json"}
        self._positions: dict[str, Position] = {}
        self._by_symbol: dict[str, str] = {}   # broker symbol -> instrument name

    def _req(self, method: str, path: str, **kw):
        r = session().request(method, f"{self.base}/accounts/{self.account}{path}",
                              headers=self.headers, timeout=TIMEOUT, **kw)
        if r.status_code >= 400:
            raise RuntimeError(f"OANDA {method} {path} -> {r.status_code}: {r.text[:300]}")
        return r.json()

    def equity(self, prices):
        return float(self._req("GET", "/summary")["account"]["NAV"])

    def positions(self):
        return self._positions

    def open(self, inst, side, qty, price, stop_loss, take_profit):
        units = int(qty) * side.sign
        if units == 0:
            return None
        body = {"order": {
            "type": "MARKET", "instrument": inst.broker_symbol, "units": str(units),
            "timeInForce": "FOK", "positionFill": "DEFAULT",
            "stopLossOnFill": {"price": _price_str(inst, stop_loss)},
            "takeProfitOnFill": {"price": _price_str(inst, take_profit)},
        }}
        res = self._req("POST", "/orders", json=body)
        fill = res.get("orderFillTransaction")
        if not fill or "tradeOpened" not in fill:
            log.error("OANDA order not filled: %s", res.get("orderCancelTransaction", res))
            return None
        pos = Position(inst.name, side, abs(units), float(fill["price"]), stop_loss, take_profit,
                       broker_id=fill["tradeOpened"]["tradeID"])
        self._positions[inst.name] = pos
        self._by_symbol[inst.broker_symbol] = inst.name
        return pos

    def close(self, inst, price, reason):
        pos = self._positions.pop(inst.name, None)
        if pos is None:
            return None
        res = self._req("PUT", f"/trades/{pos.broker_id}/close")
        fill = res["orderFillTransaction"]
        px = float(fill["price"])
        return Trade(inst.name, pos.side, pos.qty, pos.entry_price, px, float(fill.get("pl", 0.0)),
                     pos.opened_at, datetime.now(timezone.utc), reason)

    def update_stop(self, inst, pos):
        self._req("PUT", f"/trades/{pos.broker_id}/orders",
                  json={"stopLoss": {"price": _price_str(inst, pos.stop_loss)}})

    def sync(self, instruments):
        """Detect trades OANDA closed itself (server-side SL/TP) and adopt existing trades."""
        open_trades = {t["id"]: t for t in self._req("GET", "/openTrades")["trades"]}
        closed = []
        for name, pos in list(self._positions.items()):
            if pos.broker_id not in open_trades:
                t = self._req("GET", f"/trades/{pos.broker_id}")["trade"]
                self._positions.pop(name)
                closed.append(Trade(name, pos.side, pos.qty, pos.entry_price,
                                    float(t.get("averageClosePrice", pos.entry_price)),
                                    float(t.get("realizedPL", 0.0)), pos.opened_at,
                                    datetime.now(timezone.utc), "broker_stop"))
        known = {p.broker_id for p in self._positions.values()}
        by_symbol = {i.broker_symbol: i for i in instruments.values()}
        for tid, t in open_trades.items():
            inst = by_symbol.get(t["instrument"])
            if tid in known or inst is None or inst.name in self._positions:
                continue
            units = float(t["currentUnits"])
            side = Side.BUY if units > 0 else Side.SELL
            sl = float(t.get("stopLossOrder", {}).get("price", 0)) or 0.0
            tp = float(t.get("takeProfitOrder", {}).get("price", 0)) or (float("inf") if side is Side.BUY else 0.0)
            opened = pd.Timestamp(t["openTime"]).floor("s").to_pydatetime()
            self._positions[inst.name] = Position(inst.name, side, abs(units), float(t["price"]), sl, tp,
                                                  opened_at=opened, broker_id=tid)
            log.info("adopted existing OANDA trade %s on %s", tid, inst.name)
        return closed

