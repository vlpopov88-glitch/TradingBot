"""Risk management: position sizing, stops, exposure limits and circuit breakers."""
from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from datetime import date, datetime, timezone

from .models import Instrument, Position, Side

log = logging.getLogger(__name__)


@dataclass
class RiskConfig:
    risk_per_trade: float = 0.01          # fraction of equity lost if the stop is hit
    atr_stop_mult: float = 2.0            # stop distance in ATRs
    reward_risk: float = 2.0              # take-profit distance = stop distance * this
    trailing_atr_mult: float = 2.5        # 0 disables trailing stops
    max_open_positions: int = 4
    max_positions_per_class: int = 2
    max_daily_loss: float = 0.03          # halt new trades for the day past this
    max_drawdown: float = 0.15            # halt everything (until restarted) past this
    max_leverage: dict[str, float] = field(default_factory=lambda: {
        "crypto": 2.0, "forex": 10.0, "metal": 5.0})


def usd_per_quote(inst: Instrument, price: float) -> float:
    """How many USD one unit of the quote currency is worth."""
    if inst.quote in ("USD", "USDT", "USDC"):
        return 1.0
    if inst.base == "USD":
        return 1.0 / price
    raise ValueError(f"{inst.name}: only USD-based instruments are supported for sizing")


def round_qty(inst: Instrument, qty: float) -> float:
    if inst.qty_step:
        qty = math.floor(qty / inst.qty_step) * inst.qty_step
        qty = round(qty, 10)
    return qty if qty >= inst.min_qty else 0.0


class RiskManager:
    def __init__(self, cfg: RiskConfig | None = None):
        self.cfg = cfg or RiskConfig()
        self.peak_equity = 0.0
        self.day: date | None = None
        self.day_start_equity = 0.0
        self.halted_reason: str | None = None

    # ---- circuit breakers -------------------------------------------------
    def on_equity(self, equity: float, now: datetime | None = None) -> None:
        now = now or datetime.now(timezone.utc)
        if self.day != now.date():
            self.day, self.day_start_equity = now.date(), equity
            if self.halted_reason and self.halted_reason.startswith("daily"):
                log.info("new day: lifting daily-loss halt")
                self.halted_reason = None
        self.peak_equity = max(self.peak_equity, equity)
        if self.peak_equity and equity < self.peak_equity * (1 - self.cfg.max_drawdown):
            if not self.halted_reason or not self.halted_reason.startswith("max drawdown"):
                log.error("MAX DRAWDOWN hit: equity %.2f vs peak %.2f", equity, self.peak_equity)
            self.halted_reason = f"max drawdown {self.cfg.max_drawdown:.0%} breached"
        elif (self.day_start_equity and equity < self.day_start_equity * (1 - self.cfg.max_daily_loss)
              and not self.halted_reason):
            log.warning("daily loss limit hit: %.2f -> %.2f", self.day_start_equity, equity)
            self.halted_reason = f"daily loss limit {self.cfg.max_daily_loss:.0%} hit"

    # ---- entries ----------------------------------------------------------
    def can_open(self, inst: Instrument, positions: dict[str, Position],
                 instruments: dict[str, Instrument]) -> tuple[bool, str]:
        if self.halted_reason:
            return False, self.halted_reason
        if inst.name in positions:
            return False, "already in position"
        if len(positions) >= self.cfg.max_open_positions:
            return False, "max open positions"
        same = sum(1 for n in positions if instruments[n].asset_class is inst.asset_class)
        if same >= self.cfg.max_positions_per_class:
            return False, f"max {inst.asset_class.value} positions"
        return True, ""

    def plan(self, inst: Instrument, side: Side, price: float, atr: float,
             equity: float) -> tuple[float, float, float]:
        """Return (qty, stop_loss, take_profit). qty 0 means don't trade."""
        if atr <= 0 or price <= 0 or equity <= 0:
            return 0.0, 0.0, 0.0
        dist = atr * self.cfg.atr_stop_mult
        sl = price - side.sign * dist
        tp = price + side.sign * dist * self.cfg.reward_risk
        risk_usd = equity * self.cfg.risk_per_trade
        qty = risk_usd / (dist * usd_per_quote(inst, price))
        # Cap notional by leverage limit.
        notional_usd = qty * price * usd_per_quote(inst, price)
        max_notional = equity * self.cfg.max_leverage.get(inst.asset_class.value, 1.0)
        if notional_usd > max_notional:
            qty *= max_notional / notional_usd
        return round_qty(inst, qty), sl, tp

    # ---- exits ------------------------------------------------------------
    def update_trailing(self, pos: Position, high: float, low: float, atr: float) -> None:
        pos.highest, pos.lowest = max(pos.highest, high), min(pos.lowest, low)
        if not self.cfg.trailing_atr_mult or atr <= 0:
            return
        trail = atr * self.cfg.trailing_atr_mult
        if pos.side is Side.BUY:
            pos.stop_loss = max(pos.stop_loss, pos.highest - trail)
        else:
            pos.stop_loss = min(pos.stop_loss, pos.lowest + trail)

    @staticmethod
    def stop_hit(pos: Position, high: float, low: float, open_: float | None = None) -> tuple[float, str] | None:
        """Check SL/TP against a bar. Assumes the stop fills first if both are touched."""
        if pos.side is Side.BUY:
            if low <= pos.stop_loss:
                px = min(pos.stop_loss, open_) if open_ is not None else pos.stop_loss
                return px, "stop_loss"
            if high >= pos.take_profit:
                px = max(pos.take_profit, open_) if open_ is not None else pos.take_profit
                return px, "take_profit"
        else:
            if high >= pos.stop_loss:
                px = max(pos.stop_loss, open_) if open_ is not None else pos.stop_loss
                return px, "stop_loss"
            if low <= pos.take_profit:
                px = min(pos.take_profit, open_) if open_ is not None else pos.take_profit
                return px, "take_profit"
        return None

