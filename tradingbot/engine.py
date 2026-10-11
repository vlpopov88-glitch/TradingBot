"""The live loop: fetch data -> analyze -> manage risk -> execute -> journal -> sleep."""
from __future__ import annotations

import logging
import os
import signal
import time
from datetime import datetime, timezone

import pandas as pd

from . import indicators as ind
from .brokers import Broker, PaperBroker, RoutingBroker, build_broker
from .config import Config
from .data import get_provider
from .journal import Journal
from .models import Decision, Instrument, Side, Signal
from .news import NewsAnalyzer
from .positioning import PositioningAnalyzer
from .risk import RiskManager
from .strategies import Ensemble

log = logging.getLogger(__name__)

TF = {"1m": "1min", "5m": "5min", "15m": "15min", "30m": "30min", "1h": "1h", "4h": "4h", "1d": "1D"}


def closed_bars(df: pd.DataFrame, timeframe: str, now: datetime | None = None) -> pd.DataFrame:
    """Drop the still-forming last bar so signals don't repaint."""
    now = pd.Timestamp(now or datetime.now(timezone.utc))
    if len(df) and df.index[-1] + pd.Timedelta(TF[timeframe]) > now:
        return df.iloc[:-1]
    return df


def make_broker(cfg: Config) -> Broker:
    if cfg.mode == "paper":
        os.makedirs(cfg.state_dir, exist_ok=True)
        return PaperBroker(cfg.starting_cash, state_file=os.path.join(cfg.state_dir, "paper_account.json"))
    classes = {i.asset_class.value for i in cfg.instruments}
    built: dict[str, Broker] = {}
    routes = {}
    for cls in classes:
        name = cfg.brokers[cls]
        built.setdefault(name, build_broker(name))
        routes[cls] = built[name]
    return RoutingBroker(routes)


class TradingEngine:
    def __init__(self, cfg: Config, broker: Broker | None = None, journal: Journal | None = None,
                 news: NewsAnalyzer | None = None, positioning: PositioningAnalyzer | None = None):
        self.cfg = cfg
        self.instruments = cfg.by_name
        os.makedirs(cfg.state_dir, exist_ok=True)
        self.broker = broker or make_broker(cfg)
        self.journal = journal or Journal(os.path.join(cfg.state_dir, "journal.db"))
        self.ensemble = Ensemble(config=cfg.ensemble)
        self.risk = RiskManager(cfg.risk)
        self.news = news if news is not None else (
            NewsAnalyzer(cfg.news_feeds, cfg.news_half_life_hours) if cfg.news_enabled else None)
        self.positioning = positioning if positioning is not None else (
            PositioningAnalyzer() if cfg.positioning_enabled else None)
        self.prices: dict[str, float] = {}
        self._running = True
        self.kill_file = os.path.join(cfg.state_dir, "STOP")

    # ---- analysis ---------------------------------------------------------
    def context_signals(self, inst: Instrument) -> list[Signal]:
        out = []
        if self.news:
            out.append(self.news.signal(inst))
        if self.positioning:
            out.append(self.positioning.signal(inst))
        return out

    def analyze(self, inst: Instrument) -> tuple[Decision, pd.DataFrame] | None:
        df = get_provider(inst.data_source).candles(inst.data_symbol, self.cfg.timeframe, self.cfg.history_bars)
        if df.empty:
            return None
        self.prices[inst.name] = float(df["close"].iloc[-1])
        age = pd.Timestamp.now(tz="UTC") - df.index[-1]
        if age > 3 * pd.Timedelta(TF[self.cfg.timeframe]):
            # Weekend / holiday / dead feed: never trade on stale prices.
            log.info("%s: last bar is %s old - market closed or data stale, skipping", inst.name, age)
            return None
        bars = closed_bars(df, self.cfg.timeframe)
        if len(bars) < self.ensemble.min_bars:
            log.warning("%s: only %d bars, need %d", inst.name, len(bars), self.ensemble.min_bars)
            return None
        pos = self.broker.positions().get(inst.name)
        side = pos.side.sign if pos else 0
        return self.ensemble.decide(inst.name, bars, self.context_signals(inst), side), df

    # ---- one cycle --------------------------------------------------------
    def step(self) -> None:
        if self.news:
            self.news.update()
        for t in self.broker.sync(self.instruments):
            log.info("broker closed %s: pnl %.2f (%s)", t.instrument, t.pnl_usd, t.reason)
            self.journal.trade(t)

        for inst in self.instruments.values():
            try:
                self._step_instrument(inst)
            except Exception:
                log.exception("%s: cycle failed", inst.name)

        positions = self.broker.positions()
        eq = self.broker.equity(self.prices)
        self.risk.on_equity(eq)
        self.journal.equity(eq, len(positions))
        log.info("equity %.2f | open: %s%s", eq,
                 ", ".join(f"{n} {p.side.value} {p.qty:g}@{p.entry_price:.5g}" for n, p in positions.items()) or "none",
                 f" | HALTED: {self.risk.halted_reason}" if self.risk.halted_reason else "")

    def _step_instrument(self, inst: Instrument) -> None:
        res = self.analyze(inst)
        if res is None:
            return
        decision, df = res
        price = self.prices[inst.name]
        if isinstance(self.broker, PaperBroker):
            self.broker.mark(inst, price)
        atr = float(ind.atr(df).iloc[-1])
        pos = self.broker.positions().get(inst.name)

        # 1. Manage the open position: trailing stop, stop/target hit, signal exit.
        if pos:
            since = df[df.index > pd.Timestamp(pos.opened_at)]
            hi = max([price, *since["high"].tolist()])
            lo = min([price, *since["low"].tolist()])
            old_sl = pos.stop_loss
            self.risk.update_trailing(pos, hi, lo, atr)
            if pos.stop_loss != old_sl:
                self.broker.update_stop(inst, pos)
            hit = self.risk.stop_hit(pos, hi, lo)
            if hit:
                self._close(inst, hit[0], hit[1])
                return
            if decision.action == "close":
                self._close(inst, price, f"signal flip ({decision.score:+.2f})")
                return

        # 2. New entries.
        if decision.action in ("buy", "sell"):
            self.journal.decision(decision)
            side = Side(decision.action)
            if os.path.exists(self.kill_file):
                log.warning("kill switch %s present: no new entries", self.kill_file)
                return
            if side is Side.SELL and not self.broker.can_short(inst):
                return
            ok, why = self.risk.can_open(inst, self.broker.positions(), self.instruments)
            if not ok:
                log.info("%s: %s signal %+.2f blocked: %s", inst.name, side.value, decision.score, why)
                return
            equity = self.broker.equity(self.prices)
            qty, sl, tp = self.risk.plan(inst, side, price, atr, equity)
            if qty <= 0:
                log.info("%s: position size rounds to zero", inst.name)
                return
            pos = self.broker.open(inst, side, qty, price, sl, tp)
            if pos:
                log.info("OPEN %s %s %g @ %.5g  SL %.5g  TP %.5g  score %+.2f  [%s]",
                         side.value.upper(), inst.name, pos.qty, pos.entry_price, sl, tp, decision.score,
                         "; ".join(f"{s.source} {s.score:+.2f}" for s in decision.signals[:3]))
        else:
            log.debug("%s: hold (score %+.2f)", inst.name, decision.score)

    def _close(self, inst: Instrument, price: float, reason: str) -> None:
        t = self.broker.close(inst, price, reason)
        if t:
            log.info("CLOSE %s %s @ %.5g pnl %.2f USD (%s)", t.side.value, inst.name, t.exit_price, t.pnl_usd, reason)
            self.journal.trade(t)

    def flatten(self, reason: str = "manual flatten") -> None:
        for name in list(self.broker.positions()):
            inst = self.instruments[name]
            if name not in self.prices:
                self.analyze(inst)
            self._close(inst, self.prices.get(name, self.broker.positions()[name].entry_price), reason)

    # ---- loop -------------------------------------------------------------
    def run(self, once: bool = False) -> None:
        signal.signal(signal.SIGINT, self._stop)
        signal.signal(signal.SIGTERM, self._stop)
        log.info("starting %s trading on %s (%s bars, every %ss)", self.cfg.mode.upper(),
                 ", ".join(self.instruments), self.cfg.timeframe, self.cfg.loop_seconds)
        while self._running:
            started = time.time()
            self.step()
            if once:
                break
            while self._running and time.time() - started < self.cfg.loop_seconds:
                time.sleep(1)
        log.info("stopped")

    def _stop(self, *_):
        log.info("shutdown requested")
        self._running = False
