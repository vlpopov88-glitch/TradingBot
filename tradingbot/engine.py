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
from .econ_calendar import EconomicCalendar
from .journal import Journal
from .models import Decision, Instrument, Side, Signal
from .news import NewsAnalyzer
from .notify import Notifier
from .positioning import PositioningAnalyzer
from .risk import RiskManager
from .strategies import Ensemble

log = logging.getLogger(__name__)

TF = {"1m": "1min", "5m": "5min", "15m": "15min", "30m": "30min", "1h": "1h", "4h": "4h", "1d": "1D"}
INTRADAY = {"1m", "5m", "15m", "30m", "1h"}
STALE_AFTER = pd.Timedelta(hours=3)   # no new price for this long = market closed or feed dead


def closed_bars(df: pd.DataFrame, timeframe: str, now: datetime | None = None) -> pd.DataFrame:
    """Keep only bars that have fully closed, so signals don't repaint.

    Some feeds (Yahoo) append a live tick after the forming bar, so more than one
    trailing row can be unfinished: a bar is closed once start + timeframe <= now.
    """
    now = pd.Timestamp(now or datetime.now(timezone.utc))
    return df[df.index <= now - pd.Timedelta(TF[timeframe])]


def make_news(cfg: Config) -> NewsAnalyzer | None:
    if not cfg.news_enabled:
        return None
    llm = None
    if cfg.news_llm_enabled:
        from .llm_news import LLMNewsScorer, credentials_available
        if not credentials_available():
            log.info("news: no ANTHROPIC_API_KEY - using keyword sentiment")
        else:
            try:
                llm = LLMNewsScorer(cfg.news_llm_model, cfg.news_llm_effort)
                log.info("news: scoring headlines with %s", cfg.news_llm_model)
            except Exception as exc:
                log.warning("news: LLM scorer unavailable (%s) - using keyword sentiment", exc)
    return NewsAnalyzer(cfg.news_feeds, cfg.news_half_life_hours, llm=llm)


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
                 news: NewsAnalyzer | None = None, positioning: PositioningAnalyzer | None = None,
                 calendar: EconomicCalendar | None = None, notifier: Notifier | None = None):
        self.cfg = cfg
        self.instruments = cfg.by_name
        os.makedirs(cfg.state_dir, exist_ok=True)
        self.broker = broker or make_broker(cfg)
        self.journal = journal or Journal(os.path.join(cfg.state_dir, "journal.db"))
        self.ensemble = Ensemble(config=cfg.ensemble)
        self.risk = RiskManager(cfg.risk, state_file=os.path.join(cfg.state_dir, "risk_state.json"))
        self.news = news if news is not None else make_news(cfg)
        self.positioning = positioning if positioning is not None else (
            PositioningAnalyzer(crypto=cfg.positioning_crypto) if cfg.positioning_enabled else None)
        self.calendar = calendar if calendar is not None else (
            EconomicCalendar(cfg.calendar_minutes_before, cfg.calendar_minutes_after, cfg.calendar_impacts)
            if cfg.calendar_enabled else None)
        self.prices: dict[str, float] = {}
        self.last_exit: dict[str, pd.Timestamp] = {}
        self.stale: dict[str, bool] = {}
        self.notifier = notifier or Notifier(prefix="[paper] " if cfg.mode == "paper" else "")
        self._running = True
        self._failures = 0
        self._summary_day = None
        self.kill_file = os.path.join(cfg.state_dir, "STOP")
        self.heartbeat_file = os.path.join(cfg.state_dir, "heartbeat")

    # ---- analysis ---------------------------------------------------------
    def context_signals(self, inst: Instrument) -> list[Signal]:
        out = []
        if self.news:
            out.append(self.news.signal(inst))
        if self.positioning:
            out.append(self.positioning.signal(inst))
        return out

    def analyze(self, inst: Instrument, allow_stale: bool = False) -> tuple[Decision, pd.DataFrame] | None:
        """Return (decision, closed bars), or None when there is nothing safe to act on.
        ``allow_stale`` still analyzes a closed market (for display, never for trading)."""
        provider = get_provider(inst.data_source)
        df = provider.candles(inst.data_symbol, self.cfg.timeframe, self.cfg.history_bars)
        if df.empty:
            return None
        # Daily/4h bars can't tell whether the market is open right now; hourly bars can.
        live = df if self.cfg.timeframe in INTRADAY else provider.candles(inst.data_symbol, "1h", 3)
        self.prices[inst.name] = float(live["close"].iloc[-1])
        age = pd.Timestamp.now(tz="UTC") - live.index[-1]
        self.stale[inst.name] = age > STALE_AFTER
        if age > STALE_AFTER and not allow_stale:
            # Weekend / holiday / dead feed: never trade on stale prices.
            log.info("%s: no price for %s - market closed or data stale, skipping", inst.name, age)
            return None
        bars = closed_bars(df, self.cfg.timeframe)
        if len(bars) < self.ensemble.min_bars:
            log.warning("%s: only %d bars, need %d", inst.name, len(bars), self.ensemble.min_bars)
            return None
        pos = self.broker.positions().get(inst.name)
        side = pos.side.sign if pos else 0
        return self.ensemble.decide(inst.name, bars, self.context_signals(inst), side), bars

    # ---- one cycle --------------------------------------------------------
    def step(self) -> None:
        if self.news:
            self.news.update()
        if self.calendar:
            self.calendar.update()
        for t in self.broker.sync(self.instruments):
            log.info("broker closed %s: pnl %.2f (%s)", t.instrument, t.pnl_usd, t.reason)
            self.journal.trade(t)
            self.last_exit[t.instrument] = pd.Timestamp.now(tz="UTC")
            self._notify_close(t)

        for inst in self.instruments.values():
            try:
                self._step_instrument(inst)
            except Exception:
                log.exception("%s: cycle failed", inst.name)

        positions = self.broker.positions()
        eq = self.broker.equity(self.prices)
        was_halted = self.risk.halted_reason
        self.risk.on_equity(eq)
        if self.risk.halted_reason and self.risk.halted_reason != was_halted:
            self.notifier.send(f"⛔ New trades halted: {self.risk.halted_reason}. Equity {eq:,.2f}. "
                               "Open positions keep their stops. Review, then run `python -m tradingbot resume`.")
        self.journal.equity(eq, len(positions))
        log.info("equity %.2f | open: %s%s", eq,
                 ", ".join(f"{n} {p.side.value} {p.qty:g}@{p.entry_price:.5g}" for n, p in positions.items()) or "none",
                 f" | HALTED: {self.risk.halted_reason}" if self.risk.halted_reason else "")
        self._daily_summary(eq, positions)
        with open(self.heartbeat_file, "w") as f:
            f.write(datetime.now(timezone.utc).isoformat())

    def _daily_summary(self, equity: float, positions) -> None:
        now = datetime.now(timezone.utc)
        if now.hour < self.cfg.notify_summary_hour or self._summary_day == now.date():
            return
        self._summary_day = now.date()
        start = self.risk.day_start_equity or equity
        today = [t for t in self.journal.recent_trades(200) if t[0][:10] == now.date().isoformat()]
        lines = [f"📊 Daily summary {now:%Y-%m-%d}",
                 f"Equity {equity:,.2f} ({equity / start - 1:+.2%} today)",
                 f"Closed today: {len(today)} trade(s), P&L {sum(t[6] for t in today):+,.2f}"]
        lines += [f"Open: {n} {p.side.value} {p.qty:g} @ {p.entry_price:.5g} (SL {p.stop_loss:.5g})"
                  for n, p in positions.items()] or ["No open positions"]
        if self.risk.halted_reason:
            lines.append(f"⛔ Halted: {self.risk.halted_reason}")
        self.notifier.send("\n".join(lines))

    def _step_instrument(self, inst: Instrument) -> None:
        res = self.analyze(inst)
        if res is None:
            return
        decision, bars = res
        price = self.prices[inst.name]
        self.journal.snapshot(inst.name, bars.index[-1], price, decision)
        if isinstance(self.broker, PaperBroker):
            self.broker.mark(inst, price)
        atr = float(ind.atr(bars).iloc[-1])
        pos = self.broker.positions().get(inst.name)

        # 1. Manage the open position: stops/targets bar by bar, then the signal exit.
        if pos:
            if self._manage_position(inst, pos, bars, price, atr):
                return
            if decision.action == "close":
                self._close(inst, price, f"signal flip ({decision.score:+.2f})")
                return

        # 2. New entries.
        if decision.action in ("buy", "sell"):
            self.journal.decision(decision)
            side = Side(decision.action)
            exited = self.last_exit.get(inst.name)
            wait = self.cfg.risk.cooldown_bars * pd.Timedelta(TF[self.cfg.timeframe])
            if exited is not None and pd.Timestamp.now(tz="UTC") - exited < wait:
                log.info("%s: %s signal %+.2f - cooling down after exit", inst.name, side.value, decision.score)
                return
            if os.path.exists(self.kill_file):
                log.warning("kill switch %s present: no new entries", self.kill_file)
                return
            if side is Side.SELL and not self.broker.can_short(inst):
                return
            event = self.calendar.blackout(inst) if self.calendar else None
            if event:
                log.info("%s: %s signal %+.2f held back - %s %s at %s UTC", inst.name, side.value,
                         decision.score, event.currency, event.title, f"{event.time.astimezone(timezone.utc):%H:%M}")
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
                why = "; ".join(f"{s.source} {s.score:+.2f}" for s in decision.signals[:3])
                log.info("OPEN %s %s %g @ %.5g  SL %.5g  TP %.5g  score %+.2f  [%s]",
                         side.value.upper(), inst.name, pos.qty, pos.entry_price, sl, tp, decision.score, why)
                self.notifier.send(f"🟢 OPEN {side.value.upper()} {inst.name} {pos.qty:g} @ {pos.entry_price:.5g}\n"
                                   f"Stop {sl:.5g} | Target {tp:.5g} | risk ~{self.cfg.risk.risk_per_trade:.0%}\n"
                                   f"Score {decision.score:+.2f} ({why})")
        else:
            log.debug("%s: hold (score %+.2f)", inst.name, decision.score)

    def _manage_position(self, inst: Instrument, pos, bars: pd.DataFrame, price: float, atr: float) -> bool:
        """Run stops over each newly closed bar in order, then over the live price.

        Order matters: a trailing stop raised by a later high must never be compared
        with an earlier low. Within a bar the stop is checked before trailing it.
        Returns True if the position was closed.
        """
        tf = pd.Timedelta(TF[self.cfg.timeframe])
        start = (pd.Timestamp(pos.checked_until) if pos.checked_until
                 else pd.Timestamp(pos.opened_at).floor(tf))   # skip the entry bar's pre-entry range
        for ts, bar in bars[bars.index > start].iterrows():
            hit = self.risk.stop_hit(pos, bar["high"], bar["low"], bar["open"])
            if hit:
                self._close(inst, *hit)
                return True
            old_sl = pos.stop_loss
            self.risk.update_trailing(pos, bar["high"], bar["low"], atr)
            if pos.stop_loss != old_sl:
                self.broker.update_stop(inst, pos)
            pos.checked_until = ts.isoformat()
        self.broker.save_state()
        hit = self.risk.stop_hit(pos, price, price, price)   # fills at the live price
        if hit:
            self._close(inst, *hit)
            return True
        return False

    def _close(self, inst: Instrument, price: float, reason: str) -> None:
        t = self.broker.close(inst, price, reason)
        if t:
            log.info("CLOSE %s %s @ %.5g pnl %.2f USD (%s)", t.side.value, inst.name, t.exit_price, t.pnl_usd, reason)
            self.journal.trade(t)
            self.last_exit[inst.name] = pd.Timestamp.now(tz="UTC")
            self._notify_close(t)

    def _notify_close(self, t) -> None:
        icon = "✅" if t.pnl_usd > 0 else "🔴"
        self.notifier.send(f"{icon} CLOSE {t.side.value} {t.instrument} @ {t.exit_price:.5g}\n"
                           f"P&L {t.pnl_usd:+,.2f} USD ({t.reason})")

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
        if not once:
            self.notifier.send(f"🤖 TradingBot started in {self.cfg.mode.upper()} mode: "
                               f"{', '.join(self.instruments)} ({self.cfg.timeframe} bars)")
        while self._running:
            started = time.time()
            try:
                self.step()
                self._failures = 0
            except Exception:
                # A broker/network outage must not kill a bot that is holding positions.
                self._failures += 1
                log.exception("cycle failed (%d in a row)", self._failures)
                if self._failures in (3, 12, 60):
                    self.notifier.send(f"⚠️ {self._failures} cycles in a row failed - check the log "
                                       f"({self.cfg.state_dir}/bot.log). The bot keeps retrying.")
            if once:
                break
            while self._running and time.time() - started < self.cfg.loop_seconds:
                time.sleep(1)
        log.info("stopped")

    def _stop(self, *_):
        log.info("shutdown requested")
        self._running = False
