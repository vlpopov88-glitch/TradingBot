"""Bar-by-bar backtest of the technical ensemble with the same risk rules as live.

News and positioning have no reliable history here, so the backtest measures the
technical edge alone. Decisions use closed bars only and fill at the next bar's open.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from . import indicators as ind
from .brokers.paper import PaperBroker
from .models import Instrument, Side, Trade
from .risk import RiskConfig, RiskManager
from .strategies import Ensemble, EnsembleConfig


@dataclass
class BacktestResult:
    instrument: str
    start_equity: float
    end_equity: float
    trades: list[Trade] = field(default_factory=list)
    equity_curve: pd.Series = field(default_factory=pd.Series)

    @property
    def total_return(self) -> float:
        return self.end_equity / self.start_equity - 1

    @property
    def max_drawdown(self) -> float:
        eq = self.equity_curve
        return float((eq / eq.cummax() - 1).min()) if len(eq) else 0.0

    @property
    def win_rate(self) -> float:
        return sum(t.pnl_usd > 0 for t in self.trades) / len(self.trades) if self.trades else 0.0

    @property
    def profit_factor(self) -> float:
        gains = sum(t.pnl_usd for t in self.trades if t.pnl_usd > 0)
        losses = -sum(t.pnl_usd for t in self.trades if t.pnl_usd < 0)
        return gains / losses if losses else float("inf") if gains else 0.0

    def sharpe(self) -> float:
        """Annualized, inferring bars-per-year from the curve's own timestamps."""
        eq = self.equity_curve
        r = eq.pct_change().dropna()
        if len(r) < 2 or not r.std():
            return 0.0
        years = (eq.index[-1] - eq.index[0]).total_seconds() / (365.25 * 86400)
        return float(r.mean() / r.std() * np.sqrt(len(r) / years))

    def report(self) -> str:
        return (f"{self.instrument}: return {self.total_return:+.2%} | max DD {self.max_drawdown:.2%} | "
                f"Sharpe {self.sharpe():.2f} | trades {len(self.trades)} | "
                f"win {self.win_rate:.0%} | PF {self.profit_factor:.2f} | "
                f"equity {self.start_equity:,.0f} -> {self.end_equity:,.0f}")


def run_backtest(inst: Instrument, df: pd.DataFrame, cash: float = 10_000.0,
                 risk_cfg: RiskConfig | None = None, ens_cfg: EnsembleConfig | None = None,
                 window: int = 400) -> BacktestResult:
    ens = Ensemble(config=ens_cfg)
    risk = RiskManager(risk_cfg)
    broker = PaperBroker(cash)
    atr_all = ind.atr(df)
    trades: list[Trade] = []
    curve = {}
    pending: Side | str | None = None   # action decided on the previous close

    for i in range(ens.min_bars, len(df)):
        bar = df.iloc[i]
        ts = df.index[i]
        o, h, l, c = bar["open"], bar["high"], bar["low"], bar["close"]
        pos = broker.positions().get(inst.name)

        # Execute last bar's decision at this bar's open.
        if pending == "close" and pos:
            trades.append(broker.close(inst, o, "signal flip", when=ts))
            pos = None
        elif isinstance(pending, Side) and not pos:
            ok, _ = risk.can_open(inst, broker.positions(), {inst.name: inst})
            if ok:
                qty, sl, tp = risk.plan(inst, pending, o, float(atr_all.iloc[i - 1]), broker.equity({inst.name: o}))
                if qty > 0:
                    pos = broker.open(inst, pending, qty, o, sl, tp, when=ts)
        pending = None

        # Intrabar stop / target, then trail.
        if pos:
            hit = risk.stop_hit(pos, h, l, o)
            if hit:
                trades.append(broker.close(inst, hit[0], hit[1], when=ts))
                pos = None
            else:
                risk.update_trailing(pos, h, l, float(atr_all.iloc[i]))

        broker.mark(inst, c)
        eq = broker.equity({inst.name: c})
        risk.on_equity(eq, ts.to_pydatetime())
        curve[ts] = eq

        d = ens.decide(inst.name, df.iloc[max(0, i - window + 1): i + 1], (), pos.side.sign if pos else 0)
        if d.action == "close":
            pending = "close"
        elif d.action in ("buy", "sell"):
            pending = Side(d.action)

    last = df["close"].iloc[-1]
    if broker.positions():
        trades.append(broker.close(inst, last, "end of test", when=df.index[-1]))
    end = broker.equity({inst.name: last})
    return BacktestResult(inst.name, cash, end, trades, pd.Series(curve, dtype=float))
