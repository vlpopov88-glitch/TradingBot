from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd
import pytest

from tradingbot import indicators as ind
from tradingbot.backtest import run_backtest
from tradingbot.brokers.paper import PaperBroker
from tradingbot.config import Config, load_config
from tradingbot.data.market import SyntheticData
from tradingbot.engine import TradingEngine, closed_bars
from tradingbot.journal import Journal
from tradingbot.models import AssetClass, Instrument, Position, Side, Signal
from tradingbot.news import NewsAnalyzer, Headline, parse_feed, score_headline
from tradingbot.positioning import cot_score, crypto_score
from tradingbot.risk import RiskConfig, RiskManager
from tradingbot.strategies import Ensemble, TrendFollowing

XAU = Instrument("XAU/USD", AssetClass.METAL, "synthetic", "XAU", "XAU_USD", qty_step=1, min_qty=1)
EUR = Instrument("EUR/USD", AssetClass.FOREX, "synthetic", "EUR", "EUR_USD")
JPY = Instrument("USD/JPY", AssetClass.FOREX, "synthetic", "USDJPY", "USD_JPY")
BTC = Instrument("BTC/USD", AssetClass.CRYPTO, "synthetic", "BTC", "BTC/USDT", qty_step=0.0001)


def trend_df(n=300, slope=0.002, start=100.0):
    idx = pd.date_range("2025-01-01", periods=n, freq="h", tz="UTC")
    rng = np.random.default_rng(1)
    close = start * np.exp(np.cumsum(slope + rng.normal(0, 0.001, n)))
    open_ = np.r_[start, close[:-1]]
    return pd.DataFrame({"open": open_, "high": np.maximum(open_, close) * 1.001,
                         "low": np.minimum(open_, close) * 0.999, "close": close, "volume": 1.0}, index=idx)


# ---- indicators -----------------------------------------------------------
def test_indicators_basic():
    df = trend_df()
    r = ind.rsi(df["close"]).dropna()
    assert ((r >= 0) & (r <= 100)).all()
    assert r.iloc[-1] > 60                       # steady uptrend -> high RSI
    assert (ind.atr(df).dropna() > 0).all()
    assert ind.adx(df).iloc[-1] > 25             # strong trend


# ---- news -----------------------------------------------------------------
def test_hawkish_fed_is_bullish_usd_bearish_gold():
    s = score_headline("Fed signals another rate hike as Powell turns hawkish")
    assert s["USD"] > 0
    assert s.get("XAU*", 0) < 0


def test_gold_headlines():
    assert score_headline("Gold prices surge to record high on safe-haven demand")["XAU"] > 0.5
    assert score_headline("Gold slumps as dollar strengthens")["XAU"] < 0
    assert score_headline("Bitcoin plunges after exchange hack")["BTC"] < -0.5
    assert score_headline("Local bakery opens new store") == {}


def test_parse_rss_and_pair_signal():
    xml = b"""<?xml version="1.0"?><rss><channel>
      <item><title>Dollar rallies as Fed turns hawkish</title><pubDate>Sat, 11 Oct 2026 08:00:00 GMT</pubDate></item>
      <item><title>ECB's Lagarde dovish, euro slides</title><pubDate>2026-10-11T07:00:00Z</pubDate></item>
      <item><title>No date here, euro drops</title></item>
    </channel></rss>"""
    heads = parse_feed(xml, "test")
    assert len(heads) == 3
    assert all(h.published.tzinfo for h in heads)
    na = NewsAnalyzer(feeds=[])
    na.headlines = {h.title: h for h in heads}
    sig = na.signal(EUR, now=datetime(2026, 10, 11, 9, tzinfo=timezone.utc))
    assert sig.score < 0                          # USD up + EUR down => EUR/USD bearish


def test_news_decay():
    na = NewsAnalyzer(feeds=[], half_life_hours=6)
    now = datetime.now(timezone.utc)
    na.headlines = {"a": Headline("Gold soars", now - timedelta(hours=1), "t"),
                    "b": Headline("Gold plunges", now - timedelta(hours=30), "t")}
    assert na.signal(XAU, now).score > 0          # fresh bullish outweighs stale bearish


# ---- positioning ----------------------------------------------------------
def _cot_rows(nets):
    return [{"open_interest_all": "1000", "noncomm_positions_long_all": str(500 + n * 500),
             "noncomm_positions_short_all": str(500 - n * 500)} for n in nets]


def test_cot_crowded_long_is_contrarian():
    nets = [0.6] + list(np.linspace(0.5, -0.5, 50))   # newest at historic extreme
    assert cot_score(_cot_rows(nets)).score < 0
    assert cot_score(_cot_rows(nets), invert=True).score > 0


def test_cot_follow_flows_midrange():
    nets = [0.1, 0.05, 0.0, -0.05, -0.1] + list(np.linspace(0.5, -0.5, 40))
    assert cot_score(_cot_rows(nets)).score > 0


def test_crypto_positioning():
    assert crypto_score([1.0], [1.0], 0.002).score < 0         # extreme funding -> fade longs
    assert crypto_score([1.5, 1.2], [], 0.0001).score > 0      # top traders adding longs
    assert crypto_score([], [], 0.0).confidence == 0


# ---- risk -----------------------------------------------------------------
def test_sizing_risks_one_percent():
    rm = RiskManager(RiskConfig(risk_per_trade=0.01, atr_stop_mult=2, reward_risk=2))
    qty, sl, tp = rm.plan(EUR, Side.BUY, 1.10, 0.002, 10_000)
    assert sl == pytest.approx(1.096) and tp == pytest.approx(1.108)
    assert qty * (1.10 - sl) == pytest.approx(100)


def test_sizing_usd_base_pair():
    rm = RiskManager(RiskConfig(max_leverage={"forex": 100}))
    qty, sl, _ = rm.plan(JPY, Side.SELL, 150.0, 0.5, 10_000)
    assert sl == pytest.approx(151.0)
    assert qty * 1.0 / 151.0 == pytest.approx(100, rel=0.01)   # loss in JPY converted to USD


def test_leverage_cap_and_rounding():
    rm = RiskManager(RiskConfig(max_leverage={"crypto": 1.0}))
    qty, _, _ = rm.plan(BTC, Side.BUY, 60_000, 10, 10_000)     # tiny stop -> huge size, capped
    assert qty * 60_000 <= 10_000
    qty, _, _ = rm.plan(XAU, Side.BUY, 2400, 0.01, 100)       # rounds below min_qty
    assert qty in (0.0, 1.0)


def test_limits_and_halts():
    rm = RiskManager(RiskConfig(max_open_positions=1, max_daily_loss=0.03, max_drawdown=0.10))
    insts = {i.name: i for i in (XAU, EUR)}
    pos = {"XAU/USD": Position("XAU/USD", Side.BUY, 1, 2400, 2390, 2420)}
    assert rm.can_open(EUR, pos, insts) == (False, "max open positions")
    t = datetime(2026, 1, 1, tzinfo=timezone.utc)
    rm.on_equity(10_000, t)
    rm.on_equity(9_600, t)
    assert "daily" in rm.halted_reason
    rm.on_equity(9_600, t + timedelta(days=1))             # new day lifts daily halt
    assert rm.halted_reason is None
    rm.on_equity(8_900, t + timedelta(days=1))
    assert "drawdown" in rm.halted_reason


def test_stops_and_trailing():
    rm = RiskManager(RiskConfig(trailing_atr_mult=2))
    p = Position("XAU/USD", Side.BUY, 1, 100, 95, 110)
    assert rm.stop_hit(p, 101, 94) == (95, "stop_loss")
    assert rm.stop_hit(p, 111, 99) == (110, "take_profit")
    assert rm.stop_hit(p, 102, 90, open_=92) == (92, "stop_loss")   # gap through stop
    rm.update_trailing(p, 108, 100, 1.0)
    assert p.stop_loss == 106
    rm.update_trailing(p, 107, 100, 1.0)
    assert p.stop_loss == 106                                       # never loosens
    s = Position("XAU/USD", Side.SELL, 1, 100, 105, 90)
    assert rm.stop_hit(s, 106, 99) == (105, "stop_loss")
    assert rm.stop_hit(s, 100, 89) == (90, "take_profit")


# ---- paper broker ---------------------------------------------------------
def test_paper_broker_pnl_and_persistence(tmp_path):
    path = str(tmp_path / "acct.json")
    b = PaperBroker(10_000, state_file=path, cost_bps={"forex": 0})
    b.open(EUR, Side.BUY, 10_000, 1.10, 1.09, 1.12)
    assert b.equity({"EUR/USD": 1.11}) == pytest.approx(10_100)
    b2 = PaperBroker(0, state_file=path, cost_bps={"forex": 0})                # restored from disk
    assert "EUR/USD" in b2.positions()
    t = b2.close(EUR, 1.12, "tp")
    assert t.pnl_usd == pytest.approx(200)
    assert b2.cash == pytest.approx(10_200)


def test_paper_costs_and_usd_base():
    b = PaperBroker(10_000, cost_bps={"forex": 1.0})
    b.open(JPY, Side.BUY, 1_000, 150.0, 149, 152)
    t = b.close(JPY, 150.0, "flat")
    assert t.pnl_usd < 0                                  # paid the spread both ways


# ---- strategies -----------------------------------------------------------
def test_trend_strategies_agree_with_trend():
    up, down = trend_df(slope=0.002), trend_df(slope=-0.002)
    assert TrendFollowing().evaluate(up).score > 0.5
    assert TrendFollowing().evaluate(down).score < -0.5
    ens = Ensemble()
    assert ens.decide("X", up).action == "buy"
    assert ens.decide("X", down).action == "sell"


def test_ensemble_blends_context_and_exits():
    ens = Ensemble()
    up = trend_df()
    bearish = [Signal("news", -1, 1), Signal("positioning", -1, 1)]
    assert ens.decide("X", up, bearish).score < ens.decide("X", up).score
    assert ens.decide("X", trend_df(slope=-0.002), position_side=1).action == "close"
    assert ens.decide("X", up, position_side=1).action == "hold"


# ---- backtest & engine ----------------------------------------------------
def test_backtest_runs():
    df = SyntheticData().candles("XAU", limit=500)
    res = run_backtest(XAU, df)
    assert len(res.equity_curve) > 0
    assert res.end_equity > 0
    assert all(t.closed_at >= t.opened_at for t in res.trades)
    assert "return" in res.report()


def test_closed_bars_drops_forming_bar():
    df = trend_df(10)
    now = df.index[-1] + pd.Timedelta(minutes=30)
    assert len(closed_bars(df, "1h", now)) == 9
    assert len(closed_bars(df, "1h", df.index[-1] + pd.Timedelta(hours=1))) == 10


class _StubCtx:
    def __init__(self, score):
        self.score = score

    def update(self, *a, **k):
        return 0

    def signal(self, inst, *a):
        return Signal("news", self.score, 1.0, "stub")


def test_engine_cycle(tmp_path):
    cfg = Config(state_dir=str(tmp_path), instruments=[XAU, BTC], history_bars=400)
    eng = TradingEngine(cfg, broker=PaperBroker(10_000), news=_StubCtx(0.0), positioning=None)
    eng.step()
    eng.step()
    assert set(eng.prices) == {"XAU/USD", "BTC/USD"}
    rows = eng.journal.db.execute("SELECT COUNT(*) FROM equity").fetchone()[0]
    assert rows >= 1
    (tmp_path / "STOP").write_text("")
    eng.flatten()
    assert not eng.broker.positions()


def test_engine_respects_kill_switch(tmp_path):
    (tmp_path / "STOP").write_text("")
    cfg = Config(state_dir=str(tmp_path), instruments=[XAU], history_bars=400)
    cfg.ensemble.entry_threshold = 0.0           # would trade on anything
    eng = TradingEngine(cfg, broker=PaperBroker(10_000), news=_StubCtx(1.0), positioning=None)
    eng.step()
    assert not eng.broker.positions()


def test_journal(tmp_path):
    j = Journal(str(tmp_path / "j.db"))
    assert j.summary() == {"trades": 0, "pnl_usd": 0, "win_rate": 0.0}


def test_default_config_loads():
    cfg = load_config("config.yaml")
    assert cfg.mode == "paper"
    names = set(cfg.by_name)
    assert {"XAU/USD", "EUR/USD", "BTC/USD"} <= names


def test_engine_skips_stale_market(tmp_path, monkeypatch):
    stale = trend_df(400)                         # ends in Jan 2025 -> stale
    monkeypatch.setattr(SyntheticData, "candles", lambda self, *a, **k: stale)
    cfg = Config(state_dir=str(tmp_path), instruments=[XAU], history_bars=400)
    cfg.ensemble.entry_threshold = 0.0
    eng = TradingEngine(cfg, broker=PaperBroker(10_000), news=_StubCtx(1.0), positioning=None)
    eng.step()
    assert not eng.broker.positions()
