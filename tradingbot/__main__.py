"""Command line: python -m tradingbot <command>"""
from __future__ import annotations

import argparse
import logging
import os
import sys
from datetime import timezone

import pandas as pd

from .config import load_config


def _setup_logging(state_dir: str, verbose: bool) -> None:
    os.makedirs(state_dir, exist_ok=True)
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s %(levelname)-7s %(name)s: %(message)s",
        handlers=[logging.StreamHandler(), logging.FileHandler(os.path.join(state_dir, "bot.log"))],
    )
    logging.getLogger("urllib3").setLevel(logging.WARNING)


def cmd_run(cfg, args):
    from .engine import TradingEngine
    if cfg.mode == "live" and not args.confirm_live:
        sys.exit("Refusing to trade real money without --confirm-live. Paper trade first.")
    TradingEngine(cfg).run(once=args.once)


def cmd_analyze(cfg, args):
    """Print what the bot thinks of every market right now, without trading."""
    from .engine import TradingEngine
    from .brokers import PaperBroker
    eng = TradingEngine(cfg, broker=PaperBroker(cfg.starting_cash))
    if eng.news:
        eng.news.update(force=True)
    if eng.calendar:
        eng.calendar.update(force=True)
    for inst in eng.instruments.values():
        try:
            res = eng.analyze(inst, allow_stale=True)
        except Exception as exc:
            print(f"\n{inst.name}: data error: {exc}")
            continue
        if not res:
            print(f"\n{inst.name}: not enough history")
            continue
        d, _ = res
        closed = "  [market closed - last price shown]" if eng.stale.get(inst.name) else ""
        print(f"\n{inst.name} @ {eng.prices[inst.name]:.5g}  ->  {d.action.upper()}  (score {d.score:+.2f}){closed}")
        for s in d.signals:
            print(f"   {s.source:<15} {s.score:+.2f}  conf {s.confidence:.2f}  {s.reason}")
        if eng.calendar:
            for e in eng.calendar.upcoming(inst)[:3]:
                print(f"   ! {e.time.astimezone(timezone.utc):%a %H:%M} UTC  {e.currency} {e.title} (no new entries "
                      f"{cfg.calendar_minutes_before}m before to {cfg.calendar_minutes_after}m after)")


def cmd_news(cfg, args):
    from .engine import make_news
    na = make_news(cfg)
    if na is None:
        sys.exit("news is disabled in config")
    na.update(force=True)
    rows = sorted(na.headlines.values(), key=lambda h: h.published, reverse=True)
    for h in rows[: args.limit]:
        scores = na.score(h.title)
        if scores or args.all:
            tag = " ".join(f"{k}{v:+.2f}" for k, v in scores.items())
            print(f"{h.published:%m-%d %H:%M}  {tag:<28} {h.title[:110]}")


def cmd_backtest(cfg, args):
    from .backtest import run_backtest
    from .data import get_provider
    from .research import YAHOO, cot_signals
    tf = args.timeframe or cfg.timeframe
    names = [args.instrument] if args.instrument else list(cfg.by_name)
    for name in names:
        inst = cfg.by_name[name]
        if args.source:
            df = get_provider(args.source).candles(inst.data_symbol, tf, args.bars)
        else:   # Yahoo has the longest free history for every market
            df = get_provider("yahoo").candles(YAHOO.get(name, inst.data_symbol), tf, args.bars)
        context = None
        if (cfg.positioning_enabled and not args.no_positioning and inst.cot_code
                and tf in ("4h", "1d") and args.source != "synthetic"):
            context = {"positioning": (cot_signals(inst.cot_code, invert=inst.base == "USD", weeks=700), 0.6)}
        res = run_backtest(inst, df, cfg.starting_cash, cfg.risk, cfg.ensemble, context=context)
        print(res.report() + ("  [+COT positioning]" if context else ""))
        if args.trades:
            for t in res.trades:
                print(f"   {t.opened_at:%Y-%m-%d %H:%M} {t.side.value:<4} {t.entry_price:.5g} -> "
                      f"{t.exit_price:.5g}  {t.pnl_usd:+9.2f}  {t.reason}")


def cmd_research(cfg, args):
    """Measure on years of history whether each signal predicts future returns."""
    from . import research as rs
    insts = [cfg.by_name[args.instrument]] if args.instrument else cfg.instruments
    sections = []
    for what in args.what.split(","):
        if what == "technical":
            for tf in args.timeframes.split(","):
                detail, pooled = rs.research_technical(insts, tf, cfg.ensemble)
                sections.append((f"TECHNICAL strategies, {tf} bars (all markets pooled)", pooled, detail))
        elif what == "cot":
            detail, pooled = rs.research_cot(insts)
            sections.append(("POSITIONING - CFTC COT, weekly (gold + FX pooled)", pooled, detail))
        elif what == "crypto":
            detail, pooled = rs.research_crypto(insts)
            sections.append(("POSITIONING - crypto (OKX funding / trader ratios)", pooled, detail))
    text = [rs.__doc__.split("\n\n", 1)[1]]
    for title, pooled, detail in sections:
        text.append(rs.format_table(title, pooled))
        if args.detail:
            text.append(rs.format_table("  per market:", detail))
    report = "\n".join(text)
    print(report)
    path = os.path.join(cfg.state_dir, "research_report.txt")
    with open(path, "w") as f:
        f.write(report)
    print(f"\nsaved to {path}")


def cmd_evaluate(cfg, args):
    """Score the signals recorded during (paper) trading against what happened next."""
    from . import research as rs
    from .journal import Journal
    path = os.path.join(cfg.state_dir, "journal.db")
    if not os.path.exists(path):
        sys.exit("no journal yet - run the bot first")
    snaps = Journal(path).snapshots()
    if snaps.empty:
        sys.exit("no signal snapshots recorded yet - let the bot run for a while")
    days = (pd.to_datetime(snaps["ts"]).max() - pd.to_datetime(snaps["ts"]).min()).days
    print(f"{snaps['bar_ts'].nunique()} bars recorded over {days} days across {snaps['instrument'].nunique()} markets")
    if days < 14:
        print("note: under ~2 weeks of data the numbers below are mostly noise")
    detail, pooled = rs.evaluate_journal(snaps, cfg)
    print(rs.format_table("LIVE SIGNALS (all markets pooled)", pooled))
    if args.detail:
        print(rs.format_table("per market:", detail))


def cmd_status(cfg, args):
    from .journal import Journal
    path = os.path.join(cfg.state_dir, "journal.db")
    if not os.path.exists(path):
        sys.exit("no journal yet - run the bot first")
    j = Journal(path)
    print(j.summary())
    for row in j.recent_trades(args.limit):
        print("  ", row)


def cmd_flatten(cfg, args):
    from .engine import TradingEngine
    if cfg.mode == "live" and not args.confirm_live:
        sys.exit("add --confirm-live to flatten live positions")
    TradingEngine(cfg).flatten()


def main(argv=None):
    p = argparse.ArgumentParser(prog="tradingbot", description="XAU/USD, forex & crypto trading bot")
    p.add_argument("-c", "--config", default="config.yaml")
    p.add_argument("-v", "--verbose", action="store_true")
    sub = p.add_subparsers(dest="cmd", required=True)

    r = sub.add_parser("run", help="run the trading loop")
    r.add_argument("--once", action="store_true", help="run one cycle and exit")
    r.add_argument("--confirm-live", action="store_true")
    sub.add_parser("analyze", help="show current signals for every market (no trading)")
    n = sub.add_parser("news", help="show scored headlines")
    n.add_argument("--limit", type=int, default=40)
    n.add_argument("--all", action="store_true", help="include headlines with no score")
    b = sub.add_parser("backtest", help="backtest technical strategies on history")
    b.add_argument("--instrument")
    b.add_argument("--bars", type=int, default=2000)
    b.add_argument("--timeframe")
    b.add_argument("--source", help="override data source, e.g. synthetic")
    b.add_argument("--trades", action="store_true")
    b.add_argument("--no-positioning", action="store_true", help="technical signals only")
    rp = sub.add_parser("research", help="measure each signal's predictive power on history")
    rp.add_argument("--what", default="technical,cot,crypto")
    rp.add_argument("--timeframes", default="1h,4h,1d")
    rp.add_argument("--instrument")
    rp.add_argument("--detail", action="store_true", help="also show per-market rows")
    ev = sub.add_parser("evaluate", help="score the signals recorded while trading")
    ev.add_argument("--detail", action="store_true")
    s = sub.add_parser("status", help="journal summary and recent trades")
    s.add_argument("--limit", type=int, default=20)
    f = sub.add_parser("flatten", help="close all open positions")
    f.add_argument("--confirm-live", action="store_true")

    args = p.parse_args(argv)
    cfg = load_config(args.config)
    _setup_logging(cfg.state_dir, args.verbose)
    {"run": cmd_run, "analyze": cmd_analyze, "news": cmd_news, "backtest": cmd_backtest,
     "research": cmd_research, "evaluate": cmd_evaluate, "status": cmd_status,
     "flatten": cmd_flatten}[args.cmd](cfg, args)


if __name__ == "__main__":
    main()
