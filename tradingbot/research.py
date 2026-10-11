"""Does each signal actually predict prices? Measure it instead of assuming it.

Every signal's score at decision time is lined up with the return that followed
over a few horizons. Per signal and horizon the report shows:

  n      decisions with a non-zero score
  IC     rank correlation of score vs. forward return. 0 = useless. In liquid
         markets a genuine edge is usually small: |IC| of 0.02-0.05.
  t      t-statistic of IC, with overlapping horizons counted only once.
         |t| below ~2 is indistinguishable from luck.
  hit    share of decisions whose direction was right
  edge   average forward return from following the signal's sign, in basis
         points, before trading costs
  1st/2nd  IC in the first and second half of the sample. A real effect should
         keep its sign in both halves; one that flips is likely noise.

Sources with history: technical strategies (Yahoo, years), CFTC COT positioning
(weekly, 10+ years), OKX crypto funding (~3 months) and trader ratios (~3-6 months).
News has no free history, so it is measured from the bot's own journal instead
(`tradingbot evaluate`, once paper trading has collected a few weeks of data).
"""
from __future__ import annotations

import logging
import math
from dataclasses import dataclass

import numpy as np
import pandas as pd

from .config import Config
from .data.market import YahooData, get_provider
from .engine import TF
from .http import get_json
from .models import AssetClass, Instrument
from .positioning import COT_URL, OKX, cot_score
from .strategies import Ensemble, default_strategies

log = logging.getLogger(__name__)

# Yahoo symbols with long history for each market (crypto too, so no OKX paging).
YAHOO = {"XAU/USD": "GC=F", "EUR/USD": "EURUSD=X", "GBP/USD": "GBPUSD=X", "USD/JPY": "JPY=X",
         "BTC/USD": "BTC-USD", "ETH/USD": "ETH-USD"}


@dataclass
class Stat:
    source: str
    horizon: str
    n: int
    n_eff: float
    ic: float
    hit: float
    edge_bps: float
    ic_first: float
    ic_second: float
    market: str = ""

    @property
    def t(self) -> float:
        if self.n_eff <= 3 or np.isnan(self.ic):
            return float("nan")
        return self.ic * math.sqrt(self.n_eff - 2) / math.sqrt(max(1 - self.ic ** 2, 1e-9))


def _rank_ic(a: pd.Series, b: pd.Series) -> float:
    if len(a) < 10 or a.nunique() < 2 or b.nunique() < 2:   # binary (+/-) signals are fine
        return float("nan")
    return float(a.rank().corr(b.rank()))


def forward_returns(decisions: pd.DatetimeIndex, close_at_end: pd.Series, horizon: pd.Timedelta,
                    ref: pd.Series | None = None) -> pd.Series:
    """Return from each decision time to decision time + horizon.

    ``close_at_end`` is indexed by bar *end* time, so ``asof(t)`` is the last price
    that was actually known at time t (no lookahead). ``ref`` overrides the entry price.
    """
    p0 = close_at_end.asof(decisions).to_numpy() if ref is None else np.asarray(ref, dtype=float)
    # Exit at the first price at or after decision + horizon (Friday + 1 day -> Monday's
    # close, not Friday's again), NaN where that future hasn't happened yet.
    pos = close_at_end.index.searchsorted(decisions + horizon, side="left")
    closes = np.append(close_at_end.to_numpy(dtype=float), np.nan)
    return pd.Series(closes[pos] / p0 - 1, index=decisions)


def measure(source: str, label: str, score: pd.Series, fwd: pd.Series, horizon: pd.Timedelta) -> Stat:
    m = score.notna() & fwd.notna() & (score != 0)
    s, r = score[m], fwd[m]
    n = len(s)
    if n < 10:
        return Stat(source, label, n, 0, float("nan"), float("nan"), float("nan"), float("nan"), float("nan"))
    spacing = pd.Series(s.index).diff().median()
    n_eff = n * min(1.0, spacing / horizon) if pd.notna(spacing) and spacing > pd.Timedelta(0) else n
    half = n // 2
    moved = r != 0
    return Stat(source, label, n, n_eff, _rank_ic(s, r),
                float((np.sign(s[moved]) == np.sign(r[moved])).mean()), float((np.sign(s) * r).mean() * 1e4),
                _rank_ic(s.iloc[:half], r.iloc[:half]), _rank_ic(s.iloc[half:], r.iloc[half:]))


def evaluate_frame(signals: pd.DataFrame, close_at_end: pd.Series, horizons: dict[str, pd.Timedelta],
                   ref: pd.Series | None = None, market: str = "") -> list[Stat]:
    """``signals``: one column per source, indexed by decision time."""
    out = []
    for label, h in horizons.items():
        fwd = forward_returns(signals.index, close_at_end, h, ref)
        for col in signals.columns:
            st = measure(col, label, signals[col], fwd, h)
            st.market = market
            out.append(st)
    return out


def pool(stats: list[Stat]) -> list[Stat]:
    """Combine per-market stats into one row per (source, horizon), weighted by n_eff."""
    groups: dict[tuple[str, str], list[Stat]] = {}
    for s in stats:
        if s.n >= 10 and not np.isnan(s.ic):
            groups.setdefault((s.source, s.horizon), []).append(s)
    out = []
    for (src, hz), ss in groups.items():
        w = np.array([s.n_eff for s in ss])

        def avg(attr):
            vals = np.array([getattr(s, attr) for s in ss], dtype=float)
            ok = ~np.isnan(vals)
            return float(np.average(vals[ok], weights=w[ok])) if ok.any() else float("nan")

        out.append(Stat(src, hz, sum(s.n for s in ss), float(w.sum()), avg("ic"), avg("hit"),
                        avg("edge_bps"), avg("ic_first"), avg("ic_second")))
    return out


def verdict(s: Stat) -> str:
    if s.n < 30 or np.isnan(s.t):
        return "too little data"
    consistent = (np.sign(s.ic_first) == np.sign(s.ic_second) == np.sign(s.ic))
    if abs(s.t) >= 2 and consistent:
        return "PREDICTIVE" if s.ic > 0 else "PREDICTIVE (inverted)"
    if abs(s.t) >= 2:
        return "unstable"
    return "no edge"


def format_table(title: str, stats: list[Stat]) -> str:
    detail = any(s.market for s in stats)
    mk = f"{'market':<9}" if detail else ""
    lines = [f"\n{title}",
             f"  {mk}{'signal':<18}{'horizon':>8}{'n':>7}{'IC':>8}{'t':>7}{'hit':>7}{'edge':>8}"
             f"{'1st':>8}{'2nd':>8}  verdict"]
    def hz(label):
        try:
            return pd.Timedelta(label)
        except ValueError:
            return pd.Timedelta(0)

    for s in sorted(stats, key=lambda x: (x.source, x.market, hz(x.horizon))):
        m = f"{s.market:<9}" if detail else ""
        lines.append(f"  {m}{s.source:<18}{s.horizon:>8}{s.n:>7}{s.ic:>+8.3f}{s.t:>+7.1f}{s.hit:>7.0%}"
                     f"{s.edge_bps:>+7.1f}b{s.ic_first:>+8.3f}{s.ic_second:>+8.3f}  {verdict(s)}")
    return "\n".join(lines)


# ---- historical signal generators ----------------------------------------
def close_at_end(df: pd.DataFrame, bar: pd.Timedelta) -> pd.Series:
    s = df["close"].copy()
    s.index = s.index + bar
    return s[~s.index.duplicated(keep="last")].sort_index()


def technical_signals(df: pd.DataFrame, bar: pd.Timedelta, ens_cfg=None) -> pd.DataFrame:
    """Every strategy's score at each bar close (decision time = bar end), plus
    ``technical``: the blend of the strategies enabled in the config."""
    ens = Ensemble(default_strategies())
    tf = ens.technical_frame(df).drop(columns="technical")
    tf["technical"] = Ensemble(config=ens_cfg).technical_frame(df)["technical"]
    tf = tf.iloc[ens.min_bars:]
    tf.index = tf.index + bar
    return tf


def cot_signals(code: str, invert: bool, lookback: int = 156, weeks: int = 520) -> pd.Series:
    """The live COT score as it would have read each week, indexed when it was usable.

    Positions are as of Tuesday and published Friday 15:30 ET, so the score is
    stamped Saturday 12:00 UTC: the entry price is Friday's close, after release.
    """
    rows = get_json(COT_URL, params={"cftc_contract_market_code": code,
                                     "$order": "report_date_as_yyyy_mm_dd DESC", "$limit": lookback + weeks})
    out = {}
    for k in range(len(rows) - lookback):
        sig = cot_score(rows[k:k + lookback], invert)
        tuesday = pd.Timestamp(rows[k]["report_date_as_yyyy_mm_dd"]).tz_localize("UTC")
        out[tuesday + pd.Timedelta(days=4, hours=12)] = sig.score
    return pd.Series(out, dtype=float).sort_index()


def _okx_rows(path: str, params: dict, pages: int = 10) -> list:
    rows, after = [], None
    for _ in range(pages):
        p = dict(params)
        if after:
            p["after"] = after
        data = get_json(f"{OKX}/{path}", params=p)["data"]
        if not data:
            break
        rows.extend(data)
        after = data[-1]["fundingTime"] if isinstance(data[-1], dict) else data[-1][0]
        if len(data) < int(params.get("limit", 100)):
            break
    return rows


def crypto_positioning_signals(swap: str) -> tuple[pd.Series, pd.DataFrame]:
    """(funding score every 8h, daily trader-ratio scores) with the live formulas."""
    fund = _okx_rows("public/funding-rate-history", {"instId": swap, "limit": 100})
    f = pd.Series({pd.Timestamp(int(r["fundingTime"]), unit="ms", tz="UTC"): float(r["fundingRate"])
                   for r in fund}).sort_index()
    funding_score = -np.tanh((f - 0.0001) / 0.0003)

    ccy = swap.split("-")[0]
    top = get_json(f"{OKX}/rubik/stat/contracts/long-short-position-ratio-contract-top-trader",
                   params={"instId": swap, "period": "1D", "limit": 100})["data"]
    crowd = get_json(f"{OKX}/rubik/stat/contracts/long-short-account-ratio",
                     params={"ccy": ccy, "period": "1D"})["data"]
    to_s = lambda rows: pd.Series({pd.Timestamp(int(r[0]), unit="ms", tz="UTC"): float(r[1])
                                   for r in rows if float(r[1]) > 0}).sort_index()
    top_s, crowd_s = to_s(top), to_s(crowd)
    # Each daily ratio is stamped at the start of its day; it is known at the end.
    day = pd.Timedelta(days=1)
    daily = pd.DataFrame({
        "top_traders": 0.5 * np.tanh(np.log(top_s) * 4) + 0.5 * np.tanh(np.log(top_s / top_s.shift()) * 8),
        "crowd_fade": -np.tanh(np.log(crowd_s) * 2),
    })
    daily.index = daily.index + day
    daily["funding"] = funding_score.asof(daily.index).to_numpy()
    daily["positioning"] = daily[["top_traders", "crowd_fade", "funding"]].mean(axis=1, skipna=False)
    return funding_score.rename("funding"), daily


# ---- reports ---------------------------------------------------------------
def _hz(*hours: float) -> dict[str, pd.Timedelta]:
    def label(h):
        return f"{h:g}h" if h < 24 else f"{h / 24:g}d"
    return {label(h): pd.Timedelta(hours=h) for h in hours}


def _bar_horizons(bar: pd.Timedelta, counts=(1, 4, 24)) -> dict[str, pd.Timedelta]:
    return _hz(*(k * bar / pd.Timedelta(hours=1) for k in counts))


def research_technical(instruments: list[Instrument], timeframe: str, ens_cfg=None) -> tuple[list[Stat], list[Stat]]:
    bars, horizons = {
        "1h": (12_000, _hz(1, 4, 24)),
        "4h": (3_000, _hz(4, 24, 72)),
        "1d": (2_520, _hz(24, 120, 480)),
    }[timeframe]
    bar = pd.Timedelta(TF[timeframe])
    all_stats = []
    for inst in instruments:
        df = YahooData().candles(YAHOO.get(inst.name, inst.data_symbol), timeframe, bars)
        sig = technical_signals(df, bar, ens_cfg)
        stats = evaluate_frame(sig, close_at_end(df, bar), horizons, market=inst.name)
        log.info("technical %s %s: %d bars %s..%s", inst.name, timeframe, len(df),
                 df.index[0].date(), df.index[-1].date())
        all_stats.extend(stats)
    return all_stats, pool(all_stats)


def research_cot(instruments: list[Instrument]) -> tuple[list[Stat], list[Stat]]:
    all_stats = []
    for inst in instruments:
        if not inst.cot_code:
            continue
        sig = cot_signals(inst.cot_code, invert=inst.base == "USD")
        daily = YahooData().candles(YAHOO.get(inst.name, inst.data_symbol), "1d", 2_600)
        stats = evaluate_frame(sig.to_frame("positioning (COT)"), close_at_end(daily, pd.Timedelta(days=1)),
                               _hz(24 * 7, 24 * 14, 24 * 28), market=inst.name)
        all_stats.extend(stats)
        log.info("COT %s: %d weekly reports %s..%s", inst.name, len(sig), sig.index[0].date(), sig.index[-1].date())
    return all_stats, pool(all_stats)


def research_crypto(instruments: list[Instrument]) -> tuple[list[Stat], list[Stat]]:
    all_stats = []
    for inst in instruments:
        if inst.asset_class is not AssetClass.CRYPTO or not inst.okx_swap:
            continue
        funding, daily = crypto_positioning_signals(inst.okx_swap)
        hourly = YahooData().candles(YAHOO.get(inst.name, inst.data_symbol), "1h", 6_000)
        px = close_at_end(hourly, pd.Timedelta(hours=1))
        stats = evaluate_frame(funding.to_frame(), px, _hz(8, 24, 72), market=inst.name)
        stats += evaluate_frame(daily, px, _hz(24, 72, 168), market=inst.name)
        all_stats.extend(stats)
    return all_stats, pool(all_stats)


def evaluate_journal(snapshots: pd.DataFrame, cfg: Config) -> tuple[list[Stat], list[Stat]]:
    """Score the signals the bot actually saw live (including news) against what happened next."""
    if snapshots.empty:
        return [], []
    bar = pd.Timedelta(TF[cfg.timeframe])
    horizons = _bar_horizons(bar, (1, 3, 10) if bar >= pd.Timedelta(days=1) else (1, 4, 24))
    all_stats = []
    for name, g in snapshots.groupby("instrument"):
        inst = cfg.by_name.get(name)
        if inst is None:
            continue
        g = g.assign(ts=pd.to_datetime(g["ts"], utc=True))
        wide = g.pivot_table(index="ts", columns="source", values="score", aggfunc="last").sort_index()
        ref = g.groupby("ts")["price"].last().reindex(wide.index)
        span = pd.Timestamp.now(tz="UTC") - wide.index[0]
        need = int(span / bar) + 50
        df = get_provider(inst.data_source).candles(inst.data_symbol, cfg.timeframe, max(need, 100))
        stats = evaluate_frame(wide, close_at_end(df, bar), horizons, ref=ref, market=name)
        all_stats.extend(stats)
    return all_stats, pool(all_stats)
