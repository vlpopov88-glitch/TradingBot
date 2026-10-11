"""Watching other traders: what are big speculators and the crowd doing?

* Gold / forex -> CFTC Commitments of Traders (weekly). We take large speculators'
  net position as a % of open interest, rank it against the last ~3 years (the
  "COT index"), and combine *momentum* (specs adding in the direction of the move)
  with *contrarian fade at extremes* (crowded trades tend to reverse).
* Crypto -> OKX derivatives data (hourly): top traders' long/short ratio (follow
  the "smart money"), all-accounts long/short ratio (fade the retail crowd) and
  funding rate (extreme funding = over-leveraged side, fade it).
"""
from __future__ import annotations

import logging
import math
import time

from .http import get_json
from .models import AssetClass, Instrument, Signal

log = logging.getLogger(__name__)

COT_URL = "https://publicreporting.cftc.gov/resource/6dca-aqww.json"
OKX = "https://www.okx.com/api/v5"


def cot_index(nets: list[float]) -> float:
    """Where the latest value sits in its historical range, 0..1."""
    lo, hi = min(nets), max(nets)
    return 0.5 if hi == lo else (nets[0] - lo) / (hi - lo)


def cot_score(rows: list[dict], invert: bool = False) -> Signal:
    """rows: newest first, CFTC legacy futures-only records."""
    nets = []
    for r in rows:
        oi = float(r["open_interest_all"]) or 1.0
        nets.append((float(r["noncomm_positions_long_all"]) - float(r["noncomm_positions_short_all"])) / oi)
    if invert:  # e.g. USD/JPY trades the inverse of the JPY futures contract
        nets = [-n for n in nets]
    if len(nets) < 10:
        return Signal("positioning", 0.0, 0.0, "not enough COT history")
    idx = cot_index(nets)
    momentum = math.tanh((nets[0] - nets[min(4, len(nets) - 1)]) * 10)  # ~1 month change
    # Extremes (>90th or <10th pct) are crowded -> contrarian; the middle -> follow flows.
    if idx > 0.9:
        score, why = -(idx - 0.9) * 10 * 0.6, "specs extremely long (crowded)"
    elif idx < 0.1:
        score, why = (0.1 - idx) * 10 * 0.6, "specs extremely short (crowded)"
    else:
        score, why = 0.6 * momentum, "following speculator flows"
    return Signal("positioning", score, 0.6,
                  f"COT index {idx:.0%}, net {nets[0]:+.1%} of OI, 4wk chg {nets[0] - nets[min(4, len(nets) - 1)]:+.1%}; {why}")


def crypto_score(top_ratio: list[float], crowd_ratio: list[float], funding: float) -> Signal:
    """Ratios newest first (long/short). funding per 8h, e.g. 0.0001 = 0.01%."""
    parts, reasons = [], []
    if top_ratio:
        # Follow top traders: ratio > 1 means more longs. Also look at the trend.
        lvl = math.tanh(math.log(top_ratio[0]) * 4)
        trend = math.tanh(math.log(top_ratio[0] / top_ratio[-1]) * 8) if len(top_ratio) > 1 else 0
        parts.append(0.5 * lvl + 0.5 * trend)
        reasons.append(f"top-trader L/S {top_ratio[0]:.2f}")
    if crowd_ratio:
        # Fade the crowd when it is heavily one-sided.
        parts.append(-math.tanh(math.log(crowd_ratio[0]) * 2))
        reasons.append(f"crowd L/S {crowd_ratio[0]:.2f}")
    if funding:
        # 0.01%/8h is neutral-ish; well above that = overheated longs.
        parts.append(-math.tanh((funding - 0.0001) / 0.0003))
        reasons.append(f"funding {funding * 100:.4f}%")
    if not parts:
        return Signal("positioning", 0.0, 0.0, "no positioning data")
    return Signal("positioning", sum(parts) / len(parts), 0.5, ", ".join(reasons))


class PositioningAnalyzer:
    def __init__(self, cot_weeks: int = 156, cot_refresh_s: int = 6 * 3600, crypto_refresh_s: int = 900):
        self.cot_weeks = cot_weeks
        self._ttl = {"cot": cot_refresh_s, "crypto": crypto_refresh_s}
        self._cache: dict[str, tuple[float, Signal]] = {}

    def _cached(self, key: str, kind: str, fn) -> Signal:
        hit = self._cache.get(key)
        if hit and time.time() - hit[0] < self._ttl[kind]:
            return hit[1]
        try:
            sig = fn()
        except Exception as exc:
            log.warning("positioning fetch failed for %s: %s", key, exc)
            sig = hit[1] if hit else Signal("positioning", 0.0, 0.0, f"unavailable: {exc}")
        self._cache[key] = (time.time(), sig)
        return sig

    def signal(self, inst: Instrument) -> Signal:
        if inst.asset_class is AssetClass.CRYPTO and inst.okx_swap:
            return self._cached(inst.name, "crypto", lambda: self._crypto(inst.okx_swap))
        if inst.cot_code:
            # COT futures are quoted vs USD (EUR, GBP, JPY...). For USD/XXX pairs invert.
            invert = inst.base == "USD"
            return self._cached(inst.name, "cot", lambda: self._cot(inst.cot_code, invert))
        return Signal("positioning", 0.0, 0.0, "no positioning source")

    def _cot(self, code: str, invert: bool) -> Signal:
        rows = get_json(COT_URL, params={
            "cftc_contract_market_code": code,
            "$order": "report_date_as_yyyy_mm_dd DESC",
            "$limit": self.cot_weeks,
        })
        return cot_score(rows, invert)

    def _crypto(self, swap: str) -> Signal:
        ccy = swap.split("-")[0]
        top = get_json(f"{OKX}/rubik/stat/contracts/long-short-position-ratio-contract-top-trader",
                       params={"instId": swap, "period": "1H", "limit": 24})["data"]
        crowd = get_json(f"{OKX}/rubik/stat/contracts/long-short-account-ratio",
                         params={"ccy": ccy, "period": "1H"})["data"][:24]
        fund = get_json(f"{OKX}/public/funding-rate", params={"instId": swap})["data"]
        return crypto_score(
            [float(r[1]) for r in top if float(r[1]) > 0],
            [float(r[1]) for r in crowd if float(r[1]) > 0],
            float(fund[0]["fundingRate"]) if fund else 0.0,
        )
