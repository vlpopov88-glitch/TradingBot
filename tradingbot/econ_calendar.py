"""Economic calendar: avoid opening trades right around high-impact releases.

NFP, CPI, FOMC and the like can move gold and FX by several ATRs in seconds, gap
through stops and widen spreads. The bot blocks *new* entries in a window around
each high-impact event for the affected currencies; open positions keep their stops.
Source: the free Forex Factory weekly JSON feed.
"""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone

from .http import get_json
from .models import AssetClass, Instrument

log = logging.getLogger(__name__)

FEED = "https://nfs.faireconomy.media/ff_calendar_thisweek.json"


@dataclass(frozen=True)
class Event:
    title: str
    currency: str
    time: datetime
    impact: str


def parse_events(rows: list[dict]) -> list[Event]:
    out = []
    for r in rows:
        try:
            t = datetime.fromisoformat(r["date"])
        except (KeyError, ValueError):
            continue
        if t.tzinfo is None:
            t = t.replace(tzinfo=timezone.utc)
        out.append(Event(r.get("title", ""), r.get("country", "").upper(), t, r.get("impact", "")))
    return out


def affected_currencies(inst: Instrument) -> set[str]:
    """Which currencies' news can move this instrument."""
    if inst.asset_class is AssetClass.FOREX:
        return {inst.base, inst.quote}
    # Gold and crypto are priced in, and driven by, the US dollar.
    return {"USD"}


class EconomicCalendar:
    def __init__(self, minutes_before: int = 30, minutes_after: int = 30,
                 impacts: tuple[str, ...] = ("High",), refresh_seconds: int = 3600):
        self.before = timedelta(minutes=minutes_before)
        self.after = timedelta(minutes=minutes_after)
        self.impacts = set(impacts)
        self.refresh = refresh_seconds
        self.events: list[Event] = []
        self._last_fetch = 0.0

    def update(self, force: bool = False) -> None:
        if not force and time.time() - self._last_fetch < self.refresh:
            return
        self._last_fetch = time.time()
        try:
            self.events = [e for e in parse_events(get_json(FEED)) if e.impact in self.impacts]
            log.info("calendar: %d high-impact events this week", len(self.events))
        except Exception as exc:  # keep the last good copy
            log.warning("economic calendar fetch failed: %s", exc)

    def blackout(self, inst: Instrument, now: datetime | None = None) -> Event | None:
        """The event that currently blocks trading this instrument, if any."""
        now = now or datetime.now(timezone.utc)
        ccys = affected_currencies(inst)
        for e in self.events:
            if e.currency in ccys and e.time - self.before <= now <= e.time + self.after:
                return e
        return None

    def upcoming(self, inst: Instrument, now: datetime | None = None, hours: float = 24) -> list[Event]:
        now = now or datetime.now(timezone.utc)
        ccys = affected_currencies(inst)
        return sorted((e for e in self.events
                       if e.currency in ccys and now <= e.time <= now + timedelta(hours=hours)),
                      key=lambda e: e.time)
