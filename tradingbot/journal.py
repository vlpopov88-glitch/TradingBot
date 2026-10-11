"""SQLite journal of decisions, trades and equity, for review and auditing."""
from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timezone

from .models import Decision, Trade

SCHEMA = """
CREATE TABLE IF NOT EXISTS trades (
    id INTEGER PRIMARY KEY, instrument TEXT, side TEXT, qty REAL, entry REAL, exit REAL,
    pnl_usd REAL, opened_at TEXT, closed_at TEXT, reason TEXT);
CREATE TABLE IF NOT EXISTS decisions (
    id INTEGER PRIMARY KEY, ts TEXT, instrument TEXT, score REAL, action TEXT, signals TEXT);
CREATE TABLE IF NOT EXISTS equity (ts TEXT PRIMARY KEY, equity REAL, open_positions INTEGER);
"""


class Journal:
    def __init__(self, path: str):
        self.db = sqlite3.connect(path)
        self.db.executescript(SCHEMA)

    def trade(self, t: Trade) -> None:
        self.db.execute(
            "INSERT INTO trades (instrument, side, qty, entry, exit, pnl_usd, opened_at, closed_at, reason)"
            " VALUES (?,?,?,?,?,?,?,?,?)",
            (t.instrument, t.side.value, t.qty, t.entry_price, t.exit_price, t.pnl_usd,
             t.opened_at.isoformat(), t.closed_at.isoformat(), t.reason))
        self.db.commit()

    def decision(self, d: Decision) -> None:
        sigs = [{"source": s.source, "score": round(s.score, 3), "conf": round(s.confidence, 2),
                 "reason": s.reason} for s in d.signals]
        self.db.execute("INSERT INTO decisions (ts, instrument, score, action, signals) VALUES (?,?,?,?,?)",
                        (datetime.now(timezone.utc).isoformat(), d.instrument, d.score, d.action,
                         json.dumps(sigs)))
        self.db.commit()

    def equity(self, value: float, n_positions: int) -> None:
        self.db.execute("INSERT OR REPLACE INTO equity VALUES (?,?,?)",
                        (datetime.now(timezone.utc).isoformat(), value, n_positions))
        self.db.commit()

    def recent_trades(self, n: int = 20) -> list[tuple]:
        return self.db.execute(
            "SELECT closed_at, instrument, side, qty, entry, exit, pnl_usd, reason FROM trades"
            " ORDER BY id DESC LIMIT ?", (n,)).fetchall()

    def summary(self) -> dict:
        row = self.db.execute(
            "SELECT COUNT(*), COALESCE(SUM(pnl_usd),0), COALESCE(SUM(pnl_usd > 0),0) FROM trades").fetchone()
        return {"trades": row[0], "pnl_usd": row[1], "win_rate": (row[2] / row[0]) if row[0] else 0.0}
