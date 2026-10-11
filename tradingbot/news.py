"""News monitoring: pull RSS headlines and turn them into per-instrument sentiment.

Each headline is matched to the assets/currencies it talks about and scored with a
finance-specific lexicon. For a pair BASE/QUOTE the instrument score is
``sentiment(BASE) - sentiment(QUOTE)``; e.g. a hawkish Fed headline is bullish USD,
which is bearish EUR/USD and bearish XAU/USD. Older headlines decay exponentially.
"""
from __future__ import annotations

import logging
import math
import re
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime

from defusedxml import ElementTree as ET

from .http import session, TIMEOUT
from .models import Instrument, Signal

log = logging.getLogger(__name__)

DEFAULT_FEEDS = [
    "https://www.investing.com/rss/news_1.rss",        # forex
    "https://www.investing.com/rss/news_11.rss",       # commodities
    "https://www.investing.com/rss/news_301.rss",      # crypto
    "https://cointelegraph.com/rss",
    "https://feeds.content.dowjones.io/public/rss/mw_topstories",
    "https://news.google.com/rss/search?q=gold+price+OR+forex+OR+bitcoin&hl=en-US&gl=US&ceid=US:en",
]

# Which words point a headline at which asset/currency.
ENTITIES: dict[str, tuple[str, ...]] = {
    "XAU": ("gold", "bullion", "xau", "precious metal"),
    "XAG": ("silver", "xag"),
    "BTC": ("bitcoin", "btc"),
    "ETH": ("ethereum", "ether ", "eth "),
    "SOL": ("solana",),
    "CRYPTO": ("crypto", "cryptocurrency", "digital asset", "stablecoin", "altcoin"),
    "USD": ("dollar", "greenback", "fed ", "federal reserve", "fomc", "powell", "u.s. economy",
            "nonfarm", "payrolls", "us cpi", "u.s. inflation", "treasury yields", "dxy"),
    "EUR": ("euro ", "euro,", "ecb", "lagarde", "eurozone", "euro zone", "euro area"),
    "GBP": ("sterling", "pound", "boe", "bank of england", "bailey", "uk economy", "gilts"),
    "JPY": ("yen", "boj", "bank of japan", "ueda", "japan's economy"),
    "CHF": ("swiss franc", "snb"),
    "AUD": ("aussie", "australian dollar", "rba"),
    "CAD": ("loonie", "canadian dollar", "bank of canada"),
    "NZD": ("kiwi", "new zealand dollar", "rbnz"),
}

BULLISH = {
    "surge": 1.0, "soar": 1.0, "rally": 0.9, "jump": 0.8, "climb": 0.6, "rise": 0.5,
    "gain": 0.5, "higher": 0.4, "record high": 1.0, "all-time high": 1.0,
    "rebound": 0.6, "recover": 0.5, "strong": 0.5, "strengthen": 0.6, "bullish": 0.9,
    "upbeat": 0.5, "beat": 0.5, "outperform": 0.6,
    "inflow": 0.6, "approval": 0.7, "approve": 0.6, "adoption": 0.5,
    "buy": 0.3, "upgrade": 0.6, "boost": 0.5, "optimism": 0.5,
}
BEARISH = {
    "plunge": 1.0, "crash": 1.0, "tumble": 0.9, "slump": 0.9, "sink": 0.7, "drop": 0.6,
    "fall": 0.5, "slide": 0.6, "decline": 0.5, "lower": 0.4, "weak": 0.5,
    "weaken": 0.6, "bearish": 0.9, "miss": 0.5, "selloff": 0.8, "sell-off": 0.8,
    "outflow": 0.6, "hack": 0.9, "exploit": 0.8, "ban": 0.7, "lawsuit": 0.5,
    "crackdown": 0.7, "downgrade": 0.6, "recession": 0.5, "slowdown": 0.4, "fears": 0.4,
    "liquidation": 0.6, "default": 0.6, "pressure": 0.3,
}
# Monetary-policy tone: hawkish lifts the currency; it weighs on gold and crypto.
HAWKISH = ("hawkish", "rate hike", "hikes rates", "raise rates", "tightening", "hot inflation",
           "inflation accelerates", "higher for longer")
DOVISH = ("dovish", "rate cut", "cuts rates", "cut rates", "easing", "inflation cools",
          "inflation slows", "stimulus", "qe")
SAFE_HAVEN = ("war", "conflict", "geopolitical", "missile", "invasion", "tariff", "crisis",
              "uncertainty", "safe haven", "safe-haven")
NEGATIONS = ("not ", "no ", "fails to", "despite", "unlikely")


@dataclass
class Headline:
    title: str
    published: datetime
    source: str


def _parse_date(text: str | None) -> datetime:
    dt = None
    if text:
        text = text.strip()
        try:
            dt = parsedate_to_datetime(text)
        except (TypeError, ValueError):
            try:
                dt = datetime.fromisoformat(text.replace("Z", "+00:00"))
            except ValueError:
                pass
    if dt is None:
        return datetime.now(timezone.utc)
    return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)


def parse_feed(xml: bytes, source: str) -> list[Headline]:
    root = ET.fromstring(xml)
    out = []
    # RSS <item> and Atom <entry>
    for item in root.iter():
        tag = item.tag.rsplit("}", 1)[-1]
        if tag not in ("item", "entry"):
            continue
        fields = {c.tag.rsplit("}", 1)[-1]: (c.text or "") for c in item}
        title = fields.get("title", "").strip()
        if title:
            date = fields.get("pubDate") or fields.get("published") or fields.get("updated")
            out.append(Headline(title, _parse_date(date), source))
    return out


def _contains(text: str, phrase: str, inflect: bool = False) -> bool:
    """Whole-word match; with ``inflect`` also matches surges/surged/surging."""
    suffix = r"(?:s|es|d|ed|ing)?" if inflect else ""
    pattern = r"(?<![a-z])" + re.escape(phrase.strip()) + suffix + r"(?![a-z])"
    return re.search(pattern, text) is not None


def score_headline(title: str) -> dict[str, float]:
    """Return {entity: sentiment in [-1, 1]} for one headline."""
    text = f" {title.lower()} "
    entities = [e for e, words in ENTITIES.items() if any(_contains(text, w) for w in words)]
    if not entities:
        return {}

    polarity = sum(w for k, w in BULLISH.items() if _contains(text, k, inflect=True))
    polarity -= sum(w for k, w in BEARISH.items() if _contains(text, k, inflect=True))
    if any(n in text for n in NEGATIONS):
        polarity *= -0.5
    hawk = sum(_contains(text, k) for k in HAWKISH) - sum(_contains(text, k) for k in DOVISH)
    haven = any(_contains(text, k) for k in SAFE_HAVEN)

    out: dict[str, float] = {}
    for e in entities:
        s = polarity
        if e in ("USD", "EUR", "GBP", "JPY", "CHF", "AUD", "CAD", "NZD"):
            s += 0.7 * hawk
        if e in ("XAU", "BTC", "ETH", "SOL", "CRYPTO") and "USD" in entities:
            # A Fed/USD story moves gold and crypto in the opposite direction to the dollar.
            s -= 0.6 * hawk
        if e in ("XAU", "XAG") and haven:
            s += 0.5
        if s:
            out[e] = math.tanh(s)
    # Pure USD stories still matter for gold/crypto: inverse dollar correlation.
    if "USD" in out and "XAU" not in out:
        out["XAU*"] = -0.5 * out["USD"]
    return out


class NewsAnalyzer:
    def __init__(self, feeds: list[str] | None = None, half_life_hours: float = 6.0,
                 refresh_seconds: int = 300, max_age_hours: float = 48.0) -> None:
        self.feeds = feeds or DEFAULT_FEEDS
        self.half_life = half_life_hours
        self.refresh = refresh_seconds
        self.max_age = max_age_hours
        self.headlines: dict[str, Headline] = {}
        self._last_fetch = 0.0

    def update(self, force: bool = False) -> int:
        if not force and time.time() - self._last_fetch < self.refresh:
            return 0
        self._last_fetch = time.time()
        new = 0
        for url in self.feeds:
            try:
                r = session().get(url, timeout=TIMEOUT)
                r.raise_for_status()
                for h in parse_feed(r.content, url):
                    if h.title not in self.headlines:
                        self.headlines[h.title] = h
                        new += 1
            except Exception as exc:  # one dead feed must not stop the bot
                log.warning("news feed failed %s: %s", url, exc)
        cutoff = datetime.now(timezone.utc).timestamp() - self.max_age * 3600
        self.headlines = {k: h for k, h in self.headlines.items() if h.published.timestamp() >= cutoff}
        log.info("news: %d new headlines, %d tracked", new, len(self.headlines))
        return new

    def _entity_scores(self, now: datetime) -> tuple[dict[str, float], dict[str, float], dict[str, list[str]]]:
        total: dict[str, float] = {}
        weight: dict[str, float] = {}
        examples: dict[str, list[str]] = {}
        for h in self.headlines.values():
            age_h = max(0.0, (now - h.published).total_seconds() / 3600)
            decay = 0.5 ** (age_h / self.half_life)
            for e, s in score_headline(h.title).items():
                total[e] = total.get(e, 0.0) + s * decay
                weight[e] = weight.get(e, 0.0) + decay
                if abs(s) > 0.3:
                    examples.setdefault(e, []).append(h.title)
        return total, weight, examples

    def signal(self, inst: Instrument, now: datetime | None = None) -> Signal:
        now = now or datetime.now(timezone.utc)
        total, weight, examples = self._entity_scores(now)

        def ent(code: str) -> tuple[float, float]:
            keys = [code]
            if code == "XAU":
                keys.append("XAU*")
            if code in ("BTC", "ETH", "SOL"):
                keys.append("CRYPTO")
            t = sum(total.get(k, 0.0) for k in keys)
            w = sum(weight.get(k, 0.0) for k in keys)
            return t, w

        bt, bw = ent(inst.base)
        qt, qw = ent(inst.quote) if inst.asset_class.value == "forex" else (0.0, 0.0)
        w = bw + qw
        if w == 0:
            return Signal("news", 0.0, 0.0, "no relevant headlines")
        # Average sentiment, then shrink toward zero when evidence is thin.
        raw = (bt - qt) / w
        confidence = min(1.0, w / 5.0)
        top = (examples.get(inst.base) or examples.get("XAU*" if inst.base == "XAU" else "") or [""])[:1]
        return Signal("news", raw * confidence, confidence,
                      f"{w:.1f} weighted headlines; e.g. {top[0][:80]!r}" if top[0] else f"{w:.1f} weighted headlines")
