"""Shared HTTP session with retries and a sane timeout."""
from __future__ import annotations

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

TIMEOUT = 15
_session: requests.Session | None = None


def session() -> requests.Session:
    global _session
    if _session is None:
        s = requests.Session()
        retry = Retry(total=3, backoff_factor=1.0, status_forcelist=(429, 500, 502, 503, 504))
        s.mount("https://", HTTPAdapter(max_retries=retry))
        s.headers["User-Agent"] = "Mozilla/5.0 (TradingBot; +https://github.com)"
        _session = s
    return _session


def get_json(url: str, **kwargs):
    r = session().get(url, timeout=TIMEOUT, **kwargs)
    r.raise_for_status()
    return r.json()
