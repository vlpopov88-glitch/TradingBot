"""Phone alerts via Telegram: trades, halts, a daily summary and repeated errors.

Setup (2 minutes): message @BotFather on Telegram -> /newbot -> copy the token.
Send your new bot any message, then open
https://api.telegram.org/bot<TOKEN>/getUpdates and copy "chat":{"id": ...}.
Put both in the environment (or .env for Docker):
    TELEGRAM_BOT_TOKEN=...   TELEGRAM_CHAT_ID=...
Without them notifications are simply off.
"""
from __future__ import annotations

import logging
import os

from .http import session

log = logging.getLogger(__name__)


class Notifier:
    def __init__(self, token: str | None = None, chat_id: str | None = None, prefix: str = ""):
        self.token = token if token is not None else os.getenv("TELEGRAM_BOT_TOKEN", "")
        self.chat_id = chat_id if chat_id is not None else os.getenv("TELEGRAM_CHAT_ID", "")
        self.prefix = prefix
        self.sent: list[str] = []          # kept for tests / inspection

    @property
    def enabled(self) -> bool:
        return bool(self.token and self.chat_id)

    def send(self, text: str) -> bool:
        text = f"{self.prefix}{text}"
        self.sent.append(text)
        if not self.enabled:
            return False
        try:
            r = session().post(f"https://api.telegram.org/bot{self.token}/sendMessage",
                               json={"chat_id": self.chat_id, "text": text[:4000],
                                     "disable_web_page_preview": True}, timeout=10)
            if r.status_code != 200:
                log.warning("telegram send failed: HTTP %s", r.status_code)
                return False
            return True
        except Exception as exc:
            # Never log str(exc): request errors embed the URL, which contains the bot token.
            log.warning("telegram send failed: %s", type(exc).__name__)
            return False
