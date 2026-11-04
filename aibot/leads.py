"""Lead validation and storage. Everything here treats model output as untrusted input."""

from __future__ import annotations

import re
import sqlite3
import threading
from dataclasses import dataclass
from datetime import datetime

NAME_MAX, REQUEST_MAX = 60, 500


class LeadError(ValueError):
    pass


@dataclass(frozen=True)
class Lead:
    user_id: int
    username: str
    name: str
    phone: str
    request: str
    created_at: datetime


def clean_text(raw: object, limit: int) -> str:
    """Single line, no control characters, cut to `limit`."""
    text = re.sub(r"[\x00-\x1f\x7f]", " ", str(raw or ""))
    return " ".join(text.split())[:limit]


def normalize_phone(raw: object) -> str:
    text = str(raw or "").strip()
    if not re.fullmatch(r"\+?[\d\s\-().]{7,25}", text):
        raise LeadError("bad_phone")
    digits = re.sub(r"\D", "", text)
    if len(digits) == 10 and digits.startswith("0"):
        digits = "38" + digits
    if not 10 <= len(digits) <= 15:
        raise LeadError("bad_phone")
    return "+" + digits


def make_lead(data: dict, *, user_id: int, username: str, now: datetime) -> Lead:
    name = clean_text(data.get("name"), NAME_MAX)
    request = clean_text(data.get("request"), REQUEST_MAX)
    if len(name) < 2:
        raise LeadError("bad_name")
    if len(request) < 3:
        raise LeadError("bad_request")
    return Lead(user_id, clean_text(username, 40), name, normalize_phone(data.get("phone")), request, now)


class LeadStore:
    def __init__(self, path: str = ":memory:"):
        self._conn = sqlite3.connect(path, check_same_thread=False, isolation_level=None)
        self._lock = threading.Lock()
        with self._lock:
            self._conn.execute(
                "CREATE TABLE IF NOT EXISTS leads (id INTEGER PRIMARY KEY AUTOINCREMENT, user_id INTEGER, "
                "username TEXT, name TEXT, phone TEXT, request TEXT, created_at TEXT)"
            )

    def add(self, lead: Lead) -> int:
        with self._lock:
            cur = self._conn.execute(
                "INSERT INTO leads (user_id, username, name, phone, request, created_at) VALUES (?, ?, ?, ?, ?, ?)",
                (lead.user_id, lead.username, lead.name, lead.phone, lead.request, lead.created_at.isoformat()),
            )
        return cur.lastrowid

    def last_for_user(self, user_id: int) -> datetime | None:
        with self._lock:
            row = self._conn.execute(
                "SELECT created_at FROM leads WHERE user_id = ? ORDER BY id DESC LIMIT 1", (user_id,)
            ).fetchone()
        return datetime.fromisoformat(row[0]) if row else None

    def count(self) -> int:
        with self._lock:
            return self._conn.execute("SELECT COUNT(*) FROM leads").fetchone()[0]

    def close(self) -> None:
        self._conn.close()
