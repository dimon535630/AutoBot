import hashlib
import re
import sqlite3
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

# Формат: FBOT-<7|14|30>-<B5>-<B8>-<B3>-<B9>-<checksum>
# B5/B8/B3/B9 — 4 цифры с проверкой кратности на 5/8/3/9 соответственно.
KEY_PATTERN = re.compile(r"^FBOT-(7|14|30)-(\d{4})-(\d{4})-(\d{4})-(\d{4})-([A-F0-9]{2})$")


@dataclass
class LicenseStatus:
    is_active: bool
    key_value: Optional[str]
    expires_at: Optional[datetime]

    @property
    def seconds_left(self) -> int:
        if not self.expires_at:
            return 0
        return max(0, int((self.expires_at - datetime.utcnow()).total_seconds()))


class LicenseManager:
    def __init__(self, db_path: str = "licenses.db"):
        self.db_path = Path(db_path)
        self._init_db()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_db(self) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS license_state (
                    id INTEGER PRIMARY KEY CHECK(id = 1),
                    active_key TEXT,
                    activated_at TEXT,
                    expires_at TEXT
                )
                """
            )
            conn.execute("INSERT OR IGNORE INTO license_state(id) VALUES (1)")
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS used_keys (
                    key_hash TEXT PRIMARY KEY,
                    used_at TEXT NOT NULL
                )
                """
            )

    def _checksum(self, base: str) -> str:
        return hashlib.sha256(base.encode("utf-8")).hexdigest().upper()[:2]

    def _key_hash(self, key_value: str) -> str:
        return hashlib.sha256(key_value.encode("utf-8")).hexdigest().upper()

    def validate_key_format(self, key_value: str) -> int:
        key_value = key_value.strip().upper()
        match = KEY_PATTERN.match(key_value)
        if not match:
            raise ValueError("Неверный формат ключа")

        duration = int(match.group(1))
        blocks = [int(match.group(2)), int(match.group(3)), int(match.group(4)), int(match.group(5))]
        checksum = match.group(6)

        divisors = [5, 8, 3, 9]
        for idx, (block, div) in enumerate(zip(blocks, divisors), start=1):
            if block % div != 0:
                raise ValueError(f"Блок {idx} должен быть кратен {div}")

        base = "-".join(key_value.split("-")[:-1])
        if self._checksum(base) != checksum:
            raise ValueError("Неверная контрольная сумма ключа")

        return duration

    def activate_with_key(self, key_value: str) -> LicenseStatus:
        key_value = key_value.strip().upper()
        duration = self.validate_key_format(key_value)
        key_hash = self._key_hash(key_value)

        now = datetime.utcnow()
        expires_at = now + timedelta(days=duration)

        with self._connect() as conn:
            used = conn.execute("SELECT 1 FROM used_keys WHERE key_hash = ?", (key_hash,)).fetchone()
            if used:
                raise ValueError("Этот ключ уже был использован")

            conn.execute(
                "INSERT INTO used_keys(key_hash, used_at) VALUES(?, ?)",
                (key_hash, now.isoformat()),
            )
            conn.execute(
                """
                UPDATE license_state
                SET active_key = ?, activated_at = ?, expires_at = ?
                WHERE id = 1
                """,
                (key_value, now.isoformat(), expires_at.isoformat()),
            )

        return LicenseStatus(True, key_value, expires_at)

    def get_status(self) -> LicenseStatus:
        with self._connect() as conn:
            row = conn.execute("SELECT active_key, expires_at FROM license_state WHERE id = 1").fetchone()

        if not row or not row["active_key"] or not row["expires_at"]:
            return LicenseStatus(False, None, None)

        expires_at = datetime.fromisoformat(row["expires_at"])
        if datetime.utcnow() >= expires_at:
            self.deactivate()
            return LicenseStatus(False, None, None)

        return LicenseStatus(True, row["active_key"], expires_at)

    def deactivate(self) -> None:
        with self._connect() as conn:
            conn.execute("UPDATE license_state SET active_key = NULL, activated_at = NULL, expires_at = NULL WHERE id = 1")