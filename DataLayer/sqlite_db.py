from __future__ import annotations

import sqlite3
import threading
from pathlib import Path

DEFAULT_PATH = Path(__file__).resolve().parent / "events.db"

_SCHEMA = """
CREATE TABLE IF NOT EXISTS events (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    ts            REAL    NOT NULL,
    event_type    TEXT    NOT NULL,
    source_id     TEXT,
    source_name   TEXT,
    person_name   TEXT,
    track_id      INTEGER,
    score         REAL,
    snapshot_path TEXT
);
CREATE INDEX IF NOT EXISTS idx_events_ts ON events(ts DESC);
"""


class SQLiteDB:
    def __init__(self, path: str | Path = DEFAULT_PATH):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self._conn = sqlite3.connect(str(self.path), check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        with self._lock:
            self._conn.executescript(_SCHEMA)
            self._conn.commit()

    def execute(self, sql: str, params: tuple = ()) -> sqlite3.Cursor:
        with self._lock:
            cur = self._conn.execute(sql, params)
            self._conn.commit()
            return cur

    def query(self, sql: str, params: tuple = ()) -> list[sqlite3.Row]:
        with self._lock:
            return self._conn.execute(sql, params).fetchall()

    def close(self) -> None:
        with self._lock:
            self._conn.close()
