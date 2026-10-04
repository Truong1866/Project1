from __future__ import annotations

from DataLayer.sqlite_db import SQLiteDB


class EventRepository:
    def __init__(self, db: SQLiteDB):
        self.db = db

    def add(self, ts: float, event_type: str, source_id: str, source_name: str,
            person_name: str | None, track_id: int | None, score: float | None,
            snapshot_path: str | None) -> int:
        cur = self.db.execute(
            "INSERT INTO events (ts, event_type, source_id, source_name, person_name, track_id, score, snapshot_path)"
            " VALUES (?,?,?,?,?,?,?,?)",
            (ts, event_type, source_id, source_name, person_name, track_id, score, snapshot_path),
        )
        return int(cur.lastrowid)

    def recent(self, limit: int = 100) -> list[dict]:
        rows = self.db.query("SELECT * FROM events ORDER BY ts DESC LIMIT ?", (limit,))
        return [dict(r) for r in rows]

    def count(self) -> int:
        return int(self.db.query("SELECT COUNT(*) AS c FROM events")[0]["c"])

    def clear(self) -> None:
        self.db.execute("DELETE FROM events")