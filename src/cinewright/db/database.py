"""SQLite access for the core.

One connection shared by all threads behind a re-entrant lock: FastAPI runs sync routes in a
thread pool and async routes on the event loop, and a local app does not need concurrent writers.
WAL journal, foreign keys on, forward-only migrations from `db/migrations/NNNN_name.sql`, tracked in
`schema_migrations`.
"""

from __future__ import annotations

import re
import sqlite3
import threading
from collections.abc import Iterable, Iterator
from contextlib import contextmanager
from datetime import UTC, datetime
from importlib.resources import files
from importlib.resources.abc import Traversable
from pathlib import Path
from typing import Any

_MIGRATION_FILE = re.compile(r"^\d{4}_[a-z0-9_]+\.sql$")


def bundled_migrations() -> Traversable:
    return files("cinewright.db").joinpath("migrations")


class Database:
    def __init__(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        self.path = path
        self._lock = threading.RLock()
        self._conn = sqlite3.connect(path, check_same_thread=False, isolation_level=None, timeout=10)
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA foreign_keys=ON")
        self._conn.execute("PRAGMA synchronous=NORMAL")

    @classmethod
    def open(cls, path: Path) -> Database:
        db = cls(path)
        db.migrate()
        return db

    def migrate(self, directory: Traversable | Path | None = None) -> list[str]:
        """Apply pending migrations in name order, each in its own transaction. Returns the applied versions."""
        source = directory if directory is not None else bundled_migrations()
        scripts = sorted((item.name, item) for item in source.iterdir() if _MIGRATION_FILE.match(item.name))
        applied_now: list[str] = []
        with self._lock:
            self._conn.execute("CREATE TABLE IF NOT EXISTS schema_migrations (version TEXT PRIMARY KEY, applied_at TEXT NOT NULL)")
            applied = {row[0] for row in self._conn.execute("SELECT version FROM schema_migrations")}
            for name, item in scripts:
                version = name.removesuffix(".sql")
                if version in applied:
                    continue
                sql = item.read_text(encoding="utf-8")
                try:
                    self._conn.executescript("BEGIN IMMEDIATE;\n" + sql)
                    self._conn.execute(
                        "INSERT INTO schema_migrations (version, applied_at) VALUES (?, ?)",
                        (version, datetime.now(UTC).isoformat(timespec="microseconds")),
                    )
                    self._conn.execute("COMMIT")
                except BaseException:
                    if self._conn.in_transaction:
                        self._conn.execute("ROLLBACK")
                    raise
                applied_now.append(version)
        return applied_now

    @contextmanager
    def transaction(self) -> Iterator[sqlite3.Connection]:
        """Serialised write transaction; a nested call joins the outer one."""
        with self._lock:
            if self._conn.in_transaction:
                yield self._conn
                return
            self._conn.execute("BEGIN IMMEDIATE")
            try:
                yield self._conn
            except BaseException:
                self._conn.execute("ROLLBACK")
                raise
            self._conn.execute("COMMIT")

    def query(self, sql: str, params: Iterable[Any] = ()) -> list[sqlite3.Row]:
        with self._lock:
            return self._conn.execute(sql, tuple(params)).fetchall()

    def query_one(self, sql: str, params: Iterable[Any] = ()) -> sqlite3.Row | None:
        with self._lock:
            return self._conn.execute(sql, tuple(params)).fetchone()

    def execute(self, sql: str, params: Iterable[Any] = ()) -> int:
        """Run one statement in its own transaction; returns the number of changed rows."""
        with self.transaction() as conn:
            return conn.execute(sql, tuple(params)).rowcount

    def close(self) -> None:
        with self._lock:
            self._conn.close()
