"""Database: WAL, foreign keys, forward-only migrations tracked in schema_migrations."""

from __future__ import annotations

import sqlite3

import pytest

from cinewright.db.database import Database


def test_bundled_migrations_apply_once(tmp_path):
    db = Database(tmp_path / "db" / "test.sqlite3")
    try:
        assert db.migrate() == ["0001_initial", "0002_job_outputs"]
        assert db.migrate() == []
        assert db.query_one("PRAGMA journal_mode")[0] == "wal"
        assert db.query_one("PRAGMA foreign_keys")[0] == 1
        tables = {row["name"] for row in db.query("SELECT name FROM sqlite_master WHERE type = 'table'")}
        assert {"jobs", "job_events", "job_outputs", "assets", "hosts", "workers", "settings", "schema_migrations"} <= tables
        with pytest.raises(sqlite3.IntegrityError):  # workers.host_id references hosts
            db.execute(
                "INSERT INTO workers (id, host_id, engine, models, health, token_hash, registered_at, last_seen_at) "
                "VALUES ('w', 'missing', 'e', '[]', '{}', 'h', 'now', 'now')"
            )
        with pytest.raises(sqlite3.IntegrityError):  # job_outputs.job_id references jobs
            db.execute(
                "INSERT INTO job_outputs (job_id, name, sha256, media_type, size_bytes, created_at) "
                "VALUES ('missing', 'image.png', 'a', 'image/png', 1, 'now')"
            )
    finally:
        db.close()


def test_failed_migration_rolls_back_and_is_not_recorded(tmp_path):
    migrations = tmp_path / "migrations"
    migrations.mkdir()
    (migrations / "0001_good.sql").write_text("CREATE TABLE a (x INTEGER);", encoding="utf-8")
    (migrations / "0002_bad.sql").write_text("CREATE TABLE b (x INTEGER);\nTHIS IS NOT SQL;", encoding="utf-8")
    (migrations / "notes.txt").write_text("ignored", encoding="utf-8")
    db = Database(tmp_path / "test.sqlite3")
    try:
        with pytest.raises(sqlite3.OperationalError):
            db.migrate(migrations)
        tables = {row["name"] for row in db.query("SELECT name FROM sqlite_master WHERE type = 'table'")}
        assert "a" in tables and "b" not in tables
        assert [row["version"] for row in db.query("SELECT version FROM schema_migrations")] == ["0001_good"]
    finally:
        db.close()


def test_transactions_roll_back_on_error_and_nest(tmp_path):
    db = Database.open(tmp_path / "test.sqlite3")
    try:
        with pytest.raises(RuntimeError), db.transaction() as conn:
            conn.execute("INSERT INTO settings (key, value) VALUES ('a', '1')")
            raise RuntimeError
        assert db.query_one("SELECT COUNT(*) FROM settings")[0] == 0
        with db.transaction() as outer:
            outer.execute("INSERT INTO settings (key, value) VALUES ('a', '1')")
            with db.transaction() as inner:
                inner.execute("INSERT INTO settings (key, value) VALUES ('b', '2')")
        assert db.query_one("SELECT COUNT(*) FROM settings")[0] == 2
    finally:
        db.close()
