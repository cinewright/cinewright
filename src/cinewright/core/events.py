"""Job events: the `job_events` table (per-job `seq`) plus an in-process broadcaster for live SSE.

Events are written inside the transaction that changes the job and published after it commits.
The table's rowid gives a global order, which the unfiltered stream uses for replay.
"""

from __future__ import annotations

import json
import sqlite3
from collections.abc import AsyncIterator
from datetime import datetime
from typing import Any

from cinewright.api.schemas import TERMINAL_JOB_STATES, EventKind, JobEvent
from cinewright.core.clock import from_iso, to_iso
from cinewright.core.notify import Broadcaster
from cinewright.db.database import Database

_TERMINAL = {state.value for state in TERMINAL_JOB_STATES}

Published = tuple[int, JobEvent]  # (rowid, event)


def is_terminal(event: JobEvent) -> bool:
    return event.kind == EventKind.STATE and event.data.get("state") in _TERMINAL


def _event(row: sqlite3.Row) -> JobEvent:
    return JobEvent(
        job_id=row["job_id"],
        seq=row["seq"],
        kind=EventKind(row["kind"]),
        data=json.loads(row["data"]),
        created_at=from_iso(row["created_at"]),
    )


def _parse_last_event_id(value: str | None) -> tuple[str, int] | None:
    if not value or ":" not in value:
        return None
    job_id, _, seq = value.rpartition(":")
    return (job_id, int(seq)) if job_id and seq.isdigit() else None


class EventLog:
    def __init__(self, db: Database) -> None:
        self.db = db
        self.broadcaster = Broadcaster()

    @staticmethod
    def append(conn: sqlite3.Connection, job_id: str, kind: EventKind, data: dict[str, Any], now: datetime) -> Published:
        seq = conn.execute("SELECT COALESCE(MAX(seq), 0) + 1 FROM job_events WHERE job_id = ?", (job_id,)).fetchone()[0]
        cursor = conn.execute(
            "INSERT INTO job_events (job_id, seq, kind, data, created_at) VALUES (?, ?, ?, ?, ?)",
            (job_id, seq, kind.value, json.dumps(data), to_iso(now)),
        )
        return cursor.lastrowid, JobEvent(job_id=job_id, seq=seq, kind=kind, data=data, created_at=now)

    def publish(self, items: list[Published]) -> None:
        for item in items:
            self.broadcaster.publish(item)

    def close(self) -> None:
        self.broadcaster.close()

    def _replay(self, job_id: str | None, after: str | None) -> tuple[list[Published], int, bool]:
        """Events to replay, the rowid watermark and whether the filtered job already finished.

        Runs as one transaction so no commit falls between reading the events and the job state.
        """
        last = _parse_last_event_id(after)
        with self.db.transaction() as conn:
            watermark = conn.execute("SELECT COALESCE(MAX(rowid), 0) FROM job_events").fetchone()[0]
            start = None
            if last is not None:
                row = conn.execute("SELECT rowid FROM job_events WHERE job_id = ? AND seq = ?", last).fetchone()
                start = row[0] if row else None
            if job_id is not None:
                state = conn.execute("SELECT state FROM jobs WHERE id = ?", (job_id,)).fetchone()
                if state is None:
                    return [], watermark, True
                if start is not None:
                    rows = conn.execute(
                        "SELECT rowid, * FROM job_events WHERE job_id = ? AND rowid > ? ORDER BY rowid", (job_id, start)
                    ).fetchall()
                else:
                    after_seq = last[1] if last is not None and last[0] == job_id else 0
                    rows = conn.execute(
                        "SELECT rowid, * FROM job_events WHERE job_id = ? AND seq > ? ORDER BY rowid", (job_id, after_seq)
                    ).fetchall()
                finished = state["state"] in _TERMINAL
            else:
                rows = conn.execute("SELECT rowid, * FROM job_events WHERE rowid > ? ORDER BY rowid", (start,)).fetchall() if start else []
                finished = False
        return [(row["rowid"], _event(row)) for row in rows], watermark, finished

    async def stream(self, job_id: str | None, after: str | None) -> AsyncIterator[JobEvent]:
        """Replay, then follow live events. A job-filtered stream ends after the job's terminal state.

        Without `after`, a job-filtered stream replays the job's whole history and the unfiltered
        stream starts with live events.
        """
        subscription = self.broadcaster.subscribe()
        try:
            replay, watermark, finished = self._replay(job_id, after)
            for rowid, event in replay:
                watermark = max(watermark, rowid)
                yield event
                if job_id is not None and is_terminal(event):
                    return
            if finished:
                return
            while True:
                item = await subscription.queue.get()
                if item is None:
                    return
                rowid, event = item
                if rowid <= watermark or (job_id is not None and event.job_id != job_id):
                    continue
                yield event
                if job_id is not None and is_terminal(event):
                    return
        finally:
            self.broadcaster.unsubscribe(subscription)
