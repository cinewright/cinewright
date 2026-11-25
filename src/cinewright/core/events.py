"""Job events: the `job_events` table (per-job `seq`) plus an in-process broadcaster for live SSE.

Events are written inside the transaction that changes the job and published after it commits.
The table's rowid gives a global order, which the unfiltered stream uses for replay.

Worker previews are the exception: they are broadcast but never stored, because a preview of up to
256 KiB every 2 s would grow the database by megabytes per minute of generation. A preview carries
the `seq` of the job's latest stored event, so `Last-Event-ID` always names a stored event. The log
keeps the latest preview of each running job in memory and sends it to a stream after its replay.
Releases 0.0.1 and 0.0.2 did store previews; those rows stay in the table but replay never sends them.

System events (engine provisioning, model installs) are not job events: they are live only, neither stored nor
replayed nor kept, and only the unfiltered stream carries them. They are broadcast as they are, not
as a counted `(published, Published)` item, so they never touch the watermark or the publish counter.
"""

from __future__ import annotations

import json
import sqlite3
import threading
from collections.abc import AsyncGenerator
from datetime import datetime
from typing import Any, NamedTuple

from cinewright.api.schemas import TERMINAL_JOB_STATES, EventKind, JobEvent, JobState, SystemEvent
from cinewright.core.clock import from_iso, to_iso
from cinewright.core.notify import Broadcaster
from cinewright.db.database import Database

_TERMINAL = {state.value for state in TERMINAL_JOB_STATES}


class Published(NamedTuple):
    """An event of a committed transaction; `rowid` is None for a preview, which is not stored."""

    rowid: int | None
    event: JobEvent


class _Replay(NamedTuple):
    events: list[Published]
    watermark: int  # highest `job_events` rowid at the snapshot
    finished: bool  # the filtered job does not exist or already finished
    published: int  # publish counter at the snapshot
    previews: list[JobEvent]  # kept previews of the jobs the stream covers


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
        # Held by the core from a write transaction's commit through its publish, and by a stream's
        # replay snapshot. So subscribers see events in commit order, and a snapshot never sees a
        # commit whose events (and kept preview) are not yet published.
        self.lock = threading.RLock()
        # Counts broadcast job events and previews (not system events), so a stream can skip previews its snapshot covers.
        self._published = 0
        self._previews: dict[str, JobEvent] = {}  # latest preview per running job

    @staticmethod
    def append(conn: sqlite3.Connection, job_id: str, kind: EventKind, data: dict[str, Any], now: datetime) -> Published:
        seq = conn.execute("SELECT COALESCE(MAX(seq), 0) + 1 FROM job_events WHERE job_id = ?", (job_id,)).fetchone()[0]
        cursor = conn.execute(
            "INSERT INTO job_events (job_id, seq, kind, data, created_at) VALUES (?, ?, ?, ?, ?)",
            (job_id, seq, kind.value, json.dumps(data), to_iso(now)),
        )
        return Published(cursor.lastrowid, JobEvent(job_id=job_id, seq=seq, kind=kind, data=data, created_at=now))

    @staticmethod
    def live(conn: sqlite3.Connection, job_id: str, kind: EventKind, data: dict[str, Any], now: datetime) -> Published:
        """An event that is broadcast but not stored; it carries the `seq` of the job's latest stored event."""
        seq = conn.execute("SELECT COALESCE(MAX(seq), 0) FROM job_events WHERE job_id = ?", (job_id,)).fetchone()[0]
        return Published(None, JobEvent(job_id=job_id, seq=seq, kind=kind, data=data, created_at=now))

    def publish(self, items: list[Published]) -> None:
        """Broadcast committed events in order; keep each preview until the job leaves `running`."""
        with self.lock:
            for item in items:
                event = item.event
                if item.rowid is None and event.kind == EventKind.PREVIEW:
                    self._previews[event.job_id] = event
                elif event.kind == EventKind.STATE and event.data.get("state") != JobState.RUNNING.value:
                    self._previews.pop(event.job_id, None)
                self._published += 1
                self.broadcaster.publish((self._published, item))

    def kept_previews(self, job_id: str | None = None) -> list[JobEvent]:
        """The kept latest previews: of one job, or of every running job."""
        with self.lock:
            if job_id is None:
                return list(self._previews.values())
            preview = self._previews.get(job_id)
            return [preview] if preview is not None else []

    def publish_system(self, event: SystemEvent) -> None:
        """Callable from any thread; reaches open unfiltered streams only.

        Takes neither the lock nor a publish count: a system event is not ordered against job commits
        and is never replayed or kept, so no replay snapshot has to account for it.
        """
        self.broadcaster.publish(event)

    def close(self) -> None:
        self.broadcaster.close()

    def _replay(self, job_id: str | None, after: str | None) -> _Replay:
        """Events to replay, the snapshot's watermarks, whether the filtered job already finished, and the kept previews.

        Runs as one transaction under the lock, so no commit or publish falls between reading the
        events, the job state and the kept previews.
        """
        last = _parse_last_event_id(after)
        with self.lock, self.db.transaction() as conn:
            published = self._published
            watermark = conn.execute("SELECT COALESCE(MAX(rowid), 0) FROM job_events").fetchone()[0]
            start = None
            if last is not None:
                # Any kind: 0.0.1 and 0.0.2 sent previews with an `id:`, so Last-Event-ID can name an old preview row.
                row = conn.execute("SELECT rowid FROM job_events WHERE job_id = ? AND seq = ?", last).fetchone()
                start = row[0] if row else None
            # Replay reads stored events only: releases 0.0.1 and 0.0.2 also stored previews, which it skips.
            if job_id is not None:
                state = conn.execute("SELECT state FROM jobs WHERE id = ?", (job_id,)).fetchone()
                if state is None:
                    return _Replay([], watermark, True, published, [])
                if start is not None:
                    rows = conn.execute(
                        "SELECT rowid, * FROM job_events WHERE kind != 'preview' AND job_id = ? AND rowid > ? ORDER BY rowid",
                        (job_id, start),
                    ).fetchall()
                else:
                    after_seq = last[1] if last is not None and last[0] == job_id else 0
                    rows = conn.execute(
                        "SELECT rowid, * FROM job_events WHERE kind != 'preview' AND job_id = ? AND seq > ? ORDER BY rowid",
                        (job_id, after_seq),
                    ).fetchall()
                finished = state["state"] in _TERMINAL
            else:
                rows = []
                if start:
                    rows = conn.execute(
                        "SELECT rowid, * FROM job_events WHERE kind != 'preview' AND rowid > ? ORDER BY rowid", (start,)
                    ).fetchall()
                finished = False
            previews = [] if finished else self.kept_previews(job_id)
        return _Replay([Published(row["rowid"], _event(row)) for row in rows], watermark, finished, published, previews)

    async def stream(self, job_id: str | None, after: str | None) -> AsyncGenerator[JobEvent | SystemEvent, None]:
        """Replay, then the kept previews, then follow live events. A job-filtered stream ends after the job's terminal state.

        Without `after`, a job-filtered stream replays the job's whole history and the unfiltered
        stream starts with the kept previews and live events. Only the unfiltered stream carries
        system events; they are live only, so one published before the stream subscribed is not sent.
        """
        subscription = self.broadcaster.subscribe()
        try:
            replay = self._replay(job_id, after)
            for _, event in replay.events:
                yield event
                if job_id is not None and is_terminal(event):
                    return
            if replay.finished:
                return
            for event in replay.previews:
                yield event
            while True:
                item = await subscription.queue.get()
                if item is None:
                    return
                if isinstance(item, SystemEvent):
                    # Not a job event: never replayed, so neither the watermark nor the publish counter applies.
                    if job_id is None:
                        yield item
                    continue
                published, (rowid, event) = item
                if job_id is not None and event.job_id != job_id:
                    continue
                # A stored event up to the watermark was replayed. A preview published before the
                # snapshot is the kept preview sent above, or older than it.
                if (rowid <= replay.watermark) if rowid is not None else (published <= replay.published):
                    continue
                yield event
                if job_id is not None and is_terminal(event):
                    return
        finally:
            self.broadcaster.unsubscribe(subscription)
