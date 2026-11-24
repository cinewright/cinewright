"""Persistent job queue in SQLite with pull leases.

A worker leases the next `queued` job whose route it can serve (same host and engine, engine model
in the worker's list): interactive before batch, then oldest first. A lease lasts `lease_seconds`
and is extended by every authenticated worker call for the job and by the worker's heartbeat; an
expired lease puts the job back into the queue, or fails it after `max_attempts` leases.

A worker runs one job at a time and asks for a lease only when it is idle, so a job still leased
to a worker that asks for a new one was dropped without a report: it is released as abandoned
(requeued, cancelled if a cancel was requested, or failed after `max_attempts`) before the new
lease is granted. Otherwise the heartbeat would keep that job's lease alive for as long as the
worker process lives.
"""

from __future__ import annotations

import sqlite3
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any
from uuid import uuid4

from cinewright.api.schemas import EventKind, JobError, JobState, Route
from cinewright.core.clock import to_iso
from cinewright.core.notify import Notifier
from cinewright.core.registry import Registry

LEASE_SECONDS = 120.0
MAX_ATTEMPTS = 3
ACTIVE_STATES = (JobState.LEASED.value, JobState.RUNNING.value)

Emit = Callable[[str, EventKind, dict[str, Any]], None]


@dataclass(frozen=True)
class WorkerRecord:
    id: str
    host_id: str
    engine: str
    engine_version: str | None
    models: frozenset[str]


# Columns of the jobs table that transition() may set (db/migrations/0001_initial.sql).
JOB_COLUMNS = frozenset(
    {
        "state",
        "priority",
        "capability",
        "data_class",
        "request",
        "route",
        "seed",
        "client",
        "agent_id",
        "progress",
        "eta_seconds",
        "error",
        "attempts",
        "cancel_requested",
        "worker_id",
        "lease_id",
        "lease_expires_at",
        "started_at",
        "finished_at",
    }
)


@dataclass
class Tx:
    """A write transaction: the connection, its timestamp and the events to publish after commit.

    `emit` stores an event in `job_events`; `broadcast` publishes one to live subscribers only.
    """

    conn: sqlite3.Connection
    now: datetime
    emit: Emit
    broadcast: Emit
    wake_workers: bool = False


def transition(tx: Tx, job_id: str, state: JobState, data: dict[str, Any] | None = None, **columns: Any) -> None:
    """Set a job's state (plus other columns) and record the state event."""
    values = {"state": state.value, **columns}
    unknown = set(values) - JOB_COLUMNS
    if unknown:
        raise ValueError(f"not a jobs column: {', '.join(sorted(unknown))}")
    assignments = ", ".join(f"{name} = ?" for name in values)
    # Column names are checked against JOB_COLUMNS above; every value is a bound parameter.
    query = f"UPDATE jobs SET {assignments} WHERE id = ?"  # noqa: S608
    # Semgrep's raw-query rule cannot see the allowlist check, hence the inline suppression.
    tx.conn.execute(query, (*values.values(), job_id))  # nosemgrep
    tx.emit(job_id, EventKind.STATE, {"state": state.value, **(data or {})})
    if state == JobState.QUEUED:
        tx.wake_workers = True


class Scheduler:
    def __init__(self, registry: Registry, *, lease_seconds: float = LEASE_SECONDS, max_attempts: int = MAX_ATTEMPTS) -> None:
        self.registry = registry
        self.lease_seconds = lease_seconds
        self.max_attempts = max_attempts
        self.wakeup = Notifier()

    def serves(self, worker: WorkerRecord, route: Route) -> bool:
        binding = self.registry.binding(route.model_id, route.quant, route.engine)
        return (
            binding is not None
            and route.host_id == worker.host_id
            and route.engine == worker.engine
            and binding.model_type in worker.models
        )

    def lease_next(self, tx: Tx, worker: WorkerRecord, *, abandon: bool = True) -> sqlite3.Row | None:
        """Lease the next job the worker serves; first release its abandoned jobs unless `abandon` is False."""
        if abandon:
            self.abandon(tx, worker.id)
        rows = tx.conn.execute(
            "SELECT id, route FROM jobs WHERE state = 'queued' "
            "ORDER BY CASE priority WHEN 'interactive' THEN 0 ELSE 1 END, created_at, rowid"
        ).fetchall()
        job_id = next((row["id"] for row in rows if self.serves(worker, Route.model_validate_json(row["route"]))), None)
        if job_id is None:
            return None
        transition(
            tx,
            job_id,
            JobState.LEASED,
            {"worker_id": worker.id},
            worker_id=worker.id,
            lease_id=uuid4().hex,
            lease_expires_at=self.expiry(tx.now),
            attempts=tx.conn.execute("SELECT attempts + 1 FROM jobs WHERE id = ?", (job_id,)).fetchone()[0],
        )
        return tx.conn.execute("SELECT * FROM jobs WHERE id = ?", (job_id,)).fetchone()

    def expiry(self, now: datetime) -> str:
        return to_iso(now + timedelta(seconds=self.lease_seconds))

    def extend(self, tx: Tx, worker_id: str, job_id: str | None = None) -> None:
        if job_id is None:
            tx.conn.execute(
                "UPDATE jobs SET lease_expires_at = ? WHERE worker_id = ? AND state IN (?, ?)",
                (self.expiry(tx.now), worker_id, *ACTIVE_STATES),
            )
        else:
            tx.conn.execute("UPDATE jobs SET lease_expires_at = ? WHERE id = ? AND worker_id = ?", (self.expiry(tx.now), job_id, worker_id))

    def release(self, tx: Tx, row: sqlite3.Row, reason: str, error: JobError) -> JobState:
        """Take a job away from its worker: cancel it if requested, requeue it, or fail it after too many attempts."""
        cleared = {"worker_id": None, "lease_id": None, "lease_expires_at": None}
        if row["cancel_requested"]:
            transition(tx, row["id"], JobState.CANCELLED, {"reason": reason}, finished_at=to_iso(tx.now), **cleared)
            return JobState.CANCELLED
        if row["attempts"] >= self.max_attempts:
            failure = error.model_copy(update={"retryable": False})
            transition(
                tx,
                row["id"],
                JobState.FAILED,
                {"reason": reason, "error": failure.model_dump()},
                error=failure.model_dump_json(),
                finished_at=to_iso(tx.now),
                **cleared,
            )
            return JobState.FAILED
        transition(tx, row["id"], JobState.QUEUED, {"reason": reason}, progress=None, **cleared)
        return JobState.QUEUED

    def abandon(self, tx: Tx, worker_id: str) -> int:
        """Release every job still leased to a worker that asks for a new lease; returns how many."""
        rows = tx.conn.execute("SELECT * FROM jobs WHERE worker_id = ? AND state IN (?, ?)", (worker_id, *ACTIVE_STATES)).fetchall()
        error = JobError(
            code="lease_abandoned",
            message="The worker asked for a new job without reporting on this one, so the core took it back.",
            retryable=True,
        )
        for row in rows:
            self.release(tx, row, "lease_abandoned", error)
        return len(rows)

    def expire(self, tx: Tx) -> int:
        rows = tx.conn.execute(
            "SELECT * FROM jobs WHERE state IN (?, ?) AND lease_expires_at < ?", (*ACTIVE_STATES, to_iso(tx.now))
        ).fetchall()
        error = JobError(code="lease_expired", message=f"The worker stopped reporting for {self.lease_seconds:g} s, so the lease expired.")
        for row in rows:
            self.release(tx, row, "lease_expired", error)
        return len(rows)
