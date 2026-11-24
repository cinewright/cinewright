"""Scheduler: lease order, matching, long-poll wake-up and lease expiry (on the Core, without HTTP)."""

from __future__ import annotations

import asyncio
import json
import time

import pytest

from cinewright.api.schemas import ClientKind, Heartbeat, JobRequest, JobState, LeaseRequest, WorkerEvents, WorkerRegistration
from cinewright.core.errors import Conflict


def register(core, models=("test_image",)) -> str:
    registration = WorkerRegistration(host_id="local", engine="wan2gp", models=list(models), health={"state": "ready"})
    return core.register_worker(registration, core.enrollment_token).worker_id


def submit(core, priority="interactive", prompt="a lighthouse") -> str:
    request = JobRequest.model_validate({"capability": "t2i", "prompt": prompt, "priority": priority})
    return core.submit_job(request, ClientKind.UI, None).id


def lease(core, worker_id, wait=0.0):
    return asyncio.run(core.lease(worker_id, LeaseRequest(wait_seconds=wait)))


def test_interactive_jobs_are_leased_before_batch_then_oldest_first(core):
    workers = [register(core) for _ in range(4)]  # one job per worker: a new lease releases the worker's previous one
    batch = submit(core, "batch")
    first = submit(core, "interactive")
    second = submit(core, "interactive")
    order = [lease(core, worker).job.job_id for worker in workers[:3]]
    assert order == [first, second, batch]
    assert lease(core, workers[3]) is None
    assert core.get_job(first).state == JobState.LEASED


def test_a_worker_only_leases_jobs_whose_engine_model_it_serves(core):
    serving = register(core, models=("test_image",))
    other = register(core, models=("test_video",))
    job_id = submit(core)
    assert lease(core, other) is None
    granted = lease(core, serving)
    assert granted.job.job_id == job_id
    assert granted.job.engine_model == "test_image"
    assert 0 <= granted.job.seed < 2**31


def test_expired_lease_requeues_and_fails_after_max_attempts(core, clock):
    worker = register(core)
    job_id = submit(core)
    for attempt in range(1, 4):
        assert lease(core, worker).job.job_id == job_id
        clock.advance(121)
        core.heartbeat(worker, Heartbeat(health={"state": "ready"}))  # keep the worker online, but too late for the lease
        assert core.expire_leases() == 0  # the heartbeat extended the lease
        clock.advance(121)
        assert core.expire_leases() == 1
        expected = JobState.QUEUED if attempt < 3 else JobState.FAILED
        assert core.get_job(job_id).state == expected
    job = core.get_job(job_id)
    assert job.error.code == "lease_expired"


def test_cancel_requested_before_expiry_ends_cancelled(core, clock):
    worker = register(core)
    job_id = submit(core)
    lease(core, worker)
    assert core.cancel_job(job_id).state == JobState.LEASED
    assert core.heartbeat(worker, Heartbeat(health={"state": "busy"})).cancel == [job_id]
    clock.advance(121)
    assert core.expire_leases() == 1
    assert core.get_job(job_id).state == JobState.CANCELLED


def test_long_poll_wakes_when_a_job_arrives_from_another_thread(core):
    worker = register(core)

    async def scenario():
        waiting = asyncio.create_task(core.lease(worker, LeaseRequest(wait_seconds=10)))
        await asyncio.sleep(0.1)
        started = time.monotonic()
        job_id = await asyncio.to_thread(submit, core)
        granted = await asyncio.wait_for(waiting, 5)
        return job_id, granted, time.monotonic() - started

    job_id, granted, elapsed = asyncio.run(scenario())
    assert granted.job.job_id == job_id
    assert elapsed < 2


def test_long_poll_returns_nothing_after_wait_seconds(core):
    worker = register(core)
    started = time.monotonic()
    assert lease(core, worker, wait=0.2) is None
    assert 0.15 < time.monotonic() - started < 3


def queued_reasons(core, job_id) -> list[str | None]:
    rows = core.db.query("SELECT data FROM job_events WHERE job_id = ? AND kind = 'state' ORDER BY seq", (job_id,))
    return [data.get("reason") for data in map(json.loads, (row["data"] for row in rows)) if data["state"] == "queued"]


def test_a_dropped_job_is_requeued_when_its_worker_asks_for_a_new_lease(core, clock):
    worker = register(core)
    job_id = submit(core)
    dropped = lease(core, worker).job
    clock.advance(100)
    core.heartbeat(worker, Heartbeat(health={"state": "ready"}))  # the heartbeat still keeps a running job's lease alive
    assert core.expire_leases() == 0
    # The worker lost the job without reporting (e.g. its completion failed) and asks for new work.
    again = lease(core, worker).job
    assert again.job_id == job_id and again.lease_id != dropped.lease_id  # requeued, then the oldest queued job again
    assert queued_reasons(core, job_id) == [None, "lease_abandoned"]
    with pytest.raises(Conflict):
        core.worker_events(worker, job_id, WorkerEvents(lease_id=dropped.lease_id, events=[{"kind": "log", "data": {}}]))


def test_a_dropped_job_with_a_cancel_request_ends_cancelled(core):
    worker = register(core)
    job_id = submit(core)
    lease(core, worker)
    core.cancel_job(job_id)
    assert core.heartbeat(worker, Heartbeat(health={"state": "ready"})).cancel == [job_id]  # no running job matches on the worker
    assert lease(core, worker) is None
    assert core.get_job(job_id).state == JobState.CANCELLED
    assert core.heartbeat(worker, Heartbeat(health={"state": "ready"})).cancel == []


def test_a_job_dropped_on_every_attempt_fails(core):
    worker = register(core)
    job_id = submit(core)
    for _ in range(3):
        assert lease(core, worker).job.job_id == job_id
    assert lease(core, worker) is None
    job = core.get_job(job_id)
    assert job.state == JobState.FAILED
    assert (job.error.code, job.error.retryable) == ("lease_abandoned", False)


def test_other_workers_jobs_are_not_released(core):
    first, second = register(core), register(core)
    job_id = submit(core)
    lease(core, first)
    assert lease(core, second) is None
    assert core.get_job(job_id).state == JobState.LEASED


def test_a_stale_long_poll_does_not_take_back_the_job_the_workers_retry_got(core):
    worker = register(core)

    async def scenario():
        # The worker's first long-poll lost its connection, but the server side keeps waiting.
        stale = asyncio.create_task(core.lease(worker, LeaseRequest(wait_seconds=0.5)))
        await asyncio.sleep(0.05)
        job_id = submit(core)  # wakes the stale poll
        fresh = await core.lease(worker, LeaseRequest(wait_seconds=0))  # the worker's retry gets the job first
        return job_id, fresh, await asyncio.wait_for(stale, 5)

    job_id, fresh, stale_result = asyncio.run(scenario())
    assert fresh.job.job_id == job_id
    assert stale_result is None
    job = core.get_job(job_id)
    assert job.state == JobState.LEASED
    assert queued_reasons(core, job_id) == [None]
    core.worker_events(worker, job_id, WorkerEvents(lease_id=fresh.job.lease_id, events=[{"kind": "log", "data": {}}]))


def test_transition_refuses_columns_outside_the_allowlist():
    from cinewright.core import scheduler

    with pytest.raises(ValueError, match="not a jobs column"):
        scheduler.transition(None, "job", JobState.QUEUED, **{"state = 'x'; --": 1})


def test_job_columns_match_the_jobs_table():
    import re

    from cinewright.core.scheduler import JOB_COLUMNS
    from cinewright.db.database import bundled_migrations

    sql = (bundled_migrations() / "0001_initial.sql").read_text(encoding="utf-8")
    body = sql.split("CREATE TABLE jobs (", 1)[1].split(");", 1)[0]
    columns = {m.group(1) for m in re.finditer(r"^\s+([a-z_]+)\s+[A-Z]", body, re.MULTILINE)}
    assert JOB_COLUMNS == columns - {"id", "created_at"}
