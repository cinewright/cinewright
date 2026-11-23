"""Event log: live broadcast across threads, replay after Last-Event-ID, end of job-filtered streams."""

from __future__ import annotations

import asyncio

import pytest

from cinewright.api.schemas import ClientKind, JobRequest, WorkerRegistration

REQUEST = JobRequest.model_validate({"capability": "t2i", "prompt": "a lighthouse"})


@pytest.fixture(autouse=True)
def worker(core):
    registration = WorkerRegistration(host_id="local", engine="wan2gp", models=["test_image"], health={"state": "ready"})
    return core.register_worker(registration, core.enrollment_token).worker_id


def collect(core, job_id=None, after=None, count=None):
    async def run():
        events = []
        async for event in core.events(job_id, after):
            events.append(event)
            if count is not None and len(events) == count:
                break
        return events

    return asyncio.run(asyncio.wait_for(run(), 5))


def test_live_events_from_other_threads_reach_the_stream(core):
    async def scenario():
        received = []

        async def follow():
            async for event in core.events(None, None):
                received.append(event)
                if len(received) == 2:
                    return

        task = asyncio.create_task(follow())
        await asyncio.sleep(0.05)
        first = await asyncio.to_thread(core.submit_job, REQUEST, ClientKind.UI, None)
        await asyncio.to_thread(core.cancel_job, first.id)
        await asyncio.wait_for(task, 5)
        return first.id, received

    job_id, received = asyncio.run(scenario())
    assert [(e.job_id, e.seq, e.data["state"]) for e in received] == [(job_id, 1, "queued"), (job_id, 2, "cancelled")]


def test_job_stream_replays_history_and_ends_after_the_terminal_state(core):
    job_id = core.submit_job(REQUEST, ClientKind.UI, None).id
    core.cancel_job(job_id)
    assert [e.seq for e in collect(core, job_id)] == [1, 2]
    assert [e.seq for e in collect(core, job_id, after=f"{job_id}:1")] == [2]
    assert collect(core, job_id, after=f"{job_id}:2") == []
    assert collect(core, "no-such-job") == []


def test_unfiltered_stream_replays_after_last_event_id(core):
    first = core.submit_job(REQUEST, ClientKind.UI, None).id
    second = core.submit_job(REQUEST, ClientKind.UI, None).id
    core.cancel_job(first)
    events = collect(core, None, after=f"{first}:1", count=2)
    assert [(e.job_id, e.seq) for e in events] == [(second, 1), (first, 2)]


def test_malformed_last_event_id_is_ignored(core):
    job_id = core.submit_job(REQUEST, ClientKind.UI, None).id
    core.cancel_job(job_id)
    assert [e.seq for e in collect(core, job_id, after="garbage")] == [1, 2]


def test_closing_the_core_ends_open_streams(core):
    async def scenario():
        task = asyncio.create_task(_drain(core))
        await asyncio.sleep(0.05)
        await asyncio.to_thread(core.close)
        return await asyncio.wait_for(task, 5)

    assert asyncio.run(scenario()) == []


async def _drain(core):
    return [event async for event in core.events(None, None)]
