"""The SSE route: keepalives on idle streams, release of gone clients, live-only previews.

The tests call the route function and read its body iterator, so they control `is_disconnected()`.
Through the TestClient, closing a response cancels the app task, which already releases the stream.
"""

from __future__ import annotations

import asyncio

import pytest
from conftest import finish_job, leased_job, preview_event, report

from cinewright.api.routes import jobs
from cinewright.api.schemas import ClientKind, JobEvent, JobRequest, WorkerRegistration

REQUEST = JobRequest.model_validate({"capability": "t2i", "prompt": "a lighthouse"})
KEEPALIVE = b": keepalive\n\n"


class Client:
    """Stands in for the request: only `is_disconnected()` is used by the stream."""

    def __init__(self) -> None:
        self.gone = False

    async def is_disconnected(self) -> bool:
        return self.gone


def subscriptions(core) -> int:
    return len(core.event_log.broadcaster._subscriptions)


async def until(condition, timeout=5.0) -> None:
    deadline = asyncio.get_running_loop().time() + timeout
    while not condition():
        assert asyncio.get_running_loop().time() < deadline, "condition not met in time"
        await asyncio.sleep(0.005)


async def read_all(body) -> list[bytes]:
    return [chunk async for chunk in body]


async def next_event(body) -> bytes:
    while (chunk := await anext(body)).startswith(b":"):
        pass
    return chunk


def parse(chunks: list[bytes]) -> list[tuple[str | None, JobEvent]]:
    """SSE messages as (id or None, event); comments are skipped."""
    messages = []
    for block in b"".join(chunks).decode().split("\n\n"):
        lines = [line for line in block.splitlines() if not line.startswith(":")]
        if not lines:
            continue
        fields = dict(line.split(": ", 1) for line in lines)
        event = JobEvent.model_validate_json(fields["data"])
        assert fields["event"] == event.kind
        messages.append((fields.get("id"), event))
    return messages


@pytest.fixture
def short_keepalive(monkeypatch):
    monkeypatch.setattr(jobs, "SSE_KEEPALIVE_SECONDS", 0.05)


@pytest.mark.usefixtures("short_keepalive")
def test_an_idle_stream_sends_keepalives_and_still_delivers_later_events(core):
    registration = WorkerRegistration(host_id="local", engine="wan2gp", models=["test_image"], health={"state": "ready"})
    core.register_worker(registration, core.enrollment_token)  # so the job below has a route

    async def scenario():
        body = (await jobs.stream_events(Client(), core, None, None)).body_iterator
        assert (await anext(body)).startswith(b": cinewright events")
        assert await asyncio.wait_for(anext(body), 2) == KEEPALIVE
        assert await asyncio.wait_for(anext(body), 2) == KEEPALIVE
        job = await asyncio.to_thread(core.submit_job, REQUEST, ClientKind.UI, None)
        chunk = await asyncio.wait_for(next_event(body), 2)
        await body.aclose()
        return job.id, chunk

    job_id, chunk = asyncio.run(scenario())
    assert chunk.startswith(f"id: {job_id}:1\nevent: state\n".encode())
    assert subscriptions(core) == 0


@pytest.mark.usefixtures("short_keepalive")
def test_a_gone_client_is_released_without_a_new_event(core):
    async def scenario():
        client = Client()
        chunks: list[bytes] = []

        async def follow(body):
            async for chunk in body:
                chunks.append(chunk)

        reader = asyncio.create_task(follow((await jobs.stream_events(client, core, None, None)).body_iterator))
        await until(lambda: subscriptions(core) == 1)
        client.gone, sent = True, len(chunks)
        await asyncio.wait_for(reader, 2)  # only guards against a hang; no event is published
        # Checked inside the loop: asyncio.run's teardown would close a leaked stream anyway.
        return chunks[sent:], subscriptions(core)

    after_gone, left = asyncio.run(scenario())
    assert after_gone == [KEEPALIVE]  # released after the next keepalive interval, not later
    assert left == 0


def test_cancelling_an_idle_stream_releases_its_subscription(core):
    async def scenario():
        reader = asyncio.create_task(read_all((await jobs.stream_events(Client(), core, None, None)).body_iterator))
        await until(lambda: subscriptions(core) == 1)
        reader.cancel()  # what Starlette does when the server reports a disconnect
        with pytest.raises(asyncio.CancelledError):
            await reader
        return subscriptions(core)

    assert asyncio.run(scenario()) == 0


def test_previews_reach_subscribers_live_without_being_stored(core):
    worker, job_id, lease_id = leased_job(core)
    log = {"kind": "log", "data": {"message": "step done"}}

    def batch(frame: int) -> list[dict]:
        progress = {"kind": "progress", "data": {"phase": "inference", "step": frame, "total_steps": 100, "percent": frame}}
        return [[progress, preview_event(frame)], [preview_event(frame), log], [preview_event(frame)]][frame % 3]

    async def scenario():
        chunks: list[bytes] = []

        async def follow(body):
            async for chunk in body:
                chunks.append(chunk)

        reader = asyncio.create_task(follow((await jobs.stream_events(Client(), core, job_id, None)).body_iterator))
        await until(lambda: len(chunks) == 3)  # the opening comment and the replayed queued and leased states
        for frame in range(100):
            await asyncio.to_thread(report, core, worker, job_id, lease_id, *batch(frame))
        kept = core.event_log.kept_previews()
        await asyncio.to_thread(finish_job, core, worker, job_id, lease_id)
        await asyncio.wait_for(reader, 5)  # a job-filtered stream ends after the terminal state
        return chunks, kept

    chunks, kept = asyncio.run(scenario())
    messages = parse(chunks)
    previews = [(event_id, event) for event_id, event in messages if event.kind == "preview"]
    assert [event.data for _, event in previews] == [preview_event(frame)["data"] for frame in range(100)]
    assert all(event_id is None for event_id, _ in previews)
    assert [event.data for event in kept] == [preview_event(99)["data"]]

    stored = [(event_id, event) for event_id, event in messages if event.kind != "preview"]
    assert [event.seq for _, event in stored] == list(range(1, len(stored) + 1))
    assert all(event_id == f"{job_id}:{event.seq}" for event_id, event in stored)
    latest = 0
    for _, event in messages:  # a preview carries the seq of the job's latest stored event
        if event.kind == "preview":
            assert event.seq == latest
        else:
            latest = event.seq

    rows = core.db.query("SELECT seq, kind FROM job_events WHERE job_id = ? ORDER BY seq", (job_id,))
    assert "preview" not in [row["kind"] for row in rows]
    assert [row["seq"] for row in rows] == [event.seq for _, event in stored]
    assert core.event_log.kept_previews() == []
