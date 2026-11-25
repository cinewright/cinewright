"""Estimates, jobs and the job event stream (SSE)."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator

from fastapi import APIRouter, Header, Query, Request
from fastapi.responses import StreamingResponse

from cinewright.api.deps import Core
from cinewright.api.problems import PROBLEM_RESPONSE
from cinewright.api.schemas import ClientKind, Estimate, EventKind, Job, JobEvent, JobList, JobRequest, JobState

router = APIRouter(tags=["jobs"], responses={404: PROBLEM_RESPONSE, 409: PROBLEM_RESPONSE, 422: PROBLEM_RESPONSE, 501: PROBLEM_RESPONSE})

SSE_KEEPALIVE_SECONDS = 15
CLIENT_HEADER = "X-Cinewright-Client"
AGENT_HEADER = "X-Cinewright-Agent"
AGENT_PATTERN = r"^[A-Za-z0-9 ._-]{1,64}$"


@router.post("/estimate", response_model=Estimate, operation_id="estimate")
def estimate(request: JobRequest, core: Core) -> Estimate:
    return core.estimate(request)


@router.post("/jobs", response_model=Job, status_code=202, operation_id="submitJob")
def submit_job(
    request: JobRequest,
    core: Core,
    client: ClientKind = Header(default=ClientKind.UI, alias=CLIENT_HEADER, description="Client that submits the job"),
    agent: str | None = Header(
        default=None,
        alias=AGENT_HEADER,
        pattern=AGENT_PATTERN,
        description="Name of the agent behind an MCP client; self-declared, recorded only for client `mcp`",
    ),
) -> Job:
    return core.submit_job(request, client, agent if client == ClientKind.MCP else None)


@router.get("/jobs", response_model=JobList, operation_id="listJobs")
def list_jobs(
    core: Core,
    state: JobState | None = Query(default=None),
    cursor: str | None = Query(default=None),
    limit: int = Query(default=50, ge=1, le=200),
) -> JobList:
    return core.list_jobs(state, cursor, limit)


@router.get("/jobs/{job_id}", response_model=Job, operation_id="getJob")
def get_job(job_id: str, core: Core) -> Job:
    return core.get_job(job_id)


@router.post("/jobs/{job_id}/cancel", response_model=Job, operation_id="cancelJob")
def cancel_job(job_id: str, core: Core) -> Job:
    return core.cancel_job(job_id)


@router.get(
    "/events",
    operation_id="streamEvents",
    response_class=StreamingResponse,
    responses={200: {"description": "Server-sent events; each `data:` line is a JobEvent", "content": {"text/event-stream": {}}}},
)
async def stream_events(
    request: Request,
    core: Core,
    job_id: str | None = Query(default=None, description="Only events of this job"),
    last_event_id: str | None = Header(default=None, alias="Last-Event-ID"),
) -> StreamingResponse:
    async def body() -> AsyncIterator[bytes]:
        yield f": cinewright events, keepalive {SSE_KEEPALIVE_SECONDS}s\n\n".encode()
        events = core.events(job_id, last_event_id)
        # One pending wait survives keepalive timeouts: `asyncio.wait_for` would cancel the event
        # generator on timeout, which closes it, so no later event would arrive.
        pending: asyncio.Task[JobEvent | None] | None = None
        try:
            while True:
                if pending is None:
                    pending = asyncio.create_task(_next_event(events))
                done, _ = await asyncio.wait({pending}, timeout=SSE_KEEPALIVE_SECONDS)
                if not done:
                    # An idle stream writes nothing, so a silently dropped connection shows only now.
                    yield b": keepalive\n\n"
                    if await request.is_disconnected():
                        return
                    continue
                event, pending = pending.result(), None
                if event is None or await request.is_disconnected():
                    return
                yield _sse(event)
        finally:
            if pending is not None and not pending.done():
                pending.cancel()  # throws into the generator, whose `finally` unsubscribes
                await asyncio.wait({pending})
            await events.aclose()

    return StreamingResponse(body(), media_type="text/event-stream", headers={"Cache-Control": "no-store"})


async def _next_event(events: AsyncIterator[JobEvent]) -> JobEvent | None:
    return await anext(events, None)


def _sse(event: JobEvent) -> bytes:
    # A preview is not stored, so it carries no `id:` and Last-Event-ID always names a stored event.
    event_id = "" if event.kind == EventKind.PREVIEW else f"id: {event.job_id}:{event.seq}\n"
    return f"{event_id}event: {event.kind}\ndata: {event.model_dump_json()}\n\n".encode()
