"""Estimates, jobs and the event stream (SSE): job events plus live system events."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from typing import Any, get_args

from fastapi import APIRouter, Header, Query, Request
from fastapi.openapi.utils import get_openapi
from fastapi.responses import StreamingResponse
from fastapi.routing import APIRoute

from cinewright.api.deps import Core
from cinewright.api.problems import PROBLEM_RESPONSE
from cinewright.api.schemas import (
    API_VERSION,
    ClientKind,
    Estimate,
    EventKind,
    Job,
    JobEvent,
    JobList,
    JobRequest,
    JobState,
    SystemEvent,
)

router = APIRouter(tags=["jobs"], responses={404: PROBLEM_RESPONSE, 409: PROBLEM_RESPONSE, 422: PROBLEM_RESPONSE, 501: PROBLEM_RESPONSE})

SSE_KEEPALIVE_SECONDS = 15
CLIENT_HEADER = "X-Cinewright-Client"
AGENT_HEADER = "X-Cinewright-Agent"
AGENT_PATTERN = r"^[A-Za-z0-9 ._-]{1,64}$"
# The `data:` object of one SSE message; `stream_event_schemas` registers both models as OpenAPI components.
StreamEvent = JobEvent | SystemEvent
STREAM_DESCRIPTION = (
    "Server-sent events. Each message has an `event:` line with the event's kind and a `data:` line with one JSON "
    "object: a JobEvent for the job event kinds (`state`, `progress`, `preview`, `log`), a SystemEvent for the other "
    "kinds (`engine`: engine provisioning, `model`: model install progress, installs, removals and evictions). "
    "Stored job events have an `id:` line (`<job_id>:<seq>`). A stream first replays the stored events after "
    "`Last-Event-ID`; with `job_id` and without `Last-Event-ID`, the job's whole history. Previews and system events "
    "have no id and are never replayed: after its replay a stream sends the latest preview of each running job it "
    "covers once, and system events only while it is open. With `job_id` the stream carries only that job's events "
    "and ends after its final state; only a stream without `job_id` carries system events. A comment line "
    "(`: keepalive`) follows after 15 s without an event."
)


def stream_event_schemas() -> dict[str, Any]:
    """OpenAPI component schemas of the stream's `data:` objects and of every schema they refer to.

    FastAPI registers a model as a component only when a route takes or returns it as JSON, and naming it as the 200
    response model would describe the stream as `application/json`. So the schemas come from a throwaway document of
    one route that returns them, which FastAPI renders like every other component; the route is never served.
    """
    route = APIRoute("/", _schema_only, response_model=StreamEvent)
    return get_openapi(title="stream events", version=API_VERSION, routes=[route])["components"]["schemas"]


def _schema_only() -> None:
    """Endpoint of the throwaway route in `stream_event_schemas`; never served."""


def sse_frame(event: JobEvent | SystemEvent) -> bytes:
    """One SSE message named after the event's kind. Only stored job events carry an id (`<job_id>:<seq>`, which the
    browser resends as Last-Event-ID for replay). A preview (live only, never stored) and a system event have none,
    so the browser keeps the last stored event's id and Last-Event-ID always names a stored event."""
    payload = event.model_dump_json()
    if isinstance(event, JobEvent) and event.kind != EventKind.PREVIEW:
        return f"id: {event.job_id}:{event.seq}\nevent: {event.kind}\ndata: {payload}\n\n".encode()
    return f"event: {event.kind}\ndata: {payload}\n\n".encode()


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
    responses={
        200: {
            "description": STREAM_DESCRIPTION,
            "content": {
                "text/event-stream": {
                    "schema": {"oneOf": [{"$ref": f"#/components/schemas/{model.__name__}"} for model in get_args(StreamEvent)]}
                }
            },
        }
    },
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
        pending: asyncio.Task[JobEvent | SystemEvent | None] | None = None
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
                yield sse_frame(event)
        finally:
            if pending is not None and not pending.done():
                pending.cancel()  # throws into the generator, whose `finally` unsubscribes
                await asyncio.wait({pending})
            await events.aclose()

    return StreamingResponse(body(), media_type="text/event-stream", headers={"Cache-Control": "no-store"})


async def _next_event(events: AsyncIterator[JobEvent | SystemEvent]) -> JobEvent | SystemEvent | None:
    return await anext(events, None)
