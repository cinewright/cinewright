"""Estimates, jobs and the job event stream (SSE)."""

from __future__ import annotations

from collections.abc import AsyncIterator

from fastapi import APIRouter, Header, Query, Request
from fastapi.responses import StreamingResponse

from cinewright.api.deps import Core
from cinewright.api.problems import PROBLEM_RESPONSE
from cinewright.api.schemas import ClientKind, Estimate, Job, JobList, JobRequest, JobState

router = APIRouter(tags=["jobs"], responses={404: PROBLEM_RESPONSE, 409: PROBLEM_RESPONSE, 422: PROBLEM_RESPONSE, 501: PROBLEM_RESPONSE})

SSE_KEEPALIVE_SECONDS = 15


@router.post("/estimate", response_model=Estimate, operation_id="estimate")
def estimate(request: JobRequest, core: Core) -> Estimate:
    return core.estimate(request)


@router.post("/jobs", response_model=Job, status_code=202, operation_id="submitJob")
def submit_job(request: JobRequest, core: Core) -> Job:
    return core.submit_job(request, ClientKind.UI, None)


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
        async for event in core.events(job_id, last_event_id):
            if await request.is_disconnected():
                break
            payload = event.model_dump_json()
            yield f"id: {event.job_id}:{event.seq}\nevent: {event.kind}\ndata: {payload}\n\n".encode()

    return StreamingResponse(body(), media_type="text/event-stream", headers={"Cache-Control": "no-store"})
