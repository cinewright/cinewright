"""Worker protocol: registration, heartbeat, long-poll leases, events, outputs, completion."""

from __future__ import annotations

from fastapi import APIRouter, Path, Request, Response
from fastapi.responses import FileResponse
from starlette.concurrency import run_in_threadpool

from cinewright.api.deps import BearerToken, Core, WorkerId, binary_body, receive_upload, upload_media_type
from cinewright.api.problems import PROBLEM_RESPONSE
from cinewright.api.schemas import (
    Heartbeat,
    HeartbeatAck,
    JobCompletion,
    JobFailure,
    Lease,
    LeaseRequest,
    OutputUploaded,
    WorkerEvents,
    WorkerEventsAck,
    WorkerRegistered,
    WorkerRegistration,
)

router = APIRouter(
    prefix="/worker",
    tags=["worker"],
    responses={401: PROBLEM_RESPONSE, 404: PROBLEM_RESPONSE, 409: PROBLEM_RESPONSE, 422: PROBLEM_RESPONSE},
)

OutputName = Path(pattern=r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")


@router.post("/register", response_model=WorkerRegistered, status_code=201, operation_id="registerWorker")
def register(registration: WorkerRegistration, core: Core, enrollment_token: BearerToken) -> WorkerRegistered:
    return core.register_worker(registration, enrollment_token)


@router.post("/heartbeat", response_model=HeartbeatAck, operation_id="workerHeartbeat")
def heartbeat(heartbeat: Heartbeat, core: Core, worker_id: WorkerId) -> HeartbeatAck:
    return core.heartbeat(worker_id, heartbeat)


@router.post(
    "/lease",
    response_model=Lease,
    operation_id="leaseJob",
    responses={204: {"description": "No job available within wait_seconds"}},
)
async def lease(request: LeaseRequest, http: Request, core: Core, worker_id: WorkerId) -> Lease | Response:
    # A long-poll whose worker went away (e.g. a killed worker process) must not take a job it can never receive.
    granted = await core.lease(worker_id, request, gone=http.is_disconnected)
    return granted if granted is not None else Response(status_code=204)


@router.get(
    "/inputs/{sha256}",
    operation_id="downloadWorkerInput",
    response_class=FileResponse,
    responses={200: {"description": "Input asset bytes", "content": {"application/octet-stream": {}}}},
)
def download_input(core: Core, worker_id: WorkerId, sha256: str = Path(pattern=r"^[0-9a-f]{64}$")) -> FileResponse:
    return FileResponse(core.worker_input_path(worker_id, sha256))


@router.post("/jobs/{job_id}/events", response_model=WorkerEventsAck, operation_id="postWorkerEvents")
def post_events(job_id: str, events: WorkerEvents, core: Core, worker_id: WorkerId) -> WorkerEventsAck:
    return core.worker_events(worker_id, job_id, events)


@router.put(
    "/jobs/{job_id}/outputs/{name}",
    response_model=OutputUploaded,
    status_code=201,
    operation_id="uploadWorkerOutput",
    openapi_extra=binary_body("Output file as raw bytes; Content-Type names the media type."),
)
async def upload_output(request: Request, job_id: str, core: Core, worker_id: WorkerId, name: str = OutputName) -> OutputUploaded:
    media_type = upload_media_type(request)
    # The job check runs before anything is written to disk; store_output repeats it before placing the file.
    async with receive_upload(request, core, check=lambda: core.check_output(worker_id, job_id)) as upload:
        # Placing the file and recording it must not block the event loop (SSE, leases, heartbeats).
        return await run_in_threadpool(core.store_output, worker_id, job_id, name, media_type, upload)


@router.post("/jobs/{job_id}/complete", status_code=204, operation_id="completeJob")
def complete(job_id: str, completion: JobCompletion, core: Core, worker_id: WorkerId) -> Response:
    core.complete_job(worker_id, job_id, completion)
    return Response(status_code=204)


@router.post("/jobs/{job_id}/fail", status_code=204, operation_id="failJob")
def fail(job_id: str, failure: JobFailure, core: Core, worker_id: WorkerId) -> Response:
    core.fail_job(worker_id, job_id, failure)
    return Response(status_code=204)
