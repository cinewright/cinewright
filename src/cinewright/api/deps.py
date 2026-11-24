"""Request dependencies shared by the routers."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator, Callable
from contextlib import asynccontextmanager
from typing import Annotated

from fastapi import Depends, FastAPI, Header, Request
from starlette.concurrency import run_in_threadpool

from cinewright.core.assets import BlobWriter, StagedBlob
from cinewright.core.errors import Invalid, Unauthorized
from cinewright.core.service import CoreService

MAX_UPLOAD_BYTES = 2 * 1024**3  # 2 GiB per asset or worker output
# Uploads whose bodies are read at the same time; further uploads wait for a free slot. Waiting adds
# no response code, and uvicorn stops reading a waiting body after 64 KiB, so it costs no memory.
MAX_CONCURRENT_UPLOADS = 4
WRITE_BUFFER_BYTES = 1024**2  # body bytes collected before one write to disk off the event loop

# Media types accepted for uploaded inputs and worker outputs. None of them is CORS-safelisted,
# so a browser page on another origin cannot send them without a preflight the core refuses.
UPLOAD_MEDIA_TYPES = frozenset(
    {
        "image/png",
        "image/jpeg",
        "image/webp",
        "video/mp4",
        "video/webm",
        "audio/wav",
        "audio/x-wav",
        "audio/mpeg",
        "audio/flac",
        "application/octet-stream",
    }
)


def get_core(request: Request) -> CoreService:
    return request.app.state.core


Core = Annotated[CoreService, Depends(get_core)]


def bearer_token(authorization: Annotated[str | None, Header()] = None) -> str:
    if not authorization or not authorization.startswith("Bearer "):
        raise Unauthorized("Missing bearer token.")
    token = authorization.removeprefix("Bearer ").strip()
    if not token:
        raise Unauthorized("Missing bearer token.")
    return token


def worker_id(core: Core, token: Annotated[str, Depends(bearer_token)]) -> str:
    return core.authenticate_worker(token)


WorkerId = Annotated[str, Depends(worker_id)]
BearerToken = Annotated[str, Depends(bearer_token)]


def upload_media_type(request: Request) -> str:
    """The media type of a raw binary upload, checked against the allow-list; refuses a declared size above the limit."""
    media_type = (request.headers.get("content-type") or "").split(";")[0].strip().lower()
    if media_type not in UPLOAD_MEDIA_TYPES:
        raise Invalid(f"Unsupported media type {media_type or 'none'}.")
    declared = request.headers.get("content-length")
    if declared and declared.isdigit() and int(declared) > MAX_UPLOAD_BYTES:
        raise Invalid("Upload exceeds 2 GiB.")
    return media_type


def _upload_slots(app: FastAPI) -> asyncio.Semaphore:
    # Created on first use, so it belongs to the event loop that serves the app.
    slots = getattr(app.state, "upload_slots", None)
    if slots is None:
        slots = app.state.upload_slots = asyncio.Semaphore(MAX_CONCURRENT_UPLOADS)
    return slots


@asynccontextmanager
async def receive_upload(request: Request, core: CoreService, check: Callable[[], None] | None = None) -> AsyncIterator[StagedBlob]:
    """Stream a raw body into a temporary file of the asset store, hashing it on the way; yield the complete file.

    The body never sits in memory as a whole, and every write, hash and fsync runs off the event loop.
    `check` runs once the upload has a slot, before anything is written to disk. On every exit the
    temporary file is removed unless the core placed it: refused, oversized, empty or interrupted.
    """
    async with _upload_slots(request.app):
        if check is not None:
            await run_in_threadpool(check)
        writer = await run_in_threadpool(core.stage_upload)
        try:
            pending: list[bytes] = []
            buffered = size = 0
            async for chunk in request.stream():
                size += len(chunk)
                if size > MAX_UPLOAD_BYTES:
                    raise Invalid("Upload exceeds 2 GiB.")
                pending.append(chunk)
                buffered += len(chunk)
                if buffered >= WRITE_BUFFER_BYTES:
                    await run_in_threadpool(writer.write, b"".join(pending))
                    pending.clear()
                    buffered = 0
            if size == 0:
                raise Invalid("Empty upload.")
            if pending:
                await run_in_threadpool(writer.write, b"".join(pending))
                pending.clear()
            yield await run_in_threadpool(writer.finish)
        finally:
            await _discard(writer)


async def _discard(writer: BlobWriter) -> None:
    try:
        await run_in_threadpool(writer.discard)
    except asyncio.CancelledError:
        writer.discard()  # cancelled before the thread ran: remove the file here rather than leave it behind
        raise


def binary_body(description: str) -> dict:
    """OpenAPI requestBody for raw binary uploads."""
    return {
        "requestBody": {
            "required": True,
            "description": description,
            "content": {media: {"schema": {"type": "string", "format": "binary"}} for media in sorted(UPLOAD_MEDIA_TYPES)},
        }
    }
