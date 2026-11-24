"""Request dependencies shared by the routers."""

from __future__ import annotations

import asyncio
import logging
import threading
from collections.abc import AsyncIterator, Callable
from contextlib import aclosing, asynccontextmanager, suppress
from typing import Annotated

from fastapi import Depends, FastAPI, Header, Request
from starlette.concurrency import run_in_threadpool

from cinewright.core.assets import BlobWriter, StagedBlob
from cinewright.core.errors import CoreError, Invalid, Unauthorized
from cinewright.core.service import CoreService

log = logging.getLogger(__name__)

MAX_UPLOAD_BYTES = 2 * 1024**3  # 2 GiB per asset or worker output
# Uploads whose bodies are read at the same time; further uploads wait for a free slot. Waiting adds
# no response code, and uvicorn stops reading a waiting body after 64 KiB, so it costs no memory.
MAX_CONCURRENT_UPLOADS = 4
# A body that sends nothing for this long is refused, so stalled clients cannot keep the slots. It is
# below the worker client's 30 s socket timeout, so a worker waiting behind a stalled upload gets a slot.
UPLOAD_IDLE_SECONDS = 20.0
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


async def _body(request: Request) -> AsyncIterator[bytes]:
    """The body's chunks; refuses a body above the size limit or one that sends nothing for `UPLOAD_IDLE_SECONDS`."""
    chunks = request.stream()
    size = 0
    while True:
        try:
            async with asyncio.timeout(UPLOAD_IDLE_SECONDS):
                chunk = await anext(chunks)
        except StopAsyncIteration:
            return
        except TimeoutError:
            raise Invalid(f"Upload stalled: no data for {UPLOAD_IDLE_SECONDS:g} s.") from None
        size += len(chunk)
        if size > MAX_UPLOAD_BYTES:
            raise Invalid("Upload exceeds 2 GiB.")
        yield chunk


async def _drain(request: Request) -> None:
    """Read a refused body to its end without keeping it, so a client that sends it all before reading gets the answer."""
    with suppress(Invalid):  # stalled or oversized: the original refusal still answers
        async with aclosing(_body(request)) as body:
            async for _ in body:
                pass


@asynccontextmanager
async def receive_upload(request: Request, core: CoreService, check: Callable[[], None] | None = None) -> AsyncIterator[StagedBlob]:
    """Stream a raw body into a temporary file of the asset store, hashing it on the way; yield the complete file.

    The body never sits in memory as a whole, and every write, hash and fsync runs off the event loop.
    `check` runs once the upload has a slot, before anything is written to disk; when it refuses, the
    body is read and dropped first, so the client can read the refusal. A body that stalls for
    `UPLOAD_IDLE_SECONDS` is refused and gives up its slot. On every exit the temporary file is removed
    unless the core placed it: refused, oversized, stalled, empty or interrupted.
    """
    async with _upload_slots(request.app):
        if check is not None:
            try:
                await run_in_threadpool(check)
            except CoreError:
                await _drain(request)
                raise
        writer = await run_in_threadpool(core.stage_upload)
        try:
            pending: list[bytes] = []
            buffered = size = 0
            async with aclosing(_body(request)) as body:
                async for chunk in body:
                    pending.append(chunk)
                    size += len(chunk)
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
    """Remove an upload's temporary file exactly once, in a thread, also when the request is cancelled.

    This relies on the upload routes running without an anyio task group around them (the middleware is
    pure ASGI). A task group that keeps cancelling the request during shutdown could orphan the shielded
    task and keep an anyio worker thread alive at exit; revisit this before adding one.
    """
    claimed = threading.Lock()

    def discard_once() -> None:
        if claimed.acquire(blocking=False):  # the thread and the fallback below never both touch the file
            writer.discard()

    task = asyncio.ensure_future(run_in_threadpool(discard_once))
    try:
        await asyncio.shield(task)  # a cancelled request goes on cancelling while the thread finishes
    except asyncio.CancelledError:
        task.add_done_callback(lambda done: _after_cancelled_discard(done, discard_once, writer))
        raise


def _after_cancelled_discard(done: asyncio.Future, discard_once: Callable[[], None], writer: BlobWriter) -> None:
    try:
        if done.cancelled():
            discard_once()  # no thread took it, for example because the event loop is shutting down
        elif (error := done.exception()) is not None:
            raise error
    except OSError:
        log.warning("Could not remove the temporary upload file %s.", writer.path, exc_info=True)


def binary_body(description: str) -> dict:
    """OpenAPI requestBody for raw binary uploads."""
    return {
        "requestBody": {
            "required": True,
            "description": description,
            "content": {media: {"schema": {"type": "string", "format": "binary"}} for media in sorted(UPLOAD_MEDIA_TYPES)},
        }
    }
