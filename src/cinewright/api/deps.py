"""Request dependencies shared by the routers."""

from __future__ import annotations

from typing import Annotated

from fastapi import Depends, Header, Request

from cinewright.core.errors import Invalid, Unauthorized
from cinewright.core.service import CoreService

MAX_UPLOAD_BYTES = 2 * 1024**3  # 2 GiB per asset or worker output

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


async def read_upload(request: Request) -> tuple[bytes, str]:
    """Read a raw binary body and its media type, enforcing the type list and size limit."""
    media_type = (request.headers.get("content-type") or "").split(";")[0].strip().lower()
    if media_type not in UPLOAD_MEDIA_TYPES:
        raise Invalid(f"Unsupported media type {media_type or 'none'}.")
    declared = request.headers.get("content-length")
    if declared and declared.isdigit() and int(declared) > MAX_UPLOAD_BYTES:
        raise Invalid("Upload exceeds 2 GiB.")
    chunks: list[bytes] = []
    size = 0
    async for chunk in request.stream():
        size += len(chunk)
        if size > MAX_UPLOAD_BYTES:
            raise Invalid("Upload exceeds 2 GiB.")
        chunks.append(chunk)
    if size == 0:
        raise Invalid("Empty upload.")
    return b"".join(chunks), media_type


def binary_body(description: str) -> dict:
    """OpenAPI requestBody for raw binary uploads."""
    return {
        "requestBody": {
            "required": True,
            "description": description,
            "content": {media: {"schema": {"type": "string", "format": "binary"}} for media in sorted(UPLOAD_MEDIA_TYPES)},
        }
    }
