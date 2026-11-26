"""Asset store: input uploads, listing, deletion, content and provenance sidecars."""

from __future__ import annotations

from fastapi import APIRouter, Path, Query, Request, Response
from fastapi.responses import FileResponse, JSONResponse
from starlette.concurrency import run_in_threadpool

from cinewright.api.deps import UPLOAD_MEDIA_TYPES, Core, binary_body, receive_upload, upload_media_type
from cinewright.api.problems import PROBLEM_RESPONSE
from cinewright.api.schemas import Asset, AssetList, DataClass, Provenance

router = APIRouter(tags=["assets"], responses={404: PROBLEM_RESPONSE, 422: PROBLEM_RESPONSE})

ShaPath = Path(pattern=r"^[0-9a-f]{64}$")
GENERIC_MEDIA_TYPE = "application/octet-stream"
# Asset bytes never run as a page in the UI's origin, even if a browser renders them as a document.
# Framing stays forbidden by the middleware's X-Frame-Options: DENY.
CONTENT_SECURITY_POLICY = "sandbox; default-src 'none'"


@router.post(
    "/assets",
    response_model=Asset,
    status_code=201,
    operation_id="uploadAsset",
    openapi_extra=binary_body("Input file (image, clip or audio) as raw bytes; Content-Type names the media type."),
)
async def upload_asset(request: Request, core: Core, data_class: DataClass = Query(default=DataClass.PRIVATE)) -> Asset:
    media_type = upload_media_type(request)
    async with receive_upload(request, core) as upload:
        # Placing the file and recording it must not block the event loop (SSE, leases, heartbeats).
        return await run_in_threadpool(core.add_asset, upload, media_type, data_class)


@router.get("/assets", response_model=AssetList, operation_id="listAssets")
def list_assets(core: Core, cursor: str | None = Query(default=None), limit: int = Query(default=50, ge=1, le=200)) -> AssetList:
    return core.list_assets(cursor, limit)


@router.get("/assets/{sha256}", response_model=Asset, operation_id="getAsset")
def get_asset(core: Core, sha256: str = ShaPath) -> Asset:
    return core.get_asset(sha256)


@router.delete("/assets/{sha256}", status_code=204, operation_id="deleteAsset", responses={409: PROBLEM_RESPONSE})
def delete_asset(core: Core, sha256: str = ShaPath) -> Response:
    """Delete an asset for good: its record, its file and the job output records of it. There is no recycle bin.

    Jobs that made or used it keep their state and events but no longer list it as an output; assets made from it keep
    their provenance, whose `parents` still name it. Refused with 409 while a job that has not ended (awaiting approval,
    queued or running) uses it as an input, and while its file stays open elsewhere, for example in a player.
    """
    core.delete_asset(sha256)
    return Response(status_code=204)


@router.get(
    "/assets/{sha256}/content",
    operation_id="getAssetContent",
    response_class=FileResponse,
    responses={200: {"description": "The asset's bytes", "content": {"application/octet-stream": {}}}},
)
def get_asset_content(core: Core, sha256: str = ShaPath) -> FileResponse:
    asset = core.get_asset(sha256)
    headers = {"Content-Security-Policy": CONTENT_SECURITY_POLICY}
    media_type = asset.media_type.lower()
    if media_type not in UPLOAD_MEDIA_TYPES:  # recorded before the allow-list existed, or by a bug: a download only
        media_type = GENERIC_MEDIA_TYPE
        headers["Content-Disposition"] = f'attachment; filename="{sha256[:12]}.bin"'
    return FileResponse(core.asset_path(sha256), media_type=media_type, headers=headers)


@router.get("/assets/{sha256}/provenance", response_model=Provenance, operation_id="getAssetProvenance")
def get_asset_provenance(core: Core, sha256: str = ShaPath) -> JSONResponse:
    asset = core.get_asset(sha256)
    body = asset.provenance.model_dump(mode="json", by_alias=True)
    headers = {"Content-Disposition": f'attachment; filename="{sha256[:12]}.provenance.json"'}
    return JSONResponse(body, headers=headers)
