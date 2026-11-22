"""Asset store: input uploads, listing, content and provenance sidecars."""

from __future__ import annotations

from fastapi import APIRouter, Path, Query, Request
from fastapi.responses import FileResponse, JSONResponse

from cinewright.api.deps import Core, binary_body, read_upload
from cinewright.api.problems import PROBLEM_RESPONSE
from cinewright.api.schemas import Asset, AssetList, DataClass, Provenance

router = APIRouter(tags=["assets"], responses={404: PROBLEM_RESPONSE, 422: PROBLEM_RESPONSE})

ShaPath = Path(pattern=r"^[0-9a-f]{64}$")


@router.post(
    "/assets",
    response_model=Asset,
    status_code=201,
    operation_id="uploadAsset",
    openapi_extra=binary_body("Input file (image, clip or audio) as raw bytes; Content-Type names the media type."),
)
async def upload_asset(request: Request, core: Core, data_class: DataClass = Query(default=DataClass.PRIVATE)) -> Asset:
    data, media_type = await read_upload(request)
    return core.add_asset(data, media_type, data_class)


@router.get("/assets", response_model=AssetList, operation_id="listAssets")
def list_assets(core: Core, cursor: str | None = Query(default=None), limit: int = Query(default=50, ge=1, le=200)) -> AssetList:
    return core.list_assets(cursor, limit)


@router.get("/assets/{sha256}", response_model=Asset, operation_id="getAsset")
def get_asset(core: Core, sha256: str = ShaPath) -> Asset:
    return core.get_asset(sha256)


@router.get(
    "/assets/{sha256}/content",
    operation_id="getAssetContent",
    response_class=FileResponse,
    responses={200: {"description": "The asset's bytes", "content": {"application/octet-stream": {}}}},
)
def get_asset_content(core: Core, sha256: str = ShaPath) -> FileResponse:
    asset = core.get_asset(sha256)
    return FileResponse(core.asset_path(sha256), media_type=asset.media_type)


@router.get("/assets/{sha256}/provenance", response_model=Provenance, operation_id="getAssetProvenance")
def get_asset_provenance(core: Core, sha256: str = ShaPath) -> JSONResponse:
    asset = core.get_asset(sha256)
    body = asset.provenance.model_dump(mode="json", by_alias=True)
    headers = {"Content-Disposition": f'attachment; filename="{sha256[:12]}.provenance.json"'}
    return JSONResponse(body, headers=headers)
