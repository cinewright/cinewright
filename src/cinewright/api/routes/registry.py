"""Hardware, capability registry, models and calibration."""

from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, Path, Query, Response
from pydantic import Field

from cinewright.api.deps import Core
from cinewright.api.problems import PROBLEM_MEDIA_TYPE, PROBLEM_RESPONSE
from cinewright.api.schemas import CalibrationReport, CapabilityList, HardwareProfile, InstallRefusal, ModelDetail, ModelList

router = APIRouter(responses={404: PROBLEM_RESPONSE, 422: PROBLEM_RESPONSE, 501: PROBLEM_RESPONSE})

ModelIdPath = Path(pattern=r"^[a-z0-9]+(?:-[a-z0-9.]+)*$", max_length=80)
# `<model id>:<quantization>`, split at the first colon: model ids have none, quantization ids have no pattern. Only the
# colon is checked, so every quantization an `eviction_needed` refusal lists can be sent back.
EvictValue = Annotated[str, Field(pattern=r"^[^:]*:.*$")]
INSTALL_REFUSAL = {"model": InstallRefusal, "content": {PROBLEM_MEDIA_TYPE: {}}}


@router.get("/hardware", response_model=HardwareProfile, tags=["hardware"], operation_id="getHardware")
def get_hardware(core: Core) -> HardwareProfile:
    return core.hardware()


@router.post("/hardware/detect", response_model=HardwareProfile, tags=["hardware"], operation_id="detectHardware")
def detect_hardware(core: Core) -> HardwareProfile:
    return core.detect_hardware()


@router.get("/capabilities", response_model=CapabilityList, tags=["registry"], operation_id="listCapabilities")
def list_capabilities(core: Core) -> CapabilityList:
    return core.capabilities()


@router.get("/models", response_model=ModelList, tags=["registry"], operation_id="listModels")
def list_models(core: Core) -> ModelList:
    return core.list_models()


@router.get("/models/{model_id}", response_model=ModelDetail, tags=["registry"], operation_id="getModel")
def get_model(core: Core, model_id: str = ModelIdPath) -> ModelDetail:
    return core.get_model(model_id)


@router.post(
    "/models/{model_id}/install", status_code=202, tags=["registry"], operation_id="installModel", responses={409: INSTALL_REFUSAL}
)
def install_model(
    core: Core,
    model_id: str = ModelIdPath,
    quant: str | None = Query(default=None, description="Quantization to install (required)"),
    evict: list[EvictValue] = Query(
        default=[],
        description="An installed quantization you agree to remove before the download, written `<model_id>:<quant>`; repeat it "
        "for each one that an `eviction_needed` refusal listed",
    ),
) -> Response:
    """Install a quantization with the shared files it needs, in the background.

    Checks first and refuses with 409 when the model cannot be installed now: free disk space, the storage quota, an
    install in progress, the licence's country exclusions. With a quota, the least recently used quantizations no job
    needs are removed once the install is registered, so a failed download removes nothing. When the drive holds the
    download only after removing them, the install is refused with `eviction_needed`, which lists them with their sizes;
    sent again with each of them in `evict`, it removes them first, and they stay removed if the download fails.
    Progress and the result follow as `model` events on `GET /events`; `GET /models` holds the state.
    """
    keys = [(evict_model, evict_quant) for evict_model, _, evict_quant in (value.partition(":") for value in evict)]
    core.install_model(model_id, quant, evict=keys)
    return Response(status_code=202)


@router.delete("/models/{model_id}", status_code=204, tags=["registry"], operation_id="removeModel", responses={409: PROBLEM_RESPONSE})
def remove_model(
    core: Core, model_id: str = ModelIdPath, quant: str | None = Query(default=None, description="Quantization to remove; all when omitted")
) -> Response:
    """Remove installed quantizations; shared files stay while another installed model or a waiting job needs them.

    Refused with 409 while the quantization is being installed or a queued or running job needs it.
    """
    core.remove_model(model_id, quant)
    return Response(status_code=204)


@router.get("/calibration", response_model=CalibrationReport, tags=["calibration"], operation_id="getCalibration")
def get_calibration(core: Core) -> CalibrationReport:
    return core.calibration()


@router.post("/calibration/runs", status_code=202, tags=["calibration"], operation_id="startCalibration")
def start_calibration(core: Core) -> Response:
    core.start_calibration()
    return Response(status_code=202)
