"""Hardware, capability registry, models and calibration."""

from __future__ import annotations

from fastapi import APIRouter, Path, Query, Response

from cinewright.api.deps import Core
from cinewright.api.problems import PROBLEM_RESPONSE
from cinewright.api.schemas import CalibrationReport, CapabilityList, HardwareProfile, ModelDetail, ModelList

router = APIRouter(responses={404: PROBLEM_RESPONSE, 422: PROBLEM_RESPONSE, 501: PROBLEM_RESPONSE})

ModelIdPath = Path(pattern=r"^[a-z0-9]+(?:-[a-z0-9.]+)*$", max_length=80)


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
    "/models/{model_id}/install", status_code=202, tags=["registry"], operation_id="installModel", responses={409: PROBLEM_RESPONSE}
)
def install_model(
    core: Core, model_id: str = ModelIdPath, quant: str | None = Query(default=None, description="Quantization to install (required)")
) -> Response:
    """Install a quantization with the shared files it needs, in the background.

    Checks first and refuses with 409 when the model cannot be installed now: free disk space, the storage quota
    (it may remove the least recently used models first), an install in progress, the licence's country exclusions.
    Progress and the result follow as `model` events on `GET /events`; `GET /models` holds the state.
    """
    core.install_model(model_id, quant)
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
