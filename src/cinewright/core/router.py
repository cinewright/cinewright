"""Router: map a `JobRequest` to model, quantization and host.

Pure functions over snapshots the core passes in. Check order for every candidate
(model, quantization, host):

1. policy: the host accepts the job's data class (biometric data never leaves local hosts), and
   the model licence has no geo exclusion for the configured country (`EU` covers EU members)
2. feasibility: capability, pinned quantization, model limits (resolution; for a clip also frames, on the
   model's grid, fps and duration, which a still image skips), the quantization installed on the host (only
   for hosts whose installed quantizations the core passes in: the local model store), an online worker on
   the host that serves the engine model, calibration not `wont_run`
3. cost: local hosts cost 0 EUR; other hosts have no price yet, so their cost is unknown
4. runtime: `seconds_per_step * steps` when calibrated, else unknown

The first failing check is the candidate's reason. Feasible candidates are ranked by cost, the
model's registry `preference` (higher first), calibration status and runtime, then by registry order.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence, Set
from dataclasses import dataclass, field
from typing import Any

from cinewright.api.schemas import (
    CalibrationResult,
    CalibrationStatus,
    Capability,
    DataClass,
    HostKind,
    JobParams,
    JobRequest,
    ModelDescriptor,
    Route,
    Settings,
)

EU_COUNTRIES = frozenset("AT BE BG CY CZ DE DK EE ES FI FR GR HR HU IE IT LT LU LV MT NL PL PT RO SE SI SK".split())
_DATA_CLASS_RANK = {DataClass.PUBLIC: 0, DataClass.PRIVATE: 1, DataClass.BIOMETRIC: 2}
_STATUS_RANK = {CalibrationStatus.RUNS: 0, CalibrationStatus.SLOW: 1, None: 2}
# Jobs that return one still image. Wan2GP's image mode renders the model's shortest clip and keeps its first frame, so
# frames, fps and duration do not apply: neither the limit checks nor provenance (`core/assets.py`) use them.
IMAGE_CAPABILITIES = frozenset({Capability.T2I, Capability.I2I})


@dataclass(frozen=True)
class HostInfo:
    id: str
    name: str
    kind: HostKind
    allowed_data_classes: frozenset[DataClass]


@dataclass(frozen=True)
class WorkerInfo:
    id: str
    host_id: str
    engine: str
    models: frozenset[str]


@dataclass(frozen=True)
class Plan:
    route: Route | None
    reasons: list[str] = field(default_factory=list)
    eta_seconds: float | None = None
    cost_eur: float | None = None
    requires_approval: bool = False

    @property
    def feasible(self) -> bool:
        return self.route is not None


def strictest(classes: Sequence[DataClass]) -> DataClass:
    return max(classes, key=_DATA_CLASS_RANK.__getitem__)


def geo_excluded(exclusions: Sequence[str], country: str) -> bool:
    return country in exclusions or ("EU" in exclusions and country in EU_COUNTRIES)


def host_accepts(host: HostInfo, data_class: DataClass) -> bool:
    if data_class == DataClass.BIOMETRIC and host.kind != HostKind.LOCAL:
        return False
    return data_class in host.allowed_data_classes


def effective_params(request: JobRequest, model: ModelDescriptor, seed: int | None = None) -> JobParams:
    """Request parameters with the model's registry defaults filled in where the request left them open."""
    params = request.params.model_dump(exclude={"extra"})
    known = set(JobParams.model_fields) - {"extra", "seed"}
    extra: dict[str, Any] = {}
    for key, value in model.defaults.items():
        if key in known:
            if params.get(key) is None:
                params[key] = value
        elif key != "seed":
            extra[key] = value
    extra.update(request.params.extra)
    if seed is not None:
        params["seed"] = seed
    return JobParams.model_validate({**params, "extra": extra})


def _limit_problem(params: JobParams, model: ModelDescriptor, capability: Capability) -> str | None:
    limits = model.limits
    if params.resolution and limits.resolutions and params.resolution not in limits.resolutions:
        return f"{model.id} does not support {params.resolution} (supported: {', '.join(limits.resolutions)})"
    if capability in IMAGE_CAPABILITIES:  # the engine ignores frames, fps and duration for a still image
        return None
    if params.frames and limits.max_frames and params.frames > limits.max_frames:
        return f"{model.id} renders at most {limits.max_frames} frames, {params.frames} requested"
    if params.frames:
        problem = _frame_grid_problem(params.frames, model)
        if problem:
            return problem
    if params.fps and limits.fps and params.fps not in limits.fps:
        return f"{model.id} does not support {params.fps} fps"
    if params.duration_s and limits.max_duration_s and params.duration_s > limits.max_duration_s:
        return f"{model.id} renders at most {limits.max_duration_s:g} s, {params.duration_s:g} s requested"
    return None


def _frame_grid_problem(frames: int, model: ModelDescriptor) -> str | None:
    """Why a clip of `frames` frames is off the model's grid (`min_frames + n * frames_step`); None when it is on it."""
    first, step = model.limits.min_frames, model.limits.frames_step
    if first is None or step is None:
        return None
    if frames < first:
        return f"{model.id} renders at least {first} frames, {frames} requested"
    if (frames - first) % step == 0:
        return None
    lower = frames - (frames - first) % step
    nearest = [lower, lower + step] if model.limits.max_frames is None or lower + step <= model.limits.max_frames else [lower]
    return f"{model.id} renders {first} + a multiple of {step} frames, {frames} requested (nearest: {' or '.join(str(n) for n in nearest)})"


def _calibration(
    calibration: Sequence[CalibrationResult], host_id: str, model_id: str, quant: str, resolution: str | None
) -> tuple[CalibrationResult | None, bool]:
    """Best matching measurement and whether every matching measurement says `wont_run`."""
    rows = [
        r
        for r in calibration
        if r.host_id == host_id and r.model_id == model_id and r.quant == quant and (resolution is None or r.resolution == resolution)
    ]
    usable = [r for r in rows if r.status != CalibrationStatus.WONT_RUN]
    if rows and not usable:
        return rows[0], True
    best = min(usable, key=lambda r: _STATUS_RANK[r.status], default=None)
    return best, False


@dataclass(frozen=True)
class _Candidate:
    route: Route
    cost: float | None
    preference: int
    status: CalibrationStatus | None
    eta: float | None
    order: tuple[int, int, int]


def plan(
    request: JobRequest,
    data_class: DataClass,
    models: Sequence[ModelDescriptor],
    hosts: Sequence[HostInfo],
    workers: Sequence[WorkerInfo],
    calibration: Sequence[CalibrationResult],
    settings: Settings,
    *,
    installed: Mapping[str, Set[tuple[str, str]]],
) -> Plan:
    """Choose a route for `request`; `data_class` is the strictest class of the request and its inputs.

    `installed` maps a host id to the (model id, quantization) pairs installed on it. A host it does not name is
    not checked: the core knows only the local model store (remote hosts follow with #17 and #18).
    """
    if request.model is not None:
        candidates_models = [m for m in models if m.id == request.model]
        if not candidates_models:
            return Plan(None, [f"feasibility: model {request.model} is not in the registry"])
    else:
        candidates_models = [m for m in models if request.capability in m.capabilities]
        if not candidates_models:
            return Plan(None, [f"feasibility: no registry model offers {request.capability}"])

    reasons: list[str] = []
    accepted_hosts = []
    for index, host in enumerate(hosts):
        if host_accepts(host, data_class):
            accepted_hosts.append((index, host))
        elif data_class == DataClass.BIOMETRIC and host.kind != HostKind.LOCAL:
            reasons.append(f"policy: biometric data stays on local hosts, so {host.id} is excluded")
        else:
            reasons.append(f"policy: host {host.id} does not accept {data_class} data")
    if not accepted_hosts:
        reasons.append(f"policy: no host accepts {data_class} data")
        return Plan(None, _unique(reasons))

    feasible: list[_Candidate] = []
    for model_index, model in enumerate(candidates_models):
        if geo_excluded(model.license.geo_exclusions, settings.country):
            reasons.append(f"policy: the licence of {model.id} ({model.license.name}) excludes {settings.country}")
            continue
        if request.capability not in model.capabilities:
            reasons.append(f"feasibility: {model.id} does not offer {request.capability}")
            continue
        quants = [q for q in model.quantizations if request.quant is None or q.id == request.quant]
        if not quants:
            reasons.append(f"feasibility: {model.id} has no quantization {request.quant}")
            continue
        params = effective_params(request, model)
        problem = _limit_problem(params, model, request.capability)
        if problem:
            reasons.append(f"feasibility: {problem}")
            continue
        for quant_index, quant in enumerate(model.quantizations):
            if quant not in quants:
                continue
            bindings = [b for b in model.engines if b.quant == quant.id]
            if not bindings:
                reasons.append(f"feasibility: no engine binding for {model.id} {quant.id}")
                continue
            for host_index, host in accepted_hosts:
                host_installed = installed.get(host.id)
                if host_installed is not None and (model.id, quant.id) not in host_installed:
                    reasons.append(f"feasibility: {model.id} {quant.id} is not installed on {host.id}")
                    continue
                binding = next(
                    (
                        b
                        for b in bindings
                        if any(w.host_id == host.id and w.engine == b.engine and b.model_type in w.models for w in workers)
                    ),
                    None,
                )
                if binding is None:
                    served = ", ".join(f"{b.engine} {b.model_type}" for b in bindings)
                    reasons.append(f"feasibility: no online worker on {host.id} serves {served}")
                    continue
                measured, wont_run = _calibration(calibration, host.id, model.id, quant.id, params.resolution)
                if wont_run and measured is not None:
                    detail = f" ({measured.detail})" if measured.detail else ""
                    reasons.append(f"feasibility: {model.id} {quant.id} does not run on {host.id}{detail}")
                    continue
                cost = 0.0 if host.kind == HostKind.LOCAL else None
                if request.budget_eur is not None and (cost is None or cost > request.budget_eur):
                    reasons.append(f"cost: {host.id} has no price within the budget of €{request.budget_eur:.2f}")
                    continue
                status = measured.status if measured else None
                eta = measured.seconds_per_step * params.steps if measured and measured.seconds_per_step and params.steps else None
                route_reasons = [
                    f"policy: host {host.id} accepts {data_class} data",
                    f"policy: the licence of {model.id} allows {settings.country}",
                    *([f"feasibility: {model.id} {quant.id} is installed on {host.id}"] if host_installed is not None else []),
                    f"feasibility: a worker on {host.id} serves {binding.engine} {binding.model_type}",
                    f"feasibility: calibration on {host.id}: {status or 'not measured'}",
                    f"cost: €{cost:.2f} on {host.id}" if cost is not None else f"cost: unknown on {host.id}",
                    f"runtime: {eta:.0f} s" if eta is not None else "runtime: unknown until calibrated",
                ]
                route = Route(model_id=model.id, quant=quant.id, host_id=host.id, engine=binding.engine, reasons=route_reasons)
                feasible.append(_Candidate(route, cost, model.preference, status, eta, (model_index, quant_index, host_index)))

    if not feasible:
        return Plan(None, _unique(reasons))
    best = min(
        feasible,
        key=lambda c: (
            math.inf if c.cost is None else c.cost,
            -c.preference,
            _STATUS_RANK[c.status],
            math.inf if c.eta is None else c.eta,
            c.order,
        ),
    )
    host = next(h for h in hosts if h.id == best.route.host_id)
    requires_approval = host.kind != HostKind.LOCAL and (best.cost is None or best.cost > settings.approval_threshold_eur)
    return Plan(best.route, list(best.route.reasons), best.eta, best.cost, requires_approval)


def _unique(items: list[str]) -> list[str]:
    return list(dict.fromkeys(items))
