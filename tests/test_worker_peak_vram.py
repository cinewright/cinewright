"""Peak VRAM per job: the worker loop reads it from the torch its engine imported and posts it as a `log` event before
it completes or fails the job (the core refuses events for a closed job).

The engine is the mock adapter, which never touches torch; a fake `torch` module in `sys.modules` stands in for the one
Wan2GP imports. The fake core of `test_worker_loop` validates every request against the contract.
"""

from __future__ import annotations

import json
import sys
import types

import pytest
from test_worker_loop import FakeCore, make_job, start_loop, wait_closed

from cinewright_worker.engines.mock import MockAdapter

MIB = 1024 * 1024


class OrderedCore(FakeCore):
    """Also records, in order, the worker's event posts and its complete or fail call."""

    def __init__(self) -> None:
        super().__init__()
        self.calls: list[tuple[str, list[dict]]] = []

    def route(self, request, method, path, body):
        result = super().route(request, method, path, body)
        action = path.rsplit("/", 1)[-1]
        if method == "POST" and action in ("events", "complete", "fail"):
            self.calls.append((action, json.loads(body).get("events", []) if action == "events" else []))
        return result

    def peak_events(self) -> list[tuple[int, dict]]:
        """(position in `calls`, data) of every posted event with `peak_vram_mb`."""
        calls = enumerate(self.calls)
        return [(index, event["data"]) for index, (_, events) in calls for event in events if "peak_vram_mb" in event["data"]]

    def close_index(self) -> int:
        return next(index for index, (action, _) in enumerate(self.calls) if action in ("complete", "fail"))


class FakeCuda:
    def __init__(self, *, initialised: bool = True, peak_bytes: int = 12_345 * MIB) -> None:
        self.initialised = initialised
        self.peak_bytes = peak_bytes
        self.resets = 0
        self.readings = 0

    def is_initialized(self) -> bool:
        return self.initialised

    def reset_peak_memory_stats(self) -> None:
        assert self.initialised, "the worker touched CUDA before the engine initialised it"
        self.resets += 1

    def max_memory_allocated(self) -> int:
        assert self.initialised, "the worker touched CUDA before the engine initialised it"
        self.readings += 1
        return self.peak_bytes


class CudaStartingAdapter(MockAdapter):
    """Initialises (fake) CUDA when it starts the job, as Wan2GP does on its first generation."""

    def __init__(self, cuda: FakeCuda, **options) -> None:
        super().__init__(**options)
        self.cuda = cuda

    def submit(self, job):
        self.cuda.initialised = True
        return super().submit(job)


@pytest.fixture
def core():
    fake = OrderedCore()
    yield fake
    fake.shutdown()
    assert fake.violations == []


def use_torch(monkeypatch: pytest.MonkeyPatch, cuda: FakeCuda) -> None:
    torch = types.ModuleType("torch")
    torch.cuda = cuda
    monkeypatch.setitem(sys.modules, "torch", torch)


def run_one_job(core: OrderedCore, tmp_path, adapter: MockAdapter | None = None, **job) -> None:
    core.queue.append(make_job(**job))
    _, thread, result = start_loop(core, tmp_path, adapter=adapter, once=True)
    wait_closed(core, thread)
    assert result == [0]


def test_peak_vram_is_posted_before_the_job_completes(core, tmp_path, monkeypatch):
    cuda = FakeCuda()
    use_torch(monkeypatch, cuda)

    run_one_job(core, tmp_path, params={"steps": 3})

    assert len(core.completed) == 1 and core.failed == []
    [(index, data)] = core.peak_events()
    assert data["peak_vram_mb"] == 12_345 and data["message"]
    assert index < core.close_index()
    assert (cuda.resets, cuda.readings) == (1, 1)


def test_peak_vram_is_posted_before_the_job_fails(core, tmp_path, monkeypatch):
    use_torch(monkeypatch, FakeCuda(peak_bytes=15_000 * MIB + 7))

    run_one_job(core, tmp_path, params={"steps": 3, "extra": {"mock_fail": "oom"}})

    assert core.completed == [] and core.failed[0].error.code == "oom"
    [(index, data)] = core.peak_events()
    assert data["peak_vram_mb"] == 15_000
    assert index < core.close_index()


def test_peak_vram_counts_from_cuda_initialised_during_the_job(core, tmp_path, monkeypatch):
    # Before the job CUDA is not initialised, so nothing is reset; its statistics start within the job.
    cuda = FakeCuda(initialised=False)
    use_torch(monkeypatch, cuda)

    run_one_job(core, tmp_path, adapter=CudaStartingAdapter(cuda, step_seconds=0.002), params={"steps": 3})

    [(index, data)] = core.peak_events()
    assert data["peak_vram_mb"] == 12_345 and index < core.close_index()
    assert (cuda.resets, cuda.readings) == (0, 1)


def test_no_peak_vram_event_without_torch(core, tmp_path, monkeypatch):
    monkeypatch.delitem(sys.modules, "torch", raising=False)

    run_one_job(core, tmp_path, params={"steps": 3})

    assert len(core.completed) == 1
    assert core.peak_events() == []
    assert "torch" not in sys.modules  # the worker never imports it


def test_no_peak_vram_event_while_cuda_is_not_initialised(core, tmp_path, monkeypatch):
    cuda = FakeCuda(initialised=False)
    use_torch(monkeypatch, cuda)

    run_one_job(core, tmp_path, params={"steps": 3})

    assert len(core.completed) == 1 and core.peak_events() == []
    assert (cuda.resets, cuda.readings) == (0, 0)


def test_a_failing_measurement_never_keeps_the_job_from_closing(core, tmp_path, monkeypatch):
    def broken() -> int:
        raise RuntimeError("CUDA error: unspecified launch failure")

    cuda = FakeCuda()
    cuda.max_memory_allocated = broken
    use_torch(monkeypatch, cuda)

    run_one_job(core, tmp_path, params={"steps": 3})

    assert len(core.completed) == 1 and core.peak_events() == []
