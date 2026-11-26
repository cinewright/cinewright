"""The `gpu` marker: such tests run only on a computer with a GPU and a provisioned engine, never in CI."""

from __future__ import annotations

import test_wan2gp_adapter
from conftest import gpu_skip_reason

ENGINE = {"CINEWRIGHT_WAN2GP_ROOT": "C:/wan2gp", "CINEWRIGHT_WAN2GP_PYTHON": "C:/wan2gp/env/Scripts/python.exe"}


def test_gpu_tests_need_the_switch_and_a_provisioned_engine():
    assert gpu_skip_reason({}) is not None  # CI
    assert gpu_skip_reason(ENGINE) is not None
    assert gpu_skip_reason({"CINEWRIGHT_GPU_TESTS": "0", **ENGINE}) is not None
    assert gpu_skip_reason({"CINEWRIGHT_GPU_TESTS": "1"}) is not None
    assert gpu_skip_reason({"CINEWRIGHT_GPU_TESTS": "1", "CINEWRIGHT_WAN2GP_ROOT": "C:/wan2gp"}) is not None
    assert gpu_skip_reason({"CINEWRIGHT_GPU_TESTS": "1", **ENGINE}) is None


def test_the_wan2gp_live_probe_is_a_gpu_test():
    assert "gpu" in [mark.name for mark in test_wan2gp_adapter.test_wan2gp_live_probe.pytestmark]
