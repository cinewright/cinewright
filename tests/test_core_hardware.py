"""Hardware detection: nvidia-smi parsing, driver check, and no crash without NVIDIA."""

from __future__ import annotations

import subprocess

from cinewright.api.schemas import GpuInfo
from cinewright.core import hardware

RTX_5080 = "0, NVIDIA GeForce RTX 5080, 16303, 621.14, 12.0, 5, 16\n"


def gpu(cc: str, driver: str) -> GpuInfo:
    return GpuInfo(index=0, name="GPU", vram_total_mb=8000, driver_version=driver, compute_capability=cc)


def test_parse_nvidia_smi():
    output = RTX_5080 + "1, Some, GPU, 8192, [N/A], 8.9, [N/A], [N/A]\ngarbage line\n"
    first, second = hardware.parse_nvidia_smi(output)
    assert first == GpuInfo(
        index=0,
        name="NVIDIA GeForce RTX 5080",
        vram_total_mb=16303,
        driver_version="621.14",
        compute_capability="12.0",
        pcie_link="Gen5 x16",
    )
    assert (second.name, second.driver_version, second.compute_capability, second.pcie_link) == ("Some, GPU", None, "8.9", None)


def test_driver_check():
    assert hardware.driver_findings([gpu("12.0", "621.14")]) == (True, [])
    ok, notes = hardware.driver_findings([gpu("12.0", "566.36")])
    assert ok is False
    assert "needs driver 570 or newer" in notes[0]
    ok, notes = hardware.driver_findings([gpu("8.9", "550.10")])
    assert ok is True
    assert notes == ["Driver branch 550 is older than 580, which the pinned engine build (CUDA 13.0) needs."]
    assert hardware.driver_findings([])[0] is None


def test_missing_or_failing_nvidia_smi_does_not_crash(monkeypatch, tmp_path):
    monkeypatch.setattr(hardware.shutil, "which", lambda name: None)
    assert hardware.query_nvidia_smi() is None
    profile = hardware.detect(tmp_path)
    assert (profile.gpus, profile.cuda_driver_ok) == ([], None)
    assert profile.ram_total_mb > 0 and profile.disk_free_mb > 0

    def timeout(*args, **kwargs):
        raise subprocess.TimeoutExpired("nvidia-smi", 10)

    monkeypatch.setattr(hardware.shutil, "which", lambda name: "/usr/bin/nvidia-smi")
    monkeypatch.setattr(hardware.subprocess, "run", timeout)
    assert hardware.query_nvidia_smi() is None


def test_detect_uses_nvidia_smi_output(monkeypatch, tmp_path):
    monkeypatch.setattr(hardware, "query_nvidia_smi", lambda: RTX_5080)
    profile = hardware.detect(tmp_path, "local")
    assert profile.gpus[0].name == "NVIDIA GeForce RTX 5080"
    assert profile.cuda_driver_ok is True
