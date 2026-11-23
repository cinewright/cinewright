"""Hardware detection: NVIDIA GPUs via nvidia-smi, RAM via psutil, free disk of the data directory.

Never raises because a tool is missing: without nvidia-smi the profile simply lists no GPU.
"""

from __future__ import annotations

import platform
import shutil
import subprocess
import sys
from pathlib import Path

import psutil

from cinewright.api.schemas import GpuInfo, HardwareProfile
from cinewright.core.clock import utcnow

MIB = 1024 * 1024
NVIDIA_SMI_TIMEOUT_S = 10.0
NVIDIA_QUERY = "index,name,memory.total,driver_version,compute_cap,pcie.link.gen.current,pcie.link.width.current"
# Blackwell (compute capability 12.x) needs CUDA 12.8, i.e. driver branch 570 or newer.
BLACKWELL_MIN_DRIVER = 570
# The pinned engine build uses CUDA 13.0, which needs driver branch 580 or newer.
ENGINE_MIN_DRIVER = 580


def query_nvidia_smi(timeout: float = NVIDIA_SMI_TIMEOUT_S) -> str | None:
    executable = shutil.which("nvidia-smi")
    if executable is None:
        return None
    # The launcher runs without a console on Windows; keep nvidia-smi from flashing a window.
    flags = subprocess.CREATE_NO_WINDOW if sys.platform == "win32" else 0
    try:
        result = subprocess.run(  # noqa: S603 - fixed arguments, executable resolved from PATH
            [executable, f"--query-gpu={NVIDIA_QUERY}", "--format=csv,noheader,nounits"],
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
            creationflags=flags,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return result.stdout if result.returncode == 0 else None


def _value(text: str) -> str | None:
    text = text.strip()
    return None if not text or text.startswith("[") or text.upper() == "N/A" else text


def parse_nvidia_smi(output: str) -> list[GpuInfo]:
    gpus = []
    for line in output.splitlines():
        parts = [part.strip() for part in line.split(",")]
        if len(parts) < 7:
            continue
        # A GPU name could contain a comma; the other six fields cannot.
        index, name, memory, driver, compute, gen, width = parts[0], ", ".join(parts[1:-5]), *parts[-5:]
        try:
            gpu_index = int(index)
            vram = int(float(memory))
        except ValueError:
            continue
        link_gen, link_width = _value(gen), _value(width)
        gpus.append(
            GpuInfo(
                index=gpu_index,
                name=name,
                vram_total_mb=vram,
                driver_version=_value(driver),
                compute_capability=_value(compute),
                pcie_link=f"Gen{link_gen} x{link_width}" if link_gen and link_width else None,
            )
        )
    return gpus


def _major(version: str | None) -> int | None:
    try:
        return int(version.split(".")[0]) if version else None
    except ValueError:
        return None


def driver_findings(gpus: list[GpuInfo]) -> tuple[bool | None, list[str]]:
    """`cuda_driver_ok` and the notes about it. None without an NVIDIA GPU."""
    if not gpus:
        return None, ["No NVIDIA GPU found, so no local engine can generate images or video."]
    ok = True
    notes = []
    for gpu in gpus:
        capability, driver = _major(gpu.compute_capability), _major(gpu.driver_version)
        if capability is not None and capability >= 12 and driver is not None and driver < BLACKWELL_MIN_DRIVER:
            ok = False
            notes.append(
                f"Driver {gpu.driver_version} is too old for {gpu.name}: compute capability {gpu.compute_capability} "
                f"needs driver {BLACKWELL_MIN_DRIVER} or newer (CUDA 12.8)."
            )
    drivers = {_major(gpu.driver_version) for gpu in gpus} - {None}
    if drivers and min(drivers) < ENGINE_MIN_DRIVER:
        notes.append(f"Driver branch {min(drivers)} is older than {ENGINE_MIN_DRIVER}, which the pinned engine build (CUDA 13.0) needs.")
    return ok, notes


def cpu_name() -> str | None:
    if sys.platform == "win32":
        import winreg

        try:
            with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, r"HARDWARE\DESCRIPTION\System\CentralProcessor\0") as key:
                return str(winreg.QueryValueEx(key, "ProcessorNameString")[0]).strip() or None
        except OSError:
            pass
    elif sys.platform.startswith("linux"):
        try:
            for line in Path("/proc/cpuinfo").read_text(encoding="utf-8", errors="replace").splitlines():
                if line.startswith("model name"):
                    return line.split(":", 1)[1].strip() or None
        except OSError:
            pass
    return platform.processor() or None


def detect(data_dir: Path, host_id: str = "local") -> HardwareProfile:
    output = query_nvidia_smi()
    gpus = parse_nvidia_smi(output) if output else []
    driver_ok, notes = driver_findings(gpus)
    disk_dir = data_dir if data_dir.exists() else Path(data_dir.anchor or ".")
    return HardwareProfile(
        host_id=host_id,
        os=f"{platform.system()} {platform.release()} ({platform.version()})",
        cpu=cpu_name(),
        ram_total_mb=psutil.virtual_memory().total // MIB,
        disk_free_mb=shutil.disk_usage(disk_dir).free // MIB,
        gpus=gpus,
        cuda_driver_ok=driver_ok,
        notes=notes,
        detected_at=utcnow(),
    )
