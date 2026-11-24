"""Engine pins: `EnginePin` validates registry/engines.yaml; a malformed copy is an error."""

from __future__ import annotations

import copy
import shutil
from pathlib import Path

import pytest
import yaml

from cinewright.core.engines import DEFAULT_PATH, REGISTRY_DIR, EnginePin, EnginePinError, load_engine_pins

RAW = yaml.safe_load(DEFAULT_PATH.read_text(encoding="utf-8"))
FFMPEG_LATEST = "https://github.com/GyanD/codexffmpeg/releases/latest/download/ffmpeg-release-essentials.zip"
DELETE = object()


def write_copy(directory: Path, data) -> Path:
    """Write `data` as engines.yaml into `directory`, next to a copy of every lock the bundled pins name."""
    for pin in RAW.values():
        lock = directory / pin["lock"]
        lock.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(REGISTRY_DIR / pin["lock"], lock)
    path = directory / "engines.yaml"
    path.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")
    return path


def mutated(field: str, value):
    """A copy of the bundled pins with `field` (a /-separated path inside the wan2gp entry) set to `value` or deleted."""
    data = copy.deepcopy(RAW)
    *parents, key = field.split("/")
    node = data["wan2gp"]
    for part in parents:
        node = node[part]
    if value is DELETE:
        del node[key]
    else:
        node[key] = value
    return data


def test_bundled_pins_validate():
    pins = load_engine_pins()
    assert set(pins) == set(RAW)
    wan2gp = pins["wan2gp"]
    assert isinstance(wan2gp, EnginePin)
    assert wan2gp.archive_url.endswith(f"/archive/{wan2gp.commit}.zip")
    assert (REGISTRY_DIR / wan2gp.lock).is_file()
    assert wan2gp.torch_versions == {"torch": "2.10.0", "torchvision": "0.25.0", "torchaudio": "2.10.0"}
    assert set(wan2gp.ffmpeg.win_amd64.files) >= {"ffmpeg.exe", "ffprobe.exe", "ffplay.exe"}
    assert set(wan2gp.optional_wheels) == {"triton-windows", "sageattention"}


def test_unmodified_copy_validates(tmp_path):
    assert load_engine_pins(write_copy(tmp_path, RAW)) == load_engine_pins()


@pytest.mark.parametrize(
    ("field", "value"),
    [
        # A commit that is not 40 hex.
        ("commit", "38758d3"),
        ("commit", "38758D3C73366422EFCF609ABB63CAB8385F3552"),
        ("commit", "z" * 40),
        # A SHA-256 that is not 64 hex.
        ("archive_sha256", "4a65052e"),
        ("archive_sha256", "g" * 64),
        ("ffmpeg/win_amd64/sha256", "b9bbbb3a"),
        ("ffmpeg/win_amd64/files/ffmpeg.exe/sha256", "0" * 63),
        ("optional_wheels/triton-windows/sha256", "sha256:" + "0" * 64),
        ("optional_wheels/sageattention/sha256", ""),
        # A non-https URL.
        ("repo", "http://github.com/deepbeepmeep/Wan2GP"),
        ("archive_url", "http://github.com/deepbeepmeep/WanGP/archive/38758d3c73366422efcf609abb63cab8385f3552.zip"),
        ("torch_index", "http://download.pytorch.org/whl/cu130"),
        ("licence/url", "ftp://github.com/deepbeepmeep/Wan2GP/blob/main/LICENSE.txt"),
        ("ffmpeg/win_amd64/url", "http://github.com/GyanD/codexffmpeg/releases/download/9.1/ffmpeg-9.1-essentials_build.zip"),
        ("optional_wheels/sageattention/url", "file:///C:/wheels/sageattention-2.2.0-cp310-abi3-win_amd64.whl"),
        # An ffmpeg URL that points to a "latest" release, or names no version.
        ("ffmpeg/win_amd64/url", FFMPEG_LATEST),
        ("ffmpeg/win_amd64/url", "https://github.com/GyanD/codexffmpeg/releases/latest/download/ffmpeg-9.1-essentials_build.zip"),
        ("ffmpeg/win_amd64/url", "https://www.gyan.dev/ffmpeg/builds/ffmpeg-release-essentials.zip"),
        # A lock path that does not exist, or leaves the registry directory.
        ("lock", "wan2gp/pylock.missing.toml"),
        ("lock", "../pylock.toml"),
        ("lock", "wan2gp/pyproject.toml"),
        # The source archive must be the one of the pinned commit.
        ("archive_url", "https://github.com/deepbeepmeep/WanGP/archive/0000000000000000000000000000000000000000.zip"),
        # PyTorch: exactly torch, torchvision and torchaudio, each pinned with ==.
        ("torch", "torch==2.10.0 torchvision==0.25.0"),
        ("torch", "torch>=2.10.0 torchvision==0.25.0 torchaudio==2.10.0"),
        # ffmpeg must pin every file provisioning extracts.
        ("ffmpeg/win_amd64/files/ffprobe.exe", DELETE),
        # An optional wheel must be a wheel of the package it is pinned for.
        ("optional_wheels/triton-windows/url", "https://files.pythonhosted.org/packages/x/triton-3.7.1-cp311-cp311-win_amd64.whl"),
        ("optional_wheels/sageattention/url", "https://github.com/woct0rdho/SageAttention/archive/refs/tags/v2.2.0.zip"),
        ("committed_at", "2026-11-13T00:59:23"),
        ("min_driver", "R580"),
        ("python", "3.11"),
        ("verified/date", "yesterday"),
        ("licence/notice", DELETE),
        ("archive_sha256", DELETE),
        ("unknown_field", "value"),
    ],
)
def test_malformed_copy_fails_validation(tmp_path, field, value):
    path = write_copy(tmp_path, mutated(field, value))
    with pytest.raises(EnginePinError, match="wan2gp"):
        load_engine_pins(path)


def test_lock_is_resolved_next_to_the_file(tmp_path):
    path = write_copy(tmp_path, RAW)
    (tmp_path / RAW["wan2gp"]["lock"]).unlink()
    with pytest.raises(EnginePinError, match="lock"):
        load_engine_pins(path)


@pytest.mark.parametrize("text", ["", "- wan2gp\n", "wan2gp: [\n", "Wan2GP!: {}\n"])
def test_unreadable_file_is_an_error(tmp_path, text):
    path = tmp_path / "engines.yaml"
    path.write_text(text, encoding="utf-8")
    with pytest.raises(EnginePinError):
        load_engine_pins(path)


def test_missing_file_is_an_error(tmp_path):
    with pytest.raises(EnginePinError, match="engines.yaml"):
        load_engine_pins(tmp_path / "engines.yaml")
