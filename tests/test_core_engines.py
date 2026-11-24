"""Engine pins: `EnginePin` validates registry/engines.yaml; a malformed copy is an error."""

from __future__ import annotations

import copy
import shutil
from pathlib import Path

import pytest
import yaml
from pydantic import ValidationError

from cinewright.core.engines import DEFAULT_PATH, REGISTRY_DIR, EnginePin, EnginePinError, load_engine_pins

RAW = yaml.safe_load(DEFAULT_PATH.read_text(encoding="utf-8"))
COMMIT = RAW["wan2gp"]["commit"]
FFMPEG_LATEST = "https://github.com/GyanD/codexffmpeg/releases/latest/download/ffmpeg-release-essentials.zip"
WRONG_ARCHIVE = "archive_url must be the GitHub source archive of `commit` in `repo`"
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


def archive_of(commit: str) -> str:
    return f"https://github.com/deepbeepmeep/WanGP/archive/{commit}.zip"


def assert_rejected(path: Path, error: str) -> None:
    """Loading `path` fails with exactly one validation error, `error` (`wan2gp/<field path>: <part of the message>`).

    Checking the location and message, not just that loading fails, makes sure the rule under test rejects the copy
    and not another rule that happens to trip over the same change.
    """
    where, message = error.split(": ", 1)
    with pytest.raises(EnginePinError, match="wan2gp") as excinfo:
        load_engine_pins(path)
    cause = excinfo.value.__cause__
    assert isinstance(cause, ValidationError)
    errors = [("/".join(["wan2gp", *map(str, item["loc"])]), item["msg"]) for item in cause.errors()]
    assert len(errors) == 1, errors
    assert errors[0][0] == where, errors
    assert message in errors[0][1], errors


def test_bundled_pins_validate():
    pins = load_engine_pins()
    assert set(pins) == set(RAW)
    wan2gp = pins["wan2gp"]
    assert isinstance(wan2gp, EnginePin)
    assert wan2gp.archive_url == f"{wan2gp.repo}/archive/{wan2gp.commit}.zip"
    assert (REGISTRY_DIR / wan2gp.lock).is_file()
    assert wan2gp.torch_versions == {"torch": "2.10.0", "torchvision": "0.25.0", "torchaudio": "2.10.0"}
    assert set(wan2gp.ffmpeg.win_amd64.files) >= {"ffmpeg.exe", "ffprobe.exe", "ffplay.exe"}
    assert set(wan2gp.optional_wheels) == {"triton-windows", "sageattention"}


def test_unmodified_copy_validates(tmp_path):
    assert load_engine_pins(write_copy(tmp_path, RAW)) == load_engine_pins()


@pytest.mark.parametrize(
    ("field", "value", "error"),
    [
        # A SHA-256 that is not 64 hex.
        ("archive_sha256", "4a65052e", "wan2gp/archive_sha256: [0-9a-f]{64}"),
        ("archive_sha256", "g" * 64, "wan2gp/archive_sha256: [0-9a-f]{64}"),
        ("ffmpeg/win_amd64/sha256", "b9bbbb3a", "wan2gp/ffmpeg/win_amd64/sha256: [0-9a-f]{64}"),
        ("ffmpeg/win_amd64/files/ffmpeg.exe/sha256", "0" * 63, "wan2gp/ffmpeg/win_amd64/files/ffmpeg.exe/sha256: [0-9a-f]{64}"),
        ("optional_wheels/triton-windows/sha256", "sha256:" + "0" * 64, "wan2gp/optional_wheels/triton-windows/sha256: [0-9a-f]{64}"),
        ("optional_wheels/sageattention/sha256", "", "wan2gp/optional_wheels/sageattention/sha256: [0-9a-f]{64}"),
        # A non-https URL.
        ("repo", "http://github.com/deepbeepmeep/Wan2GP", "wan2gp/repo: is not an https URL"),
        (
            "archive_url",
            "http://github.com/deepbeepmeep/WanGP/archive/38758d3c73366422efcf609abb63cab8385f3552.zip",
            "wan2gp/archive_url: is not an https URL",
        ),
        ("torch_index", "http://download.pytorch.org/whl/cu130", "wan2gp/torch_index: is not an https URL"),
        ("licence/url", "ftp://github.com/deepbeepmeep/Wan2GP/blob/main/LICENSE.txt", "wan2gp/licence/url: is not an https URL"),
        (
            "ffmpeg/win_amd64/url",
            "http://github.com/GyanD/codexffmpeg/releases/download/9.1/ffmpeg-9.1-essentials_build.zip",
            "wan2gp/ffmpeg/win_amd64/url: is not an https URL",
        ),
        (
            "optional_wheels/sageattention/url",
            "file:///C:/wheels/sageattention-2.2.0-cp310-abi3-win_amd64.whl",
            "wan2gp/optional_wheels/sageattention/url: is not an https URL",
        ),
        # An ffmpeg URL that points to a "latest" release, or names no version.
        ("ffmpeg/win_amd64/url", FFMPEG_LATEST, "wan2gp/ffmpeg/win_amd64: points to a 'latest' release"),
        (
            "ffmpeg/win_amd64/url",
            "https://github.com/GyanD/codexffmpeg/releases/latest/download/ffmpeg-9.1-essentials_build.zip",
            "wan2gp/ffmpeg/win_amd64: points to a 'latest' release",
        ),
        (
            "ffmpeg/win_amd64/url",
            "https://www.gyan.dev/ffmpeg/builds/ffmpeg-release-essentials.zip",
            "wan2gp/ffmpeg/win_amd64: does not name version 9.1",
        ),
        # A lock path that does not exist. The rules for the path itself: test_lock_path_rules and the rows below.
        ("lock", "wan2gp/pylock.missing.toml", "wan2gp/lock: does not exist"),
        # Paths stay inside their directory: no absolute path, no drive (checked before the lock's existence).
        ("lock", "/registry/wan2gp/pylock.win_amd64.toml", "wan2gp/lock: must be a relative path"),
        ("lock", "C:wan2gp/pylock.win_amd64.toml", "wan2gp/lock: must be a relative path"),
        ("requirements", "/requirements.txt", "wan2gp/requirements: must be a relative path"),
        ("requirements", "C:requirements.txt", "wan2gp/requirements: must be a relative path"),
        ("ffmpeg/win_amd64/bin_dir", "/ffmpeg-9.1-essentials_build/bin", "wan2gp/ffmpeg/win_amd64/bin_dir: must be a relative path"),
        ("ffmpeg/win_amd64/bin_dir", "C:/ffmpeg-9.1-essentials_build/bin", "wan2gp/ffmpeg/win_amd64/bin_dir: must be a relative path"),
        # The source archive must be the GitHub source archive of the pinned commit in the pinned repository.
        ("archive_url", archive_of("0" * 40), f"wan2gp: {WRONG_ARCHIVE}"),
        ("archive_url", f"https://example.com/deepbeepmeep/WanGP/archive/{COMMIT}.zip", f"wan2gp: {WRONG_ARCHIVE}"),
        ("archive_url", f"https://github.com@example.com/deepbeepmeep/WanGP/archive/{COMMIT}.zip", f"wan2gp: {WRONG_ARCHIVE}"),
        ("archive_url", f"https://github.com/someone-else/WanGP/archive/{COMMIT}.zip", f"wan2gp: {WRONG_ARCHIVE}"),
        ("archive_url", f"https://github.com/mirror/deepbeepmeep/WanGP/archive/{COMMIT}.zip", f"wan2gp: {WRONG_ARCHIVE}"),
        ("archive_url", f"https://github.com/deepbeepmeep/Wan2GP/archive/{COMMIT}.zip", f"wan2gp: {WRONG_ARCHIVE}"),
        ("repo", "https://github.com/someone-else/WanGP", f"wan2gp: {WRONG_ARCHIVE}"),
        # The repository is a GitHub repository URL.
        ("repo", "https://gitlab.com/deepbeepmeep/WanGP", "wan2gp/repo: is not a GitHub repository"),
        ("repo", "https://github.com/deepbeepmeep", "wan2gp/repo: is not a GitHub repository"),
        ("repo", "https://github.com/deepbeepmeep/WanGP/tree/main", "wan2gp/repo: is not a GitHub repository"),
        # PyTorch: exactly torch, torchvision and torchaudio, each pinned with ==.
        ("torch", "torch==2.10.0 torchvision==0.25.0", "wan2gp/torch: must pin exactly torch, torchvision, torchaudio"),
        ("torch", "torch>=2.10.0 torchvision==0.25.0 torchaudio==2.10.0", "wan2gp/torch: 'torch>=2.10.0' is not name==version"),
        # ffmpeg must pin every file provisioning extracts, and nothing else but DLLs.
        ("ffmpeg/win_amd64/files/ffprobe.exe", DELETE, "wan2gp/ffmpeg/win_amd64: lacks ['ffprobe.exe']"),
        (
            "ffmpeg/win_amd64/files/readme.txt",
            {"sha256": "0" * 64, "size": 1},
            "wan2gp/ffmpeg/win_amd64: may list only ['ffmpeg.exe', 'ffplay.exe', 'ffprobe.exe'] and DLLs, not ['readme.txt']",
        ),
        # An optional wheel must be a wheel of the package it is pinned for.
        (
            "optional_wheels/triton-windows/url",
            "https://files.pythonhosted.org/packages/x/triton-3.7.1-cp311-cp311-win_amd64.whl",
            "wan2gp/optional_wheels: points to a wheel of 'triton'",
        ),
        (
            "optional_wheels/sageattention/url",
            "https://github.com/woct0rdho/SageAttention/archive/refs/tags/v2.2.0.zip",
            "wan2gp/optional_wheels/sageattention/url: is not a wheel",
        ),
        ("committed_at", "2026-11-13T00:59:23", "wan2gp/committed_at: timezone info"),
        ("min_driver", "R580", "wan2gp/min_driver: should match pattern"),
        ("python", "3.11", "wan2gp/python: should match pattern"),
        ("verified/date", "yesterday", "wan2gp/verified/date: valid date"),
        ("licence/notice", DELETE, "wan2gp/licence/notice: Field required"),
        ("archive_sha256", DELETE, "wan2gp/archive_sha256: Field required"),
        ("unknown_field", "value", "wan2gp/unknown_field: Extra inputs are not permitted"),
    ],
)
def test_malformed_copy_fails_validation(tmp_path, field, value, error):
    assert_rejected(write_copy(tmp_path, mutated(field, value)), error)


@pytest.mark.parametrize("commit", ["38758d3", "38758D3C73366422EFCF609ABB63CAB8385F3552", "z" * 40])
def test_commit_must_be_40_lowercase_hex(tmp_path, commit):
    # archive_url follows the commit, so the archive cross-check cannot be what rejects the copy.
    data = mutated("commit", commit)
    data["wan2gp"]["archive_url"] = archive_of(commit)
    assert_rejected(write_copy(tmp_path, data), "wan2gp/commit: [0-9a-f]{40}")


@pytest.mark.parametrize(
    ("lock", "message"),
    [
        ("../pylock.toml", "stays inside its directory"),
        ("wan2gp\\pylock.win_amd64.toml", "with / separators"),
        ("wan2gp/pyproject.toml", "must be a PEP 751 file"),
    ],
)
def test_lock_path_rules(tmp_path, lock, message):
    # The file exists, so the existence check cannot be what rejects the copy.
    registry = tmp_path / "registry"
    path = write_copy(registry, mutated("lock", lock))
    (registry / lock).touch()
    assert_rejected(path, f"wan2gp/lock: {message}")


def test_windows_ffmpeg_pin_may_list_dlls(tmp_path):
    path = write_copy(tmp_path, mutated("ffmpeg/win_amd64/files/avcodec-62.dll", {"sha256": "0" * 64, "size": 1}))
    assert "avcodec-62.dll" in load_engine_pins(path)["wan2gp"].ffmpeg.win_amd64.files


@pytest.mark.parametrize("engine_id", ["Wan2GP!", "wan-2gp", "2wan", "WAN2GP"])
def test_engine_id_rules(tmp_path, engine_id):
    # The entry itself is the valid bundled one (it loads under a valid id), so only the id can be what fails.
    assert set(load_engine_pins(write_copy(tmp_path, {"wan2gp_next": RAW["wan2gp"]}))) == {"wan2gp_next"}
    with pytest.raises(EnginePinError, match="is not an engine id"):
        load_engine_pins(write_copy(tmp_path, {engine_id: RAW["wan2gp"]}))


def test_lock_is_resolved_next_to_the_file(tmp_path):
    path = write_copy(tmp_path, RAW)
    (tmp_path / RAW["wan2gp"]["lock"]).unlink()
    with pytest.raises(EnginePinError, match="lock"):
        load_engine_pins(path)


@pytest.mark.parametrize("content", [b"", b"- wan2gp\n", b"wan2gp: [\n", b"Wan2GP!: {}\n", b"wan2gp:\n  name: \xff\xfe not UTF-8\n"])
def test_unreadable_file_is_an_error(tmp_path, content):
    path = tmp_path / "engines.yaml"
    path.write_bytes(content)
    with pytest.raises(EnginePinError):
        load_engine_pins(path)


def test_missing_file_is_an_error(tmp_path):
    with pytest.raises(EnginePinError, match="engines.yaml"):
        load_engine_pins(tmp_path / "engines.yaml")
