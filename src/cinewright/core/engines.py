"""Engine pins: everything engine provisioning downloads, validated from `src/cinewright/registry/engines.yaml`.

`EnginePin` is internal data, not part of the API contract. Provisioning (#1) and the driver check (#10)
read the pins through `load_engine_pins()`. An invalid file is an error, because the pins ship with the
application and a broken entry is a bug. Paths in a pin (`lock`) are relative to the directory of the file.
"""

from __future__ import annotations

import datetime as dt
import re
from pathlib import Path, PurePosixPath
from typing import Annotated
from urllib.parse import unquote, urlsplit

import yaml
from pydantic import (
    AfterValidator,
    AwareDatetime,
    BaseModel,
    ConfigDict,
    Field,
    PositiveInt,
    ValidationError,
    ValidationInfo,
    field_validator,
    model_validator,
)

REGISTRY_DIR = Path(__file__).resolve().parent.parent / "registry"
DEFAULT_PATH = REGISTRY_DIR / "engines.yaml"

TORCH_PACKAGES = ("torch", "torchvision", "torchaudio")
# Files provisioning extracts from the Windows ffmpeg archive into `<wan2gp root>/ffmpeg_bins`, plus any DLLs.
FFMPEG_WINDOWS_FILES = frozenset({"ffmpeg.exe", "ffprobe.exe", "ffplay.exe"})

_GITHUB = "github.com"  # host of every engine repository and its source archives
_ENGINE_ID = re.compile(r"^[a-z][a-z0-9_]*$")
_LOCK_NAME = re.compile(r"^pylock(\.[^.]+)?\.toml$")  # PEP 751 file name
_TORCH_PIN = re.compile(r"^(?P<name>[a-z]+)==(?P<version>\d+(\.\d+)*)$")


class EnginePinError(ValueError):
    pass


def _https_url(value: str) -> str:
    parts = urlsplit(value)
    if parts.scheme != "https" or not parts.hostname:
        raise ValueError(f"{value!r} is not an https URL")
    return value


def _relative_path(value: str) -> str:
    path = PurePosixPath(value)
    if "\\" in value or ":" in value or path.is_absolute() or ".." in path.parts or str(path) == ".":
        raise ValueError(f"{value!r} must be a relative path with / separators that stays inside its directory")
    return value


def _url_path_segments(url: str) -> list[str]:
    return [unquote(segment) for segment in urlsplit(url).path.split("/") if segment]


HttpsUrl = Annotated[str, AfterValidator(_https_url)]
RelativePath = Annotated[str, Field(min_length=1), AfterValidator(_relative_path)]
Sha256 = Annotated[str, Field(pattern=r"^[0-9a-f]{64}$", description="Lowercase hex SHA-256")]
FileName = Annotated[str, Field(pattern=r"^[A-Za-z0-9][A-Za-z0-9._+-]*$")]


class _Pin(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class Verification(_Pin):
    date: dt.date
    gpu: str = Field(min_length=1)
    result: str = Field(min_length=1)


class Licence(_Pin):
    name: str = Field(min_length=1)
    url: HttpsUrl
    notice: str = Field(min_length=1, description="Disclosure the licence requires in UI and docs, verbatim")


class PinnedFile(_Pin):
    sha256: Sha256
    size: PositiveInt


class FfmpegBuild(_Pin):
    """A versioned ffmpeg release archive and the files provisioning extracts from it."""

    version: str = Field(pattern=r"^\d+(\.\d+)*$")
    url: HttpsUrl
    sha256: Sha256
    size: PositiveInt
    bin_dir: RelativePath = Field(description="Directory inside the archive that holds `files`")
    files: dict[FileName, PinnedFile] = Field(min_length=1)

    @model_validator(mode="after")
    def _versioned_release(self) -> FfmpegBuild:
        segments = _url_path_segments(self.url)
        if any("latest" in segment.lower() for segment in segments):
            raise ValueError(f"ffmpeg url {self.url!r} points to a 'latest' release; pin a versioned one")
        version = re.compile(rf"(?<![\d.]){re.escape(self.version)}(?![\d.])")
        if not any(version.search(segment) for segment in segments):
            raise ValueError(f"ffmpeg url {self.url!r} does not name version {self.version}")
        return self


class FfmpegPins(_Pin):
    """ffmpeg per platform. Linux follows with #22."""

    win_amd64: FfmpegBuild

    @field_validator("win_amd64")
    @classmethod
    def _windows_files(cls, build: FfmpegBuild) -> FfmpegBuild:
        missing = FFMPEG_WINDOWS_FILES - set(build.files)
        if missing:
            raise ValueError(f"the Windows ffmpeg pin lacks {sorted(missing)}")
        unexpected = sorted(name for name in build.files if name not in FFMPEG_WINDOWS_FILES and not name.lower().endswith(".dll"))
        if unexpected:
            raise ValueError(f"the Windows ffmpeg pin may list only {sorted(FFMPEG_WINDOWS_FILES)} and DLLs, not {unexpected}")
        return build


class PinnedWheel(_Pin):
    url: HttpsUrl
    sha256: Sha256

    @field_validator("url")
    @classmethod
    def _is_wheel(cls, url: str) -> str:
        segments = _url_path_segments(url)
        if not segments or not segments[-1].endswith(".whl"):
            raise ValueError(f"{url!r} is not a wheel")
        return url

    @property
    def distribution(self) -> str:
        """Normalised distribution name from the wheel's file name."""
        return _normalise(_url_path_segments(self.url)[-1].split("-", 1)[0])


def _normalise(name: str) -> str:
    return re.sub(r"[-_.]+", "-", name).lower()


class EnginePin(_Pin):
    """One engine: source archive, Python environment (with its lock), ffmpeg, optional wheels, driver and licence."""

    name: str = Field(min_length=1)
    repo: HttpsUrl = Field(description="Upstream GitHub repository, https://github.com/<owner>/<name>")
    commit: str = Field(pattern=r"^[0-9a-f]{40}$")
    committed_at: AwareDatetime
    archive_url: HttpsUrl
    archive_sha256: Sha256
    archive_size: PositiveInt
    python: str = Field(pattern=r"^3\.\d+\.\d+$")
    torch: str = Field(description="`torch==X torchvision==Y torchaudio==Z`, installed from `torch_index`")
    torch_index: HttpsUrl
    requirements: RelativePath = Field(description="Path inside the engine checkout")
    lock: RelativePath = Field(description="Hash-locked environment (PEP 751), relative to the directory of engines.yaml")
    ffmpeg: FfmpegPins
    optional_wheels: dict[str, PinnedWheel] = Field(default_factory=dict)
    min_driver: str = Field(pattern=r"^\d+(\.\d+)*$", description="Minimum NVIDIA driver branch")
    verified: Verification
    licence: Licence

    @field_validator("repo")
    @classmethod
    def _github_repository(cls, value: str) -> str:
        parts = urlsplit(value)
        if parts.netloc != _GITHUB or len(_url_path_segments(value)) != 2 or parts.query or parts.fragment:
            raise ValueError(f"repo {value!r} is not a GitHub repository (https://{_GITHUB}/<owner>/<name>)")
        return value

    @field_validator("torch")
    @classmethod
    def _torch_pins(cls, value: str) -> str:
        names = []
        for item in value.split():
            match = _TORCH_PIN.match(item)
            if not match:
                raise ValueError(f"torch entry {item!r} is not name==version")
            names.append(match["name"])
        if sorted(names) != sorted(TORCH_PACKAGES):
            raise ValueError(f"torch must pin exactly {', '.join(TORCH_PACKAGES)} once each")
        return value

    @field_validator("lock")
    @classmethod
    def _lock_exists(cls, value: str, info: ValidationInfo) -> str:
        if not _LOCK_NAME.match(PurePosixPath(value).name):
            raise ValueError(f"lock {value!r} must be a PEP 751 file (pylock.toml or pylock.<name>.toml)")
        base = Path((info.context or {}).get("registry_dir", REGISTRY_DIR))
        if not (base / value).is_file():
            raise ValueError(f"lock file {value!r} does not exist next to engines.yaml")
        return value

    @field_validator("optional_wheels")
    @classmethod
    def _wheel_names(cls, wheels: dict[str, PinnedWheel]) -> dict[str, PinnedWheel]:
        for name, wheel in wheels.items():
            if _normalise(name) != wheel.distribution:
                raise ValueError(f"optional wheel {name!r} points to a wheel of {wheel.distribution!r}")
        return wheels

    @model_validator(mode="after")
    def _archive_of_commit(self) -> EnginePin:
        owner, name = _url_path_segments(self.repo)
        expected = f"https://{_GITHUB}/{owner}/{name}/archive/{self.commit}.zip"
        if self.archive_url != expected:
            raise ValueError(f"archive_url must be the GitHub source archive of `commit` in `repo`, {expected}")
        return self

    @property
    def torch_versions(self) -> dict[str, str]:
        """The PyTorch packages and their versions, in the order of `torch`."""
        return dict(item.split("==", 1) for item in self.torch.split())


def load_engine_pins(path: Path | None = None) -> dict[str, EnginePin]:
    """Load and validate the engine pins, keyed by engine id. Raises `EnginePinError` on any problem."""
    path = DEFAULT_PATH if path is None else path
    try:
        data = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, yaml.YAMLError) as exc:
        raise EnginePinError(f"Cannot read engine pins {path.name}: {exc}") from exc
    if not isinstance(data, dict) or not data:
        raise EnginePinError(f"{path.name} must map engine ids to engine pins")
    pins: dict[str, EnginePin] = {}
    for engine_id, entry in data.items():
        if not isinstance(engine_id, str) or not _ENGINE_ID.match(engine_id):
            raise EnginePinError(f"{path.name}: {engine_id!r} is not an engine id")
        try:
            pins[engine_id] = EnginePin.model_validate(entry, context={"registry_dir": path.parent})
        except ValidationError as exc:
            raise EnginePinError(f"Invalid engine pin {engine_id!r} in {path.name}: {exc}") from exc
    return pins
