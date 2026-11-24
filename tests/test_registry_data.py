"""Registry data: model descriptors and engine pins under src/cinewright/registry/."""

import re
import tomllib
from pathlib import Path
from urllib.parse import urlparse

import pytest
import yaml

import cinewright
from cinewright.api.schemas import ModelDescriptor
from cinewright.core.engines import load_engine_pins

REGISTRY_DIR = Path(cinewright.__file__).parent / "registry"
MODEL_FILES = sorted((REGISTRY_DIR / "models").glob("*.yaml"))
MVP_MODEL_IDS = {"z-image-turbo", "ltx-2.3-distilled-22b", "wan-2.2-i2v-14b", "wan-2.2-t2v-14b"}
FILE_HOSTS = {"huggingface.co", "civitai.com"}
EXTENSIONS = {".safetensors": "safetensors", ".gguf": "gguf"}
WANGP_NOTICE = (
    "Cinewright uses WanGP (Wan2GP) by deepbeepmeep as its first generation engine. WanGP is source-available under the "
    "WanGP Community License 2.0 and is downloaded from its official repository during setup; it is not part of Cinewright."
)


def load(path: Path):
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def descriptor(path: Path) -> ModelDescriptor:
    return ModelDescriptor.model_validate(load(path))


def test_mvp_models_present():
    assert {path.stem for path in MODEL_FILES} >= MVP_MODEL_IDS


@pytest.mark.parametrize("path", MODEL_FILES, ids=lambda path: path.stem)
def test_descriptor_validates_and_id_matches_file_name(path):
    assert descriptor(path).id == path.stem


def test_ids_unique():
    ids = [descriptor(path).id for path in MODEL_FILES]
    assert len(ids) == len(set(ids))


@pytest.mark.parametrize("path", MODEL_FILES, ids=lambda path: path.stem)
def test_files_are_pinned_safe_downloads(path):
    for quant in descriptor(path).quantizations:
        for file in quant.files:
            assert re.fullmatch(r"[0-9a-f]{64}", file.sha256), file.name
            url = urlparse(file.url)
            assert url.scheme == "https" and url.hostname in FILE_HOSTS, file.url
            suffix = Path(url.path).suffix
            assert suffix in EXTENSIONS, file.url
            assert EXTENSIONS[suffix] == quant.format, file.url
            assert Path(url.path).name == file.name


@pytest.mark.parametrize("path", MODEL_FILES, ids=lambda path: path.stem)
def test_engine_bindings_match_quantizations(path):
    model = descriptor(path)
    quant_ids = [quant.id for quant in model.quantizations]
    assert len(quant_ids) == len(set(quant_ids))
    bound = {binding.quant for binding in model.engines}
    assert bound <= set(quant_ids)
    assert set(quant_ids) <= bound, "every quantization needs an engine binding"


@pytest.mark.parametrize("path", MODEL_FILES, ids=lambda path: path.stem)
def test_defaults_within_limits(path):
    model = descriptor(path)
    defaults, limits = model.defaults, model.limits
    if "resolution" in defaults:
        assert defaults["resolution"] in limits.resolutions
    if "frames" in defaults:
        assert limits.max_frames is not None and defaults["frames"] <= limits.max_frames
    if "fps" in defaults:
        assert defaults["fps"] in limits.fps


def test_engine_pin():
    engines = load(REGISTRY_DIR / "engines.yaml")
    wan2gp = engines["wan2gp"]
    assert re.fullmatch(r"[0-9a-f]{40}", wan2gp["commit"])
    assert wan2gp["repo"].startswith("https://github.com/")
    assert wan2gp["licence"]["notice"] == WANGP_NOTICE


def test_every_binding_engine_is_pinned():
    engines = load(REGISTRY_DIR / "engines.yaml")
    for path in MODEL_FILES:
        assert {binding.engine for binding in descriptor(path).engines} <= set(engines)


# --------------------------------------------------------------------------- engine environment lock

PYPI_FILES = "files.pythonhosted.org"
PYTORCH_HOSTS = {"download.pytorch.org", "download-r2.pytorch.org"}
ONNXRUNTIME_FEED = "https://aiinfra.pkgs.visualstudio.com/PublicPackages/_packaging/ort-cuda-13-nightly/pypi/simple/"
# Packages that come from somewhere other than PyPI, and from which hosts. Everything else comes from PyPI,
# except the direct-URL wheels upstream names (GitHub release assets, listed in the lock input).
SCOPED_HOSTS = {
    "torch": PYTORCH_HOSTS,
    "torchvision": PYTORCH_HOSTS,
    "torchaudio": PYTORCH_HOSTS,
    "onnxruntime-gpu": {urlparse(ONNXRUNTIME_FEED).hostname},
}
DIRECT_URL_HOST = "github.com"
GLOBAL_INDEX_OPTIONS = ("--index-url", "--extra-index-url", "--find-links")


def wan2gp_pin():
    return load_engine_pins()["wan2gp"]


def engine_lock_path() -> Path:
    return REGISTRY_DIR / wan2gp_pin().lock


def engine_lock() -> dict:
    return tomllib.loads(engine_lock_path().read_text(encoding="utf-8"))


def lock_input() -> dict:
    return tomllib.loads((engine_lock_path().parent / "pyproject.toml").read_text(encoding="utf-8"))


def artifacts(package: dict) -> list[dict]:
    """Every file a lock entry offers: its wheels, its sdist and a direct-URL archive."""
    files = list(package.get("wheels", []))
    files += [package[key] for key in ("sdist", "archive") if key in package]
    return files


def test_engine_lock_entries_are_pinned_and_hashed():
    packages = engine_lock()["packages"]
    assert len(packages) > 100
    names = [package["name"] for package in packages]
    assert len(names) == len(set(names)), "one entry per package"
    for package in packages:
        assert re.fullmatch(r"[0-9][0-9A-Za-z.+!_-]*", str(package.get("version", ""))), package["name"]
        files = artifacts(package)
        assert files, package["name"]
        for file in files:
            assert re.fullmatch(r"[0-9a-f]{64}", file.get("hashes", {}).get("sha256", "")), (package["name"], file["url"])
            assert urlparse(file["url"]).scheme == "https", file["url"]


def test_engine_lock_has_no_global_index_option():
    text = engine_lock_path().read_text(encoding="utf-8")
    assert [option for option in GLOBAL_INDEX_OPTIONS if option in text] == []
    assert engine_lock()["lock-version"] == "1.0"


def test_engine_lock_has_torch_at_the_pinned_versions():
    versions = {package["name"]: package["version"] for package in engine_lock()["packages"]}
    expected = wan2gp_pin().torch_versions
    assert set(expected) == {"torch", "torchvision", "torchaudio"}
    for name, version in expected.items():
        # The local version label (+cu130) aside.
        assert versions[name].split("+", 1)[0] == version, name


def test_engine_lock_sources_are_scoped():
    """PyTorch only from torch_index, onnxruntime-gpu only from its nightly feed, and nothing else from either."""
    torch_path = urlparse(wan2gp_pin().torch_index).path.rstrip("/") + "/"
    direct_urls = {
        requirement.split(" @ ", 1)[1].split(";", 1)[0].strip()
        for requirement in lock_input()["project"]["dependencies"]
        if " @ " in requirement
    }
    for package in engine_lock()["packages"]:
        for file in artifacts(package):
            url = urlparse(file["url"])
            if package["name"] in SCOPED_HOSTS:
                assert url.hostname in SCOPED_HOSTS[package["name"]], file["url"]
                if url.hostname in PYTORCH_HOSTS:
                    assert url.path.startswith(torch_path), file["url"]
            elif url.hostname == DIRECT_URL_HOST:
                assert file is package.get("archive") and file["url"] in direct_urls, file["url"]
            else:
                assert url.hostname == PYPI_FILES, file["url"]


def test_engine_lock_targets_windows_x64():
    for package in engine_lock()["packages"]:
        for wheel in package.get("wheels", []):
            platform_tag = Path(urlparse(wheel["url"]).path).stem.rsplit("-", 1)[-1]
            assert platform_tag in {"win_amd64", "any"}, wheel["url"]


def test_engine_lock_input_scopes_the_extra_indexes():
    """The lock input has no global extra index: each extra index is explicit and serves only the packages pinned to it."""
    data = lock_input()
    pin = wan2gp_pin()
    uv = data["tool"]["uv"]
    indexes = {index["name"]: index for index in uv["index"]}
    assert all(index.get("explicit") is True for index in indexes.values())
    assert not {"index-url", "extra-index-url", "find-links"} & set(uv)
    sources = {name: indexes[source["index"]]["url"] for name, source in uv["sources"].items()}
    assert sources == {
        "torch": pin.torch_index,
        "torchvision": pin.torch_index,
        "torchaudio": pin.torch_index,
        "onnxruntime-gpu": ONNXRUNTIME_FEED,
    }
    assert data["project"]["dependencies"][:3] == [f"{name}=={version}" for name, version in pin.torch_versions.items()]


def test_optional_wheels_are_pinned_with_hashes():
    wheels = wan2gp_pin().optional_wheels
    assert set(wheels) == {"triton-windows", "sageattention"}
    for wheel in wheels.values():
        assert re.fullmatch(r"[0-9a-f]{64}", wheel.sha256)
        url = urlparse(wheel.url)
        assert url.scheme == "https" and url.path.endswith("-win_amd64.whl"), wheel.url
