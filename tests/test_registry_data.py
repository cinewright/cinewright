"""Registry data: model descriptors and engine pins under src/cinewright/registry/."""

import re
from pathlib import Path
from urllib.parse import urlparse

import pytest
import yaml

import cinewright
from cinewright.api.schemas import ModelDescriptor

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
