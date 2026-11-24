"""Registry data: model descriptors, shared components and engine pins under src/cinewright/registry/."""

import datetime as dt
import re
import tomllib
from pathlib import Path
from urllib.parse import urlparse

import pytest
import yaml

import cinewright
from cinewright.api.schemas import ModelDescriptor
from cinewright.core.engines import load_engine_pins
from cinewright.core.registry import Registry

REGISTRY_DIR = Path(cinewright.__file__).parent / "registry"
MODEL_FILES = sorted((REGISTRY_DIR / "models").glob("*.yaml"))
SHARED_FILES = sorted((REGISTRY_DIR / "shared").glob("*.yaml"))
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
# The lock follows the same supply-chain quarantine as the engine commit (ARCHITECTURE.md, "Wan2GP pin").
QUARANTINE = dt.timedelta(days=7)
FIXED_UTC_TIMESTAMP = re.compile(r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z")


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


def lock_cutoff() -> dt.datetime:
    """uv's `exclude-newer` from the lock input's [tool.uv]."""
    return dt.datetime.fromisoformat(lock_input()["tool"]["uv"]["exclude-newer"])


def test_engine_lock_input_keeps_the_quarantine():
    """The lock input sets uv's exclude-newer to a fixed UTC time at least 7 days in the past, without exceptions.

    A duration (such as `7 days`) would make a regeneration depend on the day it runs, and `exclude-newer-package`
    would exempt packages. The 7-day check reads the clock, but it can only turn from failing to passing as time goes
    on: it rejects a cutoff that is less than 7 days old when the change is tested.
    """
    uv = lock_input()["tool"]["uv"]
    assert FIXED_UTC_TIMESTAMP.fullmatch(str(uv.get("exclude-newer", ""))), "[tool.uv] exclude-newer must be a fixed UTC time"
    assert "exclude-newer-package" not in uv
    assert lock_cutoff() <= dt.datetime.now(dt.UTC) - QUARANTINE


def test_engine_lock_respects_the_quarantine():
    """Every file the lock takes from an index was uploaded before the cutoff, so the lock was resolved with it.

    The direct-URL wheels upstream names (`archive` entries) have no upload time; they are pinned by URL and hash.
    """
    cutoff = lock_cutoff()
    for package in engine_lock()["packages"]:
        for file in artifacts(package):
            if file is package.get("archive"):
                continue
            uploaded = file.get("upload-time")
            assert isinstance(uploaded, dt.datetime), (package["name"], file["url"])
            assert uploaded < cutoff, (package["name"], file["url"], uploaded.isoformat())


def test_optional_wheels_are_pinned_with_hashes():
    wheels = wan2gp_pin().optional_wheels
    assert set(wheels) == {"triton-windows", "sageattention"}
    for wheel in wheels.values():
        assert re.fullmatch(r"[0-9a-f]{64}", wheel.sha256)
        url = urlparse(wheel.url)
        assert url.scheme == "https" and url.path.endswith("-win_amd64.whl"), wheel.url


# --------------------------------------------------------------------------- shared engine files

# What Wan2GP loads for each MVP model type at the pinned commit 38758d3c73366422efcf609abb63cab8385f3552,
# besides the transformer files of the quantization: the text encoder for its text_encoder_quantization
# (query_model_def), the files query_model_files lists (tokenizer, VAE, upscaler and config files) and the
# preload_URLs of the model definition. Paths are relative to the checkpoint root. Transcribed by hand from the
# family handlers (models/z_image/z_image_handler.py, models/ltx2/ltx2_handler.py, models/wan/wan_handler.py) and
# defaults/ltx2_22B.json (issue #41). Never derive this list from the registry, or the test proves nothing.
GEMMA = "gemma-3-12b-it-qat-q4_0-unquantized/"
Z_IMAGE = {
    "repo": "DeepBeepMeep/Z-Image",
    "text_encoder": {"bf16": "Qwen3/qwen3_bf16.safetensors", "int8": "Qwen3/qwen3_quanto_bf16_int8.safetensors"},
    "files": [
        "Qwen3/tokenizer.json",
        "Qwen3/tokenizer_config.json",
        "Qwen3/vocab.json",
        "Qwen3/config.json",
        "Qwen3/merges.txt",
        "ZImageTurbo_VAE_bf16.safetensors",
        "ZImageTurbo_VAE_bf16_config.json",
        "ZImageTurbo_scheduler_config.json",
    ],
}
LTX_2_3 = {
    "repo": "DeepBeepMeep/LTX-2",
    "text_encoder": {
        "bf16": GEMMA + "gemma-3-12b-it-qat-q4_0-unquantized.safetensors",
        "int8": GEMMA + "gemma-3-12b-it-qat-q4_0-unquantized_quanto_bf16_int8.safetensors",
    },
    "files": [
        GEMMA + "added_tokens.json",
        GEMMA + "chat_template.json",
        GEMMA + "config_light.json",
        GEMMA + "generation_config.json",
        GEMMA + "preprocessor_config.json",
        GEMMA + "processor_config.json",
        GEMMA + "special_tokens_map.json",
        GEMMA + "tokenizer.json",
        GEMMA + "tokenizer.model",
        GEMMA + "tokenizer_config.json",
        "ltx-2.3-22b_vae.safetensors",
        "ltx-2.3-22b_audio_vae.safetensors",
        "ltx-2.3-22b_vocoder.safetensors",
        "ltx-2.3-22b_text_embedding_projection.safetensors",
        "ltx-2.3-22b_embeddings_connector.safetensors",
        "ltx-2.3-spatial-upscaler-x2-1.1.safetensors",
        "ltx-2.3-temporal-upscaler-x2-1.0.safetensors",
        "ltx-2.3-22b-ic-lora-hdr-scene-emb.safetensors",
    ],
}
WAN_14B = {
    "repo": "DeepBeepMeep/Wan2.1",
    "text_encoder": {
        "bf16": "umt5-xxl/models_t5_umt5-xxl-enc-bf16.safetensors",
        "int8": "umt5-xxl/models_t5_umt5-xxl-enc-quanto_int8.safetensors",
    },
    "files": [
        "umt5-xxl/special_tokens_map.json",
        "umt5-xxl/spiece.model",
        "umt5-xxl/tokenizer.json",
        "umt5-xxl/tokenizer_config.json",
        "xlm-roberta-large/models_clip_open-clip-xlm-roberta-large-vit-huge-14-bf16.safetensors",
        "xlm-roberta-large/sentencepiece.bpe.model",
        "xlm-roberta-large/special_tokens_map.json",
        "xlm-roberta-large/tokenizer.json",
        "xlm-roberta-large/tokenizer_config.json",
        "Wan2.1_VAE.safetensors",
        "Wan2.1_VAE_upscale2x_imageonly_real_v1.safetensors",
    ],
}
WAN2GP_SHARED_FILES = {
    "z_image": Z_IMAGE,
    "ltx2_22B_distilled": LTX_2_3,
    "ltx2_22B_distilled_gguf_q4_k_m": LTX_2_3,
    "ltx2_22B_distilled_gguf_q6_k": LTX_2_3,
    "ltx2_22B_distilled_gguf_q8_0": LTX_2_3,
    "t2v_2_2": WAN_14B,
    "i2v_2_2": WAN_14B,
}
TEXT_ENCODER_VARIANTS = ("bf16", "int8")
SHARED_EXTENSIONS = {".safetensors", ".gguf", ".json", ".txt", ".model"}


def expected_shared_files(model_type: str, variant: str) -> set[tuple[str, str]]:
    family = WAN2GP_SHARED_FILES[model_type]
    return {(family["repo"], path) for path in [family["text_encoder"][variant], *family["files"]]}


def required_shared_files(registry: Registry, model_id: str, variant: str) -> list[tuple[str, str]]:
    return [(component.source.repo, file.path) for component in registry.requirements(model_id) for file in component.files_for(variant)]


def mvp_model_types() -> list[tuple[str, str]]:
    pairs = {(path.stem, binding.model_type) for path in MODEL_FILES if path.stem in MVP_MODEL_IDS for binding in descriptor(path).engines}
    return sorted(pairs)


def test_every_mvp_model_type_has_a_committed_file_list():
    assert {model_type for _, model_type in mvp_model_types()} == set(WAN2GP_SHARED_FILES)


@pytest.mark.parametrize("variant", TEXT_ENCODER_VARIANTS)
@pytest.mark.parametrize(("model_id", "model_type"), mvp_model_types(), ids=str)
def test_mvp_model_requires_the_files_wan2gp_loads(model_id, model_type, variant):
    required = required_shared_files(Registry.load(), model_id, variant)
    assert len(required) == len(set(required)), "a shared file is listed twice"
    assert set(required) == expected_shared_files(model_type, variant)


def test_shared_components_present_and_used():
    registry = Registry.load()
    assert [path.stem for path in SHARED_FILES] == [component.id for component in registry.components()]
    used = {component_id for model in registry.all() for component_id in model.requires}
    assert used == {component.id for component in registry.components()}, "every shared component is required by a model"


@pytest.mark.parametrize("path", SHARED_FILES, ids=lambda path: path.stem)
def test_shared_files_are_pinned_safe_downloads(path):
    component = Registry.load().component(path.stem)
    assert component.source.provider == "huggingface"
    assert re.fullmatch(r"[0-9a-f]{40}", component.source.revision)
    for file in component.files:
        assert re.fullmatch(r"[0-9a-f]{64}", file.sha256), file.path
        assert file.size_bytes > 0, file.path
        assert Path(file.name).suffix in SHARED_EXTENSIONS, file.path
        if file.variant is not None:
            assert Path(file.name).suffix == ".safetensors", "only text encoder weights have a variant"


@pytest.mark.parametrize("path", MODEL_FILES, ids=lambda path: path.stem)
def test_quantization_urls_name_the_source_revision(path):
    model = descriptor(path)
    assert model.source.revision is not None and re.fullmatch(r"[0-9a-f]{40}", model.source.revision)
    base = f"https://huggingface.co/{model.source.repo}/resolve/{model.source.revision}/"
    for quant in model.quantizations:
        for file in quant.files:
            assert file.url == base + file.name


def test_no_resolve_main_url_in_registry():
    files = [path for path in sorted(REGISTRY_DIR.rglob("*")) if path.is_file()]
    assert files
    assert [path.relative_to(REGISTRY_DIR).as_posix() for path in files if "resolve/main" in path.read_text(encoding="utf-8")] == []
