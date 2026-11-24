"""Registry loading: YAML descriptors validated with ModelDescriptor and SharedComponent; empty is valid."""

from __future__ import annotations

import copy
import re
from pathlib import Path
from typing import Any

import pytest
import yaml
from conftest import TEST_IMAGE

from cinewright.core.registry import DEFAULT_DIRECTORY, Registry, RegistryError

REVISION = "0123456789abcdef0123456789abcdef01234567"
BASE = f"https://huggingface.co/test/shared/resolve/{REVISION}/"

TEST_SHARED: dict[str, Any] = {
    "id": "test-shared",
    "source": {"provider": "huggingface", "repo": "test/shared", "revision": REVISION},
    "files": [
        {
            "name": "encoder_bf16.safetensors",
            "subfolder": "encoder/",
            "url": BASE + "encoder/encoder_bf16.safetensors",
            "sha256": "d" * 64,
            "size_bytes": 2000,
            "variant": "bf16",
        },
        {
            "name": "encoder_int8.safetensors",
            "subfolder": "encoder/",
            "url": BASE + "encoder/encoder_int8.safetensors",
            "sha256": "e" * 64,
            "size_bytes": 1000,
            "variant": "int8",
        },
        {"name": "tokenizer.json", "subfolder": "encoder/", "url": BASE + "encoder/tokenizer.json", "sha256": "f" * 64, "size_bytes": 30},
        {"name": "spiece.model", "subfolder": "encoder/", "url": BASE + "encoder/spiece.model", "sha256": "1" * 64, "size_bytes": 40},
        {"name": "merges.txt", "subfolder": "encoder/", "url": BASE + "encoder/merges.txt", "sha256": "2" * 64, "size_bytes": 50},
        {"name": "vae.gguf", "url": BASE + "vae.gguf", "sha256": "3" * 64, "size_bytes": 60},
    ],
}

CIVITAI = "https://civitai.com/api/download/models/"
TEST_CIVITAI: dict[str, Any] = {
    "id": "test-shared",
    "source": {"provider": "civitai", "repo": "test/shared", "revision": "1"},
    "files": [
        {"name": "encoder_bf16.safetensors", "subfolder": "encoder/", "url": CIVITAI + "1", "sha256": "d" * 64, "size_bytes": 2000},
        {"name": "vae.gguf", "url": CIVITAI + "2", "sha256": "3" * 64, "size_bytes": 60},
    ],
}


def write(directory: Path, name: str, data: dict[str, Any]) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    (directory / name).write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")


def load_with_shared(tmp_path: Path, component: dict[str, Any], requires: list[str] | None = None) -> Registry:
    write(tmp_path / "models", "test-image.yaml", {**TEST_IMAGE, "requires": ["test-shared"] if requires is None else requires})
    write(tmp_path / "shared", f"{component['id']}.yaml", component)
    return Registry.load(tmp_path / "models", tmp_path / "shared")


def test_empty_and_missing_directories_are_valid(tmp_path):
    assert Registry.load(tmp_path).all() == []
    assert Registry.load(tmp_path / "missing").all() == []
    assert Registry.load(tmp_path / "missing", tmp_path / "missing-shared").components() == []


def test_loads_descriptors_and_finds_bindings(registry_dir):
    registry = Registry.load(registry_dir)
    assert [m.id for m in registry.all()] == ["test-image", "test-video"]
    assert registry.binding("test-image", "Q8_0", "wan2gp").model_type == "test_image_q8"
    assert registry.binding("test-image", "fp8", "wan2gp") is None
    assert registry.get("nope") is None
    assert registry.get("test-image").requires == []
    assert registry.requirements("test-image") == []


def test_invalid_descriptor_is_an_error(tmp_path):
    (tmp_path / "broken.yaml").write_text(yaml.safe_dump({**TEST_IMAGE, "capabilities": []}), encoding="utf-8")
    with pytest.raises(RegistryError, match=r"broken\.yaml"):
        Registry.load(tmp_path)


def test_duplicate_ids_are_an_error(tmp_path):
    for name in ("one.yaml", "two.yaml"):
        (tmp_path / name).write_text(yaml.safe_dump(TEST_IMAGE), encoding="utf-8")
    with pytest.raises(RegistryError, match="twice"):
        Registry.load(tmp_path)


def test_bundled_registry_loads():
    registry = Registry.load()
    assert registry.components()
    for model in registry.all():
        assert [component.id for component in registry.requirements(model.id)] == model.requires
    explicit = Registry.load(DEFAULT_DIRECTORY)
    assert explicit.all() == registry.all()
    assert explicit.components() == registry.components()


def civitai_model() -> dict[str, Any]:
    """TEST_IMAGE with its files on civitai.com, so a case can change one file and keep the rest valid."""
    model = copy.deepcopy(TEST_IMAGE)
    model["source"] = {"provider": "civitai", "repo": "test/test-image"}
    for number, quant in enumerate(model["quantizations"], start=1):
        quant["files"][0]["url"] = f"{CIVITAI}{number}"
    return model


QUANT_HF_HOST = r"Value error, \S+: files of a huggingface source download from https://huggingface\.co"
QUANT_CIVITAI_HOST = r"Value error, \S+: files of a civitai source download from https://civitai\.com"
QUANT_DEVICE_NAME = r"quantizations\.0\.files\.0\.name\s+Value error, \S+: {} is a reserved device name on Windows"

INVALID_QUANTIZATION_FILES = {
    "pickle .bin": ({"name": "pytorch_model.bin"}, r"quantizations\.0\s+Value error, pytorch_model\.bin: files of a safetensors"),
    "gguf file in a safetensors quantization": ({"name": "x.gguf"}, r"quantizations\.0\s+Value error, x\.gguf: files of a safetensors"),
    "slash in name": ({"name": "sub/x.safetensors"}, r"quantizations\.0\.files\.0\.name\s+String should match pattern"),
    "device name": ({"name": "CON.safetensors"}, QUANT_DEVICE_NAME.format("CON")),
    "device name in lower case": ({"name": "lpt1.safetensors"}, QUANT_DEVICE_NAME.format("lpt1")),
    "http url": (
        {"url": "http://huggingface.co/test/test-image/bf16"},
        r"Value error, http://\S+: files of a huggingface source download from",
    ),
    "foreign host": ({"url": "https://example.com/bf16"}, r"Value error, https://example\.com/bf16: files of a huggingface source"),
    # The hostname of these URLs is huggingface.co, so only the exact netloc refuses them.
    "user info in url": ({"url": "https://user:pw@huggingface.co/test/test-image/bf16"}, QUANT_HF_HOST),
    "port in url": ({"url": "https://huggingface.co:8443/test/test-image/bf16"}, QUANT_HF_HOST),
    "backslash before @ in url": ({"url": "https://evil.com\\@huggingface.co/test/test-image/bf16"}, QUANT_HF_HOST),
}


@pytest.mark.parametrize(("fields", "reason"), INVALID_QUANTIZATION_FILES.values(), ids=INVALID_QUANTIZATION_FILES.keys())
def test_invalid_quantization_file_is_an_error(tmp_path, fields, reason):
    model = copy.deepcopy(TEST_IMAGE)
    model["quantizations"][0]["files"][0].update(fields)
    write(tmp_path, "test-image.yaml", model)
    with pytest.raises(RegistryError, match=r"test-image\.yaml") as excinfo:
        Registry.load(tmp_path)
    assert re.search(reason, str(excinfo.value)), str(excinfo.value)


def test_loads_civitai_quantization_files(tmp_path):
    write(tmp_path, "test-image.yaml", civitai_model())
    model = Registry.load(tmp_path).get("test-image")
    assert [q.files[0].url for q in model.quantizations] == [CIVITAI + "1", CIVITAI + "2"]


# The hostname of each URL is civitai.com, so only the exact netloc refuses them.
INVALID_CIVITAI_QUANTIZATION_URLS = {
    "user info": "https://user:pw@civitai.com/api/download/models/1",
    "port": "https://civitai.com:8443/api/download/models/1",
    "backslash before @": "https://evil.com\\@civitai.com/x",
    "host in upper case": "https://CIVITAI.com/api/download/models/1",
}


@pytest.mark.parametrize("url", INVALID_CIVITAI_QUANTIZATION_URLS.values(), ids=INVALID_CIVITAI_QUANTIZATION_URLS.keys())
def test_civitai_quantization_url_must_name_the_host_exactly(tmp_path, url):
    model = civitai_model()
    model["quantizations"][0]["files"][0]["url"] = url
    write(tmp_path, "test-image.yaml", model)
    with pytest.raises(RegistryError, match=r"test-image\.yaml") as excinfo:
        Registry.load(tmp_path)
    assert re.search(QUANT_CIVITAI_HOST, str(excinfo.value)), str(excinfo.value)


# --------------------------------------------------------------------------- shared components


def test_loads_shared_components_and_files_per_text_encoder_variant(tmp_path):
    registry = load_with_shared(tmp_path, TEST_SHARED)
    assert [component.id for component in registry.components()] == ["test-shared"]
    (component,) = registry.requirements("test-image")
    assert registry.component("test-shared") == component
    assert registry.component("nope") is None
    common = ["encoder/tokenizer.json", "encoder/spiece.model", "encoder/merges.txt", "vae.gguf"]
    assert [f.path for f in component.files_for("bf16")] == ["encoder/encoder_bf16.safetensors", *common]
    assert [f.path for f in component.files_for("int8")] == ["encoder/encoder_int8.safetensors", *common]
    assert len(component.files_for(None)) == 6


def test_unknown_required_component_is_an_error(tmp_path):
    with pytest.raises(RegistryError, match="test-image.*requires unknown shared component nope"):
        load_with_shared(tmp_path, TEST_SHARED, requires=["test-shared", "nope"])


def test_requiring_a_component_twice_is_an_error(tmp_path):
    with pytest.raises(RegistryError, match="Model test-image requires a shared component twice"):
        load_with_shared(tmp_path, TEST_SHARED, requires=["test-shared", "test-shared"])


def test_unknown_required_component_is_an_error_without_shared_directory(registry_dir):
    write(registry_dir, "test-image.yaml", {**TEST_IMAGE, "requires": ["test-shared"]})
    with pytest.raises(RegistryError, match="unknown shared component test-shared"):
        Registry.load(registry_dir)


def test_component_id_must_match_file_name(tmp_path):
    write(tmp_path / "models", "test-image.yaml", TEST_IMAGE)
    write(tmp_path / "shared", "other-name.yaml", TEST_SHARED)
    with pytest.raises(RegistryError, match=r"other-name\.yaml"):
        Registry.load(tmp_path / "models", tmp_path / "shared")


def broken(change) -> dict[str, Any]:
    component = copy.deepcopy(TEST_SHARED)
    change(component)
    return component


def first_file(**fields):
    def change(component):
        component["files"][0].update(fields)
        for key, value in fields.items():
            if value is None:
                del component["files"][0][key]

    return change


def civitai_first_file(**fields):
    def change(component):
        component.clear()
        component.update(copy.deepcopy(TEST_CIVITAI))
        first_file(**fields)(component)

    return change


def bad_subfolder(subfolder: str):
    """The URL matches the subfolder, so only the subfolder rule can refuse it."""
    return first_file(subfolder=subfolder, url=BASE + subfolder + "encoder_bf16.safetensors")


def disallowed(name: str):
    return first_file(name=name, url=BASE + "encoder/" + name)


def duplicate_last(name: str):
    def change(component):
        component["files"].append({**component["files"][-1], "name": name, "url": BASE + name})

    return change


HF_HOST = r"files of a huggingface source download from https://huggingface\.co"
CIVITAI_HOST = r"files of a civitai source download from https://civitai\.com"
EXPECTED_URL = r"Value error, \S+: expected https://huggingface\.co/test/shared/resolve/"
EXTENSION = r"files\.0\.name\s+Value error, \S+: only \.gguf, \.json, \.model, \.safetensors, \.txt files are allowed"
SUBFOLDER = r"files\.0\.subfolder\s+String should match pattern"


def device_name(field: str, device: str) -> str:
    return rf"files\.0\.{field}\s+Value error, \S+: {device} is a reserved device name on Windows"


# Each case names the error it expects, so a case only passes when its own rule refuses it.
INVALID_COMPONENTS = {
    "missing sha256": (first_file(sha256=None), r"files\.0\.sha256\s+Field required"),
    "missing size_bytes": (first_file(size_bytes=None), r"files\.0\.size_bytes\s+Field required"),
    "short sha256": (first_file(sha256="d" * 63), r"files\.0\.sha256\s+String should match pattern"),
    "http url": (first_file(url=BASE.replace("https://", "http://") + "encoder/encoder_bf16.safetensors"), HF_HOST),
    "foreign host": (first_file(url=BASE.replace("huggingface.co", "example.com") + "encoder/encoder_bf16.safetensors"), HF_HOST),
    "civitai host for a huggingface source": (first_file(url=CIVITAI + "1"), HF_HOST),
    "civitai source: http url": (civitai_first_file(url=CIVITAI.replace("https://", "http://") + "1"), CIVITAI_HOST),
    "civitai source: foreign host": (civitai_first_file(url="https://example.com/api/download/models/1"), CIVITAI_HOST),
    "civitai source: huggingface host": (civitai_first_file(url=BASE + "encoder/encoder_bf16.safetensors"), CIVITAI_HOST),
    # The hostname of these URLs is civitai.com, so only the exact netloc refuses them.
    "civitai source: user info in url": (civitai_first_file(url="https://user:pw@civitai.com/api/download/models/1"), CIVITAI_HOST),
    "civitai source: port in url": (civitai_first_file(url="https://civitai.com:8443/api/download/models/1"), CIVITAI_HOST),
    "civitai source: backslash before @": (civitai_first_file(url="https://evil.com\\@civitai.com/x"), CIVITAI_HOST),
    "civitai source: host in upper case": (civitai_first_file(url="https://CIVITAI.com/api/download/models/1"), CIVITAI_HOST),
    "resolve/main url": (first_file(url="https://huggingface.co/test/shared/resolve/main/encoder/encoder_bf16.safetensors"), EXPECTED_URL),
    "url of another file": (first_file(url=BASE + "encoder/encoder_int8.safetensors"), EXPECTED_URL),
    "url in another subfolder": (first_file(url=BASE + "encoder_bf16.safetensors"), EXPECTED_URL),
    "pickle .bin": (disallowed("pytorch_model.bin"), EXTENSION),
    "pickle .pt": (disallowed("encoder.pt"), EXTENSION),
    "pickle .pkl": (disallowed("encoder.pkl"), EXTENSION),
    "pickle .ckpt": (disallowed("encoder.ckpt"), EXTENSION),
    "python .py": (disallowed("modeling.py"), EXTENSION),
    "no extension": (disallowed("encoder"), EXTENSION),
    "unknown variant": (first_file(variant="fp8"), r"files\.0\.variant\s+Input should be 'bf16' or 'int8'"),
    "path traversal in subfolder": (bad_subfolder("../"), SUBFOLDER),
    "subfolder without trailing slash": (bad_subfolder("encoder"), SUBFOLDER),
    "absolute subfolder": (bad_subfolder("/encoder/"), SUBFOLDER),
    "trailing dot in subfolder": (bad_subfolder("encoder./"), SUBFOLDER),
    "device name": (disallowed("CON.json"), device_name("name", "CON")),
    "device name in lower case": (disallowed("aux.safetensors"), device_name("name", "aux")),
    "device name with two extensions": (disallowed("nul.tar.json"), device_name("name", "nul")),
    "device name COM1": (disallowed("Com1.txt"), device_name("name", "Com1")),
    "device name COM9": (disallowed("COM9.json"), device_name("name", "COM9")),
    "device name LPT9": (disallowed("lpt9.model"), device_name("name", "lpt9")),
    "device name without extension": (disallowed("PRN"), device_name("name", "PRN")),
    "device name as subfolder": (bad_subfolder("aux/"), device_name("subfolder", "aux")),
    "device name as nested subfolder": (bad_subfolder("encoder/con/"), device_name("subfolder", "con")),
    "device name with extension as subfolder": (bad_subfolder("LPT1.d/"), device_name("subfolder", "LPT1")),
    "slash in name": (first_file(name="encoder/encoder_bf16.safetensors", subfolder=""), r"files\.0\.name\s+String should match pattern"),
    "duplicate path": (duplicate_last("vae.gguf"), r"vae\.gguf is listed twice"),
    "duplicate path in another case": (duplicate_last("VAE.gguf"), r"VAE\.gguf is listed twice \(paths ignore case\)"),
    "missing revision": (lambda c: c["source"].pop("revision"), r"source\.revision is required"),
    "branch as revision": (lambda c: c["source"].update(revision="main"), r"source\.revision 'main' is not a commit"),
    "unknown field": (first_file(note="extra"), r"files\.0\.note\s+Extra inputs are not permitted"),
}


@pytest.mark.parametrize(("change", "reason"), INVALID_COMPONENTS.values(), ids=INVALID_COMPONENTS.keys())
def test_invalid_shared_component_is_an_error(tmp_path, change, reason):
    with pytest.raises(RegistryError, match=r"test-shared\.yaml") as excinfo:
        load_with_shared(tmp_path, broken(change))
    assert re.search(reason, str(excinfo.value)), str(excinfo.value)


NEAR_DEVICE_NAMES = {
    "console.json": ("encoder/", "console.json"),
    "CONFIG.json": ("encoder/", "CONFIG.json"),
    "com10.txt": ("encoder/", "com10.txt"),
    "aux_config.json": ("encoder/", "aux_config.json"),
    "nul-free.model": ("encoder/", "nul-free.model"),
    "subfolder conx/": ("conx/", "encoder_bf16.safetensors"),
    "subfolder com10/lpt/": ("com10/lpt/", "encoder_bf16.safetensors"),
}


@pytest.mark.parametrize(("subfolder", "name"), NEAR_DEVICE_NAMES.values(), ids=NEAR_DEVICE_NAMES.keys())
def test_names_close_to_device_names_are_allowed(tmp_path, subfolder, name):
    component = broken(first_file(subfolder=subfolder, name=name, url=BASE + subfolder + name))
    (loaded,) = load_with_shared(tmp_path, component).requirements("test-image")
    assert loaded.files[0].path == subfolder + name


def test_loads_civitai_shared_component(tmp_path):
    (component,) = load_with_shared(tmp_path, TEST_CIVITAI).requirements("test-image")
    assert [(f.path, f.url) for f in component.files] == [("encoder/encoder_bf16.safetensors", CIVITAI + "1"), ("vae.gguf", CIVITAI + "2")]


def other_component(name: str, subfolder: str, sha256: str) -> dict[str, Any]:
    file = {"name": name, "subfolder": subfolder, "url": BASE + subfolder + name, "sha256": sha256, "size_bytes": 30}
    return {"id": "other-shared", "source": TEST_SHARED["source"], "files": [file]}


INSTALL_PATHS = {
    "same shared path in another case": (
        other_component("Tokenizer.json", "Encoder/", "9" * 64),
        "shared component other-shared and shared component test-shared name different files at one path: "
        "Encoder/Tokenizer.json, encoder/tokenizer.json",
    ),
    "shared root file named like a quantization file": (
        other_component("TEST-IMAGE-Q8.gguf", "", "9" * 64),
        "shared component other-shared and model test-image name different files at one path: TEST-IMAGE-Q8.gguf, test-image-q8.gguf",
    ),
    "same file in two components": (other_component("tokenizer.json", "encoder/", "f" * 64), None),
    "quantization file as a shared root file": (other_component("test-image-q8.gguf", "", "b" * 64), None),
}


@pytest.mark.parametrize(("other", "clash"), INSTALL_PATHS.values(), ids=INSTALL_PATHS.keys())
def test_one_install_path_holds_one_file(tmp_path, other, clash):
    write(tmp_path / "shared", "other-shared.yaml", other)
    if clash is None:
        assert len(load_with_shared(tmp_path, TEST_SHARED).components()) == 2
    else:
        with pytest.raises(RegistryError, match=rf"{re.escape(clash)} \(paths ignore case\)"):
            load_with_shared(tmp_path, TEST_SHARED)


def test_duplicate_component_ids_are_an_error():
    from cinewright.api.schemas import SharedComponent

    component = SharedComponent.model_validate(TEST_SHARED)
    with pytest.raises(RegistryError, match="twice"):
        Registry(shared=[component, component])
