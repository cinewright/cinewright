"""Registry loading: YAML descriptors validated with ModelDescriptor; empty is valid."""

from __future__ import annotations

import pytest
import yaml
from conftest import TEST_IMAGE

from cinewright.core.registry import Registry, RegistryError


def test_empty_and_missing_directories_are_valid(tmp_path):
    assert Registry.load(tmp_path).all() == []
    assert Registry.load(tmp_path / "missing").all() == []


def test_loads_descriptors_and_finds_bindings(registry_dir):
    registry = Registry.load(registry_dir)
    assert [m.id for m in registry.all()] == ["test-image", "test-video"]
    assert registry.binding("test-image", "Q8_0", "wan2gp").model_type == "test_image_q8"
    assert registry.binding("test-image", "fp8", "wan2gp") is None
    assert registry.get("nope") is None


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
    Registry.load()
