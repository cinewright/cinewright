"""Capability registry: one YAML `ModelDescriptor` per model under `src/cinewright/registry/models/`.

Engine files several models need (text encoders, VAEs, upscalers, tokenizer and config files) are
`SharedComponent`s under `src/cinewright/registry/shared/`; a descriptor names them in `requires`.

An empty or missing directory is a valid, empty registry. An invalid descriptor stops the core,
because the registry ships with the application and a broken entry is a bug.
"""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path

import yaml
from pydantic import BaseModel, ValidationError

from cinewright.api.schemas import EngineBinding, ModelDescriptor, SharedComponent

REGISTRY_ROOT = Path(__file__).resolve().parent.parent / "registry"
DEFAULT_DIRECTORY = REGISTRY_ROOT / "models"
DEFAULT_SHARED_DIRECTORY = REGISTRY_ROOT / "shared"


class RegistryError(ValueError):
    pass


def _load_directory[Entry: BaseModel](directory: Path, model: type[Entry], kind: str) -> list[tuple[Path, Entry]]:
    if not directory.is_dir():
        return []
    entries = []
    for path in sorted(directory.glob("*.yaml")):
        try:
            data = yaml.safe_load(path.read_text(encoding="utf-8"))
            entries.append((path, model.model_validate(data)))
        except (yaml.YAMLError, ValidationError) as exc:
            raise RegistryError(f"Invalid {kind} {path.name}: {exc}") from exc
    return entries


def _check_install_paths(components: Iterable[SharedComponent], models: Iterable[ModelDescriptor]) -> None:
    """Refuse two different files at one path below the checkpoint root.

    The model store (#3) installs every shared file at `<subfolder><name>` below one checkpoint
    root. This check assumes #3 installs every quantization file at `<name>` there too; #3 must
    confirm that layout. On Windows that root ignores case, so paths are compared case-folded.
    The same path with the same SHA-256 is one file and stays allowed.
    """
    entries = [(f"shared component {c.id}", f.path, f.sha256) for c in components for f in c.files]
    entries += [(f"model {m.id}", f.name, f.sha256) for m in models for q in m.quantizations for f in q.files]
    first: dict[str, tuple[str, str, str]] = {}
    for owner, path, sha256 in entries:
        first_owner, first_path, first_sha256 = first.setdefault(path.casefold(), (owner, path, sha256))
        if first_sha256 != sha256:
            raise RegistryError(f"{first_owner} and {owner} name different files at one path: {first_path}, {path} (paths ignore case).")


class Registry:
    def __init__(self, models: Iterable[ModelDescriptor] = (), shared: Iterable[SharedComponent] = ()) -> None:
        self._shared: dict[str, SharedComponent] = {}
        for component in shared:
            if component.id in self._shared:
                raise RegistryError(f"Shared component id {component.id} appears twice in the registry.")
            self._shared[component.id] = component
        self._models: dict[str, ModelDescriptor] = {}
        for model in models:
            if model.id in self._models:
                raise RegistryError(f"Model id {model.id} appears twice in the registry.")
            for component_id in model.requires:
                if component_id not in self._shared:
                    raise RegistryError(f"Model {model.id} requires unknown shared component {component_id}.")
            if len(set(model.requires)) != len(model.requires):
                raise RegistryError(f"Model {model.id} requires a shared component twice.")
            self._models[model.id] = model
        _check_install_paths(self._shared.values(), self._models.values())

    @classmethod
    def load(cls, directory: Path | None = None, shared_directory: Path | None = None) -> Registry:
        """Load descriptors from `directory` and shared components from `shared_directory`.

        Without arguments, or with the bundled `directory`, both come from the registry bundled
        with the application. A custom `directory` without a `shared_directory` has no shared
        components.
        """
        if shared_directory is None and (directory is None or directory.resolve() == DEFAULT_DIRECTORY):
            shared_directory = DEFAULT_SHARED_DIRECTORY
        directory = DEFAULT_DIRECTORY if directory is None else directory
        shared = []
        if shared_directory is not None:
            for path, component in _load_directory(shared_directory, SharedComponent, "shared component"):
                if component.id != path.stem:
                    raise RegistryError(f"Invalid shared component {path.name}: id {component.id} differs from the file name.")
                shared.append(component)
        models = [model for _, model in _load_directory(directory, ModelDescriptor, "model descriptor")]
        return cls(models, shared)

    def all(self) -> list[ModelDescriptor]:
        return list(self._models.values())

    def get(self, model_id: str) -> ModelDescriptor | None:
        return self._models.get(model_id)

    def components(self) -> list[SharedComponent]:
        return list(self._shared.values())

    def component(self, component_id: str) -> SharedComponent | None:
        return self._shared.get(component_id)

    def requirements(self, model_id: str) -> list[SharedComponent]:
        """Shared components the model requires, in descriptor order; empty for an unknown model."""
        model = self._models.get(model_id)
        return [] if model is None else [self._shared[component_id] for component_id in model.requires]

    def binding(self, model_id: str, quant: str, engine: str) -> EngineBinding | None:
        model = self._models.get(model_id)
        if model is None:
            return None
        return next((b for b in model.engines if b.quant == quant and b.engine == engine), None)
