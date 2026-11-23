"""Capability registry: one YAML `ModelDescriptor` per model under `src/cinewright/registry/models/`.

An empty or missing directory is a valid, empty registry. An invalid descriptor stops the core,
because the registry ships with the application and a broken entry is a bug.
"""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path

import yaml
from pydantic import ValidationError

from cinewright.api.schemas import EngineBinding, ModelDescriptor

DEFAULT_DIRECTORY = Path(__file__).resolve().parent.parent / "registry" / "models"


class RegistryError(ValueError):
    pass


class Registry:
    def __init__(self, models: Iterable[ModelDescriptor] = ()) -> None:
        self._models: dict[str, ModelDescriptor] = {}
        for model in models:
            if model.id in self._models:
                raise RegistryError(f"Model id {model.id} appears twice in the registry.")
            self._models[model.id] = model

    @classmethod
    def load(cls, directory: Path | None = None) -> Registry:
        directory = DEFAULT_DIRECTORY if directory is None else directory
        if not directory.is_dir():
            return cls()
        models = []
        for path in sorted(directory.glob("*.yaml")):
            try:
                data = yaml.safe_load(path.read_text(encoding="utf-8"))
                models.append(ModelDescriptor.model_validate(data))
            except (yaml.YAMLError, ValidationError) as exc:
                raise RegistryError(f"Invalid model descriptor {path.name}: {exc}") from exc
        return cls(models)

    def all(self) -> list[ModelDescriptor]:
        return list(self._models.values())

    def get(self, model_id: str) -> ModelDescriptor | None:
        return self._models.get(model_id)

    def binding(self, model_id: str, quant: str, engine: str) -> EngineBinding | None:
        model = self._models.get(model_id)
        if model is None:
            return None
        return next((b for b in model.engines if b.quant == quant and b.engine == engine), None)
