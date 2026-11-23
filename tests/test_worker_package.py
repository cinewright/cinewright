"""The worker package stays standard library only, so it can run inside engine environments."""

from __future__ import annotations

import ast
import sys
from pathlib import Path

import cinewright_worker

PACKAGE = Path(cinewright_worker.__file__).parent


def imported_modules(path: Path) -> set[str]:
    modules: set[str] = set()
    for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
        if isinstance(node, ast.Import):
            modules.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            modules.add(node.module.split(".")[0])
    return modules


def test_worker_imports_only_the_standard_library():
    # torch is imported lazily inside the Wan2GP adapter, where Wan2GP's environment provides it.
    allowed = set(sys.stdlib_module_names) | {"cinewright_worker", "__future__"}
    offenders = {
        str(path.relative_to(PACKAGE)): sorted(imported_modules(path) - allowed - ({"torch"} if path.name == "wan2gp.py" else set()))
        for path in PACKAGE.rglob("*.py")
    }
    assert {name: mods for name, mods in offenders.items() if mods} == {}


def test_every_module_uses_postponed_annotations():
    # `X | None` annotations must not be evaluated on Python 3.10 dataclasses and signatures.
    sources = {str(path.relative_to(PACKAGE)): path.read_text("utf-8") for path in PACKAGE.rglob("*.py")}
    missing = [name for name, text in sources.items() if "def " in text and "from __future__ import annotations" not in text]
    assert missing == []
