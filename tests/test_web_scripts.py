"""Runs the web UI's Node tests so pytest covers them too.

- `node --test` for the build scripts (web/scripts/*.test.mjs: token generator, dev-proxy Origin guard).
- Vitest for the component tests (web/src/**/*.test.tsx). Vitest comes from web/node_modules, which
  `npm ci` creates; in CI the Build step (`python scripts/build.py`) runs it before the Test step.
"""

import os
import shutil
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
WEB = ROOT / "web"
NODE = shutil.which("node")
NODE_TESTS = sorted(path.relative_to(ROOT).as_posix() for path in (WEB / "scripts").glob("*.test.mjs"))
COMPONENT_TESTS = sorted(path.relative_to(WEB).as_posix() for path in (WEB / "src").rglob("*.test.ts*"))
VITEST = WEB / "node_modules" / "vitest" / "vitest.mjs"


@pytest.mark.skipif(NODE is None, reason="node is not on PATH")
def test_web_script_node_tests_pass():
    assert "web/scripts/gen-tokens.test.mjs" in NODE_TESTS
    result = subprocess.run(  # noqa: S603 - fixed test invocation
        [NODE, "--test", *NODE_TESTS],
        cwd=ROOT,
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr


def test_web_component_tests_pass():
    if NODE is None:
        pytest.skip("node is not on PATH")
    if not (WEB / "node_modules").is_dir():
        pytest.skip("web/node_modules is missing; run `npm ci` in web/ (python scripts/build.py does)")
    assert "src/views/SettingsView.test.tsx" in COMPONENT_TESTS
    assert VITEST.is_file(), "web/node_modules has no vitest; run `npm ci` in web/"
    result = subprocess.run(  # noqa: S603 - fixed test invocation
        [NODE, str(VITEST), "run"],
        cwd=WEB,
        env={**os.environ, "NO_COLOR": "1"},
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=600,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr
