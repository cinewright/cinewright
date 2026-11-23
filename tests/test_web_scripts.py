"""Runs the Node tests of the web scripts (web/scripts/*.test.mjs: token generator, dev-proxy Origin guard) so pytest covers them too."""

import shutil
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
NODE = shutil.which("node")
NODE_TESTS = sorted(path.relative_to(ROOT).as_posix() for path in (ROOT / "web" / "scripts").glob("*.test.mjs"))


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
