"""Launcher: version smoke test, refusal of non-loopback binds, second start only opens the browser."""

from __future__ import annotations

import logging
import subprocess
import sys
from logging.handlers import RotatingFileHandler

import pytest

from cinewright import __version__, launcher


@pytest.fixture(autouse=True)
def detach_log_files():
    yield
    root = logging.getLogger()
    for handler in [h for h in root.handlers if isinstance(h, RotatingFileHandler)]:
        root.removeHandler(handler)
        handler.close()


def test_version_flag_prints_name_and_version(capsys):
    assert launcher.main(["--version"]) == 0
    assert capsys.readouterr().out == f"cinewright {__version__}\n"


def test_python_dash_m_cinewright_version():
    result = subprocess.run([sys.executable, "-m", "cinewright", "--version"], capture_output=True, text=True, timeout=60, check=False)
    assert result.returncode == 0
    assert result.stdout.strip() == f"cinewright {__version__}"


def test_non_loopback_bind_is_refused(monkeypatch, home):
    monkeypatch.setenv("CINEWRIGHT_HOST", "0.0.0.0")
    monkeypatch.setattr(launcher, "serve", lambda config: (_ for _ in ()).throw(AssertionError("must not start")))
    assert launcher.main([]) == 2
    assert "is refused" in (home / "logs" / "cinewright.log").read_text(encoding="utf-8")


def test_second_start_only_opens_the_browser(monkeypatch):
    opened = []
    monkeypatch.setattr(launcher, "is_running", lambda config: True)
    monkeypatch.setattr(launcher.webbrowser, "open", opened.append)
    monkeypatch.setattr(launcher, "serve", lambda config: (_ for _ in ()).throw(AssertionError("must not start")))
    assert launcher.main([]) == 0
    assert opened == ["http://127.0.0.1:8737/"]


def test_first_start_serves(monkeypatch):
    started = []
    monkeypatch.setattr(launcher, "is_running", lambda config: False)
    monkeypatch.setattr(launcher, "serve", lambda config: started.append(config) or True)
    assert launcher.main([]) == 0
    assert started[0].port == 8737


def test_failed_startup_exits_non_zero(monkeypatch):
    monkeypatch.setattr(launcher, "is_running", lambda config: False)
    monkeypatch.setattr(launcher, "serve", lambda config: False)
    assert launcher.main([]) == 1
