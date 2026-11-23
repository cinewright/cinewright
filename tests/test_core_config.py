"""Configuration from the environment and the data directory layout."""

from __future__ import annotations

from pathlib import Path

import pytest

from cinewright.config import load_config
from cinewright.paths import DataPaths, default_home


def test_defaults_and_overrides(tmp_path):
    config = load_config({"CINEWRIGHT_HOME": str(tmp_path)})
    assert (config.home, config.host, config.port, config.bind) == (tmp_path, "127.0.0.1", 8737, "127.0.0.1:8737")
    assert config.loopback
    custom = load_config({"CINEWRIGHT_HOME": str(tmp_path), "CINEWRIGHT_HOST": "0.0.0.0", "CINEWRIGHT_PORT": "9000"})  # noqa: S104
    assert (custom.port, custom.base_url, custom.loopback) == (9000, "http://0.0.0.0:9000", False)


@pytest.mark.parametrize("port", ["0", "70000", "http", "-1"])
def test_invalid_port_is_refused(port):
    with pytest.raises(ValueError, match="CINEWRIGHT_PORT"):
        load_config({"CINEWRIGHT_PORT": port})


def test_default_home_per_platform(tmp_path):
    assert default_home({"LOCALAPPDATA": "C:/Users/x/AppData/Local"}, "win32") == Path("C:/Users/x/AppData/Local/Cinewright")
    assert default_home({"XDG_DATA_HOME": str(tmp_path)}, "linux") == tmp_path / "cinewright"
    assert default_home({"XDG_DATA_HOME": "relative"}, "linux") == Path.home() / ".local" / "share" / "cinewright"
    assert default_home({}, "linux") == Path.home() / ".local" / "share" / "cinewright"


def test_data_layout(tmp_path):
    paths = DataPaths(tmp_path)
    paths.ensure()
    assert {p.name for p in tmp_path.iterdir()} == {"db", "assets", "models", "engines", "logs"}
    sha = "ab" + "0" * 62
    assert paths.asset_file(sha) == tmp_path / "assets" / "ab" / sha
