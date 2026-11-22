"""Cinewright: local image, video and audio generation with open-weight models."""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("cinewright")
except PackageNotFoundError:  # running from a source tree without installation
    __version__ = "0.0.0"

__all__ = ["__version__"]
