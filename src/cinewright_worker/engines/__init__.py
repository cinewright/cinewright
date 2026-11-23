"""Engine adapters (`mock`, `wan2gp`) and the helpers they share with the worker loop."""

from __future__ import annotations

import os

# File extension <-> media type for inputs and outputs. Fixed table instead of `mimetypes`, whose
# answers depend on the Windows registry. Every type here is accepted by the core for uploads.
MEDIA_TYPES = {
    ".png": "image/png",
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".webp": "image/webp",
    ".mp4": "video/mp4",
    ".webm": "video/webm",
    ".wav": "audio/wav",
    ".mp3": "audio/mpeg",
    ".flac": "audio/flac",
}
EXTENSIONS = {
    "image/png": ".png",
    "image/jpeg": ".jpg",
    "image/webp": ".webp",
    "video/mp4": ".mp4",
    "video/webm": ".webm",
    "audio/wav": ".wav",
    "audio/x-wav": ".wav",
    "audio/mpeg": ".mp3",
    "audio/flac": ".flac",
}
GENERIC_MEDIA_TYPE = "application/octet-stream"


class EngineError(RuntimeError):
    """An engine failure with a stable machine code, e.g. `oom`, `engine_failed`, `model_unavailable`."""

    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code
        self.message = message


def media_type_for(path: str) -> str:
    return MEDIA_TYPES.get(os.path.splitext(path)[1].lower(), GENERIC_MEDIA_TYPE)
