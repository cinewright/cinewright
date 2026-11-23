"""Time helpers. The database stores ISO 8601 strings in UTC with a fixed width, so they sort as text."""

from __future__ import annotations

from collections.abc import Callable
from datetime import UTC, datetime

Clock = Callable[[], datetime]


def utcnow() -> datetime:
    return datetime.now(UTC)


def to_iso(moment: datetime) -> str:
    return moment.astimezone(UTC).isoformat(timespec="microseconds")


def from_iso(text: str | None) -> datetime | None:
    return datetime.fromisoformat(text) if text else None
