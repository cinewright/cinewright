"""Wake coroutines on the event loop when another thread changes state.

Sync routes run in FastAPI's thread pool, while lease long-polls and the SSE stream wait on the
event loop. Both cross that boundary only through `loop.call_soon_threadsafe`.
"""

from __future__ import annotations

import asyncio
import threading
from dataclasses import dataclass, field
from typing import Any


class Notifier:
    """Edge-triggered wake-up. Take `snapshot()` before checking state, then wait on it."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._loop: asyncio.AbstractEventLoop | None = None
        self._event: asyncio.Event | None = None

    def snapshot(self) -> asyncio.Event:
        """Must run on the event loop. The returned event is set by the next `notify()`."""
        loop = asyncio.get_running_loop()
        with self._lock:
            if self._loop is not loop or self._event is None:
                self._loop = loop
                self._event = asyncio.Event()
            return self._event

    def notify(self) -> None:
        """Callable from any thread; a no-op while nobody waits."""
        with self._lock:
            loop, event = self._loop, self._event
            self._event = None
        if loop is not None and event is not None:
            _call_soon(loop, event.set)


@dataclass(eq=False)
class Subscription:
    """A queue on one event loop that receives broadcast items; `None` ends it."""

    loop: asyncio.AbstractEventLoop
    queue: asyncio.Queue[Any] = field(default_factory=asyncio.Queue)


class Broadcaster:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._subscriptions: set[Subscription] = set()

    def subscribe(self) -> Subscription:
        subscription = Subscription(asyncio.get_running_loop())
        with self._lock:
            self._subscriptions.add(subscription)
        return subscription

    def unsubscribe(self, subscription: Subscription) -> None:
        with self._lock:
            self._subscriptions.discard(subscription)

    def publish(self, item: Any) -> None:
        """Callable from any thread."""
        with self._lock:
            subscriptions = list(self._subscriptions)
        for subscription in subscriptions:
            if not _call_soon(subscription.loop, subscription.queue.put_nowait, item):
                self.unsubscribe(subscription)

    def close(self) -> None:
        """End every open subscription."""
        self.publish(None)


def _call_soon(loop: asyncio.AbstractEventLoop, callback: Any, *args: Any) -> bool:
    try:
        loop.call_soon_threadsafe(callback, *args)
    except RuntimeError:  # the loop is closed
        return False
    return True
