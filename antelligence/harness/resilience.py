"""Pause and retry a step when the disk or the local model server stalls, instead of failing the whole run.

Why: on Eshaan's Mac the model cache lives on the external drive WD_BLACK, which froze on 2026-11-01 in the middle
of the night run (and once before, for five days in April). Reads failed with EINTR or EIO, a model server hung while
loading, and three stages of the run were lost. A harness step is small, so the right move is: notice, wait for the
drive to answer again, then ask the same step again. Every wait is logged, so the report can say how long the run lost.
"""

from __future__ import annotations

import asyncio
import errno
import os
import threading
import time
from pathlib import Path
from typing import Any, Awaitable, Callable, List, Optional, Sequence, TypeVar

from antelligence.providers.base import ProviderUnavailable

T = TypeVar("T")

# errors that come from a stalled or busy disk, not from a bad program
TRANSIENT_ERRNO = frozenset({errno.EIO, errno.EINTR, errno.EAGAIN, errno.ETIMEDOUT, errno.ENXIO, errno.EBUSY,
                             getattr(errno, "ESTALE", 116), errno.EDEADLK})
WAITS = (15.0, 30.0, 60.0, 120.0, 300.0, 600.0)      # about 19 minutes of patience per step, then give up


def transient(exc: BaseException) -> bool:
    """Is this a failure that may pass if we wait (a stalled disk or a model server that isn't answering yet)?"""
    if isinstance(exc, ProviderUnavailable):
        return True
    if isinstance(exc, (TimeoutError, asyncio.TimeoutError)):
        return True
    if isinstance(exc, OSError) and exc.errno in TRANSIENT_ERRNO:
        return True
    return False


def drive_answers(path: Optional[Path], timeout_s: float = 10.0) -> bool:
    """Can we list ``path`` within ``timeout_s``? A frozen drive hangs here rather than failing, so this runs the
    listing in a thread and gives up on waiting. ``None`` means there is no drive to check."""
    if path is None:
        return True
    out: List[bool] = []

    def look() -> None:
        try:
            os.listdir(path)
            out.append(True)
        except OSError:
            out.append(False)

    t = threading.Thread(target=look, daemon=True)
    t.start()
    t.join(timeout_s)
    return bool(out and out[0])


class Patience:
    """Retries a step on transient failures, waiting longer each time, and keeps a log of every wait."""

    def __init__(self, waits: Sequence[float] = WAITS, drive: Optional[Path] = None,
                 sleep: Callable[[float], Awaitable[Any]] = asyncio.sleep, log: Callable[[str], None] = print):
        self.waits, self.drive, self.sleep, self.log = tuple(waits), drive, sleep, log
        self.events: List[dict] = []

    @property
    def waited_s(self) -> float:
        return sum(e["waited_s"] for e in self.events)

    async def run(self, what: str, fn: Callable[[], Awaitable[T]]) -> T:
        for i, wait in enumerate((*self.waits, None)):
            try:
                return await fn()
            except Exception as exc:  # noqa: BLE001 - sorted by transient() below
                if wait is None or not transient(exc):
                    raise
                start = time.monotonic()
                self.log(f"[harness] {what}: {type(exc).__name__}: {str(exc)[:120]}. Waiting {wait:.0f}s, "
                         f"then trying again ({i + 1}/{len(self.waits)})")
                await self.sleep(wait)
                while not drive_answers(self.drive):           # still frozen: keep waiting, a minute at a time
                    self.log(f"[harness] the drive {self.drive} still doesn't answer; waiting 60s")
                    await self.sleep(60.0)
                self.events.append({"what": what, "error": type(exc).__name__, "waited_s":
                                    round(time.monotonic() - start, 1)})
        raise AssertionError("unreachable")


def read_text(path: Path, waits: Sequence[float] = (5.0, 15.0, 30.0, 60.0), sleep: Callable[[float], None] = time.sleep
              ) -> str:
    """``Path.read_text`` that waits and tries again on a stalled disk."""
    for wait in (*waits, None):
        try:
            return Path(path).read_text()
        except OSError as exc:
            if wait is None or exc.errno not in TRANSIENT_ERRNO:
                raise
            sleep(wait)
    raise AssertionError("unreachable")
