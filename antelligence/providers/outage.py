"""Wait out a provider outage; never turn it into a different answer (decision 2026-10-13, mistake-2039).

``complete_with_outage_retry`` sends the SAME request again while the provider is unavailable (HTTP 401, 429, 5xx or a
transport error, i.e. :class:`~antelligence.providers.base.ProviderUnavailable`), sleeping 15 s, 30 s, 60 s, then 120 s
between tries, until ``max_wait_s`` (default 600 s) of waiting has passed. Past that it raises :class:`QueenUnavailable`,
a :class:`~antelligence.kernel.decision.DecisionAbort`: it ends the run as an error and no retry, fallback or error
decision may absorb it. Any other failure (an invalid reply, a spent budget, a bad request) is raised untouched on the
first try.

Retries are the same queen call, so they do not count against the per-run call cap; each attempt still goes through the
provider stack (usage ledger and Nous stop) as before. ``sleep`` and ``clock`` are injectable so tests run instantly.
"""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable, Optional

from antelligence.kernel.decision import DecisionAbort
from antelligence.providers.base import ChatRequest, ChatResponse, Provider, ProviderUnavailable
from antelligence.providers.redact import scrub

INITIAL_BACKOFF_S = 15.0
BACKOFF_FACTOR = 2.0
MAX_BACKOFF_S = 120.0
DEFAULT_MAX_WAIT_S = 600.0

# Exit code of the batch drivers (scripts/v41_run.py, scripts/v41_sandbox.py) when the queen provider stayed down past
# the maximum wait. 75 is EX_TEMPFAIL: nothing is wrong with the study, run it again later; the caches resume it.
EXIT_QUEEN_UNAVAILABLE = 75


class QueenUnavailable(DecisionAbort):
    """The queen provider stayed unavailable past the maximum wait. The run is an error, never a fallback."""


@dataclass
class OutagePolicy:
    """How long and how the queen waits out a provider outage. ``retries`` and ``waited_s`` accumulate over the policy's
    life (one per queen or proposer), for the run summary."""

    max_wait_s: float = DEFAULT_MAX_WAIT_S
    initial_backoff_s: float = INITIAL_BACKOFF_S
    factor: float = BACKOFF_FACTOR
    max_backoff_s: float = MAX_BACKOFF_S
    sleep: Callable[[float], Awaitable[Any]] = asyncio.sleep
    clock: Callable[[], float] = time.monotonic
    retries: int = field(default=0, init=False)
    waited_s: float = field(default=0.0, init=False)

    def __post_init__(self) -> None:
        if self.max_wait_s < 0 or self.initial_backoff_s <= 0 or self.factor < 1 or self.max_backoff_s < self.initial_backoff_s:
            raise ValueError("outage policy needs max_wait_s >= 0, initial_backoff_s > 0, factor >= 1, max_backoff_s >= initial_backoff_s")


async def complete_with_outage_retry(provider: Provider, request: ChatRequest, policy: Optional[OutagePolicy] = None) -> ChatResponse:
    policy = policy or OutagePolicy()
    started = policy.clock()
    delay = policy.initial_backoff_s
    while True:
        try:
            return await provider.complete(request)
        except ProviderUnavailable as exc:
            remaining = policy.max_wait_s - (policy.clock() - started)
            if remaining <= 0:
                raise QueenUnavailable(scrub(f"queen provider unavailable for {policy.clock() - started:.0f}s "
                                             f"(max wait {policy.max_wait_s:.0f}s): {exc}"[:300])) from None
            wait = min(delay, remaining)
            policy.retries += 1
            policy.waited_s += wait
            await policy.sleep(wait)
            delay = min(delay * policy.factor, policy.max_backoff_s)
