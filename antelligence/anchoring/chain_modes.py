"""Mode C (anchors, asynchronous) and mode D (board entries on chain) over ``backend/chain/v4_1.py``.

Both adapters take a ``ChainClientV41``-shaped client. They never hold a key and never import the
client module, so the kernel side stays free of web3 unless a chain is configured.

Mode C never blocks: :meth:`ChainAnchorMode.submit` is one SQLite INSERT (the S1 persistent queue).
Somebody else calls :meth:`ChainAnchorMode.pump`, from a background thread or at run end.

Mode D posts one batch per decision round, then waits for inclusion for at most ``timeout_s``
(goal 4.4: 60 s). Entries whose batch is not included in time are *invisible* for that round; the
timeout is logged and the batch stays in the queue, so it can still land later.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Set

from antelligence.kernel.board import ZERO_HASH, Entry

BATCH_TIMEOUT_S = 60.0

CONFIRMED = "confirmed"


class ChainModeError(RuntimeError):
    pass


# --------------------------------------------------------------------------------------------------
# Mode C
# --------------------------------------------------------------------------------------------------

class ChainAnchorMode:
    """Queue ``AnchorRegistry.anchor`` calls. ``prev_root`` is tracked per run, as the contract demands."""

    def __init__(self, client: Any) -> None:
        self.client = client
        self._last_root: Dict[str, bytes] = {}
        self.jobs: Dict[tuple, int] = {}

    def submit(self, run_id: str, size: int, root: bytes) -> int:
        """Enqueue and return the job id. No network call, no waiting."""
        prev = self._last_root.get(run_id, b"\x00" * 32)
        job = self.client.enqueue_anchor(run_id, size, root, prev)
        self._last_root[run_id] = bytes(root)
        self.jobs[(run_id, size)] = job
        return job

    def submit_once(self, single_shot_id: bytes, digest: bytes) -> int:
        job = self.client.enqueue_anchor_once(single_shot_id, digest)
        self.jobs[("once", bytes(single_shot_id).hex())] = job
        return job

    def pump(self):
        """Advance the queue one step (send, poll, replace). Safe to call from a worker thread."""
        return self.client.pump()

    def job(self, job_id: int) -> Optional[dict]:
        return self.client.queue.get(job_id)

    def tx_hash(self, job_id: int) -> Optional[str]:
        job = self.job(job_id)
        return job["tx_hashes"][-1] if job and job["tx_hashes"] else None

    def confirmed(self, job_id: int) -> bool:
        job = self.job(job_id)
        return bool(job and job["status"] == CONFIRMED)


# --------------------------------------------------------------------------------------------------
# Mode D
# --------------------------------------------------------------------------------------------------

# Integer payload fields in the INTERFACES section 4 table order. Fixed-length triples are flattened;
# strings and variable-length lists are never posted (free text does not go on chain, the id commits to it).
_COMPACT_FIELDS: Dict[str, Sequence[str]] = {
    "task": ("until_tick",),
    "knowledge": ("region_id", "centroid_dmm", "p_tumour_ppm", "volume_mm3", "unc_ppm"),
    "evidence": ("squad", "pos_vox", "look_p_ppm", "p_tumour_ppm", "dose_ug", "released_ug"),
    "route": ("squad", "priority"),
    "decision": ("squad", "round", "release", "amount_level", "p_target_ppm", "p_release_ppm", "p_amount_ppm", "deferred"),
    "edit": ("round", "prediction.delta_ppm", "kept"),
    "alarm": ("value_ppm",),
}
KEPT_NULL = 2   # `kept` is 0/1/null; null is posted as 2 (INTERFACES is silent, see the S2 report)


def payload_ints(entry: Entry) -> List[int]:
    out: List[int] = []
    for key in _COMPACT_FIELDS[entry.type]:
        value: Any = entry.payload
        for part in key.split("."):
            value = value[part]
        if key == "kept" and value is None:
            value = KEPT_NULL
        out.extend(value if isinstance(value, list) else [value])
    return [int(v) for v in out]


def compact_entry(entry: Entry) -> bytes:
    from chain.v4_1 import encode_entry_compact   # backend/ is on sys.path wherever a chain is used

    return encode_entry_compact(entry.seq, entry.type, entry.id, entry.prev_hash, payload_ints(entry))


@dataclass
class RoundResult:
    visible: Set[int]                      # seqs included on chain by the end of the wait
    invisible: Set[int]                    # seqs of this round (and earlier rounds) still not included
    timed_out: bool
    latency_ms: Dict[int, int] = field(default_factory=dict)   # seq -> post-to-inclusion, for entries seen included


class BoardPostMode:
    """Post a run's entries to ``BoardLog`` in batches and tell the harness which are visible."""

    def __init__(
        self,
        client: Any,
        timeout_s: float = BATCH_TIMEOUT_S,
        poll_s: float = 1.0,
        clock: Callable[[], float] = time.time,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        self.client = client
        self.timeout_s = timeout_s
        self.poll_s = poll_s
        self.clock = clock
        self.sleep = sleep
        self._posted_upto: Dict[str, int] = {}
        self._batches: List[dict] = []
        self.visible: Set[int] = set()
        self.log: List[dict] = []

    def post_round(self, run_id: str, entries: Sequence[Entry]) -> RoundResult:
        """Queue ``entries`` as one batch, then wait up to ``timeout_s`` for every outstanding batch."""
        if entries:
            first = entries[0].seq
            expected = self._posted_upto.get(run_id, 0)
            if first != expected or [e.seq for e in entries] != list(range(first, first + len(entries))):
                raise ChainModeError(f"batch must continue at seq {expected}, got {[e.seq for e in entries]}")
            prev = entries[0].prev_hash or ZERO_HASH
            job = self.client.enqueue_board_batch(run_id, first, prev, [compact_entry(e) for e in entries])
            self._batches.append({"job": job, "seqs": [e.seq for e in entries], "posted_at": self.clock(), "seen": False})
            self._posted_upto[run_id] = first + len(entries)
        return self._wait()

    def _wait(self) -> RoundResult:
        deadline = self.clock() + self.timeout_s
        latency: Dict[int, int] = {}
        while True:
            self.client.pump()
            now = self.clock()
            for batch in self._batches:
                if not batch["seen"] and self._included(batch["job"]):
                    batch["seen"] = True
                    ms = int(round((now - batch["posted_at"]) * 1000))
                    self.visible.update(batch["seqs"])
                    for seq in batch["seqs"]:
                        latency[seq] = ms
                    self.log.append({"event": "included", "first_seq": batch["seqs"][0], "n": len(batch["seqs"]), "latency_ms": ms})
            outstanding = [b for b in self._batches if not b["seen"]]
            if not outstanding:
                return RoundResult(set(self.visible), set(), False, latency)
            if now >= deadline:
                missing = {s for b in outstanding for s in b["seqs"]}
                self.log.append({"event": "timeout", "invisible_seqs": sorted(missing), "waited_s": self.timeout_s})
                return RoundResult(set(self.visible), missing, True, latency)
            self.sleep(self.poll_s)

    def _included(self, job_id: int) -> bool:
        job = self.client.queue.get(job_id)
        return bool(job and job["status"] == CONFIRMED)
