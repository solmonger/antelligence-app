"""The anchorer: the board calls it at every 50 entries and at run end (INTERFACES section 3).

Wiring (the board itself stays untouched)::

    anchorer = Anchorer(board.run_id, {"OTS", "R", "C"}, out_dir, ots=..., rekor=..., anchor_key=..., chain=...)
    entry = board.append(...)
    anchorer.maybe_anchor(board)      # after every append: anchors when size is a multiple of 50
    ...
    anchorer.finalize(board)          # run end: anchors the final size (once, even on a multiple of 50)

OTS and Rekor are best-effort network calls: a failure is recorded in ``errors`` and never raises into
the tick loop. Mode C only enqueues (see :mod:`chain_modes`). ``<run_id>.anchors.json`` is rewritten
after every anchor in the INTERFACES section 7 shape.
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Union

from web3 import Web3

from antelligence.anchoring.chain_modes import ChainAnchorMode
from antelligence.anchoring.ots import OtsClient
from antelligence.anchoring.rekor import AnchorKey, RekorClient
from antelligence.kernel.board import ANCHOR_EVERY, Board

MODES = ("OTS", "R", "C")
SINGLE_SHOT_IDS = {"prereg": "v4.1|prereg", "runplan": "v4.1|runplan"}


def single_shot_id(name: str) -> bytes:
    """``keccak256("v4.1|prereg")`` or ``keccak256("v4.1|runplan")`` (INTERFACES section 3)."""
    return bytes(Web3.keccak(text=SINGLE_SHOT_IDS[name]))


class Anchorer:
    def __init__(
        self,
        run_id: str,
        modes: Iterable[str],
        out_dir: Union[str, Path],
        *,
        ots: Optional[OtsClient] = None,
        rekor: Optional[RekorClient] = None,
        anchor_key: Optional[AnchorKey] = None,
        chain: Optional[ChainAnchorMode] = None,
        clock: Callable[[], float] = time.perf_counter,
    ) -> None:
        self.modes = frozenset(modes)
        unknown = self.modes - set(MODES)
        if unknown:
            raise ValueError(f"unknown anchor modes {sorted(unknown)}")
        for mode, dep, name in (("OTS", ots, "ots"), ("R", rekor, "rekor"), ("C", chain, "chain")):
            if mode in self.modes and dep is None:
                raise ValueError(f"mode {mode} needs {name}=")
        if "R" in self.modes and anchor_key is None:
            raise ValueError("mode R needs anchor_key=")
        self.run_id = run_id
        self.out_dir = Path(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.ots, self.rekor, self.anchor_key, self.chain = ots, rekor, anchor_key, chain
        self._clock = clock
        self.records: List[Dict[str, Any]] = []
        self.errors: List[Dict[str, str]] = []
        self.latency_s: Dict[str, List[float]] = {m: [] for m in MODES}
        self._jobs: Dict[int, int] = {}      # size -> chain queue job id

    @property
    def path(self) -> Path:
        return self.out_dir / f"{self.run_id}.anchors.json"

    @property
    def anchored_sizes(self) -> List[int]:
        return [r["size"] for r in self.records]

    # -- board hooks ---------------------------------------------------------------------------
    def maybe_anchor(self, board: Board) -> Optional[Dict[str, Any]]:
        """Anchor if the board just reached a multiple of 50 entries."""
        size = board.size()
        if size > 0 and size % ANCHOR_EVERY == 0 and size not in self.anchored_sizes:
            return self.anchor(board, size)
        return None

    def finalize(self, board: Board) -> Optional[Dict[str, Any]]:
        """Run end: anchor the final size unless it is already anchored (a multiple of 50 anchored on append)."""
        size = board.size()
        if size > 0 and size not in self.anchored_sizes:
            return self.anchor(board, size)
        return None

    # -- one anchor ----------------------------------------------------------------------------
    def anchor(self, board: Board, size: int) -> Dict[str, Any]:
        if board.run_id != self.run_id:
            raise ValueError("board belongs to a different run")
        root = board.root(size)
        digest = board.anchor_digest(size)
        rec: Dict[str, Any] = {"size": size, "root": root.hex(), "digest": digest.hex()}
        if "OTS" in self.modes:
            self._timed("OTS", lambda: self._ots(rec, size, digest))
        if "R" in self.modes:
            self._timed("R", lambda: self._rekor(rec, digest))
        if "C" in self.modes:
            self._timed("C", lambda: self._chain(rec, size, root))
        self.records.append(rec)
        self._write()
        return rec

    def _timed(self, mode: str, fn: Callable[[], None]) -> None:
        t0 = self._clock()
        try:
            fn()
        except Exception as exc:  # never into the tick loop
            self.errors.append({"mode": mode, "error": f"{type(exc).__name__}: {exc}"})
        finally:
            self.latency_s[mode].append(self._clock() - t0)

    def _ots(self, rec: Dict[str, Any], size: int, digest: bytes) -> None:
        path = self.out_dir / f"{self.run_id}.{size}.ots"
        self.ots.stamp(digest, path)
        rec["ots_path"] = path.name

    def _rekor(self, rec: Dict[str, Any], digest: bytes) -> None:
        entry = self.rekor.log_digest(digest, self.anchor_key)
        rec["rekor_uuid"] = entry.uuid
        rec["rekor_log_index"] = entry.log_index

    def _chain(self, rec: Dict[str, Any], size: int, root: bytes) -> None:
        self._jobs[size] = self.chain.submit(self.run_id, size, root)

    # -- later, off the hot path ---------------------------------------------------------------
    def refresh_chain(self) -> int:
        """Copy confirmed chain tx hashes into the records; returns how many are confirmed."""
        done = 0
        for rec in self.records:
            job = self._jobs.get(rec["size"])
            if job is not None and self.chain.confirmed(job):
                rec["chain_tx"] = self.chain.tx_hash(job)
                done += 1
        self._write()
        return done

    def upgrade_ots(self) -> Dict[str, int]:
        """End of run: upgrade every ``.ots`` proof. Returns counts of upgraded and still pending."""
        up = pend = 0
        for rec in self.records:
            if "ots_path" in rec:
                result = self.ots.upgrade(self.out_dir / rec["ots_path"])
                up += result.upgraded
                pend += result.pending
        return {"upgraded": up, "pending": pend}

    def _write(self) -> None:
        tmp = self.path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(self.records, indent=1, sort_keys=True) + "\n", encoding="utf-8")
        os.replace(tmp, self.path)


class SingleShotAnchorer:
    """Preregistration and run plan: one digest each, once, to OTS / Rekor / chain ``anchorOnce``."""

    def __init__(
        self, modes: Iterable[str], out_dir: Union[str, Path], *, ots: Optional[OtsClient] = None,
        rekor: Optional[RekorClient] = None, study_key: Optional[AnchorKey] = None,
        chain: Optional[ChainAnchorMode] = None,
    ) -> None:
        self.modes = frozenset(modes)
        self.out_dir = Path(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.ots, self.rekor, self.study_key, self.chain = ots, rekor, study_key, chain
        self.records: Dict[str, Dict[str, Any]] = {}
        self.errors: List[Dict[str, str]] = []

    def anchor(self, name: str, digest: bytes) -> Dict[str, Any]:
        """Anchor one document in every enabled mode. A failing mode is recorded in ``errors``, not raised."""
        if name not in SINGLE_SHOT_IDS:
            raise ValueError(f"unknown single-shot document {name!r}")
        if name in self.records:
            raise ValueError(f"{name} is single-shot and was already anchored")
        rec: Dict[str, Any] = {"digest": digest.hex()}
        if "OTS" in self.modes:
            try:
                path = self.out_dir / f"{name}.ots"
                self.ots.stamp(digest, path)
                rec["ots_path"] = path.name
            except Exception as exc:
                self.errors.append({"mode": "OTS", "error": f"{type(exc).__name__}: {exc}"})
        if "R" in self.modes:
            try:
                entry = self.rekor.log_digest(digest, self.study_key)
                rec.update(rekor_uuid=entry.uuid, rekor_log_index=entry.log_index)
            except Exception as exc:
                self.errors.append({"mode": "R", "error": f"{type(exc).__name__}: {exc}"})
        if "C" in self.modes:
            try:
                rec["chain_job"] = self.chain.submit_once(single_shot_id(name), digest)
            except Exception as exc:
                self.errors.append({"mode": "C", "error": f"{type(exc).__name__}: {exc}"})
        self.records[name] = rec
        return rec
