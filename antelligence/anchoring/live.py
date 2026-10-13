"""Production anchoring for v4.1: OTS (public calendars) + Rekor (rekor.sigstore.dev) + chain C (AnchorRegistry) for
boards, chain D (BoardLog) for the SW-D arm and ReputationRegistry for SW-Byz-chain, all over the one persistent
``ChainClientV41`` queue and nonce manager (``backend/chain/v4_1.py``).

The signer is loaded inside :func:`chain.v4_1.open_live_client` and never leaves the client; nothing here receives,
stores, logs or prints it. :func:`build_live_anchoring` runs the chain preflight (chain id, deployer, contract code
and code hash) before it returns, so no write can happen before those checks pass.

Every anchor is archived (tx hash, L2 block, L1 batch number) into one ``anchors.json`` that
``scripts/v41_archive_l1.py`` later extends with the L1 commit, prove and execute hashes.
"""

from __future__ import annotations

import json
import os
import sys
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Union

from antelligence.anchoring.anchorer import Anchorer
from antelligence.anchoring.chain_modes import BoardPostMode, ChainAnchorMode
from antelligence.anchoring.ots import HttpOtsTransport, OtsClient
from antelligence.anchoring.rekor import AnchorKey, HttpRekorTransport, RekorClient
from antelligence.kernel.board import Board

REPO = Path(__file__).resolve().parents[2]
DEFAULT_ARCHIVE = REPO / "docs/research/v4.1-20261013/chain/anchors.json"
DEFAULT_QUEUE = REPO / "ledger/v4_1_chain_queue.sqlite"
LIVE_MODES = ("OTS", "R", "C")


def _chain_module():
    try:
        from chain import v4_1
    except ImportError:                      # backend/ is not on sys.path yet
        sys.path.insert(0, str(REPO / "backend"))
        from chain import v4_1
    return v4_1


class AnchorArchive:
    """The aggregated ``anchors.json``: one row per (run, size) or single-shot name, rewritten atomically."""

    def __init__(self, path: Union[str, Path] = DEFAULT_ARCHIVE) -> None:
        self.path = Path(path)
        self._lock = threading.Lock()

    def rows(self) -> List[Dict[str, Any]]:
        if not self.path.exists():
            return []
        return json.loads(self.path.read_text(encoding="utf-8"))

    def upsert(self, row: Dict[str, Any]) -> None:
        key = (row.get("run_id"), row.get("size"), row.get("name"))
        with self._lock:
            rows = self.rows()
            for i, old in enumerate(rows):
                if (old.get("run_id"), old.get("size"), old.get("name")) == key:
                    rows[i] = {**old, **row}
                    break
            else:
                rows.append(row)
            self.path.parent.mkdir(parents=True, exist_ok=True)
            tmp = self.path.with_suffix(".json.tmp")
            tmp.write_text(json.dumps(rows, indent=1, sort_keys=True) + "\n", encoding="utf-8")
            os.replace(tmp, self.path)


class LiveAnchorer(Anchorer):
    """``Anchorer`` whose chain leg is finished by the shared pump: ``refresh_chain`` waits (bounded) for the queue,
    then copies tx hash, L2 block and L1 batch number into the record and the aggregated archive."""

    def __init__(self, *args: Any, client: Any, archive: AnchorArchive, drain_timeout_s: float = 180.0,
                 **kw: Any) -> None:
        super().__init__(*args, **kw)
        self.client, self.archive, self.drain_timeout_s = client, archive, drain_timeout_s

    def refresh_chain(self) -> int:
        if self._jobs:
            try:
                self.client.drain(timeout_s=self.drain_timeout_s)
            except Exception as exc:         # a chain problem is recorded, never raised into the run
                self.errors.append({"mode": "C", "error": f"{type(exc).__name__}: {exc}"})
        done = super().refresh_chain()
        for rec in self.records:
            row = {"kind": "board", "run_id": self.run_id, "size": rec["size"], "root": rec["root"],
                   "digest": rec["digest"], **{k: rec[k] for k in ("rekor_uuid", "rekor_log_index", "ots_path") if k in rec}}
            tx = rec.get("chain_tx")
            if tx:
                try:
                    info = self.client.receipt_info(tx)
                except Exception as exc:
                    self.errors.append({"mode": "C", "error": f"receipt: {type(exc).__name__}: {exc}"})
                    info = {}
                rec["chain_block"], rec["chain_l1_batch"] = info.get("block"), info.get("l1_batch")
                row.update(tx_hash=tx, l2_block=info.get("block"), l1_batch=info.get("l1_batch"),
                           chain_id=int(self.client.w3.eth.chain_id))
            self.archive.upsert(row)
        self._write()
        return done


class ChainPumper(threading.Thread):
    """Background pump so queued anchors keep moving while the run works. Errors are kept, never raised."""

    def __init__(self, client: Any, interval_s: float = 2.0) -> None:
        super().__init__(name="v41-chain-pump", daemon=True)
        self.client, self.interval_s = client, interval_s
        self._stop_evt = threading.Event()
        self.errors: List[str] = []

    def run(self) -> None:
        while not self._stop_evt.wait(self.interval_s):
            counts = self.client.queue.counts()
            if counts["pending"] or counts["sent"]:
                try:
                    self.client.pump()
                except Exception as exc:
                    self.errors.append(f"{type(exc).__name__}: {exc}"[:240])

    def stop(self) -> None:
        self._stop_evt.set()
        self.join(timeout=10)


@dataclass
class LiveAnchoring:
    client: Any
    chain: ChainAnchorMode
    ots: OtsClient
    rekor: RekorClient
    archive: AnchorArchive
    pumper: Optional[ChainPumper] = None
    drain_timeout_s: float = 180.0
    anchorers: List[LiveAnchorer] = field(default_factory=list)

    def anchorer_factory(self, board: Board, akey: AnchorKey, out_dir: Path) -> LiveAnchorer:
        a = LiveAnchorer(board.run_id, LIVE_MODES, out_dir, ots=self.ots, rekor=self.rekor, anchor_key=akey,
                         chain=self.chain, client=self.client, archive=self.archive,
                         drain_timeout_s=self.drain_timeout_s)
        self.anchorers.append(a)
        return a

    def post_mode_factory(self) -> BoardPostMode:
        return BoardPostMode(self.client)

    @property
    def reputation_client(self) -> Any:
        return self.client

    def close(self) -> Dict[str, int]:
        if self.pumper is not None:
            self.pumper.stop()
        counts = self.client.drain(timeout_s=self.drain_timeout_s)
        self.client.queue.close()
        return counts


def build_live_anchoring(*, queue_path: Union[str, Path, None] = None, archive_path: Union[str, Path, None] = None,
                         client: Any = None, ots: Optional[OtsClient] = None, rekor: Optional[RekorClient] = None,
                         pump_interval_s: float = 2.0, start_pumper: bool = True, drain_timeout_s: float = 180.0,
                         **open_kw: Any) -> LiveAnchoring:
    """Real OTS + Rekor + chain by default. Tests inject ``client`` (a local-node client), mock ``ots``/``rekor``."""
    if client is None:
        client = _chain_module().open_live_client(Path(queue_path) if queue_path else DEFAULT_QUEUE, **open_kw)
    else:
        client.preflight()
    pumper = ChainPumper(client, pump_interval_s) if start_pumper else None
    if pumper:
        pumper.start()
    return LiveAnchoring(client=client, chain=ChainAnchorMode(client),
                         ots=ots or OtsClient(HttpOtsTransport()), rekor=rekor or RekorClient(HttpRekorTransport()),
                         archive=AnchorArchive(archive_path or DEFAULT_ARCHIVE), pumper=pumper,
                         drain_timeout_s=drain_timeout_s)
