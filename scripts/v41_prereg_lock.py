#!/usr/bin/env python
"""Lock the preregistration, then the run plan (used after G1). WRITES TO PUBLIC NETWORKS; the lead runs it.

  .venv/bin/python scripts/v41_prereg_lock.py prereg                         # PREREGISTRATION.md
  .venv/bin/python scripts/v41_prereg_lock.py runplan --runplan runplan.json # after the prereg receipt exists

For each document: the file must be committed in HEAD and unmodified, and the study key fingerprint it states must be
the key at ledger/v4.1-study-key. Then, in this order and resumably (a re-run skips what already succeeded):

  1. AnchorRegistry.anchorOnce(keccak256("v4.1|prereg"|"v4.1|runplan"), digest) on chain 300, read back via eth_call
  2. OpenTimestamps stamp against the public calendars (<name>.ots)
  3. Rekor hashedrekord signed with the study key (<name>.rekor.json)

The receipt is written only after all three succeed: tx_hash, status 1, readback_ok, chain_id 300, prereg_sha256 (what
scripts/v41_run.py and v41_gate check), plus block, L1 batch, OTS path and Rekor uuid/index. The digest is sha256 of the
file for the preregistration and sha256 of its RFC 8785 JSON canonical form for the run plan (the verifier's rule).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any, Callable, Dict, Optional, Sequence

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "backend"))

from web3 import Web3  # noqa: E402

from antelligence.anchoring import keys as keymod  # noqa: E402
from antelligence.anchoring.anchorer import single_shot_id  # noqa: E402
from antelligence.anchoring.live import AnchorArchive  # noqa: E402
from antelligence.anchoring.pins import study_key_in  # noqa: E402
from antelligence.experiments import v41_gate  # noqa: E402
from antelligence.providers.redact import scrub  # noqa: E402
from antelligence.kernel.canonical import jcs_bytes  # noqa: E402

DOC_DIR = REPO / "docs/research/v4.1-20261013"
CHAIN_DIR = DOC_DIR / "chain"
ZERO32 = b"\x00" * 32


class LockError(RuntimeError):
    pass


def document_digest(name: str, path: Path) -> bytes:
    if name == "prereg":
        return hashlib.sha256(path.read_bytes()).digest()
    return hashlib.sha256(jcs_bytes(json.loads(path.read_text()))).digest()


def _once_digest(client: Any, sid: bytes) -> bytes:
    from chain.v4_1 import _calldata

    return bytes(client.w3.eth.call({"to": client._addr("AnchorRegistry"), "data": _calldata("AnchorRegistry", "onceDigestOf", [sid])}))


def lock_document(name: str, doc: Path, *, client: Any, ots: Any, rekor: Any, study_key: Any, out_dir: Path,
                  repo_root: Path = REPO, prereg: Optional[Path] = None, drain_timeout_s: float = 300.0) -> Dict[str, Any]:
    """Anchor one document in all three places and write ``<name>-receipt.json``. Resumable."""
    sid, digest = single_shot_id(name), document_digest(name, doc)
    prereg = prereg or doc
    try:
        v41_gate._check_prereg(repo_root, doc)           # committed in HEAD, byte-identical, inside the repo
    except v41_gate.GateError as exc:
        raise LockError(f"{doc.name}: {exc}") from None
    if study_key_in(prereg.read_bytes()) != study_key.fingerprint:
        raise LockError("the study key fingerprint in the preregistration is not the key at ledger/v4.1-study-key")
    client.preflight()                                    # chain 300, deployer, code and code hash, before any write

    existing = _once_digest(client, sid)
    if existing not in (ZERO32, b""):
        if existing != digest:
            raise LockError(f"{name} is already anchored on chain with a DIFFERENT digest; refusing")
    else:
        client.enqueue_anchor_once(sid, digest)
        client.drain(timeout_s=drain_timeout_s)
    if _once_digest(client, sid) != digest:
        raise LockError(f"{name}: chain readback does not equal the digest (see queue); nothing else was written")
    job = next((j for j in client.queue.all() if j["dedupe_key"] == f"once:{sid.hex()}"), None)
    tx = job["tx_hashes"][-1] if job and job["tx_hashes"] else None
    info = client.receipt_info(tx) if tx else {}
    if not tx or info.get("status") != 1:
        raise LockError(f"{name}: no successful receipt in this queue (anchored in an earlier session? read the explorer)")

    out_dir.mkdir(parents=True, exist_ok=True)
    ots_path = out_dir / f"{name}.ots"
    if not ots_path.exists():
        ots.stamp(digest, ots_path)
    entry = rekor.log_digest(digest, study_key)
    rekor_path = out_dir / f"{name}.rekor.json"
    rekor_path.write_text(json.dumps({"uuid": entry.uuid, "log_index": entry.log_index, "digest": digest.hex(),
                                      "study_key_sha256": study_key.fingerprint}, indent=1) + "\n")
    receipt = {"name": name, "tx_hash": tx, "status": 1, "readback_ok": True, "chain_id": int(client.w3.eth.chain_id),
               "single_shot_id": "0x" + sid.hex(), "digest": digest.hex(), "l2_block": info.get("block"),
               "l1_batch": info.get("l1_batch"), "registry": client._addr("AnchorRegistry"),
               "ots": ots_path.name, "rekor_uuid": entry.uuid, "rekor_log_index": entry.log_index,
               "prereg_sha256": hashlib.sha256(prereg.read_bytes()).hexdigest()}
    (out_dir / f"{name}-receipt.json").write_text(json.dumps(receipt, indent=1, sort_keys=True) + "\n")
    AnchorArchive(out_dir / "anchors.json").upsert(
        {"kind": "once", "name": name, "tx_hash": tx, "l2_block": info.get("block"), "l1_batch": info.get("l1_batch"),
         "chain_id": receipt["chain_id"], "digest": digest.hex(), "rekor_uuid": entry.uuid, "ots_path": ots_path.name})
    return receipt


def main(argv: Optional[Sequence[str]] = None, *, make_services: Optional[Callable[[], Dict[str, Any]]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("which", choices=("prereg", "runplan"))
    ap.add_argument("--prereg", default=str(DOC_DIR / "PREREGISTRATION.md"))
    ap.add_argument("--runplan", default=str(DOC_DIR / "runplan.json"))
    ap.add_argument("--out-dir", default=str(CHAIN_DIR))
    ap.add_argument("--repo-root", default=str(REPO))
    ap.add_argument("--ledger-dir", default=str(REPO / "ledger"))
    ap.add_argument("--queue", default=str(REPO / "ledger/v4_1_chain_queue.sqlite"))
    args = ap.parse_args(argv)
    out = Path(args.out_dir)
    if args.which == "runplan" and not (out / "prereg-receipt.json").exists():
        print("refused: lock the preregistration first (no prereg-receipt.json)", file=sys.stderr)
        return 2
    try:
        if make_services:
            svc = make_services()
        else:
            from antelligence.anchoring.live import build_live_anchoring

            live = build_live_anchoring(queue_path=args.queue, start_pumper=False)
            svc = {"client": live.client, "ots": live.ots, "rekor": live.rekor}
        study = keymod.load_study_key(args.ledger_dir)
        doc = Path(args.prereg if args.which == "prereg" else args.runplan)
        receipt = lock_document(args.which, doc, client=svc["client"], ots=svc["ots"], rekor=svc["rekor"], study_key=study,
                                out_dir=out, repo_root=Path(args.repo_root), prereg=Path(args.prereg))
    except (LockError, keymod.KeyError_) as exc:
        print(f"refused: {scrub(exc)}", file=sys.stderr)
        return 2
    print(f"{args.which} locked: tx {receipt['tx_hash']} block {receipt['l2_block']} rekor {receipt['rekor_uuid']} "
          f"-> {out / (args.which + '-receipt.json')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
