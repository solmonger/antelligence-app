#!/usr/bin/env python
"""Fill in the L1 commit, prove and execute hashes for every anchor in docs/research/v4.1-20261013/chain/anchors.json.

  .venv/bin/python scripts/v41_archive_l1.py [--anchors PATH] [--rpc URL]

Read-only: only eth_getTransactionReceipt and zks_getL1BatchDetails are ever sent (a whitelist enforces it). Run it
again later; an anchor whose batch is not yet committed/proven/executed keeps null hashes and is filled next time.
Rows get: l1_batch, l1_commit_tx, l1_prove_tx, l1_execute_tx, l1_status. The file is rewritten atomically.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence

import httpx

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

DEFAULT_ANCHORS = REPO / "docs/research/v4.1-20261013/chain/anchors.json"
DEFAULT_RPC = "https://sepolia.era.zksync.dev"
READ_ONLY = frozenset({"eth_getTransactionReceipt", "zks_getL1BatchDetails", "eth_chainId"})
Rpc = Callable[[str, list], Any]


def http_rpc(url: str, timeout_s: float = 30.0) -> Rpc:
    client = httpx.Client(timeout=timeout_s, trust_env=False)

    def call(method: str, params: list) -> Any:
        if method not in READ_ONLY:
            raise ValueError(f"{method} is not a read-only method this script may send")
        r = client.post(url, json={"jsonrpc": "2.0", "id": 1, "method": method, "params": params})
        r.raise_for_status()
        body = r.json()
        if "error" in body:
            raise RuntimeError(f"{method}: {body['error']}")
        return body.get("result")

    return call


def _int(value: Any) -> Optional[int]:
    if value is None:
        return None
    return int(value, 16) if isinstance(value, str) and value.startswith("0x") else int(value)


def fill_row(row: Dict[str, Any], rpc: Rpc) -> bool:
    """Update one anchor row in place; returns True if anything changed."""
    tx = row.get("tx_hash")
    if not tx:
        return False
    before = dict(row)
    batch = row.get("l1_batch")
    if batch is None:
        rc = rpc("eth_getTransactionReceipt", [tx])
        if rc:
            if row.get("l2_block") is None:
                row["l2_block"] = _int(rc.get("blockNumber"))
            batch = _int(rc.get("l1BatchNumber"))
            row["l1_batch"] = batch
    if batch is not None:
        d = rpc("zks_getL1BatchDetails", [int(batch)]) or {}
        row["l1_commit_tx"] = d.get("commitTxHash")
        row["l1_prove_tx"] = d.get("proveTxHash")
        row["l1_execute_tx"] = d.get("executeTxHash")
        row["l1_status"] = d.get("status")
    return row != before


def main(argv: Optional[Sequence[str]] = None, *, rpc: Optional[Rpc] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--anchors", default=str(DEFAULT_ANCHORS))
    ap.add_argument("--rpc", default=os.environ.get("ZKSYNC_SEPOLIA_RPC_URL") or DEFAULT_RPC)
    args = ap.parse_args(argv)
    path = Path(args.anchors)
    if not path.exists():
        print(f"no anchors file at {path}", file=sys.stderr)
        return 2
    call = rpc or http_rpc(args.rpc)
    chain = _int(call("eth_chainId", []))
    if chain != 300:
        print(f"refused: RPC serves chain {chain}, not 300", file=sys.stderr)
        return 2
    rows: List[Dict[str, Any]] = json.loads(path.read_text())
    changed = executed = 0
    for row in rows:
        try:
            changed += fill_row(row, call)
        except Exception as exc:
            print(f"{row.get('run_id')}:{row.get('size', row.get('name'))}: {type(exc).__name__}: {exc}", file=sys.stderr)
        executed += bool(row.get("l1_execute_tx"))
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(rows, indent=1, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(tmp, path)
    print(f"{len(rows)} anchors, {changed} updated, {executed} executed on L1")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
