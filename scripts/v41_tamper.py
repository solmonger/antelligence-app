#!/usr/bin/env python
"""v4.1 tamper suite (goal 4.4): T1 rewrite, T2 truncate, T3 equivocation, T4 backdate x B/OTS/R/C.

Offline by default (mock OTS calendars, mock Rekor, a Python model of AnchorRegistry read through the
real Web3 reader path). With --local-node it runs the chain leg against a LOCAL hardhat/anvil node
(chain id 31337/1337 only; the node's unlocked accounts are used, no key is read):

  cd blockchain && npx hardhat node --port 8545        # in another shell, after `npx hardhat compile`
  .venv/bin/python scripts/v41_tamper.py --runs 50 --local-node http://127.0.0.1:8545 \\
      --artifact blockchain/artifacts/contracts/AnchorRegistry.sol/AnchorRegistry.json --out tamper.json

Output: JSON with, per (attack, position, condition): detection rate with Wilson interval, write-time vs
audit-time counts, mean verification wall time, bytes and network calls, the unanchored tail size, and the
mean anchor latency per system. Real-network execution (public Rekor, OTS calendars, the L2) is the lead's job.
Exit code 1 if the observed detections differ from the expected table.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from antelligence.anchoring import tamper  # noqa: E402


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs", type=int, default=50, help="honest runs per cell (goal: 50 SW-B evaluation runs)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--local-node", help="URL of a local hardhat/anvil node for the chain leg")
    ap.add_argument("--artifact", default=str(REPO / "blockchain/artifacts/contracts/AnchorRegistry.sol/AnchorRegistry.json"))
    ap.add_argument("--out")
    args = ap.parse_args(argv)

    factory = None
    if args.local_node:
        from web3 import Web3

        w3 = Web3(Web3.HTTPProvider(args.local_node))
        if int(w3.eth.chain_id) not in tamper.LOCAL_CHAIN_IDS:
            ap.error("--local-node must serve chain id 31337 or 1337")
        factory = lambda: tamper.deploy_local_registry(w3, Path(args.artifact))  # noqa: E731
    matrix = tamper.run_matrix(args.runs, registry_factory=factory, base_seed=args.seed)
    matrix["table_mismatches"] = tamper.table_mismatches(matrix)
    text = json.dumps(matrix, indent=1, sort_keys=True)
    if args.out:
        Path(args.out).write_text(text + "\n")
    else:
        print(text)
    return 1 if matrix["table_mismatches"] else 0


if __name__ == "__main__":
    sys.exit(main())
