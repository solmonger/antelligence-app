#!/usr/bin/env python
"""v4.1 key custody (goal 2.5). Prints fingerprints and file names only, never key material.

  .venv/bin/python scripts/v41_keys.py init-study                 # P-256 study key at ledger/v4.1-study-key (0600), once
  .venv/bin/python scripts/v41_keys.py fingerprint                # study key fingerprint, for the preregistration
  .venv/bin/python scripts/v41_keys.py audit                      # counts and bad modes
  .venv/bin/python scripts/v41_keys.py delete-run-keys --confirm  # after the tamper suite; logged in ledger/v4.1-key-deletions.jsonl
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Optional, Sequence

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from antelligence.anchoring import keys  # noqa: E402


def main(argv: Optional[Sequence[str]] = None, *, ledger_dir: Optional[Path] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ledger-dir", default=str(ledger_dir or REPO / "ledger"))
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("init-study")
    sub.add_parser("fingerprint")
    sub.add_parser("audit")
    d = sub.add_parser("delete-run-keys")
    d.add_argument("--confirm", action="store_true")
    args = ap.parse_args(argv)
    try:
        if args.cmd == "init-study":
            print(f"study key created; fingerprint {keys.create_study_key(args.ledger_dir)}")
        elif args.cmd == "fingerprint":
            print(keys.study_fingerprint(args.ledger_dir))
        elif args.cmd == "audit":
            print(keys.audit(args.ledger_dir))
        else:
            rows = keys.delete_run_keys(args.ledger_dir, confirm=args.confirm)
            print(f"deleted {len(rows)} run key file(s); logged to ledger/{keys.DELETION_LOG}")
    except keys.KeyError_ as exc:
        print(f"refused: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
