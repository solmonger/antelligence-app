#!/usr/bin/env python
"""v4.3 freeze guard CLI. No model, no network, no key; only git.

    python scripts/v43_freeze.py write --out docs/research/v4.3-20261015/freeze.json [--allow-dirty]
    python scripts/v43_freeze.py check [--freeze-file docs/research/v4.3-20261015/freeze.json]
    python scripts/v43_freeze.py show

``write`` hashes every git-tracked file under ``antelligence/``, ``config/v4_3/`` and ``scripts/v43_*.py`` from the working tree
and writes ``{commit, freeze_hash, config_sha256 (per file and combined), date}``. It refuses a tree with uncommitted changes in
that set unless ``--allow-dirty``. ``check`` exits 1 when the file is missing or its hash differs from the current code.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import List, Optional

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from antelligence.experiments.v43_freeze import (DEFAULT_FREEZE_FILE, FreezeRefused, check_freeze, current,  # noqa: E402
                                                 write_freeze)


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    w = sub.add_parser("write")
    w.add_argument("--out", type=Path, default=DEFAULT_FREEZE_FILE)
    w.add_argument("--allow-dirty", action="store_true")
    w.add_argument("--root", type=Path, default=REPO)
    c = sub.add_parser("check")
    c.add_argument("--freeze-file", type=Path, default=DEFAULT_FREEZE_FILE)
    c.add_argument("--root", type=Path, default=REPO)
    s = sub.add_parser("show")
    s.add_argument("--root", type=Path, default=REPO)
    a = ap.parse_args(argv)
    if a.cmd == "write":
        try:
            doc = write_freeze(a.out, a.root, allow_dirty=a.allow_dirty)
        except FreezeRefused as exc:
            print(f"refused: {exc}", file=sys.stderr)
            return 2
        print(f"wrote {a.out}\ncommit {doc['commit']}\nfreeze_hash {doc['freeze_hash']}\nconfig_sha256 {doc['config_sha256']['combined']}")
        return 0
    if a.cmd == "check":
        ok, why = check_freeze(a.freeze_file, a.root)
        print(("OK: " if ok else "MISMATCH: ") + why)
        return 0 if ok else 1
    doc = current(a.root)
    print(json.dumps({k: doc[k] for k in ("commit", "freeze_hash", "n_files")} | {"config_combined": doc["config_sha256"]["combined"]}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
