#!/usr/bin/env python
"""v4.3 freeze guard CLI. No model, no network, no key; only git.

    python scripts/v43_freeze.py write --out docs/research/v4.3-20261015/freeze.json [--allow-dirty]
    python scripts/v43_freeze.py check [--freeze-file docs/research/v4.3-20261015/freeze.json]
    python scripts/v43_freeze.py show

The freeze file is written by the LEAD, once, after the last code change; builders and runners never write it.

``write`` hashes every file under ``antelligence/``, ``config/v4_3/`` and ``scripts/v43_*.py`` (tracked OR untracked-not-ignored;
no .pyc) plus the SimLook cell table, the combined-estimate coefficients and ``config/v4_1/split.json``, from the working tree. It
writes ``{commit, date, freeze_hash, config_sha256, rules_sha, estimate_sha, cell_table_sha256, allowed_world_cfg_sha
{low, mid, high}, colony_cfg_sha, dirty, uncommitted}``. It refuses a tree with uncommitted changes in that set unless
``--allow-dirty``, which records ``"dirty": true`` in the file. ``check`` exits 1 when the file is missing, differs from the current
code, or was written dirty. There is NO allow-dirty at check time: the harness refuses a dirty-recorded freeze too, so a freeze
written with ``--allow-dirty`` is good for tests of the tooling only, never for a verdict run. The harness also checks the content
hashes of the config, coefficients and cell table it was built from against the recorded ones.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import List, Optional

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from antelligence.experiments.v43_freeze import (DEFAULT_FREEZE_FILE, FreezeError, check_freeze, current,  # noqa: E402
                                                 write_freeze)


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    w = sub.add_parser("write")
    w.add_argument("--out", type=Path, default=DEFAULT_FREEZE_FILE)
    w.add_argument("--allow-dirty", action="store_true",
                   help='write from a dirty tree and record "dirty": true; check and the harness then refuse that file')
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
        except FreezeError as exc:
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
