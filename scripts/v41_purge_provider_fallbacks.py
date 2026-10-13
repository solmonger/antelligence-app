#!/usr/bin/env python
"""Drop sandbox results-cache entries whose episode had a provider-caused queen fallback (mistake-2039).

  .venv/bin/python scripts/v41_purge_provider_fallbacks.py WORK_DIR            # dry run: list what would go
  .venv/bin/python scripts/v41_purge_provider_fallbacks.py WORK_DIR --apply    # rewrite results-cache.jsonl, keep a .bak

WORK_DIR is the sandbox work dir (``results-cache.jsonl`` and ``runs/<run_id>/`` with ``cell.json`` and the board export).
A cached entry is dropped when its run export holds an ``alarm`` whose note starts ``queen fallback: ProviderError`` (what
the old queen wrote when a provider outage became an S-rule fallback), when its cell says ``queen_fallbacks_provider > 0``,
or when the cache row itself records ``queen_fallbacks.provider > 0``. Rows written by the current sandbox carry the
episode's ``run_id`` and are matched exactly; older rows have none and are matched on (harness sha, arm, case, seed) against
every export of that key: if any of them is tainted the row goes, because a cached coverage cannot be traced to one run.
Dropped episodes simply rerun (their replies are cached). Everything else, unparseable lines included, is kept verbatim.

Exit 0 on success (also for a dry run); 1 when the cache or the work dir is missing; 2 when a .bak already exists (the
first backup is the original and is never overwritten). Reads no keys and makes no network call.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from antelligence.experiments import v41_fallbacks  # noqa: E402

CACHE_NAME = "results-cache.jsonl"


def tainted_runs(work: Path) -> Tuple[Set[str], Set[Tuple[str, str, str, int]]]:
    """(run ids, (harness sha, arm, case, seed) keys) of every run under ``work`` with a provider-caused queen fallback."""
    ids: Set[str] = set()
    keys: Set[Tuple[str, str, str, int]] = set()
    for cell_path in sorted(work.rglob("cell.json")):
        run_dir = cell_path.parent
        try:
            cell = json.loads(cell_path.read_text(encoding="utf-8"))
        except ValueError:
            continue
        if v41_fallbacks.run_dir_provider_fallbacks(run_dir, cell) <= 0:
            continue
        ids.add(str(cell.get("run_id")))
        for export in v41_fallbacks.export_files(run_dir):
            first = export.read_text(encoding="utf-8").split("\n", 1)[0]
            try:
                header = json.loads(first)
            except ValueError:
                continue
            keys.add((str(header.get("harness_cfg_sha256")), str(cell.get("arm")), str(cell.get("case")), int(cell.get("seed", 0))))
    return ids, keys


def purge(work: Path, *, apply: bool = False) -> Dict[str, Any]:
    cache = work / CACHE_NAME
    if not cache.is_file():
        raise FileNotFoundError(f"{cache}: no results cache")
    ids, keys = tainted_runs(work)
    kept: List[str] = []
    dropped: List[Dict[str, Any]] = []
    for line in cache.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except ValueError:
            kept.append(line)
            continue
        recorded = (row.get("queen_fallbacks") or {}).get("provider", 0) if isinstance(row, dict) else 0
        if isinstance(row, dict) and row.get("run_id"):
            bad = str(row["run_id"]) in ids or bool(recorded)
        elif isinstance(row, dict):
            bad = bool(recorded) or (str(row.get("harness")), str(row.get("arm")), str(row.get("case")), int(row.get("seed", 0))) in keys
        else:
            bad = False
        if bad:
            dropped.append({k: row.get(k) for k in ("harness", "arm", "case", "seed")})
        else:
            kept.append(line)
    report: Dict[str, Any] = {"cache": str(cache), "rows_kept": len(kept), "rows_dropped": len(dropped), "dropped": dropped,
                              "tainted_runs": len(ids), "applied": False, "backup": None}
    if apply and dropped:
        backup = cache.with_name(cache.name + ".bak")
        if backup.exists():
            raise FileExistsError(f"{backup} already exists; the first backup is the original, move it away first")
        os.replace(cache, backup)          # keep the original; the rewrite below cannot lose it
        tmp = cache.with_name(cache.name + ".tmp")
        tmp.write_text("".join(l + "\n" for l in kept), encoding="utf-8")
        os.replace(tmp, cache)
        report.update(applied=True, backup=str(backup))
    return report


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("work_dir", type=Path)
    ap.add_argument("--apply", action="store_true", help="rewrite results-cache.jsonl (default is a dry run); keeps results-cache.jsonl.bak")
    a = ap.parse_args(argv)
    if not a.work_dir.is_dir():
        print(f"{a.work_dir}: not a directory", file=sys.stderr)
        return 1
    try:
        r = purge(a.work_dir, apply=a.apply)
    except FileNotFoundError as exc:
        print(str(exc), file=sys.stderr)
        return 1
    except FileExistsError as exc:
        print(str(exc), file=sys.stderr)
        return 2
    for d in r["dropped"]:
        print(f"drop {d['arm']} {d['case']} seed {d['seed']} harness {str(d['harness'])[:12]}")
    mode = "applied" if r["applied"] else ("dry run (pass --apply to rewrite)" if r["rows_dropped"] else "nothing to do")
    print(f"{r['tainted_runs']} tainted run(s); {r['rows_dropped']} cache row(s) dropped, {r['rows_kept']} kept: {mode}"
          + (f"; backup {r['backup']}" if r["backup"] else ""))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
