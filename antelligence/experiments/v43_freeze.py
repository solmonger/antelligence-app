"""v4.3 freeze guard (goal section 4): no sharing arm and no control runs on a development case until the lead has frozen the code.

* :func:`freeze_hash`  sha256 over the sorted (path, content sha256) of every git-tracked file under ``antelligence/``,
  ``config/v4_3/`` and ``scripts/v43_*.py``, read from the WORKING TREE (so an edit that is not committed still changes it).
* :func:`write_freeze` / :func:`check_freeze`  write and compare ``freeze.json``: ``{commit, freeze_hash, config_sha256, date}``.
* :func:`require_frozen`  every v4.3 driver that runs a sharing arm or a control on a development case calls it first. It refuses
  unless the freeze file exists and its ``freeze_hash`` equals the current one. S, S-solo and S' (gate-A style runs) do not need it.

Nothing here runs a model, reads a key or touches the network; it only runs ``git ls-files`` / ``git rev-parse`` / ``git status``.
"""

from __future__ import annotations

import fnmatch
import hashlib
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_FREEZE_FILE = REPO_ROOT / "docs" / "research" / "v4.3-20261015" / "freeze.json"
CODE_DIRS = ("antelligence/", "config/v4_3/")
SCRIPT_GLOB = "scripts/v43_*.py"
CONFIG_DIR = "config/v4_3/"
SCHEMA = "freeze_v4_3"


class FreezeRefused(PermissionError):
    """A sharing arm or control was asked to run on a development case and the code is not frozen (or changed since)."""


def _git(root: Path, *args: str) -> str:
    out = subprocess.run(["git", "-C", str(root), *args], check=True, capture_output=True, text=True)
    return out.stdout


def tracked_files(root: Path = REPO_ROOT) -> List[str]:
    """Git-tracked paths in the frozen set, sorted, repo-relative with ``/``."""
    names = _git(root, "ls-files", "-z", "--", "antelligence", "config/v4_3", "scripts").split("\0")
    keep = [n for n in names if n and (n.startswith(CODE_DIRS) or fnmatch.fnmatchcase(n, SCRIPT_GLOB) and n.count("/") == 1)]
    return sorted(set(keep))


def _file_sha(path: Path) -> str:
    try:
        return hashlib.sha256(path.read_bytes()).hexdigest()
    except FileNotFoundError:
        return "deleted"                                   # tracked but gone from the working tree: that is a change too


def file_hashes(root: Path = REPO_ROOT, paths: Optional[Iterable[str]] = None) -> Dict[str, str]:
    return {p: _file_sha(Path(root) / p) for p in (sorted(paths) if paths is not None else tracked_files(root))}


def _combine(hashes: Dict[str, str]) -> str:
    h = hashlib.sha256()
    for p in sorted(hashes):
        h.update(f"{p}\0{hashes[p]}\n".encode())
    return h.hexdigest()


def freeze_hash(root: Path = REPO_ROOT) -> str:
    return _combine(file_hashes(root))


def head_commit(root: Path = REPO_ROOT) -> Optional[str]:
    try:
        return _git(root, "rev-parse", "HEAD").strip()
    except subprocess.CalledProcessError:                  # no commit yet
        return None


def config_hashes(root: Path = REPO_ROOT) -> Dict[str, Any]:
    files = file_hashes(root, [p for p in tracked_files(root) if p.startswith(CONFIG_DIR)])
    return {"files": files, "combined": _combine(files)}


def uncommitted(root: Path = REPO_ROOT) -> List[str]:
    """Paths in the frozen set whose working tree differs from HEAD (or the index): staged, modified or deleted."""
    names = set(tracked_files(root))
    out = _git(root, "status", "--porcelain", "-z", "--", "antelligence", "config/v4_3", "scripts").split("\0")
    dirty = []
    for ent in out:
        if len(ent) > 3:
            code, path = ent[:2], ent[3:]
            if path in names and code.strip() and code != "??":
                dirty.append(path)
    return sorted(dirty)


def current(root: Path = REPO_ROOT) -> Dict[str, Any]:
    hashes = file_hashes(root)
    return {"schema": SCHEMA, "commit": head_commit(root), "freeze_hash": _combine(hashes), "n_files": len(hashes),
            "config_sha256": config_hashes(root), "date": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")}


def write_freeze(out: Path, root: Path = REPO_ROOT, *, allow_dirty: bool = False) -> Dict[str, Any]:
    """Write ``freeze.json``. A tree with uncommitted changes in the frozen set is refused (the commit would not name the
    code that was hashed) unless ``allow_dirty``."""
    dirty = uncommitted(root)
    if dirty and not allow_dirty:
        raise FreezeRefused(f"{len(dirty)} frozen-set file(s) have uncommitted changes (commit first): {', '.join(dirty[:5])}")
    doc = current(root)
    doc["uncommitted"] = dirty
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_name(out.name + ".tmp")
    tmp.write_text(json.dumps(doc, indent=1, sort_keys=True))
    tmp.replace(out)
    return doc


def check_freeze(path: Path = DEFAULT_FREEZE_FILE, root: Path = REPO_ROOT) -> Tuple[bool, str]:
    """(ok, reason). ``ok`` only if the file exists, parses, and its ``freeze_hash`` equals the current one."""
    path = Path(path)
    if not path.is_file():
        return False, f"freeze file {path} does not exist: the lead has not frozen the code"
    try:
        doc = json.loads(path.read_text())
        want = doc["freeze_hash"]
    except (ValueError, KeyError, TypeError) as exc:
        return False, f"freeze file {path} is unreadable ({exc})"
    now = freeze_hash(root)
    if want != now:
        changed = _changed_files(doc, root)
        return False, (f"the code changed since the freeze: freeze_hash {str(want)[:12]} != current {now[:12]}"
                       + (f" (config differs: {', '.join(changed)})" if changed else ""))
    return True, f"frozen at commit {doc.get('commit')} ({str(want)[:12]})"


def _changed_files(doc: Dict[str, Any], root: Path) -> List[str]:
    saved = (doc.get("config_sha256") or {}).get("files") or {}
    now = config_hashes(root)["files"]
    return sorted(p for p in set(saved) | set(now) if saved.get(p) != now.get(p))


def require_frozen(path: Path = DEFAULT_FREEZE_FILE, root: Path = REPO_ROOT) -> Dict[str, Any]:
    """Raise :class:`FreezeRefused` unless the freeze file matches the current code. Returns the freeze document."""
    ok, why = check_freeze(path, root)
    if not ok:
        raise FreezeRefused(why)
    return json.loads(Path(path).read_text())


def freeze_needed(arms: Sequence[str], *, on_dev_case: bool = True) -> bool:
    """True if any of ``arms`` is a sharing arm or a control (anything but S, S-solo, S') and a development case is involved."""
    from antelligence.experiments.v43_arms import needs_freeze
    return bool(on_dev_case and any(needs_freeze(a) for a in arms))
