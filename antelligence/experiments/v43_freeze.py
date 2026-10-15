"""v4.3 freeze guard (goal section 4): no sharing arm and no control runs on a development case until the lead has frozen the code.

* :func:`freeze_hash`  sha256 over the sorted (path, content sha256) of every file under ``antelligence/``, ``config/v4_3/`` and
  ``scripts/v43_*.py`` (tracked OR untracked-and-not-ignored, so a new file changes it; ``.pyc`` / ``__pycache__`` excluded) plus
  three explicit inputs that change results: the SimLook cell table, the combined-estimate coefficients and the v4.1 split file
  (a missing explicit file is an error). Read from the WORKING TREE, so an uncommitted edit changes it.
* :func:`write_freeze` / :func:`check_freeze`  write and compare ``freeze.json``:
  ``{commit, date, freeze_hash, config_sha256, rules_sha, estimate_sha, cell_table_sha256, allowed_world_cfg_sha{low,mid,high},
  colony_cfg_sha, dirty, uncommitted}``. The LEAD writes the file after the last code change. ``--allow-dirty`` records
  ``"dirty": true``; ``check_freeze`` and the harness refuse a dirty-recorded freeze (there is no allow-dirty at check time).
* :func:`require_frozen`      the tree check (file exists, not dirty-recorded, ``freeze_hash`` equals the current one).
* :func:`require_frozen_run`  the harness check: the tree check PLUS the live harness's CONTENT hashes (ant rules, combined
  estimate, SimLook cell table, world config, colony config) against the frozen ones, so no path (``--config``,
  ``--coefficients``, a cell-table path) can smuggle in an input the hash never covered. Mismatch raises :class:`FreezeError`
  naming the field. ``V43Harness.run_episode`` calls it before building the world, for every arm but S, S-solo and S' on a
  development case. The freeze file is the module constant :data:`FREEZE_FILE`; a test may monkeypatch the module attribute
  (``monkeypatch.setattr(v43_freeze, "FREEZE_FILE", tmp)``) or set ``harness.freeze_file``; the repo root is ``REPO_ROOT`` (same).

Nothing here runs a model, reads a key or touches the network; it only runs ``git ls-files`` / ``git rev-parse`` / ``git status``.
"""

from __future__ import annotations

import fnmatch
import hashlib
import json
import subprocess
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

REPO_ROOT = Path(__file__).resolve().parents[2]
FREEZE_FILE = REPO_ROOT / "docs" / "research" / "v4.3-20261015" / "freeze.json"
DEFAULT_FREEZE_FILE = FREEZE_FILE                        # the CLI default (same file)
CODE_DIRS = ("antelligence/", "config/v4_3/")
SCRIPT_GLOB = "scripts/v43_*.py"
CONFIG_DIR = "config/v4_3/"
WORLD_CONFIG = "config/v4_3/world.yaml"                  # the frozen config every verdict run uses
CELL_TABLE = "docs/research/v4.2-20261014/simlook/cells_v4_2.json"         # read-only
COEFFICIENTS = "docs/research/v4.3-20261015/combined/coefficients.json"
SPLIT_FILE = "config/v4_1/split.json"                    # read-only
EXPLICIT_FILES = (CELL_TABLE, COEFFICIENTS, SPLIT_FILE)  # inputs outside the code dirs that change results
SCHEMA = "freeze_v4_3"
LEVELS = ("low", "mid", "high")
REQUIRED_FIELDS = ("rules_sha", "estimate_sha", "cell_table_sha256", "allowed_world_cfg_sha", "colony_cfg_sha")


class FreezeError(PermissionError):
    """A sharing arm or control was asked to run on a development case and the code is not frozen (or something changed since)."""


FreezeRefused = FreezeError                               # the first name of this class; kept for the drivers


def _git(root: Path, *args: str) -> str:
    out = subprocess.run(["git", "-C", str(root), *args], check=True, capture_output=True, text=True)
    return out.stdout


def _in_set(n: str) -> bool:
    if not n or n.endswith(".pyc") or "__pycache__/" in n or n.startswith("__pycache__/"):
        return False
    return n.startswith(CODE_DIRS) or (fnmatch.fnmatchcase(n, SCRIPT_GLOB) and n.count("/") == 1)


def frozen_files(root: Path = REPO_ROOT) -> List[str]:
    """Repo-relative paths (``/``) the freeze hash covers, sorted: every tracked-or-untracked-not-ignored file under
    ``antelligence/``, ``config/v4_3/`` and ``scripts/v43_*.py`` (no ``.pyc`` / ``__pycache__``), plus the explicit files.
    A missing explicit file raises :class:`FreezeError`."""
    root = Path(root)
    names = _git(root, "ls-files", "-z", "--cached", "--others", "--exclude-standard", "--",
                 "antelligence", "config/v4_3", "scripts").split("\0")
    keep = {n for n in names if _in_set(n)}
    for rel in EXPLICIT_FILES:
        if not (root / rel).is_file():
            raise FreezeError(f"the freeze covers {rel} and it is missing")
        keep.add(rel)
    return sorted(keep)


tracked_files = frozen_files                              # earlier name


def _file_sha(path: Path) -> str:
    try:
        return hashlib.sha256(path.read_bytes()).hexdigest()
    except FileNotFoundError:
        return "deleted"                                   # tracked but gone from the working tree: that is a change too


def file_hashes(root: Path = REPO_ROOT, paths: Optional[Iterable[str]] = None) -> Dict[str, str]:
    return {p: _file_sha(Path(root) / p) for p in (sorted(paths) if paths is not None else frozen_files(root))}


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
    files = file_hashes(root, [p for p in frozen_files(root) if p.startswith(CONFIG_DIR)])
    return {"files": files, "combined": _combine(files)}


def uncommitted(root: Path = REPO_ROOT) -> List[str]:
    """Paths in the frozen set that differ from HEAD: staged, modified, deleted OR untracked-new (all of them are hashed)."""
    names = set(frozen_files(root))
    out = _git(root, "status", "--porcelain", "-z", "--untracked-files=all", "--", "antelligence", "config/v4_3", "scripts",
               *EXPLICIT_FILES).split("\0")
    return sorted(ent[3:] for ent in out if len(ent) > 3 and ent[3:] in names and ent[:2].strip())


# --- content hashes of what a harness is built from ------------------------------------------------------------------------

def _sha(obj: Any) -> str:
    return hashlib.sha256(json.dumps(obj, sort_keys=True, default=str).encode()).hexdigest()   # = v42_harness._sha


def _canonical(obj: Any) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"))


def derived_hashes(root: Path = REPO_ROOT) -> Dict[str, Any]:
    """The content hashes a harness built from the frozen files must reproduce (``config/v4_3/world.yaml`` plus the
    coefficient file and the cell table). The colony config goes through the same steps as ``V43Harness`` applies."""
    import yaml
    from antelligence.kernel.colony.episode import ColonyConfig
    from antelligence.worlds.tumor_mri import WorldConfig
    from antelligence.worlds.tumor_mri.ant_rules import AntRules
    from antelligence.worlds.tumor_mri.combined import CombinedEstimate

    root = Path(root)
    doc = yaml.safe_load((root / WORLD_CONFIG).read_text())
    colony = doc.get("colony", {})
    wc = WorldConfig.from_dict(doc["world"])
    rules = AntRules.from_dict(colony.get("ant_rules") or {})
    cc = replace(ColonyConfig.from_dict({**colony, "quiet_rule": False}), answer_rule=rules.answer_rule)
    est = CombinedEstimate.from_json(root / COEFFICIENTS) if rules.use_combined else None
    return {"rules_sha": rules.sha(), "estimate_sha": None if est is None else est.sha(),
            "cell_table_sha256": _file_sha(root / CELL_TABLE),
            "allowed_world_cfg_sha": {lv: _sha(wc.with_(facts=replace(wc.facts, uptake_level=lv)).to_dict()) for lv in LEVELS},
            "colony_cfg_sha": _sha(cc.to_dict())}


def current(root: Path = REPO_ROOT) -> Dict[str, Any]:
    hashes = file_hashes(root)
    return {"schema": SCHEMA, "commit": head_commit(root), "freeze_hash": _combine(hashes), "n_files": len(hashes),
            "config_sha256": config_hashes(root), **derived_hashes(root),
            "date": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")}


def write_freeze(out: Path, root: Path = REPO_ROOT, *, allow_dirty: bool = False) -> Dict[str, Any]:
    """Write ``freeze.json`` (the lead does this, once, after the last code change). A tree with uncommitted changes in the
    frozen set is refused (the commit would not name the code that was hashed) unless ``allow_dirty``, which records
    ``"dirty": true``; ``check_freeze`` and the harness refuse a dirty-recorded freeze."""
    dirty = uncommitted(root)
    if dirty and not allow_dirty:
        raise FreezeError(f"{len(dirty)} frozen-set file(s) have uncommitted changes (commit first): {', '.join(dirty[:5])}")
    doc = current(root)
    doc["uncommitted"] = dirty
    doc["dirty"] = bool(dirty)
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_name(out.name + ".tmp")
    tmp.write_text(json.dumps(doc, indent=1, sort_keys=True))
    tmp.replace(out)
    return doc


def check_freeze(path: Optional[Path] = None, root: Optional[Path] = None) -> Tuple[bool, str]:
    """(ok, reason). ``ok`` only if the file exists, parses, is not dirty-recorded, and its ``freeze_hash`` equals the current one.
    ``path`` / ``root`` default to the module constants :data:`FREEZE_FILE` / :data:`REPO_ROOT`, read at call time."""
    path = Path(path) if path is not None else FREEZE_FILE
    root = Path(root) if root is not None else REPO_ROOT
    if not path.is_file():
        return False, f"freeze file {path} does not exist: the lead has not frozen the code"
    try:
        doc = json.loads(path.read_text())
        want = doc["freeze_hash"]
    except (ValueError, KeyError, TypeError) as exc:
        return False, f"freeze file {path} is unreadable ({exc})"
    if doc.get("dirty"):
        return False, f"freeze file {path} was written from a dirty tree ('dirty': true); the lead must commit and freeze again"
    try:
        now = freeze_hash(root)
    except FreezeError as exc:
        return False, str(exc)
    if want != now:
        changed = _changed_files(doc, root)
        return False, (f"the code changed since the freeze: freeze_hash {str(want)[:12]} != current {now[:12]}"
                       + (f" (config differs: {', '.join(changed)})" if changed else ""))
    return True, f"frozen at commit {doc.get('commit')} ({str(want)[:12]})"


def _changed_files(doc: Dict[str, Any], root: Path) -> List[str]:
    saved = (doc.get("config_sha256") or {}).get("files") or {}
    now = config_hashes(root)["files"]
    return sorted(p for p in set(saved) | set(now) if saved.get(p) != now.get(p))


def require_frozen(path: Optional[Path] = None, root: Optional[Path] = None) -> Dict[str, Any]:
    """Raise :class:`FreezeError` unless the freeze file matches the current code. Returns the freeze document."""
    ok, why = check_freeze(path, root)
    if not ok:
        raise FreezeError(why)
    return json.loads((Path(path) if path is not None else FREEZE_FILE).read_text())


def live_hashes(harness: Any, root: Path) -> Dict[str, Any]:
    """The content hashes of what ``harness`` is actually built from (never a path)."""
    est = getattr(harness, "estimate", None)
    table = harness.cell_table()
    return {"rules_sha": harness.rules.sha(), "estimate_sha": None if est is None else est.sha(),
            "cell_table": _sha_text(_canonical(table)), "world_cfg_sha": _sha(harness.world_cfg.to_dict()),
            "colony_cfg_sha": _sha(harness.colony_cfg.to_dict())}


def _sha_text(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


def require_frozen_run(harness: Any, *, path: Optional[Path] = None, root: Optional[Path] = None) -> Dict[str, Any]:
    """The check ``V43Harness.run_episode`` makes before building a world for a sharing arm or control on a development case.

    The file is ``path``, else ``harness.freeze_file``, else the module constant :data:`FREEZE_FILE`. It must pass
    :func:`require_frozen` (tree hash, not dirty-recorded) AND the harness must be built from the frozen CONTENT: its ant
    rules, combined estimate, SimLook cell table (content), world config (one of the three uptake levels, nothing else
    different) and colony config must hash to the recorded values. A mismatch raises :class:`FreezeError` naming the field.
    A harness with an injected ``look_fn_factory`` is refused (its sensor cannot be hashed)."""
    root = Path(root) if root is not None else REPO_ROOT
    path = Path(path or getattr(harness, "freeze_file", None) or FREEZE_FILE)
    doc = require_frozen(path, root)
    for f in REQUIRED_FIELDS:
        if f not in doc:
            raise FreezeError(f"freeze file {path} lacks {f!r} (written by an older freeze tool); the lead must freeze again")
    if getattr(harness, "look_fn_factory", None) is not None:
        raise FreezeError("look_fn_factory: an injected sensor cannot be covered by the freeze")
    live = live_hashes(harness, root)
    for f in ("rules_sha", "estimate_sha", "colony_cfg_sha"):
        if live[f] != doc[f]:
            raise FreezeError(f"{f}: the harness has {str(live[f])[:12]} but the freeze recorded {str(doc[f])[:12]}")
    table_file = root / CELL_TABLE
    if _file_sha(table_file) != doc["cell_table_sha256"]:
        raise FreezeError(f"cell_table_sha256: {CELL_TABLE} differs from the freeze")
    want_table = _sha_text(_canonical(json.loads(table_file.read_text())))
    if live["cell_table"] != want_table:
        raise FreezeError(f"cell_table_sha256: the harness uses a SimLook table whose content differs from {CELL_TABLE}")
    allowed = doc["allowed_world_cfg_sha"]
    if live["world_cfg_sha"] not in set(allowed.values()):
        raise FreezeError(f"allowed_world_cfg_sha: the harness world config ({live['world_cfg_sha'][:12]}) is none of the frozen "
                          f"{WORLD_CONFIG} at levels low/mid/high")
    return doc


def freeze_needed(arms: Sequence[str], *, on_dev_case: bool = True) -> bool:
    """True if any of ``arms`` is a sharing arm or a control (anything but S, S-solo, S') and a development case is involved."""
    from antelligence.experiments.v43_arms import needs_freeze
    return bool(on_dev_case and any(needs_freeze(a) for a in arms))
