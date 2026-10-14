"""Provider-caused queen fallbacks in finished runs (decision 2026-10-13, mistake-2039).

A run whose queen fell back to the S-rule because the PROVIDER failed is invalid for the threshold/router fit and for the
sandbox keep rule. New runs say so in their cell (``queen_fallbacks_provider``). Runs from before that field existed say
so only in their board export, as an ``alarm`` entry whose note starts ``queen fallback: ProviderError`` (the class name of
every provider failure the old queen absorbed). This module reads either. It reads no keys and makes no network call.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterator, Mapping, Optional

PROVIDER_FALLBACK_PREFIXES = ("queen fallback: ProviderError", "queen fallback: ProviderUnavailable")


# Before ProviderTruncated existed, a length cut-off was written as "queen fallback: ProviderError: ... (length)". It is an
# invalid reply, not a provider failure, so those old alarms are not provider-tainted either.
LEGACY_LENGTH_MARKER = "did not finish normally (length)"


def is_provider_fallback_note(note: Any) -> bool:
    """True for an alarm note that records a provider failure. ProviderTruncated (a length cut-off) is an invalid reply."""
    return isinstance(note, str) and note.startswith(PROVIDER_FALLBACK_PREFIXES) and LEGACY_LENGTH_MARKER not in note


def export_files(run_dir: Path) -> Iterator[Path]:
    """The run's board export(s): every ``*.jsonl`` in the run directory except the decision log."""
    for f in sorted(Path(run_dir).glob("*.jsonl")):
        if f.name != "decisions.jsonl":
            yield f


def export_provider_fallbacks(path: Path) -> int:
    """Number of provider-fallback alarm entries in one board export (JSONL: header line, then one entry per line)."""
    n = 0
    for i, line in enumerate(Path(path).read_text(encoding="utf-8").splitlines()):
        if i == 0 or not line.strip():
            continue
        try:
            row = json.loads(line)
        except ValueError:
            continue
        if isinstance(row, dict) and row.get("type") == "alarm" and is_provider_fallback_note((row.get("payload") or {}).get("note")):
            n += 1
    return n


def run_dir_provider_fallbacks(run_dir: Path, cell: Optional[Mapping[str, Any]] = None) -> int:
    """Provider-caused queen fallbacks of one run: the cell's counter when it has one, else the alarms in the export."""
    if cell is not None and cell.get("queen_fallbacks_provider") is not None:
        return int(cell["queen_fallbacks_provider"])
    return sum(export_provider_fallbacks(f) for f in export_files(run_dir))
