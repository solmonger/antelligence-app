"""One scrubber for every piece of text that is stored (sqlite, cells, ledgers, archives) or printed.

Removes: URL userinfo, path and query (``scheme://host[:port]`` is kept); Bearer / Authorization values; ``sk-`` style
and 64-hex tokens; and any secret value passed in or registered with :func:`register_secret` (RPC URL, pod secret,
Nous key). Servers may echo what they were sent, so response bodies go through this too.
"""

from __future__ import annotations

import re
import threading
from typing import Iterable, Set

_URL = re.compile(r"(https?://)(?:[^/\s'\"<>@]*@)?([^/\s'\"<>?#]+)[^\s'\"<>)\]]*", re.IGNORECASE)
_BEARER = re.compile(r"(?i)\b(?:authorization:?\s*)?bearer\s+['\"]?[^\s'\",;)]+|\b(?:authorization:?|api[_-]?key[=:])\s*['\"]?(?!bearer\b)[^\s'\",;)]+")
_URLPATH = re.compile(r"(?i)\b(url:?\s*)/\S+")
_TOKENS = re.compile(r"\bsk-[A-Za-z0-9_\-]{8,}|\b(?:0x)?[0-9a-fA-F]{64}\b")
_lock = threading.Lock()
_registered: Set[str] = set()


def register_secret(*values: str) -> None:
    with _lock:
        _registered.update(v for v in values if v and len(v) >= 6)


def register_url(url: str) -> None:
    """Register an endpoint URL and its secret-bearing parts (userinfo, path, query), which libraries echo on their own."""
    from urllib.parse import urlsplit

    register_secret(url)
    parts = urlsplit(url)
    register_secret(parts.password or "", parts.path.strip("/"), parts.query)
    for seg in parts.path.split("/") + parts.query.replace("&", "=").split("="):
        if len(seg) >= 12:
            register_secret(seg)


def scrub(text: object, extra_secrets: Iterable[str] = ()) -> str:
    out = str(text)
    with _lock:
        secrets = sorted(_registered | {s for s in extra_secrets if s}, key=len, reverse=True)
    for s in secrets:                      # exact values first, before URL stripping rewrites the context
        out = out.replace(s, "***")
    out = _URL.sub(lambda m: m.group(1) + m.group(2), out)
    out = _BEARER.sub("Bearer ***", out)
    out = _URLPATH.sub(lambda m: m.group(1) + "<path>", out)
    return _TOKENS.sub("***", out)
