"""Run a short program a model wrote, as safely as plain Python allows.

The code solver asks a model to write a program that checks every option against a puzzle's rules. That program is
model output, so before it runs it must pass a strict check of its parse tree:

* imports only from a small list of pure modules (itertools, math, fractions, collections, functools, re, string,
  json, decimal, operator, statistics);
* none of ``open``, ``exec``, ``eval``, ``compile``, ``__import__``, ``input``, ``globals``, ``locals``, ``vars``,
  ``getattr``, ``setattr``, ``delattr``, ``breakpoint``, ``exit``, ``quit``, ``help``, ``memoryview``;
* no name or attribute that starts with an underscore (that closes the usual ways out through ``__class__`` and
  friends).

Then it runs in a separate ``python -I -S`` process (no site packages, no user paths, no environment variables) in an
empty temporary folder, with a time limit and, where the system allows, CPU and memory limits. This is a guard
against mistakes, not against a determined attacker: never feed it code from an untrusted person.
"""

from __future__ import annotations

import ast
import os
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from typing import Optional

ALLOWED_MODULES = frozenset({"itertools", "math", "fractions", "collections", "functools", "re", "string", "json",
                             "decimal", "operator", "statistics"})
BANNED_NAMES = frozenset({"open", "exec", "eval", "compile", "__import__", "input", "globals", "locals", "vars",
                          "getattr", "setattr", "delattr", "breakpoint", "exit", "quit", "help", "memoryview",
                          "classmethod", "staticmethod", "super", "type", "object"})
MAX_CODE_CHARS = 12_000
MAX_OUTPUT_CHARS = 4_000


class Refused(ValueError):
    """The program uses something the sandbox doesn't allow."""


def check(code: str) -> None:
    """Raise :class:`Refused` with a plain reason when the program uses anything not allowed."""
    if len(code) > MAX_CODE_CHARS:
        raise Refused(f"the program is longer than {MAX_CODE_CHARS} characters")
    try:
        tree = ast.parse(code)
    except SyntaxError as exc:
        raise Refused(f"syntax error on line {exc.lineno}: {exc.msg}") from None
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name.split(".")[0] not in ALLOWED_MODULES:
                    raise Refused(f"import of {alias.name!r} is not allowed")
        elif isinstance(node, ast.ImportFrom):
            if node.level or (node.module or "").split(".")[0] not in ALLOWED_MODULES:
                raise Refused(f"import from {node.module!r} is not allowed")
        elif isinstance(node, ast.Name):
            if node.id in BANNED_NAMES or (node.id.startswith("_") and node.id not in ("_", "__name__")):
                raise Refused(f"the name {node.id!r} is not allowed")
        elif isinstance(node, ast.Attribute):
            if node.attr.startswith("_"):
                raise Refused(f"the attribute {node.attr!r} is not allowed")
        elif isinstance(node, (ast.Global, ast.Nonlocal)):
            raise Refused("global and nonlocal are not allowed")


@dataclass(frozen=True)
class Result:
    ok: bool                      # it ran and exited cleanly
    stdout: str
    error: str                    # a short plain reason when not ok
    seconds: float


_RUNNER = """\
import sys
try:
    import resource
    cpu, mem = int(sys.argv[2]), int(sys.argv[3])
    resource.setrlimit(resource.RLIMIT_CPU, (cpu, cpu))
    resource.setrlimit(resource.RLIMIT_FSIZE, (1 << 20, 1 << 20))
    if sys.platform != "darwin":   # macOS refuses an address-space limit
        resource.setrlimit(resource.RLIMIT_AS, (mem << 20, mem << 20))
except (ImportError, ValueError, OSError):
    pass
with open(sys.argv[1]) as fh:
    source = fh.read()
exec(compile(source, "prog.py", "exec"), {"__name__": "__main__"})
"""


def run(code: str, *, timeout_s: float = 20.0, mem_mb: int = 1024, python: Optional[str] = None) -> Result:
    """Check, then run, a program. Never raises for the program's own failures; they come back in ``error``."""
    import time

    try:
        check(code)
    except Refused as exc:
        return Result(False, "", f"refused: {exc}", 0.0)
    start = time.monotonic()
    with tempfile.TemporaryDirectory(prefix="harness-sbx-") as d:
        path = os.path.join(d, "prog.py")
        with open(path, "w") as fh:
            fh.write(code)
        try:
            p = subprocess.run([python or sys.executable, "-I", "-S", "-c", _RUNNER, path, str(int(timeout_s) + 1),
                                str(mem_mb)], cwd=d, env={}, capture_output=True, text=True, timeout=timeout_s,
                               stdin=subprocess.DEVNULL)
        except subprocess.TimeoutExpired:
            return Result(False, "", f"it ran longer than {timeout_s:.0f} seconds", time.monotonic() - start)
    secs = time.monotonic() - start
    if p.returncode != 0:
        tail = (p.stderr or "").strip().splitlines()[-3:]
        return Result(False, p.stdout[:MAX_OUTPUT_CHARS], "it crashed: " + " | ".join(tail)[:500], secs)
    return Result(True, p.stdout[:MAX_OUTPUT_CHARS], "", secs)
