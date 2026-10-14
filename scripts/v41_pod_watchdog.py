#!/usr/bin/env python3
"""Pod-side idle watchdog for the v4.1 worker pod (standard library only; fetched and sha256-verified like serve.sh).

Run by ``v41_pod_serve.sh`` once every server is ready. Every ``V41_WATCHDOG_POLL_S`` (default 60) seconds it scrapes the
local Prometheus ``/metrics`` of the vLLM servers (:8100, :8101) and the d1 shim (:8102). If no request started or
finished on any server for ``V41_IDLE_SHUTDOWN_S`` seconds (default 900) the pod terminates itself.

Observability (S19). Every poll appends one JSON line to ``V41_WATCHDOG_LOG`` (default ``/opt/v41/watchdog-log.jsonl``):
per server the scrape HTTP status, the metric values used, the core names that were missing and any error; the idle
seconds; and the decision. Every self-termination attempt appends one line (method, path with the pod id elided, HTTP
status, redacted body). The d1 shim serves the tail at ``GET /v41/watchdog-log?n=200`` and ``v41_pod.py diag`` prints
it. The same lines go to stdout, flushed (a block-buffered pipe can hide them otherwise).

Metric names. vLLM changed its names between 0.6 and 0.30 and some builds export ``vllm_`` instead of ``vllm:``, so
names are canonicalised to the ``vllm:`` form and several aliases are accepted. A reachable ``/metrics`` with none of
the known names is an ERROR, not idleness: it holds the idle clock (a pod we cannot read is never killed on that basis)
and is logged loudly. The Mac watcher and the run-end ``down`` remain the guard in that case.

Self-termination uses the pod-scoped ``RUNPOD_API_KEY`` and ``RUNPOD_POD_ID`` that RunPod injects into every pod
(https://docs.runpod.io/pods/references/environment-variables). Routes are tried in the order of ``ROUTES`` and each
result is logged. The account key is never sent to the pod.

    v41_pod_watchdog.py                 # watch loop
    v41_pod_watchdog.py --terminate-now # one self-termination attempt (used by the launcher when serve.sh ends)
"""
from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import sys
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, List, Mapping, Optional, Tuple, Union

REST_URL = "https://rest.runpod.io/v1"
REST_V2_URL = "https://api.runpod.io/v2"
GRAPHQL_URL = "https://api.runpod.io/graphql"
DEFAULT_IDLE_S = 900.0
DEFAULT_POLL_S = 60.0
DEFAULT_LOG = "/opt/v41/watchdog-log.jsonl"
PORTS = (8100, 8101, 8102)

# Counters: any change means a request was served. Gauges: a non-zero value means one is in flight.
# Canonical names use the "vllm:" prefix. Aliases cover vLLM v0.6 .. v0.30 (request counters, token counters and the
# histogram ``_count`` series, which only ever increase when a request finishes or an iteration runs).
_VLLM_COUNTERS = ("request_success_total", "request_success", "prompt_tokens_total", "generation_tokens_total",
                  "e2e_request_latency_seconds_count", "request_queue_time_seconds_count",
                  "request_prefill_time_seconds_count", "request_decode_time_seconds_count",
                  "request_prompt_tokens_count", "request_generation_tokens_count", "num_preemptions_total")
_VLLM_GAUGES = ("num_requests_running", "num_requests_waiting", "num_requests_swapped")
COUNTERS = tuple(f"vllm:{n}" for n in _VLLM_COUNTERS) + ("v41_d1_requests_total",)
GAUGES = tuple(f"vllm:{n}" for n in _VLLM_GAUGES) + ("v41_d1_in_flight",)
# What a healthy server must expose; names absent from a scrape are reported as "missing" (diagnostics only).
CORE_VLLM = ("vllm:num_requests_running", "vllm:num_requests_waiting", "vllm:request_success_total")
CORE_D1 = ("v41_d1_requests_total", "v41_d1_in_flight")
_LINE = re.compile(r"^([A-Za-z_:][A-Za-z0-9_:]*)(?:\{[^}]*\})?\s+([-+0-9.eEinfNa]+)\s*$")
_KEY_LIKE = re.compile(r"(rpa_[A-Za-z0-9_\-]{6,}|Bearer\s+\S+)")

Activity = Dict[str, float]


def core_names(port: int) -> Tuple[str, ...]:
    return CORE_D1 if port == 8102 else CORE_VLLM


def canonical(name: str) -> str:
    """``vllm_foo`` -> ``vllm:foo`` (some builds export the sanitised form); everything else unchanged."""
    return "vllm:" + name[len("vllm_"):] if name.startswith("vllm_") else name


def metric_names(text: str) -> List[str]:
    """Distinct canonical metric names in a Prometheus text body (diagnostics: shows what a server really exports)."""
    seen: Dict[str, None] = {}
    for line in text.splitlines():
        if line.startswith("#"):
            continue
        m = _LINE.match(line.strip())
        if m:
            seen[canonical(m.group(1))] = None
    return list(seen)


def parse_activity(text: str) -> Activity:
    """Sum each watched metric over its label sets. Unknown metrics are ignored; none known -> ``{}``."""
    out: Activity = {}
    for line in text.splitlines():
        if line.startswith("#"):
            continue
        m = _LINE.match(line.strip())
        if not m:
            continue
        name = canonical(m.group(1))
        if name in COUNTERS or name in GAUGES:
            try:
                out[name] = out.get(name, 0.0) + float(m.group(2))
            except ValueError:
                continue
    return out


def missing_core(act: Activity, port: int) -> List[str]:
    """Core names absent from ``act`` (``request_success`` counts for ``request_success_total``)."""
    return [n for n in core_names(port) if n not in act and not (n == "vllm:request_success_total" and "vllm:request_success" in act)]


def is_busy(act: Activity) -> bool:
    return any(act.get(g, 0.0) > 0 for g in GAUGES)


class IdleTracker:
    """Idle time across sources. ``observe`` takes {source: Activity or None (unreachable)} and returns idle seconds.

    The clock starts at the first successful scrape (a booting pod is not idle). Busy gauges or any counter change
    reset it; an unreachable source counts as no activity. A reachable source with no known metric names (``{}``) is
    unreadable, not idle: it holds the clock and is listed in ``unreadable``."""

    def __init__(self, clock: Callable[[], float] = time.monotonic) -> None:
        self.clock = clock
        self.last: Dict[str, Activity] = {}
        self.last_active: Optional[float] = None
        self.unreadable: List[str] = []

    def observe(self, snapshot: Mapping[str, Optional[Activity]]) -> float:
        now = self.clock()
        reachable = {k: v for k, v in snapshot.items() if v is not None}
        self.unreadable = sorted(k for k, v in reachable.items() if not v)
        if not reachable and self.last_active is None:
            return 0.0
        if self.last_active is None:
            self.last_active = now  # first sight: the idle clock starts here
        if self.unreadable:
            self.last_active = now
        for src, act in reachable.items():
            if not act:
                continue
            prev = self.last.get(src)
            counters = {c: act.get(c) for c in COUNTERS if c in act}
            if is_busy(act) or (prev is not None and prev != counters):
                self.last_active = now
            self.last[src] = counters
        return now - self.last_active


# --- one scrape, with enough detail to log ------------------------------------------------------------------------


@dataclass
class Scrape:
    """Result of one /metrics request. ``activity`` None = unreachable or non-200; ``{}`` = no known names."""
    status: Optional[int]
    activity: Optional[Activity]
    missing: List[str] = field(default_factory=list)
    error: Optional[str] = None

    def record(self) -> Dict[str, Any]:
        out: Dict[str, Any] = {"status": self.status, "values": self.activity or {}, "missing": self.missing}
        if self.error:
            out["error"] = self.error
        return out


def scrape_port(port: int, token: str, timeout: float = 10.0) -> Scrape:
    try:
        req = urllib.request.Request(f"http://127.0.0.1:{port}/metrics", headers={"Authorization": f"Bearer {token}"})
        with urllib.request.urlopen(req, timeout=timeout) as r:
            status, text = int(r.status), r.read().decode("utf-8", "replace")
    except urllib.error.HTTPError as exc:
        return Scrape(int(exc.code), None, list(core_names(port)), f"HTTP {exc.code}")
    except Exception as exc:  # connection refused while a server restarts, timeouts, ...
        return Scrape(None, None, list(core_names(port)), type(exc).__name__)
    act = parse_activity(text)
    err = None if act else "metrics reachable but no known metric names (exported: " + ", ".join(metric_names(text)[:12]) + ")"
    return Scrape(status, act, missing_core(act, port), err)


def read_metrics(port: int, token: str) -> Optional[Activity]:
    """The activity of one port, None if unreachable."""
    return scrape_port(port, token).activity


# --- logging ------------------------------------------------------------------------------------------------------


def _now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def redact(text: str, env: Optional[Mapping[str, str]] = None, limit: int = 300) -> str:
    """Strip key-looking tokens and the exact secret values from a log string, then truncate."""
    for name in ("RUNPOD_API_KEY", "VLLM_API_KEY"):
        secret = (env or {}).get(name, "")
        if secret:
            text = text.replace(secret, "[redacted]")
    return _KEY_LIKE.sub("[redacted]", text)[:limit]


class Sink:
    """Appends JSON lines to the watchdog log file and echoes them (flushed) to stdout. Never raises."""

    def __init__(self, path: Optional[str] = None, env: Optional[Mapping[str, str]] = None, echo: bool = True) -> None:
        self.path, self.env, self.echo = path, env, echo

    def event(self, record: Mapping[str, Any]) -> None:
        line = json.dumps({"ts": _now(), **record}, sort_keys=True, default=str)
        if self.path:
            try:
                Path(self.path).parent.mkdir(parents=True, exist_ok=True)
                with open(self.path, "a", encoding="utf-8") as fh:
                    fh.write(line + "\n")
            except OSError:
                pass
        if self.echo:
            print(line, flush=True)

    def msg(self, text: str) -> None:
        self.event({"event": "msg", "msg": redact(text, self.env, 500)})


# --- self-termination --------------------------------------------------------------------------------------------

Response = Union[int, Tuple[int, str]]
Request = Callable[..., Response]
RunCmd = Callable[[List[str], Mapping[str, str]], Tuple[int, str]]

# (name, kind, method, url or argv template, GraphQL document). Order matters: the first success wins. Terminate routes
# stop billing for compute and container disk; stop routes are the fallback (a stopped pod still bills storage).
# REST v1 is what docs.runpod.io/pods/manage-pods documents; v2 and GraphQL are what the other docs pages show.
ROUTES: Tuple[Tuple[str, str, str, str, Optional[str]], ...] = (
    # every terminate route first: a stop that succeeds early would leave the pod billing storage
    ("rest-v1-delete", "http", "DELETE", REST_URL + "/pods/{id}", None),
    ("rest-v2-delete", "http", "DELETE", REST_V2_URL + "/pods/{id}", None),
    ("graphql-podTerminate", "http", "POST", GRAPHQL_URL, 'mutation { podTerminate(input: {podId: "{id}"}) }'),
    ("runpodctl-pod-delete", "cli", "", "runpodctl pod delete {id}", None),  # spelling in docs.runpod.io/pods/manage-pods
    ("runpodctl-remove-pod", "cli", "", "runpodctl remove pod {id}", None),  # older runpodctl spelling
    # stop fallbacks (GPU billing stops; storage still bills)
    ("rest-v1-stop", "http", "POST", REST_URL + "/pods/{id}/stop", None),
    ("graphql-podStop", "http", "POST", GRAPHQL_URL, 'mutation { podStop(input: {podId: "{id}"}) { id desiredStatus } }'),
    ("runpodctl-pod-stop", "cli", "", "runpodctl pod stop {id}", None),
    ("runpodctl-stop-pod", "cli", "", "runpodctl stop pod {id}", None),
)
TERMINATE_ROUTES = ("rest-v1-delete", "rest-v2-delete", "graphql-podTerminate", "runpodctl-pod-delete", "runpodctl-remove-pod")


def _http(method: str, url: str, headers: Mapping[str, str], body: Optional[str] = None) -> Tuple[int, str]:
    try:
        data = json.dumps({"query": body}).encode() if body is not None else None
        hdrs = dict(headers)
        if data is not None:
            hdrs["Content-Type"] = "application/json"
        req = urllib.request.Request(url, method=method, headers=hdrs, data=data)
        with urllib.request.urlopen(req, timeout=30) as r:
            return int(r.status), r.read(2000).decode("utf-8", "replace")
    except urllib.error.HTTPError as exc:
        try:
            text = exc.read(2000).decode("utf-8", "replace")
        except Exception:
            text = ""
        return int(exc.code), text


def _run(argv: List[str], env: Mapping[str, str]) -> Tuple[int, str]:
    out = subprocess.run(argv, capture_output=True, text=True, timeout=60, env=dict(env))
    return out.returncode, (out.stdout + out.stderr)


def _unpack(res: Response) -> Tuple[int, str]:
    return (res[0], res[1]) if isinstance(res, tuple) else (int(res), "")


def terminate_self(env: Mapping[str, str], request: Request = _http, log: Callable[[str], None] = print, *,
                   events: Optional[Callable[[Mapping[str, Any]], None]] = None, run: RunCmd = _run,
                   which: Callable[[str], Optional[str]] = shutil.which) -> str:
    """'terminated', 'stopped', 'no-credentials' or 'failed'. Tries ``ROUTES`` in order, logs each result, never logs
    the key. ``request(method, url, headers[, graphql_document])`` returns a status or ``(status, body)``."""
    emit = events or (lambda _r: None)
    pod_id, key = env.get("RUNPOD_POD_ID", ""), env.get("RUNPOD_API_KEY", "")
    if not pod_id or not key:
        log("watchdog: RUNPOD_POD_ID / RUNPOD_API_KEY missing; cannot self-terminate")
        emit({"event": "selfterm", "route": "-", "outcome": "no-credentials",
              "have_pod_id": bool(pod_id), "have_api_key": bool(key)})
        return "no-credentials"
    headers = {"Authorization": f"Bearer {key}", "User-Agent": "v41-pod-watchdog/1"}
    for name, kind, method, template, doc in ROUTES:
        word = "terminated" if name in TERMINATE_ROUTES else "stopped"
        path = template.replace("{id}", "<id>")
        record: Dict[str, Any] = {"event": "selfterm", "route": name, "method": method or "CLI",
                                  "path": path.replace("https://", "")}
        try:
            if kind == "cli":
                argv = template.replace("{id}", pod_id).split()
                if not which(argv[0]):
                    record.update(status=None, outcome="skipped", body="runpodctl not in the image")
                    emit(record)
                    log(f"watchdog: {name} skipped (runpodctl not found)")
                    continue
                status, text = run(argv, env)
                ok = status == 0
            else:
                url = template.replace("{id}", pod_id)
                if doc is not None:
                    status, text = _unpack(request(method, url, headers, doc.replace("{id}", pod_id)))
                    ok = 200 <= status < 300 and '"errors"' not in text  # GraphQL refuses with 200 + errors
                else:
                    status, text = _unpack(request(method, url, headers))
                    ok = 200 <= status < 300
        except Exception as exc:  # network error: try the next route
            record.update(status=None, outcome="error", body=redact(type(exc).__name__, env))
            emit(record)
            log(f"watchdog: {name} raised {type(exc).__name__}")
            continue
        record.update(status=status, outcome="ok" if ok else "refused", body=redact(text, env))
        emit(record)
        log(f"watchdog: {name} {path} -> {'exit' if kind == 'cli' else 'HTTP'} {status}")
        if ok:
            return word
    return "failed"


# --- the loop -----------------------------------------------------------------------------------------------------


def _as_scrape(value: Any, port: str) -> Scrape:
    if isinstance(value, Scrape):
        return value
    if value is None:
        return Scrape(None, None, list(core_names(int(port) if port.isdigit() else 0)), "unreachable")
    return Scrape(200, value, [])


def watch(env: Mapping[str, str], *, idle_s: float, poll_s: float, clock: Callable[[], float] = time.monotonic,
          sleep: Callable[[float], None] = time.sleep,
          scrape: Optional[Callable[[], Mapping[str, Any]]] = None,
          request: Request = _http, log: Callable[[str], None] = print, max_polls: Optional[int] = None,
          events: Optional[Callable[[Mapping[str, Any]], None]] = None, run: RunCmd = _run,
          which: Callable[[str], Optional[str]] = shutil.which) -> int:
    """Loop until idle for ``idle_s`` then self-terminate. 0 if the pod was terminated or stopped, 3 if it could not be."""
    token = env.get("VLLM_API_KEY", "")
    scrape = scrape or (lambda: {str(p): scrape_port(p, token) for p in PORTS})
    emit = events or (lambda _r: None)
    tracker = IdleTracker(clock)
    log(f"watchdog: started, idle limit {idle_s:.0f}s, poll {poll_s:.0f}s")
    emit({"event": "start", "idle_limit_s": idle_s, "poll_s": poll_s, "pod_id_present": bool(env.get("RUNPOD_POD_ID")),
          "api_key_present": bool(env.get("RUNPOD_API_KEY")), "runpodctl": bool(which("runpodctl"))})
    polls = 0
    while max_polls is None or polls < max_polls:
        polls += 1
        scrapes = {k: _as_scrape(v, k) for k, v in scrape().items()}
        idle = tracker.observe({k: s.activity for k, s in scrapes.items()})
        decision = "terminate" if idle >= idle_s else "wait"
        poll_rec: Dict[str, Any] = {"event": "poll", "servers": {k: s.record() for k, s in scrapes.items()},
                                    "idle_s": round(idle, 1), "decision": decision}
        if tracker.unreadable:
            poll_rec["error"] = "metrics reachable but no known names on: " + ",".join(tracker.unreadable)
        emit(poll_rec)
        if decision == "terminate":
            log(f"watchdog: idle {idle:.0f}s >= {idle_s:.0f}s; terminating the pod")
            outcome = terminate_self(env, request, log, events=events, run=run, which=which)
            emit({"event": "selfterm-result", "outcome": outcome})
            if outcome in ("terminated", "stopped"):
                log(f"watchdog: pod {outcome}")
                return 0
            log(f"watchdog: self-termination {outcome}; will retry")
            return 3
        log(f"watchdog: idle {idle:.0f}s")
        sleep(poll_s)
    return 1


def main(argv: Optional[List[str]] = None, env: Optional[Mapping[str, str]] = None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    env = os.environ if env is None else env
    sink = Sink(env.get("V41_WATCHDOG_LOG", DEFAULT_LOG), env)
    if "--terminate-now" in argv:
        sink.event({"event": "terminate-now"})
        return 0 if terminate_self(env, _http, sink.msg, events=sink.event) in ("terminated", "stopped") else 3
    idle_s = float(env.get("V41_IDLE_SHUTDOWN_S", DEFAULT_IDLE_S))
    poll_s = float(env.get("V41_WATCHDOG_POLL_S", DEFAULT_POLL_S))
    while True:
        rc = watch(env, idle_s=idle_s, poll_s=poll_s, log=sink.msg, events=sink.event)
        if rc != 3:
            return rc
        time.sleep(poll_s)  # termination refused: keep trying, keep logging


if __name__ == "__main__":
    sys.exit(main())
