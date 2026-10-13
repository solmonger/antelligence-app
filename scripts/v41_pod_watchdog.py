#!/usr/bin/env python3
"""Pod-side idle watchdog for the v4.1 worker pod (standard library only; fetched and sha256-verified like serve.sh).

Run by ``v41_pod_serve.sh`` once every server is ready. Every ``V41_WATCHDOG_POLL_S`` (default 60) seconds it scrapes the
local Prometheus ``/metrics`` of the vLLM servers (:8100, :8101) and the d1 shim (:8102). If no request started or
finished on any server for ``V41_IDLE_SHUTDOWN_S`` seconds (default 900) the pod terminates itself.

Self-termination uses the pod-scoped ``RUNPOD_API_KEY`` and ``RUNPOD_POD_ID`` that RunPod injects into every pod
(https://docs.runpod.io/pods/references/environment-variables): ``DELETE https://rest.runpod.io/v1/pods/<id>``
(terminate: compute and container disk stop billing), then ``POST .../stop`` if the delete is refused (stop: GPU
stops billing, a stopped pod still bills storage). The account key is never sent to the pod. Decisions go to stdout.

    v41_pod_watchdog.py                 # watch loop
    v41_pod_watchdog.py --terminate-now # one self-termination attempt (used by the launcher when serve.sh ends)
"""
from __future__ import annotations

import os
import re
import sys
import time
import urllib.request
from typing import Callable, Dict, List, Mapping, Optional, Tuple

REST_URL = "https://rest.runpod.io/v1"
DEFAULT_IDLE_S = 900.0
DEFAULT_POLL_S = 60.0
PORTS = (8100, 8101, 8102)
# Counters: any change means a request was served. Gauges: a non-zero value means one is in flight.
# vLLM names ("vllm:" prefix, Prometheus text) per the vLLM metrics docs; d1 names come from scripts/v41_d1_server.py.
COUNTERS = ("vllm:request_success_total", "vllm:prompt_tokens_total", "vllm:generation_tokens_total",
            "v41_d1_requests_total")
GAUGES = ("vllm:num_requests_running", "vllm:num_requests_waiting", "v41_d1_in_flight")
_LINE = re.compile(r"^([A-Za-z_:][A-Za-z0-9_:]*)(?:\{[^}]*\})?\s+([-+0-9.eEinfNa]+)\s*$")

Activity = Dict[str, float]


def parse_activity(text: str) -> Activity:
    """Sum each watched metric over its label sets. Missing metrics are absent from the result."""
    out: Activity = {}
    for line in text.splitlines():
        if line.startswith("#"):
            continue
        m = _LINE.match(line.strip())
        if m and m.group(1) in COUNTERS + GAUGES:
            try:
                out[m.group(1)] = out.get(m.group(1), 0.0) + float(m.group(2))
            except ValueError:
                continue
    return out


def is_busy(act: Activity) -> bool:
    return any(act.get(g, 0.0) > 0 for g in GAUGES)


class IdleTracker:
    """Idle time across sources. ``observe`` takes {source: Activity or None (unreachable)} and returns idle seconds.

    The clock starts at the first successful scrape (a booting pod is not idle). Busy gauges or any counter change
    reset it; an unreachable source counts as no activity."""

    def __init__(self, clock: Callable[[], float] = time.monotonic) -> None:
        self.clock = clock
        self.last: Dict[str, Activity] = {}
        self.last_active: Optional[float] = None

    def observe(self, snapshot: Mapping[str, Optional[Activity]]) -> float:
        now = self.clock()
        reachable = {k: v for k, v in snapshot.items() if v is not None}
        if not reachable and self.last_active is None:
            return 0.0
        if self.last_active is None:
            self.last_active = now  # first sight: the idle clock starts here
        for src, act in reachable.items():
            prev = self.last.get(src)
            counters = {c: act.get(c) for c in COUNTERS if c in act}
            if is_busy(act) or (prev is not None and prev != counters):
                self.last_active = now
            self.last[src] = counters
        return now - self.last_active


Request = Callable[[str, str, Mapping[str, str]], int]


def _http(method: str, url: str, headers: Mapping[str, str]) -> int:
    try:
        req = urllib.request.Request(url, method=method, headers=dict(headers))
        with urllib.request.urlopen(req, timeout=30) as r:
            return int(r.status)
    except urllib.error.HTTPError as exc:
        return int(exc.code)


def terminate_self(env: Mapping[str, str], request: Request = _http, log: Callable[[str], None] = print) -> str:
    """'terminated', 'stopped', 'no-credentials' or 'failed'. Never logs the key."""
    pod_id, key = env.get("RUNPOD_POD_ID", ""), env.get("RUNPOD_API_KEY", "")
    if not pod_id or not key:
        log("watchdog: RUNPOD_POD_ID / RUNPOD_API_KEY missing; cannot self-terminate")
        return "no-credentials"
    headers = {"Authorization": f"Bearer {key}", "User-Agent": "v41-pod-watchdog/1"}
    for method, suffix, word in (("DELETE", "", "terminated"), ("POST", "/stop", "stopped")):
        try:
            status = request(method, f"{REST_URL}/pods/{pod_id}{suffix}", headers)
        except Exception as exc:  # network error: try the next route
            log(f"watchdog: {method} {suffix or '/'} raised {type(exc).__name__}")
            continue
        log(f"watchdog: {method} /pods/<id>{suffix} -> HTTP {status}")
        if 200 <= status < 300:
            return word
    return "failed"


def read_metrics(port: int, token: str) -> Optional[Activity]:
    try:
        req = urllib.request.Request(f"http://127.0.0.1:{port}/metrics", headers={"Authorization": f"Bearer {token}"})
        with urllib.request.urlopen(req, timeout=10) as r:
            return parse_activity(r.read().decode("utf-8", "replace"))
    except Exception:
        return None


def watch(env: Mapping[str, str], *, idle_s: float, poll_s: float, clock: Callable[[], float] = time.monotonic,
          sleep: Callable[[float], None] = time.sleep,
          scrape: Optional[Callable[[], Mapping[str, Optional[Activity]]]] = None,
          request: Request = _http, log: Callable[[str], None] = print, max_polls: Optional[int] = None) -> int:
    """Loop until idle for ``idle_s`` then self-terminate. 0 if the pod was terminated or stopped, 3 if it could not be."""
    token = env.get("VLLM_API_KEY", "")
    scrape = scrape or (lambda: {str(p): read_metrics(p, token) for p in PORTS})
    tracker = IdleTracker(clock)
    log(f"watchdog: started, idle limit {idle_s:.0f}s, poll {poll_s:.0f}s")
    polls = 0
    while max_polls is None or polls < max_polls:
        polls += 1
        idle = tracker.observe(scrape())
        if idle >= idle_s:
            log(f"watchdog: idle {idle:.0f}s >= {idle_s:.0f}s; terminating the pod")
            outcome = terminate_self(env, request, log)
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
    if "--terminate-now" in argv:
        return 0 if terminate_self(env) in ("terminated", "stopped") else 3
    idle_s = float(env.get("V41_IDLE_SHUTDOWN_S", DEFAULT_IDLE_S))
    poll_s = float(env.get("V41_WATCHDOG_POLL_S", DEFAULT_POLL_S))
    while True:
        rc = watch(env, idle_s=idle_s, poll_s=poll_s)
        if rc != 3:
            return rc
        time.sleep(poll_s)  # termination refused: keep trying, keep logging


if __name__ == "__main__":
    sys.exit(main())
