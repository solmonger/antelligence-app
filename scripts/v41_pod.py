#!/usr/bin/env python3
"""v4.1 RunPod wrapper around scripts/v4_pod.py (reuses its functions; nothing is forked).

Differences from v4: hard stop $24 (budget $25, ``v41_budget.STOPS``), ledger ``ledger/v4.1-pod-costs.jsonl`` (same
schema ``v4_budget.pod_spend`` reads, plus an ``image`` field), state ``ledger/v4.1-pod-state.json`` (0600), a
read-only ``quote`` command, a pinned image recorded in state and ledger, and an ``--auto-down-after-min`` deadline
that ``status`` reports and ``cost`` warns about once passed. The key is never printed.
"""
from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import sys
import time
from contextlib import contextmanager
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Callable, Dict, Iterator, List, Optional

import httpx

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from antelligence.experiments import v41_budget  # noqa: E402
from scripts import v4_pod  # noqa: E402
from scripts import v41_pod_watchdog as wd  # noqa: E402
from scripts.v41_pod_watchdog import IdleTracker, parse_activity  # noqa: E402

HARD_STOP_USD = v41_budget.STOPS["runpod"]  # 24.00
GPU = "NVIDIA L40S"
IMAGE = v4_pod.VLLM_IMAGE  # pinned by v4_pod (vLLM >= 0.17)
MODEL = v4_pod.WORKERS_KEY
DEFAULT_IDLE_DOWN_MIN = 15.0
WATCH_POLL_S = 60.0
READY_TIMEOUT_ENV = "V41_READY_TIMEOUT_S"  # the launcher's own ready timeout (scripts/v41_pod_serve.sh), default 900 s
DEFAULT_READY_TIMEOUT_S = 900.0
# Usage-ledger ``backend`` values that are NOT served by the pod (Nous queen/FW, Bonsai on the Mac). Any other named
# backend counts as pod activity: an unknown name keeps the pod up (bounded by the deadline) rather than killing a
# working pod. Rows without a backend name (e.g. Nous preflight probes) are not pod activity.
NON_POD_BACKENDS = frozenset({"nous", "nous_fw", "llamacpp", "bonsai"})
WATCH_PORTS = (8100, 8101, 8102)
KEEP_POD_FLAG = "--keep-pod"
Echo = v4_pod.Echo

# Read-only: a ``query`` operation only. ``quote`` refuses to send anything else (see ``assert_readonly``).
QUOTE_QUERY = ("query($id: String!) { gpuTypes(input: {id: $id}) { id displayName securePrice communityPrice "
               "lowestPrice(input: {gpuCount: 1}) { stockStatus availableGpuCounts } } }")


def v41_paths(root: Path = ROOT) -> v4_pod.Paths:
    return v4_pod.Paths(state=root / "ledger" / "v4.1-pod-state.json", costs=root / "ledger" / "v4.1-pod-costs.jsonl")


def validate_image(image: str) -> str:
    name, sep, tag = image.rpartition(":")
    if not sep or "/" in tag or not tag or tag == "latest":
        raise ValueError(f"image {image!r} must carry a pinned tag, never latest")
    return image


def assert_readonly(query: str) -> None:
    if not query.lstrip().startswith("query") or "mutation" in query.lower():
        raise ValueError("quote may only send a GraphQL query")


@contextmanager
def v41_bindings(image: str) -> Iterator[None]:
    """Point v4_pod's module-level knobs at v4.1 for one command: hard stop, image, ledger image field."""
    saved = (v4_pod.VLLM_IMAGE, v4_pod.HARD_STOP_USD, v4_pod.refuse_over_budget, v4_pod.cost_row)
    orig_refuse, orig_row = saved[2], saved[3]

    def refuse(spent: float, rate: float, limit: float = HARD_STOP_USD) -> Optional[str]:
        return orig_refuse(spent, rate, limit)

    def row(event: str, pod_id: str, rate: float, utc: Optional[str] = None, **extra: Any) -> Dict[str, Any]:
        return {**orig_row(event, pod_id, rate, utc, **extra), "image": image}

    v4_pod.VLLM_IMAGE, v4_pod.HARD_STOP_USD = image, HARD_STOP_USD
    v4_pod.refuse_over_budget, v4_pod.cost_row = refuse, row
    try:
        yield
    finally:
        v4_pod.VLLM_IMAGE, v4_pod.HARD_STOP_USD, v4_pod.refuse_over_budget, v4_pod.cost_row = saved


def deadline_passed(state: Dict[str, Any], now: Optional[datetime] = None) -> bool:
    dl = state.get("auto_down_deadline_utc")
    return bool(dl) and (now or datetime.now(timezone.utc)) > datetime.fromisoformat(dl)


def quote(api_key: str, transport: Optional[httpx.BaseTransport] = None) -> Dict[str, Any]:
    """Read-only L40S price and availability. Sends one GraphQL query; starts nothing."""
    assert_readonly(QUOTE_QUERY)
    out: Dict[str, Any] = {"ok": False, "gpu": GPU, "secure_per_hr": None, "community_per_hr": None,
                           "availability": None, "hard_stop_usd": HARD_STOP_USD}
    try:
        with httpx.Client(headers={"Authorization": f"Bearer {api_key}"}, timeout=30.0, trust_env=False,
                          follow_redirects=False, transport=transport) as http:
            r = http.post(v4_pod.GRAPHQL_URL, json={"query": QUOTE_QUERY, "variables": {"id": GPU}})
        out["http"] = r.status_code
        body = r.json() if r.status_code == 200 else {}
        if body.get("errors"):
            out["error"] = "graphql errors"
        rows = (body.get("data") or {}).get("gpuTypes") or []
        if rows:
            g = rows[0]
            out["secure_per_hr"], out["community_per_hr"] = g.get("securePrice"), g.get("communityPrice")
            low = g.get("lowestPrice") or {}
            out["availability"] = low.get("stockStatus")
            out["available_gpu_counts"] = low.get("availableGpuCounts")
            out["ok"] = out["secure_per_hr"] is not None or out["community_per_hr"] is not None
    except (httpx.HTTPError, ValueError) as exc:
        out["error"] = type(exc).__name__
    return out


def spawn_watcher(paths: v4_pod.Paths, pod_id: str, idle_min: float) -> int:
    """Start the detached idle watcher (its own session; output to ``<state>.watch.log``) and return its pid.
    The watcher reads the RunPod key itself, so no key appears in its argv or environment."""
    log = open(str(paths.state) + ".watch.log", "ab")
    proc = subprocess.Popen([sys.executable, str(Path(__file__).resolve()), "watch", "--pod-id", pod_id,
                             "--state", str(paths.state), "--costs", str(paths.costs), "--idle-down-min", str(idle_min)],
                            stdin=subprocess.DEVNULL, stdout=log, stderr=log, start_new_session=True, close_fds=True)
    log.close()
    return proc.pid


def cmd_up(client: v4_pod.RunPod, paths: v4_pod.Paths, image: str, auto_down_min: Optional[float], echo: Echo,
           wait: bool = False, transport: Optional[httpx.BaseTransport] = None,
           sleep: Callable[[float], None] = time.sleep, scripts_commit: Optional[str] = None,
           idle_down_min: float = 0.0, spawn: Callable[[v4_pod.Paths, str, float], int] = spawn_watcher) -> int:
    with v41_bindings(validate_image(image)):
        rc = v4_pod.cmd_up(MODEL, client, paths, echo, False, transport, sleep, scripts_commit)
    state = v4_pod.load_state(paths.state)
    if rc == 0 and state:
        state["image"] = image
        if auto_down_min:
            state["auto_down_deadline_utc"] = (datetime.now(timezone.utc) + timedelta(minutes=auto_down_min)).isoformat()
            echo(f"auto-down deadline {state['auto_down_deadline_utc']} (enforced by the idle watcher)")
        v4_pod.save_state(paths.state, state)
        if idle_down_min > 0:
            state["watcher_pid"] = spawn(paths, state["pod_id"], idle_down_min)
            v4_pod.save_state(paths.state, state)
            echo(f"idle watcher pid {state['watcher_pid']}: pod goes down after {idle_down_min:g} idle min")
        if wait:
            def gone() -> Optional[str]:
                if not v4_pod.load_state(paths.state):
                    return "the pod was taken down (no pod recorded)"
                try:
                    if v4_pod.pod_status(client, state["pod_id"]) == "GONE":
                        return f"pod {state['pod_id']} is gone"
                except (httpx.HTTPError, ValueError):
                    pass
                return None

            return v4_pod.cmd_wait(paths, echo, transport, sleep, abort=gone)
    return rc


def cmd_status(client: v4_pod.RunPod, paths: v4_pod.Paths, echo: Echo) -> int:
    rc = v4_pod.cmd_status(client, paths, echo)
    state = v4_pod.load_state(paths.state)
    if state and state.get("auto_down_deadline_utc"):
        flag = " PASSED: run `down` now" if deadline_passed(state) else ""
        echo(f"auto-down deadline {state['auto_down_deadline_utc']}{flag}")
    return rc


def cmd_cost(paths: v4_pod.Paths, echo: Echo) -> int:
    spent = v4_pod.spend_so_far(paths.costs)
    echo(f"v4.1 runpod spend ${spent:.3f} of ${HARD_STOP_USD:.2f} hard stop (${HARD_STOP_USD - spent:.3f} left)")
    state = v4_pod.load_state(paths.state)
    if state and deadline_passed(state):
        echo(f"WARNING: auto-down deadline {state['auto_down_deadline_utc']} has passed and pod "
             f"{state['pod_id']} is still recorded; run `down`")
    return 0


def _cmdline(pid: int) -> str:
    out = subprocess.run(["ps", "-p", str(pid), "-o", "command="], capture_output=True, text=True)
    return out.stdout.strip()


def kill_watcher(state: Optional[Dict[str, Any]], *, killer: Callable[[int, int], None] = os.kill,
                 cmdline: Callable[[int], str] = _cmdline) -> bool:
    """SIGTERM the recorded idle watcher, only if that pid is still a ``v41_pod.py watch`` (guards against pid reuse)
    and is not this process. True if a signal was sent."""
    pid = (state or {}).get("watcher_pid")
    if not isinstance(pid, int) or pid == os.getpid():
        return False
    cmd = cmdline(pid)
    if "v41_pod.py" not in cmd or "watch" not in cmd:
        return False
    try:
        killer(pid, signal.SIGTERM)
    except OSError:
        return False
    return True


def cmd_down(client: v4_pod.RunPod, paths: v4_pod.Paths, echo: Echo, *,
             killer: Callable[[int, int], None] = os.kill, cmdline: Callable[[int], str] = _cmdline) -> int:
    """Stop the watcher, then terminate the pod. A watcher is stopped even if the terminate fails (the next `down` retries)."""
    state = v4_pod.load_state(paths.state)
    if kill_watcher(state, killer=killer, cmdline=cmdline):
        echo(f"stopped idle watcher pid {state['watcher_pid']}")  # type: ignore[index]
    with v41_bindings(state_image(paths)):
        return v4_pod.cmd_down(client, paths, echo)


def proxy_stem(state: Dict[str, Any]) -> str:
    base = str(state["base_url"]).rstrip("/").removesuffix("/v1")
    return base[: -len("-8100.proxy.runpod.net")]


def proxy_url(state: Dict[str, Any], port: int, path: str) -> str:
    return f"{proxy_stem(state)}-{port}.proxy.runpod.net{path}"


def fetch_activity(state: Dict[str, Any], http: httpx.Client) -> Dict[str, Optional[Dict[str, float]]]:
    """Scrape /metrics of each server through the RunPod proxy. Unreachable, non-200, or reachable with none of the known
    metric names (nothing we can read) -> None, so the caller falls back to the usage ledger."""
    out: Dict[str, Optional[Dict[str, float]]] = {}
    for port in WATCH_PORTS:
        try:
            r = http.get(proxy_url(state, port, "/metrics"), headers={"Authorization": f"Bearer {state['api_secret']}"})
            act = parse_activity(r.text) if r.status_code == 200 else None
            out[str(port)] = act or None
        except httpx.HTTPError:
            out[str(port)] = None
    return out


def models_reachable(state: Dict[str, Any], http: httpx.Client) -> bool:
    """True if the 9B server answers /v1/models through the proxy (the pod is up even if /metrics is not forwarded)."""
    try:
        r = http.get(proxy_url(state, 8100, "/v1/models"), headers={"Authorization": f"Bearer {state['api_secret']}"})
        return r.status_code == 200
    except httpx.HTTPError:
        return False


def read_ledger_rows(path: Path) -> List[Dict[str, Any]]:
    from antelligence.experiments import v4_budget

    return v4_budget.read_jsonl(path)


def last_pod_row_utc(rows: List[Dict[str, Any]]) -> Optional[datetime]:
    """Newest ``utc`` among usage-ledger rows written by pod backends, failed calls included.

    Fail-safe: every named backend outside ``NON_POD_BACKENDS`` (vllm, d1, slm, anything new) counts as the pod."""
    best: Optional[datetime] = None
    for row in rows:
        backend = str(row.get("backend") or "").strip().lower()
        if not backend or backend in NON_POD_BACKENDS or row.get("provider") == "nous":
            continue
        try:
            when = datetime.fromisoformat(str(row["utc"]).replace("Z", "+00:00"))
        except (KeyError, ValueError):
            continue
        if when.tzinfo is None:
            when = when.replace(tzinfo=timezone.utc)
        if best is None or when > best:
            best = when
    return best


def ready_timeout_s(environ: Optional[Dict[str, str]] = None) -> float:
    """The launcher's ready timeout (``V41_READY_TIMEOUT_S``, default 900 s); the watcher's boot grace is built on it."""
    raw = (os.environ if environ is None else environ).get(READY_TIMEOUT_ENV, "")
    try:
        value = float(raw)
    except ValueError:
        return DEFAULT_READY_TIMEOUT_S
    return value if value > 0 else DEFAULT_READY_TIMEOUT_S


def usage_ledger_path(paths: v4_pod.Paths) -> Path:
    return paths.state.parent / "v4.1-usage.jsonl"


def watch_loop(client: v4_pod.RunPod, paths: v4_pod.Paths, pod_id: str, idle_min: float, echo: Echo, *,
               transport: Optional[httpx.BaseTransport] = None, clock: Callable[[], float] = time.monotonic,
               sleep: Callable[[float], None] = time.sleep, now: Callable[[], datetime] = lambda: datetime.now(timezone.utc),
               poll_s: float = WATCH_POLL_S, ready_timeout: Optional[float] = None,
               max_polls: Optional[int] = None, usage: Optional[Path] = None) -> int:
    """Poll status, /metrics and the usage ledger every ``poll_s``; `down` after ``idle_min`` idle minutes, once the
    auto-down deadline passes, or if the pod is gone. Exits quietly when the state file is gone or now belongs to
    another pod (someone ran `down`).

    Idle time is measured from the newest of: the pod's creation time (state ``created_utc``; the watcher's start if that
    is missing), the last metrics-observed activity, the first moment the pod answered, and the last pod-backend usage
    ledger row. It is never measured from before this pod existed (a ledger row from a previous session is irrelevant).

    Boot grace: until the first scrape that shows every server up through /metrics, no idle action is taken for
    ``ready_timeout`` (the launcher's V41_READY_TIMEOUT_S, default 900 s) plus the idle window after the pod's creation.
    Past that, if metrics never became reachable, the usage ledger alone decides (a proxy that does not forward /metrics
    must not get a working pod killed) and the log says why. The deadline and GONE rules apply even during the grace."""
    usage = usage or usage_ledger_path(paths)
    ready_s = ready_timeout_s() if ready_timeout is None else ready_timeout
    grace_s = ready_s + idle_min * 60
    tracker, started, polls = IdleTracker(clock), clock(), 0
    first_ready: Optional[float] = None
    all_up_seen, grace_over_said = False, False
    mode = ""
    with httpx.Client(timeout=20.0, trust_env=False, follow_redirects=False, transport=transport) as http:
        while max_polls is None or polls < max_polls:
            polls += 1
            state = v4_pod.load_state(paths.state)
            if not state or state.get("pod_id") != pod_id:
                echo("watcher: pod no longer recorded; exiting")
                return 0
            reason = None
            try:
                status = v4_pod.pod_status(client, pod_id)
            except (httpx.HTTPError, ValueError):
                status = None
            baseline = started
            created = parse_utc(state.get("created_utc"))
            if created is not None:
                baseline = min(started, clock() - max((now() - created).total_seconds(), 0.0))
            activity = fetch_activity(state, http)
            metrics_ok = any(v is not None for v in activity.values())
            all_up_seen = all_up_seen or all(activity.get(str(p)) is not None for p in WATCH_PORTS)
            tracker.observe(activity)
            ready = metrics_ok or models_reachable(state, http)
            if ready and first_ready is None:
                first_ready = clock()
            new_mode = "metrics+ledger" if metrics_ok else "ledger-only"
            if new_mode != mode and (metrics_ok or ready or polls > 1):
                mode = new_mode
                if not metrics_ok:
                    echo("watcher: /metrics is not readable through the proxy; judging idleness on the usage ledger alone")
            ledger_dt = last_pod_row_utc(read_ledger_rows(usage))
            candidates = [t for t in (baseline, tracker.last_active, first_ready,
                                      clock() - (now() - ledger_dt).total_seconds() if ledger_dt else None) if t is not None]
            idle = clock() - max(candidates)
            in_grace = not all_up_seen and (clock() - baseline) < grace_s
            if status == "GONE":
                reason = "pod is gone"
            elif deadline_passed(state, now()):
                reason = "auto-down deadline passed"
            elif idle >= idle_min * 60 and not in_grace:
                how = "metrics and usage ledger" if metrics_ok else "usage ledger only (metrics unreachable)"
                reason = f"idle {idle / 60:.1f} min by {how}"
                if not all_up_seen:
                    reason += (f"; metrics never showed every server up within the {grace_s / 60:.0f} min boot grace "
                               f"({ready_s:.0f} s ready timeout + {idle_min:g} min idle window), so the usage ledger decided")
            if not all_up_seen and not in_grace and not grace_over_said and not reason:
                grace_over_said = True
                echo(f"watcher: boot grace of {grace_s / 60:.0f} min is over and /metrics never showed every server up; "
                     "the usage-ledger idle rule now applies")
            if reason:
                echo(f"watcher: {reason}; taking the pod down")
                return cmd_down(client, paths, echo)
            sleep(poll_s)
    return 0


def parse_utc(raw: Any) -> Optional[datetime]:
    try:
        when = datetime.fromisoformat(str(raw).replace("Z", "+00:00"))
    except ValueError:
        return None
    return when if when.tzinfo else when.replace(tzinfo=timezone.utc)


DIAG_TAIL_DEFAULT = 40


def _scrub(text: str, secret: str) -> str:
    return wd.redact(text.replace(secret, "[redacted]") if secret else text, None, limit=100_000)


def cmd_diag(paths: v4_pod.Paths, echo: Echo, *, transport: Optional[httpx.BaseTransport] = None,
             tail: int = DIAG_TAIL_DEFAULT) -> int:
    """Plain-text pod health through the proxy: per port /v1/models and /metrics status, which watchdog metric names are
    present (with values) or missing, then the watchdog log tail from the d1 shim. Makes only GET requests, needs no
    RunPod key, and never prints any secret (every line is scrubbed). Safe to paste."""
    state = v4_pod.load_state(paths.state)
    if not state or "base_url" not in state:
        echo("diag: no pod recorded")
        return 1
    secret = str(state.get("api_secret", ""))
    say = lambda line: echo(_scrub(line, secret))  # noqa: E731
    headers = {"Authorization": f"Bearer {secret}"}
    say(f"diag: pod {state.get('pod_id')} image {state.get('image', IMAGE)}")
    problems: List[str] = []
    with httpx.Client(timeout=20.0, trust_env=False, follow_redirects=False, transport=transport) as http:
        def get(port: int, path: str) -> Optional[httpx.Response]:
            try:
                return http.get(proxy_url(state, port, path), headers=headers)
            except httpx.HTTPError as exc:
                say(f"  :{port}{path.split('?')[0]} -> {type(exc).__name__}")
                return None

        for port in WATCH_PORTS:
            models = get(port, "/v1/models")
            if models is not None:
                say(f"  :{port} /v1/models -> HTTP {models.status_code}")
            resp = get(port, "/metrics")
            if resp is None:
                problems.append(f":{port} /metrics unreachable through the proxy")
                continue
            say(f"  :{port} /metrics -> HTTP {resp.status_code}")
            if resp.status_code != 200:
                problems.append(f":{port} /metrics HTTP {resp.status_code} (proxy may not forward it)")
                continue
            act = parse_activity(resp.text)
            names = wd.metric_names(resp.text)
            missing = wd.missing_core(act, port)
            say("    present: " + (", ".join(f"{k}={v:g}" for k, v in sorted(act.items())) or "(none of the watched names)"))
            say("    missing core names: " + (", ".join(missing) or "(none)"))
            say(f"    exported metric names ({len(names)}): " + ", ".join(names[:25]) + (" ..." if len(names) > 25 else ""))
            if not act:
                problems.append(f":{port} /metrics reachable but none of the watched names are present")
        log = get(8102, f"/v41/watchdog-log?n={tail}")
    say(f"watchdog log tail (:8102 /v41/watchdog-log) -> " + ("unreachable" if log is None else f"HTTP {log.status_code}"))
    lines: List[Any] = []
    if log is not None and log.status_code == 200:
        try:
            body = log.json()
            lines = body.get("lines", [])
            if not body.get("exists"):
                problems.append("watchdog log file does not exist on the pod (the watchdog never started)")
        except ValueError:
            problems.append("watchdog log reply was not JSON")
    elif log is not None:
        problems.append(f"watchdog log HTTP {log.status_code}")
    for line in lines:
        say("  " + json.dumps(line, sort_keys=True))
    attempts = [ln for ln in lines if isinstance(ln, dict) and ln.get("event") == "selfterm"]
    if attempts:
        say("self-termination attempts: " + "; ".join(f"{a.get('route')}={a.get('status')}/{a.get('outcome')}" for a in attempts[-8:]))
    for problem in problems:
        say(f"PROBLEM: {problem}")
    return 0


class _PodDown:
    """SIGTERM handler context: turns SIGTERM into SystemExit so ``finally`` blocks run (SIGINT already raises)."""

    def __enter__(self) -> None:
        self.old = None
        try:
            self.old = signal.signal(signal.SIGTERM, lambda _s, _f: sys.exit(143))
        except ValueError:  # not the main thread
            pass

    def __exit__(self, *exc: Any) -> None:
        if self.old is not None:
            signal.signal(signal.SIGTERM, self.old)


def quiet_down(paths: Optional[v4_pod.Paths] = None, echo: Echo = lambda m: print(m, file=sys.stderr), *,
               key_fn: Callable[[], str] = v4_pod.runpod_key,
               transport: Optional[httpx.BaseTransport] = None) -> int:
    """Bring the pod down if one is recorded; silent when none is. Never raises, never prints the key."""
    paths = paths or v41_paths()
    try:
        if not v4_pod.load_state(paths.state):
            return 0
        return cmd_down(v4_pod.RunPod(key_fn(), transport), paths, echo)
    except (Exception, SystemExit) as exc:
        echo(f"pod down FAILED ({type(exc).__name__}); the pod may still be billing: run scripts/v41_pod.py down")
        return 1


def run_with_pod_down(fn: Callable[[], int], *, keep_pod: bool = False, paths: Optional[v4_pod.Paths] = None,
                      down: Callable[..., int] = quiet_down) -> int:
    """Run ``fn`` and bring the pod down however it ends: normal return, error, KeyboardInterrupt or SIGTERM.
    ``keep_pod`` skips the down."""
    if keep_pod:
        return fn()
    with _PodDown():
        try:
            return fn()
        finally:
            down(paths)


def run_main_with_pod_down(main: Callable[[Optional[List[str]]], int], argv: Optional[List[str]] = None, *,
                           parser: Optional[Callable[[], argparse.ArgumentParser]] = None) -> int:
    """Thin wrapper for the ``__main__`` of v41_run / v41_sandbox / v41_latency.

    Strips ``--keep-pod``, then validates the remaining arguments with the script's own ``parser()`` (passed as
    ``parser=``, else the ``parser`` attribute of ``main``'s module) BEFORE entering the down-on-exit scope. So
    ``--help``, ``-h``, an unknown flag or a missing required argument exits through argparse (SystemExit) without
    touching the pod: a help call must never terminate a live pod another process is using (mistake-2043).

    Once the arguments parse, the run has started: ANY exit (return, error, exception, KeyboardInterrupt, SIGTERM,
    exit 75) brings the pod down unless ``--keep-pod`` was given. A ``main`` module with no ``parser`` is run
    without pre-validation (legacy behaviour)."""
    args = list(sys.argv[1:] if argv is None else argv)
    keep = KEEP_POD_FLAG in args
    rest = [a for a in args if a != KEEP_POD_FLAG]
    make = parser or getattr(sys.modules.get(getattr(main, "__module__", "")), "parser", None)
    if make is not None:
        make().parse_args(rest)  # argparse help / usage errors raise SystemExit here, before any down()
    return run_with_pod_down(lambda: main(rest), keep_pod=keep)


def state_image(paths: v4_pod.Paths) -> str:
    state = v4_pod.load_state(paths.state)
    return (state or {}).get("image") or IMAGE


def main(argv: Optional[List[str]] = None, *, paths: Optional[v4_pod.Paths] = None,
         transport: Optional[httpx.BaseTransport] = None, api_key: Optional[str] = None,
         sleep: Callable[[float], None] = time.sleep, echo: Echo = print,
         spawn: Callable[[v4_pod.Paths, str, float], int] = spawn_watcher) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="cmd", required=True)
    sub.add_parser("quote", help="read-only: L40S price and availability as one JSON line")
    up = sub.add_parser("up")
    up.add_argument("model", choices=[MODEL])
    up.add_argument("--image", default=IMAGE, help=f"pinned image tag (default {IMAGE})")
    up.add_argument("--scripts-commit", required=True, metavar="SHA",
                    help="40-hex commit; the pod downloads its launcher from GitHub at this commit and verifies sha256")
    up.add_argument("--wait", action="store_true")
    up.add_argument("--auto-down-after-min", type=float, default=None, metavar="N")
    up.add_argument("--idle-down-min", type=float, default=DEFAULT_IDLE_DOWN_MIN, metavar="N",
                    help="a detached watcher takes the pod down after N idle minutes (default 15; 0 = no watcher)")
    watch = sub.add_parser("watch", help="internal: the idle watcher `up` spawns")
    watch.add_argument("--pod-id", required=True)
    watch.add_argument("--state", required=True)
    watch.add_argument("--costs", required=True)
    watch.add_argument("--idle-down-min", type=float, default=DEFAULT_IDLE_DOWN_MIN)
    diag = sub.add_parser("diag", help="plain-text pod health through the proxy plus the watchdog log tail (no key needed)")
    diag.add_argument("--tail", type=int, default=DIAG_TAIL_DEFAULT, metavar="N", help="watchdog log lines to print")
    for name in ("status", "wait", "down", "cost"):
        sub.add_parser(name)
    args = parser.parse_args(argv)
    paths = paths or v41_paths()
    if args.cmd == "watch":
        paths = v4_pod.Paths(state=Path(args.state), costs=Path(args.costs))
    if args.cmd == "cost":
        return cmd_cost(paths, echo)
    if args.cmd == "diag":
        return cmd_diag(paths, echo, transport=transport, tail=args.tail)
    if args.cmd == "wait":
        return v4_pod.cmd_wait(paths, echo, transport, sleep)
    key = api_key or v4_pod.runpod_key()
    if args.cmd == "quote":
        echo(json.dumps(quote(key, transport)))
        return 0
    client = v4_pod.RunPod(key, transport)
    if args.cmd == "up":
        try:
            return cmd_up(client, paths, args.image, args.auto_down_after_min, echo, args.wait, transport, sleep,
                          args.scripts_commit, args.idle_down_min, spawn)
        except ValueError as exc:
            echo(f"refused: {exc}")
            return 2
    if args.cmd == "status":
        return cmd_status(client, paths, echo)
    if args.cmd == "watch":
        return watch_loop(client, paths, args.pod_id, args.idle_down_min, echo, transport=transport, sleep=sleep)
    return cmd_down(client, paths, echo)


if __name__ == "__main__":
    sys.exit(main())
