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
from scripts.v41_pod_watchdog import IdleTracker, parse_activity  # noqa: E402

HARD_STOP_USD = v41_budget.STOPS["runpod"]  # 24.00
GPU = "NVIDIA L40S"
IMAGE = v4_pod.VLLM_IMAGE  # pinned by v4_pod (vLLM >= 0.17)
MODEL = v4_pod.WORKERS_KEY
DEFAULT_IDLE_DOWN_MIN = 15.0
WATCH_POLL_S = 60.0
BOOT_TIMEOUT_MIN = 45.0  # the watcher gives up on a pod that never serves metrics
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
            return v4_pod.cmd_wait(paths, echo, transport, sleep)
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


def fetch_activity(state: Dict[str, Any], http: httpx.Client) -> Dict[str, Optional[Dict[str, float]]]:
    """Scrape /metrics of each server through the RunPod proxy. Unreachable -> None."""
    base = str(state["base_url"]).rstrip("/").removesuffix("/v1")
    stem = base[: -len("-8100.proxy.runpod.net")]
    out: Dict[str, Optional[Dict[str, float]]] = {}
    for port in WATCH_PORTS:
        try:
            r = http.get(f"{stem}-{port}.proxy.runpod.net/metrics", headers={"Authorization": f"Bearer {state['api_secret']}"})
            out[str(port)] = parse_activity(r.text) if r.status_code == 200 else None
        except httpx.HTTPError:
            out[str(port)] = None
    return out


def watch_loop(client: v4_pod.RunPod, paths: v4_pod.Paths, pod_id: str, idle_min: float, echo: Echo, *,
               transport: Optional[httpx.BaseTransport] = None, clock: Callable[[], float] = time.monotonic,
               sleep: Callable[[float], None] = time.sleep, now: Callable[[], datetime] = lambda: datetime.now(timezone.utc),
               poll_s: float = WATCH_POLL_S, boot_timeout_min: float = BOOT_TIMEOUT_MIN,
               max_polls: Optional[int] = None) -> int:
    """Poll status and /metrics every ``poll_s``; `down` after ``idle_min`` idle minutes, once the auto-down deadline
    passes, if the pod is gone, or if it never serves metrics within ``boot_timeout_min``. Exits quietly when the state
    file is gone or now belongs to another pod (someone ran `down`)."""
    tracker, started, polls = IdleTracker(clock), clock(), 0
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
            idle = tracker.observe(fetch_activity(state, http))
            if status == "GONE":
                reason = "pod is gone"
            elif deadline_passed(state, now()):
                reason = "auto-down deadline passed"
            elif idle >= idle_min * 60:
                reason = f"idle {idle / 60:.1f} min"
            elif tracker.last_active is None and (clock() - started) >= boot_timeout_min * 60:
                reason = f"no metrics {boot_timeout_min:g} min after start"
            if reason:
                echo(f"watcher: {reason}; taking the pod down")
                return cmd_down(client, paths, echo)
            sleep(poll_s)
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


def run_main_with_pod_down(main: Callable[[Optional[List[str]]], int], argv: Optional[List[str]] = None) -> int:
    """Thin wrapper for the ``__main__`` of v41_run / v41_sandbox / v41_latency: strips ``--keep-pod`` and runs main."""
    args = list(sys.argv[1:] if argv is None else argv)
    keep = KEEP_POD_FLAG in args
    rest = [a for a in args if a != KEEP_POD_FLAG]
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
    for name in ("status", "wait", "down", "cost"):
        sub.add_parser(name)
    args = parser.parse_args(argv)
    paths = paths or v41_paths()
    if args.cmd == "watch":
        paths = v4_pod.Paths(state=Path(args.state), costs=Path(args.costs))
    if args.cmd == "cost":
        return cmd_cost(paths, echo)
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
