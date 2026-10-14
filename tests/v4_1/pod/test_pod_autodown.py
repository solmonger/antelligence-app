"""S17: the pod never bills idle. Fake transports, fake clock, fake metrics server; no network, no RunPod, no key."""
from __future__ import annotations

import importlib.util
import json
import os
import signal
import subprocess
import sys
import threading
from datetime import datetime, timedelta, timezone
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

import httpx
import pytest

from scripts import v41_pod, v4_pod
from scripts import v41_pod_watchdog as wd

REPO = Path(__file__).resolve().parents[3]
FAKE_KEY = "rpa_FAKEKEY_do_not_print_123456"
POD_KEY = "rpa_PODSCOPED_do_not_print_987"


def metrics(success=0, running=0, waiting=0, prompt=0):
    return (f'# HELP vllm:request_success_total x\nvllm:request_success_total{{finished_reason="stop"}} {success}.0\n'
            f'vllm:request_success_total{{finished_reason="length"}} 0.0\nvllm:prompt_tokens_total{{model_name="m"}} {prompt}.0\n'
            f'vllm:num_requests_running{{model_name="m"}} {running}.0\nvllm:num_requests_waiting{{model_name="m"}} {waiting}.0\n')


class Clock:
    def __init__(self):
        self.t = 1000.0

    def __call__(self):
        return self.t

    def sleep(self, s):
        self.t += s


# --- pod side: parsing, idle tracking, termination ------------------------------------------------------------


def test_parse_activity_sums_labels_and_ignores_comments():
    a = wd.parse_activity(metrics(success=3, running=2, prompt=40))
    assert a["vllm:request_success_total"] == 3.0 and a["vllm:num_requests_running"] == 2.0
    assert a["vllm:prompt_tokens_total"] == 40.0 and "vllm:num_requests_waiting" in a
    assert wd.parse_activity("garbage\n# x 1\nother_metric 5\n") == {}


def test_idle_clock_starts_at_first_scrape_not_at_boot():
    c = Clock()
    t = wd.IdleTracker(c)
    assert t.observe({"8100": None}) == 0.0
    c.t += 5000  # a long boot, nothing reachable yet
    assert t.observe({"8100": None}) == 0.0
    assert t.observe({"8100": wd.parse_activity(metrics())}) == 0.0
    c.t += 600
    assert t.observe({"8100": wd.parse_activity(metrics())}) == 600


def test_no_false_trigger_while_requests_flow():
    c = Clock()
    t = wd.IdleTracker(c)
    for n in range(30):  # 30 min of one request finishing every minute
        assert t.observe({"8100": wd.parse_activity(metrics(success=n))}) <= 60
        c.t += 60
    # in-flight request with no completed ones: the running gauge counts as activity
    for _ in range(30):
        assert t.observe({"8101": wd.parse_activity(metrics(success=7, running=1))}) == 0
        c.t += 60


def test_activity_on_any_one_server_resets_the_clock():
    c = Clock()
    t = wd.IdleTracker(c)
    t.observe({"8100": wd.parse_activity(metrics(success=1)), "8102": {"v41_d1_requests_total": 0.0}})
    c.t += 800
    assert t.observe({"8100": wd.parse_activity(metrics(success=1)), "8102": {"v41_d1_requests_total": 1.0}}) == 0
    c.t += 800
    assert t.observe({"8100": wd.parse_activity(metrics(success=1)), "8102": {"v41_d1_requests_total": 1.0}}) == 800


def test_watch_terminates_after_idle_limit_and_logs_no_key():
    c, logs, calls = Clock(), [], []

    def request(method, url, headers):
        calls.append((method, url, headers["Authorization"]))
        return 204

    env = {"RUNPOD_POD_ID": "podX", "RUNPOD_API_KEY": POD_KEY}
    rc = wd.watch(env, idle_s=900, poll_s=60, clock=c, sleep=c.sleep, request=request, log=logs.append,
                  scrape=lambda: {"8100": wd.parse_activity(metrics(success=1))})
    assert rc == 0
    assert calls == [("DELETE", "https://rest.runpod.io/v1/pods/podX", f"Bearer {POD_KEY}")]
    assert c.t - 1000.0 == 900  # 15 polls of 60 s
    assert not any(POD_KEY in line for line in logs) and any("terminating" in line for line in logs)


def test_watch_does_not_terminate_while_busy():
    c, calls = Clock(), []
    n = iter(range(10_000))
    rc = wd.watch({"RUNPOD_POD_ID": "p", "RUNPOD_API_KEY": "k"}, idle_s=900, poll_s=60, clock=c, sleep=c.sleep,
                  request=lambda *a: calls.append(a) or 204, log=lambda _m: None, max_polls=100,
                  scrape=lambda: {"8100": wd.parse_activity(metrics(success=next(n)))})
    assert rc == 1 and calls == []


def test_terminate_falls_back_to_stop_then_reports_failure():
    seen = []

    def delete_refused(method, url, headers):
        seen.append(method)
        return 403 if method == "DELETE" else 200

    assert wd.terminate_self({"RUNPOD_POD_ID": "p", "RUNPOD_API_KEY": "k"}, delete_refused, lambda _m: None) == "stopped"
    assert seen == ["DELETE", "DELETE", "POST"]
    assert wd.terminate_self({"RUNPOD_POD_ID": "p", "RUNPOD_API_KEY": "k"}, lambda *a: 401, lambda _m: None) == "failed"

    def boom(*_a):
        raise OSError("down")

    assert wd.terminate_self({"RUNPOD_POD_ID": "p", "RUNPOD_API_KEY": "k"}, boom, lambda _m: None) == "failed"
    assert wd.terminate_self({"RUNPOD_POD_ID": "p"}, lambda *a: 204, lambda _m: None) == "no-credentials"


def test_watch_returns_3_when_termination_fails():
    c = Clock()
    rc = wd.watch({"RUNPOD_POD_ID": "p", "RUNPOD_API_KEY": "k"}, idle_s=120, poll_s=60, clock=c, sleep=c.sleep,
                  request=lambda *a: 500, log=lambda _m: None, scrape=lambda: {"8100": wd.parse_activity(metrics())})
    assert rc == 3


class _FakeMetrics(BaseHTTPRequestHandler):
    body = metrics()
    auth_seen: list = []

    def do_GET(self):
        type(self).auth_seen.append(self.headers.get("Authorization"))
        raw = type(self).body.encode()
        self.send_response(200 if self.path == "/metrics" else 404)
        self.send_header("Content-Length", str(len(raw)))
        self.end_headers()
        self.wfile.write(raw)

    def log_message(self, *_):
        return


def test_watchdog_against_a_fake_local_metrics_server():
    handler = type("H", (_FakeMetrics,), {"body": metrics(success=1), "auth_seen": []})
    srv = HTTPServer(("127.0.0.1", 0), handler)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    try:
        port = srv.server_address[1]
        c, calls = Clock(), []
        tick = {"n": 0}

        def scrape():
            tick["n"] += 1
            if tick["n"] == 5:  # a request lands on poll 5: the clock must restart there
                handler.body = metrics(success=2)
            return {"8100": wd.read_metrics(port, "tok"), "8101": wd.read_metrics(1, "tok")}  # 8101 unreachable

        rc = wd.watch({"RUNPOD_POD_ID": "p", "RUNPOD_API_KEY": "k"}, idle_s=900, poll_s=60, clock=c, sleep=c.sleep,
                      request=lambda m, u, h: calls.append(m) or 204, log=lambda _m: None, scrape=scrape)
        assert rc == 0 and calls == ["DELETE"] and tick["n"] == 5 + 15
        assert set(handler.auth_seen) == {"Bearer tok"}
    finally:
        srv.shutdown()


def test_main_terminate_now_and_env_knobs(monkeypatch):
    sent = []
    monkeypatch.setattr(wd, "_http", lambda m, u, h: sent.append((m, u)) or 204)
    monkeypatch.setattr(wd, "terminate_self", lambda env, request=wd._http, log=print, **kw: "terminated")
    assert wd.main(["--terminate-now"], {"RUNPOD_POD_ID": "p", "RUNPOD_API_KEY": "k"}) == 0
    seen = {}
    monkeypatch.setattr(wd, "watch", lambda env, idle_s, poll_s, **kw: seen.update(idle=idle_s, poll=poll_s) or 0)
    assert wd.main([], {"V41_IDLE_SHUTDOWN_S": "30", "V41_WATCHDOG_POLL_S": "5"}) == 0 and seen == {"idle": 30.0, "poll": 5.0}
    assert wd.main([], {}) == 0 and seen == {"idle": 900.0, "poll": 60.0}  # default 15 min, 60 s


def test_d1_shim_exposes_activity_counters():
    spec = importlib.util.spec_from_file_location("v41_d1_server", REPO / "scripts" / "v41_d1_server.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    act = wd.parse_activity(mod.metrics_text())
    assert act == {"v41_d1_requests_total": 0.0, "v41_d1_in_flight": 0.0}


# --- launcher wiring -------------------------------------------------------------------------------------------


def test_launcher_starts_watchdog_after_all_servers_ready_and_is_fetched():
    text = (REPO / "scripts" / "v41_pod_serve.sh").read_text()
    assert subprocess.run(["bash", "-n", str(REPO / "scripts" / "v41_pod_serve.sh")]).returncode == 0
    assert text.index("ready: d1-3B") < text.index('python3 "$WATCHDOG" &')
    assert "V41_IDLE_SHUTDOWN_S" in text and "RUNPOD_API_KEY" not in text.replace("pod-scoped", "")
    assert v4_pod.WORKERS_FETCH["v41_pod_watchdog.py"] == "scripts/v41_pod_watchdog.py"


def test_start_script_never_idles_and_self_terminates_on_exit():
    pin = v4_pod.ScriptsPin("a" * 40, {n: "b" * 64 for n in v4_pod.WORKERS_FETCH})
    cmd = v4_pod.deploy_payload("v41-workers", "secret", "NVIDIA L40S", "tok", pin)["dockerStartCmd"][0]
    assert "sleep infinity" not in cmd and "trap on_exit EXIT" in cmd and "kill 1" in cmd
    assert cmd.index("trap on_exit EXIT") < cmd.index("fetch_verified")  # covers fetch/hash failures too
    assert "RUNPOD_API_KEY" in cmd and "secret" not in cmd and len(cmd) < 8000
    assert subprocess.run(["bash", "-n", "-c", cmd]).returncode == 0


def test_inline_self_terminate_snippet_runs_and_refuses_without_credentials():
    ast_ok = compile(v4_pod.SELF_TERMINATE_PY, "<selfterm>", "exec")
    assert ast_ok
    env = {k: v for k, v in os.environ.items() if not k.startswith("RUNPOD")}
    r = subprocess.run([sys.executable, "-c", v4_pod.SELF_TERMINATE_PY], capture_output=True, text=True, env=env)
    assert r.returncode == 1 and "no pod credentials" in r.stdout


# --- Mac side: finally/signal path ----------------------------------------------------------------------------


def test_down_runs_on_normal_end_error_interrupt_and_sigterm():
    for how in ("ok", "error", "interrupt", "sigterm"):
        downs = []

        def fn(how=how):
            if how == "error":
                raise RuntimeError("boom")
            if how == "interrupt":
                raise KeyboardInterrupt
            if how == "sigterm":
                os.kill(os.getpid(), signal.SIGTERM)
            return 0

        before = signal.getsignal(signal.SIGTERM)
        if how == "ok":
            assert v41_pod.run_with_pod_down(fn, down=lambda p: downs.append(1)) == 0
        else:
            with pytest.raises((RuntimeError, KeyboardInterrupt, SystemExit)):
                v41_pod.run_with_pod_down(fn, down=lambda p: downs.append(1))
        assert downs == [1], how
        assert signal.getsignal(signal.SIGTERM) is before  # handler restored


def test_keep_pod_skips_down_and_flag_is_stripped():
    downs, got = [], []
    rc = v41_pod.run_with_pod_down(lambda: 0, keep_pod=True, down=lambda p: downs.append(1))
    assert rc == 0 and downs == []
    v41_pod.run_main_with_pod_down(lambda argv: got.append(argv) or 0, ["--keep-pod", "--x", "1"])
    assert got == [["--x", "1"]]


def test_quiet_down_is_silent_without_a_pod_and_never_prints_the_key(tmp_path):
    paths = v4_pod.Paths(state=tmp_path / "s.json", costs=tmp_path / "c.jsonl")
    out: list = []
    assert v41_pod.quiet_down(paths, out.append, key_fn=lambda: pytest.fail("key must not be read")) == 0
    assert out == []
    v4_pod.save_state(paths.state, {"pod_id": "p1", "api_secret": "s", "created_utc": "2026-10-13T00:00:00+00:00"})

    def refuse(req):
        return httpx.Response(500)

    assert v41_pod.quiet_down(paths, out.append, key_fn=lambda: FAKE_KEY, transport=httpx.MockTransport(refuse)) == 1
    assert out and not any(FAKE_KEY in m for m in out)
    assert v41_pod.quiet_down(paths, out.append, key_fn=lambda: (_ for _ in ()).throw(SystemExit("no key"))) == 1
    assert "may still be billing" in out[-1]


def test_entry_points_are_wrapped_in_pod_down():
    for name in ("v41_run", "v41_sandbox", "v41_latency"):
        text = (REPO / "scripts" / f"{name}.py").read_text()
        assert "raise SystemExit(run_main_with_pod_down(main))" in text, name


# --- Mac side: watcher ------------------------------------------------------------------------------------------


def started(tmp_path, idle_min=15.0, auto_down=None, spawn=None):
    log: list = []
    spawned: list = []

    def handler(req: httpx.Request) -> httpx.Response:
        log.append((req.method, str(req.url)))
        if req.url.host == "api.runpod.io":
            return httpx.Response(200, json={"data": {"gpuTypes": [{"securePrice": 1.0}]}})
        if req.method == "POST":
            return httpx.Response(201, json={"id": "pod1", "costPerHr": 1.0})
        if req.method == "DELETE":
            return httpx.Response(204)
        if req.url.host == "rest.runpod.io":
            return httpx.Response(200, json={"desiredStatus": "RUNNING"})
        return metrics_response(req)

    state = {"body": metrics(success=1), "up": True}

    def metrics_response(req):
        if not state["up"]:
            raise httpx.ConnectError("boot")
        return httpx.Response(200, text=state["body"])

    paths = v4_pod.Paths(state=tmp_path / "ledger" / "st.json", costs=tmp_path / "ledger" / "c.jsonl")
    transport = httpx.MockTransport(handler)
    argv = ["up", "v41-workers", "--scripts-commit", subprocess.run(["git", "rev-parse", "HEAD"], cwd=REPO, capture_output=True,
                                                                     text=True).stdout.strip(), "--idle-down-min", str(idle_min)]
    if auto_down:
        argv += ["--auto-down-after-min", str(auto_down)]
    out: list = []
    rc = v41_pod.main(argv, paths=paths, transport=transport, api_key=FAKE_KEY, echo=out.append,
                      spawn=spawn or (lambda p, pid, m: spawned.append((pid, m)) or 777))
    assert rc == 0
    return transport, paths, log, state, spawned, out


def test_up_spawns_watcher_with_default_15_and_records_pid(tmp_path):
    _, paths, _, _, spawned, out = started(tmp_path, idle_min=15.0)
    assert spawned == [("pod1", 15.0)] and v4_pod.load_state(paths.state)["watcher_pid"] == 777
    assert FAKE_KEY not in json.dumps(out)
    parser_default = v41_pod.DEFAULT_IDLE_DOWN_MIN
    assert parser_default == 15.0


def test_idle_down_zero_spawns_nothing(tmp_path):
    _, paths, _, _, spawned, _ = started(tmp_path, idle_min=0)
    assert spawned == [] and "watcher_pid" not in v4_pod.load_state(paths.state)


def run_watch(transport, paths, c, **kw):
    out: list = []
    client = v4_pod.RunPod(FAKE_KEY, transport)
    rc = v41_pod.watch_loop(client, paths, "pod1", 15.0, out.append, transport=transport, clock=c, sleep=c.sleep, **kw)
    return rc, out


def test_watcher_takes_pod_down_after_idle_and_not_before(tmp_path):
    transport, paths, log, state, _, _ = started(tmp_path)
    c = Clock()
    rc, out = run_watch(transport, paths, c, max_polls=10)  # 10 min of constant counters: still up
    assert rc == 0 and not any(m == "DELETE" for m, _ in log) and paths.state.exists()
    rc, out = run_watch(transport, paths, c)  # run on to 15 idle minutes
    assert ("DELETE", "https://rest.runpod.io/v1/pods/pod1") in log and not paths.state.exists()
    assert any("idle 15.0 min" in m for m in out)
    stops = [json.loads(line) for line in paths.costs.read_text().splitlines()]
    assert stops[-1]["event"] == "stop"


def test_watcher_does_not_fire_while_requests_flow(tmp_path):
    transport, paths, log, state, _, _ = started(tmp_path)
    c, n = Clock(), {"i": 0}

    def sleep(s):
        n["i"] += 1
        state["body"] = metrics(success=n["i"])  # a request finishes every poll
        c.sleep(s)

    out: list = []
    v41_pod.watch_loop(v4_pod.RunPod(FAKE_KEY, transport), paths, "pod1", 15.0, out.append, transport=transport, clock=c,
                       sleep=sleep, max_polls=120)
    assert paths.state.exists() and not any(m == "DELETE" for m, _ in log)


def test_booting_pod_with_old_ledger_row_is_not_killed_during_grace_then_is_after(tmp_path):
    """Live incident 2026-10-13: a row from a session 230 min earlier plus unreachable metrics killed a booting pod."""
    transport, paths, log, state, _, _ = started(tmp_path)
    state["up"] = False  # /metrics unreachable while booting
    c, base = Clock(), datetime.now(timezone.utc)
    now = lambda: base + timedelta(seconds=c.t - 1000.0)  # noqa: E731
    usage = paths.state.parent / "v4.1-usage.jsonl"
    old = (base - timedelta(minutes=230)).strftime("%Y-%m-%dT%H:%M:%S.%fZ")
    usage.write_text(json.dumps({"utc": old, "backend": "vllm", "model": "m", "role": "worker"}) + "\n")
    out: list = []
    client = v4_pod.RunPod(FAKE_KEY, transport)
    kw = dict(transport=transport, clock=c, sleep=c.sleep, now=now, ready_timeout=900.0)
    v41_pod.watch_loop(client, paths, "pod1", 15.0, out.append, max_polls=29, **kw)  # 29 min < 15 + 15 grace
    assert paths.state.exists() and not any(m == "DELETE" for m, _ in log)
    assert not any("taking the pod down" in m for m in out)
    v41_pod.watch_loop(client, paths, "pod1", 15.0, out.append, **kw)  # no activity ever: down after the grace
    assert not paths.state.exists() and ("DELETE", "https://rest.runpod.io/v1/pods/pod1") in log
    assert any("usage ledger only" in m and "boot grace" in m for m in out)
    assert (c.t - 1000.0) / 60 < 40  # measured from this pod's creation, not from the 230-minute-old row


def test_idle_is_measured_from_pod_creation_never_before(tmp_path):
    transport, paths, log, state, _, _ = started(tmp_path)
    state["up"] = False  # nothing answers: the baseline is the creation time
    s = v4_pod.load_state(paths.state)
    c, base = Clock(), datetime.now(timezone.utc)
    s["created_utc"] = (base - timedelta(minutes=5)).isoformat()  # created 5 min before the watcher started
    v4_pod.save_state(paths.state, s)
    out: list = []
    v41_pod.watch_loop(v4_pod.RunPod(FAKE_KEY, transport), paths, "pod1", 15.0, out.append, transport=transport, clock=c,
                       sleep=c.sleep, now=lambda: base + timedelta(seconds=c.t - 1000.0), ready_timeout=0.001)
    # idle counted from creation (t=-300 s): down ~10 min into the watcher's life, not 15
    assert any("idle 16.0 min" in m for m in out), out  # 15 min grace end + one 60 s poll
    assert (c.t - 1000.0) <= 11 * 60


def test_active_pod_with_fresh_ledger_rows_is_never_taken_down(tmp_path):
    transport, paths, log, state, _, _ = started(tmp_path)
    state["up"] = False  # metrics never reachable: only the ledger can keep it up
    c, base = Clock(), datetime.now(timezone.utc)
    now = lambda: base + timedelta(seconds=c.t - 1000.0)  # noqa: E731
    usage = paths.state.parent / "v4.1-usage.jsonl"

    def sleep(s):
        stamp = now().strftime("%Y-%m-%dT%H:%M:%S.%fZ")
        usage.write_text(json.dumps({"utc": stamp, "backend": "d1", "model": "m", "role": "worker"}) + "\n")
        c.sleep(s)

    out: list = []
    v41_pod.watch_loop(v4_pod.RunPod(FAKE_KEY, transport), paths, "pod1", 15.0, out.append, transport=transport, clock=c,
                       sleep=sleep, now=now, ready_timeout=900.0, max_polls=180)
    assert paths.state.exists() and not any(m == "DELETE" for m, _ in log)


def test_ready_timeout_env_knob():
    assert v41_pod.ready_timeout_s({}) == 900.0
    assert v41_pod.ready_timeout_s({"V41_READY_TIMEOUT_S": "120"}) == 120.0
    assert v41_pod.ready_timeout_s({"V41_READY_TIMEOUT_S": "junk"}) == 900.0


def test_up_wait_stops_when_the_pod_disappears(tmp_path):
    log: list = []
    gone = {"v": False}

    def handler(req: httpx.Request) -> httpx.Response:
        log.append((req.method, str(req.url)))
        if req.url.host == "api.runpod.io":
            return httpx.Response(200, json={"data": {"gpuTypes": [{"securePrice": 1.0}]}})
        if req.method == "POST":
            return httpx.Response(201, json={"id": "pod1", "costPerHr": 1.0})
        if req.url.host == "rest.runpod.io":
            return httpx.Response(404) if gone["v"] else httpx.Response(200, json={"desiredStatus": "RUNNING"})
        raise httpx.ConnectError("booting")

    paths = v4_pod.Paths(state=tmp_path / "ledger" / "st.json", costs=tmp_path / "ledger" / "c.jsonl")
    sleeps: list = []

    def sleep(s):
        sleeps.append(s)
        gone["v"] = len(sleeps) >= 3  # the pod vanishes while we wait

    commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=REPO, capture_output=True, text=True).stdout.strip()
    out: list = []
    rc = v41_pod.main(["up", "v41-workers", "--scripts-commit", commit, "--idle-down-min", "0", "--wait"], paths=paths,
                      transport=httpx.MockTransport(handler), api_key=FAKE_KEY, echo=out.append, sleep=sleep)
    assert rc == 1 and len(sleeps) <= 4 and any("ABORTED" in m and "gone" in m for m in out)
    assert FAKE_KEY not in json.dumps(out)


def test_up_wait_stops_when_state_is_removed(tmp_path):
    transport, paths, *_ = started(tmp_path)
    calls = {"n": 0}

    def sleep(s):
        calls["n"] += 1
        paths.state.unlink(missing_ok=True)  # someone ran `down`

    s = v4_pod.load_state(paths.state)
    out: list = []
    rc = v4_pod.cmd_wait(paths, out.append, httpx.MockTransport(lambda r: httpx.Response(502)), sleep,
                         abort=lambda: None if paths.state.exists() else "the pod was taken down (no pod recorded)")
    assert rc == 1 and calls["n"] == 1 and "ABORTED" in out[0] and s


def test_watcher_enforces_auto_down_deadline(tmp_path):
    transport, paths, log, state, _, _ = started(tmp_path, auto_down=30)
    c, base = Clock(), datetime.now(timezone.utc)
    n = {"i": 0}

    def sleep(s):
        n["i"] += 1
        state["body"] = metrics(success=n["i"])  # busy: only the deadline can stop it
        c.sleep(s)

    out: list = []
    v41_pod.watch_loop(v4_pod.RunPod(FAKE_KEY, transport), paths, "pod1", 15.0, out.append, transport=transport, clock=c,
                       sleep=sleep, now=lambda: base + timedelta(seconds=c.t - 1000.0), max_polls=100)
    assert not paths.state.exists() and any("deadline passed" in m for m in out)


def test_watcher_exits_when_state_removed_or_replaced(tmp_path):
    transport, paths, log, state, _, _ = started(tmp_path)
    s = v4_pod.load_state(paths.state)
    s["pod_id"] = "other"
    v4_pod.save_state(paths.state, s)
    rc, out = run_watch(transport, paths, Clock())
    assert rc == 0 and "no longer recorded" in out[0] and not any(m == "DELETE" for m, _ in log)
    paths.state.unlink()
    assert run_watch(transport, paths, Clock())[0] == 0


def test_down_kills_watcher_only_if_it_is_ours(tmp_path):
    transport, paths, log, state, _, _ = started(tmp_path)
    killed = []
    client = v4_pod.RunPod(FAKE_KEY, transport)
    out: list = []
    # pid reused by an unrelated process: left alone
    s = v4_pod.load_state(paths.state)
    assert not v41_pod.kill_watcher(s, killer=lambda p, sig: killed.append(p), cmdline=lambda p: "/usr/bin/vim")
    assert v41_pod.cmd_down(client, paths, out.append, killer=lambda p, sig: killed.append((p, sig)),
                            cmdline=lambda p: "python scripts/v41_pod.py watch --pod-id pod1") == 0
    assert killed == [(777, signal.SIGTERM)] and not paths.state.exists()
    assert any("stopped idle watcher" in m for m in out)


def test_watcher_does_not_kill_itself(tmp_path):
    assert not v41_pod.kill_watcher({"watcher_pid": os.getpid()}, killer=lambda *a: pytest.fail("self-kill"),
                                    cmdline=lambda p: "v41_pod.py watch")
