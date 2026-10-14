"""S19: observable pod auto-shutdown. Fakes only: no network, no RunPod, no key."""
from __future__ import annotations

import importlib.util
import json
import re
import subprocess
import threading
import urllib.request
from datetime import datetime, timedelta, timezone
from http.server import ThreadingHTTPServer
from pathlib import Path

import httpx
import pytest

from scripts import v41_pod, v4_pod
from scripts import v41_pod_watchdog as wd

REPO = Path(__file__).resolve().parents[3]
FAKE_KEY = "rpa_FAKEKEY_do_not_print_123456"
POD_KEY = "rpa_PODSCOPED_do_not_print_987"
SECRET = "sk-vllm-secret-abc123"


class Clock:
    def __init__(self):
        self.t = 1000.0

    def __call__(self):
        return self.t

    def sleep(self, s):
        self.t += s


V030 = ('vllm:num_requests_running{engine="0"} 0.0\nvllm:num_requests_waiting{engine="0"} 0.0\n'
        'vllm:request_success_total{engine="0",finished_reason="stop"} 5.0\nvllm:prompt_tokens_total{engine="0"} 90.0\n')


# --- metric names -----------------------------------------------------------------------------------------------


def test_underscore_prefix_and_old_names_are_accepted():
    assert wd.parse_activity("vllm_request_success_total 4\nvllm_num_requests_running 1\n") == {
        "vllm:request_success_total": 4.0, "vllm:num_requests_running": 1.0}
    old = wd.parse_activity("vllm:request_success 3\nvllm:num_requests_swapped 2\nvllm:e2e_request_latency_seconds_count 9\n")
    assert old == {"vllm:request_success": 3.0, "vllm:num_requests_swapped": 2.0,
                   "vllm:e2e_request_latency_seconds_count": 9.0}
    assert wd.is_busy(old)  # swapped requests are in flight
    assert wd.missing_core(wd.parse_activity("vllm:request_success 1\nvllm:num_requests_running 0\n"), 8100) == [
        "vllm:num_requests_waiting"]
    assert wd.missing_core(wd.parse_activity(V030), 8100) == []
    assert wd.missing_core({}, 8102) == list(wd.CORE_D1)


def test_reachable_metrics_without_known_names_is_an_error_not_idle():
    c, calls, events = Clock(), [], []
    unreadable = wd.Scrape(200, {}, list(wd.CORE_VLLM), "metrics reachable but no known metric names")
    rc = wd.watch({"RUNPOD_POD_ID": "p", "RUNPOD_API_KEY": "k"}, idle_s=900, poll_s=60, clock=c, sleep=c.sleep,
                  request=lambda *a: calls.append(a) or 204, log=lambda _m: None, events=events.append, max_polls=60,
                  scrape=lambda: {"8100": unreadable})
    assert rc == 1 and calls == []  # an hour of unreadable metrics never terminates
    polls = [e for e in events if e["event"] == "poll"]
    assert polls and all("no known names on: 8100" in e["error"] and e["idle_s"] == 0 for e in polls)


# --- watchdog log lines -----------------------------------------------------------------------------------------


def test_poll_lines_carry_status_values_missing_idle_and_decision():
    c, events = Clock(), []
    scrapes = {"8100": wd.Scrape(200, wd.parse_activity(V030), []),
               "8101": wd.Scrape(None, None, list(wd.CORE_VLLM), "ConnectionRefusedError"),
               "8102": wd.Scrape(404, None, list(wd.CORE_D1), "HTTP 404")}
    rc = wd.watch({"RUNPOD_POD_ID": "p", "RUNPOD_API_KEY": POD_KEY}, idle_s=120, poll_s=60, clock=c, sleep=c.sleep,
                  request=lambda *a: (204, ""), log=lambda _m: None, events=events.append, scrape=lambda: scrapes)
    assert rc == 0
    polls = [e for e in events if e["event"] == "poll"]
    assert [p["decision"] for p in polls] == ["wait", "wait", "terminate"]
    first = polls[0]
    assert first["servers"]["8100"] == {"status": 200, "missing": [], "values": wd.parse_activity(V030)}
    assert first["servers"]["8101"]["error"] == "ConnectionRefusedError" and first["servers"]["8102"]["status"] == 404
    assert polls[-1]["idle_s"] == 120.0
    assert events[0]["event"] == "start" and events[0]["api_key_present"] is True
    assert POD_KEY not in json.dumps(events)


def test_sink_writes_jsonl_to_file_and_flushes_stdout(tmp_path, capsys):
    log = tmp_path / "sub" / "wd.jsonl"
    sink = wd.Sink(str(log), {"RUNPOD_API_KEY": POD_KEY})
    sink.event({"event": "poll", "idle_s": 3})
    sink.msg(f"calling with Bearer {POD_KEY} and {POD_KEY}")
    lines = [json.loads(x) for x in log.read_text().splitlines()]
    assert lines[0]["event"] == "poll" and "ts" in lines[0] and POD_KEY not in log.read_text()
    assert "[redacted]" in lines[1]["msg"]
    assert capsys.readouterr().out.count("\n") == 2
    wd.Sink("/proc/nonexistent/x.jsonl", echo=False).event({"event": "x"})  # an unwritable path never raises


# --- self-termination routes ------------------------------------------------------------------------------------


def run_terminate(responses, *, cli=False, run_rc=1):
    seen, events, logs = [], [], []

    def request(method, url, headers, body=None):
        seen.append((method, url, body))
        assert headers["Authorization"] == f"Bearer {POD_KEY}"
        return responses.pop(0)

    ran = []
    out = wd.terminate_self({"RUNPOD_POD_ID": "podZ", "RUNPOD_API_KEY": POD_KEY}, request, logs.append, events=events.append,
                            run=lambda argv, env: ran.append(argv) or (run_rc, "denied"),
                            which=lambda name: "/usr/bin/runpodctl" if cli else None)
    return out, seen, events, logs, ran


def test_routes_are_tried_in_documented_order_and_each_result_is_logged():
    refused = ([(403, '{"error":"forbidden %s"}' % POD_KEY)] * 2 + [(200, '{"errors":[{"message":"not allowed"}]}')]
               + [(403, '{"error":"forbidden %s"}' % POD_KEY)] + [(200, '{"errors":[{"message":"not allowed"}]}')])
    out, seen, events, logs, ran = run_terminate(refused, cli=True)
    assert out == "failed"
    assert [(m, u.split("/pods/")[0] if "/pods/" in u else u) for m, u, _ in seen] == [
        ("DELETE", "https://rest.runpod.io/v1"), ("DELETE", "https://api.runpod.io/v2"),
        ("POST", "https://api.runpod.io/graphql"), ("POST", "https://rest.runpod.io/v1"),
        ("POST", "https://api.runpod.io/graphql")]
    assert 'podTerminate(input: {podId: "podZ"})' in seen[2][2] and "podStop" in seen[4][2]
    assert ran == [["runpodctl", "pod", "delete", "podZ"], ["runpodctl", "remove", "pod", "podZ"],
                   ["runpodctl", "pod", "stop", "podZ"], ["runpodctl", "stop", "pod", "podZ"]]
    routes = [e["route"] for e in events]
    assert routes == ["rest-v1-delete", "rest-v2-delete", "graphql-podTerminate", "runpodctl-pod-delete",
                      "runpodctl-remove-pod", "rest-v1-stop", "graphql-podStop", "runpodctl-pod-stop",
                      "runpodctl-stop-pod"]
    assert [e["outcome"] for e in events] == ["refused"] * 9
    assert events[0]["status"] == 403 and events[0]["path"] == "rest.runpod.io/v1/pods/<id>"
    blob = json.dumps(events) + "".join(logs)
    assert POD_KEY not in blob and "[redacted]" in events[0]["body"] and "podZ" not in blob


def test_runpodctl_is_skipped_when_absent_and_stop_counts_as_stopped():
    out, _, events, _, ran = run_terminate([(403, "")] * 3 + [(204, "")], cli=False)
    assert out == "stopped" and ran == [] and [e["route"] for e in events] == [
        "rest-v1-delete", "rest-v2-delete", "graphql-podTerminate", "runpodctl-pod-delete", "runpodctl-remove-pod",
        "rest-v1-stop"]
    assert [e["outcome"] for e in events[3:5]] == ["skipped", "skipped"]
    out, _, events, _, _ = run_terminate([(401, "")] * 5, cli=False)
    assert out == "failed" and [e["outcome"] for e in events[-2:]] == ["skipped", "skipped"]


def test_graphql_success_and_runpodctl_success():
    out, _, events, _, _ = run_terminate([(403, "")] * 2 + [(200, '{"data":{"podTerminate":null}}')])
    assert out == "terminated" and events[-1]["route"] == "graphql-podTerminate" and events[-1]["outcome"] == "ok"
    out, _, _, _, ran = run_terminate([(403, "")] * 3, cli=True, run_rc=0)
    assert out == "terminated" and ran == [["runpodctl", "pod", "delete", "podZ"]]
    out, _, events, _, ran = run_terminate([(403, "")] * 3, cli=True, run_rc=0)
    assert out == "terminated" and "rest-v1-stop" not in [e["route"] for e in events]


def test_missing_credentials_are_logged_with_which_one_is_absent():
    events = []
    assert wd.terminate_self({"RUNPOD_POD_ID": "p"}, lambda *a: 204, lambda _m: None, events=events.append) == "no-credentials"
    assert events == [{"event": "selfterm", "route": "-", "outcome": "no-credentials", "have_pod_id": True, "have_api_key": False}]


def test_network_error_is_logged_and_next_route_runs():
    calls = []

    def request(method, url, headers, body=None):
        calls.append(method)
        if len(calls) == 1:
            raise OSError("down")
        return 204

    events = []
    assert wd.terminate_self({"RUNPOD_POD_ID": "p", "RUNPOD_API_KEY": "k"}, request, lambda _m: None, events=events.append) == "terminated"
    assert [e["outcome"] for e in events] == ["error", "ok"] and events[0]["body"] == "OSError"


def test_every_terminate_route_precedes_every_stop_route():
    names = [route[0] for route in wd.ROUTES]
    idx = {n: i for i, n in enumerate(names)}
    stops = [n for n in names if n not in wd.TERMINATE_ROUTES]
    assert stops and set(wd.TERMINATE_ROUTES) <= set(names)
    assert max(idx[n] for n in wd.TERMINATE_ROUTES) < min(idx[n] for n in stops)


# --- d1 serves the log ------------------------------------------------------------------------------------------


def load_d1():
    spec = importlib.util.spec_from_file_location("v41_d1_server_s19", REPO / "scripts" / "v41_d1_server.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_watchdog_tail_parses_bounds_and_survives_a_missing_file(tmp_path):
    d1 = load_d1()
    assert d1.watchdog_tail(str(tmp_path / "none"), 5) == {"path": str(tmp_path / "none"), "exists": False, "lines": []}
    f = tmp_path / "wd.jsonl"
    f.write_text("\n".join(json.dumps({"i": i}) for i in range(10)) + "\nnot json at all\n")
    got = d1.watchdog_tail(str(f), 3)
    assert got["lines"] == [{"i": 8}, {"i": 9}, {"raw": "not json at all"}]
    assert len(d1.watchdog_tail(str(f), 0)["lines"]) == 1  # n is clamped to at least one line
    assert len(d1.watchdog_tail(str(f), 10**9)["lines"]) == 11


def test_d1_route_needs_bearer_auth_and_honours_n(tmp_path, monkeypatch):
    d1 = load_d1()
    f = tmp_path / "wd.jsonl"
    f.write_text("".join(json.dumps({"i": i}) + "\n" for i in range(30)))
    monkeypatch.setenv("V41_WATCHDOG_LOG", str(f))
    srv = ThreadingHTTPServer(("127.0.0.1", 0), d1.make_handler(None, {}, threading.Lock(), "tok"))
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    try:
        base = f"http://127.0.0.1:{srv.server_address[1]}"

        def get(path, token=None):
            req = urllib.request.Request(base + path, headers={"Authorization": f"Bearer {token}"} if token else {})
            try:
                with urllib.request.urlopen(req, timeout=10) as r:
                    return r.status, json.loads(r.read())
            except urllib.error.HTTPError as exc:
                return exc.code, None

        assert get("/v41/watchdog-log?n=5") == (401, None)
        assert get("/v41/watchdog-log?n=5", "wrong") == (401, None)
        status, body = get("/v41/watchdog-log?n=5", "tok")
        assert status == 200 and [x["i"] for x in body["lines"]] == [25, 26, 27, 28, 29]
        assert len(get("/v41/watchdog-log", "tok")[1]["lines"]) == 30  # default n=200
        assert d1.ACTIVITY == {"requests": 0, "in_flight": 0}  # reading the log is not request activity
    finally:
        srv.shutdown()


# --- launcher ---------------------------------------------------------------------------------------------------


def test_launcher_logs_watchdog_start_and_failure_and_is_unbuffered():
    path = REPO / "scripts" / "v41_pod_serve.sh"
    text = path.read_text()
    assert subprocess.run(["bash", "-n", str(path)]).returncode == 0
    assert "export PYTHONUNBUFFERED=1" in text and "V41_WATCHDOG_LOG" in text
    assert "watchdog started pid" in text and "watchdog FAILED to start" in text and "watchdog file missing" in text
    assert text.index("ready: d1-3B") < text.index('python3 "$WATCHDOG" &') < text.index("watchdog started pid")
    assert "RUNPOD_API_KEY" not in text


def test_launcher_wlog_writes_valid_json_lines(tmp_path):
    text = (REPO / "scripts" / "v41_pod_serve.sh").read_text()
    fn = re.search(r"^wlog\(\) \{.*?^\}\n", text, re.S | re.M).group(0)
    log = tmp_path / "d" / "wd.jsonl"
    out = subprocess.run(["bash", "-c", fn + 'wlog "watchdog started pid 7"; wlog "second"'], capture_output=True, text=True,
                         env={"V41_WATCHDOG_LOG": str(log), "PATH": "/usr/bin:/bin"})
    assert out.returncode == 0
    rows = [json.loads(x) for x in log.read_text().splitlines()]
    assert [r["msg"] for r in rows] == ["watchdog started pid 7", "second"] and rows[0]["event"] == "launcher"
    assert out.stdout.count("\n") == 2


# --- v41_pod.py diag --------------------------------------------------------------------------------------------


def make_state(tmp_path):
    paths = v4_pod.Paths(state=tmp_path / "ledger" / "st.json", costs=tmp_path / "ledger" / "c.jsonl")
    v4_pod.save_state(paths.state, {"pod_id": "pod1", "api_secret": SECRET, "image": "vllm/vllm-openai:v0.30.0",
                                    "base_url": "https://pod1-8100.proxy.runpod.net/v1",
                                    "created_utc": "2026-10-13T00:00:00+00:00"})
    return paths


def diag_transport(seen, *, metrics_status=200, metrics_body=V030, log_status=200, leak=False):
    log_lines = [{"event": "start", "idle_limit_s": 900}, {"event": "poll", "idle_s": 60, "decision": "wait"},
                 {"event": "selfterm", "route": "rest-v1-delete", "status": 403, "outcome": "refused",
                  "body": f"token {SECRET}" if leak else "forbidden"}]

    def handler(req: httpx.Request) -> httpx.Response:
        seen.append((str(req.url), req.headers.get("authorization")))
        path = req.url.path
        if path == "/v1/models":
            return httpx.Response(200 if "8102" not in req.url.host else 404, json={})
        if path == "/metrics":
            return httpx.Response(metrics_status, text=metrics_body if metrics_status == 200 else "no")
        if path == "/v41/watchdog-log":
            return httpx.Response(log_status, json={"exists": True, "lines": log_lines[: int(req.url.params["n"])]})
        return httpx.Response(404)

    return httpx.MockTransport(handler)


def test_diag_reports_statuses_names_values_and_log_tail_without_the_key(tmp_path, monkeypatch):
    paths = make_state(tmp_path)
    monkeypatch.setattr(v4_pod, "runpod_key", lambda: pytest.fail("diag must not read the RunPod key"))
    seen, out = [], []
    rc = v41_pod.main(["diag", "--tail", "2"], paths=paths, transport=diag_transport(seen, leak=True), echo=out.append)
    text = "\n".join(out)
    assert rc == 0 and SECRET not in text and FAKE_KEY not in text
    assert ":8100 /v1/models -> HTTP 200" in text and ":8102 /v1/models -> HTTP 404" in text
    assert ":8101 /metrics -> HTTP 200" in text
    assert "vllm:num_requests_running=0" in text and "vllm:request_success_total=5" in text
    assert "missing core names: (none)" in text  # 8100/8101 have everything
    assert "missing core names: v41_d1_requests_total, v41_d1_in_flight" in text  # 8102 fake serves vLLM names only
    assert "watchdog log tail (:8102 /v41/watchdog-log) -> HTTP 200" in text
    assert text.count('"event": "start"') == 1 and '"event": "selfterm"' not in text  # tail 2 of 3 = poll + selfterm
    seen_urls = [u for u, _ in seen]
    assert any(u.endswith("pod1-8102.proxy.runpod.net/v41/watchdog-log?n=2") for u in seen_urls)
    assert {a for _, a in seen} == {f"Bearer {SECRET}"}


def test_diag_scrubs_a_secret_that_leaks_into_the_log(tmp_path):
    paths = make_state(tmp_path)
    out = []
    v41_pod.main(["diag", "--tail", "3"], paths=paths, transport=diag_transport([], leak=True), echo=out.append)
    text = "\n".join(out)
    assert SECRET not in text and "token [redacted]" in text
    assert "self-termination attempts: rest-v1-delete=403/refused" in text


def test_diag_flags_a_proxy_that_does_not_forward_metrics_and_missing_names(tmp_path):
    paths = make_state(tmp_path)
    out = []
    v41_pod.main(["diag"], paths=paths, transport=diag_transport([], metrics_status=404), echo=out.append)
    text = "\n".join(out)
    assert "PROBLEM: :8100 /metrics HTTP 404 (proxy may not forward it)" in text
    out = []
    v41_pod.main(["diag"], paths=paths, transport=diag_transport([], metrics_body="process_cpu_seconds_total 3\n"), echo=out.append)
    text = "\n".join(out)
    assert "PROBLEM: :8100 /metrics reachable but none of the watched names are present" in text
    assert "process_cpu_seconds_total" in text  # the real exported names are shown so the alias list can be fixed
    out = []
    assert v41_pod.main(["diag"], paths=v4_pod.Paths(state=tmp_path / "none.json", costs=tmp_path / "c"), echo=out.append) == 1
    assert out == ["diag: no pod recorded"]


def test_diag_reports_a_pod_with_no_watchdog_log(tmp_path):
    paths = make_state(tmp_path)

    def handler(req):
        if req.url.path == "/v41/watchdog-log":
            return httpx.Response(200, json={"exists": False, "lines": []})
        return httpx.Response(200, text=V030)

    out = []
    v41_pod.main(["diag"], paths=paths, transport=httpx.MockTransport(handler), echo=out.append)
    assert any("watchdog never started" in m for m in out)


# --- Mac watcher: the ledger keeps a working pod alive -----------------------------------------------------------


def write_usage(paths, rows):
    usage = paths.state.parent / "v4.1-usage.jsonl"
    usage.parent.mkdir(parents=True, exist_ok=True)
    usage.write_text("".join(json.dumps(r) + "\n" for r in rows))
    return usage


def watcher_env(tmp_path, *, metrics="down", models=True):
    """A started pod whose proxy serves ``metrics`` ('down' = 404, 'unknown' = no watched names, 'live')."""
    from tests.v4_1.pod.test_pod_autodown import started

    transport, paths, log, state, _, _ = started(tmp_path)

    def handler(req: httpx.Request) -> httpx.Response:
        log.append((req.method, str(req.url)))
        if req.url.host == "rest.runpod.io":
            return httpx.Response(204) if req.method == "DELETE" else httpx.Response(200, json={"desiredStatus": "RUNNING"})
        if req.url.path == "/metrics":
            if metrics == "down":
                return httpx.Response(404)
            return httpx.Response(200, text="process_cpu_seconds_total 1\n" if metrics == "unknown" else V030)
        if req.url.path == "/v1/models":
            return httpx.Response(200 if models else 502, json={})
        return httpx.Response(404)

    return httpx.MockTransport(handler), paths, log


def pod_row(when, backend="vllm"):
    return {"utc": when.strftime("%Y-%m-%dT%H:%M:%S.%fZ"), "backend": backend, "model": "m", "role": "worker"}


def run_watch(transport, paths, c, base, **kw):
    out = []
    v41_pod.watch_loop(v4_pod.RunPod(FAKE_KEY, transport), paths, "pod1", 15.0, out.append, transport=transport, clock=c,
                       sleep=c.sleep, now=lambda: base + timedelta(seconds=c.t - 1000.0), **kw)
    return out


def test_metrics_not_forwarded_but_ledger_active_never_kills_the_pod(tmp_path):
    transport, paths, log = watcher_env(tmp_path, metrics="down")
    c, base = Clock(), datetime.now(timezone.utc)
    n = {"i": 0}

    def sleep(s):  # a pod-backend row lands every 5 minutes for 2 hours
        n["i"] += 1
        if n["i"] % 5 == 0:
            rows = [pod_row(base + timedelta(seconds=c.t - 1000.0))]
            write_usage(paths, rows)
        c.sleep(s)

    out = []
    v41_pod.watch_loop(v4_pod.RunPod(FAKE_KEY, transport), paths, "pod1", 15.0, out.append, transport=transport, clock=c,
                       sleep=sleep, now=lambda: base + timedelta(seconds=c.t - 1000.0), max_polls=120)
    assert paths.state.exists() and not any(m == "DELETE" for m, _ in log)
    assert sum("usage ledger alone" in m for m in out) == 1  # said once, not every poll


def test_ledger_only_mode_goes_down_after_the_idle_window_without_rows(tmp_path):
    transport, paths, log = watcher_env(tmp_path, metrics="down")
    c, base = Clock(), datetime.now(timezone.utc)
    write_usage(paths, [pod_row(base)])  # one row at t=0, then silence
    out = run_watch(transport, paths, c, base, max_polls=14)  # 14 min after the row: still inside the 15 min window
    assert paths.state.exists()
    out = run_watch(transport, paths, c, base)
    assert not paths.state.exists() and any("usage ledger only (metrics unreachable)" in m for m in out)


def test_rows_from_non_pod_backends_do_not_count(tmp_path):
    transport, paths, log = watcher_env(tmp_path, metrics="down")
    c, base = Clock(), datetime.now(timezone.utc)
    n = {"i": 0}

    def sleep(s):
        n["i"] += 1
        write_usage(paths, [pod_row(base + timedelta(seconds=c.t - 1000.0), backend="nous_fw")])
        c.sleep(s)

    out = []
    v41_pod.watch_loop(v4_pod.RunPod(FAKE_KEY, transport), paths, "pod1", 15.0, out.append, transport=transport, clock=c,
                       sleep=sleep, now=lambda: base + timedelta(seconds=c.t - 1000.0))
    assert not paths.state.exists()  # Nous traffic is not the pod working


def test_metrics_idle_but_ledger_active_keeps_the_pod(tmp_path):
    transport, paths, log = watcher_env(tmp_path, metrics="live")  # counters never change: idle by metrics
    c, base = Clock(), datetime.now(timezone.utc)
    n = {"i": 0}

    def sleep(s):
        n["i"] += 1
        write_usage(paths, [pod_row(base + timedelta(seconds=c.t - 1000.0), backend="d1")])
        c.sleep(s)

    out = []
    v41_pod.watch_loop(v4_pod.RunPod(FAKE_KEY, transport), paths, "pod1", 15.0, out.append, transport=transport, clock=c,
                       sleep=sleep, now=lambda: base + timedelta(seconds=c.t - 1000.0), max_polls=60)
    assert paths.state.exists()


def test_unknown_metric_names_fall_back_to_the_ledger(tmp_path):
    transport, paths, log = watcher_env(tmp_path, metrics="unknown")
    c, base = Clock(), datetime.now(timezone.utc)
    out = run_watch(transport, paths, c, base)  # no rows at all: idle by the ledger from first readiness
    assert not paths.state.exists()
    assert any("usage ledger alone" in m for m in out) and any("usage ledger only" in m for m in out)


def test_deadline_rule_still_applies_in_ledger_mode(tmp_path):
    from tests.v4_1.pod.test_pod_autodown import started

    transport, paths, log, state, _, _ = started(tmp_path, auto_down=30)
    c, base = Clock(), datetime.now(timezone.utc)
    n = {"i": 0}

    def sleep(s):
        n["i"] += 1
        write_usage(paths, [pod_row(base + timedelta(seconds=c.t - 1000.0))])
        state["body"] = "process_cpu_seconds_total 1\n"
        c.sleep(s)

    out = []
    v41_pod.watch_loop(v4_pod.RunPod(FAKE_KEY, transport), paths, "pod1", 15.0, out.append, transport=transport, clock=c,
                       sleep=sleep, now=lambda: base + timedelta(seconds=c.t - 1000.0), max_polls=100)
    assert not paths.state.exists() and any("deadline passed" in m for m in out)


# --- pinning ----------------------------------------------------------------------------------------------------


def test_scripts_pin_hashes_the_new_launcher_files_as_committed():
    head = subprocess.run(["git", "rev-parse", "HEAD"], cwd=REPO, capture_output=True, text=True).stdout.strip()
    pin = v4_pod.scripts_pin(head, REPO)
    assert set(pin.hashes) == set(v4_pod.WORKERS_FETCH) and all(re.fullmatch(r"[0-9a-f]{64}", h) for h in pin.hashes.values())
    frag = v4_pod.fetch_fragment(pin)
    assert "v41_pod_watchdog.py" in frag and pin.hashes["v41_pod_watchdog.py"] in frag
