#!/usr/bin/env python3
"""RunPod lifecycle for the v4 pilot: one on-demand vLLM pod at a time (goal section 8).

Commands: ``up <model>`` | ``status`` | ``wait`` | ``down`` | ``cost``. Port of colony_v2_pod.py.
Every pod start and stop is appended to ``ledger/v4-pod-costs.jsonl`` in the schema
``antelligence.experiments.v4_budget.pod_spend`` reads. ``up`` refuses when that spend plus one hour at the
quoted rate would pass the $2.75 hard stop. The RunPod key is read from ~/.hermes/.env in ``runpod_key`` only.
A fresh API secret is generated per pod and kept only in the gitignored 0600 state file; it is never printed.
Pure helpers (payload, command line, cost rows, refusal, state) are separate from HTTP so tests need no network.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import secrets
import shlex
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, List, Mapping, Optional

import httpx

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from antelligence.experiments import v4_budget as budget  # noqa: E402

REST_URL = "https://rest.runpod.io/v1"
GRAPHQL_URL = "https://api.runpod.io/graphql"
# Pinned and verified 2026-10-10; never ``latest``.
VLLM_IMAGE = "vllm/vllm-openai:v0.30.0"
GPU_TYPES = ("NVIDIA L40S", "NVIDIA A40")  # first choice, then fallback
CONTAINER_DISK_GB = 80
MAX_WALL_HOURS = 2.0
HARD_STOP_USD = budget.LIMITS["runpod"]
ASSUMED_RATE_PER_HR = 1.20  # used for the pre-flight check only when the API gives no quote
# Served name is the HF id, so the response-model identity check pins the weights. Revisions are HF commits.
MODELS: Dict[str, Dict[str, Optional[str]]] = {
    "qwen3.5-0.8b": {"hf": "Qwen/Qwen3.5-0.8B", "revision": "2fc06364715b967f1860aea9cf38778875588b17"},
    "qwen3.5-4b": {"hf": "Qwen/Qwen3.5-4B", "revision": "851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a"},
    "qwen3.5-9b": {"hf": "Qwen/Qwen3.5-9B", "revision": "c202236235762e1c871ad0ccb60c8ee5ba337b9a"},
    # Hybrid (Mamba) layers: one cache block per running sequence, so 64 fit on an L40S where 256 do not.
    "qwen3.5-27b": {"hf": "Qwen/Qwen3.5-27B-FP8", "revision": "97f5941bf617e31c5e237364a8602ce3f03a551a",
                    "max_num_seqs": "64"},
    "qwen3.8-27b": {"hf": "Qwen/Qwen3.8-27B-FP8", "revision": "017b9c7af6b5689d5dd426a76e0bc077eb5ca20a",
                    "max_num_seqs": "64"},
}


# v4.1 worker pod (goal 4.5): Qwen3.5-9B + 4B on vLLM and d1-3B, launched by scripts/v41_pod_serve.sh. It reuses the
# same budget guard, cost ledger and one-pod rule; only the start command and the ports differ.
WORKERS_KEY = "v41-workers"
WORKERS_HF, WORKERS_REVISION = "RedHatAI/Qwen3.5-9B-FP8-dynamic", "790f0576d2d77dd5322aa0603a470bd9e3a3d1f6"
WORKERS_PORTS = ["8100/http", "8101/http", "8102/http", "8001/http"]  # 9B, 4B, d1 shim, read-only log
# Launcher files the pod downloads at start-up: pod-side name -> repo path. Fetched from GitHub at a pinned commit and
# sha256-verified (the pod exposes only HTTP ports and the vLLM image has no sshd, so nothing can be pushed to it).
WORKERS_FETCH = {"serve.sh": "scripts/v41_pod_serve.sh", "v41_d1_server.py": "scripts/v41_d1_server.py",
                 "v41_pod_watchdog.py": "scripts/v41_pod_watchdog.py"}
RAW_BASE = "https://raw.githubusercontent.com/solmonger/antelligence-app"
WORKERS_DEST = "/opt/v41"
COMMIT_RE = re.compile(r"[0-9a-f]{40}")


class ScriptsCommitError(ValueError):
    """The --scripts-commit value is malformed, unknown locally, or lacks a launcher file."""


@dataclass(frozen=True)
class ScriptsPin:
    """A git commit and the sha256 of each launcher file at that commit (name -> hex digest)."""
    commit: str
    hashes: Mapping[str, str]


@dataclass
class Paths:
    state: Path = ROOT / "ledger" / "v4-pod-state.json"
    costs: Path = ROOT / "ledger" / "v4-pod-costs.jsonl"


# --- pure helpers -----------------------------------------------------------------


def served_name(model: str) -> str:
    if model == WORKERS_KEY:  # the 9B's name; the launcher waits for all three servers before it finishes
        return f"{WORKERS_HF}@{WORKERS_REVISION}"
    return str(MODELS[model]["hf"])


def vllm_args(model: str) -> List[str]:
    """``vllm serve`` arguments. The API key is NOT here: vLLM reads it from env VLLM_API_KEY."""
    spec = MODELS[model]
    parts = [spec["hf"], "--served-model-name", served_name(model), "--max-model-len", "8192",
             "--gpu-memory-utilization", "0.92", "--seed", "0", "--revision", spec["revision"]]
    if spec.get("max_num_seqs"):
        parts += ["--max-num-seqs", spec["max_num_seqs"]]
    # Qwen templates think by default and would spend the answer budget on it; answer directly.
    return parts + ["--default-chat-template-kwargs", '{"enable_thinking":false}']


def start_script(model: str, log_token: str) -> str:
    """Container command: vLLM on :8000; its log (key redacted) served read-only on :8001 under a secret path."""
    serve = " ".join(shlex.quote(a) for a in ["vllm", "serve", *vllm_args(model)])
    log = f"/srv/{log_token}/vllm.log"
    # An empty index.html stops http.server from listing /srv (which would reveal the token).
    return (f"mkdir -p /srv/{log_token} && : > /srv/index.html && "
            f"(python3 -m http.server 8001 --directory /srv >/dev/null 2>&1 &) && "
            f"{{ {serve} 2>&1 | sed -u \"s/$VLLM_API_KEY/[redacted]/g\" > {log}; }}; "
            f"echo \"vllm exited\" >> {log}; sleep infinity")


def scripts_pin(commit: str, root: Path = ROOT) -> ScriptsPin:
    """Validate ``commit`` (40 lowercase hex), require it to exist locally, and hash each launcher file as stored in
    that commit (``git show <sha>:<path>``), so the pod is pinned to exactly what the lead has reviewed."""
    if not COMMIT_RE.fullmatch(commit or ""):
        raise ScriptsCommitError(f"--scripts-commit must be a full 40-char lowercase hex SHA, got {commit!r}")
    probe = subprocess.run(["git", "cat-file", "-e", f"{commit}^{{commit}}"], cwd=root, capture_output=True)
    if probe.returncode != 0:
        raise ScriptsCommitError(f"commit {commit} not found in the local repo (git fetch it first)")
    hashes: Dict[str, str] = {}
    for name, path in WORKERS_FETCH.items():
        shown = subprocess.run(["git", "show", f"{commit}:{path}"], cwd=root, capture_output=True)
        if shown.returncode != 0:
            raise ScriptsCommitError(f"{path} does not exist at commit {commit}")
        hashes[name] = hashlib.sha256(shown.stdout).hexdigest()
    return ScriptsPin(commit, hashes)


_FETCH_FUNCS = r"""fetch_one() { if command -v curl >/dev/null 2>&1; then curl -fsSL -o "$2" "$1"; else python3 -c 'import sys,urllib.request;open(sys.argv[2],"wb").write(urllib.request.urlopen(sys.argv[1],timeout=60).read())' "$1" "$2"; fi; }
sha_of() { python3 -c 'import hashlib,sys;print(hashlib.sha256(open(sys.argv[1],"rb").read()).hexdigest())' "$1"; }
fetch_verified() { n=$1; u=$2; want=$3; d="$DEST/$n"; ok=; i=0
  while [ $i -lt 5 ]; do i=$((i+1)); if fetch_one "$u" "$d.part" 2>>"$LOG"; then ok=1; break; fi; echo "fetch $n attempt $i failed" >> "$LOG"; sleep 3; done
  if [ -z "$ok" ]; then echo "FETCH FAILED: $n from $u; refusing to run" >> "$LOG"; exit 1; fi
  got=$(sha_of "$d.part")
  if [ "$got" != "$want" ]; then rm -f "$d.part"; echo "HASH MISMATCH: $n expected $want got $got; refusing to run" >> "$LOG"; exit 1; fi
  mv "$d.part" "$d"; echo "verified $n sha256=$got" >> "$LOG"; }
"""


def fetch_fragment(pin: ScriptsPin, dest: str = WORKERS_DEST, log: str = "/dev/null", base: str = RAW_BASE) -> str:
    """Shell fragment: download each launcher file from ``<base>/<commit>/<path>`` (curl, else python3 urllib), check its
    sha256 against the value embedded here, and ``exit 1`` with a log line on any failure. Pure; ``base`` is
    overridable so tests can point it at a file:// fixture."""
    if not COMMIT_RE.fullmatch(pin.commit) or any(not re.fullmatch(r"[0-9a-f]{64}", h) for h in pin.hashes.values()):
        raise ScriptsCommitError("malformed commit or hash")
    lines = [f"DEST={shlex.quote(dest)}; LOG={shlex.quote(log)}; mkdir -p \"$DEST\"",
             f"echo \"fetching launcher at commit {pin.commit}\" >> \"$LOG\"", _FETCH_FUNCS.rstrip("\n")]
    for name, path in WORKERS_FETCH.items():
        url = f"{base}/{pin.commit}/{path}"
        lines.append(f"fetch_verified {shlex.quote(name)} {shlex.quote(url)} {pin.hashes[name]}")
    return "\n".join(lines)


# Pod-side self-termination with the pod-scoped RUNPOD_API_KEY / RUNPOD_POD_ID RunPod injects (never our account key):
# DELETE the pod (terminate), else POST /stop. Inline so it works even when the launcher fetch itself failed.
SELF_TERMINATE_PY = (
    "import os,sys,urllib.request as u\n"
    "i=os.environ.get('RUNPOD_POD_ID');k=os.environ.get('RUNPOD_API_KEY')\n"
    "if not i or not k: print('selfterm: no pod credentials'); sys.exit(1)\n"
    "for m,p in (('DELETE',''),('POST','/stop')):\n"
    "  try:\n"
    "    r=u.urlopen(u.Request('https://rest.runpod.io/v1/pods/'+i+p,method=m,headers={'Authorization':'Bearer '+k,"
    "'User-Agent':'v41-pod/1'}),timeout=30);print('selfterm',m,r.status);sys.exit(0)\n"
    "  except Exception as e: print('selfterm',m,'failed',type(e).__name__,getattr(e,'code',''))\n"
    "sys.exit(1)\n")
CRASH_GRACE_S = 120  # after the launcher ends, keep the log port up this long so a failure can be read


def workers_start_script(log_token: str, pin: ScriptsPin) -> str:
    """Container command for the v4.1 worker pod. It serves the log read-only on :8001 first (so fetch progress is
    visible), downloads the launcher from GitHub at the pinned commit and verifies its sha256 (see
    ``fetch_fragment``; a mismatch exits non-zero before anything runs), then runs ``serve.sh``. Log (key redacted)
    is under a secret path, as for ``up``. An EXIT trap terminates the pod (``SELF_TERMINATE_PY``) however the script
    ends (fetch failure, hash mismatch, launcher exit or crash) after ``CRASH_GRACE_S``; if even that fails it
    ``kill 1``s the container. It never idles on ``sleep infinity``."""
    log = f"/srv/{log_token}/workers.log"
    selfterm = shlex.quote(SELF_TERMINATE_PY)
    return (f"mkdir -p /srv/{log_token} && : > /srv/index.html && "
            f"(python3 -m http.server 8001 --directory /srv >/dev/null 2>&1 &) && "
            f"LOG={shlex.quote(log)}; "
            f"on_exit() {{ trap - EXIT; echo \"launcher ended; self-terminating in {CRASH_GRACE_S}s\" >> \"$LOG\"; "
            f"sleep {CRASH_GRACE_S}; python3 -c {selfterm} >> \"$LOG\" 2>&1 || "
            f"{{ echo \"self-terminate failed; killing the container\" >> \"$LOG\"; kill 1; }}; }}; "
            f"trap on_exit EXIT\n"
            f"{fetch_fragment(pin, WORKERS_DEST, log)}\n"
            f"{{ bash {WORKERS_DEST}/serve.sh 2>&1 | sed -u \"s/$VLLM_API_KEY/[redacted]/g\" >> {log}; }}; "
            f"echo \"workers exited\" >> {log}")


def deploy_payload_workers(api_secret: str, gpu: str, log_token: str, pin: Optional[ScriptsPin]) -> Dict[str, Any]:
    """REST body for the v4.1 worker pod. Pure: builds, never sends."""
    if pin is None:
        raise ScriptsCommitError("the worker pod needs a scripts pin (--scripts-commit)")
    return {"name": "v41-workers", "imageName": VLLM_IMAGE, "gpuTypeIds": [gpu], "gpuCount": 1,
            "containerDiskInGb": CONTAINER_DISK_GB, "volumeInGb": 0, "ports": WORKERS_PORTS,
            "env": {"VLLM_API_KEY": api_secret}, "dockerEntrypoint": ["bash", "-c"],
            "dockerStartCmd": [workers_start_script(log_token, pin)]}


def deploy_payload(model: str, api_secret: str, gpu: str, log_token: str,
                   pin: Optional[ScriptsPin] = None) -> Dict[str, Any]:
    """REST body for one on-demand pod. Pure: builds, never sends."""
    if model == WORKERS_KEY:
        return deploy_payload_workers(api_secret, gpu, log_token, pin)
    return {"name": f"v4-pilot-{model}", "imageName": VLLM_IMAGE, "gpuTypeIds": [gpu], "gpuCount": 1,
            "containerDiskInGb": CONTAINER_DISK_GB, "volumeInGb": 0, "ports": ["8000/http", "8001/http"],
            "env": {"VLLM_API_KEY": api_secret}, "dockerEntrypoint": ["bash", "-c"],
            "dockerStartCmd": [start_script(model, log_token)]}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def cost_row(event: str, pod_id: str, rate_per_hr: float, utc: Optional[str] = None, **extra: Any) -> Dict[str, Any]:
    """One cost record in the schema ``v4_budget.pod_spend`` reads."""
    if event not in ("start", "stop"):
        raise ValueError(f"event must be start or stop, not {event!r}")
    return {"event": event, "pod_id": pod_id, "utc": utc or utc_now(), "rate_per_hr": float(rate_per_hr), **extra}


def append_cost(path: Path, row: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a") as handle:
        handle.write(json.dumps(row) + "\n")


def spend_so_far(costs: Path, now: Optional[datetime] = None) -> float:
    return budget.pod_spend(budget.read_jsonl(costs), now)


def refuse_over_budget(spent: float, rate_per_hr: float, limit: float = HARD_STOP_USD) -> Optional[str]:
    """Reason to refuse starting a pod (spend plus one hour at the quoted rate would pass the stop), else None."""
    if spent + rate_per_hr > limit:
        return f"budget: ${spent:.3f} spent + 1h at ${rate_per_hr:.3f}/hr would exceed the ${limit:.2f} hard stop"
    return None


def load_state(path: Path) -> Optional[Dict[str, Any]]:
    return json.loads(path.read_text()) if path.exists() else None


def save_state(path: Path, state: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(fd, "w") as handle:
        handle.write(json.dumps(state, indent=1))
    os.chmod(path, 0o600)


def public_view(state: Mapping[str, Any]) -> Dict[str, Any]:
    """State fields that are safe to print (no api_secret, no log url token)."""
    return {k: state.get(k) for k in ("pod_id", "model", "served", "gpu", "image", "revision", "cost_per_hr",
                                      "created_utc", "base_url")}


def elapsed_line(state: Mapping[str, Any], now: Optional[datetime] = None) -> str:
    start = datetime.fromisoformat(state["created_utc"])
    hours = ((now or datetime.now(timezone.utc)) - start).total_seconds() / 3600
    return f"elapsed {hours * 60:.1f} min, estimated spend ${hours * float(state.get('cost_per_hr') or 0):.3f}"


def serves_model(payload: Mapping[str, Any], expected: str) -> bool:
    return expected in [m.get("id") for m in payload.get("data", []) if isinstance(m, dict)]


# --- HTTP -------------------------------------------------------------------------


def runpod_key() -> str:
    """RUNPOD_API_KEY from ~/.hermes/.env. The only place the key file is read; never printed."""
    for line in Path("~/.hermes/.env").expanduser().read_text().splitlines():
        name, _, value = line.partition("=")
        if name.strip() == "RUNPOD_API_KEY":
            return value.strip().strip("'\"")
    raise SystemExit("RUNPOD_API_KEY not found in ~/.hermes/.env")


class RunPod:
    """Thin RunPod client; the transport is injectable so tests never touch the network."""

    def __init__(self, api_key: str, transport: Optional[httpx.BaseTransport] = None) -> None:
        self._http = httpx.Client(headers={"Authorization": f"Bearer {api_key}"}, timeout=60.0,
                                  follow_redirects=False, trust_env=False, transport=transport)

    def create_pod(self, body: Dict[str, Any]) -> httpx.Response:
        return self._http.post(f"{REST_URL}/pods", json=body)

    def get_pod(self, pod_id: str) -> httpx.Response:
        return self._http.get(f"{REST_URL}/pods/{pod_id}")

    def delete_pod(self, pod_id: str) -> httpx.Response:
        return self._http.delete(f"{REST_URL}/pods/{pod_id}")

    def quote(self, gpu: str) -> Optional[float]:
        """Secure-cloud on-demand $/hr for one GPU type, or None if the API gives no number."""
        query = "query($id: String!) { gpuTypes(input: {id: $id}) { securePrice } }"
        try:
            r = self._http.post(GRAPHQL_URL, json={"query": query, "variables": {"id": gpu}})
            price = r.json()["data"]["gpuTypes"][0]["securePrice"] if r.status_code == 200 else None
            return float(price) if price else None
        except (httpx.HTTPError, KeyError, IndexError, TypeError, ValueError):
            return None


Echo = Callable[[str], None]


def pod_status(client: RunPod, pod_id: str) -> Optional[str]:
    """desiredStatus of a pod, "GONE" on 404, None when the API could not say."""
    r = client.get_pod(pod_id)
    if r.status_code == 404:
        return "GONE"
    return r.json().get("desiredStatus") if r.status_code == 200 else None


class PodGone(RuntimeError):
    """The pod disappeared while we were waiting for it to serve."""


def wait_ready(state: Mapping[str, Any], timeout_s: float, transport: Optional[httpx.BaseTransport] = None,
               sleep: Callable[[float], None] = time.sleep, interval_s: float = 15.0,
               abort: Optional[Callable[[], Optional[str]]] = None) -> bool:
    """Poll the pod's /v1/models until it lists the expected served name. ``abort`` returns a message once waiting is
    pointless (pod gone); it is raised as PodGone."""
    deadline = time.monotonic() + timeout_s
    with httpx.Client(headers={"Authorization": f"Bearer {state['api_secret']}"}, timeout=15.0, trust_env=False,
                      transport=transport) as http:
        while True:
            try:
                r = http.get(f"{state['base_url']}/models")
                if r.status_code == 200 and serves_model(r.json(), state["served"]):
                    return True
            except (httpx.HTTPError, ValueError):
                pass
            if time.monotonic() >= deadline:
                return False
            reason = abort() if abort else None
            if reason:
                raise PodGone(reason)
            sleep(interval_s)


# --- commands ---------------------------------------------------------------------


def cmd_up(model: str, client: RunPod, paths: Paths, echo: Echo = print, wait: bool = False,
           transport: Optional[httpx.BaseTransport] = None, sleep: Callable[[float], None] = time.sleep,
           scripts_commit: Optional[str] = None) -> int:
    if model not in MODELS and model != WORKERS_KEY:
        echo(f"unknown model {model!r}; choose from {sorted([*MODELS, WORKERS_KEY])}")
        return 2
    pin: Optional[ScriptsPin] = None
    if model == WORKERS_KEY:  # validated before any API call or state change
        try:
            pin = scripts_pin(scripts_commit or "")
        except ScriptsCommitError as exc:
            echo(f"refused: {exc}")
            return 2
    state = load_state(paths.state)
    if state:
        status = pod_status(client, state["pod_id"])
        if status != "GONE":
            echo(f"refused: pod {state['pod_id']} ({state['model']}) may still be running "
                 f"(status {status}); run `down` first (one pod at a time)")
            return 1
        echo("stale state file (pod gone); closing its cost record and replacing it")
        append_cost(paths.costs, cost_row("stop", state["pod_id"], state.get("cost_per_hr") or 0.0))
        paths.state.unlink()
    spent = spend_so_far(paths.costs)
    pod: Optional[Dict[str, Any]] = None
    gpu = ""
    api_secret, log_token = secrets.token_urlsafe(32), secrets.token_hex(16)
    for gpu in GPU_TYPES:
        reason = refuse_over_budget(spent, client.quote(gpu) or ASSUMED_RATE_PER_HR)
        if reason:
            echo(f"refused: {reason}")
            return 1
        r = client.create_pod(deploy_payload(model, api_secret, gpu, log_token, pin))
        body = r.json() if r.status_code in (200, 201) else None
        if body and body.get("id"):
            pod = body
            break
        echo(f"{gpu} unavailable: HTTP {r.status_code}")
    if pod is None:
        echo("no GPU available")
        return 1
    rate = float(pod.get("costPerHr") or ASSUMED_RATE_PER_HR)
    pin_fields: Dict[str, Any] = {"scripts_commit": pin.commit, "scripts_sha256": dict(pin.hashes)} if pin else {}
    append_cost(paths.costs, cost_row("start", pod["id"], rate, **pin_fields))
    state = {"pod_id": pod["id"], "api_secret": api_secret, "model": model, "served": served_name(model),
             "hf": WORKERS_HF if model == WORKERS_KEY else MODELS[model]["hf"],
             "revision": WORKERS_REVISION if model == WORKERS_KEY else MODELS[model]["revision"],
             "image": VLLM_IMAGE, "gpu": gpu, "cost_per_hr": rate, "created_utc": utc_now(),
             "base_url": f"https://{pod['id']}-{8100 if model == WORKERS_KEY else 8000}.proxy.runpod.net/v1",
             "log_url": f"https://{pod['id']}-8001.proxy.runpod.net/{log_token}/"
                        f"{'workers' if model == WORKERS_KEY else 'vllm'}.log", **pin_fields}
    save_state(paths.state, state)
    echo(f"pod {pod['id']} on {gpu} at ${rate:.3f}/hr; base_url {state['base_url']}")
    if wait:
        return cmd_wait(paths, echo, transport, sleep)
    return 0


def cmd_status(client: RunPod, paths: Paths, echo: Echo = print) -> int:
    state = load_state(paths.state)
    if not state:
        echo("no pod recorded")
        return 0
    view = public_view(state)
    view["desiredStatus"] = pod_status(client, state["pod_id"])
    echo(json.dumps(view))
    echo(elapsed_line(state))
    return 0


def cmd_wait(paths: Paths, echo: Echo = print, transport: Optional[httpx.BaseTransport] = None,
             sleep: Callable[[float], None] = time.sleep, timeout_s: float = MAX_WALL_HOURS * 3600,
             abort: Optional[Callable[[], Optional[str]]] = None) -> int:
    state = load_state(paths.state)
    if not state:
        echo("no pod recorded")
        return 1
    try:
        ok = wait_ready(state, timeout_s, transport, sleep, abort=abort)
    except PodGone as exc:
        echo(f"ABORTED: stopped waiting, {exc}; the pod will never become ready")
        return 1
    echo(("ready: serving " + state["served"]) if ok else "NOT READY within the wall-clock cap; run `down`")
    echo(elapsed_line(state))
    return 0 if ok else 1


def cmd_down(client: RunPod, paths: Paths, echo: Echo = print) -> int:
    state = load_state(paths.state)
    if not state:
        echo("no pod recorded")
        return 0
    r = client.delete_pod(state["pod_id"])
    if r.status_code not in (200, 204) and pod_status(client, state["pod_id"]) != "GONE":
        echo(f"terminate failed: HTTP {r.status_code}; pod may still be billing")
        return 1
    append_cost(paths.costs, cost_row("stop", state["pod_id"], state.get("cost_per_hr") or 0.0))
    echo(f"terminated {state['pod_id']} | {elapsed_line(state)}")
    paths.state.unlink()
    return 0


def cmd_cost(paths: Paths, echo: Echo = print) -> int:
    spent = spend_so_far(paths.costs)
    echo(f"runpod spend ${spent:.3f} of ${HARD_STOP_USD:.2f} hard stop (${HARD_STOP_USD - spent:.3f} left)")
    return 0


def main(argv: Optional[List[str]] = None, *, paths: Optional[Paths] = None,
         transport: Optional[httpx.BaseTransport] = None, api_key: Optional[str] = None,
         sleep: Callable[[float], None] = time.sleep, echo: Echo = print) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="cmd", required=True)
    up = sub.add_parser("up")
    up.add_argument("model", choices=sorted([*MODELS, WORKERS_KEY]))
    up.add_argument("--wait", action="store_true", help="after creating, block until vLLM serves the model")
    up.add_argument("--scripts-commit", metavar="SHA", help="v41-workers only: 40-hex commit the pod downloads and "
                    "sha256-verifies its launcher from (must exist locally and be pushed to origin)")
    for name in ("status", "wait", "down", "cost"):
        sub.add_parser(name)
    args = parser.parse_args(argv)
    paths = paths or Paths()
    if args.cmd == "cost":
        return cmd_cost(paths, echo)
    if args.cmd == "wait":  # talks to the pod with its own secret, not the RunPod key
        return cmd_wait(paths, echo, transport, sleep)
    client = RunPod(api_key or runpod_key(), transport)
    if args.cmd == "up":
        return cmd_up(args.model, client, paths, echo, args.wait, transport, sleep, args.scripts_commit)
    return {"status": cmd_status, "down": cmd_down}[args.cmd](client, paths, echo)


if __name__ == "__main__":
    sys.exit(main())
