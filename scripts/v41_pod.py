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

HARD_STOP_USD = v41_budget.STOPS["runpod"]  # 24.00
GPU = "NVIDIA L40S"
IMAGE = v4_pod.VLLM_IMAGE  # pinned by v4_pod (vLLM >= 0.17)
MODEL = v4_pod.WORKERS_KEY
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


def cmd_up(client: v4_pod.RunPod, paths: v4_pod.Paths, image: str, auto_down_min: Optional[float], echo: Echo,
           wait: bool = False, transport: Optional[httpx.BaseTransport] = None,
           sleep: Callable[[float], None] = time.sleep, scripts_commit: Optional[str] = None) -> int:
    with v41_bindings(validate_image(image)):
        rc = v4_pod.cmd_up(MODEL, client, paths, echo, False, transport, sleep, scripts_commit)
    state = v4_pod.load_state(paths.state)
    if rc == 0 and state:
        state["image"] = image
        if auto_down_min:
            state["auto_down_deadline_utc"] = (datetime.now(timezone.utc) + timedelta(minutes=auto_down_min)).isoformat()
            echo(f"auto-down deadline {state['auto_down_deadline_utc']} (reported by status; run `down` yourself)")
        v4_pod.save_state(paths.state, state)
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


def state_image(paths: v4_pod.Paths) -> str:
    state = v4_pod.load_state(paths.state)
    return (state or {}).get("image") or IMAGE


def main(argv: Optional[List[str]] = None, *, paths: Optional[v4_pod.Paths] = None,
         transport: Optional[httpx.BaseTransport] = None, api_key: Optional[str] = None,
         sleep: Callable[[float], None] = time.sleep, echo: Echo = print) -> int:
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
    for name in ("status", "wait", "down", "cost"):
        sub.add_parser(name)
    args = parser.parse_args(argv)
    paths = paths or v41_paths()
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
                          args.scripts_commit)
        except ValueError as exc:
            echo(f"refused: {exc}")
            return 2
    if args.cmd == "status":
        return cmd_status(client, paths, echo)
    with v41_bindings(state_image(paths)):
        return v4_pod.cmd_down(client, paths, echo)


if __name__ == "__main__":
    sys.exit(main())
