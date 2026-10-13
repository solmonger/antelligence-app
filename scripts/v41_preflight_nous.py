"""v4.1 preflight: one tiny request per variant to the queen model on Nous, to find the thinking-off parameter.

Lead-only (it reads the Nous key). The key is read in-process from ~/.hermes/auth.json exactly as the
v3/v4 collectors do, and is never printed or logged. Each probe appends one usage row to
ledger/v4.1-usage.jsonl (gitignored): status, served model, token usage, how much reasoning text came
back, and latency. Only the first 40 characters of the reply are stored.

    .venv/bin/python scripts/v41_preflight_nous.py
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import httpx

URL = "https://inference-api.nousresearch.com/v1/chat/completions"
MODEL = "deepseek/deepseek-v4.1-flash"
LEDGER = Path(__file__).resolve().parents[1] / "ledger" / "v4.1-usage.jsonl"
PROBES = [
    ("none", {}),
    ("thinking_type_disabled", {"thinking": {"type": "disabled"}}),
    ("chat_template_kwargs_thinking_false", {"chat_template_kwargs": {"thinking": False}}),
    ("reasoning_enabled_false", {"reasoning": {"enabled": False}}),
    ("reasoning_effort_none", {"reasoning": {"effort": "none"}}),
    ("reasoning_max_tokens_0", {"reasoning": {"max_tokens": 0}}),
]


def _key() -> str:
    return json.loads(Path("~/.hermes/auth.json").expanduser().read_text())["providers"]["nous"]["agent_key"]


def main() -> None:
    headers = {"Authorization": f"Bearer {_key()}"}
    LEDGER.parent.mkdir(parents=True, exist_ok=True)
    import sys
    only = set(sys.argv[sys.argv.index("--only") + 1].split(",")) if "--only" in sys.argv else None
    for label, extra in PROBES:
        if only and label not in only:
            continue
        body = {"model": MODEL, "messages": [{"role": "user", "content": "Reply with the single word: ok"}],
                "max_tokens": 64, "temperature": 0, **extra}
        t0 = time.time()
        r = httpx.post(URL, headers=headers, json=body, timeout=60)
        dt = round(time.time() - t0, 2)
        try:
            j = r.json()
        except ValueError:
            j = {}
        msg = ((j.get("choices") or [{}])[0] or {}).get("message") or {}
        reasoning = msg.get("reasoning") or msg.get("reasoning_content") or ""
        err = j.get("error")
        row = {"t": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "provider": "nous", "phase": "preflight",
               "probe": label, "status": r.status_code, "model": j.get("model"), "usage": j.get("usage"),
               "reasoning_chars": len(reasoning), "content_head": (msg.get("content") or "")[:40],
               "error": err.get("message") if isinstance(err, dict) else err, "latency_s": dt}
        with LEDGER.open("a") as f:
            f.write(json.dumps(row) + "\n")
        print(json.dumps(row))


if __name__ == "__main__":
    main()
