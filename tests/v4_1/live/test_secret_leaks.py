"""Secrets embedded in URLs or echoed by servers must not reach errors, cells, sqlite, stdout or stderr."""
import asyncio
import copy
import json
import pickle

import httpx
import pytest

from antelligence.experiments.v41_harness import error_cell
from antelligence.providers.base import ChatRequest, ProviderError
from antelligence.providers.openai_compat import OpenAICompatProvider
from antelligence.providers.redact import register_url, scrub

FAKE = "RPCKEY_distinctive_9f8e7d6c5b4a"
QUEEN_KEY = "queenkey_distinctive_aa11bb22cc33"


def test_scrub_rules():
    assert scrub("GET https://user:pw@host.io:8545/v2/SECRETPATH?k=v failed") == "GET https://host.io:8545 failed"
    assert "tok123" not in scrub("Authorization: Bearer tok123 rejected") and "tok123" not in scrub("bearer tok123")
    assert "sk-abcdefgh12345678" not in scrub("key sk-abcdefgh12345678")
    assert scrub("hash " + "ab" * 32) == "hash ***"
    assert scrub("my secret hunter2hunter2", ["hunter2hunter2"]) == "my secret ***"


def test_dead_rpc_url_with_key_leaks_nowhere(deployment, tmp_path, capsys):
    from chain.v4_1 import ChainClientV41, LocalSigner, TxQueue
    from web3 import Web3

    dead = Web3(Web3.HTTPProvider(f"http://127.0.0.1:1/v2/{FAKE}", request_kwargs={"timeout": 2}))
    live_w3 = deployment["w3"]
    register_url(f"http://127.0.0.1:1/v2/{FAKE}")     # what open_live_client does with the configured RPC URL
    q = TxQueue(tmp_path / "q.sqlite")
    c = ChainClientV41(live_w3, LocalSigner(deployment["key"]), q, deployment["addrs"], expected_deployer=deployment["deployer"],
                       expected_chain_id=31337, code_hashes=deployment["hashes"])
    jid = c.enqueue_anchor("aa" * 32, 1, "bb" * 32)
    c.preflight()
    c.w3 = type("W", (), {"__getattr__": lambda s, n: getattr(live_w3, n)})()
    # estimate_gas fails with an error that carries the URL
    c.w3.eth = type("E", (), {"__getattr__": lambda s, n: getattr(live_w3.eth, n),
                              "estimate_gas": lambda s, *a, **k: (_ for _ in ()).throw(
                                  ConnectionError(f"HTTPConnectionPool(host='127.0.0.1', port=1): url: /v2/{FAKE} refused http://127.0.0.1:1/v2/{FAKE}"))})()
    c.pump()
    job = q.get(jid)
    assert job["status"] == "failed" and FAKE not in json.dumps(job)
    q.close()
    raw = (tmp_path / "q.sqlite").read_bytes() + b"".join(p.read_bytes() for p in tmp_path.glob("q.sqlite-*"))
    assert FAKE.encode() not in raw
    cell = error_cell("SW-B", "c", 1, "dev", RuntimeError(f"boom http://x.io/v2/{FAKE}"))
    assert FAKE not in json.dumps(cell)
    out = capsys.readouterr()
    assert FAKE not in out.out + out.err
    # a real dead endpoint through web3 itself
    with pytest.raises(Exception) as ei:
        dead.eth.chain_id
    assert FAKE not in scrub(f"{type(ei.value).__name__}: {ei.value}")


def test_archive_script_scrubs_rpc_errors(tmp_path, capsys):
    from scripts import v41_archive_l1

    path = tmp_path / "a.json"
    path.write_text(json.dumps([{"run_id": "r", "size": 50, "tx_hash": "0x" + "ab" * 32}]))

    def rpc(method, params):
        if method == "eth_chainId":
            return "0x12c"
        raise RuntimeError(f"HTTPSConnectionPool: https://rpc.example/v2/{FAKE} refused")

    assert v41_archive_l1.main(["--anchors", str(path)], rpc=rpc) == 0
    out = capsys.readouterr()
    assert FAKE not in out.out + out.err and "https://rpc.example" in out.err


def test_queen_401_echoing_the_bearer_leaks_nowhere(tmp_path):
    from antelligence.experiments import v41_prod

    def handler(req: httpx.Request) -> httpx.Response:
        return httpx.Response(401, text=f"invalid credentials: Authorization: Bearer {QUEEN_KEY} at https://api.x/{QUEEN_KEY}")

    usage = tmp_path / "u.jsonl"
    q = v41_prod.build_live_queen(usage_ledger=usage, pod_costs=tmp_path / "pc.jsonl", cache_dir=None, nous_key=lambda: QUEEN_KEY)
    provider = q.inner.inner
    provider._transport = httpx.MockTransport(handler)
    req = ChatRequest(model="deepseek/deepseek-v4.1-flash", messages=({"role": "user", "content": "hi"},), max_tokens=8, temperature=0.0, seed=1) \
        if "seed" in ChatRequest.__dataclass_fields__ else ChatRequest(model="deepseek/deepseek-v4.1-flash", messages=({"role": "user", "content": "hi"},))
    with pytest.raises(ProviderError) as ei:
        asyncio.run(q.complete(req))
    assert QUEEN_KEY not in str(ei.value) and QUEEN_KEY not in repr(ei.value.__cause__)
    assert QUEEN_KEY not in usage.read_text()
    cell = error_cell("SW-B", "c", 1, "dev", ei.value)
    assert QUEEN_KEY not in json.dumps(cell)


def test_local_signer_cannot_be_pickled_or_copied_and_holds_no_key_attribute(deployment):
    from chain.v4_1 import LocalSigner

    s = LocalSigner(deployment["key"])
    for fn in (pickle.dumps, copy.copy, copy.deepcopy):
        with pytest.raises(TypeError):
            fn(s)
    key_hex = deployment["key"].removeprefix("0x").lower()
    assert key_hex not in repr(vars(s)).lower() and not any(isinstance(v, (bytes, str)) and key_hex in str(v).lower() for v in vars(s).values())
    assert len(s.sign({"to": "0x" + "11" * 20, "value": 0, "nonce": 0, "gas": 21000, "gasPrice": 1, "chainId": 31337})) > 50
