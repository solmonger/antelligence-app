"""Local hardhat node (chain 31337) with the three real v4.1 contracts deployed, for the live-wiring tests.

Hardhat's published development mnemonic stands in for the deployer: no real key is read anywhere. Tests skip when no
sibling checkout has node_modules and compiled artifacts.
"""
import json
import shutil
import socket
import subprocess
import sys
import time
from pathlib import Path

import pytest
from eth_account import Account
from web3 import Web3

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "backend"))
NODE = "/opt/homebrew/opt/node@22/bin"
HARDHAT_MNEMONIC = "test test test test test test test test test test test junk"
NAMES = ("AnchorRegistry", "BoardLog", "ReputationRegistry")


def _bc_dir():
    for cand in (REPO / "blockchain", REPO.parent / "v41-integration" / "blockchain",
                 REPO.parent / "v41-s1-contracts" / "blockchain"):
        if (cand / "node_modules/.bin/hardhat").exists() and (cand / "artifacts/contracts/AnchorRegistry.sol/AnchorRegistry.json").exists():
            return cand
    return None


def _free_port():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


@pytest.fixture(scope="session")
def node_url():
    bc = _bc_dir()
    npx = shutil.which("npx", path=f"{NODE}:/usr/bin:/usr/local/bin:/opt/homebrew/bin")
    if bc is None or not npx:
        pytest.skip("no hardhat checkout with node_modules + compiled artifacts")
    port = _free_port()
    env = {"PATH": f"{NODE}:/usr/bin:/bin:/usr/local/bin:/opt/homebrew/bin", "HOME": str(Path.home())}
    proc = subprocess.Popen([npx, "hardhat", "node", "--port", str(port)], cwd=bc, env=env,
                            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    url = f"http://127.0.0.1:{port}"
    w3 = Web3(Web3.HTTPProvider(url))
    for _ in range(80):
        try:
            if w3.eth.chain_id == 31337:
                break
        except Exception:
            time.sleep(0.5)
    else:
        proc.kill()
        pytest.skip("hardhat node did not start")
    yield url
    proc.terminate()
    proc.wait(timeout=10)


@pytest.fixture(scope="session")
def deployer_key():
    Account.enable_unaudited_hdwallet_features()
    return Account.from_mnemonic(HARDHAT_MNEMONIC).key.hex()


@pytest.fixture
def deployment(node_url, deployer_key):
    """Fresh contracts per test (fresh addresses, so queues and runs never collide). Returns a dict."""
    w3 = Web3(Web3.HTTPProvider(node_url))
    bc = _bc_dir()
    account = w3.eth.accounts[0]
    addrs, hashes = {}, {}
    for name in NAMES:
        art = json.loads((bc / f"artifacts/contracts/{name}.sol/{name}.json").read_text())
        rc = w3.eth.wait_for_transaction_receipt(
            w3.eth.contract(abi=art["abi"], bytecode=art["bytecode"]).constructor().transact({"from": account}))
        addrs[name] = rc["contractAddress"]
        hashes[name] = "0x" + bytes(Web3.keccak(bytes(w3.eth.get_code(rc["contractAddress"])))).hex()
    return {"w3": w3, "addrs": addrs, "hashes": hashes, "deployer": account, "key": deployer_key}


@pytest.fixture
def live_client(deployment, tmp_path):
    from chain.v4_1 import ChainClientV41, LocalSigner, TxQueue

    c = ChainClientV41(deployment["w3"], LocalSigner(deployment["key"]), TxQueue(tmp_path / "q.sqlite"),
                       deployment["addrs"], expected_deployer=deployment["deployer"], expected_chain_id=31337,
                       code_hashes=deployment["hashes"], replace_after_s=5)
    yield c
    c.queue.close()


@pytest.fixture
def mocks_lab():
    from antelligence.anchoring.tamper import Lab

    return Lab()
