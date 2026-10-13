"""The real AnchorRegistry on a LOCAL hardhat node (chain id 31337). Skipped when node/hardhat is unavailable."""

import shutil
import socket
import subprocess
import time
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[3]
ARTIFACT = REPO / "blockchain/artifacts/contracts/AnchorRegistry.sol/AnchorRegistry.json"
NODE = "/opt/homebrew/opt/node@22/bin"


def _free_port():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


@pytest.fixture(scope="module")
def local_node():
    npx = shutil.which("npx", path=f"{NODE}:/usr/bin:/usr/local/bin:/opt/homebrew/bin")
    if not npx or not ARTIFACT.exists() or not (REPO / "blockchain/node_modules/.bin/hardhat").exists():
        pytest.skip("hardhat not installed/compiled (npm ci && npx hardhat compile in blockchain/)")
    port = _free_port()
    env = {"PATH": f"{NODE}:/usr/bin:/bin:/usr/local/bin:/opt/homebrew/bin", "HOME": str(Path.home())}
    proc = subprocess.Popen([npx, "hardhat", "node", "--port", str(port)], cwd=REPO / "blockchain", env=env,
                            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    from web3 import Web3

    w3 = Web3(Web3.HTTPProvider(f"http://127.0.0.1:{port}"))
    for _ in range(60):
        try:
            if w3.eth.chain_id == 31337:
                break
        except Exception:
            time.sleep(0.5)
    else:
        proc.kill()
        pytest.skip("hardhat node did not start")
    yield w3
    proc.terminate()
    proc.wait(timeout=10)


def test_tamper_matrix_on_real_contract_local_node(local_node):
    from antelligence.anchoring import tamper

    w3 = local_node
    factory = lambda: tamper.deploy_local_registry(w3, ARTIFACT)  # noqa: E731
    for attack, position, k, tail in [("T1", "middle", 0, False), ("T2", "truncate_10", 10, False), ("T4", "-", 0, False),
                                      ("T1", "tail", 0, True)]:
        for o in tamper.run_scenario(attack, position, seed=21, k=k, tail=tail, registry_factory=factory):
            label = (f"tail_truncate_{k}" if attack == "T2" else "tail") if tail else position
            assert o.detected == tamper.expected_detects(attack, label, o.condition), (attack, label, o.condition)
            if o.condition == "C" and o.detected:
                assert o.when == "write"                       # the real contract reverts the second history
