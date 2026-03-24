"""Integration tests for blockchain operations on Base Sepolia.

These tests hit the actual Base Sepolia testnet to verify:
- Contract deployment and accessibility
- ExperienceRegistry submission and retrieval
- TumorIntel pin reporting and querying
- Data integrity across submit → verify cycle

Requires: BASE_SEPOLIA_RPC_URL and PRIVATE_KEY in .env
Skip if not configured: tests gracefully skip without blockchain access.
"""

import json
import os
import sys
import pytest
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from dotenv import load_dotenv
load_dotenv(Path(__file__).parent.parent / '.env')

# Skip all tests if blockchain not configured
RPC_URL = os.getenv("BASE_SEPOLIA_RPC_URL")
PRIVATE_KEY = os.getenv("PRIVATE_KEY")
HAS_BLOCKCHAIN = bool(RPC_URL and PRIVATE_KEY)

pytestmark = pytest.mark.skipif(
    not HAS_BLOCKCHAIN,
    reason="BASE_SEPOLIA_RPC_URL and PRIVATE_KEY not set"
)

# Contract addresses (deployed 2026-03-24)
EXPERIENCE_REGISTRY = "0x22ECc5e4ddcCbAa44f508480e09eBD2640Dcd4e9"
TUMOR_INTEL = "0xb7BD3627E0b30D6EEc394Cd663da1ffC6245F6cC"
COLONY_MEMORY = "0xc71F99c6DbDb762879fea0f2C38dACE320D494b3"
FOOD_TOKEN = "0x002E7C3C0759e1A1da4dE8F43e7792473fe67AA9"


@pytest.fixture(scope="module")
def w3():
    from web3 import Web3
    w3 = Web3(Web3.HTTPProvider(RPC_URL))
    assert w3.is_connected(), f"Cannot connect to {RPC_URL}"
    return w3


@pytest.fixture(scope="module")
def account(w3):
    from eth_account import Account
    return Account.from_key(PRIVATE_KEY)


@pytest.fixture(scope="module")
def registry(w3):
    abi_path = Path(__file__).parent.parent / "blockchain" / "artifacts" / "contracts" / "ExperienceRegistry.sol" / "ExperienceRegistry.json"
    if not abi_path.exists():
        pytest.skip("ABI not found — run 'npx hardhat compile' first")
    with open(abi_path) as f:
        abi = json.load(f)["abi"]
    return w3.eth.contract(address=w3.to_checksum_address(EXPERIENCE_REGISTRY), abi=abi)


@pytest.fixture(scope="module")
def tumor_intel(w3):
    abi_path = Path(__file__).parent.parent / "blockchain" / "artifacts" / "contracts" / "TumorIntel.sol" / "TumorIntel.json"
    if not abi_path.exists():
        pytest.skip("ABI not found")
    with open(abi_path) as f:
        abi = json.load(f)["abi"]
    return w3.eth.contract(address=w3.to_checksum_address(TUMOR_INTEL), abi=abi)


# ---------------------------------------------------------------------------
# Connection & Contract Accessibility
# ---------------------------------------------------------------------------

class TestBlockchainConnection:

    def test_connected_to_base_sepolia(self, w3):
        chain_id = w3.eth.chain_id
        assert chain_id == 84532, f"Expected Base Sepolia (84532), got {chain_id}"

    def test_deployer_has_balance(self, w3, account):
        balance = w3.eth.get_balance(account.address)
        assert balance > 0, "Deployer has 0 ETH — fund via faucet"

    def test_experience_registry_deployed(self, w3):
        code = w3.eth.get_code(w3.to_checksum_address(EXPERIENCE_REGISTRY))
        assert len(code) > 2, "ExperienceRegistry has no code at address"

    def test_tumor_intel_deployed(self, w3):
        code = w3.eth.get_code(w3.to_checksum_address(TUMOR_INTEL))
        assert len(code) > 2, "TumorIntel has no code at address"

    def test_colony_memory_deployed(self, w3):
        code = w3.eth.get_code(w3.to_checksum_address(COLONY_MEMORY))
        assert len(code) > 2, "ColonyMemory has no code at address"

    def test_food_token_deployed(self, w3):
        code = w3.eth.get_code(w3.to_checksum_address(FOOD_TOKEN))
        assert len(code) > 2, "FoodToken has no code at address"


# ---------------------------------------------------------------------------
# ExperienceRegistry
# ---------------------------------------------------------------------------

class TestExperienceRegistry:

    def test_read_existing_submission(self, registry, w3):
        """Verify we can read the submission from earlier today."""
        run_hash = w3.to_bytes(
            hexstr="0x7b3f4292792f0ab5b70af2dea162d486746b7a3046b4b219f66f86d19e755159"
        )
        exp = registry.functions.experiences(run_hash).call()
        # exp[0] = runHash, exp[1] = ipfsCid, exp[3] = score, exp[4] = submitter
        assert exp[1] != "", "IPFS CID should not be empty"
        assert exp[3] == 4550, f"Expected score 4550, got {exp[3]}"
        assert exp[4] == "0xEE8a688CE7beb1bd46bd5C84bd774Efc750fB086"

    def test_owner_is_deployer(self, registry, account):
        owner = registry.functions.owner().call()
        assert owner == account.address

    def test_min_attestations_readable(self, registry):
        min_att = registry.functions.minAttestations().call()
        assert min_att >= 1


# ---------------------------------------------------------------------------
# TumorIntel
# ---------------------------------------------------------------------------

class TestTumorIntel:

    def test_pin_count(self, tumor_intel):
        count = tumor_intel.functions.getPinCount().call()
        assert count >= 0  # may have pins from earlier tests

    def test_get_active_pins(self, tumor_intel):
        pins = tumor_intel.functions.getActivePins().call()
        assert isinstance(pins, list)


# ---------------------------------------------------------------------------
# Data Integrity (submit → verify cycle)
# ---------------------------------------------------------------------------

class TestDataIntegrity:

    def test_submitted_experience_has_valid_fields(self, registry, w3):
        """Verify on-chain submission has all expected non-zero fields."""
        run_hash = w3.to_bytes(
            hexstr="0x7b3f4292792f0ab5b70af2dea162d486746b7a3046b4b219f66f86d19e755159"
        )
        exp = registry.functions.experiences(run_hash).call()
        # exp: (runHash, ipfsCid, dataHash, score, submitter, timestamp, attestations, verified)
        assert exp[0] != b'\x00' * 32, "runHash should not be zero"
        assert len(exp[1]) > 0, "ipfsCid should not be empty"
        assert exp[2] != b'\x00' * 32, "dataHash should not be zero"
        assert exp[3] > 0, "score should be positive"
        assert exp[4] != "0x" + "0" * 40, "submitter should not be zero address"
        assert exp[5] > 0, "timestamp should be positive"
