"""Python reputation rule == the table the Solidity test also asserts == the constants in the .sol file."""

import json
import re
from pathlib import Path

import pytest

from chain.v4_1 import (
    REP_EXCLUDE_BELOW_PPM,
    REP_START_PPM,
    REP_STEP_UP_PPM,
    is_excluded,
    reputation_update,
)

ROOT = Path(__file__).resolve().parents[3]
TABLE = json.loads((ROOT / "tests/v4_1/vectors/reputation_parity.json").read_text())
SOL = (ROOT / "blockchain/contracts/ReputationRegistry.sol").read_text()


@pytest.mark.parametrize("case", TABLE["cases"], ids=lambda c: f"{c['r']}-{'bad' if c['contradicted'] else 'ok'}")
def test_rule_matches_table(case):
    assert reputation_update(case["r"], case["contradicted"]) == case["expect"]


@pytest.mark.parametrize("case", TABLE["excluded"], ids=lambda c: str(c["r"]))
def test_exclusion_threshold(case):
    assert is_excluded(case["r"]) is case["excluded"]


def test_constants_match_table_and_solidity_source():
    assert (REP_START_PPM, REP_EXCLUDE_BELOW_PPM) == (TABLE["start_ppm"], TABLE["exclude_below_ppm"])
    sol = {m[0]: int(m[1].replace("_", "")) for m in re.findall(r"uint32 public constant (\w+) = ([\d_]+);", SOL)}
    assert sol == {"START_PPM": REP_START_PPM, "STEP_UP_PPM": REP_STEP_UP_PPM, "EXCLUDE_BELOW_PPM": REP_EXCLUDE_BELOW_PPM}
    assert "r / 2" in SOL  # floor halving, integer division


def test_rule_properties():
    assert reputation_update(REP_START_PPM, False) == REP_START_PPM  # cap
    r = REP_START_PPM
    for _ in range(40):
        r = reputation_update(r, True)
    assert r == 0
    for _ in range(10):
        r = reputation_update(r, False)
    assert r == REP_START_PPM
    with pytest.raises(ValueError):
        reputation_update(-1, True)
    with pytest.raises(ValueError):
        reputation_update(REP_START_PPM + 1, False)
