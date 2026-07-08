"""Determinism contract for fixture-only chain-reader ingestion buffers."""

import json
import subprocess
import sys
from pathlib import Path


def _fixture_events() -> list[dict]:
    return [
        {
            "chain_id": 84532,
            "source": "TumorIntel.PinConfirmed",
            "block_height": 120,
            "tx_index": 2,
            "log_index": 5,
            "payload": {"pin_id": 7, "priority": 9},
        },
        {
            "chain_id": 84532,
            "source": "ExperienceRegistry.ExperienceSubmitted",
            "block_height": 119,
            "tx_index": 7,
            "log_index": 1,
            "payload": {"run_hash": "0xabc", "score": 9000},
        },
        {
            "chain_id": 84532,
            "source": "TumorIntel.PinConfirmed",
            "block_height": 120,
            "tx_index": 2,
            "log_index": 4,
            "payload": {"pin_id": 6, "priority": 8},
        },
    ]


def test_fixture_ingestion_is_byte_identical_across_order_and_process_runs(tmp_path: Path):
    from backend.chain.deterministic_buffer import (
        DETERMINISM_RISK_NOTE,
        DeterministicChainBuffer,
        pheromone_provenance_checksum,
        trust_tier_label,
    )

    first = DeterministicChainBuffer.from_events(_fixture_events())
    second = DeterministicChainBuffer.from_events(list(reversed(_fixture_events())))

    assert first.canonical_bytes() == second.canonical_bytes()
    assert first.digest() == second.digest()
    assert json.loads(first.canonical_bytes()) == {
        "events": [
            {
                "chain_id": 84532,
                "source": "ExperienceRegistry.ExperienceSubmitted",
                "block_height": 119,
                "tx_index": 7,
                "log_index": 1,
                "payload": {"run_hash": "0xabc", "score": 9000},
            },
            {
                "chain_id": 84532,
                "source": "TumorIntel.PinConfirmed",
                "block_height": 120,
                "tx_index": 2,
                "log_index": 4,
                "payload": {"pin_id": 6, "priority": 8},
            },
            {
                "chain_id": 84532,
                "source": "TumorIntel.PinConfirmed",
                "block_height": 120,
                "tx_index": 2,
                "log_index": 5,
                "payload": {"pin_id": 7, "priority": 9},
            },
        ]
    }

    fixture_path = tmp_path / "events.json"
    fixture_path.write_text(json.dumps(list(reversed(_fixture_events()))), encoding="utf-8")
    code = (
        "import json, pathlib; "
        "from backend.chain.deterministic_buffer import DeterministicChainBuffer; "
        f"events=json.loads(pathlib.Path({str(fixture_path)!r}).read_text()); "
        "buffer=DeterministicChainBuffer.from_events(events); "
        "print(buffer.digest()); "
        "print(buffer.canonical_bytes().decode())"
    )
    proc = subprocess.run(
        [sys.executable, "-c", code],
        cwd=Path(__file__).resolve().parents[1],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=True,
    )
    digest, canonical = proc.stdout.splitlines()[0], "\n".join(proc.stdout.splitlines()[1:])

    assert digest == first.digest()
    assert canonical.encode() == first.canonical_bytes()
    assert trust_tier_label(event_block_height=120, observer_height=120, finality_height=None) == "T2 observer@120"
    assert trust_tier_label(event_block_height=120, observer_height=None, finality_height=None) == "T1 local"
    assert trust_tier_label(event_block_height=120, observer_height=120, finality_height=125) == "T3 finality"
    assert pheromone_provenance_checksum("run-7", first.digest(), "prov-1") == pheromone_provenance_checksum("run-7", second.digest(), "prov-1")
    assert "identical-height" in DETERMINISM_RISK_NOTE


def test_duplicate_chain_coordinates_are_still_order_insensitive():
    from backend.chain.deterministic_buffer import DeterministicChainBuffer

    events = [
        {
            "chain_id": 84532,
            "source": "TumorIntel.PinConfirmed",
            "block_height": 120,
            "tx_index": 2,
            "log_index": 5,
            "payload": {"pin_id": 7, "priority": 9},
        },
        {
            "chain_id": 84532,
            "source": "TumorIntel.PinConfirmed",
            "block_height": 120,
            "tx_index": 2,
            "log_index": 5,
            "payload": {"pin_id": 8, "priority": 3},
        },
    ]

    first = DeterministicChainBuffer.from_events(events)
    second = DeterministicChainBuffer.from_events(list(reversed(events)))

    assert first.canonical_bytes() == second.canonical_bytes()
    assert first.digest() == second.digest()
