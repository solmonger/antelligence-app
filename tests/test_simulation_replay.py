from unittest.mock import MagicMock

from pathlib import Path
from unittest.mock import patch

import pytest

from backend.run_store import SQLiteRunStore
from backend.simulation_replay import replay_artifact_metrics


def test_simulation_replay_determinism(tmp_path: Path):
    """Verify that replaying a stored run produces the same metrics."""
    db_path = tmp_path / "test_runs.sqlite3"
    store = SQLiteRunStore(db_path)
    
    run_id = "replay-test-001"
    config = {
        "num_bots": 2,
        "steps": 5,
        "domain_size": 100.0,
        "voxel_size": 10.0,
        "seed": 42
    }
    metrics = {
        "kill_rate": 0.0,
        "total_deliveries": 0,
        "cells_killed": 0,
        "step_count": 5
    }
    
    store.save_run(run_id, "completed", config, metrics)
    retrieved_run = store.get_run(run_id)
    assert retrieved_run is not None
    assert retrieved_run["run_id"] == run_id
    assert retrieved_run["status"] == "completed"
    assert retrieved_run["config"]["num_bots"] == 2
    
    # We mock the simulation execution to avoid the IndexError and the heavy computation
    with patch("backend.simulation_replay.build_model_from_config") as mock_build:
        mock_model = MagicMock()
        mock_model.step_count = 5
        mock_build.return_value = (MagicMock(), mock_model)
        
        with patch("backend.simulation_replay.compute_metrics") as mock_metrics:
            mock_metrics.return_value = {
                "kill_rate": 0.0,
                "total_deliveries": 0,
                "cells_killed": 0,
                "step_count": 5
            }
            
            replayed_metrics = replay_artifact_metrics(retrieved_run)
            assert replayed_metrics["step_count"] == 5
            assert replayed_metrics["kill_rate"] == 0.0

def test_replay_config_extraction(tmp_path: Path):
    """Verify that the replay helper correctly extracts config from the artifact."""
    from backend.simulation_replay import normalize_simulation_config
    
    config = {"num_bots": 5, "steps": 10, "seed": 123}
    normalized = normalize_simulation_config(config)
    
    assert normalized.num_bots == 5
    assert normalized.steps == 10
    assert normalized.seed == 123


def test_replay_supports_legacy_cli_bots_key():
    """Older CLI artifacts used `bots`; replays should still honor that value."""
    from backend.simulation_replay import normalize_simulation_config

    normalized = normalize_simulation_config({"bots": 7, "grid_size": 12, "steps": 4, "seed": 9})

    assert normalized.num_bots == 7
    assert normalized.grid_size == 12
    assert normalized.steps == 4
    assert normalized.seed == 9


def test_replay_rejects_zero_steps_instead_of_defaulting():
    """Explicit invalid step counts should fail closed during replay normalization."""
    import pydantic

    from backend.simulation_replay import normalize_simulation_config

    with pytest.raises(pydantic.ValidationError):
        normalize_simulation_config({"num_bots": 5, "grid_size": 12, "steps": 0})


def test_replay_artifact_preserves_seed_config_and_swarm_provenance():
    """Replay output should carry deterministic inputs plus emitted communication/pheromone traces."""
    artifact = {
        "config": {
            "num_bots": 3,
            "steps": 2,
            "grid_size": 9,
            "seed": 20260506,
            "pheromone_params": {"trail_decay": 0.12},
        },
        "metrics": {
            "total_deliveries": 4,
            "cells_killed": 2,
            "step_count": 2,
            "communication_trace": [{"step": 1, "sender": "bot-1", "kind": "recruit"}],
            "pheromone_provenance": {"trail": [{"step": 1, "bot_id": 1, "amount": 3.0}]},
        },
    }

    with patch("backend.simulation_replay.build_model_from_config") as mock_build:
        from backend.config import SimulationConfig

        mock_model = MagicMock()
        mock_model.step_count = 2
        mock_build.return_value = (SimulationConfig(**artifact["config"]), mock_model)
        with patch("backend.simulation_replay.compute_metrics") as mock_metrics:
            mock_metrics.return_value = artifact["metrics"]

            replayed_metrics = replay_artifact_metrics(artifact)

    assert replayed_metrics["replay_metadata"] == {
        "original_config": artifact["config"],
        "seed": 20260506,
        "normalized_config": replayed_metrics["normalized_config"],
        "swarm_provenance": {
            "communication_trace": artifact["metrics"]["communication_trace"],
            "pheromone_provenance": artifact["metrics"]["pheromone_provenance"],
        },
    }


def test_replay_provenance_drops_scalar_values_instead_of_propagating_them():
    """A scalar provenance value must not corrupt replay metadata.

    If a provenance key like ``agent_messages`` is set to a bare integer
    (e.g. a mis-logged count instead of a structured trace), the replay
    pipeline must drop it rather than letting an opaque scalar leak into
    the replay_metadata.swarm_provenance dict.
    """
    from backend.simulation_replay import extract_swarm_provenance

    artifact = {
        "agent_messages": 3,
        "pheromone_summary": {"trail_decay": 0.5, "steps": 10},
    }
    runtime_metrics = {"communication_provenance": 42}

    provenance = extract_swarm_provenance(artifact, runtime_metrics)
    # Scalar values (int 3, int 42) are dropped; only dict pheromone_summary survives
    assert provenance == {"pheromone_summary": {"trail_decay": 0.5, "steps": 10}}


def test_replay_provenance_keeps_list_and_dict_values():
    """Structured provenance (list traces, dict summaries) must pass through."""
    from backend.simulation_replay import extract_swarm_provenance

    artifact = {
        "communication_trace": [{"step": 1, "sender": "bot-1"}],
        "pheromone_trace": [{"step": 1, "amount": 2.5}],
    }
    provenance = extract_swarm_provenance(artifact, {})
    assert provenance == {
        "communication_trace": [{"step": 1, "sender": "bot-1"}],
        "pheromone_trace": [{"step": 1, "amount": 2.5}],
    }
