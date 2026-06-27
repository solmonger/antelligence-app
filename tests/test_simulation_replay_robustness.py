import pytest
from backend.config import SimulationConfig
from backend.simulation_replay import normalize_simulation_config

def test_normalize_simulation_config_invalid_types():
    """Ensure that non-integer inputs for num_bots/grid_size are cast correctly,
    but explicit invalid values like negative bot counts trigger validation."""

    # Valid float string should cast
    normalized = normalize_simulation_config({"num_bots": "5.0", "steps": 5})
    assert normalized.num_bots == 5

    # Negative bots should fail if config validation enforces uint
    with pytest.raises(ValueError):
        normalize_simulation_config({"num_bots": -1, "steps": 5})
