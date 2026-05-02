"""Tests for backend/config.py — SimulationConfig schema."""

import json
import tempfile
from pathlib import Path

import pytest
from pydantic import ValidationError

from backend.config import PheromoneParams, SimulationConfig, load_config, save_config

class TestSimulationConfig:
    def test_defaults(self):
        cfg = SimulationConfig()
        assert cfg.num_bots == 10
        assert cfg.grid_size == 60
        assert cfg.steps == 100
        assert cfg.queen_enabled is False
        assert cfg.seed is None
        assert isinstance(cfg.pheromone_params, PheromoneParams)

    def test_custom_values(self):
        cfg = SimulationConfig(num_bots=5, grid_size=20, steps=50, queen_enabled=True, seed=42)
        assert cfg.num_bots == 5
        assert cfg.grid_size == 20
        assert cfg.steps == 50
        assert cfg.queen_enabled is True
        assert cfg.seed == 42

    def test_num_bots_validation_error(self):
        with pytest.raises(ValidationError):
            SimulationConfig(num_bots=0)

    def test_grid_size_validation_error(self):
        with pytest.raises(ValidationError):
            SimulationConfig(grid_size=1)

    def test_steps_validation_error(self):
        with pytest.raises(ValidationError):
            SimulationConfig(steps=0)

    def test_pheromone_params_nested(self):
        cfg = SimulationConfig(pheromone_params={"trail_diffusion": 2e-6, "alarm_diffusion": 1e-5})
        assert cfg.pheromone_params.trail_diffusion == pytest.approx(2e-6)
        assert cfg.pheromone_params.alarm_diffusion == pytest.approx(1e-5)

    def test_pheromone_params_invalid_diffusion(self):
        with pytest.raises(ValidationError):
            SimulationConfig(pheromone_params={"trail_diffusion": -1.0})

    def test_to_model_kwargs(self):
        cfg = SimulationConfig(num_bots=3, grid_size=5)
        kwargs = cfg.to_model_kwargs()
        assert kwargs["n_nanobots"] == 3
        assert kwargs["domain_size"] == pytest.approx(50.0)
        assert kwargs["voxel_size"] == pytest.approx(10.0)
        assert kwargs["with_queen"] is False
        assert kwargs["seed"] is None
        assert "pheromone_params" in kwargs

    def test_to_model_kwargs_carries_pheromone_params(self):
        cfg = SimulationConfig(
            pheromone_params={
                "trail_diffusion": 2e-6,
                "alarm_diffusion": 9e-6,
                "recruitment_decay": 0.33,
            }
        )
        kwargs = cfg.to_model_kwargs()
        assert kwargs["pheromone_params"]["trail_diffusion"] == pytest.approx(2e-6)
        assert kwargs["pheromone_params"]["alarm_diffusion"] == pytest.approx(9e-6)
        assert kwargs["pheromone_params"]["recruitment_decay"] == pytest.approx(0.33)

    def test_json_roundtrip(self):
        cfg = SimulationConfig(num_bots=7, grid_size=30, steps=200, seed=99)
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            tmp = Path(f.name)
        try:
            save_config(cfg, tmp)
            loaded = load_config(tmp)
            assert loaded.num_bots == 7
            assert loaded.grid_size == 30
            assert loaded.steps == 200
            assert loaded.seed == 99
        finally:
            tmp.unlink(missing_ok=True)

    def test_load_config_from_json_string(self):
        data = {"num_bots": 4, "grid_size": 10, "steps": 20}
        with tempfile.NamedTemporaryFile(suffix=".json", mode="w", delete=False) as f:
            json.dump(data, f)
            tmp = Path(f.name)
        try:
            cfg = load_config(tmp)
            assert cfg.num_bots == 4
        finally:
            tmp.unlink(missing_ok=True)

    def test_runtime_factory_config_guard(self):
        """Verify that runtime factory can handle different config states without crashing."""
        import sys
        from unittest.mock import MagicMock

        # We must prevent the import of the real module because it has missing dependencies (biofvm)
        # This is a hack for the test to run in an environment where biofvm isn't installed.
        sys.modules["backend.nanobot_simulation"] = MagicMock()

        from backend.runtime_factory import build_model, run_simulation

        # Mock the model to avoid heavy computation and dependency issues
        mock_model = MagicMock()
        mock_model.step.return_value = None
        mock_model.metrics = {}
        mock_model.geometry.get_tumor_statistics.return_value = {"total_cells": 100, "living_cells": 50}
        mock_factory = MagicMock(return_value=mock_model)

        cfg = SimulationConfig(num_bots=5, grid_size=10, steps=1)

        # Test build_model
        model = build_model(cfg, model_factory=mock_factory)
        assert model == mock_model

        # Test run_simulation
        sim_model, metrics = run_simulation(cfg, model_factory=mock_factory)
        assert sim_model == mock_model
        assert metrics["kill_rate"] == 0.5

        # Cleanup
        del sys.modules["backend.nanobot_simulation"]
