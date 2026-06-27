"""Tests for backend/api_server.py using FastAPI TestClient."""

from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from backend.api_server import app, _RUNS
from backend.run_store import SQLiteRunStore

client = TestClient(app)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _fake_model_factory(**kwargs):
    """Return a lightweight mock TumorNanobotModel."""
    model = MagicMock()
    model.step_count = 5
    model.metrics = {
        "total_deliveries": 2,
        "total_drug_delivered": 10.0,
        "cells_killed": 1,
        "hypoxic_cells": 3,
        "viable_cells": 50,
        "necrotic_cells": 0,
        "apoptotic_cells": 1,
        "total_api_calls": 0,
        "food_collected_by_llm": 0,
        "food_collected_by_rule": 2,
        "deliveries_by_llm": 0,
        "deliveries_by_rule": 2,
    }
    model.geometry.get_tumor_statistics.return_value = {
        "total_cells": 60,
        "living_cells": 55,
    }
    return model


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestHealthEndpoint:
    def test_health_returns_200(self):
        resp = client.get("/health")
        assert resp.status_code == 200

    def test_health_body(self):
        resp = client.get("/health")
        data = resp.json()
        assert data["status"] == "ok"
        assert "version" in data
        assert data["run_store_backend"] == "sqlite"
        assert data["run_store_ready"] is True

    def test_health_returns_503_when_run_store_is_unavailable(self):
        failing_store = MagicMock()
        failing_store.is_ready.return_value = False

        with patch("backend.api_server.RUN_STORE", failing_store):
            resp = client.get("/health")

        assert resp.status_code == 503
        assert "run store" in resp.json()["detail"].lower()


class TestSimulateEndpoint:
    def test_simulate_returns_run_id(self):
        with patch("backend.api_server.TumorNanobotModel", side_effect=_fake_model_factory):
            resp = client.post("/simulate", json={"num_bots": 2, "grid_size": 5, "steps": 3})
        assert resp.status_code == 200
        data = resp.json()
        assert "run_id" in data
        assert data["status"] == "completed"

    def test_simulate_returns_metrics(self):
        with patch("backend.api_server.TumorNanobotModel", side_effect=_fake_model_factory):
            resp = client.post("/simulate", json={"num_bots": 2, "grid_size": 5, "steps": 3})
        data = resp.json()
        assert "metrics" in data
        assert "kill_rate" in data["metrics"]

    def test_simulate_invalid_num_bots(self):
        resp = client.post("/simulate", json={"num_bots": 0, "grid_size": 5, "steps": 3})
        assert resp.status_code == 422

    def test_simulate_rejects_unknown_request_fields(self):
        resp = client.post("/simulate", json={"num_bots": 2, "grid_size": 5, "steps": 3, "unexpected_flag": True})
        assert resp.status_code == 422
        assert any(
            error.get("loc") == ["body", "unexpected_flag"]
            and error.get("type") == "extra_forbidden"
            for error in resp.json()["detail"]
        )

    def test_simulate_accepts_protocol_fields_and_stores_config(self):
        payload = {
            "num_bots": 2,
            "grid_size": 5,
            "steps": 3,
            "queen_enabled": True,
            "seed": 123,
            "pheromone_params": {
                "trail_diffusion": 2e-6,
                "alarm_diffusion": 9e-6,
                "recruitment_diffusion": 3e-6,
                "trail_decay": 0.07,
                "alarm_decay": 0.24,
                "recruitment_decay": 0.11,
            },
        }
        with patch("backend.api_server.TumorNanobotModel", side_effect=_fake_model_factory):
            resp = client.post("/simulate", json=payload)

        assert resp.status_code == 200
        run_id = resp.json()["run_id"]
        stored_config = _RUNS[run_id]["config"]
        assert stored_config["queen_enabled"] is True
        assert stored_config["seed"] == 123
        assert stored_config["pheromone_params"] == payload["pheromone_params"]

    def test_simulate_rejects_unknown_pheromone_fields(self):
        resp = client.post(
            "/simulate",
            json={"num_bots": 2, "grid_size": 5, "steps": 3, "pheromone_params": {"trail_decya": 0.1}},
        )
        assert resp.status_code == 422
        assert any(
            error.get("loc") == ["body", "pheromone_params", "trail_decya"]
            and error.get("type") == "extra_forbidden"
            for error in resp.json()["detail"]
        )

    def test_simulate_run_stored(self):
        _RUNS.clear()
        with patch("backend.api_server.TumorNanobotModel", side_effect=_fake_model_factory):
            resp = client.post("/simulate", json={"num_bots": 2, "grid_size": 5, "steps": 3})
        run_id = resp.json()["run_id"]
        assert run_id in _RUNS

    def test_simulate_with_seed(self):
        with patch("backend.api_server.TumorNanobotModel", side_effect=_fake_model_factory):
            resp = client.post("/simulate", json={"num_bots": 2, "grid_size": 5, "steps": 2, "seed": 42})
        assert resp.status_code == 200


class TestGetRunEndpoint:
    def test_get_run_not_found(self):
        resp = client.get("/runs/nonexistent-run-id")
        assert resp.status_code == 404

    def test_get_run_returns_stored_result(self):
        _RUNS.clear()
        with patch("backend.api_server.TumorNanobotModel", side_effect=_fake_model_factory):
            post_resp = client.post("/simulate", json={"num_bots": 2, "grid_size": 5, "steps": 2})
        run_id = post_resp.json()["run_id"]
        get_resp = client.get(f"/runs/{run_id}")
        assert get_resp.status_code == 200
        data = get_resp.json()
        assert data["run_id"] == run_id
        assert data["status"] == "completed"
        assert "config" in data
        assert "metrics" in data

    def test_get_run_restores_persisted_result_after_memory_cache_clear(self, tmp_path):
        _RUNS.clear()
        temp_store = SQLiteRunStore(tmp_path / "api_runs.sqlite3")
        with (
            patch("backend.api_server.RUN_STORE", temp_store),
            patch("backend.api_server.TumorNanobotModel", side_effect=_fake_model_factory),
        ):
            post_resp = client.post("/simulate", json={"num_bots": 2, "grid_size": 5, "steps": 2})
            run_id = post_resp.json()["run_id"]

            _RUNS.clear()

            get_resp = client.get(f"/runs/{run_id}")

        assert get_resp.status_code == 200
        data = get_resp.json()
        assert data["run_id"] == run_id
        assert data["status"] == "completed"
        assert run_id in _RUNS
