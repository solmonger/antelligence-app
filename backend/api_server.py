"""api_server.py — lightweight FastAPI simulation API.

Endpoints
---------
POST /simulate      Run a simulation; returns run_id + metrics immediately.
GET  /runs/{run_id} Retrieve stored results for a previous run.
GET  /health        Liveness check.

Usage
-----
    uvicorn backend.api_server:app --reload
    # or via pyproject.toml script: antelligence-api
"""

from __future__ import annotations

import sys
import os
import uuid
from pathlib import Path
from typing import Any, Dict, Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, ConfigDict, ValidationError

# Ensure backend package is importable when running from repo root.
_backend_dir = os.path.dirname(os.path.abspath(__file__))
if _backend_dir not in sys.path:
    sys.path.insert(0, _backend_dir)

from backend.config import PheromoneParams, SimulationConfig  # noqa: E402
from backend.run_store import SQLiteRunStore  # noqa: E402
from backend.runtime_factory import run_simulation  # noqa: E402
from chain.proof_adapter import create_proof_bundle  # noqa: E402

# Import simulation model at module level so tests can patch backend.api_server.TumorNanobotModel
try:
    from nanobot_simulation import TumorNanobotModel
except ImportError:
    try:
        from backend.nanobot_simulation import TumorNanobotModel  # type: ignore[no-redef]
    except ImportError:
        TumorNanobotModel = None  # type: ignore[assignment,misc]

app = FastAPI(
    title="Antelligence Simulation API",
    description="REST interface for the nanobot tumor-simulation engine.",
    version="1.0.0",
)

# ---------------------------------------------------------------------------
# Run storage
# ---------------------------------------------------------------------------
_RUNS: Dict[str, Dict[str, Any]] = {}
_DEFAULT_DB_PATH = Path(os.environ.get("ANTELLIGENCE_RUN_DB", Path(_backend_dir).parent / "data" / "api_runs.sqlite3"))
RUN_STORE = SQLiteRunStore(_DEFAULT_DB_PATH)


# ---------------------------------------------------------------------------
# Request / response models
# ---------------------------------------------------------------------------


class SimulateRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    num_bots: int = 10
    grid_size: int = 60
    steps: int = 100
    queen_enabled: bool = False
    seed: Optional[int] = None
    pheromone_params: Optional[PheromoneParams] = None


class SimulateResponse(BaseModel):
    run_id: str
    status: str
    metrics: Dict[str, Any]
    provenance: Dict[str, Any]


class RunResponse(BaseModel):
    run_id: str
    status: str
    config: Dict[str, Any]
    metrics: Dict[str, Any]
    provenance: Optional[Dict[str, Any]] = None


class HealthResponse(BaseModel):
    status: str
    version: str


def _config_validation_detail(exc: ValidationError) -> list[dict[str, Any]]:
    """Return FastAPI-style validation errors for SimulationConfig failures."""
    detail: list[dict[str, Any]] = []
    for error in exc.errors():
        item = dict(error)
        item["loc"] = ["body", *item.get("loc", ())]
        if "ctx" in item:
            item["ctx"] = {key: str(value) for key, value in item["ctx"].items()}
        detail.append(item)
    return detail


def _provenance_config(cfg: SimulationConfig) -> Dict[str, Any]:
    """Return simulation config normalized for proof/public-value metadata."""
    model_kwargs = cfg.to_model_kwargs()
    config = cfg.model_dump()
    config.update(
        {
            "nanobot_count": cfg.num_bots,
            "tumor_radius": int(model_kwargs["tumor_radius"]),
        }
    )
    return config


def _build_run_provenance(run_id: str, cfg: SimulationConfig, metrics: Dict[str, Any]) -> Dict[str, Any]:
    """Build machine-readable proof/provenance metadata for an API run."""
    bundle = create_proof_bundle(_provenance_config(cfg), metrics, run_id=run_id)
    return {
        "run_id": run_id,
        "trust_tier": bundle["trust_tier"],
        "verification_status": bundle["verification_status"],
        "proof_lifecycle": bundle["proof_lifecycle"],
        "onchain": bundle["onchain"],
        "proof_bundle": bundle["proof_bundle"],
    }


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------


@app.get("/health", response_model=HealthResponse, tags=["ops"])
def health() -> HealthResponse:
    """Return service liveness status."""
    return HealthResponse(status="ok", version=app.version)


@app.post("/simulate", response_model=SimulateResponse, tags=["simulation"])
def simulate(request: SimulateRequest) -> SimulateResponse:
    """Run a simulation synchronously and store the results.

    Returns a ``run_id`` that can later be used with ``GET /runs/{run_id}``.
    """
    # Build a validated config (raises HTTP 422 on bad input automatically).
    pheromone_kwargs = request.pheromone_params or {}
    try:
        cfg = SimulationConfig(
            num_bots=request.num_bots,
            grid_size=request.grid_size,
            steps=request.steps,
            queen_enabled=request.queen_enabled,
            seed=request.seed,
            pheromone_params=pheromone_kwargs,
        )
    except ValidationError as exc:
        raise HTTPException(status_code=422, detail=_config_validation_detail(exc)) from exc

    try:
        _, metrics = run_simulation(cfg, model_factory=TumorNanobotModel)

    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=500, detail=f"Simulation error: {exc}") from exc

    run_id = str(uuid.uuid4())
    provenance = _build_run_provenance(run_id, cfg, metrics)
    entry = {
        "status": "completed",
        "config": cfg.model_dump(),
        "metrics": metrics,
        "provenance": provenance,
    }
    _RUNS[run_id] = entry
    RUN_STORE.save_run(run_id, entry["status"], entry["config"], entry["metrics"], entry["provenance"])

    return SimulateResponse(run_id=run_id, status="completed", metrics=metrics, provenance=provenance)


@app.get("/runs/{run_id}", response_model=RunResponse, tags=["simulation"])
def get_run(run_id: str) -> RunResponse:
    """Retrieve stored results for a previous simulation run."""
    entry = _RUNS.get(run_id)
    if entry is None:
        persisted = RUN_STORE.get_run(run_id)
        if persisted is not None:
            entry = {
                "status": persisted["status"],
                "config": persisted["config"],
                "metrics": persisted["metrics"],
                "provenance": persisted.get("provenance"),
            }
            _RUNS[run_id] = entry
    if entry is None:
        raise HTTPException(status_code=404, detail=f"Run '{run_id}' not found.")
    return RunResponse(
        run_id=run_id,
        status=entry["status"],
        config=entry["config"],
        metrics=entry["metrics"],
        provenance=entry.get("provenance"),
    )


# ---------------------------------------------------------------------------
# Entry-point for pyproject.toml script
# ---------------------------------------------------------------------------


def main() -> None:
    import uvicorn

    uvicorn.run("backend.api_server:app", host="0.0.0.0", port=8001, reload=False)


if __name__ == "__main__":
    main()
