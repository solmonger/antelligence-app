"""Shared replay helpers for truthful verification and spot-checking."""

from __future__ import annotations

from typing import Any, Dict, Tuple

from backend.config import SimulationConfig


DEFAULT_REPLAY_STEPS = 10


def build_model(cfg: SimulationConfig) -> Any:
    """Import the heavy runtime only when replay actually executes."""
    from backend.runtime_factory import build_model as runtime_build_model

    return runtime_build_model(cfg)


def compute_metrics(model: Any) -> Dict[str, Any]:
    """Keep metric computation patchable without eager runtime imports."""
    from backend.runtime_factory import compute_metrics as runtime_compute_metrics

    return runtime_compute_metrics(model)


def normalize_simulation_config(config: Dict[str, Any]) -> SimulationConfig:
    num_bots = (
        config.get("num_bots")
        or config.get("nanobot_count")
        or config.get("n_nanobots")
        or config.get("n_bots")
        or 10
    )
    steps = config.get("steps") or config.get("max_steps") or config.get("n_steps") or DEFAULT_REPLAY_STEPS
    grid_size = config.get("grid_size")
    if grid_size is None:
        domain_size = float(config.get("domain_size", 600.0))
        voxel_size = float(config.get("voxel_size", 10.0))
        grid_size = max(2, int(round(domain_size / voxel_size)))
    return SimulationConfig(
        num_bots=int(num_bots),
        grid_size=int(grid_size),
        steps=int(steps),
        queen_enabled=bool(
            config.get("queen_enabled", config.get("with_queen", config.get("use_queen", False)))
        ),
        seed=config.get("seed"),
        pheromone_params=config.get("pheromone_params") or {},
    )


def build_model_from_config(config: Dict[str, Any]) -> Tuple[SimulationConfig, Any]:
    normalized = normalize_simulation_config(config)
    model = build_model(normalized)
    return normalized, model


def replay_artifact_metrics(artifact: Dict[str, Any]) -> Dict[str, Any]:
    config = artifact.get("config") or {}
    normalized, model = build_model_from_config(config)
    for _ in range(normalized.steps):
        model.step()

    runtime_metrics = compute_metrics(model)
    kill_rate = runtime_metrics.get("kill_rate", 0) * 100

    return {
        "kill_rate": round(kill_rate, 2),
        "deliveries": int(runtime_metrics.get("total_deliveries", 0)),
        "cells_killed": int(runtime_metrics.get("cells_killed", 0)),
        "step_count": int(runtime_metrics.get("step_count", model.step_count)),
        "normalized_config": normalized.model_dump(),
    }
