"""Shared replay helpers for truthful verification and spot-checking."""

from __future__ import annotations

from typing import Any, Dict, Tuple

from backend.config import SimulationConfig


DEFAULT_REPLAY_STEPS = 10

SWARM_PROVENANCE_KEYS = {
    "agent_communication",
    "agent_messages",
    "communication_provenance",
    "communication_trace",
    "message_trace",
    "pheromone_provenance",
    "pheromone_summary",
    "pheromone_trace",
}


def first_present_value(config: Dict[str, Any], *keys: str, default: Any) -> Any:
    """Return the first present config value without treating explicit zero as missing."""
    for key in keys:
        if key in config:
            return config[key]
    return default


def extract_swarm_provenance(artifact: Dict[str, Any], runtime_metrics: Dict[str, Any]) -> Dict[str, Any]:
    """Preserve communication/pheromone trace fields already emitted by the simulator."""
    provenance: Dict[str, Any] = {}
    for source in (artifact, artifact.get("metrics") or {}, runtime_metrics):
        if not isinstance(source, dict):
            continue
        for key in SWARM_PROVENANCE_KEYS:
            if key in source:
                provenance[key] = source[key]
    return _validate_swarm_provenance(provenance)


def _validate_swarm_provenance(provenance: Dict[str, Any]) -> Dict[str, Any]:
    """Enforce structural invariants on extracted swarm provenance.

    Each provenance value must be a list or dict (a structured trace or summary),
    not a bare scalar.  A scalar provenance value — e.g. an integer step count
    mistakenly placed under ``communication_trace`` — would corrupt replay metadata
    and break downstream trust accounting.  Invalid entries are silently dropped
    so that replay never propagates malformed communication/pheromone claims.
    """
    valid: Dict[str, Any] = {}
    for key, value in provenance.items():
        if isinstance(value, (list, dict)):
            valid[key] = value
    return valid


def build_model(cfg: SimulationConfig) -> Any:
    """Import the heavy runtime only when replay actually executes."""
    from backend.runtime_factory import build_model as runtime_build_model

    return runtime_build_model(cfg)


def compute_metrics(model: Any) -> Dict[str, Any]:
    """Keep metric computation patchable without eager runtime imports."""
    from backend.runtime_factory import compute_metrics as runtime_compute_metrics

    return runtime_compute_metrics(model)


def normalize_simulation_config(config: Dict[str, Any]) -> SimulationConfig:
    num_bots = first_present_value(
        config,
        "num_bots",
        "bots",
        "nanobot_count",
        "n_nanobots",
        "n_bots",
        default=10,
    )
    steps = first_present_value(config, "steps", "max_steps", "n_steps", default=DEFAULT_REPLAY_STEPS)
    grid_size = config.get("grid_size")
    if grid_size is None:
        domain_size = float(config.get("domain_size", 600.0))
        voxel_size = float(config.get("voxel_size", 10.0))
        grid_size = max(2, int(round(domain_size / voxel_size)))
    return SimulationConfig(
        num_bots=int(float(num_bots)),
        grid_size=int(float(grid_size)),
        steps=int(float(steps)),
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

    normalized_config = normalized.model_dump()
    replay_metadata = {
        "original_config": config,
        "seed": normalized_config.get("seed"),
        "normalized_config": normalized_config,
        "swarm_provenance": extract_swarm_provenance(artifact, runtime_metrics),
    }

    return {
        "kill_rate": round(kill_rate, 2),
        "deliveries": int(runtime_metrics.get("total_deliveries", 0)),
        "cells_killed": int(runtime_metrics.get("cells_killed", 0)),
        "step_count": int(runtime_metrics.get("step_count", model.step_count)),
        "normalized_config": normalized_config,
        "replay_metadata": replay_metadata,
    }
