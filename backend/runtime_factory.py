"""Shared runtime helpers for building and executing nanobot simulations."""

from __future__ import annotations

import random
from typing import Any, Callable, Dict, Tuple

import numpy as np

from backend.config import SimulationConfig
from backend.nanobot_simulation import TumorNanobotModel


ModelFactory = Callable[..., TumorNanobotModel]


def seed_rng(seed: int | None) -> None:
    if seed is None:
        return
    random.seed(seed)
    np.random.seed(seed)


def build_model(cfg: SimulationConfig, model_factory: ModelFactory = TumorNanobotModel) -> TumorNanobotModel:
    seed_rng(cfg.seed)
    return model_factory(**cfg.to_model_kwargs())


def compute_metrics(model: TumorNanobotModel) -> Dict[str, Any]:
    metrics = dict(model.metrics)
    stats = model.geometry.get_tumor_statistics()
    total = max(1, stats.get("total_cells", 1))
    living = stats.get("living_cells", total)
    metrics["kill_rate"] = (total - living) / total
    metrics["step_count"] = model.step_count
    metrics["total_cells"] = total
    metrics["living_cells"] = living
    return metrics


def run_simulation(
    cfg: SimulationConfig,
    model_factory: ModelFactory = TumorNanobotModel,
) -> Tuple[TumorNanobotModel, Dict[str, Any]]:
    model = build_model(cfg, model_factory=model_factory)
    for _ in range(cfg.steps):
        model.step()
    return model, compute_metrics(model)
