"""Queen episodic policy wrapper for nanobot swarm coordination.

The Queen adjusts worker nanobot parameters every K simulation steps
based on swarm performance metrics. This enables adaptive strategy
that improves kill rate over fixed-policy baselines.

Configurable worker parameters:
- exploration_bias: tendency to explore vs exploit (0-1)
- trail_secretion_rate: pheromone trail deposit amount
- alarm_sensitivity: how strongly to avoid alarm zones
- search_radius: max distance to find targets
- drug_delivery_amount: µg per delivery step
"""

import numpy as np
from typing import Dict, Optional, List
from dataclasses import dataclass, field


@dataclass
class WorkerParams:
    """Configurable parameters for nanobot workers."""
    exploration_bias: float = 0.3       # 0 = pure exploit, 1 = pure explore
    trail_secretion_rate: float = 3.0   # pheromone units per delivery
    alarm_sensitivity: float = 0.5      # weight for alarm avoidance
    search_radius: float = 100.0        # µm, max target search distance
    drug_delivery_amount: float = 3.0   # µg per delivery step
    chemokine_weight: float = 1.2       # attraction to success signals
    speed_multiplier: float = 1.0       # movement speed scaling

    def to_dict(self) -> dict:
        return {
            "exploration_bias": self.exploration_bias,
            "trail_secretion_rate": self.trail_secretion_rate,
            "alarm_sensitivity": self.alarm_sensitivity,
            "search_radius": self.search_radius,
            "drug_delivery_amount": self.drug_delivery_amount,
            "chemokine_weight": self.chemokine_weight,
            "speed_multiplier": self.speed_multiplier,
        }

    @classmethod
    def from_dict(cls, d: dict) -> 'WorkerParams':
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


@dataclass
class SwarmMetrics:
    """Metrics snapshot for queen decision-making."""
    step: int = 0
    total_living_cells: int = 0
    cells_killed_this_epoch: int = 0
    deliveries_this_epoch: int = 0
    drug_delivered_this_epoch: float = 0.0
    avg_nanobot_payload: float = 0.0
    nanobots_searching: int = 0
    nanobots_delivering: int = 0
    nanobots_returning: int = 0
    hypoxic_cell_count: int = 0
    kill_rate: float = 0.0


class QueenPolicy:
    """Episodic policy that adjusts worker parameters every K steps.

    The queen observes swarm metrics at each epoch boundary and adjusts
    worker parameters to optimize kill rate. Uses a simple adaptive
    heuristic by default, with optional LLM override.
    """

    def __init__(
        self,
        epoch_interval: int = 50,
        initial_params: Optional[WorkerParams] = None,
    ):
        self.epoch_interval = epoch_interval
        self.current_params = initial_params or WorkerParams()
        self.epoch_count = 0
        self.history: List[Dict] = []
        self._prev_kills = 0
        self._prev_deliveries = 0

    def should_update(self, step: int) -> bool:
        """Check if it's time for a queen policy update."""
        return step > 0 and step % self.epoch_interval == 0

    def update(self, metrics: SwarmMetrics) -> WorkerParams:
        """Adjust worker parameters based on observed metrics.

        Returns updated WorkerParams.
        """
        self.epoch_count += 1
        kills_delta = metrics.cells_killed_this_epoch
        deliveries = metrics.deliveries_this_epoch

        # Record history for evaluation
        self.history.append({
            "epoch": self.epoch_count,
            "step": metrics.step,
            "kills": kills_delta,
            "deliveries": deliveries,
            "kill_rate": metrics.kill_rate,
            "params": self.current_params.to_dict(),
        })

        # Adaptive heuristic: adjust based on performance
        new_params = WorkerParams(**self.current_params.to_dict())

        # If kill rate is low, increase exploration and search radius
        if metrics.kill_rate < 0.2:
            new_params.exploration_bias = min(0.8, new_params.exploration_bias + 0.1)
            new_params.search_radius = min(200.0, new_params.search_radius + 20.0)

        # If kill rate is high, shift to exploitation
        elif metrics.kill_rate > 0.5:
            new_params.exploration_bias = max(0.1, new_params.exploration_bias - 0.1)

        # If many nanobots are searching (not finding targets), spread them out
        if metrics.nanobots_searching > len(self.history) * 0.7:
            new_params.alarm_sensitivity = max(0.2, new_params.alarm_sensitivity - 0.1)

        # If deliveries are low despite having targets, increase delivery amount
        if deliveries < 5 and metrics.total_living_cells > 10:
            new_params.drug_delivery_amount = min(5.0, new_params.drug_delivery_amount + 0.5)

        # If too many hypoxic cells remain, increase trail secretion to guide others
        if metrics.hypoxic_cell_count > 10:
            new_params.trail_secretion_rate = min(8.0, new_params.trail_secretion_rate + 1.0)

        self.current_params = new_params
        return new_params

    def get_epoch_summary(self) -> dict:
        """Return summary of policy performance across epochs."""
        if not self.history:
            return {"epochs": 0, "avg_kills_per_epoch": 0}

        kills = [h["kills"] for h in self.history]
        return {
            "epochs": self.epoch_count,
            "avg_kills_per_epoch": np.mean(kills) if kills else 0,
            "total_kills": sum(kills),
            "best_epoch_kills": max(kills) if kills else 0,
            "final_params": self.current_params.to_dict(),
        }


def apply_params_to_nanobot(nanobot, params: WorkerParams):
    """Apply queen-decided parameters to a worker nanobot."""
    nanobot.chemotaxis_weights["alarm"] = -params.alarm_sensitivity
    nanobot.chemotaxis_weights["trail"] = 0.8 * (1.0 - params.exploration_bias)
    nanobot.chemotaxis_weights["chemokine_signal"] = params.chemokine_weight
    nanobot.speed = 30.0 * params.speed_multiplier
