"""Gymnasium environment wrapper for TumorNanobotModel.

Wraps the antelligence tumor simulation as a standard Gymnasium environment
for reinforcement learning. The RL agent controls nanobot swarm parameters
(queen policy) and is rewarded for tumor cell kills.

Observation space: cell counts, pheromone levels, nanobot states
Action space: WorkerParams adjustments (continuous)

Usage:
    import gymnasium as gym
    from backend.gym_env import TumorNanobotEnv

    env = TumorNanobotEnv(domain_size=400, n_nanobots=10, tumor_radius=150)
    obs, info = env.reset()
    for _ in range(1000):
        action = env.action_space.sample()  # or RL policy
        obs, reward, terminated, truncated, info = env.step(action)
"""

import gymnasium as gym
import numpy as np
from gymnasium import spaces

# Lazy imports to avoid circular deps and heavy module-level loads
_sim_imported = False


def _ensure_imports():
    global _sim_imported
    if _sim_imported:
        return
    import sys, os, types
    for mod_name in ["dotenv", "litellm_client", "blockchain", "blockchain.client"]:
        if mod_name not in sys.modules:
            mock = types.ModuleType(mod_name)
            if mod_name == "dotenv":
                mock.load_dotenv = lambda *a, **kw: None
            if mod_name == "litellm_client":
                mock.create_client = lambda *a, **kw: None
            if mod_name == "blockchain.client":
                mock.w3 = None
                mock.acct = None
                mock.tumor_intel_contract = None
                mock.TUMOR_INTEL_CONTRACT_ADDRESS = None
            sys.modules[mod_name] = mock
    sys.path.insert(0, os.path.join(os.path.dirname(__file__)))
    _sim_imported = True


class TumorNanobotEnv(gym.Env):
    """Gymnasium environment for nanobot tumor treatment.

    The agent acts as the Queen, adjusting worker nanobot parameters
    every `epoch_interval` simulation steps. The goal is to maximize
    tumor cell kills while minimizing drug usage.
    """

    metadata = {"render_modes": ["human"]}

    def __init__(
        self,
        domain_size: float = 400.0,
        n_nanobots: int = 10,
        tumor_radius: float = 150.0,
        max_steps: int = 300,
        epoch_interval: int = 50,
        seed: int = 42,
    ):
        super().__init__()
        _ensure_imports()

        self.domain_size = domain_size
        self.n_nanobots = n_nanobots
        self.tumor_radius = tumor_radius
        self.max_steps = max_steps
        self.epoch_interval = epoch_interval
        self._seed = seed

        # Observation space: 10 features
        # [living_cells_frac, hypoxic_frac, kill_rate, avg_payload,
        #  searching_frac, delivering_frac, returning_frac,
        #  trail_mean, alarm_mean, drug_mean]
        self.observation_space = spaces.Box(
            low=0.0, high=1.0, shape=(10,), dtype=np.float32
        )

        # Action space: 5 continuous params (normalized to [0, 1])
        # [exploration_bias, trail_secretion_rate, alarm_sensitivity,
        #  search_radius_frac, drug_delivery_frac]
        self.action_space = spaces.Box(
            low=0.0, high=1.0, shape=(5,), dtype=np.float32
        )

        self.model = None
        self.step_count = 0
        self.initial_alive = 0
        self.prev_kills = 0

    def _get_obs(self) -> np.ndarray:
        """Extract observation from current simulation state."""
        from nanobot_simulation import NanobotState
        from tumor_environment import CellPhase

        living = sum(1 for c in self.model.geometry.tumor_cells if c.is_alive)
        hypoxic = sum(1 for c in self.model.geometry.get_living_cells()
                      if c.phase == CellPhase.HYPOXIC)
        kills = self.initial_alive - living
        kill_rate = kills / max(self.initial_alive, 1)
        avg_payload = np.mean([b.drug_payload / b.max_payload for b in self.model.nanobots])

        state_counts = {}
        for bot in self.model.nanobots:
            state_counts[bot.state] = state_counts.get(bot.state, 0) + 1
        n = len(self.model.nanobots)

        trail = self.model.microenv.get_substrate("trail")
        alarm = self.model.microenv.get_substrate("alarm")
        drug = self.model.microenv.get_substrate("drug")

        trail_mean = float(np.mean(trail.concentration)) / 10.0 if trail else 0
        alarm_mean = float(np.mean(alarm.concentration)) / 10.0 if alarm else 0
        drug_mean = float(np.mean(drug.concentration)) / 10.0 if drug else 0

        return np.array([
            living / max(self.initial_alive, 1),
            hypoxic / max(living, 1),
            kill_rate,
            avg_payload,
            state_counts.get(NanobotState.SEARCHING, 0) / n,
            state_counts.get(NanobotState.DELIVERING, 0) / n,
            state_counts.get(NanobotState.RETURNING, 0) / n,
            np.clip(trail_mean, 0, 1),
            np.clip(alarm_mean, 0, 1),
            np.clip(drug_mean, 0, 1),
        ], dtype=np.float32)

    def _apply_action(self, action: np.ndarray):
        """Apply RL action to nanobot worker parameters."""
        from queen_policy import WorkerParams, apply_params_to_nanobot

        params = WorkerParams(
            exploration_bias=float(action[0]),
            trail_secretion_rate=float(action[1]) * 8.0,   # scale to [0, 8]
            alarm_sensitivity=float(action[2]),
            search_radius=50.0 + float(action[3]) * 150.0, # [50, 200]
            drug_delivery_amount=1.0 + float(action[4]) * 4.0,  # [1, 5]
        )
        for bot in self.model.nanobots:
            apply_params_to_nanobot(bot, params)

    def reset(self, seed=None, options=None):
        """Reset simulation to initial state."""
        from nanobot_simulation import TumorNanobotModel

        if seed is not None:
            self._seed = seed
        np.random.seed(self._seed)

        self.model = TumorNanobotModel(
            domain_size=self.domain_size,
            voxel_size=10.0,
            n_nanobots=self.n_nanobots,
            tumor_radius=self.tumor_radius,
            agent_type="Rule-Based",
            with_queen=False,
        )
        self.initial_alive = sum(1 for c in self.model.geometry.tumor_cells if c.is_alive)
        self.step_count = 0
        self.prev_kills = 0

        return self._get_obs(), {}

    def step(self, action):
        """Run epoch_interval sim steps with given action, return reward."""
        self._apply_action(action)

        for _ in range(self.epoch_interval):
            self.model.step()
            self.step_count += 1

        obs = self._get_obs()

        # Reward: kills this epoch (normalized)
        living = sum(1 for c in self.model.geometry.tumor_cells if c.is_alive)
        current_kills = self.initial_alive - living
        new_kills = current_kills - self.prev_kills
        self.prev_kills = current_kills

        reward = new_kills / max(self.initial_alive, 1)

        # Bonus for total kill rate
        total_kill_rate = current_kills / max(self.initial_alive, 1)
        reward += total_kill_rate * 0.1

        terminated = (living == 0)  # all cells killed
        truncated = (self.step_count >= self.max_steps)

        info = {
            "kill_rate": total_kill_rate,
            "living": living,
            "kills": current_kills,
            "step": self.step_count,
        }

        return obs, float(reward), terminated, truncated, info
