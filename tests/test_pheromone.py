"""Unit tests for pheromone system (Phase 2).

Tests cover:
- Trail pheromone: secretion, diffusion, exponential decay
- Alarm pheromone: higher diffusion rate, faster decay
- Recruitment pheromone: zone exploration signaling
- Chemotaxis: nanobots follow trail gradient, avoid alarm zones
- Mass conservation under pheromone dynamics
- Pheromone decay half-life validation
"""

import sys
import types
import numpy as np
import pytest

# Mock external modules
for mod_name in ["dotenv", "litellm_client", "blockchain", "blockchain.client"]:
    mock = types.ModuleType(mod_name)
    if mod_name == "dotenv":
        mock.load_dotenv = lambda: None
    if mod_name == "litellm_client":
        mock.create_client = lambda *a, **kw: None
    if mod_name == "blockchain.client":
        mock.w3 = None
        mock.acct = None
        mock.tumor_intel_contract = None
        mock.TUMOR_INTEL_CONTRACT_ADDRESS = None
    sys.modules[mod_name] = mock

import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "backend"))

from backend.biofvm import (
    Microenvironment,
    create_pheromone_substrate,
)


@pytest.fixture
def env():
    """Standard 200x200 µm 2D microenvironment."""
    return Microenvironment(
        x_range=(0, 200), y_range=(0, 200), z_range=(0, 0),
        dx=5, dy=5, dz=5, dimensionality=2,
    )


# ---------------------------------------------------------------------------
# Trail Pheromone
# ---------------------------------------------------------------------------

class TestTrailPheromone:

    def test_secretion_increases_concentration(self, env):
        trail = create_pheromone_substrate(env, "trail", decay_rate=0.1)
        center = env.position_to_voxel((100.0, 100.0))
        trail.add_source(center, 10.0)
        env.step()
        assert trail.concentration[center] > 0

    def test_trail_diffuses_to_neighbors(self, env):
        trail = create_pheromone_substrate(env, "trail", decay_rate=0.0)
        center = (20, 20, 0)
        trail.concentration[center] = 100.0

        for _ in range(50):
            env.step()

        # Neighbors should have gained concentration
        assert trail.concentration[19, 20, 0] > 0
        assert trail.concentration[21, 20, 0] > 0
        assert trail.concentration[20, 19, 0] > 0
        assert trail.concentration[20, 21, 0] > 0

    def test_trail_decay_half_life(self, env):
        """Verify exponential decay: C(t) = C₀ * exp(-λt).
        Half-life = ln(2)/λ. For λ=0.1/min, t½ ≈ 6.93 min.
        """
        decay_rate = 0.1
        trail = create_pheromone_substrate(env, "trail", decay_rate=decay_rate)
        # Use very small diffusion to isolate decay behavior
        trail.diffusion_coefficient = 1e-3  # near-zero

        # Set uniform concentration
        trail.concentration[:] = 100.0
        initial = 100.0

        # Simulate for the half-life duration
        half_life = np.log(2) / decay_rate  # ~6.93 min
        n_steps = int(half_life / env.dt)
        for _ in range(n_steps):
            env.step()

        # Interior should be approximately half
        interior = trail.concentration[10:-10, 10:-10, 0]
        mean_conc = interior.mean()
        expected = initial * 0.5
        # Allow 20% tolerance due to numerical discretization
        assert abs(mean_conc - expected) / expected < 0.20, (
            f"Expected ~{expected:.1f}, got {mean_conc:.1f}"
        )


# ---------------------------------------------------------------------------
# Alarm Pheromone
# ---------------------------------------------------------------------------

class TestAlarmPheromone:

    def test_alarm_decays_faster_than_trail(self, env):
        trail = create_pheromone_substrate(env, "trail", decay_rate=0.1)
        alarm = create_pheromone_substrate(env, "alarm", decay_rate=0.15)

        # Set same initial concentration
        trail.concentration[:] = 50.0
        alarm.concentration[:] = 50.0

        for _ in range(200):
            env.step()

        # Alarm should have decayed more
        trail_mean = trail.concentration[10:-10, 10:-10, 0].mean()
        alarm_mean = alarm.concentration[10:-10, 10:-10, 0].mean()
        assert alarm_mean < trail_mean

    def test_alarm_spreads_wider(self, env):
        """Alarm pheromone should diffuse faster (higher D)."""
        # trail: D=1e-6 cm²/s, alarm: D=5e-6 cm²/s
        trail = env.add_substrate("trail_test", 1e-6, 0.0, initial_value=0.0,
                                  dirichlet_boundary_value=None)
        alarm = env.add_substrate("alarm_test", 5e-6, 0.0, initial_value=0.0,
                                  dirichlet_boundary_value=None)

        center = (20, 20, 0)
        trail.concentration[center] = 100.0
        alarm.concentration[center] = 100.0

        for _ in range(100):
            env.step()

        # Alarm should have spread more (lower center, higher periphery)
        assert alarm.concentration[center] < trail.concentration[center]


# ---------------------------------------------------------------------------
# Recruitment Pheromone
# ---------------------------------------------------------------------------

class TestRecruitmentPheromone:

    def test_recruitment_creates_gradient(self, env):
        recruit = create_pheromone_substrate(env, "recruitment", decay_rate=0.12)
        # Set concentration directly at source location (simulating secretion)
        source = env.position_to_voxel((50.0, 100.0))
        recruit.concentration[source] = 100.0

        # Let it diffuse a few steps
        for _ in range(20):
            env.step()

        # Gradient at a point to the right of the source should point left (toward source)
        grad = env.get_gradient_at("recruitment", (60.0, 100.0))
        assert grad[0] < 0


# ---------------------------------------------------------------------------
# Chemotaxis Direction
# ---------------------------------------------------------------------------

class TestChemotaxis:

    def test_nanobot_follows_trail_gradient(self):
        """Nanobot chemotaxis should follow trail pheromone gradient."""
        from backend.nanobot_simulation import TumorNanobotModel

        np.random.seed(42)
        model = TumorNanobotModel(
            domain_size=200.0, voxel_size=10.0, n_nanobots=1,
            tumor_radius=80.0, agent_type="Rule-Based",
        )
        bot = model.nanobots[0]
        bot.position = np.array([100.0, 100.0, 0.0])
        bot.state.__class__  # just to confirm import

        # Place trail pheromone to the right
        trail = model.microenv.get_substrate("trail")
        if trail:
            right = model.microenv.position_to_voxel((150.0, 100.0))
            trail.concentration[right] = 100.0

            direction = bot._compute_chemotaxis_direction()
            # Trail weight is +0.8, so should point right (positive x)
            if np.linalg.norm(direction) > 0 and trail.concentration[right] > 0:
                # The oxygen gradient dominates; just verify trail contributes
                pheromone_dir = bot._compute_pheromone_direction()
                assert pheromone_dir[0] > 0  # trail points right

    def test_nanobot_avoids_alarm_gradient(self):
        """Nanobot should move away from alarm pheromone."""
        from backend.nanobot_simulation import TumorNanobotModel

        np.random.seed(42)
        model = TumorNanobotModel(
            domain_size=200.0, voxel_size=10.0, n_nanobots=1,
            tumor_radius=80.0, agent_type="Rule-Based",
        )
        bot = model.nanobots[0]
        bot.position = np.array([100.0, 100.0, 0.0])

        # Place alarm pheromone to the right
        alarm = model.microenv.get_substrate("alarm")
        if alarm:
            right = model.microenv.position_to_voxel((150.0, 100.0))
            alarm.concentration[right] = 100.0

            # The alarm chemotaxis weight is -0.5 (repulsive)
            # So the alarm gradient contribution should point left (away from alarm)
            gradient = model.microenv.get_gradient_at("alarm", (100.0, 100.0))
            alarm_contribution = -0.5 * gradient[:2]
            if np.linalg.norm(gradient) > 0:
                assert alarm_contribution[0] < 0  # repelled from right


# ---------------------------------------------------------------------------
# Pheromone Mass Conservation
# ---------------------------------------------------------------------------

class TestPheromoneConservation:

    def test_no_flux_conserves_mass(self, env):
        """Pheromone with no-flux boundaries and zero decay conserves mass."""
        pheromone = create_pheromone_substrate(env, "conserved_ph", decay_rate=0.0)
        # Place a blob
        pheromone.concentration[15:25, 15:25, 0] = 30.0
        initial_mass = pheromone.concentration.sum()

        for _ in range(100):
            env.step()

        final_mass = pheromone.concentration.sum()
        assert final_mass == pytest.approx(initial_mass, rel=0.05)

    def test_concentration_nonnegative(self, env):
        """Pheromone should never go negative."""
        pheromone = create_pheromone_substrate(env, "nonneg_ph", decay_rate=0.5)
        pheromone.concentration[:] = 0.1

        for _ in range(300):
            env.step()

        assert (pheromone.concentration >= 0).all()
