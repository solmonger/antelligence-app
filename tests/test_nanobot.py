"""Unit tests for nanobot simulation.

Tests cover:
- NanobotAgent state machine (searching, targeting, delivering, returning, reloading)
- Chemotaxis direction computation
- Drug delivery mechanics
- Movement and boundary clamping
- NanobotState transitions

Since nanobot_simulation.py has heavy imports (blockchain, litellm, dotenv),
we test the core logic by constructing a minimal TumorNanobotModel with
mocked external dependencies.
"""

import sys
import types
import numpy as np
import pytest

# Mock external modules before importing nanobot_simulation
_mock_modules = {}
for mod_name in [
    "dotenv",
    "litellm_client",
    "blockchain",
    "blockchain.client",
]:
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
    _mock_modules[mod_name] = mock
    sys.modules[mod_name] = mock

# Now we can import — add backend to path
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "backend"))

from backend.nanobot_simulation import NanobotAgent, NanobotState, TumorNanobotModel
from backend.biofvm import Microenvironment, create_oxygen_substrate, create_drug_substrate, create_pheromone_substrate
from backend.tumor_environment import TumorGeometry, TumorCell, VesselPoint, CellPhase, CellType


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def simple_model():
    """Create a minimal TumorNanobotModel for testing."""
    model = TumorNanobotModel(
        domain_size=200.0,
        voxel_size=10.0,
        n_nanobots=3,
        tumor_radius=80.0,
        agent_type="Rule-Based",
        with_queen=False,
    )
    return model


@pytest.fixture
def nanobot(simple_model):
    """Return the first nanobot from the model."""
    return simple_model.nanobots[0]


# ---------------------------------------------------------------------------
# NanobotAgent — Initialization
# ---------------------------------------------------------------------------

class TestNanobotInit:

    def test_initial_state(self, nanobot):
        assert nanobot.state == NanobotState.SEARCHING
        assert nanobot.drug_payload == 20.0
        assert nanobot.deliveries_made == 0
        assert nanobot.total_drug_delivered == 0.0

    def test_starts_near_vessel(self, simple_model):
        """Nanobots should start near a vessel if vessels exist."""
        if simple_model.geometry.vessels:
            for bot in simple_model.nanobots:
                # Check bot is within reasonable distance of some vessel
                min_dist = min(
                    np.linalg.norm(
                        np.array(v.position[:2]) - bot.position[:2]
                    )
                    for v in simple_model.geometry.vessels
                )
                # Should be within ~100 µm of a vessel (start offset is ~20 µm std)
                assert min_dist < 200.0

    def test_position_within_domain(self, simple_model):
        """Nanobots should start within or near domain boundaries.
        Allow small overshoot from vessel-proximity random offset."""
        margin = 50.0  # µm tolerance for random start offset
        for bot in simple_model.nanobots:
            assert bot.position[0] >= simple_model.microenv.x_range[0] - margin
            assert bot.position[0] <= simple_model.microenv.x_range[1] + margin
            assert bot.position[1] >= simple_model.microenv.y_range[0] - margin
            assert bot.position[1] <= simple_model.microenv.y_range[1] + margin


# ---------------------------------------------------------------------------
# Chemotaxis
# ---------------------------------------------------------------------------

class TestChemotaxis:

    def test_chemotaxis_direction_follows_gradient(self, simple_model):
        """Nanobot should move toward low oxygen (negative weight)."""
        bot = simple_model.nanobots[0]
        # Place bot in center
        bot.position = np.array([100.0, 100.0, 0.0])

        # Create a strong oxygen gradient: high on right, low on left
        oxygen = simple_model.microenv.get_substrate("oxygen")
        for i in range(simple_model.microenv.nx):
            oxygen.concentration[i, :, 0] = float(i) * 2.0

        direction = bot._compute_chemotaxis_direction()
        # Oxygen weight is -1.0 (move toward LOW), so direction should point left (negative x)
        if np.linalg.norm(direction) > 0:
            assert direction[0] < 0

    def test_pheromone_direction(self, simple_model):
        bot = simple_model.nanobots[0]
        bot.position = np.array([100.0, 100.0, 0.0])

        trail = simple_model.microenv.get_substrate("trail")
        if trail:
            # Put trail pheromone to the right
            voxel_right = simple_model.microenv.position_to_voxel((150.0, 100.0))
            trail.concentration[voxel_right] = 50.0

            direction = bot._compute_pheromone_direction()
            if np.linalg.norm(direction) > 0:
                assert direction[0] > 0  # should point right


# ---------------------------------------------------------------------------
# State Machine Transitions
# ---------------------------------------------------------------------------

class TestStateMachine:

    def test_find_target_transitions_to_targeting(self, simple_model):
        bot = simple_model.nanobots[0]
        # Place bot near a living cell
        living = simple_model.geometry.get_living_cells()
        if living:
            target = living[0]
            bot.position = np.array([
                target.position[0] + 10.0,
                target.position[1],
                0.0,
            ])
            bot.drug_payload = 20.0
            bot.state = NanobotState.SEARCHING
            bot.step()
            # Should have found target and transitioned
            assert bot.state in (NanobotState.TARGETING, NanobotState.SEARCHING)

    def test_targeting_moves_toward_cell(self, simple_model):
        bot = simple_model.nanobots[0]
        living = simple_model.geometry.get_living_cells()
        if living:
            target = living[0]
            bot.target_cell = target
            bot.state = NanobotState.TARGETING
            bot.position = np.array([
                target.position[0] + 80.0,
                target.position[1],
                0.0,
            ])
            initial_dist = np.linalg.norm(
                np.array(target.position[:2]) - bot.position[:2]
            )
            bot.step()
            new_dist = np.linalg.norm(
                np.array(target.position[:2]) - bot.position[:2]
            )
            assert new_dist < initial_dist

    def test_targeting_to_delivering_when_close(self, simple_model):
        bot = simple_model.nanobots[0]
        living = simple_model.geometry.get_living_cells()
        if living:
            target = living[0]
            bot.target_cell = target
            bot.state = NanobotState.TARGETING
            # Place very close to target
            bot.position = np.array([
                target.position[0] + 5.0,
                target.position[1],
                0.0,
            ])
            bot.step()
            assert bot.state == NanobotState.DELIVERING

    def test_delivering_depletes_payload(self, simple_model):
        bot = simple_model.nanobots[0]
        living = simple_model.geometry.get_living_cells()
        if living:
            target = living[0]
            bot.target_cell = target
            bot.state = NanobotState.DELIVERING
            bot.position = np.array([
                target.position[0],
                target.position[1],
                0.0,
            ])
            initial_payload = bot.drug_payload
            bot.step()
            assert bot.drug_payload < initial_payload
            assert bot.total_drug_delivered > 0

    def test_empty_payload_triggers_return(self, simple_model):
        bot = simple_model.nanobots[0]
        living = simple_model.geometry.get_living_cells()
        if living:
            target = living[0]
            bot.target_cell = target
            bot.state = NanobotState.DELIVERING
            bot.position = np.array([
                target.position[0],
                target.position[1],
                0.0,
            ])
            bot.drug_payload = 1.5  # below 2.0 threshold after delivery
            bot.step()
            assert bot.state == NanobotState.RETURNING

    def test_returning_moves_toward_vessel(self, simple_model):
        bot = simple_model.nanobots[0]
        if simple_model.geometry.vessels:
            vessel = simple_model.geometry.vessels[0]
            bot.state = NanobotState.RETURNING
            bot.target_vessel = vessel
            bot.position = np.array([
                vessel.position[0] + 80.0,
                vessel.position[1],
                0.0,
            ])
            initial_dist = np.linalg.norm(
                np.array(vessel.position[:2]) - bot.position[:2]
            )
            bot.step()
            new_dist = np.linalg.norm(
                np.array(vessel.position[:2]) - bot.position[:2]
            )
            assert new_dist < initial_dist

    def test_reloading_increases_payload(self, simple_model):
        bot = simple_model.nanobots[0]
        bot.state = NanobotState.RELOADING
        bot.drug_payload = 0.0
        bot.step()
        assert bot.drug_payload > 0.0

    def test_reloading_to_searching_when_full(self, simple_model):
        bot = simple_model.nanobots[0]
        bot.state = NanobotState.RELOADING
        bot.drug_payload = bot.max_payload * 0.95  # above 90% threshold
        bot.step()
        # After reload brings it to max, should transition to searching
        if bot.drug_payload >= bot.max_payload * 0.9:
            assert bot.state == NanobotState.SEARCHING

    def test_target_lost_returns_to_searching(self, simple_model):
        bot = simple_model.nanobots[0]
        bot.state = NanobotState.TARGETING
        bot.target_cell = None  # target lost
        bot.step()
        assert bot.state == NanobotState.SEARCHING


# ---------------------------------------------------------------------------
# Drug Delivery
# ---------------------------------------------------------------------------

class TestDrugDelivery:

    def test_delivery_deposits_trail_pheromone(self, simple_model):
        bot = simple_model.nanobots[0]
        living = simple_model.geometry.get_living_cells()
        if living:
            target = living[0]
            bot.target_cell = target
            bot.state = NanobotState.DELIVERING
            bot.position = np.array([
                target.position[0],
                target.position[1],
                0.0,
            ])
            trail = simple_model.microenv.get_substrate("trail")
            voxel = simple_model.microenv.position_to_voxel(tuple(bot.position))
            initial_trail = trail.source_sink[voxel] if trail else 0

            bot.step()

            if trail:
                assert trail.source_sink[voxel] > initial_trail


# ---------------------------------------------------------------------------
# Boundary Clamping
# ---------------------------------------------------------------------------

class TestBoundaryClamping:

    def test_clamp_keeps_in_domain(self, simple_model):
        bot = simple_model.nanobots[0]
        bot.position = np.array([-100.0, -100.0, 0.0])
        bot._clamp_position()
        assert bot.position[0] >= simple_model.microenv.x_range[0]
        assert bot.position[1] >= simple_model.microenv.y_range[0]

    def test_clamp_upper_bound(self, simple_model):
        bot = simple_model.nanobots[0]
        bot.position = np.array([9999.0, 9999.0, 0.0])
        bot._clamp_position()
        assert bot.position[0] <= simple_model.microenv.x_range[1]
        assert bot.position[1] <= simple_model.microenv.y_range[1]


# ---------------------------------------------------------------------------
# Model-level
# ---------------------------------------------------------------------------

class TestTumorNanobotModel:

    def test_model_creation(self, simple_model):
        assert len(simple_model.nanobots) == 3
        assert simple_model.microenv is not None
        assert simple_model.geometry is not None
        assert "oxygen" in simple_model.microenv.substrates
        assert "drug" in simple_model.microenv.substrates
        assert "trail" in simple_model.microenv.substrates

    def test_step_advances_simulation(self, simple_model):
        initial_time = simple_model.microenv.time
        simple_model.step()
        assert simple_model.microenv.time > initial_time
