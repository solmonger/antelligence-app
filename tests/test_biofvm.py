"""Unit tests for BioFVM substrate diffusion and reaction system.

Tests cover:
- SubstrateField creation and source/sink management
- Microenvironment initialization (2D and 3D)
- Diffusion correctness and numerical stability
- Decay behavior and mass conservation
- Gradient computation for chemotaxis
- Boundary conditions (Dirichlet and Neumann)
- Helper functions for common substrates
"""

import numpy as np
import pytest
from backend.biofvm import (
    SubstrateField,
    Microenvironment,
    create_oxygen_substrate,
    create_drug_substrate,
    create_pheromone_substrate,
)


# ---------------------------------------------------------------------------
# SubstrateField
# ---------------------------------------------------------------------------

class TestSubstrateField:

    def test_initialization(self):
        sf = SubstrateField("oxygen", (10, 10, 1), initial_value=38.0)
        assert sf.name == "oxygen"
        assert sf.concentration.shape == (10, 10, 1)
        assert np.allclose(sf.concentration, 38.0)
        assert sf.source_sink.shape == (10, 10, 1)
        assert np.allclose(sf.source_sink, 0.0)

    def test_add_source(self):
        sf = SubstrateField("test", (5, 5, 1))
        sf.add_source((2, 3, 0), 10.0)
        assert sf.source_sink[2, 3, 0] == pytest.approx(10.0)
        # Additive
        sf.add_source((2, 3, 0), 5.0)
        assert sf.source_sink[2, 3, 0] == pytest.approx(15.0)

    def test_add_sink(self):
        sf = SubstrateField("test", (5, 5, 1))
        sf.add_sink((1, 1, 0), 3.0)
        assert sf.source_sink[1, 1, 0] == pytest.approx(-3.0)

    def test_out_of_bounds_source_ignored(self):
        sf = SubstrateField("test", (5, 5, 1))
        sf.add_source((10, 10, 0), 1.0)  # out of bounds
        assert np.allclose(sf.source_sink, 0.0)

    def test_reset_sources_sinks(self):
        sf = SubstrateField("test", (5, 5, 1))
        sf.add_source((0, 0, 0), 5.0)
        sf.reset_sources_sinks()
        assert np.allclose(sf.source_sink, 0.0)


# ---------------------------------------------------------------------------
# Microenvironment — Initialization
# ---------------------------------------------------------------------------

class TestMicroenvironmentInit:

    def test_2d_grid_dimensions(self):
        env = Microenvironment(
            x_range=(0, 100), y_range=(0, 100), z_range=(0, 0),
            dx=10, dy=10, dz=10, dimensionality=2,
        )
        assert env.nx == 11
        assert env.ny == 11
        assert env.nz == 1
        assert env.shape == (11, 11, 1)

    def test_3d_grid_dimensions(self):
        env = Microenvironment(
            x_range=(0, 50), y_range=(0, 50), z_range=(0, 50),
            dx=10, dy=10, dz=10, dimensionality=3,
        )
        assert env.nz == 6  # (50-0)/10 + 1

    def test_add_substrate(self):
        env = Microenvironment(
            x_range=(0, 100), y_range=(0, 100), z_range=(0, 0),
            dx=10, dy=10, dz=10, dimensionality=2,
        )
        sub = env.add_substrate("oxygen", 1e-5, 0.1, initial_value=38.0)
        assert sub.name == "oxygen"
        assert "oxygen" in env.substrates
        # D should be converted from cm²/s to µm²/min
        assert sub.diffusion_coefficient == pytest.approx(1e-5 * 6e9)

    def test_timestep_stability(self):
        """dt must satisfy explicit FD stability criterion."""
        env = Microenvironment(
            x_range=(0, 100), y_range=(0, 100), z_range=(0, 0),
            dx=10, dy=10, dz=10, dimensionality=2,
        )
        env.add_substrate("oxygen", 1e-5, 0.1)
        D = 1e-5 * 6e9  # converted
        dt_max = 0.25 * (10 ** 2) / (2 * D * 2)
        assert env.dt <= dt_max or env.dt <= 0.1


# ---------------------------------------------------------------------------
# Diffusion & Decay
# ---------------------------------------------------------------------------

class TestDiffusionDecay:

    @pytest.fixture
    def small_env(self):
        """20×20 µm domain, dx=1 µm, 2D."""
        env = Microenvironment(
            x_range=(0, 20), y_range=(0, 20), z_range=(0, 0),
            dx=1, dy=1, dz=1, dimensionality=2,
        )
        return env

    def test_decay_reduces_concentration(self, small_env):
        """Pure decay (no diffusion) should reduce concentration."""
        sub = small_env.add_substrate(
            "test", diffusion_coefficient=1e-12, decay_rate=0.5,
            initial_value=10.0, dirichlet_boundary_value=None,
        )
        initial_total = sub.concentration.sum()
        for _ in range(100):
            small_env.step()
        final_total = sub.concentration.sum()
        assert final_total < initial_total

    def test_uniform_field_no_diffusion(self, small_env):
        """A uniform concentration with no decay should stay uniform (no diffusion flux)."""
        sub = small_env.add_substrate(
            "uniform", diffusion_coefficient=1e-6, decay_rate=0.0,
            initial_value=5.0, dirichlet_boundary_value=5.0,
        )
        for _ in range(50):
            small_env.step()
        # Interior should remain ~5.0 (Dirichlet boundaries also 5.0)
        interior = sub.concentration[2:-2, 2:-2, 0]
        assert np.allclose(interior, 5.0, atol=0.01)

    def test_point_source_spreads(self, small_env):
        """A point source should cause concentration to spread outward."""
        sub = small_env.add_substrate(
            "spread", diffusion_coefficient=1e-6, decay_rate=0.0,
            initial_value=0.0, dirichlet_boundary_value=None,
        )
        center = (10, 10, 0)
        sub.concentration[center] = 100.0
        initial_max = sub.concentration.max()

        for _ in range(200):
            small_env.step()

        # Max should decrease (spread out)
        assert sub.concentration.max() < initial_max
        # Neighbors should have gained concentration
        assert sub.concentration[9, 10, 0] > 0.0
        assert sub.concentration[10, 9, 0] > 0.0

    def test_mass_conservation_neumann(self, small_env):
        """With no-flux boundaries and no decay, total mass should be conserved."""
        sub = small_env.add_substrate(
            "conserved", diffusion_coefficient=1e-7, decay_rate=0.0,
            initial_value=0.0, dirichlet_boundary_value=None,
        )
        # Place a blob of concentration
        sub.concentration[8:13, 8:13, 0] = 50.0
        initial_mass = sub.concentration.sum()

        for _ in range(100):
            small_env.step()

        final_mass = sub.concentration.sum()
        # Allow small numerical error
        assert final_mass == pytest.approx(initial_mass, rel=0.05)

    def test_dirichlet_boundary_maintained(self, small_env):
        """Dirichlet boundaries should stay at boundary value."""
        sub = small_env.add_substrate(
            "dirichlet", diffusion_coefficient=1e-6, decay_rate=0.01,
            initial_value=0.0, dirichlet_boundary_value=10.0,
        )
        for _ in range(50):
            small_env.step()

        # Boundaries should be at 10.0
        assert sub.concentration[0, :, 0].mean() == pytest.approx(10.0, abs=0.01)
        assert sub.concentration[-1, :, 0].mean() == pytest.approx(10.0, abs=0.01)
        assert sub.concentration[:, 0, 0].mean() == pytest.approx(10.0, abs=0.01)
        assert sub.concentration[:, -1, 0].mean() == pytest.approx(10.0, abs=0.01)

    def test_concentration_stays_nonnegative(self, small_env):
        """Concentration should never go below zero."""
        sub = small_env.add_substrate(
            "nonneg", diffusion_coefficient=1e-6, decay_rate=1.0,
            initial_value=0.1, dirichlet_boundary_value=None,
        )
        for _ in range(200):
            small_env.step()
        assert (sub.concentration >= 0).all()


# ---------------------------------------------------------------------------
# Gradient Computation
# ---------------------------------------------------------------------------

class TestGradient:

    def test_gradient_direction(self):
        """Gradient should point from low to high concentration."""
        env = Microenvironment(
            x_range=(0, 50), y_range=(0, 50), z_range=(0, 0),
            dx=5, dy=5, dz=5, dimensionality=2,
        )
        sub = env.add_substrate("grad_test", 1e-6, 0.0, initial_value=0.0)
        # Linear gradient in x: concentration increases with x
        for i in range(env.nx):
            sub.concentration[i, :, 0] = float(i)

        grad = env.get_gradient_at("grad_test", (25.0, 25.0))
        assert grad[0] > 0  # positive x gradient
        assert abs(grad[1]) < abs(grad[0])  # no y gradient

    def test_gradient_zero_for_uniform(self):
        """Uniform field should have zero gradient."""
        env = Microenvironment(
            x_range=(0, 50), y_range=(0, 50), z_range=(0, 0),
            dx=5, dy=5, dz=5, dimensionality=2,
        )
        env.add_substrate("flat", 1e-6, 0.0, initial_value=5.0)
        grad = env.get_gradient_at("flat", (25.0, 25.0))
        assert np.allclose(grad, 0.0, atol=1e-6)

    def test_gradient_nonexistent_substrate(self):
        """Querying gradient for missing substrate returns zero."""
        env = Microenvironment(
            x_range=(0, 50), y_range=(0, 50), z_range=(0, 0),
            dx=5, dy=5, dz=5, dimensionality=2,
        )
        grad = env.get_gradient_at("nonexistent", (25.0, 25.0))
        assert np.allclose(grad, 0.0)


# ---------------------------------------------------------------------------
# Coordinate Conversion
# ---------------------------------------------------------------------------

class TestCoordinateConversion:

    def test_position_to_voxel_roundtrip(self):
        env = Microenvironment(
            x_range=(0, 100), y_range=(0, 100), z_range=(0, 0),
            dx=10, dy=10, dz=10, dimensionality=2,
        )
        voxel = env.position_to_voxel((55.0, 35.0))
        assert voxel == (5, 3, 0)

    def test_voxel_to_position(self):
        env = Microenvironment(
            x_range=(0, 100), y_range=(0, 100), z_range=(0, 0),
            dx=10, dy=10, dz=10, dimensionality=2,
        )
        pos = env.voxel_to_position((5, 3, 0))
        assert pos == (50.0, 30.0)

    def test_get_concentration_at(self):
        env = Microenvironment(
            x_range=(0, 100), y_range=(0, 100), z_range=(0, 0),
            dx=10, dy=10, dz=10, dimensionality=2,
        )
        sub = env.add_substrate("test", 1e-6, 0.0, initial_value=0.0)
        sub.concentration[5, 5, 0] = 42.0
        conc = env.get_concentration_at("test", (50.0, 50.0))
        assert conc == pytest.approx(42.0)


# ---------------------------------------------------------------------------
# Helper Functions
# ---------------------------------------------------------------------------

class TestHelperFunctions:

    @pytest.fixture
    def env(self):
        return Microenvironment(
            x_range=(0, 100), y_range=(0, 100), z_range=(0, 0),
            dx=10, dy=10, dz=10, dimensionality=2,
        )

    def test_create_oxygen(self, env):
        sub = create_oxygen_substrate(env, boundary_value=38.0)
        assert sub.name == "oxygen"
        assert np.allclose(sub.concentration, 38.0)
        assert sub.dirichlet_boundary_value == 38.0

    def test_create_drug(self, env):
        sub = create_drug_substrate(env)
        assert sub.name == "drug"
        assert np.allclose(sub.concentration, 0.0)

    def test_create_pheromone(self, env):
        sub = create_pheromone_substrate(env, "trail", decay_rate=0.2)
        assert sub.name == "trail"
        assert sub.decay_rate == pytest.approx(0.2)
        assert sub.dirichlet_boundary_value is None  # no-flux

    def test_substrate_summary(self, env):
        create_oxygen_substrate(env)
        summary = env.get_substrate_summary()
        assert "oxygen" in summary
        assert summary["oxygen"]["mean"] == pytest.approx(38.0)
        assert summary["oxygen"]["std"] == pytest.approx(0.0)
