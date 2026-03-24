"""Unit tests for tumor microenvironment components.

Tests cover:
- TumorCell creation, oxygen status, drug accumulation, growth/division
- VesselPoint initialization
- TumorGeometry: circular tumor generation, cell distributions, vasculature
- ImmuneCell basics
"""

import numpy as np
import pytest
from backend.tumor_environment import (
    CellPhase,
    CellType,
    ImmuneCellType,
    TumorCell,
    ImmuneCell,
    VesselPoint,
    TumorGeometry,
)


# ---------------------------------------------------------------------------
# TumorCell
# ---------------------------------------------------------------------------

class TestTumorCell:

    def test_creation_defaults(self):
        cell = TumorCell(cell_id=0, position=(100.0, 100.0, 0.0))
        assert cell.cell_id == 0
        assert cell.phase == CellPhase.VIABLE
        assert cell.cell_type == CellType.DIFFERENTIATED
        assert cell.is_alive is True
        assert cell.accumulated_drug == 0.0

    def test_cell_types_have_different_params(self):
        stem = TumorCell(0, (0, 0, 0), cell_type=CellType.STEM_CELL)
        diff = TumorCell(1, (0, 0, 0), cell_type=CellType.DIFFERENTIATED)
        invasive = TumorCell(2, (0, 0, 0), cell_type=CellType.INVASIVE)

        # Stem cells: lower drug sensitivity, higher resistance
        assert stem.drug_sensitivity < diff.drug_sensitivity
        assert stem.resistance_level > diff.resistance_level
        # Invasive: faster division
        assert invasive.division_threshold < diff.division_threshold

    def test_oxygen_viable_to_hypoxic(self):
        cell = TumorCell(0, (0, 0, 0))
        # Oxygen below hypoxic threshold
        cell.update_oxygen_status(oxygen_concentration=1.0, dt=1.0)
        assert cell.phase == CellPhase.HYPOXIC
        assert cell.hypoxic_duration > 0

    def test_oxygen_recovery(self):
        cell = TumorCell(0, (0, 0, 0))
        cell.update_oxygen_status(1.0, dt=1.0)
        assert cell.phase == CellPhase.HYPOXIC
        # Restore oxygen
        cell.update_oxygen_status(38.0, dt=1.0)
        assert cell.phase == CellPhase.VIABLE
        assert cell.hypoxic_duration == 0.0

    def test_prolonged_hypoxia_causes_necrosis(self):
        cell = TumorCell(0, (0, 0, 0))
        # Exceed necrotic time threshold
        for _ in range(int(cell.necrotic_time_threshold) + 5):
            cell.update_oxygen_status(0.5, dt=1.0)
        assert cell.phase == CellPhase.NECROTIC
        assert cell.is_alive is False

    def test_dead_cell_ignores_oxygen(self):
        cell = TumorCell(0, (0, 0, 0))
        cell.is_alive = False
        cell.phase = CellPhase.NECROTIC
        cell.update_oxygen_status(38.0, dt=1.0)
        assert cell.phase == CellPhase.NECROTIC  # unchanged

    def test_accumulate_drug_kills_cell(self):
        cell = TumorCell(0, (0, 0, 0), cell_type=CellType.DIFFERENTIATED)
        # Deliver enough drug to kill (lethal_drug_dose = 0.5 for differentiated)
        # accumulate_drug applies 8x enhancement
        killed = cell.accumulate_drug(0.1)  # 0.1 * 8 = 0.8 > 0.5
        assert killed is True
        assert cell.phase == CellPhase.APOPTOTIC
        assert cell.is_alive is False

    def test_accumulate_drug_sublethal(self):
        cell = TumorCell(0, (0, 0, 0), cell_type=CellType.STEM_CELL)
        # Stem cell lethal_drug_dose = 2.0; 0.01 * 8 = 0.08 < 2.0
        killed = cell.accumulate_drug(0.01)
        assert killed is False
        assert cell.is_alive is True
        assert cell.accumulated_drug > 0

    def test_accumulate_drug_dead_cell(self):
        cell = TumorCell(0, (0, 0, 0))
        cell.is_alive = False
        killed = cell.accumulate_drug(10.0)
        assert killed is False

    def test_absorb_drug_from_environment(self):
        cell = TumorCell(0, (0, 0, 0))
        initial_drug = cell.accumulated_drug
        cell.absorb_drug(drug_concentration=5.0, dt=1.0)
        assert cell.accumulated_drug > initial_drug

    def test_growth_requires_oxygen(self):
        cell = TumorCell(0, (0, 0, 0))
        # Low oxygen — no growth
        result = cell.update_growth(dt=1.0, oxygen_concentration=1.0)
        assert result is False
        assert cell.growth_progress == 0.0

    def test_growth_with_good_oxygen(self):
        cell = TumorCell(0, (0, 0, 0))
        # Well oxygenated (above 1.2 * hypoxic_threshold)
        o2 = cell.hypoxic_threshold * 1.5
        cell.update_growth(dt=1.0, oxygen_concentration=o2)
        assert cell.growth_progress > 0.0

    def test_division(self):
        cell = TumorCell(0, (50.0, 50.0, 0.0))
        cell.growth_progress = cell.division_threshold  # ready to divide
        daughter = cell.divide(new_cell_id=1)
        assert daughter is not None
        assert daughter.cell_id == 1
        assert daughter.generation == 1
        assert cell.growth_progress == 0.0  # reset
        # Daughter is nearby
        dist = np.sqrt(
            (daughter.position[0] - cell.position[0]) ** 2
            + (daughter.position[1] - cell.position[1]) ** 2
        )
        assert dist == pytest.approx(2.0 * cell.radius, abs=1.0)

    def test_division_dead_cell(self):
        cell = TumorCell(0, (0, 0, 0))
        cell.is_alive = False
        daughter = cell.divide(1)
        assert daughter is None

    def test_oxygen_consumption_varies_by_phase(self):
        cell = TumorCell(0, (0, 0, 0))
        viable_rate = cell.get_oxygen_consumption()
        cell.phase = CellPhase.HYPOXIC
        hypoxic_rate = cell.get_oxygen_consumption()
        cell.is_alive = False
        dead_rate = cell.get_oxygen_consumption()

        assert viable_rate > hypoxic_rate > dead_rate
        assert dead_rate == 0.0

    def test_to_dict(self):
        cell = TumorCell(0, (10.0, 20.0, 0.0))
        d = cell.to_dict()
        assert d["id"] == 0
        assert d["phase"] == "viable"
        assert d["is_alive"] is True


# ---------------------------------------------------------------------------
# VesselPoint
# ---------------------------------------------------------------------------

class TestVesselPoint:

    def test_creation(self):
        v = VesselPoint(position=(100.0, 100.0, 0.0))
        assert v.oxygen_supply == 38.0
        assert v.vessel_type == "normal"
        assert v.bbb_permeability == 0.1

    def test_to_dict(self):
        v = VesselPoint((50.0, 50.0, 0.0), vessel_type="bbb")
        d = v.to_dict()
        assert d["vessel_type"] == "bbb"
        assert d["position"] == (50.0, 50.0, 0.0)


# ---------------------------------------------------------------------------
# TumorGeometry
# ---------------------------------------------------------------------------

class TestTumorGeometry:

    def test_init(self):
        geom = TumorGeometry(center=(250.0, 250.0, 0.0), tumor_radius=200.0)
        assert geom.center == (250.0, 250.0, 0.0)
        assert geom.tumor_radius == 200.0
        assert len(geom.tumor_cells) == 0
        assert len(geom.vessels) == 0

    def test_generate_circular_tumor(self):
        np.random.seed(42)
        geom = TumorGeometry(
            center=(250.0, 250.0, 0.0),
            tumor_radius=200.0,
            necrotic_core_radius=50.0,
        )
        geom.generate_circular_tumor(cell_density=0.0005, dimensionality=2)

        assert len(geom.tumor_cells) > 0
        assert len(geom.vessels) > 0

    def test_cells_in_annular_region(self):
        """All generated cells should be between necrotic core and tumor edge."""
        np.random.seed(42)
        geom = TumorGeometry(
            center=(250.0, 250.0, 0.0),
            tumor_radius=200.0,
            necrotic_core_radius=50.0,
        )
        geom.generate_circular_tumor(cell_density=0.0005)

        center = np.array([250.0, 250.0])
        for cell in geom.tumor_cells:
            dist = np.sqrt(
                (cell.position[0] - center[0]) ** 2
                + (cell.position[1] - center[1]) ** 2
            )
            assert dist >= 50.0 - 1.0  # allow small float error
            assert dist <= 200.0 + 1.0

    def test_has_hypoxic_cells(self):
        """Inner cells should start hypoxic."""
        np.random.seed(42)
        geom = TumorGeometry(
            center=(250.0, 250.0, 0.0),
            tumor_radius=200.0,
            necrotic_core_radius=50.0,
        )
        geom.generate_circular_tumor(cell_density=0.0005)

        hypoxic = [c for c in geom.tumor_cells if c.phase == CellPhase.HYPOXIC]
        assert len(hypoxic) > 0

    def test_cell_type_diversity(self):
        """Generated tumor should have multiple cell types."""
        np.random.seed(42)
        geom = TumorGeometry(
            center=(250.0, 250.0, 0.0),
            tumor_radius=200.0,
            necrotic_core_radius=50.0,
        )
        geom.generate_circular_tumor(cell_density=0.001)

        types = {c.cell_type for c in geom.tumor_cells}
        assert CellType.DIFFERENTIATED in types
        # With enough cells, should have at least 2 types
        assert len(types) >= 2

    def test_get_living_cells(self):
        geom = TumorGeometry(center=(0, 0, 0))
        c1 = TumorCell(0, (10, 10, 0))
        c2 = TumorCell(1, (20, 20, 0))
        c2.is_alive = False
        geom.tumor_cells = [c1, c2]
        living = geom.get_living_cells()
        assert len(living) == 1
        assert living[0].cell_id == 0

    def test_find_nearest_vessel(self):
        geom = TumorGeometry(center=(0, 0, 0))
        v1 = VesselPoint((10.0, 10.0, 0.0))
        v2 = VesselPoint((100.0, 100.0, 0.0))
        geom.vessels = [v1, v2]
        nearest = geom.find_nearest_vessel((12.0, 12.0, 0.0))
        assert nearest is v1


# ---------------------------------------------------------------------------
# ImmuneCell
# ---------------------------------------------------------------------------

class TestImmuneCell:

    def test_creation(self):
        ic = ImmuneCell(0, (50.0, 50.0, 0.0), ImmuneCellType.T_CELL)
        assert ic.is_active is True
        assert ic.cell_type == ImmuneCellType.T_CELL
        assert ic.cytotoxicity > 0

    def test_types_have_different_properties(self):
        t = ImmuneCell(0, (0, 0, 0), ImmuneCellType.T_CELL)
        m = ImmuneCell(1, (0, 0, 0), ImmuneCellType.MACROPHAGE)
        nk = ImmuneCell(2, (0, 0, 0), ImmuneCellType.NK_CELL)

        # NK cells are fastest and most cytotoxic
        assert nk.migration_speed > t.migration_speed > m.migration_speed
        assert nk.cytotoxicity > t.cytotoxicity > m.cytotoxicity
