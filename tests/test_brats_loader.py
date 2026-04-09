import pytest
import numpy as np
from pathlib import Path

def test_synthetic_sample_creation():
    from backend.brats_loader import load_brats_patient
    vol = load_brats_patient()
    assert vol is not None
    assert vol.seg is not None
    assert vol.seg.shape == (240, 240, 155)
    assert vol.tumor_center is not None

def test_tumor_statistics():
    from backend.brats_loader import load_brats_patient
    vol = load_brats_patient()
    stats = vol.get_statistics()
    assert 'ncr_cm3' in stats
    assert 'et_cm3' in stats
    assert 'ed_cm3' in stats
    assert stats['tumor_total_cm3'] > 0

def test_brats_to_geometry():
    from backend.brats_loader import load_brats_patient, brats_volume_to_tumor_geometry
    vol = load_brats_patient()
    geom = brats_volume_to_tumor_geometry(vol, domain_size=600.0, max_cells=100)
    assert geom is not None
    assert len(geom.tumor_cells) > 0
    assert len(geom.vessels) > 0
    # All cells within domain
    for cell in geom.tumor_cells:
        assert 0 <= cell.position[0] <= 600.0
        assert 0 <= cell.position[1] <= 600.0

def test_cell_type_mapping():
    from backend.brats_loader import load_brats_patient, brats_volume_to_tumor_geometry
    from backend.tumor_environment import CellType, CellPhase
    vol = load_brats_patient()
    geom = brats_volume_to_tumor_geometry(vol, domain_size=600.0, max_cells=200)
    cell_types = {c.cell_type for c in geom.tumor_cells}
    # Should have multiple cell types from real BraTS anatomy
    assert len(cell_types) > 1

def test_create_brats_tumor_geometry_fallback():
    from backend.tumor_environment import create_brats_tumor_geometry
    # Should not raise NotImplementedError anymore
    geom = create_brats_tumor_geometry(domain_size=600.0, max_cells=50)
    assert geom is not None
    assert len(geom.tumor_cells) > 0
