"""
BraTS NIfTI Loader for Antelligence

Converts BraTS MRI segmentation volumes into TumorGeometry objects
that the nanobot simulation can use.

BraTS segmentation labels:
  0 = background
  1 = necrotic tumor core (NCR) -> maps to NECROTIC phase cells
  2 = peritumoral edema (ED) -> maps to HYPOXIC phase cells  
  4 = GD-enhancing tumor (ET) -> maps to VIABLE phase cells (active, targetable)

Coordinate system:
  BraTS: voxel indices (i, j, k) with 1mm isotropic spacing
  Simulation: micrometers (µm), centered at (domain_size/2, domain_size/2)
  Conversion: sim_pos = (voxel_idx - tumor_center_voxel) * voxel_spacing_mm * 1000 + domain_center
  But we scale to fit in domain_size, so:
  scale = domain_size / (max_tumor_extent_mm * 1000)
"""
import numpy as np
import json
import os
from pathlib import Path
from typing import Optional, Tuple, List, Dict

DATA_DIR = Path(__file__).parent.parent / 'data' / 'brats'


class BraTSVolume:
    """Loaded and parsed BraTS patient volume."""
    
    def __init__(self, patient_dir: Path):
        self.patient_dir = patient_dir
        self.patient_id = patient_dir.name
        self.seg = None          # uint8 (240,240,155)
        self.t1ce = None         # float32 (240,240,155) optional
        self.affine = None       # 4x4 affine matrix
        self.voxel_spacing = (1.0, 1.0, 1.0)  # mm
        self.shape = None
        self.tumor_center = None  # voxel coords
        self.tumor_bbox = None    # (min_i, max_i, min_j, max_j, min_k, max_k)
        self._load()
    
    def _load(self):
        try:
            import nibabel as nib
        except ImportError:
            raise ImportError("nibabel required: pip install nibabel")
        
        seg_path = self.patient_dir / 'seg.nii.gz'
        if not seg_path.exists():
            # Try other naming conventions
            segs = list(self.patient_dir.glob('*seg*.nii.gz'))
            if not segs:
                raise FileNotFoundError(f"No seg.nii.gz in {self.patient_dir}")
            seg_path = segs[0]
        
        img = nib.load(str(seg_path))
        self.seg = np.array(img.get_fdata(), dtype=np.uint8)
        self.affine = img.affine
        self.shape = self.seg.shape
        
        # Extract voxel spacing from affine
        self.voxel_spacing = tuple(abs(self.affine[i,i]) for i in range(3))
        
        # Load T1ce if available (for intensity context)
        t1ce_path = self.patient_dir / 't1ce.nii.gz'
        if t1ce_path.exists():
            t1ce_img = nib.load(str(t1ce_path))
            self.t1ce = np.array(t1ce_img.get_fdata(), dtype=np.float32)
        
        # Compute tumor bounding box and center
        tumor_mask = self.seg > 0
        if tumor_mask.any():
            coords = np.where(tumor_mask)
            self.tumor_bbox = (
                int(coords[0].min()), int(coords[0].max()),
                int(coords[1].min()), int(coords[1].max()),
                int(coords[2].min()), int(coords[2].max())
            )
            self.tumor_center = (
                int((coords[0].min() + coords[0].max()) / 2),
                int((coords[1].min() + coords[1].max()) / 2),
                int((coords[2].min() + coords[2].max()) / 2)
            )
    
    def get_tumor_slice(self, axis: int = 2, offset: int = 0) -> np.ndarray:
        """Get 2D cross-section through tumor center."""
        if self.tumor_center is None:
            return np.zeros((self.shape[0], self.shape[1]), dtype=np.uint8)
        z = self.tumor_center[axis] + offset
        z = np.clip(z, 0, self.shape[axis]-1)
        if axis == 0: return self.seg[z, :, :]
        if axis == 1: return self.seg[:, z, :]
        return self.seg[:, :, z]
    
    def get_statistics(self) -> Dict:
        if self.seg is None:
            return {}
        total_voxels = self.seg.size
        ncr = int((self.seg == 1).sum())
        ed = int((self.seg == 2).sum())
        et = int((self.seg == 4).sum())
        tumor_total = ncr + ed + et
        # Convert to physical volume (mm^3 -> cm^3)
        vox_vol = self.voxel_spacing[0] * self.voxel_spacing[1] * self.voxel_spacing[2]
        return {
            "patient_id": self.patient_id,
            "shape": list(self.shape),
            "voxel_spacing_mm": list(self.voxel_spacing),
            "tumor_center_voxel": list(self.tumor_center) if self.tumor_center else None,
            "tumor_bbox": list(self.tumor_bbox) if self.tumor_bbox else None,
            "ncr_voxels": ncr, "ed_voxels": ed, "et_voxels": et,
            "ncr_cm3": round(ncr * vox_vol / 1000, 2),
            "ed_cm3": round(ed * vox_vol / 1000, 2),
            "et_cm3": round(et * vox_vol / 1000, 2),
            "tumor_total_cm3": round(tumor_total * vox_vol / 1000, 2),
            "necrotic_fraction": round(ncr / max(1, tumor_total), 3),
            "edema_fraction": round(ed / max(1, tumor_total), 3),
            "enhancing_fraction": round(et / max(1, tumor_total), 3),
        }


def load_brats_patient(patient_id: Optional[str] = None) -> Optional[BraTSVolume]:
    """
    Load a BraTS patient volume.
    If patient_id is None, loads the first available patient.
    Falls back to synthetic sample if no real data exists.
    """
    if not DATA_DIR.exists():
        DATA_DIR.mkdir(parents=True, exist_ok=True)
    
    # Find patient directories
    patients = sorted([
        d for d in DATA_DIR.rglob('*')
        if d.is_dir() and (d / 'seg.nii.gz').exists()
    ])
    
    # Also check one level up in samples/
    sample_patients = sorted([
        d for d in (DATA_DIR / 'samples').rglob('*')
        if d.is_dir() and (d / 'seg.nii.gz').exists()
    ]) if (DATA_DIR / 'samples').exists() else []
    
    all_patients = patients + sample_patients
    
    if not all_patients:
        # Create synthetic sample
        print("[BraTS] No patient data found. Creating synthetic sample...")
        try:
            import subprocess, sys
            script = Path(__file__).parent.parent / 'scripts' / 'download_brats.py'
            subprocess.run([sys.executable, str(script), '--sample'], check=True)
            # Retry
            all_patients = sorted([
                d for d in DATA_DIR.rglob('*')
                if d.is_dir() and (d / 'seg.nii.gz').exists()
            ])
        except Exception as e:
            print(f"[BraTS] Failed to create synthetic sample: {e}")
            return None
    
    if not all_patients:
        return None
    
    if patient_id:
        match = [p for p in all_patients if patient_id in p.name]
        patient_dir = match[0] if match else all_patients[0]
    else:
        patient_dir = all_patients[0]
    
    try:
        print(f"[BraTS] Loading patient: {patient_dir.name}")
        vol = BraTSVolume(patient_dir)
        stats = vol.get_statistics()
        print(f"[BraTS] Tumor volume: {stats.get('tumor_total_cm3', '?')} cm³")
        print(f"[BraTS] NCR: {stats.get('ncr_cm3', '?')} cm³, ET: {stats.get('et_cm3', '?')} cm³, ED: {stats.get('ed_cm3', '?')} cm³")
        return vol
    except Exception as e:
        print(f"[BraTS] Error loading {patient_dir}: {e}")
        return None


def brats_volume_to_tumor_geometry(
    vol: BraTSVolume,
    domain_size: float = 600.0,
    target_slice_axis: int = 2,
    max_cells: int = 500,
    cell_density_scale: float = 1.0,
    include_3d: bool = False
):
    """
    Convert BraTS segmentation volume to TumorGeometry.
    
    Maps BraTS labels to simulation cell types:
      Label 1 (NCR, necrotic) -> NECROTIC phase cells, STEM_CELL type (necrotic core has stem cells)
      Label 4 (ET, enhancing) -> VIABLE phase cells, mixed DIFFERENTIATED/RESISTANT
      Label 2 (ED, edema) -> HYPOXIC phase cells, INVASIVE type (infiltrating edge)
    
    Coordinate mapping:
      Tumor region in voxel space is cropped to bounding box
      Scaled to fit within domain_size (µm)
      Centered at (domain_size/2, domain_size/2)
    
    Args:
        vol: loaded BraTSVolume
        domain_size: simulation domain in µm
        target_slice_axis: 0=sagittal, 1=coronal, 2=axial (default axial)
        max_cells: maximum cells to place (performance limit)
        cell_density_scale: multiplier on cell sampling density
        include_3d: if True, use 3D cell placement instead of 2D slice
    """
    import sys
    sys.path.insert(0, str(Path(__file__).parent))
    from tumor_environment import (
        TumorCell, TumorGeometry, VesselPoint, ImmuneCell,
        CellPhase, CellType, ImmuneCellType,
        create_simple_tumor_environment
    )
    
    if vol.tumor_bbox is None:
        print("[BraTS] No tumor found in segmentation, falling back to synthetic geometry")
        return create_simple_tumor_environment(domain_size=domain_size)
    
    # Get 2D cross-section at tumor center (axial slice is most informative for GBM)
    seg_slice = vol.get_tumor_slice(axis=target_slice_axis)
    
    # Crop to tumor bounding box on this slice
    tumor_mask = seg_slice > 0
    if not tumor_mask.any():
        # Try adjacent slices
        for offset in range(1, 20):
            for sign in [1, -1]:
                seg_slice = vol.get_tumor_slice(axis=target_slice_axis, offset=sign*offset)
                if (seg_slice > 0).any():
                    break
            if (seg_slice > 0).any():
                break
    
    tumor_mask = seg_slice > 0
    if not tumor_mask.any():
        print("[BraTS] Tumor not visible in any axial slice, using synthetic")
        return create_simple_tumor_environment(domain_size=domain_size)
    
    # Bounding box of tumor on this slice
    rows, cols = np.where(tumor_mask)
    r_min, r_max = rows.min(), rows.max()
    c_min, c_max = cols.min(), cols.max()
    
    # Extent in voxels
    r_extent = r_max - r_min
    c_extent = c_max - c_min
    max_extent = max(r_extent, c_extent)
    
    # Scale: fit tumor into 70% of domain (leave margin for vessels)
    usable_domain = domain_size * 0.7
    vox_to_um = (usable_domain / max(1, max_extent)) * cell_density_scale
    
    domain_center = domain_size / 2.0
    r_center = (r_min + r_max) / 2.0
    c_center = (c_min + c_max) / 2.0
    
    def voxel_to_sim(r, c):
        """Convert voxel (row, col) to simulation (x, y) in µm."""
        x = (c - c_center) * vox_to_um + domain_center
        y = (r - r_center) * vox_to_um + domain_center
        return float(x), float(y)
    
    # Sample cells from the segmentation
    # We can't place every voxel as a cell (too many)
    # Instead, sample up to max_cells voxels proportionally by label
    ncr_voxels = list(zip(*np.where(seg_slice == 1)))
    et_voxels = list(zip(*np.where(seg_slice == 4)))
    ed_voxels = list(zip(*np.where(seg_slice == 2)))
    
    total_tumor_voxels = len(ncr_voxels) + len(et_voxels) + len(ed_voxels)
    if total_tumor_voxels == 0:
        return create_simple_tumor_environment(domain_size=domain_size)
    
    # Proportional sampling
    n_ncr = max(1, int(max_cells * len(ncr_voxels) / total_tumor_voxels))
    n_et = max(1, int(max_cells * len(et_voxels) / total_tumor_voxels))
    n_ed = max(1, int(max_cells * len(ed_voxels) / total_tumor_voxels))
    
    rng = np.random.default_rng(42)
    
    def sample_voxels(voxel_list, n):
        if len(voxel_list) <= n:
            return voxel_list
        idx = rng.choice(len(voxel_list), n, replace=False)
        return [voxel_list[i] for i in idx]
    
    sampled_ncr = sample_voxels(ncr_voxels, n_ncr)
    sampled_et = sample_voxels(et_voxels, n_et)
    sampled_ed = sample_voxels(ed_voxels, n_ed)
    
    # Build TumorGeometry
    # Estimate tumor radius from extent
    tumor_radius = (max_extent / 2) * vox_to_um
    tumor_center_3d = (domain_center, domain_center, 0.0)
    
    geometry = TumorGeometry(
        center=tumor_center_3d,
        tumor_radius=tumor_radius,
        necrotic_core_radius=tumor_radius * (len(ncr_voxels) / max(1, total_tumor_voxels)) * 1.5
    )
    
    cell_id = 0
    
    # NCR (necrotic core) -> NECROTIC phase, STEM_CELL type
    # Necrotic core harbors therapy-resistant stem cells
    for r, c in sampled_ncr:
        x, y = voxel_to_sim(r, c)
        cell = TumorCell(
            cell_id=cell_id,
            position=(x, y, 0.0),
            radius=10.0,
            initial_phase=CellPhase.NECROTIC,
            cell_type=CellType.STEM_CELL
        )
        cell.is_alive = True  # necrotic but still present
        geometry.tumor_cells.append(cell)
        cell_id += 1
    
    # ET (enhancing tumor) -> VIABLE, mixed DIFFERENTIATED and RESISTANT
    for r, c in sampled_et:
        x, y = voxel_to_sim(r, c)
        # 30% resistant in enhancing region (they survived initial treatment)
        cell_type = CellType.RESISTANT if rng.random() < 0.3 else CellType.DIFFERENTIATED
        cell = TumorCell(
            cell_id=cell_id,
            position=(x, y, 0.0),
            radius=10.0,
            initial_phase=CellPhase.VIABLE,
            cell_type=cell_type
        )
        geometry.tumor_cells.append(cell)
        cell_id += 1
    
    # ED (edema) -> HYPOXIC, INVASIVE type (infiltrating cells at margin)
    for r, c in sampled_ed:
        x, y = voxel_to_sim(r, c)
        cell = TumorCell(
            cell_id=cell_id,
            position=(x, y, 0.0),
            radius=10.0,
            initial_phase=CellPhase.HYPOXIC,
            cell_type=CellType.INVASIVE
        )
        geometry.tumor_cells.append(cell)
        cell_id += 1
    
    # Generate vasculature around tumor (same as before but scaled)
    geometry._generate_peripheral_vasculature(dimensionality=2)
    
    # Generate immune cells
    geometry._generate_immune_cells(dimensionality=2)
    
    # Store BraTS metadata on geometry object
    geometry.brats_metadata = {
        "patient_id": vol.patient_id,
        "source": "brats",
        "slice_axis": target_slice_axis,
        "tumor_stats": vol.get_statistics(),
        "vox_to_um": vox_to_um,
        "cells_from_ncr": len(sampled_ncr),
        "cells_from_et": len(sampled_et),
        "cells_from_ed": len(sampled_ed),
    }
    
    print(f"[BraTS] Geometry created: {len(geometry.tumor_cells)} cells, {len(geometry.vessels)} vessels")
    print(f"[BraTS] NCR={len(sampled_ncr)}, ET={len(sampled_et)}, ED={len(sampled_ed)}")
    print(f"[BraTS] Tumor radius: {tumor_radius:.1f}µm, voxel scale: {vox_to_um:.2f}µm/voxel")
    
    return geometry
