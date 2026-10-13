"""Antelligence v4.1 perception pipeline (goal 4.1).

One MSD Task01 case in -> calibrated per-voxel tumour probability, uncertainty, crop and
region signals out. Lives outside ``antelligence.kernel``: the kernel never imports a world.

Module imports are lazy on purpose: ``msd``, ``calibrate``, ``crop``, ``regions`` and ``store``
need only numpy/scipy/nibabel; ``model`` and ``infer`` additionally need torch + monai.
"""

PIPELINE_VERSION = "v4.1-perception-1"
MODEL_ID = "MONAI/brats_mri_segmentation"
MODEL_REVISION = "370f7f9d062745fbac445e7fe6d6616d35df04ec"
MODEL_PROV = f"{MODEL_ID}@{MODEL_REVISION}"
TARGET_SPACING_MM = 2.0
