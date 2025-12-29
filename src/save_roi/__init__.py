"""
Spectral ROI - Extract spectral data from TIFF stacks using ImageJ ROIs or automated grid-based analysis
"""

__version__ = "0.1.0"

from .core import (
    extract_roi_spectra,
    extract_full_image_spectrum,
    extract_pixel_spectra,
    extract_grid_spectra,
    apply_tilt_correction,
    load_tiff_stack,
    load_imagej_rois,
    discover_tiff_files,
    discover_roi_files,
    discover_directories_with_wildcard,
    merge_roi_files,
    sum_roi_spectra_from_folders,
)

__all__ = [
    "extract_roi_spectra",
    "extract_full_image_spectrum",
    "extract_pixel_spectra",
    "extract_grid_spectra",
    "apply_tilt_correction",
    "load_tiff_stack",
    "load_imagej_rois",
    "discover_tiff_files",
    "discover_roi_files",
    "discover_directories_with_wildcard",
    "merge_roi_files",
    "sum_roi_spectra_from_folders",
]
