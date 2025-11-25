"""
Spectral ROI - Extract spectral data from TIFF stacks using ImageJ ROIs or automated grid-based analysis
"""

__version__ = "0.1.0"

from .core import (
    extract_roi_spectra,
    extract_full_image_spectrum,
    extract_pixel_spectra,
    extract_grid_spectra,
)

__all__ = [
    "extract_roi_spectra",
    "extract_full_image_spectrum",
    "extract_pixel_spectra",
    "extract_grid_spectra",
]

# Optional interactive tools (require additional dependencies)
try:
    from .interactive import (
        InteractiveROISelector,
        launch_interactive_tool,
    )
    __all__.extend([
        "InteractiveROISelector",
        "launch_interactive_tool",
    ])
except ImportError:
    # Interactive tools not available (missing dependencies)
    pass
