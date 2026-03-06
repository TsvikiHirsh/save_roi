# Python API Reference

```python
from save_roi import (
    extract_roi_spectra,
    extract_full_image_spectrum,
    extract_grid_spectra,
    extract_pixel_spectra,
    apply_tilt_correction,
    load_tiff_stack,
    load_imagej_rois,
    calculate_smooth_spectrum,
)
```

## Core Extraction Functions

### `extract_roi_spectra`

```python
extract_roi_spectra(
    tiff_path,
    roi_path=None,
    output_dir=None,
    save_csv=True,
    tilt_roi_name=None,
    smooth=False,
    prefix=None,
) -> dict[str, pd.DataFrame]
```

Extract a spectrum for each ROI in an ImageJ ROI file.

**Parameters:**

| Name | Type | Description |
|------|------|-------------|
| `tiff_path` | str / Path | TIFF stack file |
| `roi_path` | str / Path | `.roi` or `.zip` ROI file |
| `output_dir` | str / Path | Output directory (default: next to TIFF) |
| `save_csv` | bool | Write CSV files (default: True) |
| `tilt_roi_name` | str | ROI name for tilt correction |
| `smooth` | bool | Use open-beam smoothing |
| `prefix` | str | Prefix for output CSV filenames |

**Returns:** `{roi_name: DataFrame}` with columns `stack`, `counts`, `err`.

**Example:**

```python
results = extract_roi_spectra("image.tiff", roi_path="rois.zip")
df = results["ROI_1"]
print(df.head())
#    stack  counts       err
# 0      1  1234.0  35.1283
# 1      2  1189.0  34.4819
```

### `extract_full_image_spectrum`

```python
extract_full_image_spectrum(
    tiff_path,
    output_dir=None,
    save_csv=True,
) -> pd.DataFrame
```

Sum all pixels at each slice. Returns a single DataFrame.

### `extract_grid_spectra`

```python
extract_grid_spectra(
    tiff_path,
    grid_size=4,
    output_dir=None,
    save_csv=True,
    n_jobs=10,
    tilt_roi_name=None,
    roi_path=None,
    smooth=False,
    prefix=None,
) -> dict[str, pd.DataFrame]
```

Divide the image into `grid_size × grid_size` blocks and extract a spectrum per block.

Keys in the returned dict: `grid_4x4_x{col}_y{row}` (or `{prefix}_x{col}_y{row}` if prefix is set).

### `extract_pixel_spectra`

```python
extract_pixel_spectra(
    tiff_path,
    output_dir=None,
    save_csv=True,
    stride=1,
    n_jobs=10,
    tilt_roi_name=None,
    roi_path=None,
    smooth=False,
    prefix=None,
) -> dict[str, pd.DataFrame]
```

Extract one spectrum per pixel. Use `stride > 1` to sample every Nth pixel.

Keys: `pixel_x{col}_y{row}` (or `{prefix}_x{col}_y{row}`).

## Smoothing

### `calculate_smooth_spectrum`

```python
calculate_smooth_spectrum(
    stack: np.ndarray,        # (slices, height, width)
    mask: np.ndarray,         # (height, width) bool
    full_image_spectrum: np.ndarray,  # (slices,)
) -> pd.DataFrame
```

Compute the open-beam smoothed spectrum for a single mask. Usually called internally by the extraction functions when `smooth=True`, but available for custom workflows.

The model is `I_smooth(t) = N * S(t) / S_total` where `N` is the masked total counts, `S(t)` is the full-image spectrum, and `S_total = S.sum()`. Uncertainty accounts for all Poisson cross-correlations.

**Example:**

```python
import numpy as np
from save_roi import load_tiff_stack, calculate_smooth_spectrum

stack = load_tiff_stack("openbeam.tiff")
full_spectrum = stack.sum(axis=(1, 2)).astype(float)

# Custom mask (e.g. a circle)
mask = np.zeros(stack.shape[1:], dtype=bool)
cy, cx, r = 64, 64, 20
y, x = np.ogrid[:stack.shape[1], :stack.shape[2]]
mask[(y - cy)**2 + (x - cx)**2 <= r**2] = True

df = calculate_smooth_spectrum(stack, mask, full_spectrum)
```

## Utility Functions

### `load_tiff_stack`

```python
load_tiff_stack(tiff_path) -> np.ndarray  # shape (slices, height, width)
```

Load a TIFF stack into a numpy array. Supports `.tiff`, `.tif`, and `.tiff.gz`.

### `load_imagej_rois`

```python
load_imagej_rois(roi_path) -> list[ImagejRoi]
```

Load ROIs from a `.roi` file or `.zip` archive.

### `apply_tilt_correction`

```python
apply_tilt_correction(
    stack: np.ndarray,
    roi_path,
    tilt_roi_name: str,
    threshold: float = 0.4,
) -> tuple[np.ndarray, float, tuple]
```

Rotate and shift the stack so the named line ROI becomes vertical and centered.

Returns `(corrected_stack, rotation_angle_degrees, (center_y, center_x))`.

### `discover_tiff_files` / `discover_roi_files`

```python
discover_tiff_files(directory) -> list[Path]
discover_roi_files(directory) -> list[Path]
```

Recursively find TIFF or ROI files in a directory.

### `merge_roi_files`

```python
merge_roi_files(
    roi_files: list[Path],
    warn_on_conflict: bool = True,
) -> tuple[list[dict], list[str]]
```

Merge ROIs from multiple files. Returns `(roi_list, warnings)`. Conflicts (same ROI name from different files) are reported as warnings.

## Workflow Example

```python
from pathlib import Path
from save_roi import (
    load_tiff_stack,
    discover_tiff_files,
    discover_roi_files,
    merge_roi_files,
    extract_roi_spectra,
)

# Discover files
tiff_files = discover_tiff_files("data/final/")
roi_files = discover_roi_files(".")
merged_rois, warnings = merge_roi_files(roi_files)

# Process each TIFF
for tiff_path in tiff_files:
    results = extract_roi_spectra(
        tiff_path,
        roi_path="rois.zip",
        output_dir=Path("results") / tiff_path.stem,
        smooth=True,
        prefix="ob",
    )
    for name, df in results.items():
        print(f"{tiff_path.name} / {name}: {df['counts'].sum():.0f} total counts")
```
