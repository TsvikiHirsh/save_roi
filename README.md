# Save ROI
![logo](notebooks/save-roi_logo.png)

Quickly extract spectral data from TIFF stacks using ImageJ ROIs.

## Overview

**Save ROI** is a Python package that provides tools for extracting spectral profiles from TIFF image stacks. It supports multiple analysis modes:

- **ImageJ ROI files**: Use ROI definitions from ImageJ/Fiji
- **Tilt correction**: Automatically straighten and center images based on a symmetry line ROI
- **Full image analysis**: Analyze the entire image without ROI constraints
- **Grid-based analysis**: Systematic extraction using pixel grid patterns (e.g., 4x4 blocks)
- **Pixel-by-pixel analysis**: Extract spectra for individual pixels with optional stride

This package is ideal for analyzing spectroscopic imaging data, z-stacks, or any multi-slice TIFF data where you need to extract intensity profiles across slices.

## Features

- **Tilt Correction**: Straighten and center images using a symmetry line ROI
- **Multiple input formats**: Works with `.roi` and `.zip` ROI files from ImageJ
- **Flexible analysis modes**: ROI-based, full image, grid patterns, or pixel-by-pixel
- **Parallel processing**: Fast grid/pixel analysis using multiple CPU cores (default: 10 cores)
- **CSV output**: Generates CSV files with consistent structure (stack, counts, err)
- **Command-line interface**: Easy batch processing with CLI
- **Python API**: Use directly in Python scripts or Jupyter notebooks
- **Pip installable**: Standard Python package installation

## Installation

### From source (development)

```bash
# Clone or download the repository
cd save_roi

# Extract test data (TIFF file is compressed to save space)
python scripts/setup_test_data.py

# Install in editable mode with development dependencies
pip install -e ".[dev]"
```

### From PyPI (when published)

```bash
pip install save-roi
```

## Quick Start

### Command Line Usage

```bash
# Extract spectra using ImageJ ROI file
save-roi --tiff image.tiff --roi roi_file.zip

# Extract spectra with tilt correction
save-roi --tiff image.tiff --roi roi_file.zip --tilt symmetry_line

# Extract spectrum for entire image (no ROI)
save-roi --tiff image.tiff

# Extract spectra for 4x4 pixel grid (uses 10 cores by default)
save-roi --tiff image.tiff --mode grid --grid-size 4

# Extract spectra for every 4th pixel using all available cores
save-roi --tiff image.tiff --mode pixel --stride 4 --jobs -1
```

### Python API Usage

```python
from spectral_roi import extract_roi_spectra

# Extract spectra using ImageJ ROI file
results = extract_roi_spectra(
    tiff_path="image.tiff",
    roi_path="roi_file.zip",
    save_csv=True
)

# Extract spectra with tilt correction
results = extract_roi_spectra(
    tiff_path="image.tiff",
    roi_path="roi_file.zip",
    tilt_roi_name="symmetry_line",
    save_csv=True
)

# Access results as pandas DataFrames
for roi_name, df in results.items():
    print(f"{roi_name}:")
    print(df.head())
```

### Bash Script Usage

```bash
# Use the provided bash wrapper
./scripts/save_roi.sh --tiff image.tiff --roi roi_file.zip
```

## Tilt Correction

Tilt correction allows you to automatically straighten and center images based on a symmetry line defined by an ROI. This is useful for correcting sample tilt in microscopy or spectroscopy data.

### How it works

1. Create a line ROI in ImageJ that follows a symmetry axis of your sample
2. Save the ROI with a descriptive name (e.g., "symmetry_line")
3. Use the `--tilt` argument to apply correction before extracting spectra

### Command Line Usage

```bash
# Apply tilt correction using an ROI named "symmetry_line"
save-roi --tiff image.tiff --roi roi_file.zip --tilt symmetry_line

# Combine tilt correction with grid analysis
save-roi --tiff image.tiff --roi roi_file.zip --tilt symmetry_line --mode grid
```

### Python API Usage

```python
from spectral_roi import apply_tilt_correction, load_tiff_stack
import tifffile

# Load stack
stack = load_tiff_stack("image.tiff")

# Apply tilt correction
corrected_stack, angle, center = apply_tilt_correction(
    stack,
    roi_path="roi_file.zip",
    tilt_roi_name="symmetry_line"
)

# Save corrected stack
tifffile.imwrite("corrected.tiff", corrected_stack)

# Or extract spectra directly with tilt correction
from spectral_roi import extract_roi_spectra

results = extract_roi_spectra(
    tiff_path="image.tiff",
    roi_path="roi_file.zip",
    tilt_roi_name="symmetry_line"
)
```

### ROI Preparation in ImageJ

1. Open your TIFF stack in ImageJ
2. Use the **Line** tool to draw a line along a symmetry axis
   - For best results, draw the line through features that should be vertical
   - The line can be straight or follow multiple points
3. Add to ROI Manager (Ctrl+T or Cmd+T)
4. Rename the ROI to something meaningful (e.g., "symmetry_line")
5. Save all ROIs: **ROI Manager → More → Save**

## Command Line Options

```
usage: save-roi [-h] -t TIFF [-r ROI] [-o OUTPUT] [-m {roi,full,pixel,grid}]
                [--grid-size GRID_SIZE] [--stride STRIDE] [-j JOBS]
                [--no-save] [--tilt TILT] [--version]

Options:
  -h, --help            Show help message
  -t, --tiff TIFF       Path to TIFF stack file (required)
  -r, --roi ROI         Path to ImageJ ROI file (.roi or .zip)
  -o, --output OUTPUT   Output directory for CSV files
  -m, --mode {roi,full,pixel,grid}
                        Analysis mode (default: roi)
  --grid-size GRID_SIZE
                        Grid block size for grid mode (default: 4)
  --stride STRIDE       Stride for pixel mode (default: 1)
  -j, --jobs JOBS       Number of parallel jobs (default: 10, use -1 for all cores)
  --no-save             Do not save CSV files
  --tilt TILT           Name of ROI to use for tilt correction
  --version             Show version number
```

## Analysis Modes

### 1. ROI Mode (Default)

Use ImageJ ROI files to define regions of interest.

```bash
save-roi --tiff image.tiff --roi roi_file.zip
```

```python
from spectral_roi import extract_roi_spectra

results = extract_roi_spectra(
    tiff_path="image.tiff",
    roi_path="roi_file.zip"
)
```

### 2. Full Image Mode

Analyze the entire image without ROI constraints.

```bash
save-roi --tiff image.tiff --mode full
```

```python
from spectral_roi import extract_full_image_spectrum

df = extract_full_image_spectrum(tiff_path="image.tiff")
```

### 3. Grid Mode

Extract spectra for grid-based pixel blocks.

```bash
# 4x4 pixel blocks
save-roi --tiff image.tiff --mode grid --grid-size 4

# 8x8 pixel blocks
save-roi --tiff image.tiff --mode grid --grid-size 8
```

```python
from spectral_roi import extract_grid_spectra

results = extract_grid_spectra(
    tiff_path="image.tiff",
    grid_size=4
)
```

### 4. Pixel Mode

Extract spectra for individual pixels with optional stride.

```bash
# Every pixel (warning: creates many files!)
save-roi --tiff image.tiff --mode pixel

# Every 4th pixel in each direction
save-roi --tiff image.tiff --mode pixel --stride 4
```

```python
from spectral_roi import extract_pixel_spectra

results = extract_pixel_spectra(
    tiff_path="image.tiff",
    stride=4
)
```

## Output Format

All analysis modes produce CSV files with the same structure:

| Column | Description |
|--------|-------------|
| `stack` | Slice number (1-indexed, matching ImageJ) |
| `counts` | Total intensity counts in the region (sum of pixel values) |
| `err` | Error estimate (square root of counts, assuming Poisson statistics) |

### Default Output Location

By default, CSV files are saved to a subfolder next to the input TIFF:

```
/path/to/image.tiff
/path/to/image_ROI_Spectra/
    ├── ROI_1.csv
    ├── ROI_2.csv
    └── ...
```

You can specify a custom output directory with the `--output` option:

```bash
save-roi --tiff image.tiff --roi roi.zip --output ./my_results
```

## Python API Reference

### Core Functions

#### `extract_roi_spectra(tiff_path, roi_path=None, output_dir=None, save_csv=True, tilt_roi_name=None)`

Extract spectral data for ROIs from a TIFF stack.

**Parameters:**
- `tiff_path`: Path to TIFF stack file
- `roi_path`: Path to ImageJ ROI file (`.roi` or `.zip`)
- `output_dir`: Output directory for CSV files
- `save_csv`: Whether to save CSV files (default: True)
- `tilt_roi_name`: Name of ROI for tilt correction (optional)

**Returns:** Dictionary mapping ROI names to pandas DataFrames

#### `apply_tilt_correction(stack, roi_path, tilt_roi_name, threshold=0.4)`

Apply tilt correction to a TIFF stack based on a symmetry line ROI.

**Parameters:**
- `stack`: 3D numpy array (slices, height, width)
- `roi_path`: Path to ImageJ ROI file
- `tilt_roi_name`: Name of the ROI to use for tilt correction
- `threshold`: Threshold value for edge detection (default: 0.4)

**Returns:** Tuple of (corrected_stack, rotation_angle, center_coords)

#### `extract_full_image_spectrum(tiff_path, output_dir=None, save_csv=True)`

Extract spectral data for the entire image.

**Parameters:**
- `tiff_path`: Path to TIFF stack file
- `output_dir`: Output directory for CSV files
- `save_csv`: Whether to save CSV files (default: True)

**Returns:** pandas DataFrame with spectral data

#### `extract_grid_spectra(tiff_path, grid_size=4, output_dir=None, save_csv=True, n_jobs=10, tilt_roi_name=None, roi_path=None)`

Extract spectral data using grid-based pixel blocks.

**Parameters:**
- `tiff_path`: Path to TIFF stack file
- `grid_size`: Size of grid blocks (e.g., 4 for 4x4 pixels)
- `output_dir`: Output directory for CSV files
- `save_csv`: Whether to save CSV files (default: True)
- `n_jobs`: Number of parallel jobs (default: 10)
- `tilt_roi_name`: Name of ROI for tilt correction (optional)
- `roi_path`: Path to ROI file (required if using tilt correction)

**Returns:** Dictionary mapping grid cell names to pandas DataFrames

#### `extract_pixel_spectra(tiff_path, output_dir=None, save_csv=True, stride=1, n_jobs=10, tilt_roi_name=None, roi_path=None)`

Extract spectral data for individual pixels.

**Parameters:**
- `tiff_path`: Path to TIFF stack file
- `output_dir`: Output directory for CSV files
- `save_csv`: Whether to save CSV files (default: True)
- `stride`: Sampling stride (e.g., 4 for every 4th pixel)
- `n_jobs`: Number of parallel jobs (default: 10)
- `tilt_roi_name`: Name of ROI for tilt correction (optional)
- `roi_path`: Path to ROI file (required if using tilt correction)

**Returns:** Dictionary mapping pixel coordinates to pandas DataFrames

## Use Cases

**Spectral ROI** is designed for:

- **Spectroscopic imaging**: Extract spectra from hyperspectral image stacks
- **Time-lapse analysis**: Analyze intensity changes over time
- **Z-stack analysis**: Extract depth profiles from confocal microscopy
- **Fluorescence spectroscopy**: Measure emission spectra from fluorescent samples
- **Raman imaging**: Extract Raman spectra from imaging datasets
- **Sample alignment**: Correct for sample tilt using symmetry-based correction
- **Any multi-slice TIFF analysis**: General-purpose spectral extraction

## Project Structure

```
save_roi/
├── src/
│   └── spectral_roi/
│       ├── __init__.py
│       ├── core.py          # Core extraction and tilt correction functions
│       └── cli.py           # Command-line interface
├── notebooks/
│   ├── image.tiff          # Test TIFF stack
│   └── ROI2.zip            # Test ROI file
├── scripts/
│   ├── save_roi.sh         # Bash wrapper script
│   └── save_roi.ljm        # Original ImageJ macro
├── tests/
│   └── test_core.py        # Core functionality tests
├── pyproject.toml          # Package configuration
├── setup.py
└── README.md
```

## Dependencies

### Core Dependencies
- Python >= 3.8
- numpy >= 1.20.0
- scipy >= 1.7.0
- tifffile >= 2021.0.0
- roifile >= 2021.0.0
- pandas >= 1.3.0

### Optional Dependencies

For development:
- pytest >= 7.0.0
- matplotlib >= 3.5.0

## Testing

Run tests with pytest:

```bash
pytest tests/
```

## Comparison with ImageJ Macro

This package provides the same core functionality as the original ImageJ macro (`save_roi.ljm`) with several enhancements:

| Feature | ImageJ Macro | Spectral ROI |
|---------|--------------|--------------|
| ImageJ ROI support | ✓ | ✓ |
| Tilt correction | ✗ | ✓ |
| Command-line interface | ✗ | ✓ |
| Python API | ✗ | ✓ |
| Full image analysis | ✗ | ✓ |
| Grid-based analysis | ✗ | ✓ |
| Pixel-by-pixel analysis | ✗ | ✓ |
| Custom output directory | ✗ | ✓ |
| Pip installable | ✗ | ✓ |
| Parallel processing | ✗ | ✓ |

## Contributing

Contributions are welcome! Please feel free to submit issues or pull requests.

## License

MIT License - see LICENSE file for details.

## Citation

If you use this package in your research, please cite:

```
Save ROI: A Python package for extracting spectral data from TIFF stacks
https://github.com/TsvikiHirsh/save_roi
```

## Acknowledgments

This package was developed as a Python adaptation of an ImageJ macro for spectral ROI analysis. It uses the excellent `tifffile` and `roifile` libraries for file I/O, and `scipy` for image processing.

## Support

For questions, issues, or feature requests, please open an issue on GitHub.
