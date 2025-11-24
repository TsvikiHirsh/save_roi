# Save ROI

Extract spectral data from TIFF stacks using ImageJ ROIs or automated grid-based analysis.

## Overview

**Save ROI** is a Python package that provides flexible tools for extracting spectral profiles from TIFF image stacks. It supports multiple analysis modes:

- **ImageJ ROI files**: Use existing ROI definitions from ImageJ/Fiji
- **Full image analysis**: Analyze the entire image without ROI constraints
- **Grid-based analysis**: Systematic extraction using pixel grid patterns (e.g., 4x4 blocks)
- **Pixel-by-pixel analysis**: Extract spectra for individual pixels with optional stride

This package is ideal for analyzing spectroscopic imaging data, z-stacks, or any multi-slice TIFF data where you need to extract intensity profiles across slices.

## Features

- **Multiple input formats**: Works with `.roi` and `.zip` ROI files from ImageJ
- **Flexible analysis modes**: ROI-based, full image, grid patterns, or pixel-by-pixel
- **CSV output**: Generates CSV files with consistent structure (stack, counts, err)
- **Command-line interface**: Easy batch processing with CLI
- **Python API**: Use directly in Jupyter notebooks or Python scripts
- **Pip installable**: Standard Python package installation

## Installation

### From source (development)

```bash
# Clone or download the repository
cd save_roi

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

# Extract spectrum for entire image (no ROI)
save-roi --tiff image.tiff

# Extract spectra for 4x4 pixel grid
save-roi --tiff image.tiff --mode grid --grid-size 4

# Extract spectra for every 4th pixel
save-roi --tiff image.tiff --mode pixel --stride 4
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

## Command Line Options

```
usage: save-roi [-h] -t TIFF [-r ROI] [-o OUTPUT] [-m {roi,full,pixel,grid}]
                    [--grid-size GRID_SIZE] [--stride STRIDE] [--no-save]
                    [--version]

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
  --no-save             Do not save CSV files
  --version             Show version number
```

## Analysis Modes

### 1. ROI Mode (Default)

Use ImageJ ROI files to define regions of interest.

```bash
spectral-roi --tiff image.tiff --roi roi_file.zip
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
spectral-roi --tiff image.tiff --mode full
```

```python
from spectral_roi import extract_full_image_spectrum

df = extract_full_image_spectrum(tiff_path="image.tiff")
```

### 3. Grid Mode

Extract spectra for grid-based pixel blocks.

```bash
# 4x4 pixel blocks
spectral-roi --tiff image.tiff --mode grid --grid-size 4

# 8x8 pixel blocks
spectral-roi --tiff image.tiff --mode grid --grid-size 8
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
spectral-roi --tiff image.tiff --mode pixel

# Every 4th pixel in each direction
spectral-roi --tiff image.tiff --mode pixel --stride 4
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
spectral-roi --tiff image.tiff --roi roi.zip --output ./my_results
```

## Python API Reference

### Core Functions

#### `extract_roi_spectra(tiff_path, roi_path=None, output_dir=None, save_csv=True)`

Extract spectral data for ROIs from a TIFF stack.

**Parameters:**
- `tiff_path`: Path to TIFF stack file
- `roi_path`: Path to ImageJ ROI file (`.roi` or `.zip`)
- `output_dir`: Output directory for CSV files
- `save_csv`: Whether to save CSV files (default: True)

**Returns:** Dictionary mapping ROI names to pandas DataFrames

#### `extract_full_image_spectrum(tiff_path, output_dir=None, save_csv=True)`

Extract spectral data for the entire image.

**Parameters:**
- `tiff_path`: Path to TIFF stack file
- `output_dir`: Output directory for CSV files
- `save_csv`: Whether to save CSV files (default: True)

**Returns:** pandas DataFrame with spectral data

#### `extract_grid_spectra(tiff_path, grid_size=4, output_dir=None, save_csv=True)`

Extract spectral data using grid-based pixel blocks.

**Parameters:**
- `tiff_path`: Path to TIFF stack file
- `grid_size`: Size of grid blocks (e.g., 4 for 4x4 pixels)
- `output_dir`: Output directory for CSV files
- `save_csv`: Whether to save CSV files (default: True)

**Returns:** Dictionary mapping grid cell names to pandas DataFrames

#### `extract_pixel_spectra(tiff_path, output_dir=None, save_csv=True, stride=1)`

Extract spectral data for individual pixels.

**Parameters:**
- `tiff_path`: Path to TIFF stack file
- `output_dir`: Output directory for CSV files
- `save_csv`: Whether to save CSV files (default: True)
- `stride`: Sampling stride (e.g., 4 for every 4th pixel)

**Returns:** Dictionary mapping pixel coordinates to pandas DataFrames

## Examples

See the [example notebook](notebooks/example_usage.ipynb) for detailed examples including:
- Loading and plotting spectral data
- Comparing multiple ROIs
- Custom analysis workflows
- Working with results in memory

## Use Cases

**Spectral ROI** is designed for:

- **Spectroscopic imaging**: Extract spectra from hyperspectral image stacks
- **Time-lapse analysis**: Analyze intensity changes over time
- **Z-stack analysis**: Extract depth profiles from confocal microscopy
- **Fluorescence spectroscopy**: Measure emission spectra from fluorescent samples
- **Raman imaging**: Extract Raman spectra from imaging datasets
- **Any multi-slice TIFF analysis**: General-purpose spectral extraction

## Project Structure

```
save_roi/
├── src/
│   └── spectral_roi/
│       ├── __init__.py
│       ├── core.py          # Core extraction functions
│       └── cli.py           # Command-line interface
├── notebooks/
│   ├── example_usage.ipynb  # Example notebook
│   ├── image.tiff          # Test TIFF stack
│   └── ROI2.zip            # Test ROI file
├── scripts/
│   ├── save_roi.sh         # Bash wrapper script
│   └── save_roi.ljm        # Original ImageJ macro
├── tests/
│   └── test_core.py        # Unit tests
├── pyproject.toml          # Package configuration
├── setup.py
└── README.md
```

## Dependencies

- Python >= 3.8
- numpy >= 1.20.0
- tifffile >= 2021.0.0
- roifile >= 2021.0.0
- pandas >= 1.3.0

Optional (for development):
- pytest >= 7.0.0
- jupyter >= 1.0.0
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
| Command-line interface | ✗ | ✓ |
| Python API | ✗ | ✓ |
| Jupyter notebook support | ✗ | ✓ |
| Full image analysis | ✗ | ✓ |
| Grid-based analysis | ✗ | ✓ |
| Pixel-by-pixel analysis | ✗ | ✓ |
| Custom output directory | ✗ | ✓ |
| Pip installable | ✗ | ✓ |

## Contributing

Contributions are welcome! Please feel free to submit issues or pull requests.

## License

MIT License - see LICENSE file for details.

## Citation

If you use this package in your research, please cite:

```
Save ROI: A Python package for extracting spectral data from TIFF stacks
https://github.com/yourusername/save_roi
```

## Acknowledgments

This package was developed as a Python adaptation of an ImageJ macro for spectral ROI analysis. It uses the excellent `tifffile` and `roifile` libraries for file I/O.

## Support

For questions, issues, or feature requests, please open an issue on GitHub.
