# Save ROI
![logo](notebooks/save-roi_logo.png)

Quickly extract spectral data from TIFF stacks using ImageJ ROIs.

## TLDR

```bash
# Quick workflow - auto-discovers everything
save-roi data

# Single file with ROI
save-roi --tiff image.tiff --roi rois.zip

# Full image analysis (no ROI)
save-roi --tiff image.tiff --mode full

# Grid-based analysis (4×4 pixel blocks)
save-roi --tiff image.tiff --mode grid --grid-size 4
```

See [`notebooks/example.sh`](notebooks/example.sh) and [`notebooks/example_usage.ipynb`](notebooks/example_usage.ipynb) for more examples.

## Quick Start

**Organize your files:**
```
your_project/
├── ROI2.zip              # Your ImageJ ROI file
└── data/
    └── final/
        └── image.tiff    # Your TIFF stack
```

**Run one command:**
```bash
cd your_project
save-roi data
```

**Output:**
- Auto-discovers ROI files in current directory
- Auto-discovers TIFF files in `data/final/`
- Creates CSV files in `data/image_ROI_Spectra/`
- Shows progress bar

## Overview

**Save ROI** is a Python package that provides tools for extracting spectral profiles from TIFF image stacks. It supports multiple analysis modes:

- **ImageJ ROI files**: Use ROI definitions from ImageJ/Fiji
- **Tilt correction**: Automatically straighten and center images based on a symmetry line ROI
- **Full image analysis**: Analyze the entire image without ROI constraints
- **Grid-based analysis**: Systematic extraction using pixel grid patterns (e.g., 4x4 blocks)
- **Pixel-by-pixel analysis**: Extract spectra for individual pixels with optional stride

This package is ideal for analyzing spectroscopic imaging data, z-stacks, or any multi-slice TIFF data where you need to extract intensity profiles across slices.

## Features

- **Quick Workflow**: Simple `save-roi data` command
- **Auto-Discovery**: Finds TIFF and ROI files automatically
- **Multiple ROI Files**: Merges multiple `.roi`/`.zip` files
- **Progress Bars**: Real-time progress tracking
- **Parallel Processing**: Fast grid analysis (up to 10 CPU cores)
- **Memory Efficient**: Handles 3GB+ TIFF stacks
- **Python API**: Use in scripts or Jupyter notebooks

## Installation

### From source 

```bash
# Clone or download the repository
cd save_roi

# Extract test data (TIFF file is compressed to save space)
python scripts/setup_test_data.py

# Install in editable mode 
pip install -e .
```

**Key features:**
- Multiple ROI files are automatically merged
- Conflicts are detected and reported
- Works with any number of TIFF files in `data/final/`
- Perfect for batch processing spectroscopy data

## Common Usage

### Command Line

```bash
# Quick workflow - auto-discovers everything
save-roi data

# Specific files
save-roi --tiff image.tiff --roi rois.zip

# Full image (no ROI)
save-roi --tiff image.tiff --mode full

# Grid analysis (4×4 blocks)
save-roi --tiff image.tiff --mode grid --grid-size 4

# Use all CPU cores
save-roi --tiff image.tiff --mode grid --jobs -1
```

### Python API

```python
from spectral_roi import extract_roi_spectra

# Extract spectra with ROI file
results = extract_roi_spectra(
    tiff_path="image.tiff",
    roi_path="rois.zip"
)

# Access results as pandas DataFrames
for roi_name, df in results.items():
    print(f"{roi_name}: {len(df)} slices")
```

See [`notebooks/example_usage.ipynb`](notebooks/example_usage.ipynb) and [`notebooks/example.sh`](notebooks/example.sh) for detailed examples.

## Analysis Modes

### ROI Mode (Default)

Extract spectra from ImageJ ROI regions.

```bash
save-roi --tiff image.tiff --roi rois.zip
```

### Full Image Mode

Analyze the entire image (no ROI needed).

```bash
save-roi --tiff image.tiff --mode full
```

### Grid Mode

Divide image into blocks and average each block.

```bash
# 4×4 pixel blocks
save-roi --tiff image.tiff --mode grid --grid-size 4

# 8×8 pixel blocks (faster, lower resolution)
save-roi --tiff image.tiff --mode grid --grid-size 8
```

**For large files (3GB+)**: Use `--jobs 4` to reduce memory usage or increase `--grid-size` for faster processing.

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
