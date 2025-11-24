# Quick Start Guide - Save ROI

## Installation

```bash
# Install in editable mode for development
pip install -e ".[dev]"

# Or install for use
pip install -e .
```

## Usage Examples

### 1. Command Line Interface

```bash
# Extract spectra using ImageJ ROI file
save-roi --tiff notebooks/image.tiff --roi notebooks/ROI2.zip

# Extract spectrum for entire image (no ROI)
save-roi --tiff notebooks/image.tiff --mode full

# Extract spectra for 4x4 pixel grid (uses 10 cores by default)
save-roi --tiff notebooks/image.tiff --mode grid --grid-size 4

# Extract spectra for every 4th pixel using all available cores
save-roi --tiff notebooks/image.tiff --mode pixel --stride 4 --jobs -1

# Specify custom output directory
save-roi --tiff notebooks/image.tiff --roi notebooks/ROI2.zip --output ./my_results
```

### 2. Bash Script Wrapper

```bash
# Use the bash wrapper script (same interface as CLI)
./scripts/save_roi.sh --tiff notebooks/image.tiff --roi notebooks/ROI2.zip
```

### 3. Python API

```python
from spectral_roi import extract_roi_spectra, extract_full_image_spectrum

# Extract using ImageJ ROI file
results = extract_roi_spectra(
    tiff_path="notebooks/image.tiff",
    roi_path="notebooks/ROI2.zip"
)

# Access results (dict of pandas DataFrames)
for roi_name, df in results.items():
    print(f"\n{roi_name}:")
    print(df.head())
    print(f"Total counts: {df['counts'].sum()}")

# Extract full image spectrum
df = extract_full_image_spectrum(
    tiff_path="notebooks/image.tiff",
    save_csv=True
)
print(df)
```

### 4. Jupyter Notebook

See [notebooks/example_usage.ipynb](notebooks/example_usage.ipynb) for detailed examples with plotting.

## Project Structure

```
save_roi/
├── src/spectral_roi/          # Main package source
│   ├── __init__.py
│   ├── core.py               # Core extraction functions
│   └── cli.py                # Command-line interface
├── notebooks/
│   ├── example_usage.ipynb   # Example notebook
│   ├── image.tiff           # Test TIFF stack
│   └── ROI2.zip             # Test ROI file
├── scripts/
│   ├── save_roi.sh          # Bash wrapper script
│   └── save_roi.ljm         # Original ImageJ macro
├── tests/
│   └── test_core.py         # Unit tests
├── README.md                # Full documentation
├── QUICK_START.md           # This file
├── pyproject.toml           # Package configuration
└── setup.py
```

## Output Format

All analysis modes produce CSV files with the same structure:

| Column | Description |
|--------|-------------|
| `stack` | Slice number (1-indexed, matching ImageJ) |
| `counts` | Total intensity counts (sum of pixel values) |
| `err` | Error estimate (√counts, Poisson statistics) |

## Analysis Modes

| Mode | Description | Command Example |
|------|-------------|-----------------|
| **ROI** | Use ImageJ ROI files | `--roi roi.zip` |
| **Full** | Analyze entire image | `--mode full` |
| **Grid** | Grid-based blocks | `--mode grid --grid-size 4` |
| **Pixel** | Individual pixels | `--mode pixel --stride 4` |

## Running Tests

```bash
# Run all tests
pytest tests/

# Run with verbose output
pytest tests/ -v

# Run with coverage
pytest tests/ --cov=spectral_roi
```

## Common Use Cases

### Spectroscopic Imaging
Extract spectral profiles from different ROIs in your sample:
```bash
save-roi --tiff spectrum_stack.tiff --roi regions.zip
```

### Grid-Based Analysis
Systematic spatial analysis with grid patterns (parallel processing):
```bash
# Use 10 cores (default)
save-roi --tiff data.tiff --mode grid --grid-size 8

# Use all available cores
save-roi --tiff data.tiff --mode grid --grid-size 8 --jobs -1
```

### Pixel-Level Analysis
Detailed pixel-by-pixel analysis (parallel processing, use stride to reduce data):
```bash
# Use 10 cores (default)
save-roi --tiff data.tiff --mode pixel --stride 4

# Use all available cores
save-roi --tiff data.tiff --mode pixel --stride 4 --jobs -1
```

## Need Help?

- Full documentation: See [README.md](README.md)
- Examples: See [notebooks/example_usage.ipynb](notebooks/example_usage.ipynb)
- Command help: `save-roi --help`
