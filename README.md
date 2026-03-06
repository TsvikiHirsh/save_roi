# Save ROI
![logo](notebooks/save-roi_logo.png)

Quickly extract spectral data from TIFF stacks using ImageJ ROIs.

## Quick Start

**Install:**
```bash
cd save_roi
pip install -e .
```

**Organize your files:**
```
your_project/
├── ROI2.zip              # ImageJ ROI file
└── data/
    └── final/
        └── image.tiff    # TIFF stack
```

**Run:**
```bash
cd your_project
save-roi data
```

Output CSV files appear in `data/image_ROI_Spectra/`.

## Common Commands

```bash
# Auto-discover everything in data/
save-roi data

# Specific TIFF and ROI file
save-roi --tiff image.tiff --roi rois.zip

# Full image (no ROI)
save-roi --tiff image.tiff

# Grid analysis (4×4 pixel blocks)
save-roi --tiff image.tiff --mode grid
```

## Python API

```python
from save_roi import extract_roi_spectra

results = extract_roi_spectra("image.tiff", roi_path="rois.zip")
for roi_name, df in results.items():
    print(roi_name, df.head())
```

## Output Format

Each CSV has three columns:

| Column | Description |
|--------|-------------|
| `stack` | Slice number (1-indexed) |
| `counts` | Total intensity in the region |
| `err` | Poisson uncertainty (sqrt of counts) |

## Advanced Options

```bash
# See all options
save-roi --help --advanced
```

Advanced features include open-beam smoothing (`--smooth`), filename prefixes (`--prefix`), tilt correction (`--tilt`), and parallel processing (`--jobs`).

See the [docs/](docs/) folder for detailed documentation:
- [CLI reference](docs/cli.md)
- [Advanced options](docs/advanced.md)
- [Python API](docs/api.md)

## Analysis Modes

| Mode | Command | Use case |
|------|---------|---------|
| `roi` | `--mode roi --roi rois.zip` | ImageJ ROI regions |
| `full` | `--mode full` | Entire image |
| `grid` | `--mode grid --grid-size 4` | 4×4 pixel blocks |
| `pixel` | `--mode pixel --stride 2` | Per-pixel, every 2nd |

## Testing

```bash
pytest tests/
```

## License

MIT License
