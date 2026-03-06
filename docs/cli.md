# CLI Reference

## Synopsis

```
save-roi [data_dir] [options]
```

## Quick Workflow

```bash
save-roi data
```

Searches for TIFF files in `data/final/`, ROI files in the current directory, and writes output to `data/<tiff>_ROI_Spectra/`.

## Options

### Input

| Option | Short | Description |
|--------|-------|-------------|
| `data_dir` | | Data directory for quick workflow |
| `--tiff PATH` | `-t` | Path to TIFF stack file or directory |
| `--roi PATH` | `-r` | Path to ImageJ ROI file (`.roi` or `.zip`), or directory. Multiple files are merged automatically |

### Output

| Option | Short | Description |
|--------|-------|-------------|
| `--output DIR` | `-o` | Output directory for CSV files. Default: `{tiff}_ROI_Spectra/` next to the TIFF |
| `--suffix NAME` | `-s` | Create a named subfolder inside the output directory |
| `--no-save` | | Skip writing CSV files (useful for testing the API) |

### Analysis Mode

| Option | Short | Description |
|--------|-------|-------------|
| `--mode MODE` | `-m` | `roi` (default), `full`, `pixel`, or `grid` |
| `--grid-size N` | | Block size for grid mode (default: 4 → 4×4 pixels) |
| `--stride N` | | Sampling stride for pixel mode (default: 1 → every pixel) |
| `--jobs N` | `-j` | Parallel workers for grid/pixel modes. Use `-1` for all cores (default: 10) |

### Corrections

| Option | Description |
|--------|-------------|
| `--tilt ROI_NAME` | Apply tilt correction using the named symmetry-line ROI |

### Other

| Option | Description |
|--------|-------------|
| `--version` | Print version and exit |
| `--help`, `-h` | Show basic help |
| `--help --advanced` | Show full help including advanced options |

## Analysis Modes

### roi (default)

Extract a spectrum for each ROI defined in an ImageJ `.roi` or `.zip` file.

```bash
save-roi --tiff image.tiff --roi rois.zip
```

If no ROI file is found, the tool automatically falls back to `full` mode.

### full

Average over the entire image — no ROI needed.

```bash
save-roi --tiff image.tiff --mode full
```

### grid

Divide the image into rectangular blocks and average each block.

```bash
# 4×4 blocks (default)
save-roi --tiff image.tiff --mode grid

# 8×8 blocks
save-roi --tiff image.tiff --mode grid --grid-size 8
```

Output filename: `grid_4x4_x{col}_y{row}.csv`

### pixel

Extract one spectrum per pixel (or per sampled pixel when using `--stride`).

```bash
# Every pixel
save-roi --tiff image.tiff --mode pixel

# Every 4th pixel (faster)
save-roi --tiff image.tiff --mode pixel --stride 4

# All CPU cores
save-roi --tiff image.tiff --mode pixel --stride 4 --jobs -1
```

Output filename: `pixel_x{col}_y{row}.csv`

## Output

All modes produce CSV files with:

| Column | Description |
|--------|-------------|
| `stack` | Slice number (1-indexed, matches ImageJ) |
| `counts` | Total intensity counts in the region |
| `err` | Uncertainty (sqrt of counts by default; see `--smooth` for Poisson propagation) |

Default output directory: `{tiff_dir}/{tiff_stem}_ROI_Spectra/`

With `--suffix run1`: `{tiff_dir}/{tiff_stem}_ROI_Spectra/run1/`

## Examples

```bash
# Quick workflow
save-roi data

# Single file, ROI mode
save-roi --tiff image.tiff --roi rois.zip

# Full image
save-roi --tiff image.tiff --mode full

# Grid, custom output, with suffix
save-roi --tiff image.tiff --mode grid --grid-size 4 --output ./results --suffix run1

# Pixel-by-pixel with stride and parallel workers
save-roi --tiff image.tiff --mode pixel --stride 4 --jobs -1

# Tilt correction
save-roi --tiff image.tiff --roi rois.zip --tilt symmetry_line

# Batch: all TIFFs in a directory
save-roi --tiff data/final/ --roi rois.zip
```
