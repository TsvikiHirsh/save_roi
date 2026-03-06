# Advanced Options

Access advanced help with:

```bash
save-roi --help --advanced
```

## Open-Beam Smoothing (`--smooth`)

For spatially uniform samples (e.g. open-beam neutron measurements) where individual ROIs have low counts, `--smooth` improves statistics by borrowing the TOF spectral shape from the full image.

**How it works:**

The smoothed spectrum for region `r` at TOF slice `t` is:

```
I_smooth(r, t) = N(r) * S(t) / S_total
```

where:
- `N(r)` = total counts in region `r` across all slices
- `S(t)` = full-image sum at slice `t`
- `S_total` = grand total counts

Each region's total counts `N(r)` are preserved, so the spatial contrast is maintained.

**Uncertainty:**

All three quantities (`N(r)`, `S(t)`, `S_total`) are correlated Poisson random variables (they share raw pixel counts). The exact variance after propagation is:

```
sigma^2 = I_smooth^2 * (1/N(r) + 1/S(t) - 3/S_total + 2*I_raw(r,t)/(N(r)*S(t)))
```

For the full-image mask this reduces to `sigma^2 = S(t)` (standard Poisson), confirming the formula is correct.

**Usage:**

```bash
# ROI mode
save-roi --tiff openbeam.tiff --roi rois.zip --smooth

# Grid mode
save-roi --tiff openbeam.tiff --mode grid --grid-size 4 --smooth

# Pixel mode
save-roi --tiff openbeam.tiff --mode pixel --stride 2 --smooth
```

**When to use:** Open-beam or flat-field measurements where spatial variation is small compared to statistical noise. Do not use for samples with strong spatial gradients, as the smoothing assumes spatial uniformity.

## Filename Prefix (`--prefix`)

By default, output filenames include a mode tag:

- ROI mode: `ROI_1.csv`, `background.csv`
- Grid mode: `grid_4x4_x0_y0.csv`, `grid_4x4_x0_y4.csv`
- Pixel mode: `pixel_x0_y0.csv`, `pixel_x0_y1.csv`

`--prefix STRING` replaces the mode tag with your prefix:

- ROI mode: `ob_ROI_1.csv`, `ob_background.csv`
- Grid/pixel mode: `ob_x0_y0.csv`, `ob_x0_y4.csv`

```bash
# Label open-beam run
save-roi --tiff ob.tiff --roi rois.zip --prefix ob_run1

# Label sample run
save-roi --tiff sample.tiff --roi rois.zip --prefix sample_run1

# Combine with smooth for open-beam
save-roi --tiff ob.tiff --mode grid --smooth --prefix ob
save-roi --tiff sample.tiff --mode grid --prefix sample
```

This makes it easy to compare open-beam and sample runs side by side in the same output folder.

## Tilt Correction (`--tilt`)

For samples where the image is physically tilted, `--tilt ROI_NAME` straightens the image before analysis.

The named ROI must be a line ROI in the ImageJ file that defines a symmetry axis (e.g. the edge of a sample). The stack is rotated so this line becomes vertical and centered.

```bash
save-roi --tiff image.tiff --roi rois.zip --tilt symmetry_line
```

The `symmetry_line` ROI is not extracted as a spectrum — it is used only for the geometric correction.

## Parallel Processing (`--jobs`)

Grid and pixel modes use multiprocessing. By default 10 workers are used.

```bash
# Use all available CPU cores
save-roi --tiff image.tiff --mode grid --jobs -1

# Limit to 4 cores (reduces memory for large files)
save-roi --tiff image.tiff --mode grid --jobs 4
```

For large TIFF stacks (3 GB+), use `--jobs 4` or increase `--grid-size` to reduce peak memory.

## Organizing Multiple Analyses

Combine `--suffix` and `--prefix` to keep different analyses organized:

```bash
# Open-beam and sample, both in results/
save-roi --tiff ob.tiff --roi rois.zip --smooth --prefix ob --output results --suffix run1
save-roi --tiff sample.tiff --roi rois.zip --prefix sample --output results --suffix run1

# Different corrections of the same data
save-roi data --suffix raw
save-roi data --suffix tilt_corrected --tilt symmetry_line
save-roi data --suffix smooth --smooth
```

Output structure with `--suffix`:
```
results/
└── run1/
    ├── ob_ROI_1.csv
    ├── ob_ROI_2.csv
    ├── sample_ROI_1.csv
    └── sample_ROI_2.csv
```
