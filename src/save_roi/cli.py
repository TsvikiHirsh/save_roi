"""
Command-line interface for spectral-roi
"""

import argparse
import sys
from pathlib import Path
from .core import (
    extract_roi_spectra,
    extract_full_image_spectrum,
    extract_pixel_spectra,
    extract_grid_spectra,
    apply_tilt_correction,
    load_tiff_stack,
    discover_tiff_files,
    discover_roi_files,
    merge_roi_files,
    calculate_smooth_spectrum,
)
import tifffile
from tqdm import tqdm


_BASIC_EPILOG = """\
Examples:
  save-roi data                                    # quick workflow
  save-roi --tiff image.tiff --roi rois.zip        # ROI spectra
  save-roi --tiff image.tiff                       # full-image spectrum
  save-roi --tiff image.tiff --mode grid           # grid analysis (4x4 blocks)

For advanced options (open-beam smoothing, filename prefix, …):
  save-roi --help --advanced
"""

_ADVANCED_EPILOG = """\
Examples:
  # Quick workflow: discover TIFFs in data/final/, ROIs in PWD, output to data/
  save-roi data

  # Extract spectra using ImageJ ROI file
  save-roi --tiff image.tiff --roi rois.zip

  # Full image (no ROI)
  save-roi --tiff image.tiff

  # Grid analysis (4×4 blocks)
  save-roi --tiff image.tiff --mode grid --grid-size 4

  # Pixel-by-pixel, every 4th pixel, all CPU cores
  save-roi --tiff image.tiff --mode pixel --stride 4 --jobs -1

  # Custom output directory with a suffix subfolder
  save-roi --tiff image.tiff --roi rois.zip --output ./results --suffix run1

  # Tilt correction
  save-roi --tiff image.tiff --roi rois.zip --tilt symmetry_line

  # Open-beam smoothing (improved statistics for uniform samples)
  save-roi --tiff openbeam.tiff --roi rois.zip --smooth
  save-roi --tiff openbeam.tiff --mode grid --grid-size 4 --smooth

  # Short filenames with a custom prefix
  save-roi --tiff openbeam.tiff --roi rois.zip --smooth --prefix ob_run1
  save-roi --tiff openbeam.tiff --mode grid --smooth --prefix ob
"""

_SMOOTH_HELP = (
    'Open-beam smoothing: average the TOF spectrum over the full image '
    'and use its shape (scaled to the local spatial intensity) for each '
    'ROI / pixel / grid cell.  Improves statistics for spatially uniform '
    'samples (e.g. open beam).  Each region\'s total counts are preserved '
    'so the spatial background is maintained.  Uncertainty is propagated '
    'rigorously from Poisson statistics, accounting for all cross-correlations.'
)

_PREFIX_HELP = (
    'Prefix string prepended to every output CSV filename.  '
    'ROI/full mode: {prefix}_{roi_name}.csv.  '
    'Pixel/grid mode: {prefix}_x{x}_y{y}.csv '
    '(replaces the long grid_4x4_ / pixel_ mode tag).  '
    'Example: --prefix ob_run1'
)


def main():
    """Main CLI entry point"""
    show_advanced = '--advanced' in sys.argv

    parser = argparse.ArgumentParser(
        description="Extract spectral data from TIFF stacks using ImageJ ROIs or automated analysis",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=_ADVANCED_EPILOG if show_advanced else _BASIC_EPILOG,
        add_help=False,
    )

    # ── Help flags ────────────────────────────────────────────────────────────
    parser.add_argument(
        '-h', '--help',
        action='store_true', default=False,
        help='Show this help message and exit'
    )
    parser.add_argument(
        '--advanced',
        action='store_true', default=False,
        help='Show advanced options in help (use with --help)'
    )

    # ── Main arguments ────────────────────────────────────────────────────────
    parser.add_argument(
        'data_dir',
        nargs='?',
        default=None,
        help='Data directory for quick workflow. Searches for TIFFs in DATA_DIR/final/, ROIs in PWD, outputs to DATA_DIR/'
    )

    parser.add_argument(
        '-t', '--tiff',
        type=str, default=None,
        help='Path to TIFF stack file or directory.'
    )

    parser.add_argument(
        '-r', '--roi',
        type=str, default=None,
        help='Path to ImageJ ROI file (.roi or .zip), or directory to discover ROI files. Multiple files are merged automatically.'
    )

    parser.add_argument(
        '-o', '--output',
        type=str, default=None,
        help='Output directory for CSV files. Default: {tiff}_ROI_Spectra/ next to the TIFF.'
    )

    parser.add_argument(
        '-s', '--suffix',
        type=str, default=None,
        help='Subfolder appended to the output directory (e.g. "run1", "corrected").'
    )

    parser.add_argument(
        '-m', '--mode',
        type=str,
        choices=['roi', 'full', 'pixel', 'grid'],
        default='roi',
        help='Analysis mode: roi | full | pixel | grid.  Default: roi (or full when no ROI is found).'
    )

    parser.add_argument(
        '--grid-size',
        type=int, default=4,
        help='Block size for grid mode (default: 4 → 4×4 pixels).'
    )

    parser.add_argument(
        '--stride',
        type=int, default=1,
        help='Sampling stride for pixel mode (default: 1 → every pixel).'
    )

    parser.add_argument(
        '-j', '--jobs',
        type=int, default=10,
        help='Parallel jobs for grid/pixel modes. Use -1 for all cores. Default: 10'
    )

    parser.add_argument(
        '--no-save',
        action='store_true',
        help='Do not write CSV files (useful for testing).'
    )

    parser.add_argument(
        '--tilt',
        type=str, default=None,
        help='ROI name used for tilt correction. The ROI defines a symmetry line that is straightened to vertical and centred.'
    )

    parser.add_argument(
        '--version',
        action='version',
        version='%(prog)s 0.1.0'
    )

    # ── Advanced options ──────────────────────────────────────────────────────
    # Visible only with --help --advanced; still accepted on the command line.
    if show_advanced:
        adv = parser.add_argument_group('advanced options')
        adv.add_argument('--smooth', action='store_true', help=_SMOOTH_HELP)
        adv.add_argument('--prefix', type=str, default=None, help=_PREFIX_HELP)
    else:
        parser.add_argument('--smooth', action='store_true', help=argparse.SUPPRESS)
        parser.add_argument('--prefix', type=str, default=None, help=argparse.SUPPRESS)

    # ── Handle help early ─────────────────────────────────────────────────────
    if '-h' in sys.argv or '--help' in sys.argv:
        parser.print_help()
        sys.exit(0)

    args = parser.parse_args()

    # Check if using quick workflow (positional data_dir argument)
    use_quick_workflow = args.data_dir is not None and args.tiff is None
    custom_output_base = None

    if use_quick_workflow:
        # Quick workflow: data_dir/final/ contains TIFFs, PWD contains ROIs, output to data_dir/
        data_dir = Path(args.data_dir)
        if not data_dir.exists():
            print(f"Error: Data directory not found: {data_dir}", file=sys.stderr)
            sys.exit(1)
        if not data_dir.is_dir():
            print(f"Error: Data path is not a directory: {data_dir}", file=sys.stderr)
            sys.exit(1)

        print(f"Using quick workflow with data directory: {data_dir}")

        # Look for TIFFs in data_dir/final/
        tiff_search_dir = data_dir / "final"
        if not tiff_search_dir.exists():
            print(f"Error: Expected subdirectory not found: {tiff_search_dir}", file=sys.stderr)
            print(f"Quick workflow expects TIFFs in {data_dir}/final/", file=sys.stderr)
            sys.exit(1)

        tiff_input = tiff_search_dir
        custom_output_base = data_dir  # Output goes to data_dir, not data_dir/final/

        # ROI search defaults to PWD for quick workflow
        if args.roi is None:
            roi_search_dir = Path.cwd()
            print(f"Searching for ROI files in: {roi_search_dir}")
        else:
            roi_search_dir = Path(args.roi)

    elif args.data_dir is not None and args.tiff is not None:
        print("Error: Cannot specify both positional data_dir and --tiff argument", file=sys.stderr)
        sys.exit(1)
    else:
        # Standard workflow
        if args.tiff is None:
            tiff_input = Path.cwd()
            print(f"No TIFF path specified, using current directory: {tiff_input}")
        else:
            tiff_input = Path(args.tiff)
            if not tiff_input.exists():
                print(f"Error: Path not found: {tiff_input}", file=sys.stderr)
                sys.exit(1)

        roi_search_dir = None  # Will be determined later

    # Discover TIFF files
    if tiff_input.is_dir():
        tiff_files = discover_tiff_files(tiff_input)
        if not tiff_files:
            print(f"Error: No TIFF files found in directory: {tiff_input}", file=sys.stderr)
            sys.exit(1)
        print(f"Discovered {len(tiff_files)} TIFF file(s) in {tiff_input}")
        for tf in tiff_files:
            print(f"  - {tf.name}")
    else:
        # Single TIFF file
        tiff_files = [tiff_input]

    # Determine ROI search directory for standard workflow
    if not use_quick_workflow:
        if args.roi is None:
            # Auto-discover ROI files in the same directory as TIFF files
            roi_search_dir = tiff_input if tiff_input.is_dir() else tiff_input.parent
        else:
            roi_search_dir = Path(args.roi)

    # Discover ROI files
    if roi_search_dir is not None:
        if roi_search_dir.is_dir():
            roi_files = discover_roi_files(roi_search_dir)
            if roi_files:
                print(f"\nDiscovered {len(roi_files)} ROI file(s) in {roi_search_dir}")
                for rf in roi_files:
                    print(f"  - {rf.name}")
            else:
                print(f"\nNo ROI files found in {roi_search_dir}")
                roi_files = []
        elif roi_search_dir.exists() and roi_search_dir.is_file():
            # Single ROI file
            roi_files = [roi_search_dir]
            print(f"\nUsing ROI file: {roi_search_dir}")
        else:
            print(f"Error: ROI path not found: {roi_search_dir}", file=sys.stderr)
            sys.exit(1)
    else:
        roi_files = []

    # Merge ROI files if multiple
    all_rois = []
    roi_warnings = []
    if roi_files:
        all_rois, roi_warnings = merge_roi_files(roi_files, warn_on_conflict=True)
        if roi_warnings:
            print("\nROI Conflicts:")
            for warning in roi_warnings:
                print(f"  {warning}")
        print(f"\nTotal ROIs loaded: {len(all_rois)}")
        for roi_info in all_rois:
            print(f"  - {roi_info['name']}")

    # Validate tilt correction requirements
    if args.tilt and not all_rois:
        print("Error: --tilt requires ROI files to be available", file=sys.stderr)
        sys.exit(1)

    # Determine mode
    mode = args.mode
    if mode == 'roi' and not all_rois:
        print("\nWarning: No ROI files provided, switching to 'full' mode")
        mode = 'full'

    save_csv = not args.no_save

    # Create a helper function to process a single TIFF with merged ROIs
    def process_single_tiff_with_merged_rois(tiff_path, merged_rois, mode, args, base_output_dir, save_csv, custom_output_base):
        """Process a single TIFF file with pre-merged ROIs"""
        import tempfile
        import zipfile
        from roifile import roiwrite

        # If we have merged ROIs, we need to create a temporary ROI file for tilt correction
        temp_roi_path = None
        if args.tilt and merged_rois:
            # Create a temporary zip file with all merged ROIs
            temp_dir = Path(tempfile.mkdtemp())
            temp_roi_path = temp_dir / "merged_rois.zip"

            with zipfile.ZipFile(temp_roi_path, 'w') as zf:
                for roi_info in merged_rois:
                    roi_obj = roi_info['roi_object']
                    roi_name = roi_info['name']
                    # Write ROI to temporary file
                    temp_roi_file = temp_dir / f"{roi_name}.roi"
                    roiwrite(str(temp_roi_file), roi_obj)
                    zf.write(temp_roi_file, arcname=f"{roi_name}.roi")

        # Determine output directory
        if base_output_dir is not None:
            # User specified output directory
            output_dir = base_output_dir
            # Add suffix if provided
            if args.suffix:
                output_dir = output_dir / args.suffix
        elif custom_output_base is not None:
            # Quick workflow: output to custom_output_base/tiff_name_ROI_Spectra/
            output_dir = custom_output_base / f"{tiff_path.stem}_ROI_Spectra"
            # Add suffix if provided
            if args.suffix:
                output_dir = output_dir / args.suffix
        else:
            # Default: output next to TIFF file
            if args.suffix:
                # If suffix provided, create default output dir with suffix
                output_dir = tiff_path.parent / f"{tiff_path.stem}_ROI_Spectra" / args.suffix
            else:
                output_dir = None

        try:
            if mode == 'roi':
                # Process with merged ROIs directly
                results = extract_roi_spectra_with_merged_rois(
                    tiff_path,
                    merged_rois=merged_rois,
                    output_dir=output_dir,
                    save_csv=save_csv,
                    tilt_roi_name=args.tilt,
                    temp_roi_path=temp_roi_path,
                    smooth=args.smooth,
                    prefix=args.prefix
                )
                return len(results), 'ROI(s)'

            elif mode == 'full':
                result = extract_full_image_spectrum(
                    tiff_path,
                    output_dir=output_dir,
                    save_csv=save_csv
                )
                return len(result), 'slices'

            elif mode == 'pixel':
                results = extract_pixel_spectra(
                    tiff_path,
                    output_dir=output_dir,
                    save_csv=save_csv,
                    stride=args.stride,
                    n_jobs=args.jobs,
                    tilt_roi_name=args.tilt,
                    roi_path=temp_roi_path if args.tilt else None,
                    smooth=args.smooth,
                    prefix=args.prefix
                )
                return len(results), 'pixels'

            elif mode == 'grid':
                results = extract_grid_spectra(
                    tiff_path,
                    grid_size=args.grid_size,
                    output_dir=output_dir,
                    save_csv=save_csv,
                    n_jobs=args.jobs,
                    tilt_roi_name=args.tilt,
                    roi_path=temp_roi_path if args.tilt else None,
                    smooth=args.smooth,
                    prefix=args.prefix
                )
                return len(results), 'grid cells'

        finally:
            # Clean up temporary ROI file
            if temp_roi_path and temp_roi_path.exists():
                import shutil
                shutil.rmtree(temp_roi_path.parent)

    def extract_roi_spectra_with_merged_rois(tiff_path, merged_rois, output_dir, save_csv, tilt_roi_name, temp_roi_path, smooth=False, prefix=None):
        """Extract ROI spectra using pre-merged ROI list"""
        from .core import load_tiff_stack, apply_tilt_correction, get_roi_mask, calculate_spectrum, calculate_smooth_spectrum
        import warnings

        stack = load_tiff_stack(tiff_path)

        # Apply tilt correction if requested
        if tilt_roi_name is not None:
            if temp_roi_path is None:
                raise ValueError("temp_roi_path must be provided when using tilt correction")
            stack, angle, center = apply_tilt_correction(stack, temp_roi_path, tilt_roi_name)

        # Setup output directory
        if output_dir is None:
            output_dir = tiff_path.parent / f"{tiff_path.stem}_ROI_Spectra"
        else:
            output_dir = Path(output_dir)

        if save_csv:
            output_dir.mkdir(parents=True, exist_ok=True)

        # Pre-compute full-image spectrum once if smooth mode is active
        full_image_spectrum = stack.sum(axis=(1, 2)).astype(float) if smooth else None

        results = {}

        # Process each ROI from merged list
        for roi_info in merged_rois:
            roi_name = roi_info['name']
            roi_object = roi_info['roi_object']

            # Create mask for this ROI
            mask = get_roi_mask(roi_object, stack.shape[1:])

            if not mask.any():
                warnings.warn(f"ROI '{roi_name}' has no pixels, skipping")
                continue

            # Calculate spectrum
            if smooth:
                df = calculate_smooth_spectrum(stack, mask, full_image_spectrum)
            else:
                df = calculate_spectrum(stack, mask)
            results[roi_name] = df

            if save_csv:
                stem = f"{prefix}_{roi_name}" if prefix else roi_name
                csv_path = output_dir / f"{stem}.csv"
                df.to_csv(csv_path, index=False)

        return results

    # Execute based on mode with progress bar for multiple files
    try:
        print(f"\n{'='*60}")
        print(f"Processing {len(tiff_files)} TIFF file(s) in {mode} mode")
        print(f"{'='*60}\n")

        # Determine base output directory
        base_output_dir = Path(args.output) if args.output else None

        # Process each TIFF file with progress bar
        total_results = {}
        for tiff_path in tqdm(tiff_files, desc="Processing TIFFs", unit="file"):
            tqdm.write(f"\nProcessing: {tiff_path.name}")

            count, unit = process_single_tiff_with_merged_rois(
                tiff_path,
                all_rois,
                mode,
                args,
                base_output_dir,
                save_csv,
                custom_output_base
            )

            total_results[tiff_path.name] = (count, unit)
            tqdm.write(f"  Completed: {count} {unit}")

        # Summary
        print(f"\n{'='*60}")
        print("Processing Summary:")
        print(f"{'='*60}")
        for filename, (count, unit) in total_results.items():
            print(f"  {filename}: {count} {unit}")
        print(f"\nTotal files processed: {len(tiff_files)}")
        print("\nDone!")

    except Exception as e:
        import traceback
        print(f"\nError: {e}", file=sys.stderr)
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()
