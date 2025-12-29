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
    discover_directories_with_wildcard,
    merge_roi_files,
    sum_roi_spectra_from_folders,
)
import tifffile
from tqdm import tqdm


def main():
    """Main CLI entry point"""
    parser = argparse.ArgumentParser(
        description="Extract spectral data from TIFF stacks using ImageJ ROIs or automated analysis",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Quick workflow: discover TIFFs in data/final/, ROIs in PWD, output to data/
  save-roi data

  # Multi-folder workflow: analyze multiple folders and create SUM folder
  save-roi "data/run23_*"

  # Extract spectra using ImageJ ROI file
  save-roi --tiff image.tiff --roi roi_file.zip

  # Extract spectra with tilt correction
  save-roi --tiff image.tiff --roi roi_file.zip --tilt symmetry_line

  # Extract spectrum for entire image (no ROI)
  save-roi --tiff image.tiff

  # Extract spectra for 4x4 pixel grid
  save-roi --tiff image.tiff --mode grid --grid-size 4

  # Extract spectra for every pixel
  save-roi --tiff image.tiff --mode pixel

  # Extract spectra for every 4th pixel
  save-roi --tiff image.tiff --mode pixel --stride 4

  # Use all available cores for parallel processing
  save-roi --tiff image.tiff --mode grid --grid-size 4 --jobs -1

  # Specify custom output directory
  save-roi --tiff image.tiff --roi roi.zip --output ./results

  # Apply tilt correction and extract grid spectra
  save-roi --tiff image.tiff --roi roi.zip --tilt symmetry --mode grid
        """
    )

    # Positional argument for quick workflow
    parser.add_argument(
        'data_dir',
        nargs='?',
        default=None,
        help='Data directory for quick workflow. Searches for TIFFs in DATA_DIR/final/, ROIs in PWD, outputs to DATA_DIR/. '
             'Supports wildcards (e.g., "data/run23_*") for multi-folder analysis with automatic SUM folder creation.'
    )

    # Input arguments
    parser.add_argument(
        '-t', '--tiff',
        type=str,
        default=None,
        help='Path to TIFF stack file or directory. If directory, discovers all TIFF files. Default: current directory'
    )

    # Optional arguments
    parser.add_argument(
        '-r', '--roi',
        type=str,
        default=None,
        help='Path to ImageJ ROI file (.roi or .zip), or directory to discover ROI files. If not provided, auto-discovers ROI files in the same directory as TIFF files. Multiple ROI files will be merged automatically.'
    )

    parser.add_argument(
        '-o', '--output',
        type=str,
        default=None,
        help='Output directory for CSV files. Default: creates ROI_Spectra subfolder next to TIFF file.'
    )

    parser.add_argument(
        '-s', '--suffix',
        type=str,
        default=None,
        help='Optional subfolder name within the output directory. Useful for organizing different analyses (e.g., "corrected", "run1", "background").'
    )

    parser.add_argument(
        '-m', '--mode',
        type=str,
        choices=['roi', 'full', 'pixel', 'grid'],
        default='roi',
        help=(
            'Analysis mode: '
            'roi (use ImageJ ROI file), '
            'full (entire image), '
            'pixel (pixel-by-pixel or strided), '
            'grid (grid-based blocks). '
            'Default: roi if --roi is provided, otherwise full.'
        )
    )

    parser.add_argument(
        '--grid-size',
        type=int,
        default=4,
        help='Grid block size for grid mode (e.g., 4 for 4x4 blocks). Default: 4'
    )

    parser.add_argument(
        '--stride',
        type=int,
        default=1,
        help='Stride for pixel mode (e.g., 4 to sample every 4th pixel). Default: 1'
    )

    parser.add_argument(
        '-j', '--jobs',
        type=int,
        default=10,
        help='Number of parallel jobs for grid/pixel modes. Use -1 for all available cores. Default: 10'
    )

    parser.add_argument(
        '--no-save',
        action='store_true',
        help='Do not save CSV files (only useful for testing)'
    )

    parser.add_argument(
        '--tilt',
        type=str,
        default=None,
        help='Name of ROI to use for tilt correction. The ROI should define a symmetry line that will be straightened to vertical and centered.'
    )

    parser.add_argument(
        '--version',
        action='version',
        version='%(prog)s 0.1.0'
    )

    args = parser.parse_args()

    # Check if using quick workflow (positional data_dir argument)
    use_quick_workflow = args.data_dir is not None and args.tiff is None
    use_multi_folder_workflow = False
    data_dirs = []
    custom_output_base = None

    if use_quick_workflow:
        # Check if data_dir contains wildcards
        data_dir_str = args.data_dir
        if '*' in data_dir_str or '?' in data_dir_str:
            # Multi-folder workflow with wildcards
            use_multi_folder_workflow = True
            data_dirs = discover_directories_with_wildcard(data_dir_str)

            if not data_dirs:
                print(f"Error: No directories found matching pattern: {data_dir_str}", file=sys.stderr)
                sys.exit(1)

            print(f"Found {len(data_dirs)} directories matching pattern '{data_dir_str}':")
            for d in data_dirs:
                print(f"  - {d}")

        else:
            # Single directory quick workflow
            data_dir = Path(args.data_dir)
            if not data_dir.exists():
                print(f"Error: Data directory not found: {data_dir}", file=sys.stderr)
                sys.exit(1)
            if not data_dir.is_dir():
                print(f"Error: Data path is not a directory: {data_dir}", file=sys.stderr)
                sys.exit(1)

            data_dirs = [data_dir]
            print(f"Using quick workflow with data directory: {data_dir}")

        # For both single and multi-folder workflows
        if not use_multi_folder_workflow:
            data_dir = data_dirs[0]
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

    # Multi-folder workflow - process each directory separately
    if use_multi_folder_workflow:
        print(f"\n{'='*60}")
        print(f"Multi-Folder Analysis Mode")
        print(f"Processing {len(data_dirs)} directories separately")
        print(f"{'='*60}\n")

        # ROI search defaults to PWD for multi-folder workflow
        if args.roi is None:
            roi_search_dir = Path.cwd()
            print(f"Searching for ROI files in: {roi_search_dir}")
        else:
            roi_search_dir = Path(args.roi)

        # Discover and merge ROI files once (shared across all folders)
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
            roi_files = [roi_search_dir]
            print(f"\nUsing ROI file: {roi_search_dir}")
        else:
            print(f"Error: ROI path not found: {roi_search_dir}", file=sys.stderr)
            sys.exit(1)

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

        # Process each directory
        roi_spectra_output_dirs = []

        for data_dir in data_dirs:
            print(f"\n{'='*60}")
            print(f"Processing directory: {data_dir}")
            print(f"{'='*60}\n")

            # Look for TIFFs in data_dir/final/
            tiff_search_dir = data_dir / "final"
            if not tiff_search_dir.exists():
                print(f"Warning: Skipping {data_dir} - no 'final' subdirectory found", file=sys.stderr)
                continue

            # Discover TIFF files
            tiff_files = discover_tiff_files(tiff_search_dir)
            if not tiff_files:
                print(f"Warning: Skipping {data_dir} - no TIFF files found in {tiff_search_dir}", file=sys.stderr)
                continue

            print(f"Discovered {len(tiff_files)} TIFF file(s) in {tiff_search_dir}")
            for tf in tiff_files:
                print(f"  - {tf.name}")

            # Process TIFFs in this directory
            custom_output_base = data_dir
            base_output_dir = Path(args.output) if args.output else None

            # Import the helper function (defined later in the file)
            from types import FunctionType

            # Process each TIFF file
            for tiff_path in tqdm(tiff_files, desc=f"Processing {data_dir.name}", unit="file"):
                tqdm.write(f"\nProcessing: {tiff_path.name}")

                # Determine output directory for this specific data_dir
                if base_output_dir is not None:
                    output_dir = base_output_dir
                    if args.suffix:
                        output_dir = output_dir / args.suffix
                else:
                    # Output to data_dir/tiff_name_ROI_Spectra/
                    output_dir = custom_output_base / f"{tiff_path.stem}_ROI_Spectra"
                    if args.suffix:
                        output_dir = output_dir / args.suffix

                # Store this output directory for later summing
                if output_dir not in roi_spectra_output_dirs:
                    roi_spectra_output_dirs.append(output_dir)

                # Process this TIFF (we'll call the processing function directly)
                try:
                    import tempfile
                    import zipfile
                    from roifile import roiwrite

                    # Create temporary ROI file if needed for tilt correction
                    temp_roi_path = None
                    if args.tilt and all_rois:
                        temp_dir = Path(tempfile.mkdtemp())
                        temp_roi_path = temp_dir / "merged_rois.zip"

                        with zipfile.ZipFile(temp_roi_path, 'w') as zf:
                            for roi_info in all_rois:
                                roi_obj = roi_info['roi_object']
                                roi_name = roi_info['name']
                                temp_roi_file = temp_dir / f"{roi_name}.roi"
                                roiwrite(str(temp_roi_file), roi_obj)
                                zf.write(temp_roi_file, arcname=f"{roi_name}.roi")

                    # Process based on mode
                    if mode == 'roi':
                        from .core import get_roi_mask, calculate_spectrum

                        stack = load_tiff_stack(tiff_path)

                        if args.tilt and temp_roi_path:
                            stack, angle, center = apply_tilt_correction(stack, temp_roi_path, args.tilt)

                        output_dir.mkdir(parents=True, exist_ok=True)

                        for roi_info in all_rois:
                            roi_name = roi_info['name']
                            roi_object = roi_info['roi_object']
                            mask = get_roi_mask(roi_object, stack.shape[1:])

                            if not mask.any():
                                continue

                            df = calculate_spectrum(stack, mask)

                            if save_csv:
                                csv_path = output_dir / f"{roi_name}.csv"
                                df.to_csv(csv_path, index=False)

                        count = len(all_rois)
                        unit = 'ROI(s)'

                    elif mode == 'full':
                        result = extract_full_image_spectrum(
                            tiff_path,
                            output_dir=output_dir,
                            save_csv=save_csv
                        )
                        count = len(result)
                        unit = 'slices'

                    elif mode == 'pixel':
                        results = extract_pixel_spectra(
                            tiff_path,
                            output_dir=output_dir,
                            save_csv=save_csv,
                            stride=args.stride,
                            n_jobs=args.jobs,
                            tilt_roi_name=args.tilt,
                            roi_path=temp_roi_path if args.tilt else None
                        )
                        count = len(results)
                        unit = 'pixels'

                    elif mode == 'grid':
                        results = extract_grid_spectra(
                            tiff_path,
                            grid_size=args.grid_size,
                            output_dir=output_dir,
                            save_csv=save_csv,
                            n_jobs=args.jobs,
                            tilt_roi_name=args.tilt,
                            roi_path=temp_roi_path if args.tilt else None
                        )
                        count = len(results)
                        unit = 'grid cells'

                    tqdm.write(f"  Completed: {count} {unit}")

                    # Clean up temp ROI file
                    if temp_roi_path and temp_roi_path.exists():
                        import shutil
                        shutil.rmtree(temp_roi_path.parent)

                except Exception as e:
                    import traceback
                    print(f"\nError processing {tiff_path}: {e}", file=sys.stderr)
                    traceback.print_exc()
                    continue

        # After processing all directories, create SUM folder
        if roi_spectra_output_dirs:
            print(f"\n{'='*60}")
            print(f"Creating SUM folder")
            print(f"{'='*60}\n")

            # Determine parent directory for SUM folder
            # We'll place it alongside the first ROI spectra directory
            if roi_spectra_output_dirs:
                first_output_dir = roi_spectra_output_dirs[0]
                sum_output_dir = first_output_dir.parent / "SUM"
                if args.suffix:
                    sum_output_dir = sum_output_dir / args.suffix

                try:
                    sum_roi_spectra_from_folders(
                        roi_spectra_dirs=roi_spectra_output_dirs,
                        output_dir=sum_output_dir,
                        save_csv=save_csv
                    )
                    print(f"\nSUM folder created at: {sum_output_dir}")
                except Exception as e:
                    import traceback
                    print(f"\nError creating SUM folder: {e}", file=sys.stderr)
                    traceback.print_exc()

        print(f"\n{'='*60}")
        print(f"Multi-Folder Analysis Complete!")
        print(f"Processed {len(data_dirs)} directories")
        if roi_spectra_output_dirs:
            print(f"Created SUM folder at: {sum_output_dir}")
        print(f"{'='*60}\n")

        sys.exit(0)

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
                    temp_roi_path=temp_roi_path
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
                    roi_path=temp_roi_path if args.tilt else None
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
                    roi_path=temp_roi_path if args.tilt else None
                )
                return len(results), 'grid cells'

        finally:
            # Clean up temporary ROI file
            if temp_roi_path and temp_roi_path.exists():
                import shutil
                shutil.rmtree(temp_roi_path.parent)

    def extract_roi_spectra_with_merged_rois(tiff_path, merged_rois, output_dir, save_csv, tilt_roi_name, temp_roi_path):
        """Extract ROI spectra using pre-merged ROI list"""
        from .core import load_tiff_stack, apply_tilt_correction, get_roi_mask, calculate_spectrum
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
            df = calculate_spectrum(stack, mask)
            results[roi_name] = df

            if save_csv:
                csv_path = output_dir / f"{roi_name}.csv"
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
