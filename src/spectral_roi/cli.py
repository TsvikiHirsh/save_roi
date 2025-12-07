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
)
import tifffile


def main():
    """Main CLI entry point"""
    parser = argparse.ArgumentParser(
        description="Extract spectral data from TIFF stacks using ImageJ ROIs or automated analysis",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Auto-discover folder structure (archive/final/*.tiff, *.roi.zip in parent)
  save-roi archive

  # Extract spectra using ImageJ ROI file
  save-roi --tiff image.tiff --roi roi_file.zip

  # Extract spectra with tilt correction
  save-roi --tiff image.tiff --roi roi_file.zip --tilt symmetry_line

  # Extract spectrum for entire image (no ROI)
  save-roi --tiff image.tiff

  # Extract spectra for 4x4 pixel grid
  save-roi --tiff image.tiff --mode grid --grid-size 4

  # Use all available cores for parallel processing
  save-roi --tiff image.tiff --mode grid --grid-size 4 --jobs -1

  # Specify custom output directory
  save-roi --tiff image.tiff --roi roi.zip --output ./results

  # Apply tilt correction and extract grid spectra
  save-roi --tiff image.tiff --roi roi.zip --tilt symmetry --mode grid

Folder structure mode:
  When using folder mode (e.g., 'save-roi archive'), the expected structure is:
    ./archive/              - Archive folder
    ./archive/final/        - TIFF stacks location
    ./archive/final/*.tiff  - TIFF files to process
    ./*.zip                 - ROI file (searched in current directory)
    ./archive/ROI_Spectra/  - Output directory (created automatically)
        """
    )

    # Positional argument for folder mode
    parser.add_argument(
        'folder',
        nargs='?',
        type=str,
        default=None,
        help='Archive folder name (enables auto-discovery mode)'
    )

    # Optional arguments
    parser.add_argument(
        '-t', '--tiff',
        type=str,
        default=None,
        help='Path to TIFF stack file (not needed in folder mode)'
    )

    # Optional arguments
    parser.add_argument(
        '-r', '--roi',
        type=str,
        default=None,
        help='Path to ImageJ ROI file (.roi or .zip). If not provided, analyzes full image or uses specified mode.'
    )

    parser.add_argument(
        '-o', '--output',
        type=str,
        default=None,
        help='Output directory for CSV files. Default: creates ROI_Spectra subfolder next to TIFF file.'
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

    # Determine mode: folder auto-discovery or explicit files
    if args.folder and not args.tiff:
        # Folder mode: auto-discover structure
        folder_mode = True
        archive_dir = Path(args.folder)

        if not archive_dir.exists() or not archive_dir.is_dir():
            print(f"Error: Archive folder not found: {archive_dir}", file=sys.stderr)
            sys.exit(1)

        # Look for final subfolder
        final_dir = archive_dir / "final"
        if not final_dir.exists() or not final_dir.is_dir():
            print(f"Error: 'final' subfolder not found in {archive_dir}", file=sys.stderr)
            print(f"Expected structure: {archive_dir}/final/*.tiff", file=sys.stderr)
            sys.exit(1)

        # Find TIFF files in final directory
        tiff_files = list(final_dir.glob("*.tiff")) + list(final_dir.glob("*.tif"))
        if not tiff_files:
            print(f"Error: No TIFF files found in {final_dir}", file=sys.stderr)
            sys.exit(1)

        print(f"Found {len(tiff_files)} TIFF file(s) in {final_dir}")

        # Look for ROI zip file in parent directory (current working directory)
        if args.roi:
            roi_path = Path(args.roi)
        else:
            # Auto-discover ROI file
            roi_files = list(Path.cwd().glob("*.zip"))
            # Filter for files that can be opened with roifile
            valid_roi_files = []
            for roi_file in roi_files:
                try:
                    from roifile import roiread
                    # Try to read it to verify it's a valid ROI file
                    roiread(str(roi_file))
                    valid_roi_files.append(roi_file)
                except Exception:
                    # Not a valid ROI file, skip it
                    pass

            if not valid_roi_files:
                print(f"Error: No valid ROI zip file found in {Path.cwd()}", file=sys.stderr)
                print("Expected: *.zip file in current directory", file=sys.stderr)
                sys.exit(1)
            elif len(valid_roi_files) > 1:
                print(f"Warning: Multiple ROI files found, using: {valid_roi_files[0]}")

            roi_path = valid_roi_files[0]
            print(f"Using ROI file: {roi_path}")

        if not roi_path.exists():
            print(f"Error: ROI file not found: {roi_path}", file=sys.stderr)
            sys.exit(1)

        # Set output directory to archive/ROI_Spectra
        if args.output:
            output_dir = Path(args.output)
        else:
            output_dir = archive_dir / "ROI_Spectra"

        print(f"Output directory: {output_dir}")

    elif args.tiff:
        # Explicit file mode
        folder_mode = False
        tiff_files = [Path(args.tiff)]

        if not tiff_files[0].exists():
            print(f"Error: TIFF file not found: {tiff_files[0]}", file=sys.stderr)
            sys.exit(1)

        # Validate tilt correction requirements
        if args.tilt and not args.roi:
            print("Error: --tilt requires --roi to be specified", file=sys.stderr)
            sys.exit(1)

        if args.roi:
            roi_path = Path(args.roi)
            if not roi_path.exists():
                print(f"Error: ROI file not found: {roi_path}", file=sys.stderr)
                sys.exit(1)
        else:
            roi_path = None

        output_dir = Path(args.output) if args.output else None

    else:
        print("Error: Either specify a folder name or use --tiff to specify a TIFF file", file=sys.stderr)
        parser.print_help()
        sys.exit(1)

    # Determine mode
    mode = args.mode
    if mode == 'roi' and roi_path is None:
        print("Warning: No ROI file provided, switching to 'full' mode")
        mode = 'full'

    save_csv = not args.no_save

    # Execute based on mode
    try:
        # Process all TIFF files
        for tiff_idx, tiff_path in enumerate(tiff_files, 1):
            if folder_mode and len(tiff_files) > 1:
                print(f"\n{'='*60}")
                print(f"Processing TIFF {tiff_idx}/{len(tiff_files)}: {tiff_path.name}")
                print(f"{'='*60}")
            else:
                print(f"Processing TIFF: {tiff_path}")

            if mode == 'roi':
                print(f"Using ROI file: {roi_path}")
                results = extract_roi_spectra(
                    tiff_path,
                    roi_path=roi_path,
                    output_dir=output_dir,
                    save_csv=save_csv,
                    tilt_roi_name=args.tilt
                )
                print(f"\nProcessed {len(results)} ROI(s)")

            elif mode == 'full':
                print("Analyzing full image")
                result = extract_full_image_spectrum(
                    tiff_path,
                    output_dir=output_dir,
                    save_csv=save_csv
                )
                print(f"\nExtracted spectrum with {len(result)} slices")

            elif mode == 'pixel':
                print(f"Analyzing pixels with stride={args.stride}")
                results = extract_pixel_spectra(
                    tiff_path,
                    output_dir=output_dir,
                    save_csv=save_csv,
                    stride=args.stride,
                    n_jobs=args.jobs,
                    tilt_roi_name=args.tilt,
                    roi_path=roi_path if args.tilt else None
                )
                print(f"\nProcessed {len(results)} pixels")

            elif mode == 'grid':
                print(f"Analyzing grid with {args.grid_size}x{args.grid_size} blocks")
                results = extract_grid_spectra(
                    tiff_path,
                    grid_size=args.grid_size,
                    output_dir=output_dir,
                    save_csv=save_csv,
                    n_jobs=args.jobs,
                    tilt_roi_name=args.tilt,
                    roi_path=roi_path if args.tilt else None
                )
                print(f"\nProcessed {len(results)} grid cells")

        if folder_mode and len(tiff_files) > 1:
            print(f"\n{'='*60}")
            print(f"All {len(tiff_files)} TIFF files processed successfully!")
            print(f"{'='*60}")

        print("\nDone!")

    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()
