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
)


def main():
    """Main CLI entry point"""
    parser = argparse.ArgumentParser(
        description="Extract spectral data from TIFF stacks using ImageJ ROIs or automated analysis",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Launch interactive ROI selection tool
  save-roi -t image.tiff -i

  # Launch interactive tool with specific stack range
  save-roi -t image.tiff -i --stack-range 0:10

  # Extract spectra using ImageJ ROI file
  save-roi --tiff image.tiff --roi roi_file.zip

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
        """
    )

    # Required arguments
    parser.add_argument(
        '-t', '--tiff',
        type=str,
        required=True,
        help='Path to TIFF stack file'
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
        '-i', '--interactive',
        action='store_true',
        help='Launch interactive ROI selection tool (requires Jupyter or IPython)'
    )

    parser.add_argument(
        '--stack-range',
        type=str,
        default=None,
        help='Stack range to sum for interactive mode, format: "start:end" (e.g., "0:10")'
    )

    parser.add_argument(
        '--version',
        action='version',
        version='%(prog)s 0.1.0'
    )

    args = parser.parse_args()

    # Validate inputs
    tiff_path = Path(args.tiff)
    if not tiff_path.exists():
        print(f"Error: TIFF file not found: {tiff_path}", file=sys.stderr)
        sys.exit(1)

    # Handle interactive mode
    if args.interactive:
        try:
            from .interactive import launch_interactive_tool

            # Parse stack range if provided
            stack_range = None
            if args.stack_range:
                try:
                    start, end = map(int, args.stack_range.split(':'))
                    stack_range = (start, end)
                except ValueError:
                    print(f"Error: Invalid stack range format. Use 'start:end' (e.g., '0:10')", file=sys.stderr)
                    sys.exit(1)

            print("Launching interactive ROI selection tool...")
            print("Use the drawing tools in the figure to create ROIs")
            print("Tip: Install with 'pip install -e .[interactive]' if dependencies are missing")

            selector = launch_interactive_tool(tiff_path, stack_range=stack_range, mode='jupyter')
            return  # Interactive mode doesn't exit automatically

        except ImportError as e:
            print(f"Error: Interactive mode requires additional dependencies.", file=sys.stderr)
            print(f"Install with: pip install save-roi[interactive]", file=sys.stderr)
            print(f"Details: {e}", file=sys.stderr)
            sys.exit(1)
        except Exception as e:
            print(f"Error launching interactive tool: {e}", file=sys.stderr)
            sys.exit(1)

    if args.roi:
        roi_path = Path(args.roi)
        if not roi_path.exists():
            print(f"Error: ROI file not found: {roi_path}", file=sys.stderr)
            sys.exit(1)
    else:
        roi_path = None

    # Determine mode
    mode = args.mode
    if mode == 'roi' and roi_path is None:
        print("Warning: No ROI file provided, switching to 'full' mode")
        mode = 'full'

    output_dir = Path(args.output) if args.output else None
    save_csv = not args.no_save

    # Execute based on mode
    try:
        print(f"Processing TIFF: {tiff_path}")

        if mode == 'roi':
            print(f"Using ROI file: {roi_path}")
            results = extract_roi_spectra(
                tiff_path,
                roi_path=roi_path,
                output_dir=output_dir,
                save_csv=save_csv
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
                n_jobs=args.jobs
            )
            print(f"\nProcessed {len(results)} pixels")

        elif mode == 'grid':
            print(f"Analyzing grid with {args.grid_size}x{args.grid_size} blocks")
            results = extract_grid_spectra(
                tiff_path,
                grid_size=args.grid_size,
                output_dir=output_dir,
                save_csv=save_csv,
                n_jobs=args.jobs
            )
            print(f"\nProcessed {len(results)} grid cells")

        print("\nDone!")

    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == '__main__':
    main()
