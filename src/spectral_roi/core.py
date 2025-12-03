"""
Core functionality for extracting spectral data from TIFF stacks
"""

import numpy as np
import pandas as pd
import tifffile
from pathlib import Path
from typing import Union, List, Optional, Tuple
import warnings
from multiprocessing import Pool, cpu_count
from functools import partial
from scipy.ndimage import rotate
from scipy.optimize import fsolve
from scipy.interpolate import UnivariateSpline


def load_tiff_stack(tiff_path: Union[str, Path]) -> np.ndarray:
    """
    Load a TIFF stack from file.

    Parameters
    ----------
    tiff_path : str or Path
        Path to the TIFF file

    Returns
    -------
    np.ndarray
        3D array with shape (slices, height, width) or (height, width) for single slice
    """
    tiff_path = Path(tiff_path)
    if not tiff_path.exists():
        raise FileNotFoundError(f"TIFF file not found: {tiff_path}")

    data = tifffile.imread(str(tiff_path))

    # Ensure 3D array (slices, height, width)
    if data.ndim == 2:
        data = data[np.newaxis, :, :]
    elif data.ndim > 3:
        # Handle multi-channel images - take first channel
        warnings.warn(f"Image has {data.ndim} dimensions, using first channel only")
        data = data[0] if data.shape[0] < data.shape[-1] else data[..., 0]

    return data


def load_imagej_rois(roi_path: Union[str, Path]) -> List[dict]:
    """
    Load ImageJ ROIs from a .roi or .zip file.

    Parameters
    ----------
    roi_path : str or Path
        Path to the ROI file (.roi or .zip)

    Returns
    -------
    List[dict]
        List of ROI dictionaries containing name and coordinates
    """
    from roifile import ImagejRoi, roiread

    roi_path = Path(roi_path)
    if not roi_path.exists():
        raise FileNotFoundError(f"ROI file not found: {roi_path}")

    rois = []

    if roi_path.suffix.lower() == '.zip':
        # Load multiple ROIs from zip
        roi_list = roiread(str(roi_path))
        if not isinstance(roi_list, list):
            roi_list = [roi_list]

        for idx, roi in enumerate(roi_list):
            roi_name = getattr(roi, 'name', f'ROI_{idx + 1}')
            rois.append({
                'name': roi_name,
                'roi_object': roi,
            })
    else:
        # Load single ROI
        roi = ImagejRoi.fromfile(str(roi_path))
        roi_name = getattr(roi, 'name', 'ROI_1')
        rois.append({
            'name': roi_name,
            'roi_object': roi,
        })

    return rois


def get_roi_mask(roi_object, image_shape: Tuple[int, int]) -> np.ndarray:
    """
    Create a binary mask from an ImageJ ROI object.

    Parameters
    ----------
    roi_object : ImagejRoi
        ROI object from roifile
    image_shape : tuple
        (height, width) of the image

    Returns
    -------
    np.ndarray
        Binary mask of shape (height, width)
    """
    mask = np.zeros(image_shape, dtype=bool)

    # Get ROI coordinates
    if hasattr(roi_object, 'coordinates'):
        coords = roi_object.coordinates()
        if coords is not None and len(coords) > 0:
            # Handle different ROI types
            if roi_object.roitype in [0, 1, 2, 3]:  # Polygon, rectangle, oval, line
                from matplotlib.path import Path as MplPath

                # Create path from coordinates
                path = MplPath(coords)

                # Create grid of points
                height, width = image_shape
                y, x = np.mgrid[:height, :width]
                points = np.vstack((x.ravel(), y.ravel())).T

                # Test which points are inside the path
                mask_flat = path.contains_points(points)
                mask = mask_flat.reshape(image_shape)

            elif roi_object.roitype == 10:  # Point
                # For point ROIs, use the coordinates directly
                for coord in coords:
                    x, y = int(coord[0]), int(coord[1])
                    if 0 <= y < image_shape[0] and 0 <= x < image_shape[1]:
                        mask[y, x] = True

    # Fallback: try to get mask directly from ROI if available
    if not mask.any() and hasattr(roi_object, 'mask'):
        roi_mask = roi_object.mask()
        if roi_mask is not None:
            # Pad or crop to match image shape
            mask = np.zeros(image_shape, dtype=bool)
            h, w = min(roi_mask.shape[0], image_shape[0]), min(roi_mask.shape[1], image_shape[1])
            mask[:h, :w] = roi_mask[:h, :w]

    return mask


def _process_single_mask(args):
    """
    Helper function for parallel processing of masks.

    Parameters
    ----------
    args : tuple
        (mask_name, mask, stack) tuple

    Returns
    -------
    tuple
        (mask_name, DataFrame) tuple
    """
    mask_name, mask, stack = args
    df = calculate_spectrum(stack, mask)
    return mask_name, df


def calculate_spectrum(stack: np.ndarray, mask: np.ndarray) -> pd.DataFrame:
    """
    Calculate spectral profile for a masked region.

    Parameters
    ----------
    stack : np.ndarray
        3D array (slices, height, width)
    mask : np.ndarray
        2D binary mask (height, width)

    Returns
    -------
    pd.DataFrame
        DataFrame with columns: stack, counts, err
    """
    if stack.ndim == 2:
        stack = stack[np.newaxis, :, :]

    n_slices = stack.shape[0]
    results = []

    for z in range(n_slices):
        slice_data = stack[z, :, :]
        masked_data = slice_data[mask]

        if len(masked_data) > 0:
            # Calculate statistics
            counts = np.sum(masked_data)
            err = np.sqrt(counts) if counts > 0 else 0
        else:
            counts = 0
            err = 0

        results.append({
            'stack': z + 1,  # 1-indexed like ImageJ
            'counts': counts,
            'err': err
        })

    return pd.DataFrame(results)


def apply_tilt_correction(
    stack: np.ndarray,
    roi_path: Union[str, Path],
    tilt_roi_name: str,
    threshold: float = 0.4
) -> Tuple[np.ndarray, float, Tuple[float, float]]:
    """
    Apply tilt correction to a TIFF stack based on a symmetry line ROI.

    This function finds a symmetry line from the specified ROI, calculates
    its angle from vertical, rotates the entire stack to make it vertical,
    and centers the image.

    Parameters
    ----------
    stack : np.ndarray
        3D array (slices, height, width)
    roi_path : str or Path
        Path to ImageJ ROI file (.roi or .zip)
    tilt_roi_name : str
        Name of the ROI to use for tilt correction
    threshold : float, default=0.4
        Threshold value for finding symmetry line edges

    Returns
    -------
    corrected_stack : np.ndarray
        Tilt-corrected and centered stack
    angle : float
        Rotation angle applied (in degrees)
    center : tuple
        (x, y) coordinates of the symmetry line center
    """
    # Load the specified ROI
    rois = load_imagej_rois(roi_path)

    # Find the tilt ROI
    tilt_roi = None
    for roi_info in rois:
        if roi_info['name'] == tilt_roi_name:
            tilt_roi = roi_info['roi_object']
            break

    if tilt_roi is None:
        raise ValueError(f"ROI '{tilt_roi_name}' not found in {roi_path}")

    # Get ROI coordinates
    if not hasattr(tilt_roi, 'coordinates'):
        raise ValueError(f"ROI '{tilt_roi_name}' does not have coordinates")

    coords = tilt_roi.coordinates()
    if coords is None or len(coords) < 2:
        raise ValueError(f"ROI '{tilt_roi_name}' must have at least 2 points")

    # Calculate angle from vertical
    # For a line ROI, use first and last points
    # Or for more complex ROIs, fit a line through all points
    if len(coords) == 2:
        # Simple line ROI
        x1, y1 = coords[0]
        x2, y2 = coords[1]
    else:
        # Fit line through multiple points using least squares
        x_coords = coords[:, 0]
        y_coords = coords[:, 1]

        # Fit line: y = mx + b
        # Using polyfit for linear regression
        if np.std(x_coords) > np.std(y_coords):
            # More variation in x, fit y = mx + b
            m, b = np.polyfit(x_coords, y_coords, 1)
            x1, y1 = x_coords[0], m * x_coords[0] + b
            x2, y2 = x_coords[-1], m * x_coords[-1] + b
        else:
            # More variation in y, fit x = my + b
            m, b = np.polyfit(y_coords, x_coords, 1)
            y1, y2 = y_coords[0], y_coords[-1]
            x1, x2 = m * y1 + b, m * y2 + b

    # Calculate angle from vertical (vertical = 0 degrees)
    # Vector from point 1 to point 2
    dx = x2 - x1
    dy = y2 - y1

    # Angle from vertical (0 degrees is straight down)
    # We want to rotate so the line becomes vertical
    angle_from_horizontal = np.degrees(np.arctan2(dy, dx))
    angle_from_vertical = angle_from_horizontal - 90.0

    # Rotation angle needed to make line vertical
    rotation_angle = -angle_from_vertical

    print(f"Symmetry line angle from vertical: {angle_from_vertical:.2f} degrees")
    print(f"Applying rotation: {rotation_angle:.2f} degrees")

    # Calculate center of symmetry line
    center_x = (x1 + x2) / 2
    center_y = (y1 + y2) / 2

    # Rotate the entire stack
    corrected_stack = np.zeros_like(stack)
    for i in range(stack.shape[0]):
        corrected_stack[i] = rotate(
            stack[i],
            rotation_angle,
            reshape=False,
            order=3,  # Bicubic interpolation
            mode='constant',
            cval=0
        )

    # Calculate new center position after rotation
    height, width = stack.shape[1:]
    old_center = np.array([width / 2, height / 2])

    # Rotation matrix
    theta = np.radians(rotation_angle)
    rot_matrix = np.array([
        [np.cos(theta), -np.sin(theta)],
        [np.sin(theta), np.cos(theta)]
    ])

    # Calculate where the symmetry line center moved to
    relative_pos = np.array([center_x, center_y]) - old_center
    new_relative_pos = rot_matrix @ relative_pos
    new_center = old_center + new_relative_pos

    # Calculate shift to center the symmetry line
    shift_x = int(width / 2 - new_center[0])
    shift_y = int(height / 2 - new_center[1])

    print(f"Centering: shifting by ({shift_x}, {shift_y}) pixels")

    # Apply centering shift
    centered_stack = np.zeros_like(corrected_stack)
    for i in range(corrected_stack.shape[0]):
        centered_stack[i] = np.roll(corrected_stack[i], (shift_y, shift_x), axis=(0, 1))

    return centered_stack, rotation_angle, (center_x, center_y)


def extract_roi_spectra(
    tiff_path: Union[str, Path],
    roi_path: Optional[Union[str, Path]] = None,
    output_dir: Optional[Union[str, Path]] = None,
    save_csv: bool = True,
    tilt_roi_name: Optional[str] = None
) -> dict:
    """
    Extract spectral data for ROIs from a TIFF stack.

    Parameters
    ----------
    tiff_path : str or Path
        Path to the TIFF stack file
    roi_path : str or Path, optional
        Path to ImageJ ROI file (.roi or .zip). If None, analyzes full image.
    output_dir : str or Path, optional
        Directory to save CSV files. If None, creates ROI_Spectra subfolder
        next to the TIFF file.
    save_csv : bool, default=True
        Whether to save results to CSV files
    tilt_roi_name : str, optional
        Name of ROI to use for tilt correction. If provided, applies tilt
        correction before extracting spectra.

    Returns
    -------
    dict
        Dictionary mapping ROI names to DataFrames with spectral data
    """
    tiff_path = Path(tiff_path)
    stack = load_tiff_stack(tiff_path)

    # Apply tilt correction if requested
    if tilt_roi_name is not None:
        if roi_path is None:
            raise ValueError("roi_path must be provided when using tilt correction")
        print(f"\nApplying tilt correction using ROI: {tilt_roi_name}")
        stack, angle, center = apply_tilt_correction(stack, roi_path, tilt_roi_name)
        print(f"Tilt correction complete: rotated {angle:.2f} degrees, centered at {center}\n")

    # Setup output directory
    if output_dir is None:
        output_dir = tiff_path.parent / f"{tiff_path.stem}_ROI_Spectra"
    else:
        output_dir = Path(output_dir)

    if save_csv:
        output_dir.mkdir(parents=True, exist_ok=True)

    results = {}

    if roi_path is None:
        # Analyze full image
        full_mask = np.ones(stack.shape[1:], dtype=bool)
        df = calculate_spectrum(stack, full_mask)
        results['full_image'] = df

        if save_csv:
            csv_path = output_dir / 'full_image.csv'
            df.to_csv(csv_path, index=False)
            print(f"Saved: {csv_path}")
    else:
        # Load and process ROIs
        rois = load_imagej_rois(roi_path)
        print(f"Found {len(rois)} ROI(s)")

        for roi_info in rois:
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
                print(f"Saved: {csv_path}")

    return results


def extract_full_image_spectrum(
    tiff_path: Union[str, Path],
    output_dir: Optional[Union[str, Path]] = None,
    save_csv: bool = True
) -> pd.DataFrame:
    """
    Extract spectral data for the entire image (no ROI).

    Parameters
    ----------
    tiff_path : str or Path
        Path to the TIFF stack file
    output_dir : str or Path, optional
        Directory to save CSV file
    save_csv : bool, default=True
        Whether to save results to CSV

    Returns
    -------
    pd.DataFrame
        DataFrame with spectral data
    """
    results = extract_roi_spectra(tiff_path, roi_path=None, output_dir=output_dir, save_csv=save_csv)
    return results['full_image']


def extract_pixel_spectra(
    tiff_path: Union[str, Path],
    output_dir: Optional[Union[str, Path]] = None,
    save_csv: bool = True,
    stride: int = 1,
    n_jobs: int = 10,
    tilt_roi_name: Optional[str] = None,
    roi_path: Optional[Union[str, Path]] = None
) -> dict:
    """
    Extract spectral data for individual pixels or pixel groups on a grid.

    Parameters
    ----------
    tiff_path : str or Path
        Path to the TIFF stack file
    output_dir : str or Path, optional
        Directory to save CSV files
    save_csv : bool, default=True
        Whether to save results to CSV files
    stride : int, default=1
        Stride for grid sampling. stride=1 means every pixel, stride=4 means
        every 4th pixel in each direction.
    n_jobs : int, default=10
        Number of parallel jobs. Use -1 for all available cores.
    tilt_roi_name : str, optional
        Name of ROI to use for tilt correction.
    roi_path : str or Path, optional
        Path to ROI file (required if tilt_roi_name is provided).

    Returns
    -------
    dict
        Dictionary mapping pixel coordinates to DataFrames with spectral data
    """
    tiff_path = Path(tiff_path)
    stack = load_tiff_stack(tiff_path)

    # Apply tilt correction if requested
    if tilt_roi_name is not None:
        if roi_path is None:
            raise ValueError("roi_path must be provided when using tilt correction")
        print(f"\nApplying tilt correction using ROI: {tilt_roi_name}")
        stack, angle, center = apply_tilt_correction(stack, roi_path, tilt_roi_name)
        print(f"Tilt correction complete: rotated {angle:.2f} degrees, centered at {center}\n")

    # Setup output directory
    if output_dir is None:
        output_dir = tiff_path.parent / f"{tiff_path.stem}_ROI_Spectra"
    else:
        output_dir = Path(output_dir)

    if save_csv:
        output_dir.mkdir(parents=True, exist_ok=True)

    height, width = stack.shape[1:]

    # Determine number of jobs
    if n_jobs == -1:
        n_jobs = cpu_count()
    else:
        n_jobs = min(n_jobs, cpu_count())

    print(f"Extracting pixel spectra with stride={stride} using {n_jobs} cores...")

    # Prepare all masks and their names
    tasks = []
    for y in range(0, height, stride):
        for x in range(0, width, stride):
            # Create single-pixel mask
            mask = np.zeros((height, width), dtype=bool)
            mask[y, x] = True
            pixel_name = f"pixel_x{x}_y{y}"
            tasks.append((pixel_name, mask, stack))

    # Process in parallel
    with Pool(processes=n_jobs) as pool:
        results_list = pool.map(_process_single_mask, tasks)

    # Convert to dictionary
    results = dict(results_list)

    # Save CSVs if requested
    if save_csv:
        for pixel_name, df in results.items():
            csv_path = output_dir / f"{pixel_name}.csv"
            df.to_csv(csv_path, index=False)

    print(f"Saved {len(results)} pixel spectra to {output_dir}")

    return results


def extract_grid_spectra(
    tiff_path: Union[str, Path],
    grid_size: int = 4,
    output_dir: Optional[Union[str, Path]] = None,
    save_csv: bool = True,
    n_jobs: int = 10,
    tilt_roi_name: Optional[str] = None,
    roi_path: Optional[Union[str, Path]] = None
) -> dict:
    """
    Extract spectral data for grid-based pixel groups (e.g., 4x4 blocks).

    Parameters
    ----------
    tiff_path : str or Path
        Path to the TIFF stack file
    grid_size : int, default=4
        Size of grid blocks (e.g., 4 means 4x4 pixel blocks)
    output_dir : str or Path, optional
        Directory to save CSV files
    save_csv : bool, default=True
        Whether to save results to CSV files
    n_jobs : int, default=10
        Number of parallel jobs. Use -1 for all available cores.
    tilt_roi_name : str, optional
        Name of ROI to use for tilt correction.
    roi_path : str or Path, optional
        Path to ROI file (required if tilt_roi_name is provided).

    Returns
    -------
    dict
        Dictionary mapping grid cell coordinates to DataFrames with spectral data
    """
    tiff_path = Path(tiff_path)
    stack = load_tiff_stack(tiff_path)

    # Apply tilt correction if requested
    if tilt_roi_name is not None:
        if roi_path is None:
            raise ValueError("roi_path must be provided when using tilt correction")
        print(f"\nApplying tilt correction using ROI: {tilt_roi_name}")
        stack, angle, center = apply_tilt_correction(stack, roi_path, tilt_roi_name)
        print(f"Tilt correction complete: rotated {angle:.2f} degrees, centered at {center}\n")

    # Setup output directory
    if output_dir is None:
        output_dir = tiff_path.parent / f"{tiff_path.stem}_ROI_Spectra"
    else:
        output_dir = Path(output_dir)

    if save_csv:
        output_dir.mkdir(parents=True, exist_ok=True)

    height, width = stack.shape[1:]

    # Determine number of jobs
    if n_jobs == -1:
        n_jobs = cpu_count()
    else:
        n_jobs = min(n_jobs, cpu_count())

    print(f"Extracting grid spectra with {grid_size}x{grid_size} blocks using {n_jobs} cores...")

    # Prepare all masks and their names
    tasks = []
    for y in range(0, height, grid_size):
        for x in range(0, width, grid_size):
            # Create grid block mask
            mask = np.zeros((height, width), dtype=bool)
            y_end = min(y + grid_size, height)
            x_end = min(x + grid_size, width)
            mask[y:y_end, x:x_end] = True

            grid_name = f"grid_{grid_size}x{grid_size}_x{x}_y{y}"
            tasks.append((grid_name, mask, stack))

    # Process in parallel
    with Pool(processes=n_jobs) as pool:
        results_list = pool.map(_process_single_mask, tasks)

    # Convert to dictionary
    results = dict(results_list)

    # Save CSVs if requested
    if save_csv:
        for grid_name, df in results.items():
            csv_path = output_dir / f"{grid_name}.csv"
            df.to_csv(csv_path, index=False)

    print(f"Saved {len(results)} grid spectra to {output_dir}")

    return results
