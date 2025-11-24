#!/usr/bin/env python3
"""
Setup script to extract test data for save_roi package
"""

import gzip
import shutil
from pathlib import Path


def setup_test_data():
    """Extract compressed test TIFF file if needed"""
    notebooks_dir = Path(__file__).parent.parent / "notebooks"
    tiff_gz = notebooks_dir / "image.tiff.gz"
    tiff_file = notebooks_dir / "image.tiff"

    if not tiff_gz.exists():
        print(f"Compressed test data not found: {tiff_gz}")
        return False

    if tiff_file.exists():
        print(f"Test data already extracted: {tiff_file}")
        return True

    print(f"Extracting test data: {tiff_gz} -> {tiff_file}")
    try:
        with gzip.open(tiff_gz, 'rb') as f_in:
            with open(tiff_file, 'wb') as f_out:
                shutil.copyfileobj(f_in, f_out)
        print(f"Successfully extracted test data ({tiff_file.stat().st_size / 1024 / 1024:.1f} MB)")
        return True
    except Exception as e:
        print(f"Error extracting test data: {e}")
        return False


if __name__ == "__main__":
    setup_test_data()
