"""
Tests for save_roi core functionality
"""

import pytest
import numpy as np
import pandas as pd
from pathlib import Path
import tempfile
import shutil

from save_roi import (
    extract_roi_spectra,
    extract_full_image_spectrum,
    extract_pixel_spectra,
    extract_grid_spectra,
)
from save_roi.core import (
    load_tiff_stack,
    load_imagej_rois,
    calculate_spectrum,
    calculate_smooth_spectrum,
)


@pytest.fixture
def test_data_dir():
    """Return path to test data directory"""
    return Path(__file__).parent.parent / "notebooks"


@pytest.fixture
def test_tiff_path(test_data_dir):
    """Return path to test TIFF file"""
    tiff_path = test_data_dir / "image.tiff"
    if not tiff_path.exists():
        pytest.skip(f"Test TIFF file not found: {tiff_path}")
    return tiff_path


@pytest.fixture
def test_roi_path(test_data_dir):
    """Return path to test ROI file"""
    roi_path = test_data_dir / "ROI2.zip"
    if not roi_path.exists():
        pytest.skip(f"Test ROI file not found: {roi_path}")
    return roi_path


@pytest.fixture
def temp_output_dir():
    """Create temporary output directory"""
    temp_dir = tempfile.mkdtemp()
    yield Path(temp_dir)
    # Cleanup after test
    shutil.rmtree(temp_dir)


class TestLoadTiffStack:
    """Tests for load_tiff_stack function"""

    def test_load_tiff_stack(self, test_tiff_path):
        """Test loading a TIFF stack"""
        stack = load_tiff_stack(test_tiff_path)
        assert isinstance(stack, np.ndarray)
        assert stack.ndim == 3  # Should be 3D (slices, height, width)
        assert stack.shape[0] > 0  # Should have at least one slice
        assert stack.shape[1] > 0  # Should have height
        assert stack.shape[2] > 0  # Should have width

    def test_load_nonexistent_file(self):
        """Test loading a non-existent file raises error"""
        with pytest.raises(FileNotFoundError):
            load_tiff_stack("nonexistent.tiff")


class TestLoadImagejRois:
    """Tests for load_imagej_rois function"""

    def test_load_roi_zip(self, test_roi_path):
        """Test loading ROIs from zip file"""
        rois = load_imagej_rois(test_roi_path)
        assert isinstance(rois, list)
        assert len(rois) > 0
        # Check structure of first ROI
        assert 'name' in rois[0]
        assert 'roi_object' in rois[0]

    def test_load_nonexistent_roi(self):
        """Test loading a non-existent ROI file raises error"""
        with pytest.raises(FileNotFoundError):
            load_imagej_rois("nonexistent.roi")


class TestCalculateSpectrum:
    """Tests for calculate_spectrum function"""

    def test_calculate_spectrum_full_mask(self):
        """Test spectrum calculation with full mask"""
        # Create simple test stack: 3 slices, 10x10 pixels
        stack = np.ones((3, 10, 10))
        stack[0, :, :] = 10
        stack[1, :, :] = 20
        stack[2, :, :] = 30

        # Full mask
        mask = np.ones((10, 10), dtype=bool)

        df = calculate_spectrum(stack, mask)

        assert isinstance(df, pd.DataFrame)
        assert len(df) == 3  # 3 slices
        assert list(df.columns) == ['stack', 'counts', 'err']
        assert df['stack'].tolist() == [1, 2, 3]
        assert df['counts'].tolist() == [1000, 2000, 3000]  # 100 pixels each

    def test_calculate_spectrum_partial_mask(self):
        """Test spectrum calculation with partial mask"""
        stack = np.ones((2, 10, 10))
        stack[0, :, :] = 10
        stack[1, :, :] = 20

        # Only select top-left 5x5 region
        mask = np.zeros((10, 10), dtype=bool)
        mask[:5, :5] = True  # 25 pixels

        df = calculate_spectrum(stack, mask)

        assert len(df) == 2
        assert df['counts'].tolist() == [250, 500]  # 25 pixels each

    def test_calculate_spectrum_empty_mask(self):
        """Test spectrum calculation with empty mask"""
        stack = np.ones((2, 10, 10))
        mask = np.zeros((10, 10), dtype=bool)

        df = calculate_spectrum(stack, mask)

        assert len(df) == 2
        assert df['counts'].tolist() == [0, 0]
        assert df['err'].tolist() == [0, 0]


class TestCalculateSmoothSpectrum:
    """Tests for calculate_smooth_spectrum function"""

    def _full_spectrum(self, stack):
        return stack.sum(axis=(1, 2)).astype(float)

    def test_output_shape_and_columns(self):
        """Smoothed spectrum has the right shape and column names"""
        stack = np.ones((5, 10, 10))
        mask = np.ones((10, 10), dtype=bool)
        S = self._full_spectrum(stack)
        df = calculate_smooth_spectrum(stack, mask, S)
        assert list(df.columns) == ['stack', 'counts', 'err']
        assert len(df) == 5
        assert df['stack'].tolist() == [1, 2, 3, 4, 5]

    def test_full_mask_preserves_total_counts(self):
        """With a full-image mask the smoothed counts equal the raw counts"""
        rng = np.random.default_rng(42)
        stack = rng.poisson(50, size=(8, 12, 12)).astype(float)
        mask = np.ones((12, 12), dtype=bool)
        S = self._full_spectrum(stack)
        df_smooth = calculate_smooth_spectrum(stack, mask, S)
        df_raw = calculate_spectrum(stack, mask)
        # For a full mask, I_smooth(t) = N * S(t)/S_total = S_total * S(t)/S_total = S(t)
        np.testing.assert_allclose(df_smooth['counts'].values, df_raw['counts'].values, rtol=1e-10)

    def test_spatial_scaling(self):
        """Smoothed spectrum scales proportionally with local intensity"""
        # Two regions: left half has 2x the counts of right half
        stack = np.zeros((4, 10, 10))
        stack[:, :, :5] = 2.0   # left half
        stack[:, :, 5:] = 1.0   # right half

        mask_left = np.zeros((10, 10), dtype=bool)
        mask_left[:, :5] = True
        mask_right = np.zeros((10, 10), dtype=bool)
        mask_right[:, 5:] = True

        S = self._full_spectrum(stack)
        df_left = calculate_smooth_spectrum(stack, mask_left, S)
        df_right = calculate_smooth_spectrum(stack, mask_right, S)

        # Left region has 2x more total counts so smoothed values must be 2x right
        np.testing.assert_allclose(
            df_left['counts'].values,
            2.0 * df_right['counts'].values,
            rtol=1e-10
        )

    def test_tof_shape_matches_full_image(self):
        """Each region's smoothed spectrum shape matches the full-image TOF shape"""
        # Asymmetric TOF profile
        stack = np.zeros((6, 8, 8))
        weights = np.array([1, 3, 5, 4, 2, 1], dtype=float)
        for t, w in enumerate(weights):
            stack[t] = w

        mask = np.zeros((8, 8), dtype=bool)
        mask[:4, :4] = True   # top-left quadrant

        S = self._full_spectrum(stack)
        df = calculate_smooth_spectrum(stack, mask, S)

        # Normalised shape should match S/S_total
        S_total = S.sum()
        expected_shape = S / S_total
        actual_shape = df['counts'].values / df['counts'].values.sum()
        np.testing.assert_allclose(actual_shape, expected_shape, rtol=1e-10)

    def test_uncertainty_is_nonnegative(self):
        """Propagated uncertainty must always be >= 0"""
        rng = np.random.default_rng(7)
        stack = rng.poisson(10, size=(10, 20, 20)).astype(float)
        mask = np.zeros((20, 20), dtype=bool)
        mask[:5, :5] = True
        S = self._full_spectrum(stack)
        df = calculate_smooth_spectrum(stack, mask, S)
        assert (df['err'].values >= 0).all()

    def test_uncertainty_smaller_than_raw_for_small_region(self):
        """Smooth uncertainty < raw Poisson err for low-count region (many TOF bins)"""
        rng = np.random.default_rng(99)
        # Large image, many TOF bins, sparse region
        n_tof = 100
        stack = rng.poisson(100, size=(n_tof, 50, 50)).astype(float)

        mask = np.zeros((50, 50), dtype=bool)
        mask[0, 0] = True   # single pixel

        S = self._full_spectrum(stack)
        df_smooth = calculate_smooth_spectrum(stack, mask, S)
        df_raw = calculate_spectrum(stack, mask)

        # Total smooth err (in quadrature) should be less than total raw err
        smooth_total_err = np.sqrt((df_smooth['err'].values ** 2).sum())
        raw_total_err = np.sqrt((df_raw['err'].values ** 2).sum())
        assert smooth_total_err < raw_total_err

    def test_empty_mask_returns_zeros(self):
        """Empty mask gives zero counts and zero uncertainty"""
        stack = np.ones((3, 5, 5))
        mask = np.zeros((5, 5), dtype=bool)
        S = self._full_spectrum(stack)
        df = calculate_smooth_spectrum(stack, mask, S)
        assert (df['counts'].values == 0).all()
        assert (df['err'].values == 0).all()

    def test_zero_stack_returns_zeros(self):
        """All-zero stack gives zero smoothed spectrum"""
        stack = np.zeros((4, 6, 6))
        mask = np.ones((6, 6), dtype=bool)
        S = self._full_spectrum(stack)
        df = calculate_smooth_spectrum(stack, mask, S)
        assert (df['counts'].values == 0).all()
        assert (df['err'].values == 0).all()

    def test_full_image_uncertainty_is_poisson(self):
        """For the full-image mask, err² = S(t) (standard Poisson variance).

        Derivation: with N = S_total and I_raw(t) = S(t), the formula gives
            σ²/f² = 1/S_total + 1/S(t) − 3/S_total + 2/S_total = 1/S(t)
        so σ² = S(t)² / S(t) = S(t).

        Note: this is *not* the binomial S(t)·(1 − S(t)/S_total) because S_total
        is itself a Poisson random variable, not a fixed conditioning quantity.
        """
        stack = np.zeros((5, 10, 10))
        weights = np.array([10.0, 50.0, 100.0, 60.0, 20.0])
        for t, w in enumerate(weights):
            stack[t] = w

        mask = np.ones((10, 10), dtype=bool)
        S = self._full_spectrum(stack)
        df = calculate_smooth_spectrum(stack, mask, S)

        # σ² = S(t) for the full-image mask
        np.testing.assert_allclose(df['err'].values ** 2, S, rtol=1e-10)

    def test_consistency_with_extract_roi_spectra(self):
        """extract_roi_spectra with smooth=True uses calculate_smooth_spectrum"""
        import tifffile, tempfile
        from save_roi import extract_roi_spectra

        stack = np.random.default_rng(3).poisson(20, (6, 8, 8)).astype(np.uint16)
        with tempfile.NamedTemporaryFile(suffix='.tiff', delete=False) as f:
            tiff_path = f.name
        tifffile.imwrite(tiff_path, stack)

        results_smooth = extract_roi_spectra(tiff_path, roi_path=None, save_csv=False, smooth=True)
        results_raw = extract_roi_spectra(tiff_path, roi_path=None, save_csv=False, smooth=False)

        import os; os.unlink(tiff_path)

        df_s = results_smooth['full_image']
        df_r = results_raw['full_image']

        # For full image, smoothed counts == raw counts
        np.testing.assert_allclose(df_s['counts'].values, df_r['counts'].values, rtol=1e-10)
        # But errors differ (smooth uses binomial correction; raw uses sqrt)
        # They should be close but not identical for non-trivial spectra
        assert df_s['err'].values.sum() > 0


class TestExtractRoiSpectra:
    """Tests for extract_roi_spectra function"""

    def test_extract_with_roi_file(self, test_tiff_path, test_roi_path, temp_output_dir):
        """Test extracting spectra with ROI file"""
        results = extract_roi_spectra(
            tiff_path=test_tiff_path,
            roi_path=test_roi_path,
            output_dir=temp_output_dir,
            save_csv=True
        )

        assert isinstance(results, dict)
        assert len(results) > 0

        # Check each result
        for roi_name, df in results.items():
            assert isinstance(df, pd.DataFrame)
            assert list(df.columns) == ['stack', 'counts', 'err']
            assert len(df) > 0

            # Check CSV was saved
            csv_path = temp_output_dir / f"{roi_name}.csv"
            assert csv_path.exists()

    def test_extract_without_roi_file(self, test_tiff_path, temp_output_dir):
        """Test extracting spectra without ROI file (full image)"""
        results = extract_roi_spectra(
            tiff_path=test_tiff_path,
            roi_path=None,
            output_dir=temp_output_dir,
            save_csv=True
        )

        assert 'full_image' in results
        df = results['full_image']
        assert isinstance(df, pd.DataFrame)
        assert list(df.columns) == ['stack', 'counts', 'err']

        # Check CSV was saved
        csv_path = temp_output_dir / "full_image.csv"
        assert csv_path.exists()

    def test_extract_no_save(self, test_tiff_path, test_roi_path, temp_output_dir):
        """Test extracting spectra without saving CSV"""
        results = extract_roi_spectra(
            tiff_path=test_tiff_path,
            roi_path=test_roi_path,
            output_dir=temp_output_dir,
            save_csv=False
        )

        assert isinstance(results, dict)
        assert len(results) > 0

        # Check that no CSV files were created
        csv_files = list(temp_output_dir.glob("*.csv"))
        assert len(csv_files) == 0


class TestExtractFullImageSpectrum:
    """Tests for extract_full_image_spectrum function"""

    def test_extract_full_image(self, test_tiff_path, temp_output_dir):
        """Test extracting full image spectrum"""
        df = extract_full_image_spectrum(
            tiff_path=test_tiff_path,
            output_dir=temp_output_dir,
            save_csv=True
        )

        assert isinstance(df, pd.DataFrame)
        assert list(df.columns) == ['stack', 'counts', 'err']
        assert len(df) > 0

        # Check CSV was saved
        csv_path = temp_output_dir / "full_image.csv"
        assert csv_path.exists()


class TestExtractPixelSpectra:
    """Tests for extract_pixel_spectra function"""

    def test_extract_pixel_spectra_stride1(self, test_tiff_path, temp_output_dir):
        """Test extracting pixel spectra with stride=1"""
        # Load stack to get dimensions
        stack = load_tiff_stack(test_tiff_path)
        height, width = stack.shape[1:]

        results = extract_pixel_spectra(
            tiff_path=test_tiff_path,
            output_dir=temp_output_dir,
            save_csv=False,  # Don't save to avoid too many files
            stride=1
        )

        assert isinstance(results, dict)
        # Should have height * width pixels
        assert len(results) == height * width

        # Check structure of one result
        first_key = list(results.keys())[0]
        assert first_key.startswith('pixel_')
        df = results[first_key]
        assert isinstance(df, pd.DataFrame)
        assert list(df.columns) == ['stack', 'counts', 'err']

    def test_extract_pixel_spectra_stride4(self, test_tiff_path, temp_output_dir):
        """Test extracting pixel spectra with stride=4"""
        results = extract_pixel_spectra(
            tiff_path=test_tiff_path,
            output_dir=temp_output_dir,
            save_csv=False,
            stride=4
        )

        assert isinstance(results, dict)
        # Should have fewer pixels with stride=4
        stack = load_tiff_stack(test_tiff_path)
        height, width = stack.shape[1:]
        expected_count = ((height + 3) // 4) * ((width + 3) // 4)
        assert len(results) <= expected_count


class TestExtractGridSpectra:
    """Tests for extract_grid_spectra function"""

    def test_extract_grid_spectra_4x4(self, test_tiff_path, temp_output_dir):
        """Test extracting grid spectra with 4x4 blocks"""
        results = extract_grid_spectra(
            tiff_path=test_tiff_path,
            grid_size=4,
            output_dir=temp_output_dir,
            save_csv=False
        )

        assert isinstance(results, dict)
        assert len(results) > 0

        # Check structure of one result
        first_key = list(results.keys())[0]
        assert 'grid_4x4' in first_key
        df = results[first_key]
        assert isinstance(df, pd.DataFrame)
        assert list(df.columns) == ['stack', 'counts', 'err']

    def test_extract_grid_spectra_8x8(self, test_tiff_path, temp_output_dir):
        """Test extracting grid spectra with 8x8 blocks"""
        results = extract_grid_spectra(
            tiff_path=test_tiff_path,
            grid_size=8,
            output_dir=temp_output_dir,
            save_csv=False
        )

        assert isinstance(results, dict)
        assert len(results) > 0

        # 8x8 blocks should create fewer grid cells than 4x4
        results_4x4 = extract_grid_spectra(
            tiff_path=test_tiff_path,
            grid_size=4,
            output_dir=temp_output_dir,
            save_csv=False
        )
        assert len(results) < len(results_4x4)


class TestDataConsistency:
    """Tests to ensure data consistency across different methods"""

    def test_full_image_equals_single_roi(self, test_tiff_path):
        """Test that full image analysis gives same result as full-image ROI"""
        # Get full image spectrum
        full_results = extract_roi_spectra(
            tiff_path=test_tiff_path,
            roi_path=None,
            save_csv=False
        )
        full_df = full_results['full_image']

        # Sum all pixel spectra should equal full image
        pixel_results = extract_pixel_spectra(
            tiff_path=test_tiff_path,
            save_csv=False,
            stride=1
        )

        # Sum counts across all pixels for each slice
        summed_counts = {}
        for pixel_name, df in pixel_results.items():
            for _, row in df.iterrows():
                slice_num = row['stack']
                if slice_num not in summed_counts:
                    summed_counts[slice_num] = 0
                summed_counts[slice_num] += row['counts']

        # Compare with full image (allow small numerical differences)
        for _, row in full_df.iterrows():
            slice_num = row['stack']
            assert np.isclose(summed_counts[slice_num], row['counts'], rtol=1e-6)


class TestEdgeCases:
    """Tests for edge cases and error handling"""

    def test_nonexistent_tiff(self, temp_output_dir):
        """Test with non-existent TIFF file"""
        with pytest.raises(FileNotFoundError):
            extract_roi_spectra(
                tiff_path="nonexistent.tiff",
                roi_path=None,
                output_dir=temp_output_dir
            )

    def test_nonexistent_roi(self, test_tiff_path, temp_output_dir):
        """Test with non-existent ROI file"""
        with pytest.raises(FileNotFoundError):
            extract_roi_spectra(
                tiff_path=test_tiff_path,
                roi_path="nonexistent.roi",
                output_dir=temp_output_dir
            )

    def test_grid_size_larger_than_image(self, test_tiff_path, temp_output_dir):
        """Test grid size larger than image dimensions"""
        results = extract_grid_spectra(
            tiff_path=test_tiff_path,
            grid_size=10000,  # Very large grid
            output_dir=temp_output_dir,
            save_csv=False
        )

        # Should create at least one grid cell
        assert len(results) >= 1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
