"""
Tests for multi-folder ROI analysis functionality
"""

import pytest
import numpy as np
import pandas as pd
from pathlib import Path
import tempfile
import shutil

from save_roi import (
    discover_directories_with_wildcard,
    sum_roi_spectra_from_folders,
)
from save_roi.core import (
    load_tiff_stack,
    calculate_spectrum,
)


@pytest.fixture
def temp_multi_folder_structure():
    """Create temporary multi-folder structure for testing"""
    temp_dir = tempfile.mkdtemp()
    base_path = Path(temp_dir)

    # Create multiple run directories
    run_dirs = []
    for i in range(1, 4):
        run_dir = base_path / f"run23_{i}"
        run_dir.mkdir(parents=True)
        run_dirs.append(run_dir)

    yield base_path, run_dirs

    # Cleanup
    shutil.rmtree(temp_dir)


@pytest.fixture
def temp_roi_spectra_dirs():
    """Create temporary ROI spectra directories with sample CSV files"""
    temp_dir = tempfile.mkdtemp()
    base_path = Path(temp_dir)

    # Create ROI spectra directories
    roi_dirs = []
    for i in range(1, 4):
        roi_dir = base_path / f"run{i}_ROI_Spectra"
        roi_dir.mkdir(parents=True)
        roi_dirs.append(roi_dir)

        # Create sample CSV files with different counts
        for roi_name in ['ROI_1', 'ROI_2']:
            data = {
                'stack': [1, 2, 3, 4, 5],
                'counts': [100 * i, 200 * i, 300 * i, 400 * i, 500 * i],
                'err': [10 * i, 14.14 * i, 17.32 * i, 20 * i, 22.36 * i]
            }
            df = pd.DataFrame(data)
            csv_path = roi_dir / f"{roi_name}.csv"
            df.to_csv(csv_path, index=False)

    yield base_path, roi_dirs

    # Cleanup
    shutil.rmtree(temp_dir)


class TestDiscoverDirectoriesWithWildcard:
    """Tests for discover_directories_with_wildcard function"""

    def test_discover_with_wildcard(self, temp_multi_folder_structure):
        """Test discovering directories with wildcard pattern"""
        base_path, run_dirs = temp_multi_folder_structure

        # Test wildcard pattern
        pattern = base_path / "run23_*"
        discovered = discover_directories_with_wildcard(pattern)

        assert len(discovered) == 3
        assert all(d.is_dir() for d in discovered)
        assert all(d.name.startswith('run23_') for d in discovered)

    def test_discover_with_question_mark(self, temp_multi_folder_structure):
        """Test discovering directories with ? wildcard"""
        base_path, run_dirs = temp_multi_folder_structure

        # Test ? pattern
        pattern = base_path / "run23_?"
        discovered = discover_directories_with_wildcard(pattern)

        assert len(discovered) == 3

    def test_discover_no_match(self, temp_multi_folder_structure):
        """Test discovering with pattern that matches nothing"""
        base_path, run_dirs = temp_multi_folder_structure

        # Test non-matching pattern
        pattern = base_path / "nonexistent_*"
        discovered = discover_directories_with_wildcard(pattern)

        assert len(discovered) == 0

    def test_discover_filters_files(self, temp_multi_folder_structure):
        """Test that only directories are returned, not files"""
        base_path, run_dirs = temp_multi_folder_structure

        # Create a file that matches the pattern
        file_path = base_path / "run23_file.txt"
        file_path.write_text("test")

        pattern = base_path / "run23_*"
        discovered = discover_directories_with_wildcard(pattern)

        # Should only find directories, not the file
        assert len(discovered) == 3
        assert all(d.is_dir() for d in discovered)


class TestSumRoiSpectraFromFolders:
    """Tests for sum_roi_spectra_from_folders function"""

    def test_sum_basic(self, temp_roi_spectra_dirs):
        """Test basic summing of ROI spectra from multiple folders"""
        base_path, roi_dirs = temp_roi_spectra_dirs

        # Create output directory
        output_dir = base_path / "SUM"

        # Sum the spectra
        results = sum_roi_spectra_from_folders(
            roi_spectra_dirs=roi_dirs,
            output_dir=output_dir,
            save_csv=True
        )

        # Check results
        assert 'ROI_1' in results
        assert 'ROI_2' in results

        # Check ROI_1 data
        df_roi1 = results['ROI_1']
        assert len(df_roi1) == 5  # 5 slices

        # Check that counts are summed correctly
        # run1: 100, run2: 200, run3: 300 -> sum: 600
        assert df_roi1.loc[0, 'counts'] == 600
        assert df_roi1.loc[1, 'counts'] == 1200
        assert df_roi1.loc[2, 'counts'] == 1800
        assert df_roi1.loc[3, 'counts'] == 2400
        assert df_roi1.loc[4, 'counts'] == 3000

        # Check that errors are recalculated as sqrt(sum)
        assert np.isclose(df_roi1.loc[0, 'err'], np.sqrt(600))
        assert np.isclose(df_roi1.loc[1, 'err'], np.sqrt(1200))

    def test_sum_saves_csv(self, temp_roi_spectra_dirs):
        """Test that CSV files are saved correctly"""
        base_path, roi_dirs = temp_roi_spectra_dirs

        output_dir = base_path / "SUM"

        sum_roi_spectra_from_folders(
            roi_spectra_dirs=roi_dirs,
            output_dir=output_dir,
            save_csv=True
        )

        # Check that CSV files were created
        assert (output_dir / "ROI_1.csv").exists()
        assert (output_dir / "ROI_2.csv").exists()

        # Check that CSV can be read and has correct data
        df = pd.read_csv(output_dir / "ROI_1.csv")
        assert len(df) == 5
        assert list(df.columns) == ['stack', 'counts', 'err']

    def test_sum_no_save(self, temp_roi_spectra_dirs):
        """Test summing without saving CSV files"""
        base_path, roi_dirs = temp_roi_spectra_dirs

        output_dir = base_path / "SUM"

        results = sum_roi_spectra_from_folders(
            roi_spectra_dirs=roi_dirs,
            output_dir=output_dir,
            save_csv=False
        )

        # Check results exist
        assert 'ROI_1' in results

        # Check that no CSV files were created
        assert not output_dir.exists()

    def test_sum_missing_csv_in_some_dirs(self, temp_roi_spectra_dirs):
        """Test summing when some directories are missing certain CSV files"""
        base_path, roi_dirs = temp_roi_spectra_dirs

        # Remove ROI_2 from the first directory
        (roi_dirs[0] / "ROI_2.csv").unlink()

        output_dir = base_path / "SUM"

        results = sum_roi_spectra_from_folders(
            roi_spectra_dirs=roi_dirs,
            output_dir=output_dir,
            save_csv=True
        )

        # ROI_1 should be summed from all 3 directories
        df_roi1 = results['ROI_1']
        assert df_roi1.loc[0, 'counts'] == 600  # 100 + 200 + 300

        # ROI_2 should be summed from only 2 directories
        df_roi2 = results['ROI_2']
        assert df_roi2.loc[0, 'counts'] == 500  # 200 + 300 (missing from first dir)

    def test_sum_with_inconsistent_slices(self, temp_roi_spectra_dirs):
        """Test summing when directories have different numbers of slices"""
        base_path, roi_dirs = temp_roi_spectra_dirs

        # Modify one directory to have fewer slices
        data = {
            'stack': [1, 2, 3],  # Only 3 slices instead of 5
            'counts': [100, 200, 300],
            'err': [10, 14.14, 17.32]
        }
        df = pd.DataFrame(data)
        (roi_dirs[0] / "ROI_1.csv").write_text("")  # Clear first
        df.to_csv(roi_dirs[0] / "ROI_1.csv", index=False)

        output_dir = base_path / "SUM"

        # Should handle gracefully and use minimum number of slices
        with pytest.warns(UserWarning):
            results = sum_roi_spectra_from_folders(
                roi_spectra_dirs=roi_dirs,
                output_dir=output_dir,
                save_csv=True
            )

        # Should only have 3 slices (the minimum)
        df_roi1 = results['ROI_1']
        assert len(df_roi1) == 3

    def test_sum_nonexistent_directory(self):
        """Test that error is raised for non-existent directories"""
        with pytest.raises(FileNotFoundError):
            sum_roi_spectra_from_folders(
                roi_spectra_dirs=[Path("/nonexistent/path")],
                output_dir=Path("/tmp/output")
            )

    def test_sum_empty_directories(self, temp_roi_spectra_dirs):
        """Test that error is raised when no CSV files are found"""
        base_path, roi_dirs = temp_roi_spectra_dirs

        # Create empty directory
        empty_dir = base_path / "empty"
        empty_dir.mkdir()

        output_dir = base_path / "SUM"

        with pytest.raises(ValueError, match="No CSV files found"):
            sum_roi_spectra_from_folders(
                roi_spectra_dirs=[empty_dir],
                output_dir=output_dir
            )


class TestDataConsistency:
    """Tests to ensure data consistency in multi-folder workflows"""

    def test_sum_preserves_poisson_statistics(self, temp_roi_spectra_dirs):
        """Test that summing preserves Poisson statistics correctly"""
        base_path, roi_dirs = temp_roi_spectra_dirs

        output_dir = base_path / "SUM"

        results = sum_roi_spectra_from_folders(
            roi_spectra_dirs=roi_dirs,
            output_dir=output_dir,
            save_csv=True
        )

        # For Poisson statistics: error = sqrt(counts)
        for roi_name, df in results.items():
            for _, row in df.iterrows():
                expected_err = np.sqrt(row['counts']) if row['counts'] > 0 else 0
                assert np.isclose(row['err'], expected_err), \
                    f"Error mismatch for {roi_name}, slice {row['stack']}"

    def test_sum_is_associative(self, temp_roi_spectra_dirs):
        """Test that summing is associative (sum(A,B,C) == sum(sum(A,B), C))"""
        base_path, roi_dirs = temp_roi_spectra_dirs

        # Sum all three directories at once
        output_dir1 = base_path / "SUM1"
        results1 = sum_roi_spectra_from_folders(
            roi_spectra_dirs=roi_dirs,
            output_dir=output_dir1,
            save_csv=False
        )

        # Sum first two, then sum result with third
        output_dir2 = base_path / "SUM_AB"
        results_ab = sum_roi_spectra_from_folders(
            roi_spectra_dirs=roi_dirs[:2],
            output_dir=output_dir2,
            save_csv=True
        )

        output_dir3 = base_path / "SUM2"
        results2 = sum_roi_spectra_from_folders(
            roi_spectra_dirs=[output_dir2, roi_dirs[2]],
            output_dir=output_dir3,
            save_csv=False
        )

        # Results should be identical
        for roi_name in results1.keys():
            df1 = results1[roi_name]
            df2 = results2[roi_name]

            assert len(df1) == len(df2)
            for i in range(len(df1)):
                assert np.isclose(df1.loc[i, 'counts'], df2.loc[i, 'counts'])
                assert np.isclose(df1.loc[i, 'err'], df2.loc[i, 'err'])


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
