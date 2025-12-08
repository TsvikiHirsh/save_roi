"""
Tests for file discovery and merging functionality
"""

import pytest
import tempfile
import shutil
from pathlib import Path
import numpy as np
import tifffile
from roifile import ImagejRoi, roiwrite

from spectral_roi import (
    discover_tiff_files,
    discover_roi_files,
    merge_roi_files,
)


@pytest.fixture
def temp_dir():
    """Create a temporary directory for tests"""
    temp_path = Path(tempfile.mkdtemp())
    yield temp_path
    shutil.rmtree(temp_path)


@pytest.fixture
def sample_tiff_stack():
    """Create a sample TIFF stack for testing"""
    # Create a simple 3D stack: 10 slices, 50x50 pixels
    stack = np.random.randint(0, 1000, (10, 50, 50), dtype=np.uint16)
    return stack


@pytest.fixture
def sample_roi():
    """Get a sample ROI from test data"""
    # Use the existing ROI2.zip from notebooks for testing
    test_roi_path = Path(__file__).parent.parent / "notebooks" / "ROI2.zip"
    if test_roi_path.exists():
        from spectral_roi import load_imagej_rois
        rois = load_imagej_rois(test_roi_path)
        if rois:
            return rois[0]['roi_object']
    # Fallback: skip tests that need ROI
    pytest.skip("Test ROI file not available")


class TestDiscoverTiffFiles:
    """Tests for TIFF file discovery"""

    def test_discover_single_tiff(self, temp_dir, sample_tiff_stack):
        """Test discovering a single TIFF file"""
        tiff_path = temp_dir / "test.tiff"
        tifffile.imwrite(tiff_path, sample_tiff_stack)

        discovered = discover_tiff_files(temp_dir)
        assert len(discovered) == 1
        assert discovered[0].name == "test.tiff"

    def test_discover_multiple_tiffs(self, temp_dir, sample_tiff_stack):
        """Test discovering multiple TIFF files"""
        for i in range(3):
            tiff_path = temp_dir / f"test_{i}.tiff"
            tifffile.imwrite(tiff_path, sample_tiff_stack)

        discovered = discover_tiff_files(temp_dir)
        assert len(discovered) == 3
        assert all(f.suffix == ".tiff" for f in discovered)

    def test_discover_mixed_extensions(self, temp_dir, sample_tiff_stack):
        """Test discovering TIFF files with different extensions"""
        extensions = ['.tif', '.tiff', '.TIF', '.TIFF']
        for i, ext in enumerate(extensions):
            tiff_path = temp_dir / f"test_{i}{ext}"
            tifffile.imwrite(tiff_path, sample_tiff_stack)

        discovered = discover_tiff_files(temp_dir)
        assert len(discovered) == 4

    def test_discover_no_tiffs(self, temp_dir):
        """Test discovering when no TIFF files exist"""
        # Create a non-TIFF file
        (temp_dir / "test.txt").write_text("not a tiff")

        discovered = discover_tiff_files(temp_dir)
        assert len(discovered) == 0

    def test_discover_nonexistent_directory(self):
        """Test discovering in a non-existent directory"""
        with pytest.raises(FileNotFoundError):
            discover_tiff_files("/nonexistent/directory")

    def test_discover_file_not_directory(self, temp_dir):
        """Test discovering when path is a file, not directory"""
        file_path = temp_dir / "test.txt"
        file_path.write_text("test")

        with pytest.raises(ValueError):
            discover_tiff_files(file_path)

    def test_discover_sorted_output(self, temp_dir, sample_tiff_stack):
        """Test that discovered files are sorted"""
        names = ["z_last.tiff", "a_first.tiff", "m_middle.tiff"]
        for name in names:
            tiff_path = temp_dir / name
            tifffile.imwrite(tiff_path, sample_tiff_stack)

        discovered = discover_tiff_files(temp_dir)
        assert len(discovered) == 3
        assert discovered[0].name == "a_first.tiff"
        assert discovered[1].name == "m_middle.tiff"
        assert discovered[2].name == "z_last.tiff"


class TestDiscoverRoiFiles:
    """Tests for ROI file discovery"""

    def test_discover_single_roi(self, temp_dir, sample_roi):
        """Test discovering a single ROI file"""
        roi_path = temp_dir / "test.roi"
        roiwrite(roi_path, sample_roi)

        discovered = discover_roi_files(temp_dir)
        assert len(discovered) == 1
        assert discovered[0].name == "test.roi"

    def test_discover_single_zip(self, temp_dir, sample_roi):
        """Test discovering a single ZIP file"""
        import zipfile

        zip_path = temp_dir / "test.zip"
        roi_file = temp_dir / "temp.roi"
        roiwrite(roi_file, sample_roi)

        with zipfile.ZipFile(zip_path, 'w') as zf:
            zf.write(roi_file, arcname="test.roi")

        roi_file.unlink()  # Remove temp file

        discovered = discover_roi_files(temp_dir)
        assert len(discovered) == 1
        assert discovered[0].name == "test.zip"

    def test_discover_mixed_roi_files(self, temp_dir, sample_roi):
        """Test discovering both .roi and .zip files"""
        import zipfile

        # Create .roi file
        roi_path = temp_dir / "test.roi"
        roiwrite(roi_path, sample_roi)

        # Create .zip file
        zip_path = temp_dir / "test2.zip"
        roi_file = temp_dir / "temp.roi"
        roiwrite(roi_file, sample_roi)
        with zipfile.ZipFile(zip_path, 'w') as zf:
            zf.write(roi_file, arcname="test2.roi")
        roi_file.unlink()

        discovered = discover_roi_files(temp_dir)
        assert len(discovered) == 2

    def test_discover_no_rois(self, temp_dir):
        """Test discovering when no ROI files exist"""
        (temp_dir / "test.txt").write_text("not an roi")

        discovered = discover_roi_files(temp_dir)
        assert len(discovered) == 0

    def test_discover_case_insensitive(self, temp_dir, sample_roi):
        """Test case-insensitive extension matching"""
        extensions = ['.roi', '.ROI', '.zip', '.ZIP']
        for i, ext in enumerate(extensions):
            if ext.lower() == '.roi':
                roi_path = temp_dir / f"test_{i}{ext}"
                roiwrite(roi_path, sample_roi)
            else:
                import zipfile
                zip_path = temp_dir / f"test_{i}{ext}"
                roi_file = temp_dir / "temp.roi"
                roiwrite(roi_file, sample_roi)
                with zipfile.ZipFile(zip_path, 'w') as zf:
                    zf.write(roi_file, arcname=f"test_{i}.roi")
                roi_file.unlink()

        discovered = discover_roi_files(temp_dir)
        assert len(discovered) == 4


class TestMergeRoiFiles:
    """Tests for merging multiple ROI files"""

    def test_merge_single_file(self, temp_dir):
        """Test merging a single ROI file"""
        # Create a simple ROI
        roi = ImagejRoi.frompoints([[10, 10], [30, 10], [30, 30], [10, 30]])
        roi.name = "roi1"
        roi_path = temp_dir / "test.roi"
        roiwrite(roi_path, roi)

        merged, warnings = merge_roi_files([roi_path])
        assert len(merged) == 1
        assert merged[0]['name'] == "roi1"
        assert len(warnings) == 0

    def test_merge_multiple_files_no_conflict(self, temp_dir):
        """Test merging multiple files with unique ROI names"""
        roi_files = []
        for i in range(3):
            roi = ImagejRoi.frompoints([[10, 10], [30, 10], [30, 30], [10, 30]])
            roi.name = f"roi_{i}"
            roi_path = temp_dir / f"test_{i}.roi"
            roiwrite(roi_path, roi)
            roi_files.append(roi_path)

        merged, warnings = merge_roi_files(roi_files)
        assert len(merged) == 3
        assert len(warnings) == 0
        roi_names = [r['name'] for r in merged]
        assert "roi_0" in roi_names
        assert "roi_1" in roi_names
        assert "roi_2" in roi_names

    def test_merge_with_name_conflict(self, temp_dir):
        """Test merging files with conflicting ROI names"""
        # Create two ROI files with the same ROI name
        roi1 = ImagejRoi.frompoints([[10, 10], [20, 10], [20, 20], [10, 20]])
        roi1.name = "duplicate"
        roi_path1 = temp_dir / "test1.roi"
        roiwrite(roi_path1, roi1)

        roi2 = ImagejRoi.frompoints([[30, 30], [40, 30], [40, 40], [30, 40]])
        roi2.name = "duplicate"
        roi_path2 = temp_dir / "test2.roi"
        roiwrite(roi_path2, roi2)

        with pytest.warns(UserWarning, match="ROI name 'duplicate' appears in both"):
            merged, warnings = merge_roi_files([roi_path1, roi_path2])

        # Should only have one ROI (from the second file)
        assert len(merged) == 1
        assert merged[0]['name'] == "duplicate"
        assert len(warnings) == 1
        assert "duplicate" in warnings[0]

    def test_merge_zip_with_multiple_rois(self, temp_dir):
        """Test merging a ZIP file containing multiple ROIs"""
        import zipfile

        # Create multiple ROIs
        roi_files = []
        for i in range(3):
            roi = ImagejRoi.frompoints([[10+i*10, 10], [20+i*10, 10], [20+i*10, 20], [10+i*10, 20]])
            roi.name = f"roi_{i}"
            roi_path = temp_dir / f"roi_{i}.roi"
            roiwrite(roi_path, roi)
            roi_files.append(roi_path)

        # Create ZIP file
        zip_path = temp_dir / "rois.zip"
        with zipfile.ZipFile(zip_path, 'w') as zf:
            for roi_file in roi_files:
                zf.write(roi_file, arcname=roi_file.name)

        # Clean up individual ROI files
        for roi_file in roi_files:
            roi_file.unlink()

        merged, warnings = merge_roi_files([zip_path])
        assert len(merged) == 3
        assert len(warnings) == 0

    def test_merge_empty_list(self):
        """Test merging an empty list of ROI files"""
        merged, warnings = merge_roi_files([])
        assert len(merged) == 0
        assert len(warnings) == 0

    def test_merge_invalid_file(self, temp_dir):
        """Test merging with an invalid ROI file"""
        invalid_file = temp_dir / "invalid.roi"
        invalid_file.write_text("not a valid roi file")

        with pytest.warns(UserWarning, match="Error loading ROI file"):
            merged, warnings = merge_roi_files([invalid_file])

        assert len(merged) == 0
        assert len(warnings) == 1
        assert "Error loading" in warnings[0]

    def test_merge_preserves_roi_objects(self, temp_dir):
        """Test that merged ROIs preserve their roi_object"""
        roi = ImagejRoi.frompoints([[10, 10], [30, 10], [30, 30], [10, 30]])
        roi.name = "test_roi"
        roi_path = temp_dir / "test.roi"
        roiwrite(roi_path, roi)

        merged, warnings = merge_roi_files([roi_path])
        assert len(merged) == 1
        assert 'roi_object' in merged[0]
        assert merged[0]['roi_object'] is not None


class TestQuickWorkflowStructure:
    """Tests for the quick workflow directory structure"""

    def test_quick_workflow_structure(self, temp_dir, sample_tiff_stack, sample_roi):
        """Test the expected directory structure for quick workflow"""
        # Create the expected structure
        data_dir = temp_dir / "data"
        final_dir = data_dir / "final"
        final_dir.mkdir(parents=True)

        # Create TIFF in data/final/
        tiff_path = final_dir / "image.tiff"
        tifffile.imwrite(tiff_path, sample_tiff_stack)

        # Create ROI in root
        roi_path = temp_dir / "ROI.zip"
        import zipfile
        roi_file = temp_dir / "temp.roi"
        roiwrite(roi_file, sample_roi)
        with zipfile.ZipFile(roi_path, 'w') as zf:
            zf.write(roi_file, arcname="test.roi")
        roi_file.unlink()

        # Verify structure
        assert final_dir.exists()
        assert (final_dir / "image.tiff").exists()
        assert roi_path.exists()

        # Test discovery
        tiffs = discover_tiff_files(final_dir)
        assert len(tiffs) == 1

        rois_in_root = discover_roi_files(temp_dir)
        assert len(rois_in_root) == 1

    def test_multiple_tiffs_in_final(self, temp_dir, sample_tiff_stack):
        """Test discovering multiple TIFFs in final directory"""
        final_dir = temp_dir / "data" / "final"
        final_dir.mkdir(parents=True)

        # Create multiple TIFFs
        for i in range(3):
            tiff_path = final_dir / f"image_{i}.tiff"
            tifffile.imwrite(tiff_path, sample_tiff_stack)

        tiffs = discover_tiff_files(final_dir)
        assert len(tiffs) == 3

    def test_multiple_roi_files_in_root(self, temp_dir, sample_roi):
        """Test discovering multiple ROI files in root directory"""
        import zipfile

        # Create multiple ROI files
        for i in range(2):
            roi = ImagejRoi.frompoints([[10+i*10, 10], [20+i*10, 10], [20+i*10, 20], [10+i*10, 20]])
            roi.name = f"roi_{i}"

            if i == 0:
                roi_path = temp_dir / f"ROI_{i}.roi"
                roiwrite(roi_path, roi)
            else:
                zip_path = temp_dir / f"ROI_{i}.zip"
                roi_file = temp_dir / "temp.roi"
                roiwrite(roi_file, roi)
                with zipfile.ZipFile(zip_path, 'w') as zf:
                    zf.write(roi_file, arcname=f"roi_{i}.roi")
                roi_file.unlink()

        rois = discover_roi_files(temp_dir)
        assert len(rois) == 2

        # Test merging
        merged, warnings = merge_roi_files(rois)
        assert len(merged) == 2
