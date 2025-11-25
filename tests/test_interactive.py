"""
Tests for interactive ROI selection functionality
"""

import pytest
import numpy as np
from pathlib import Path
import tempfile
import zipfile

# Try to import interactive module
try:
    from spectral_roi.interactive import InteractiveROISelector
    from roifile import ImagejRoi
    INTERACTIVE_AVAILABLE = True
except ImportError:
    INTERACTIVE_AVAILABLE = False


# Skip all tests if interactive dependencies not available
pytestmark = pytest.mark.skipif(
    not INTERACTIVE_AVAILABLE,
    reason="Interactive dependencies not installed"
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
        pytest.skip("Test TIFF file not found. Run scripts/setup_test_data.py first.")
    return tiff_path


@pytest.fixture
def temp_output_dir():
    """Create a temporary output directory"""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


class TestInteractiveROISelector:
    """Test InteractiveROISelector class"""

    def test_initialization(self, test_tiff_path):
        """Test selector initialization"""
        selector = InteractiveROISelector(test_tiff_path)

        assert selector.tiff_path == test_tiff_path
        assert selector.stack is not None
        assert selector.summed_image is not None
        assert len(selector.rois) == 0
        assert selector.roi_counter == 1

    def test_initialization_with_stack_range(self, test_tiff_path):
        """Test selector initialization with stack range"""
        selector = InteractiveROISelector(test_tiff_path, stack_range=(0, 5))

        assert selector.stack.shape[0] == 5
        assert selector.summed_image.shape[0] > 0

    def test_sum_stack_range(self, test_tiff_path):
        """Test summing specific stack range"""
        selector = InteractiveROISelector(test_tiff_path)

        summed = selector.sum_stack_range(0, 5)
        assert summed.ndim == 2
        assert summed.shape == selector.summed_image.shape

    def test_create_figure(self, test_tiff_path):
        """Test figure creation"""
        selector = InteractiveROISelector(test_tiff_path)
        fig = selector.create_figure()

        assert fig is not None
        assert len(fig.data) > 0
        assert fig.layout.dragmode == 'drawrect'

    def test_add_rectangle_roi(self, test_tiff_path):
        """Test adding a rectangle ROI"""
        selector = InteractiveROISelector(test_tiff_path)

        # Create a mock rectangle shape
        shape_data = {
            'type': 'rect',
            'x0': 10,
            'y0': 10,
            'x1': 50,
            'y1': 50
        }

        selector.add_roi_from_shape(shape_data, name="test_rect")

        assert len(selector.rois) == 1
        assert selector.rois[0]['name'] == "test_rect"
        assert selector.rois[0]['shape'] == 'rect'
        assert selector.rois[0]['coords'].shape == (4, 2)

    def test_add_circle_roi(self, test_tiff_path):
        """Test adding a circle ROI"""
        selector = InteractiveROISelector(test_tiff_path)

        # Create a mock circle shape
        shape_data = {
            'type': 'circle',
            'x0': 10,
            'y0': 10,
            'x1': 50,
            'y1': 50
        }

        selector.add_roi_from_shape(shape_data, name="test_circle")

        assert len(selector.rois) == 1
        assert selector.rois[0]['name'] == "test_circle"
        assert selector.rois[0]['shape'] == 'circle'
        assert selector.rois[0]['coords'].shape[0] == 100  # 100 points in ellipse

    def test_add_polygon_roi(self, test_tiff_path):
        """Test adding a polygon ROI"""
        selector = InteractiveROISelector(test_tiff_path)

        # Create a mock path shape
        shape_data = {
            'type': 'path',
            'path': 'M 10,10 L 50,10 L 50,50 L 10,50'
        }

        selector.add_roi_from_shape(shape_data, name="test_polygon")

        assert len(selector.rois) == 1
        assert selector.rois[0]['name'] == "test_polygon"
        assert selector.rois[0]['shape'] == 'path'
        assert selector.rois[0]['coords'].shape[0] > 0

    def test_rename_roi(self, test_tiff_path):
        """Test renaming an ROI"""
        selector = InteractiveROISelector(test_tiff_path)

        # Add an ROI
        shape_data = {
            'type': 'rect',
            'x0': 10,
            'y0': 10,
            'x1': 50,
            'y1': 50
        }
        selector.add_roi_from_shape(shape_data, name="old_name")

        # Rename it
        selector.rename_roi("old_name", "new_name")

        assert len(selector.rois) == 1
        assert selector.rois[0]['name'] == "new_name"

    def test_delete_roi(self, test_tiff_path):
        """Test deleting an ROI"""
        selector = InteractiveROISelector(test_tiff_path)

        # Add two ROIs
        shape_data = {
            'type': 'rect',
            'x0': 10,
            'y0': 10,
            'x1': 50,
            'y1': 50
        }
        selector.add_roi_from_shape(shape_data, name="roi1")
        selector.add_roi_from_shape(shape_data, name="roi2")

        # Delete one
        selector.delete_roi("roi1")

        assert len(selector.rois) == 1
        assert selector.rois[0]['name'] == "roi2"

    def test_get_roi_names(self, test_tiff_path):
        """Test getting ROI names"""
        selector = InteractiveROISelector(test_tiff_path)

        # Add multiple ROIs
        shape_data = {
            'type': 'rect',
            'x0': 10,
            'y0': 10,
            'x1': 50,
            'y1': 50
        }
        selector.add_roi_from_shape(shape_data, name="roi1")
        selector.add_roi_from_shape(shape_data, name="roi2")
        selector.add_roi_from_shape(shape_data, name="roi3")

        names = selector.get_roi_names()

        assert len(names) == 3
        assert "roi1" in names
        assert "roi2" in names
        assert "roi3" in names

    def test_save_rois(self, test_tiff_path, temp_output_dir):
        """Test saving ROIs to zip file"""
        selector = InteractiveROISelector(test_tiff_path)

        # Add an ROI
        shape_data = {
            'type': 'rect',
            'x0': 10,
            'y0': 10,
            'x1': 50,
            'y1': 50
        }
        selector.add_roi_from_shape(shape_data, name="test_roi")

        # Save ROIs
        output_path = temp_output_dir / "test_rois.zip"
        saved_path = selector.save_rois(output_path)

        assert saved_path.exists()
        assert zipfile.is_zipfile(saved_path)

        # Check contents
        with zipfile.ZipFile(saved_path, 'r') as zf:
            names = zf.namelist()
            assert "test_roi.roi" in names

    def test_load_rois(self, test_tiff_path, test_data_dir):
        """Test loading ROIs from zip file"""
        selector = InteractiveROISelector(test_tiff_path)

        roi_path = test_data_dir / "ROI2.zip"
        if not roi_path.exists():
            pytest.skip("Test ROI file not found")

        # Load ROIs
        selector.load_rois(roi_path)

        assert len(selector.rois) > 0
        # Check that ROI names are loaded
        names = selector.get_roi_names()
        assert len(names) > 0

    def test_create_grid_rois(self, test_tiff_path):
        """Test creating grid ROIs"""
        selector = InteractiveROISelector(test_tiff_path)

        # Create 4x4 grid
        selector.create_grid_rois(grid_size=4)

        assert len(selector.rois) > 0
        # All grid ROIs should be rectangles
        for roi in selector.rois:
            assert roi['shape'] == 'rect'
            assert 'grid_4x4' in roi['name']

    def test_extract_spectra(self, test_tiff_path, temp_output_dir):
        """Test extracting spectra from ROIs"""
        selector = InteractiveROISelector(test_tiff_path)

        # Add an ROI
        shape_data = {
            'type': 'rect',
            'x0': 10,
            'y0': 10,
            'x1': 50,
            'y1': 50
        }
        selector.add_roi_from_shape(shape_data, name="test_roi")

        # Extract spectra (without saving)
        results = selector.extract_spectra(output_dir=None)

        assert len(results) == 1
        assert "test_roi" in results
        assert len(results["test_roi"]) > 0

    def test_convert_roi_to_imagej_rectangle(self, test_tiff_path):
        """Test converting rectangle ROI to ImageJ format"""
        selector = InteractiveROISelector(test_tiff_path)

        roi = {
            'name': 'test_rect',
            'shape': 'rect',
            'coords': np.array([[10, 10], [50, 10], [50, 50], [10, 50]])
        }

        imagej_roi = selector.convert_roi_to_imagej(roi)

        assert imagej_roi.name == 'test_rect'
        assert imagej_roi.roitype == ImagejRoi.RECT

    def test_convert_roi_to_imagej_oval(self, test_tiff_path):
        """Test converting oval ROI to ImageJ format"""
        selector = InteractiveROISelector(test_tiff_path)

        # Create ellipse coordinates
        theta = np.linspace(0, 2*np.pi, 100)
        x = 30 + 20 * np.cos(theta)
        y = 30 + 20 * np.sin(theta)
        coords = np.column_stack([x, y])

        roi = {
            'name': 'test_oval',
            'shape': 'circle',
            'coords': coords
        }

        imagej_roi = selector.convert_roi_to_imagej(roi)

        assert imagej_roi.name == 'test_oval'
        assert imagej_roi.roitype == ImagejRoi.OVAL

    def test_auto_roi_naming(self, test_tiff_path):
        """Test automatic ROI naming"""
        selector = InteractiveROISelector(test_tiff_path)

        shape_data = {
            'type': 'rect',
            'x0': 10,
            'y0': 10,
            'x1': 50,
            'y1': 50
        }

        # Add ROIs without names
        selector.add_roi_from_shape(shape_data)
        selector.add_roi_from_shape(shape_data)
        selector.add_roi_from_shape(shape_data)

        names = selector.get_roi_names()
        assert "ROI_1" in names
        assert "ROI_2" in names
        assert "ROI_3" in names

    def test_empty_save_warning(self, test_tiff_path):
        """Test warning when saving with no ROIs"""
        selector = InteractiveROISelector(test_tiff_path)

        # Try to save without any ROIs
        with pytest.warns(UserWarning):
            selector.save_rois()

    def test_extract_spectra_no_rois(self, test_tiff_path):
        """Test extracting spectra with no ROIs"""
        selector = InteractiveROISelector(test_tiff_path)

        # Try to extract without any ROIs
        with pytest.warns(UserWarning):
            results = selector.extract_spectra()

        assert len(results) == 0


class TestLaunchInteractiveTool:
    """Test launch_interactive_tool function"""

    def test_launch_jupyter_mode(self, test_tiff_path):
        """Test launching in Jupyter mode (without actually displaying)"""
        from spectral_roi.interactive import launch_interactive_tool

        # This should create a selector but not display
        # We can't test the actual display without a Jupyter environment
        selector = launch_interactive_tool.__wrapped__ if hasattr(launch_interactive_tool, '__wrapped__') else None

        # Just verify the function exists and can be imported
        assert launch_interactive_tool is not None

    def test_dash_mode_not_implemented(self, test_tiff_path):
        """Test that Dash mode raises NotImplementedError"""
        from spectral_roi.interactive import launch_interactive_tool

        with pytest.raises(NotImplementedError):
            launch_interactive_tool(test_tiff_path, mode='dash')

    def test_invalid_mode(self, test_tiff_path):
        """Test invalid mode raises ValueError"""
        from spectral_roi.interactive import launch_interactive_tool

        with pytest.raises(ValueError):
            launch_interactive_tool(test_tiff_path, mode='invalid')


class TestEdgeCases:
    """Test edge cases and error handling"""

    def test_nonexistent_tiff(self):
        """Test with non-existent TIFF file"""
        with pytest.raises(FileNotFoundError):
            InteractiveROISelector("nonexistent.tiff")

    def test_nonexistent_roi_load(self, test_tiff_path):
        """Test loading non-existent ROI file"""
        selector = InteractiveROISelector(test_tiff_path)

        with pytest.raises(FileNotFoundError):
            selector.load_rois("nonexistent.zip")

    def test_rename_nonexistent_roi(self, test_tiff_path):
        """Test renaming non-existent ROI"""
        selector = InteractiveROISelector(test_tiff_path)

        with pytest.warns(UserWarning):
            selector.rename_roi("nonexistent", "new_name")

    def test_invalid_stack_range(self, test_tiff_path):
        """Test invalid stack range"""
        # This should work but produce a smaller or empty stack
        selector = InteractiveROISelector(test_tiff_path, stack_range=(1000, 2000))

        # Stack should be empty or very small
        assert selector.stack.shape[0] == 0 or selector.stack.shape[0] < 1000
