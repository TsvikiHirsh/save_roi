"""
Interactive ROI selection tool using Plotly and ipywidgets.

This module provides an interactive tool similar to ImageJ's ROI Manager
for selecting and managing ROIs on TIFF stacks.
"""

import numpy as np
import warnings
from pathlib import Path
from typing import Union, Optional, List, Dict, Tuple
import zipfile
import tempfile

try:
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False
    warnings.warn("Plotly not installed. Install with: pip install save-roi[interactive]")

try:
    import ipywidgets as widgets
    from IPython.display import display, clear_output
    IPYWIDGETS_AVAILABLE = True
except ImportError:
    IPYWIDGETS_AVAILABLE = False

try:
    from roifile import ImagejRoi, roiwrite
    ROIFILE_AVAILABLE = True
except ImportError:
    ROIFILE_AVAILABLE = False
    warnings.warn("roifile not installed. Install with: pip install roifile")

from .core import load_tiff_stack, extract_roi_spectra, extract_grid_spectra


class InteractiveROISelector:
    """
    Interactive ROI selection tool for TIFF stacks.

    This class provides functionality similar to ImageJ's ROI Manager,
    allowing users to:
    - Display summed TIFF stacks
    - Draw and edit ROIs (rectangle, ellipse, polygon)
    - Name and manage ROIs
    - Save/load ROI files
    - Extract spectra from ROIs
    - Create grid-based ROIs

    Parameters
    ----------
    tiff_path : str or Path
        Path to the TIFF stack file
    stack_range : tuple of int, optional
        (start, end) slice indices to sum. If None, sums all slices.
    """

    def __init__(self, tiff_path: Union[str, Path], stack_range: Optional[Tuple[int, int]] = None):
        if not PLOTLY_AVAILABLE:
            raise ImportError("Plotly is required. Install with: pip install save-roi[interactive]")

        self.tiff_path = Path(tiff_path)
        if not self.tiff_path.exists():
            raise FileNotFoundError(f"TIFF file not found: {self.tiff_path}")

        # Load TIFF stack
        self.stack = load_tiff_stack(self.tiff_path)

        # Apply stack range if specified
        if stack_range is not None:
            start, end = stack_range
            self.stack = self.stack[start:end]

        # Sum the stack for display
        self.summed_image = np.sum(self.stack, axis=0)

        # ROI storage: list of dicts with 'name', 'shape', 'coords'
        self.rois: List[Dict] = []
        self.roi_counter = 1

        # UI components (will be initialized in Jupyter mode)
        self.fig = None
        self.roi_list_widget = None
        self.name_input = None
        self.shape_selector = None

    def sum_stack_range(self, start: int, end: int) -> np.ndarray:
        """
        Sum a specific range of slices from the stack.

        Parameters
        ----------
        start : int
            Starting slice index (0-indexed)
        end : int
            Ending slice index (exclusive)

        Returns
        -------
        np.ndarray
            Summed 2D image
        """
        return np.sum(self.stack[start:end], axis=0)

    def create_figure(self) -> go.Figure:
        """
        Create a Plotly figure with the summed image and drawing tools.

        Returns
        -------
        go.Figure
            Interactive Plotly figure
        """
        fig = go.Figure()

        # Add the image
        fig.add_trace(go.Heatmap(
            z=self.summed_image,
            colorscale='Greys_r',
            showscale=True,
            hovertemplate='x: %{x}<br>y: %{y}<br>intensity: %{z}<extra></extra>'
        ))

        # Configure layout for drawing
        fig.update_layout(
            title=f"ROI Selection - {self.tiff_path.name}",
            xaxis=dict(title="X", constrain='domain'),
            yaxis=dict(title="Y", scaleanchor="x", scaleratio=1, constrain='domain'),
            dragmode='drawrect',  # Default drawing mode
            newshape=dict(
                line=dict(color='cyan', width=2),
                fillcolor='rgba(0, 255, 255, 0.1)',
                opacity=0.5
            ),
            height=700,
            width=900
        )

        # Add drawing mode config
        fig.update_layout(
            modebar_add=[
                'drawrect',
                'drawcircle',
                'drawclosedpath',
                'eraseshape'
            ]
        )

        self.fig = fig
        return fig

    def add_roi_from_shape(self, shape_data: Dict, name: Optional[str] = None):
        """
        Add an ROI from Plotly shape data.

        Parameters
        ----------
        shape_data : dict
            Plotly shape dictionary containing type and coordinates
        name : str, optional
            Name for the ROI. If None, auto-generates a name.
        """
        if name is None:
            name = f"ROI_{self.roi_counter}"
            self.roi_counter += 1

        roi = {
            'name': name,
            'shape': shape_data['type'],
            'coords': self._extract_coords_from_shape(shape_data)
        }

        self.rois.append(roi)
        print(f"Added ROI: {name}")

    def _extract_coords_from_shape(self, shape_data: Dict) -> np.ndarray:
        """
        Extract coordinates from Plotly shape data.

        Parameters
        ----------
        shape_data : dict
            Plotly shape dictionary

        Returns
        -------
        np.ndarray
            Array of coordinates
        """
        shape_type = shape_data['type']

        if shape_type == 'rect':
            x0, y0 = shape_data['x0'], shape_data['y0']
            x1, y1 = shape_data['x1'], shape_data['y1']
            # Return corner coordinates
            return np.array([[x0, y0], [x1, y0], [x1, y1], [x0, y1]])

        elif shape_type == 'circle':
            x0, y0 = shape_data['x0'], shape_data['y0']
            x1, y1 = shape_data['x1'], shape_data['y1']
            # Calculate center and radii
            cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
            rx, ry = abs(x1 - x0) / 2, abs(y1 - y0) / 2
            # Generate ellipse points
            theta = np.linspace(0, 2*np.pi, 100)
            x = cx + rx * np.cos(theta)
            y = cy + ry * np.sin(theta)
            return np.column_stack([x, y])

        elif shape_type == 'path':
            # Extract path coordinates
            path = shape_data.get('path', '')
            coords = self._parse_svg_path(path)
            return coords

        else:
            raise ValueError(f"Unsupported shape type: {shape_type}")

    def _parse_svg_path(self, path_string: str) -> np.ndarray:
        """
        Parse SVG path string to extract coordinates.

        Parameters
        ----------
        path_string : str
            SVG path string

        Returns
        -------
        np.ndarray
            Array of (x, y) coordinates
        """
        coords = []
        # Simple parser for M (move) and L (line) commands
        commands = path_string.split()
        i = 0
        while i < len(commands):
            cmd = commands[i]
            if cmd in ['M', 'L']:
                i += 1
                x, y = float(commands[i].rstrip(',')), float(commands[i+1])
                coords.append([x, y])
                i += 2
            else:
                i += 1

        return np.array(coords) if coords else np.array([[0, 0]])

    def convert_roi_to_imagej(self, roi: Dict) -> ImagejRoi:
        """
        Convert an ROI dictionary to ImageJ ROI format.

        Parameters
        ----------
        roi : dict
            ROI dictionary with 'name', 'shape', and 'coords'

        Returns
        -------
        ImagejRoi
            ImageJ ROI object
        """
        if not ROIFILE_AVAILABLE:
            raise ImportError("roifile is required. Install with: pip install roifile")

        coords = roi['coords']
        shape_type = roi['shape']

        # Determine ROI type
        if shape_type == 'rect':
            # Rectangle ROI (type 1)
            x_coords = coords[:, 0]
            y_coords = coords[:, 1]
            left = int(np.min(x_coords))
            top = int(np.min(y_coords))
            width = int(np.max(x_coords) - np.min(x_coords))
            height = int(np.max(y_coords) - np.min(y_coords))

            imagej_roi = ImagejRoi.frompoints(coords, roitype=ImagejRoi.RECT)
            imagej_roi.left = left
            imagej_roi.top = top
            imagej_roi.right = left + width
            imagej_roi.bottom = top + height

        elif shape_type == 'circle':
            # Oval ROI (type 2)
            x_coords = coords[:, 0]
            y_coords = coords[:, 1]
            left = int(np.min(x_coords))
            top = int(np.min(y_coords))
            width = int(np.max(x_coords) - np.min(x_coords))
            height = int(np.max(y_coords) - np.min(y_coords))

            imagej_roi = ImagejRoi.frompoints(coords, roitype=ImagejRoi.OVAL)
            imagej_roi.left = left
            imagej_roi.top = top
            imagej_roi.right = left + width
            imagej_roi.bottom = top + height

        else:  # path or polygon
            # Polygon ROI (type 0)
            imagej_roi = ImagejRoi.frompoints(coords, roitype=ImagejRoi.POLYGON)

        # Set the name
        imagej_roi.name = roi['name']

        return imagej_roi

    def save_rois(self, output_path: Optional[Union[str, Path]] = None):
        """
        Save all ROIs to a zip file (ImageJ format).

        Parameters
        ----------
        output_path : str or Path, optional
            Output path for the ROI zip file. If None, uses default name
            based on TIFF filename.
        """
        if not self.rois:
            warnings.warn("No ROIs to save")
            return

        if output_path is None:
            output_path = self.tiff_path.parent / f"{self.tiff_path.stem}_ROIs.zip"
        else:
            output_path = Path(output_path)

        # Create a temporary directory for individual ROI files
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir_path = Path(tmpdir)
            roi_files = []

            # Convert and save each ROI
            for roi in self.rois:
                imagej_roi = self.convert_roi_to_imagej(roi)
                roi_file = tmpdir_path / f"{roi['name']}.roi"
                imagej_roi.tofile(str(roi_file))
                roi_files.append(roi_file)

            # Create zip file
            with zipfile.ZipFile(output_path, 'w') as zf:
                for roi_file in roi_files:
                    zf.write(roi_file, arcname=roi_file.name)

        print(f"Saved {len(self.rois)} ROI(s) to: {output_path}")
        return output_path

    def load_rois(self, roi_path: Union[str, Path]):
        """
        Load ROIs from an ImageJ ROI zip file.

        Parameters
        ----------
        roi_path : str or Path
            Path to the ROI file (.roi or .zip)
        """
        from .core import load_imagej_rois, get_roi_mask

        roi_path = Path(roi_path)
        if not roi_path.exists():
            raise FileNotFoundError(f"ROI file not found: {roi_path}")

        # Load ROIs using core functionality
        loaded_rois = load_imagej_rois(roi_path)

        # Convert to internal format
        for roi_info in loaded_rois:
            roi_obj = roi_info['roi_object']
            coords = roi_obj.coordinates()

            # Determine shape type
            if roi_obj.roitype == ImagejRoi.RECT:
                shape_type = 'rect'
            elif roi_obj.roitype == ImagejRoi.OVAL:
                shape_type = 'circle'
            else:
                shape_type = 'path'

            roi = {
                'name': roi_info['name'],
                'shape': shape_type,
                'coords': coords
            }
            self.rois.append(roi)

        print(f"Loaded {len(loaded_rois)} ROI(s) from: {roi_path}")

        # Update counter
        self.roi_counter = len(self.rois) + 1

    def rename_roi(self, old_name: str, new_name: str):
        """
        Rename an ROI.

        Parameters
        ----------
        old_name : str
            Current ROI name
        new_name : str
            New ROI name
        """
        for roi in self.rois:
            if roi['name'] == old_name:
                roi['name'] = new_name
                print(f"Renamed ROI: {old_name} -> {new_name}")
                return

        warnings.warn(f"ROI not found: {old_name}")

    def delete_roi(self, name: str):
        """
        Delete an ROI by name.

        Parameters
        ----------
        name : str
            Name of the ROI to delete
        """
        self.rois = [roi for roi in self.rois if roi['name'] != name]
        print(f"Deleted ROI: {name}")

    def get_roi_names(self) -> List[str]:
        """
        Get list of all ROI names.

        Returns
        -------
        List[str]
            List of ROI names
        """
        return [roi['name'] for roi in self.rois]

    def extract_spectra(self, output_dir: Optional[Union[str, Path]] = None):
        """
        Extract spectra for all ROIs using the save_roi functionality.

        Parameters
        ----------
        output_dir : str or Path, optional
            Output directory for CSV files

        Returns
        -------
        dict
            Dictionary mapping ROI names to DataFrames
        """
        if not self.rois:
            warnings.warn("No ROIs defined")
            return {}

        # Save ROIs to temporary file
        with tempfile.NamedTemporaryFile(suffix='.zip', delete=False) as tmp:
            tmp_roi_path = Path(tmp.name)

        try:
            self.save_rois(tmp_roi_path)

            # Extract spectra using core functionality
            results = extract_roi_spectra(
                self.tiff_path,
                roi_path=tmp_roi_path,
                output_dir=output_dir,
                save_csv=(output_dir is not None)
            )

            return results

        finally:
            # Clean up temporary file
            if tmp_roi_path.exists():
                tmp_roi_path.unlink()

    def create_grid_rois(self, grid_size: int = 4):
        """
        Create grid-based ROIs.

        Parameters
        ----------
        grid_size : int
            Size of grid blocks (e.g., 4 for 4x4 pixel blocks)
        """
        height, width = self.summed_image.shape

        # Clear existing ROIs
        self.rois = []
        self.roi_counter = 1

        # Create grid ROIs
        for y in range(0, height, grid_size):
            for x in range(0, width, grid_size):
                x_end = min(x + grid_size, width)
                y_end = min(y + grid_size, height)

                # Create rectangle coordinates
                coords = np.array([
                    [x, y],
                    [x_end, y],
                    [x_end, y_end],
                    [x, y_end]
                ])

                roi = {
                    'name': f"grid_{grid_size}x{grid_size}_x{x}_y{y}",
                    'shape': 'rect',
                    'coords': coords
                }

                self.rois.append(roi)

        print(f"Created {len(self.rois)} grid ROIs with size {grid_size}x{grid_size}")
        self.roi_counter = len(self.rois) + 1

    def show_jupyter(self):
        """
        Display interactive UI in Jupyter notebook.

        This creates an interactive widget interface with:
        - Plotly figure for drawing ROIs
        - ROI list widget
        - Buttons for various operations
        """
        if not IPYWIDGETS_AVAILABLE:
            raise ImportError("ipywidgets is required. Install with: pip install save-roi[interactive]")

        # Create figure
        fig = self.create_figure()

        # Create widgets
        output = widgets.Output()

        # ROI management widgets
        roi_list = widgets.Textarea(
            value='\n'.join(self.get_roi_names()),
            description='ROIs:',
            disabled=True,
            layout=widgets.Layout(width='300px', height='200px')
        )
        self.roi_list_widget = roi_list

        # Name input
        name_input = widgets.Text(
            description='ROI Name:',
            placeholder='Enter ROI name',
            layout=widgets.Layout(width='300px')
        )
        self.name_input = name_input

        # Rename widgets
        old_name_input = widgets.Text(
            description='Old Name:',
            placeholder='Current name',
            layout=widgets.Layout(width='300px')
        )

        new_name_input = widgets.Text(
            description='New Name:',
            placeholder='New name',
            layout=widgets.Layout(width='300px')
        )

        # Shape selector
        shape_selector = widgets.Dropdown(
            options=['Rectangle', 'Circle', 'Polygon'],
            value='Rectangle',
            description='Shape:',
            layout=widgets.Layout(width='300px')
        )
        self.shape_selector = shape_selector

        # Buttons
        add_roi_btn = widgets.Button(description='Add ROI from Drawing', button_style='success')
        save_rois_btn = widgets.Button(description='Save ROIs', button_style='primary')
        load_rois_btn = widgets.Button(description='Load ROIs', button_style='info')
        rename_btn = widgets.Button(description='Rename ROI', button_style='warning')
        delete_btn = widgets.Button(description='Delete ROI', button_style='danger')
        extract_btn = widgets.Button(description='Extract Spectra', button_style='success')
        grid_btn = widgets.Button(description='Create Grid', button_style='info')

        # Grid size input
        grid_size_input = widgets.IntText(
            value=4,
            description='Grid Size:',
            layout=widgets.Layout(width='150px')
        )

        # File path inputs
        save_path_input = widgets.Text(
            description='Save Path:',
            placeholder='Optional: path/to/rois.zip',
            layout=widgets.Layout(width='400px')
        )

        load_path_input = widgets.Text(
            description='Load Path:',
            placeholder='path/to/rois.zip',
            layout=widgets.Layout(width='400px')
        )

        output_dir_input = widgets.Text(
            description='Output Dir:',
            placeholder='Optional: output directory',
            layout=widgets.Layout(width='400px')
        )

        # Button callbacks
        def on_add_roi(b):
            with output:
                clear_output(wait=True)
                if fig.layout.shapes:
                    last_shape = fig.layout.shapes[-1]
                    roi_name = name_input.value or None
                    self.add_roi_from_shape(last_shape, roi_name)
                    roi_list.value = '\n'.join(self.get_roi_names())
                    name_input.value = ''
                else:
                    print("No shapes drawn. Use the drawing tools in the figure.")

        def on_save_rois(b):
            with output:
                clear_output(wait=True)
                save_path = save_path_input.value or None
                self.save_rois(save_path)

        def on_load_rois(b):
            with output:
                clear_output(wait=True)
                if load_path_input.value:
                    try:
                        self.load_rois(load_path_input.value)
                        roi_list.value = '\n'.join(self.get_roi_names())
                    except Exception as e:
                        print(f"Error loading ROIs: {e}")
                else:
                    print("Please enter a path to load ROIs from")

        def on_rename(b):
            with output:
                clear_output(wait=True)
                if old_name_input.value and new_name_input.value:
                    self.rename_roi(old_name_input.value, new_name_input.value)
                    roi_list.value = '\n'.join(self.get_roi_names())
                    old_name_input.value = ''
                    new_name_input.value = ''
                else:
                    print("Please enter both old and new names")

        def on_delete(b):
            with output:
                clear_output(wait=True)
                if old_name_input.value:
                    self.delete_roi(old_name_input.value)
                    roi_list.value = '\n'.join(self.get_roi_names())
                    old_name_input.value = ''
                else:
                    print("Please enter the name of the ROI to delete")

        def on_extract(b):
            with output:
                clear_output(wait=True)
                output_dir = output_dir_input.value or None
                results = self.extract_spectra(output_dir)
                print(f"Extracted spectra for {len(results)} ROI(s)")

        def on_create_grid(b):
            with output:
                clear_output(wait=True)
                self.create_grid_rois(grid_size_input.value)
                roi_list.value = '\n'.join(self.get_roi_names())

        # Attach callbacks
        add_roi_btn.on_click(on_add_roi)
        save_rois_btn.on_click(on_save_rois)
        load_rois_btn.on_click(on_load_rois)
        rename_btn.on_click(on_rename)
        delete_btn.on_click(on_delete)
        extract_btn.on_click(on_extract)
        grid_btn.on_click(on_create_grid)

        # Layout
        left_panel = widgets.VBox([
            widgets.HTML("<h3>ROI Manager</h3>"),
            roi_list,
            widgets.HTML("<br><b>Add ROI:</b>"),
            shape_selector,
            name_input,
            add_roi_btn,
            widgets.HTML("<br><b>Rename/Delete:</b>"),
            old_name_input,
            new_name_input,
            widgets.HBox([rename_btn, delete_btn]),
            widgets.HTML("<br><b>Grid Creation:</b>"),
            widgets.HBox([grid_size_input, grid_btn]),
        ])

        right_panel = widgets.VBox([
            widgets.HTML("<h3>File Operations</h3>"),
            save_path_input,
            save_rois_btn,
            widgets.HTML("<br>"),
            load_path_input,
            load_rois_btn,
            widgets.HTML("<br><b>Extract Spectra:</b>"),
            output_dir_input,
            extract_btn,
        ])

        controls = widgets.HBox([left_panel, right_panel])

        # Display
        display(widgets.VBox([
            go.FigureWidget(fig),
            controls,
            output
        ]))


def launch_interactive_tool(
    tiff_path: Union[str, Path],
    stack_range: Optional[Tuple[int, int]] = None,
    mode: str = 'jupyter'
):
    """
    Launch the interactive ROI selection tool.

    Parameters
    ----------
    tiff_path : str or Path
        Path to the TIFF stack file
    stack_range : tuple of int, optional
        (start, end) slice indices to sum. If None, sums all slices.
    mode : str, default='jupyter'
        Display mode: 'jupyter' for Jupyter notebook, 'dash' for standalone app

    Returns
    -------
    InteractiveROISelector
        The ROI selector instance
    """
    selector = InteractiveROISelector(tiff_path, stack_range)

    if mode == 'jupyter':
        selector.show_jupyter()
    elif mode == 'dash':
        # TODO: Implement Dash-based standalone app
        raise NotImplementedError("Dash mode not yet implemented. Use mode='jupyter' for now.")
    else:
        raise ValueError(f"Unknown mode: {mode}. Use 'jupyter' or 'dash'.")

    return selector
