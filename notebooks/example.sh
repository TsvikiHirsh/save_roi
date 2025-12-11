#!/bin/bash
#
# Save ROI - Example Usage Script
#
# This script demonstrates common usage patterns for the save-roi tool.
# Make it executable: chmod +x example.sh
#

set -e  # Exit on error

echo "======================================================================"
echo "Save ROI - Example Usage"
echo "======================================================================"
echo ""

# ============================================================================
# Example 1: Quick Workflow (Auto-Discovery)
# ============================================================================
echo "Example 1: Quick Workflow"
echo "------------------------"
echo "Assumes directory structure:"
echo "  your_project/"
echo "  ├── ROI2.zip"
echo "  └── data/"
echo "      └── final/"
echo "          └── image.tiff"
echo ""
echo "Command:"
echo "  save-roi data"
echo ""
echo "What it does:"
echo "  - Auto-discovers ROI files in current directory"
echo "  - Auto-discovers TIFF files in data/final/"
echo "  - Outputs CSV files to data/image_ROI_Spectra/"
echo ""

# ============================================================================
# Example 2: Specific TIFF and ROI Files
# ============================================================================
echo "Example 2: Specific Files"
echo "------------------------"
echo "Command:"
echo "  save-roi --tiff image.tiff --roi ROI2.zip"
echo ""
echo "What it does:"
echo "  - Processes image.tiff with ROI2.zip"
echo "  - Outputs CSV files to image_ROI_Spectra/"
echo ""

# ============================================================================
# Example 3: Full Image Analysis (No ROI)
# ============================================================================
echo "Example 3: Full Image (No ROI)"
echo "------------------------------"
echo "Command:"
echo "  save-roi --tiff image.tiff --mode full"
echo ""
echo "What it does:"
echo "  - Analyzes entire image without ROI constraints"
echo "  - Outputs single CSV file: full_image.csv"
echo ""

# ============================================================================
# Example 4: Grid Mode - Spatial Downsampling
# ============================================================================
echo "Example 4: Grid Mode (4×4 blocks)"
echo "---------------------------------"
echo "Command:"
echo "  save-roi --tiff image.tiff --mode grid --grid-size 4"
echo ""
echo "What it does:"
echo "  - Divides image into 4×4 pixel blocks"
echo "  - Each block is averaged"
echo "  - Good for spatial downsampling"
echo "  - Uses 10 CPU cores by default"
echo ""

# ============================================================================
# Example 5: Grid Mode with Custom Settings
# ============================================================================
echo "Example 5: Grid Mode (Custom)"
echo "----------------------------"
echo "Command:"
echo "  save-roi --tiff image.tiff --mode grid --grid-size 8 --jobs 4"
echo ""
echo "What it does:"
echo "  - Uses 8×8 pixel blocks (faster, lower resolution)"
echo "  - Uses only 4 CPU cores (good for large files)"
echo "  - Reduces memory usage"
echo ""

# ============================================================================
# Example 6: Custom Output Directory
# ============================================================================
echo "Example 6: Custom Output"
echo "-----------------------"
echo "Command:"
echo "  save-roi --tiff image.tiff --roi ROI2.zip --output ./my_results"
echo ""
echo "What it does:"
echo "  - Saves CSV files to ./my_results/ instead of default location"
echo ""

# ============================================================================
# Example 7: Use All CPU Cores
# ============================================================================
echo "Example 7: Maximum Performance"
echo "-----------------------------"
echo "Command:"
echo "  save-roi --tiff image.tiff --mode grid --jobs -1"
echo ""
echo "What it does:"
echo "  - Uses all available CPU cores"
echo "  - Fastest processing (but uses more memory)"
echo ""

# ============================================================================
# Advanced Examples (see notebooks/example_usage.ipynb)
# ============================================================================
echo ""
echo "======================================================================"
echo "Advanced Features"
echo "======================================================================"
echo ""
echo "The following features are available for advanced users:"
echo ""
echo "  - Tilt correction: --tilt <roi_name>"
echo "    Straighten and center images using a symmetry line ROI"
echo ""
echo "  - Pixel mode: --mode pixel --stride <N>"
echo "    Extract individual pixel spectra (creates many files!)"
echo ""
echo "See notebooks/example_usage.ipynb for detailed examples."
echo ""

# ============================================================================
# Memory Optimization Tips
# ============================================================================
echo "======================================================================"
echo "Memory Optimization (for 3GB+ files)"
echo "======================================================================"
echo ""
echo "If you're processing large TIFF stacks (3GB+), try these tips:"
echo ""
echo "  1. Reduce parallel jobs:"
echo "     save-roi data --jobs 4"
echo ""
echo "  2. Use larger grid size:"
echo "     save-roi data --mode grid --grid-size 16"
echo ""
echo "  3. ROI mode is most memory-efficient:"
echo "     save-roi data --mode roi"
echo ""
echo "Memory usage with shared memory optimization:"
echo "  - 3GB TIFF stack = ~3GB RAM (regardless of CPU cores)"
echo "  - Old behavior (pre-v0.2.0) = ~30GB RAM with 10 cores"
echo ""

# ============================================================================
# Output Format
# ============================================================================
echo "======================================================================"
echo "Output Format"
echo "======================================================================"
echo ""
echo "All CSV files have the same structure:"
echo ""
echo "  | stack | counts | err    |"
echo "  |-------|--------|--------|"
echo "  | 1     | 12345  | 111.11 |"
echo "  | 2     | 12389  | 111.30 |"
echo "  | ...   | ...    | ...    |"
echo ""
echo "Where:"
echo "  - stack: Slice number (1-indexed, matching ImageJ)"
echo "  - counts: Sum of pixel values in the region"
echo "  - err: Error estimate (sqrt of counts, Poisson statistics)"
echo ""

echo "======================================================================"
echo "For more examples, see:"
echo "  - notebooks/example_usage.ipynb (Jupyter notebook)"
echo "  - README.md (full documentation)"
echo "======================================================================"
