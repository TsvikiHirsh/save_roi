#!/bin/bash

# Bash wrapper script for save-roi
# This provides a convenient shell interface to the Python package

# Check if save-roi is installed
if ! command -v save-roi &> /dev/null; then
    echo "Error: save-roi is not installed"
    echo "Please install it with: pip install -e ."
    exit 1
fi

# Forward all arguments to the save-roi command
save-roi "$@"
