#!/bin/bash
# Script to generate documentation for V3

# Activate the virtual environment
source ../../.venv/bin/activate

# Add the V3 src directory to PYTHONPATH for autodoc
export PYTHONPATH="/home/feynman/projects/PharmaControl/V3/src:${PYTHONPATH}"

# Build the documentation
sphinx-build -b html . _build/html

echo "Documentation generated successfully!"
echo "Open _build/html/index.html to view the documentation."