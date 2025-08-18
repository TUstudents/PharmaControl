#!/bin/bash
# Script to generate documentation for V2

# Activate the virtual environment
source ../../.venv/bin/activate

# Generate API documentation
sphinx-apidoc -o . ../robust_mpc/ --force

# Build the documentation
sphinx-build -b html . _build/html

echo "Documentation generated successfully!"
echo "Open _build/html/index.html to view the documentation."