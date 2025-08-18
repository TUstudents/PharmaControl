#!/bin/bash
# Script to generate documentation for V3

# Activate the virtual environment
source ../../.venv/bin/activate

# Generate API documentation
sphinx-apidoc -o . ../src/ --force

# Build the documentation
sphinx-build -b html . _build/html

echo "Documentation generated successfully!"
echo "Open _build/html/index.html to view the documentation."