#!/bin/bash
# Make sure the script stops if any command fails
set -e

# Install required packages using python3
python3 install_requirements.py

# Run the main code using python3
python3 main.py
