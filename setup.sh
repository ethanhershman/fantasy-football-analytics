#!/usr/bin/env bash
# One-command setup for the fantasy-football-analytics project.
# Usage: bash setup.sh
#
# Requires Python 3.12 (3.14 is too new for many dependencies).

set -e

PYTHON=${PYTHON:-python3.12}

echo "Creating virtual environment with $PYTHON..."
$PYTHON -m venv venv
source venv/bin/activate

echo "Upgrading pip, setuptools, wheel..."
pip install --upgrade pip setuptools wheel

echo "Installing requirements..."
pip install -r requirements.txt

echo "Installing nfl-data-py (skipping its stale pandas/numpy pins)..."
pip install --no-deps nfl-data-py

echo ""
echo "Done! Activate the environment with:"
echo "  source venv/bin/activate"
