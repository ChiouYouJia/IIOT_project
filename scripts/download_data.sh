#!/bin/bash
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
RAW_DIR="$PROJECT_DIR/data/raw"

echo "=== ICS-Flow Dataset Downloader ==="
echo "Target: $RAW_DIR"
echo ""

mkdir -p "$RAW_DIR"

if ! command -v kaggle &> /dev/null; then
    echo "Installing kaggle CLI..."
    pip install kaggle
fi

if [ ! -f ~/.kaggle/kaggle.json ]; then
    echo "ERROR: Kaggle credentials not found."
    echo ""
    echo "Steps to set up:"
    echo "  1. Go to https://www.kaggle.com/settings"
    echo "  2. Click 'Create New Token' to download kaggle.json"
    echo "  3. Run: mkdir -p ~/.kaggle && mv kaggle.json ~/.kaggle/ && chmod 600 ~/.kaggle/kaggle.json"
    echo ""
    echo "Then run this script again."
    exit 1
fi

echo "Downloading ICS-Flow dataset from Kaggle..."
kaggle datasets download -d alirezadehlaghi/icssim -p "$RAW_DIR" --unzip

echo ""
echo "Download complete! Files in $RAW_DIR:"
ls -lh "$RAW_DIR"
echo ""
echo "Done."
