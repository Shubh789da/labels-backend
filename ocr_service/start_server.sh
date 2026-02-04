#!/bin/bash
# Startup script for DeepSeek-OCR-2 Service on RunPod

set -e

echo "======================================"
echo "DeepSeek-OCR-2 Service Startup"
echo "======================================"

# Navigate to workspace
cd /workspace/ocr_service

# Install system dependencies if not present
echo "Checking system dependencies..."
if ! command -v pdftoppm &> /dev/null; then
    echo "Installing poppler-utils..."
    apt-get update && apt-get install -y poppler-utils
fi

# FORCE REINSTALL PyTorch stack to fix "torchvision::nms" error & "conflicting dependencies"
# The correct compatibility matrix for Torch 2.6.0 is:
# - torch==2.6.0
# - torchvision==0.21.0
# - torchaudio==2.6.0  <-- This was the missing piece!
echo "Aligning PyTorch versions..."
pip uninstall -y torch torchvision torchaudio
pip install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/cu118

# Install other Python dependencies
echo "Installing other dependencies..."
pip install -r requirements.txt

# Install flash-attn separately (requires special handling)
echo "Installing flash-attn..."
pip install flash-attn==2.7.3 --no-build-isolation || echo "flash-attn may already be installed"

# Create temp directories
mkdir -p /tmp/ocr_processing/output

# Set environment variables (if .env exists)
if [ -f .env ]; then
    echo "Loading environment from .env file..."
    export $(cat .env | grep -v '^#' | xargs)
fi

# Download model on first run (will be cached)
echo "Pre-loading DeepSeek-OCR-2 model..."
python -c "from ocr_model import get_ocr_model; get_ocr_model().load_model()" || echo "Model will load on first request"

# Start the FastAPI server
echo "Starting OCR Service on port 8000..."
uvicorn main:app --host 0.0.0.0 --port 8000
