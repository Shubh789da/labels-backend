#!/bin/bash
# DeepSeek-OCR FastAPI Setup Script for RunPod
# Run this script on the RunPod instance

set -e

echo "=========================================="
echo "DeepSeek-OCR FastAPI Setup"
echo "=========================================="

# Update system
echo "[1/7] Updating system packages..."
apt update && apt upgrade -y

# Install Python 3.12
echo "[2/7] Installing Python 3.12..."
apt install -y python3.12 python3.12-venv python3.12-dev python3-pip

# Create virtual environment
echo "[3/7] Creating virtual environment..."
python3.12 -m venv ~/ocr_env
source ~/ocr_env/bin/activate

# Upgrade pip
echo "[4/7] Upgrading pip..."
python -m pip install --upgrade pip setuptools wheel

# Create app directory
echo "[5/7] Setting up application directory..."
mkdir -p ~/deepseek-ocr
cd ~/deepseek-ocr

# Install PyTorch first (with CUDA 11.8)
echo "[6/7] Installing PyTorch with CUDA..."
pip install torch==2.3.1 torchvision==0.18 --index-url https://download.pytorch.org/whl/cu118

# Install other dependencies
echo "[7/7] Installing dependencies..."
pip install fastapi uvicorn[standard] pydantic requests PyYAML
pip install addict transformers==4.46.3 tokenizers==0.20.3 PyMuPDF img2pdf einops easydict Pillow numpy
pip install 'accelerate>=0.26.0'

# Install flash-attn (this might take a while)
echo "Installing flash-attention (this may take 10-20 minutes)..."
pip install flash-attn==2.7.3 --no-build-isolation

echo "=========================================="
echo "Setup complete!"
echo "=========================================="
echo ""
echo "Next steps:"
echo "1. Copy app.py to ~/deepseek-ocr/"
echo "2. Run: source ~/ocr_env/bin/activate"
echo "3. Run: cd ~/deepseek-ocr && python app.py"
echo ""
echo "The API will be available at http://0.0.0.0:8000"
echo "=========================================="
