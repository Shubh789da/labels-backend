#!/bin/bash
# DeepSeek-OCR FastAPI Setup Script for H100 SXM
# Container: nvidia/cuda:12.1.0-cudnn8-devel-ubuntu22.04

set -e

echo "=============================================="
echo "DeepSeek-OCR FastAPI Setup (H100 SXM)"
echo "=============================================="

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

# Install PyTorch with CUDA 12.1 (optimized for H100)
echo "[6/7] Installing PyTorch with CUDA 12.1 (H100 optimized)..."
pip install torch==2.3.1 torchvision==0.18 --index-url https://download.pytorch.org/whl/cu121

# Install other dependencies
echo "[7/7] Installing dependencies..."
pip install fastapi uvicorn[standard] pydantic requests PyYAML
pip install addict transformers==4.46.3 tokenizers==0.20.3 PyMuPDF img2pdf einops easydict Pillow numpy
pip install 'accelerate>=0.26.0'

# Install flash-attn (H100 benefits significantly from Flash Attention 2)
echo "Installing flash-attention 2 (this may take 10-20 minutes)..."
pip install flash-attn==2.7.3 --no-build-isolation

echo "=============================================="
echo "H100 SXM Setup Complete!"
echo "=============================================="
echo ""
echo "Next steps:"
echo "1. Copy app_h100.py to ~/deepseek-ocr/"
echo "2. Run: source ~/ocr_env/bin/activate"
echo "3. Run: cd ~/deepseek-ocr && python app_h100.py"
echo ""
echo "H100 Optimizations:"
echo "  - CUDA 12.1 (Hopper architecture support)"
echo "  - Flash Attention 2"
echo "  - BFloat16 precision"
echo "  - Higher resolution support (3072px)"
echo "  - TF32 matrix multiplication"
echo ""
echo "The API will be available at http://0.0.0.0:8000"
echo "=============================================="
