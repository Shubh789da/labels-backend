# DeepSeek-OCR RunPod Serverless Dockerfile
# Optimized for minimal cold start with model baked into image

FROM runpod/pytorch:2.2.0-py3.10-cuda12.1.1-devel-ubuntu22.04

WORKDIR /app

# Set environment variables for optimal performance
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    HF_HOME=/app/hf_cache \
    TRANSFORMERS_CACHE=/app/hf_cache \
    TORCH_HOME=/app/torch_cache

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Copy and install Python dependencies
COPY requirements_serverless.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements_serverless.txt

# Install RunPod SDK
RUN pip install --no-cache-dir runpod

# Install flash-attention (pre-compiled for faster cold start)
RUN pip install --no-cache-dir flash-attn==2.7.3 --no-build-isolation

# Create cache directories
RUN mkdir -p /app/hf_cache /app/torch_cache

# Pre-download the DeepSeek-OCR model during build
# This bakes the ~15GB model into the Docker image
# Critical for minimizing cold start time
RUN python -c "\
from transformers import AutoModel, AutoTokenizer; \
print('Downloading DeepSeek-OCR model...'); \
tokenizer = AutoTokenizer.from_pretrained('deepseek-ai/DeepSeek-OCR', trust_remote_code=True); \
print('Tokenizer downloaded'); \
model = AutoModel.from_pretrained('deepseek-ai/DeepSeek-OCR', trust_remote_code=True); \
print('Model downloaded successfully'); \
"

# Copy handler script
COPY rp_handler.py .

# Health check (optional)
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD python -c "import torch; print('CUDA available:', torch.cuda.is_available())"

# Run the handler
CMD ["python", "-u", "rp_handler.py"]
