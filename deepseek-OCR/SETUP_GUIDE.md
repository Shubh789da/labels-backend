# DeepSeek-OCR FastAPI Setup Guide

## Quick Setup on RunPod Instance

### Step 1: Connect to RunPod via SSH

```bash
ssh i43wxscinj4qlt-64410ed2@ssh.runpod.io -i $env:USERPROFILE\.ssh\runpod_key
```

### Step 2: Copy Files to RunPod

**Option A: Using SCP (from your local machine)**
```bash
scp -i $env:USERPROFILE\.ssh\runpod_key app.py setup.sh requirements.txt i43wxscinj4qlt-64410ed2@ssh.runpod.io:~/
```

**Option B: Create files directly on RunPod (recommended)**

After SSH'ing into RunPod, run:

```bash
# Create app directory
mkdir -p ~/deepseek-ocr
cd ~/deepseek-ocr

# Create app.py (copy and paste the content)
nano app.py
# Paste the content from the local app.py file, then Ctrl+X, Y, Enter to save

# Create requirements.txt
nano requirements.txt
# Paste the content from the local requirements.txt file
```

### Step 3: Run Setup Commands on RunPod

```bash
# Update system
apt update && apt upgrade -y

# Install Python 3.12
apt install -y python3.12 python3.12-venv python3.12-dev python3-pip

# Create and activate virtual environment
python3.12 -m venv ~/ocr_env
source ~/ocr_env/bin/activate

# Upgrade pip
python -m pip install --upgrade pip setuptools wheel

# Navigate to app directory
cd ~/deepseek-ocr

# Install PyTorch with CUDA 11.8
pip install torch==2.3.0 torchvision==0.18 --index-url https://download.pytorch.org/whl/cu118

# Install core dependencies
pip install fastapi uvicorn[standard] pydantic requests PyYAML
pip install addict transformers==4.46.3 tokenizers==0.20.3 PyMuPDF img2pdf einops easydict Pillow numpy
pip install 'accelerate>=0.26.0'

# Install flash-attention (takes 10-20 minutes)
pip install flash-attn==2.7.3 --no-build-isolation
```

### Step 4: Start the Server

```bash
# Make sure you're in the right directory with venv activated
source ~/ocr_env/bin/activate
cd ~/deepseek-ocr

# Run the server
python app.py
```

The server will:
1. Load the DeepSeek-OCR model (first run downloads ~15GB model)
2. Start listening on `http://0.0.0.0:8000`

---

## API Endpoints

### Health Check
```
GET /
GET /health
```

### Process PDF from URL
```
POST /ocr/pdf
Content-Type: application/json

{
    "pdf_url": "https://example.com/document.pdf",
    "model_size": "Gundam",
    "prompt": "<|grounding|>Convert the document to markdown.",
    "pages": [0, 1, 2]  // Optional: specific pages (0-indexed), null for all
}
```

### Process Image from URL
```
POST /ocr/image
Content-Type: application/json

{
    "image_url": "https://example.com/image.png",
    "model_size": "Gundam",
    "prompt": "<|grounding|>Convert the document to markdown."
}
```

---

## Model Sizes

| Size | Resolution | Speed | Accuracy |
|------|------------|-------|----------|
| Tiny | 512x512 | Fastest | Lower |
| Small | 640x640 | Fast | Good |
| Base | 1024x1024 | Medium | Better |
| Large | 1280x1280 | Slower | Best |
| **Gundam** | 1024+640 crop | Optimized | **Recommended for documents** |

---

## Available Prompts

- `<|grounding|>Convert the document to markdown.` - Document to Markdown
- `<|grounding|>OCR this image.` - Standard OCR
- `Free OCR.` - Text extraction without layout
- `Parse the figure.` - Charts/diagrams
- `Describe this image in detail.` - Image description

---

## Testing the API

### Using cURL

```bash
# Health check
curl http://YOUR_RUNPOD_IP:8000/health

# Process PDF
curl -X POST http://YOUR_RUNPOD_IP:8000/ocr/pdf \
  -H "Content-Type: application/json" \
  -d '{
    "pdf_url": "https://arxiv.org/pdf/2301.00001.pdf",
    "model_size": "Gundam"
  }'

# Process Image
curl -X POST http://YOUR_RUNPOD_IP:8000/ocr/image \
  -H "Content-Type: application/json" \
  -d '{
    "image_url": "https://example.com/document.png",
    "model_size": "Gundam"
  }'
```

### Using Python

```python
import requests

# Process PDF
response = requests.post(
    "http://YOUR_RUNPOD_IP:8000/ocr/pdf",
    json={
        "pdf_url": "https://arxiv.org/pdf/2301.00001.pdf",
        "model_size": "Gundam",
        "prompt": "<|grounding|>Convert the document to markdown."
    }
)
print(response.json())
```

---

## RunPod Port Forwarding

To access the API from outside RunPod, you may need to:

1. **Use RunPod's HTTP Service**:
   - Go to RunPod dashboard → Your Pod → Connect → HTTP Service
   - Set port to 8000

2. **Or use SSH Port Forwarding** (from your local machine):
   ```bash
   ssh -i $env:USERPROFILE\.ssh\runpod_key -L 8000:localhost:8000 i43wxscinj4qlt-64410ed2@ssh.runpod.io
   ```
   Then access API at `http://localhost:8000`

---

## Running as Background Service

To keep the server running after disconnecting SSH:

```bash
# Using nohup
nohup python app.py > server.log 2>&1 &

# Or using screen
screen -S ocr
source ~/ocr_env/bin/activate
cd ~/deepseek-ocr
python app.py
# Press Ctrl+A, then D to detach
# Reconnect with: screen -r ocr
```

---

## Troubleshooting

### CUDA Out of Memory
- Use smaller model_size (Tiny or Small)
- Process fewer pages at once

### Model Download Issues
- Ensure enough disk space (~20GB)
- Check internet connection

### Port Already in Use
```bash
# Find and kill process on port 8000
lsof -i :8000
kill -9 <PID>
```

### Flash Attention Install Fails
```bash
# Try without flash attention (slower but works)
# Edit app.py line with _attn_implementation and change to:
# _attn_implementation='eager'
```
