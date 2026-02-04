# DeepSeek-OCR RunPod Serverless Deployment Guide

## Overview

This guide deploys DeepSeek-OCR as a serverless endpoint that:
- Scales to **zero** when idle (no cost)
- Spins up **on-demand** when API is called
- Uses **FlashBoot** for fast cold starts (~10-30 seconds)

---

## Prerequisites

1. **Docker Desktop** installed locally
2. **Docker Hub** account (free at hub.docker.com)
3. **RunPod** account with API key

---

## Step 1: Build Docker Image

```bash
# Navigate to project directory
cd d:\CT_FDA\drug_history\deepseek-OCR

# Login to Docker Hub
docker login

# Build the image (this takes 20-30 minutes due to model download)
docker build --platform linux/amd64 -t YOUR_DOCKERHUB_USERNAME/deepseek-ocr-serverless:v1 .

# Example:
# docker build --platform linux/amd64 -t shubhanshu/deepseek-ocr-serverless:v1 .
```

**Note:** The build downloads the ~15GB model and bakes it into the image. Final image size will be ~25-30GB.

---

## Step 2: Push to Docker Hub

```bash
# Push the image
docker push YOUR_DOCKERHUB_USERNAME/deepseek-ocr-serverless:v1

# Example:
# docker push shubhanshu/deepseek-ocr-serverless:v1
```

---

## Step 3: Deploy on RunPod

### 3.1 Create Endpoint

1. Go to [RunPod Console](https://www.runpod.io/console/serverless)
2. Click **"+ New Endpoint"**
3. Select **"Import from Docker Registry"**
4. Enter your image: `YOUR_DOCKERHUB_USERNAME/deepseek-ocr-serverless:v1`

### 3.2 Configure Settings

| Setting | Recommended Value | Notes |
|---------|------------------|-------|
| **GPU Type** | A6000 (48GB) or L40 (48GB) | H100 for fastest |
| **Active Workers** | 0 | Scale to zero |
| **Max Workers** | 3-5 | Based on expected load |
| **Idle Timeout** | 5 seconds | How long to stay warm |
| **Execution Timeout** | 600 seconds | Max job time (10 min) |
| **FlashBoot** | Enabled (default) | Faster cold starts |

### 3.3 GPU Selection

Select one or more GPU types (RunPod will use what's available):

| GPU | VRAM | Price/sec | Recommended For |
|-----|------|-----------|-----------------|
| RTX 4090 | 24GB | $0.00044 | Basic usage |
| A6000 | 48GB | $0.00044 | **Best value** |
| L40 | 48GB | $0.00069 | Good balance |
| A100 | 80GB | $0.00130 | High volume |
| H100 | 80GB | $0.00189 | Maximum speed |

### 3.4 Deploy

1. Click **"Deploy"**
2. Wait for endpoint to initialize
3. Copy your **Endpoint ID** (e.g., `abc123xyz`)

---

## Step 4: Get Your API Key

1. Go to [RunPod Settings](https://www.runpod.io/console/user/settings)
2. Navigate to **"API Keys"**
3. Create a new key or copy existing one

---

## Step 5: Test the Endpoint

### Synchronous Request (wait for result)

```bash
curl -X POST "https://api.runpod.ai/v2/YOUR_ENDPOINT_ID/runsync" \
  -H "Authorization: Bearer YOUR_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "input": {
      "pdf_url": "https://arxiv.org/pdf/2301.00001.pdf",
      "prompt": "<|grounding|>Convert the document to markdown.",
      "model_size": "Gundam",
      "dpi_scale": 2.0
    }
  }'
```

### Asynchronous Request (get job ID, check later)

```bash
# Submit job
curl -X POST "https://api.runpod.ai/v2/YOUR_ENDPOINT_ID/run" \
  -H "Authorization: Bearer YOUR_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "input": {
      "pdf_url": "https://example.com/document.pdf"
    }
  }'

# Response: {"id": "job-abc123", "status": "IN_QUEUE"}

# Check status
curl "https://api.runpod.ai/v2/YOUR_ENDPOINT_ID/status/job-abc123" \
  -H "Authorization: Bearer YOUR_API_KEY"
```

### Python Example

```python
import requests

RUNPOD_API_KEY = "your_api_key"
ENDPOINT_ID = "your_endpoint_id"

response = requests.post(
    f"https://api.runpod.ai/v2/{ENDPOINT_ID}/runsync",
    headers={
        "Authorization": f"Bearer {RUNPOD_API_KEY}",
        "Content-Type": "application/json"
    },
    json={
        "input": {
            "pdf_url": "https://arxiv.org/pdf/2301.00001.pdf",
            "prompt": "<|grounding|>Convert the document to markdown.",
            "model_size": "Gundam"
        }
    },
    timeout=300  # 5 minute timeout for long PDFs
)

result = response.json()
print(result)
```

---

## API Reference

### Input Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `pdf_url` | string | - | URL of PDF to process |
| `image_url` | string | - | URL of image to process (alternative to pdf_url) |
| `prompt` | string | `<\|grounding\|>Convert the document to markdown.` | OCR prompt |
| `model_size` | string | `Gundam` | Model size: Tiny/Small/Base/Large/Gundam |
| `pages` | array | null | Specific pages to process (0-indexed) |
| `process_all` | bool | true | Process all pages |
| `dpi_scale` | float | 2.0 | PDF render quality (1.0-3.0) |
| `max_image_size` | int | 3072 | Max image dimension in pixels |

### Response Format

**PDF Response:**
```json
{
  "id": "job-abc123",
  "status": "COMPLETED",
  "output": {
    "success": true,
    "pages": [
      {
        "page_number": 0,
        "raw_text": "...",
        "parsed_text": "# Document Title\n\n..."
      }
    ],
    "total_pages": 5,
    "message": "Processed 5 pages"
  }
}
```

**Image Response:**
```json
{
  "id": "job-abc123",
  "status": "COMPLETED",
  "output": {
    "success": true,
    "raw_text": "...",
    "parsed_text": "...",
    "message": "Processed image"
  }
}
```

---

## Cost Estimation

| Scenario | GPU | Time | Cost |
|----------|-----|------|------|
| 1-page PDF | A6000 | ~15s | ~$0.007 |
| 10-page PDF | A6000 | ~2min | ~$0.05 |
| 50-page PDF | A6000 | ~10min | ~$0.26 |
| 1000 PDFs/month (10 pages avg) | A6000 | - | ~$50 |

**Scale to Zero = No cost when not processing**

---

## Troubleshooting

### Cold Start Too Slow

- Ensure model is baked into image (check Dockerfile)
- Enable FlashBoot in endpoint settings
- Consider keeping 1 active worker for instant response

### Out of Memory

- Use smaller `model_size` (Tiny or Small)
- Reduce `max_image_size` to 2048
- Select GPU with more VRAM (48GB+)

### Timeout Errors

- Increase execution timeout in endpoint settings
- Process fewer pages per request
- Use async `/run` endpoint instead of `/runsync`

### Image Build Fails

- Ensure Docker has enough disk space (50GB+ recommended)
- Check internet connection for model download
- Try building without cache: `docker build --no-cache ...`

---

## Updating the Endpoint

```bash
# Build new version
docker build --platform linux/amd64 -t YOUR_USERNAME/deepseek-ocr-serverless:v2 .

# Push
docker push YOUR_USERNAME/deepseek-ocr-serverless:v2

# Update in RunPod Console:
# 1. Go to your endpoint
# 2. Click "Edit"
# 3. Change image tag to :v2
# 4. Save
```

---

## Files Reference

| File | Purpose |
|------|---------|
| `rp_handler.py` | RunPod serverless handler |
| `Dockerfile` | Container build instructions |
| `requirements_serverless.txt` | Python dependencies |
| `SERVERLESS_DEPLOY.md` | This guide |
