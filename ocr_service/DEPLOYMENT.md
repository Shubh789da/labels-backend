# DeepSeek-OCR-2 Service Deployment Guide

## Prerequisites

1. RunPod instance with GPU (RTX 4090 or A100 recommended)
2. SSH key registered with RunPod
3. AWS S3 bucket created
4. RunPod secrets configured

## Your RunPod Secrets

You have these secrets configured:
- `HF_TOKEN` - Hugging Face token
- `AWS_SECRET_ACCESS_KEY` - AWS secret key
- `AWS_SECRET_ACCESS_ID` - AWS access key ID
- `DEEPSEEK_API` - DeepSeek API key

## Step 1: Create a Custom Template on RunPod

1. Go to [RunPod Templates](https://www.runpod.io/console/user/templates)
2. Click **New Template**
3. Configure:
   - **Template Name**: `DeepSeek-OCR-Service`
   - **Container Image**: `runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04` (Recommended)
     * *Note: The older `pytorch:2.1.2` image you mentioned is too old for DeepSeek-OCR-2.*
   - **Container Disk**: `50 GB`
   - **Volume Disk**: `50 GB`
   - **Expose TCP Ports**: `22, 8000`
   
4. **Environment Variables** (Add these):
   - `HF_TOKEN`: (Your Hugging Face Token)
   - `AWS_ACCESS_KEY_ID`: (Your AWS Key ID)
   - `AWS_SECRET_ACCESS_KEY`: (Your AWS Secret Key)
   - `DEEPSEEK_API`: (Your DeepSeek API Key)
   - `PUBLIC_KEY`: (Copy your `id_ed25519.pub` or `runpod_key.pub` content here)

5. **Docker Start Command** (CRITICAL: Copy exactly):
   ```bash
   bash -c "apt-get update && apt-get install -y poppler-utils tesseract-ocr openssh-server && mkdir -p ~/.ssh && chmod 700 ~/.ssh && echo \"$PUBLIC_KEY\" >> ~/.ssh/authorized_keys && chmod 600 ~/.ssh/authorized_keys && service ssh start && pip install torch==2.6.0 torchvision==0.21.0 torchaudio==2.5.0 --index-url https://download.pytorch.org/whl/cu118 && pip install pdf2image pillow boto3 fastapi uvicorn[standard] transformers accelerate python-multipart aiofiles flash-attn --no-build-isolation && sleep infinity"
   ```
   > **Updates included**: 
   > 1. Installs SSH Server (fixes connection refused)
   > 2. Upgrades PyTorch to **2.6.0** (Required for DeepSeek)
   > 3. Installs `flash-attn` (Required for performance)

## Step 2: Deploy the Pod

1. Go to [RunPod Console](https://www.runpod.io/console/pods) -> **Deploy**
2. Select:
   - **GPU**: RTX 4090 (24GB) or A100 (80GB) recommended
   - **Template**: Select your `DeepSeek-OCR-Service` template
3. Click **Deploy**

## Step 3: Get Connection Details

1. Wait for the Pod to show **Running**
2. Click **Connect**
3. Look for **TCP Port Mapping**:
   - Find the IP (e.g., `213.173...`)
   - Find the Port mapping to `22` (e.g., `47566`)

## Step 4: Upload Files

You have three options (choose one):

### Option A: Use Self-Healing Script (Recommended)
Run this script and enter the NEW IP/Port when asked:
```powershell
.\deploy_to_runpod.ps1
```

### Option B: Use runpodctl (No SSH needed)
```powershell
.\upload_with_runpodctl.ps1
```

### Option C: Manual SCP
```powershell
# Replace with your NEW IP and Port
scp -P <PORT> -i ~/.ssh/runpod_key config.py pdf_processor.py ocr_model.py s3_service.py main.py requirements.txt start_server.sh root@<IP>:/workspace/
```

## Step 5: Start the Service

1. SSH into the pod:
   ```bash
   ssh -p <PORT> -i ~/.ssh/runpod_key root@<IP>
   ```
2. Run the server:
   ```bash
   cd /workspace
   # Run directly
   uvicorn main:app --host 0.0.0.0 --port 8000
   ```

## Step 6: Verify Secrets are Loaded

After the pod starts, verify your secrets are available:

```bash
# Check if env vars are set (shows first 10 chars only for security)
echo "HF_TOKEN: $(echo $HF_TOKEN | head -c 10)..."
echo "AWS_ACCESS_KEY_ID: $(echo $AWS_ACCESS_KEY_ID | head -c 10)..."
echo "AWS_SECRET_ACCESS_KEY: $(echo $AWS_SECRET_ACCESS_KEY | head -c 10)..."
```

## Step 7: Install Dependencies

```bash
cd /workspace/ocr_service

# Install system dependencies
apt-get update && apt-get install -y poppler-utils

# Install Python dependencies
pip install -r requirements.txt

# Install flash-attn (required for DeepSeek-OCR-2)
pip install flash-attn==2.7.3 --no-build-isolation
```

## Step 8: Start the Server

```bash
cd /workspace/ocr_service
chmod +x start_server.sh
./start_server.sh
```

Or run directly:
```bash
uvicorn main:app --host 0.0.0.0 --port 8000
```

## Step 9: Run in Background (Optional)

```bash
nohup uvicorn main:app --host 0.0.0.0 --port 8000 > /workspace/server.log 2>&1 &
```

## Step 10: Test the Service

The service will be available at:
```
https://ijb7en31qwkhvp-64411c51-8000.proxy.runpod.net
```

Test endpoints:
```bash
# Health check
curl https://ijb7en31qwkhvp-64411c51-8000.proxy.runpod.net/health

# Process a PDF
curl -X POST https://ijb7en31qwkhvp-64411c51-8000.proxy.runpod.net/ocr/process \
  -H "Content-Type: application/json" \
  -d '{"url": "http://www.accessdata.fda.gov/drugsatfda_docs/label/2020/125504s031lbl.pdf", "document_id": "125504s031", "document_type": "Label"}'
```

## Troubleshooting

### Secrets Not Available
If env vars are empty, check your RunPod template configuration:
1. Go to RunPod Console → Templates
2. Edit your template
3. Add the environment variable mappings from Step 1
4. Restart your pod

### Model Loading Issues
```bash
# Check GPU availability
nvidia-smi

# Check CUDA version
nvcc --version

# Test model loading
python -c "from ocr_model import get_ocr_model; get_ocr_model().load_model()"
```

### Memory Issues
- DeepSeek-OCR-2 requires ~12GB VRAM
- For large PDFs, reduce DPI: Add `DPI=150` to template env vars
- Reduce MAX_PAGES if needed

### S3 Upload Issues
```bash
# Test AWS credentials
python -c "import boto3; print(boto3.client('s3').list_buckets())"
```

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | Service info |
| `/health` | GET | Health check (model loaded status) |
| `/ocr/process` | POST | Process single PDF |
| `/ocr/batch` | POST | Process multiple PDFs |

## Integration with Drug History API

The drug history backend calls this service via:
```
POST /ocr/process
{
  "url": "http://www.accessdata.fda.gov/drugsatfda_docs/label/2020/125504s031lbl.pdf",
  "document_id": "125504s031",
  "document_type": "Label"
}
```

Response:
```json
{
  "success": true,
  "document_id": "125504s031",
  "s3_url": "https://pharma-labels-ocr.s3.us-east-1.amazonaws.com/fda_documents/Label_125504s031_20260129_123456.md",
  "page_count": 45,
  "message": "Successfully processed 45 pages"
}
```
