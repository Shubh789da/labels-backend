# DeepSeek-OCR FastAPI Endpoint - H100/H200 SXM Optimized
# Accepts PDF URLs, processes them, and returns OCR results
# Optimized for NVIDIA H100/H200 SXM (80GB+ VRAM, Hopper architecture)

import os
import sys
import re
import tempfile
import logging
import requests
import fitz  # PyMuPDF
import asyncio
from concurrent.futures import ThreadPoolExecutor
from io import StringIO, BytesIO
from pathlib import Path
from typing import Optional, List
from PIL import Image

# Set CUDA debugging environment variables BEFORE importing torch
os.environ["CUDA_LAUNCH_BLOCKING"] = "0"  # Set to "1" for debugging
os.environ["TORCH_USE_CUDA_DSA"] = "0"    # Device-side assertions

import torch
import warnings
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, HttpUrl
import uvicorn

# Suppress transformer warnings for cleaner output
warnings.filterwarnings("ignore", message=".*do_sample.*")
warnings.filterwarnings("ignore", message=".*attention_mask.*")
warnings.filterwarnings("ignore", message=".*pad_token_id.*")
warnings.filterwarnings("ignore", message=".*seen_tokens.*")
warnings.filterwarnings("ignore", message=".*get_max_cache.*")
warnings.filterwarnings("ignore", message=".*position_ids.*")
warnings.filterwarnings("ignore", message=".*position_embeddings.*")
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

# Bounding box constants
BOUNDING_BOX_PATTERN = re.compile(r"<\|det\|>\[\[(\d+),\s*(\d+),\s*(\d+),\s*(\d+)\]\]<\|/det\|>")

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    stream=sys.stderr,
    force=True
)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="DeepSeek-OCR API (H100/H200 Optimized)",
    description="OCR API optimized for NVIDIA H100/H200 SXM GPU with CUDA error recovery",
    version="2.1.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global model variables
model = None
tokenizer = None

# Reusable temp directory
_temp_dir = None

# Thread pool for parallel processing
_executor = None


def get_temp_dir():
    """Get or create a reusable temp directory"""
    global _temp_dir
    if _temp_dir is None or not os.path.exists(_temp_dir):
        _temp_dir = tempfile.mkdtemp(prefix="ocr_h100_")
    return _temp_dir


def get_executor():
    """Get thread pool executor for parallel processing"""
    global _executor
    if _executor is None:
        # H100 can handle more concurrent operations
        _executor = ThreadPoolExecutor(max_workers=4)
    return _executor


class PDFRequest(BaseModel):
    pdf_url: HttpUrl
    model_size: str = "Gundam"  # Tiny/Small=fast, Base/Large=accurate, Gundam=balanced
    prompt: str = "<|grounding|>Convert the document to markdown."
    pages: Optional[List[int]] = None  # None or empty list means ALL pages
    process_all: bool = True  # Set to True to process all pages (default)
    dpi_scale: float = 1.5  # Slightly reduced for better CUDA compatibility
    max_image_size: int = 2048  # Reduced to prevent masked_scatter CUDA errors


class ImageRequest(BaseModel):
    image_url: HttpUrl
    model_size: str = "Gundam"
    prompt: str = "<|grounding|>Convert the document to markdown."
    max_image_size: int = 2048  # Reduced for better CUDA compatibility


class OCRResponse(BaseModel):
    success: bool
    pages: List[dict]
    total_pages: int
    message: Optional[str] = None


class SingleOCRResponse(BaseModel):
    success: bool
    raw_text: str
    parsed_text: str
    message: Optional[str] = None


def parse_ocr_output(raw_output: str) -> str:
    """Parse raw OCR output to remove debug info and format cleanly"""
    lines = raw_output.split('\n')
    parsed_lines = []

    skip_patterns = [
        'BASE:', 'PATCHES:', 'NO PATCHES', 'directly resize',
        'image size:', 'valid image tokens:', 'output texts tokens',
        'compression ratio:', 'save results:', '====', '===',
    ]

    for line in lines:
        stripped = line.strip()

        if not stripped or any(pattern in line for pattern in skip_patterns):
            continue

        if '<|ref|>' in line:
            pattern = r'<\|ref\|>(.*?)<\|/ref\|>(?:<\|det\|>\[\[(.*?)\]\]<\|/det\|>)?'
            matches = re.findall(pattern, line)

            if matches:
                for ref_text, coords in matches:
                    if coords:
                        parsed_lines.append(f"**{ref_text}** -> [{coords}]")
                    else:
                        parsed_lines.append(ref_text.strip())
            continue

        parsed_lines.append(stripped)

    result = '\n'.join(parsed_lines)
    return result if result.strip() else raw_output


def get_model_config(model_size: str) -> dict:
    """Get model configuration based on size - Matching reference implementation for stability"""
    configs = {
        "Tiny": {"base_size": 512, "image_size": 512, "crop_mode": False},
        "Small": {"base_size": 640, "image_size": 640, "crop_mode": False},
        "Base": {"base_size": 1024, "image_size": 1024, "crop_mode": False},
        "Large": {"base_size": 1280, "image_size": 1280, "crop_mode": False},
        "Gundam": {"base_size": 1024, "image_size": 640, "crop_mode": True}
    }
    return configs.get(model_size, configs["Gundam"])


def download_file(url: str) -> bytes:
    """Download file from URL with optimized settings"""
    logger.info(f"Downloading file from: {url}")
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
    }
    # Increase timeout and chunk size for faster downloads
    response = requests.get(str(url), headers=headers, timeout=180, stream=True)
    response.raise_for_status()
    return response.content


def pdf_to_images(pdf_bytes: bytes, dpi_scale: float = 2.0) -> List[Image.Image]:
    """Convert PDF to list of PIL Images - H100 optimized with higher default DPI"""
    images = []
    pdf_document = fitz.open(stream=pdf_bytes, filetype="pdf")

    for page_num in range(len(pdf_document)):
        page = pdf_document[page_num]
        mat = fitz.Matrix(dpi_scale, dpi_scale)
        pix = page.get_pixmap(matrix=mat)
        img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
        images.append(img)

    pdf_document.close()
    return images


def resize_if_needed(image: Image.Image, max_size: int = 3072) -> Image.Image:
    """Resize image if larger than max_size - H100 allows larger images"""
    w, h = image.size
    if max(w, h) > max_size:
        ratio = max_size / max(w, h)
        new_size = (int(w * ratio), int(h * ratio))
        return image.resize(new_size, Image.Resampling.LANCZOS)
    return image





def process_single_image(image: Image.Image, prompt: str, model_size: str, max_image_size: int = 2048) -> dict:
    """Process a single image with the OCR model - H100/H200 optimized with error recovery"""
    global model, tokenizer

    config_model = get_model_config(model_size)
    
    image = resize_if_needed(image, max_image_size)
    
    logger.info(f"Image size after preprocessing: {image.size}")

    if "<image>" not in prompt:
        full_prompt = f"<image>\n{prompt}"
    else:
        full_prompt = prompt

    temp_dir = get_temp_dir()
    temp_image_path = os.path.join(temp_dir, f"input_{id(image)}.jpg")
    image.save(temp_image_path, "JPEG", quality=90)

    # Retry logic for CUDA errors
    max_retries = 2
    last_error = None

    for attempt in range(max_retries + 1):
        captured_output = StringIO()
        old_stdout = sys.stdout

        try:
            # Clear CUDA cache before inference to prevent fragmentation
            torch.cuda.empty_cache()
            torch.cuda.synchronize()

            sys.stdout = captured_output
            # Use torch.inference_mode() for faster inference (disables gradient computation)
            with torch.inference_mode():
                result = model.infer(
                    tokenizer,
                    prompt=full_prompt,
                    image_file=temp_image_path,
                    output_path=temp_dir,
                    base_size=config_model["base_size"],
                    image_size=config_model["image_size"],
                    crop_mode=config_model["crop_mode"],
                    save_results=False,
                    test_compress=False
                )

            # Success - break out of retry loop
            sys.stdout = old_stdout
            break

        except RuntimeError as e:
            sys.stdout = old_stdout
            last_error = e
            error_str = str(e).lower()

            # Check if it's a CUDA error that might be recoverable
            if "cuda" in error_str or "device-side assert" in error_str:
                logger.warning(f"CUDA error on attempt {attempt + 1}, clearing cache and retrying...")

                # Reset CUDA state
                torch.cuda.empty_cache()
                torch.cuda.synchronize()

                if attempt < max_retries:
                    # Try with smaller image on retry
                    if attempt == 1:
                        logger.info("Retrying with smaller image...")
                        image = resize_if_needed(image, max_image_size // 2)
                        image.save(temp_image_path, "JPEG", quality=85)
                    continue

            logger.error(f"Model inference error: {e}")
            raise

        except Exception as e:
            sys.stdout = old_stdout
            logger.error(f"Model inference error: {e}")
            raise

        finally:
            if sys.stdout != old_stdout:
                sys.stdout = old_stdout
    else:
        # All retries failed
        if last_error:
            raise last_error

    # Clean up temp file
    try:
        os.remove(temp_image_path)
    except:
        pass

    console_output = captured_output.getvalue()
    text_result = console_output if console_output else str(result)
    parsed_result = parse_ocr_output(text_result)

    return {
        "raw_text": text_result,
        "parsed_text": parsed_result
    }


@app.on_event("startup")
async def load_model():
    """Load model on startup with H100/H200 optimizations"""
    global model, tokenizer

    import torch
    from transformers import AutoModel, AutoTokenizer

    logger.info("=" * 60)
    logger.info("Loading DeepSeek-OCR model (H100/H200 SXM Optimized)")
    logger.info("=" * 60)

    # Detect GPU
    gpu_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "Unknown"
    logger.info(f"Detected GPU: {gpu_name}")

    # H100/H200 Hopper Architecture Optimizations
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    # Enable flash attention optimizations (safe for H100/H200)
    try:
        torch.backends.cuda.enable_flash_sdp(True)
        torch.backends.cuda.enable_mem_efficient_sdp(True)
        torch.backends.cuda.enable_math_sdp(True)
        logger.info("Flash SDP backends enabled")
    except Exception as e:
        logger.warning(f"Could not enable all SDP backends: {e}")

    # Set CUDA memory allocation config for large VRAM (80GB+)
    # Use max_split_size to prevent fragmentation on H200
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True,max_split_size_mb:512"
    os.environ["CUDA_VISIBLE_DEVICES"] = '0'

    model_name = 'deepseek-ai/DeepSeek-OCR'

    logger.info("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

    # Use simple, direct loading like app.py but with H200 optimizations
    # Valid Optimizations for H200:
    # 1. flash_attention_2 (Critical)
    # 2. bfloat16 (Native support)
    # 3. device_map="auto" (Standard)
    
    logger.info("Loading model with bfloat16 + flash_attention_2...")
    model = AutoModel.from_pretrained(
        model_name,
        _attn_implementation='flash_attention_2',
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
        use_safetensors=True
    )
    
    # Ensure model is in eval mode
    model = model.eval()

    # Log GPU info
    gpu_name = torch.cuda.get_device_name(0)
    gpu_memory = torch.cuda.get_device_properties(0).total_memory / (1024**3)
    logger.info(f"GPU: {gpu_name}")
    logger.info(f"GPU Memory: {gpu_memory:.1f} GB")
    logger.info("Model loaded successfully!")

    logger.info("=" * 60)
    logger.info("H100/H200 Optimizations Active:")
    logger.info("  - Dynamic attention implementation")
    logger.info("  - TF32 Matrix Multiplication")
    logger.info("  - Optimized dtype for H200")
    logger.info("  - CUDA Memory Optimization")
    logger.info("  - Higher Resolution Support (2048px)")
    logger.info("  - Auto CUDA cache clearing every 5 pages")
    logger.info("=" * 60)


@app.get("/")
async def root():
    """Health check endpoint"""
    return {
        "status": "ok",
        "message": "DeepSeek-OCR API (H100 Optimized) is running",
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "N/A"
    }


@app.get("/health")
async def health():
    """Health check endpoint with GPU stats"""
    global model

    gpu_stats = {}
    if torch.cuda.is_available():
        gpu_stats = {
            "gpu_name": torch.cuda.get_device_name(0),
            "gpu_memory_total_gb": round(torch.cuda.get_device_properties(0).total_memory / (1024**3), 2),
            "gpu_memory_allocated_gb": round(torch.cuda.memory_allocated(0) / (1024**3), 2),
            "gpu_memory_cached_gb": round(torch.cuda.memory_reserved(0) / (1024**3), 2),
        }

    return {
        "status": "healthy",
        "model_loaded": model is not None,
        "gpu": gpu_stats
    }


@app.post("/ocr/pdf", response_model=OCRResponse)
async def process_pdf(request: PDFRequest):
    """
    Process a PDF from URL and return OCR results for each page.
    H100 optimized with higher default resolution and larger image support.
    """
    global model

    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded yet")

    try:
        pdf_bytes = download_file(str(request.pdf_url))
        logger.info(f"Downloaded PDF: {len(pdf_bytes)} bytes")

        images = pdf_to_images(pdf_bytes, dpi_scale=request.dpi_scale)
        total_pages = len(images)
        logger.info(f"PDF has {total_pages} pages (rendered at {request.dpi_scale}x DPI)")

        if request.process_all or request.pages is None or len(request.pages) == 0:
            pages_to_process = list(range(total_pages))
        else:
            pages_to_process = [p for p in request.pages if 0 <= p < total_pages]

        logger.info(f"Will process {len(pages_to_process)} pages: {pages_to_process}")

        results = []
        for idx, page_num in enumerate(pages_to_process):
            logger.info(f"Processing page {page_num + 1}/{total_pages} (iteration {idx + 1}/{len(pages_to_process)})")
            sys.stderr.flush()

            image = images[page_num]

            try:
                # Clear CUDA cache between pages to prevent memory fragmentation
                if idx > 0 and idx % 5 == 0:
                    torch.cuda.empty_cache()
                    logger.info("Cleared CUDA cache")

                page_result = process_single_image(image, request.prompt, request.model_size, request.max_image_size)
                results.append({
                    "page_number": page_num,
                    "raw_text": page_result["raw_text"],
                    "parsed_text": page_result["parsed_text"]
                })
                logger.info(f"Completed page {page_num + 1}/{total_pages}, parsed text length: {len(page_result['parsed_text'])}")

            except RuntimeError as cuda_error:
                error_str = str(cuda_error).lower()
                if "cuda" in error_str or "device-side assert" in error_str:
                    logger.error(f"CUDA error on page {page_num + 1}: {cuda_error}")
                    # Try to recover CUDA state
                    torch.cuda.empty_cache()
                    torch.cuda.synchronize()

                    results.append({
                        "page_number": page_num,
                        "raw_text": f"Error: CUDA error - page skipped",
                        "parsed_text": f"Error: CUDA error on this page. Try processing separately with smaller dpi_scale."
                    })
                else:
                    raise

            except Exception as page_error:
                logger.error(f"Error processing page {page_num + 1}: {page_error}")
                results.append({
                    "page_number": page_num,
                    "raw_text": f"Error: {str(page_error)}",
                    "parsed_text": f"Error: {str(page_error)}"
                })

        logger.info(f"Finished processing all pages. Total results: {len(results)}")

        return OCRResponse(
            success=True,
            pages=results,
            total_pages=total_pages,
            message=f"Successfully processed {len(results)} pages (H100 optimized)"
        )

    except requests.exceptions.RequestException as e:
        logger.error(f"Failed to download PDF: {e}")
        raise HTTPException(status_code=400, detail=f"Failed to download PDF: {str(e)}")
    except Exception as e:
        logger.error(f"Error processing PDF: {e}")
        raise HTTPException(status_code=500, detail=f"Error processing PDF: {str(e)}")


@app.post("/ocr/image", response_model=SingleOCRResponse)
async def process_image_url(request: ImageRequest):
    """
    Process an image from URL and return OCR results.
    H100 optimized with larger image support.
    """
    global model

    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded yet")

    try:
        image_bytes = download_file(str(request.image_url))
        image = Image.open(BytesIO(image_bytes))

        if image.mode != 'RGB':
            image = image.convert('RGB')

        logger.info(f"Processing image: {image.size}")

        result = process_single_image(image, request.prompt, request.model_size, request.max_image_size)

        return SingleOCRResponse(
            success=True,
            raw_text=result["raw_text"],
            parsed_text=result["parsed_text"],
            message="Successfully processed image (H100 optimized)"
        )

    except requests.exceptions.RequestException as e:
        logger.error(f"Failed to download image: {e}")
        raise HTTPException(status_code=400, detail=f"Failed to download image: {str(e)}")
    except Exception as e:
        logger.error(f"Error processing image: {e}")
        raise HTTPException(status_code=500, detail=f"Error processing image: {str(e)}")


if __name__ == "__main__":
    uvicorn.run(
        "app_h100:app",
        host="0.0.0.0",
        port=8000,
        reload=False,
        workers=1  # Single worker to maximize GPU utilization
    )
