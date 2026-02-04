# DeepSeek-OCR FastAPI Endpoint
# Accepts PDF URLs, processes them, and returns OCR results

import os
import sys
import re
import tempfile
import logging
import requests
import fitz  # PyMuPDF
from io import StringIO, BytesIO
from pathlib import Path
from typing import Optional, List
from PIL import Image

import torch
import warnings
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, HttpUrl
import uvicorn

# Suppress transformer warnings for cleaner output and minor speed improvement
# warnings.filterwarnings("ignore", message=".*do_sample.*")
# warnings.filterwarnings("ignore", message=".*attention_mask.*")
# warnings.filterwarnings("ignore", message=".*pad_token_id.*")
# warnings.filterwarnings("ignore", message=".*seen_tokens.*")
# warnings.filterwarnings("ignore", message=".*get_max_cache.*")
# warnings.filterwarnings("ignore", message=".*position_ids.*")
# warnings.filterwarnings("ignore", message=".*position_embeddings.*")

# Bounding box constants
BOUNDING_BOX_PATTERN = re.compile(r"<\|det\|>\[\[(\d+),\s*(\d+),\s*(\d+),\s*(\d+)\]\]<\|/det\|>")

# Setup logging - explicitly use stderr to avoid stdout capture issues
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    stream=sys.stderr,
    force=True
)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="DeepSeek-OCR API",
    description="OCR API for processing PDFs and images using DeepSeek-OCR model",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global model variables (loaded on startup)
model = None
tokenizer = None

# Reusable temp directory to avoid creation overhead
_temp_dir = None


def get_temp_dir():
    """Get or create a reusable temp directory"""
    global _temp_dir
    if _temp_dir is None or not os.path.exists(_temp_dir):
        _temp_dir = tempfile.mkdtemp(prefix="ocr_")
    return _temp_dir


class PDFRequest(BaseModel):
    pdf_url: HttpUrl
    model_size: str = "Gundam"  # Tiny/Small=fast, Base/Large=accurate, Gundam=balanced
    prompt: str = "<|grounding|>Convert the document to markdown."
    pages: Optional[List[int]] = None  # None or empty list means ALL pages
    process_all: bool = True  # Set to True to process all pages (default)
    dpi_scale: float = 1.5  # PDF render quality: 1.0=fast, 1.5=balanced, 2.0=high quality
    max_image_size: int = 2048  # Max image dimension in pixels (smaller=faster)


class ImageRequest(BaseModel):
    image_url: HttpUrl
    model_size: str = "Gundam"  # Tiny/Small=fast, Base/Large=accurate, Gundam=balanced
    prompt: str = "<|grounding|>Convert the document to markdown."
    max_image_size: int = 2048  # Max image dimension in pixels (smaller=faster)


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

    # Patterns to skip (debug/metadata)
    skip_patterns = [
        'BASE:', 'PATCHES:', 'NO PATCHES', 'directly resize',
        'image size:', 'valid image tokens:', 'output texts tokens',
        'compression ratio:', 'save results:', '====', '===',
    ]

    for line in lines:
        stripped = line.strip()

        # Skip empty lines and debug patterns
        if not stripped or any(pattern in line for pattern in skip_patterns):
            continue

        # Handle ref/det structured data
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

        # Regular content - add as is
        parsed_lines.append(stripped)

    result = '\n'.join(parsed_lines)
    return result if result.strip() else raw_output


def get_model_config(model_size: str) -> dict:
    """Get model configuration based on size"""
    configs = {
        "Tiny": {"base_size": 512, "image_size": 512, "crop_mode": False},
        "Small": {"base_size": 640, "image_size": 640, "crop_mode": False},
        "Base": {"base_size": 1024, "image_size": 1024, "crop_mode": False},
        "Large": {"base_size": 1280, "image_size": 1280, "crop_mode": False},
        "Gundam": {"base_size": 1024, "image_size": 640, "crop_mode": True}
    }
    return configs.get(model_size, configs["Gundam"])


def download_file(url: str) -> bytes:
    """Download file from URL"""
    logger.info(f"Downloading file from: {url}")
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
    }
    response = requests.get(str(url), headers=headers, timeout=120)
    response.raise_for_status()
    return response.content


def pdf_to_images(pdf_bytes: bytes, dpi_scale: float = 1.5) -> List[Image.Image]:
    """Convert PDF to list of PIL Images

    Args:
        pdf_bytes: PDF file bytes
        dpi_scale: Resolution multiplier (1.0=72dpi, 1.5=108dpi, 2.0=144dpi)
                   Lower = faster but less accurate, Higher = slower but more accurate
    """
    images = []
    pdf_document = fitz.open(stream=pdf_bytes, filetype="pdf")

    for page_num in range(len(pdf_document)):
        page = pdf_document[page_num]
        mat = fitz.Matrix(dpi_scale, dpi_scale)
        pix = page.get_pixmap(matrix=mat)

        # Convert to PIL Image
        img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
        images.append(img)

    pdf_document.close()
    return images


def resize_if_needed(image: Image.Image, max_size: int = 2048) -> Image.Image:
    """Resize image if larger than max_size while maintaining aspect ratio"""
    w, h = image.size
    if max(w, h) > max_size:
        ratio = max_size / max(w, h)
        new_size = (int(w * ratio), int(h * ratio))
        return image.resize(new_size, Image.Resampling.LANCZOS)
    return image


def process_single_image(image: Image.Image, prompt: str, model_size: str, max_image_size: int = 2048) -> dict:
    """Process a single image with the OCR model"""
    global model, tokenizer

    config_model = get_model_config(model_size)

    # Resize large images for faster processing
    image = resize_if_needed(image, max_image_size)

    # Build prompt
    if "<image>" not in prompt:
        full_prompt = f"<image>\n{prompt}"
    else:
        full_prompt = prompt

    # Use reusable temp directory to avoid creation overhead
    temp_dir = get_temp_dir()
    temp_image_path = os.path.join(temp_dir, "input_image.jpg")
    # Use lower JPEG quality for speed (85 is still good quality)
    image.save(temp_image_path, "JPEG", quality=85)

    # Capture stdout to get model output
    captured_output = StringIO()
    old_stdout = sys.stdout

    try:
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
                test_compress=False  # Disable for speed
            )
    except Exception as e:
        logger.error(f"Model inference error: {e}")
        raise
    finally:
        # Always restore stdout
        sys.stdout = old_stdout

    console_output = captured_output.getvalue()
    text_result = console_output if console_output else str(result)
    parsed_result = parse_ocr_output(text_result)

    return {
        "raw_text": text_result,
        "parsed_text": parsed_result
    }


@app.on_event("startup")
async def load_model():
    """Load model on startup"""
    global model, tokenizer

    import torch
    from transformers import AutoModel, AutoTokenizer

    logger.info("Loading DeepSeek-OCR model...")

    # Enable CUDA optimizations for faster inference
    torch.backends.cudnn.benchmark = True  # Optimize for consistent input sizes
    torch.backends.cuda.matmul.allow_tf32 = True  # Use TF32 for faster matmul
    torch.backends.cudnn.allow_tf32 = True

    os.environ["CUDA_VISIBLE_DEVICES"] = '0'
    model_name = 'deepseek-ai/DeepSeek-OCR'

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModel.from_pretrained(
        model_name,
        _attn_implementation='flash_attention_2',
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
        use_safetensors=True
    )
    model = model.eval().cuda()

    logger.info("Model loaded successfully!")

    # Warmup inference to initialize CUDA kernels (makes subsequent inferences faster)
    logger.info("Running warmup inference...")
    try:
        temp_dir = get_temp_dir()
        warmup_img = Image.new('RGB', (256, 256), color='white')
        warmup_path = os.path.join(temp_dir, "warmup.jpg")
        warmup_img.save(warmup_path, "JPEG")

        with torch.inference_mode():
            _ = model.infer(
                tokenizer,
                prompt="<image>\nOCR",
                image_file=warmup_path,
                output_path=temp_dir,
                base_size=512,
                image_size=512,
                crop_mode=False,
                save_results=False,
                test_compress=False
            )
        logger.info("Warmup complete - model ready for fast inference!")
    except Exception as e:
        logger.warning(f"Warmup failed (non-critical): {e}")


@app.get("/")
async def root():
    """Health check endpoint"""
    return {"status": "ok", "message": "DeepSeek-OCR API is running"}


@app.get("/health")
async def health():
    """Health check endpoint"""
    global model
    return {
        "status": "healthy",
        "model_loaded": model is not None
    }


@app.post("/ocr/pdf", response_model=OCRResponse)
async def process_pdf(request: PDFRequest):
    """
    Process a PDF from URL and return OCR results for each page.

    - **pdf_url**: URL of the PDF to process
    - **model_size**: Model size (Tiny, Small, Base, Large, Gundam)
    - **prompt**: OCR prompt to use
    - **pages**: Optional list of page numbers (0-indexed) to process. If None, processes all pages.
    """
    global model

    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded yet")

    try:
        # Download PDF
        pdf_bytes = download_file(str(request.pdf_url))
        logger.info(f"Downloaded PDF: {len(pdf_bytes)} bytes")

        # Convert to images
        images = pdf_to_images(pdf_bytes, dpi_scale=request.dpi_scale)
        total_pages = len(images)
        logger.info(f"PDF has {total_pages} pages (rendered at {request.dpi_scale}x DPI)")

        # Determine which pages to process
        # Default: process ALL pages unless specific pages are requested AND process_all is False
        if request.process_all or request.pages is None or len(request.pages) == 0:
            pages_to_process = list(range(total_pages))
        else:
            pages_to_process = [p for p in request.pages if 0 <= p < total_pages]

        logger.info(f"Will process {len(pages_to_process)} pages: {pages_to_process}")

        # Process each page
        results = []
        for idx, page_num in enumerate(pages_to_process):
            logger.info(f"Processing page {page_num + 1}/{total_pages} (iteration {idx + 1}/{len(pages_to_process)})")
            sys.stderr.flush()  # Force flush to ensure log is visible

            image = images[page_num]

            try:
                page_result = process_single_image(image, request.prompt, request.model_size, request.max_image_size)
                results.append({
                    "page_number": page_num,
                    "raw_text": page_result["raw_text"],
                    "parsed_text": page_result["parsed_text"]
                })
                logger.info(f"Completed page {page_num + 1}/{total_pages}, parsed text length: {len(page_result['parsed_text'])}")
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
            message=f"Successfully processed {len(results)} pages"
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

    - **image_url**: URL of the image to process
    - **model_size**: Model size (Tiny, Small, Base, Large, Gundam)
    - **prompt**: OCR prompt to use
    """
    global model

    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded yet")

    try:
        # Download image
        image_bytes = download_file(str(request.image_url))
        image = Image.open(BytesIO(image_bytes))

        # Convert to RGB if needed
        if image.mode != 'RGB':
            image = image.convert('RGB')

        logger.info(f"Processing image: {image.size}")

        result = process_single_image(image, request.prompt, request.model_size, request.max_image_size)

        return SingleOCRResponse(
            success=True,
            raw_text=result["raw_text"],
            parsed_text=result["parsed_text"],
            message="Successfully processed image"
        )

    except requests.exceptions.RequestException as e:
        logger.error(f"Failed to download image: {e}")
        raise HTTPException(status_code=400, detail=f"Failed to download image: {str(e)}")
    except Exception as e:
        logger.error(f"Error processing image: {e}")
        raise HTTPException(status_code=500, detail=f"Error processing image: {str(e)}")


if __name__ == "__main__":
    uvicorn.run(
        "app:app",
        host="0.0.0.0",
        port=8000,
        reload=False,
        workers=1
    )
