# DeepSeek-OCR RunPod Serverless Handler
# Accepts PDF/Image URLs, processes them, and returns OCR results
# Optimized for serverless deployment with cold start minimization

import os
import sys
import re
import tempfile
import requests
import fitz  # PyMuPDF
from io import StringIO, BytesIO
from typing import Optional, List
from PIL import Image

import torch
import warnings
import runpod

# Suppress all warnings for cleaner logs and faster startup
warnings.filterwarnings("ignore")

# Global model variables (loaded once, reused across requests)
model = None
tokenizer = None
_temp_dir = None


def get_temp_dir():
    """Get or create a reusable temp directory"""
    global _temp_dir
    if _temp_dir is None or not os.path.exists(_temp_dir):
        _temp_dir = tempfile.mkdtemp(prefix="ocr_serverless_")
    return _temp_dir


def load_model():
    """Load model once at container startup - critical for cold start optimization"""
    global model, tokenizer

    if model is not None:
        return  # Already loaded

    from transformers import AutoModel, AutoTokenizer

    print("Loading DeepSeek-OCR model...", file=sys.stderr)

    # CUDA optimizations
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    # H100/A100 specific optimizations
    if torch.cuda.is_available():
        torch.backends.cuda.enable_flash_sdp(True)
        torch.backends.cuda.enable_mem_efficient_sdp(True)

    model_name = 'deepseek-ai/DeepSeek-OCR'

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModel.from_pretrained(
        model_name,
        _attn_implementation='flash_attention_2',
        torch_dtype=torch.bfloat16,
        device_map="cuda:0",
        trust_remote_code=True,
        use_safetensors=True
    ).eval()

    print(f"Model loaded on {torch.cuda.get_device_name(0)}", file=sys.stderr)

    # Warmup inference
    print("Running warmup inference...", file=sys.stderr)
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
        torch.cuda.synchronize()
        print("Warmup complete - ready for inference!", file=sys.stderr)
    except Exception as e:
        print(f"Warmup failed (non-critical): {e}", file=sys.stderr)


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
    """Get model configuration based on size"""
    configs = {
        "Tiny": {"base_size": 640, "image_size": 640, "crop_mode": False},
        "Small": {"base_size": 768, "image_size": 768, "crop_mode": False},
        "Base": {"base_size": 1280, "image_size": 1280, "crop_mode": False},
        "Large": {"base_size": 1536, "image_size": 1536, "crop_mode": False},
        "Gundam": {"base_size": 1280, "image_size": 768, "crop_mode": True}
    }
    return configs.get(model_size, configs["Gundam"])


def download_file(url: str) -> bytes:
    """Download file from URL"""
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
    }
    response = requests.get(str(url), headers=headers, timeout=180)
    response.raise_for_status()
    return response.content


def pdf_to_images(pdf_bytes: bytes, dpi_scale: float = 2.0) -> List[Image.Image]:
    """Convert PDF to list of PIL Images"""
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
    """Resize image if larger than max_size"""
    w, h = image.size
    if max(w, h) > max_size:
        ratio = max_size / max(w, h)
        new_size = (int(w * ratio), int(h * ratio))
        return image.resize(new_size, Image.Resampling.LANCZOS)
    return image


def process_single_image(image: Image.Image, prompt: str, model_size: str, max_image_size: int = 3072) -> dict:
    """Process a single image with the OCR model"""
    global model, tokenizer

    config_model = get_model_config(model_size)
    image = resize_if_needed(image, max_image_size)

    if "<image>" not in prompt:
        full_prompt = f"<image>\n{prompt}"
    else:
        full_prompt = prompt

    temp_dir = get_temp_dir()
    temp_image_path = os.path.join(temp_dir, f"input_{id(image)}.jpg")
    image.save(temp_image_path, "JPEG", quality=90)

    captured_output = StringIO()
    old_stdout = sys.stdout

    try:
        sys.stdout = captured_output
        with torch.inference_mode():
            torch.cuda.synchronize()
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
            torch.cuda.synchronize()
    finally:
        sys.stdout = old_stdout
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


def handler(job):
    """
    RunPod Serverless Handler

    Input format:
    {
        "input": {
            "pdf_url": "https://...",          # OR
            "image_url": "https://...",
            "prompt": "<|grounding|>Convert the document to markdown.",
            "model_size": "Gundam",
            "pages": [0, 1, 2],                # Optional: specific pages (PDF only)
            "process_all": true,               # Process all pages (default)
            "dpi_scale": 2.0,                  # PDF render quality
            "max_image_size": 3072             # Max image dimension
        }
    }

    Output format:
    {
        "success": true,
        "pages": [...],
        "total_pages": N
    }
    """
    global model

    # Ensure model is loaded
    load_model()

    job_input = job.get('input', {})

    # Extract parameters
    pdf_url = job_input.get('pdf_url')
    image_url = job_input.get('image_url')
    prompt = job_input.get('prompt', '<|grounding|>Convert the document to markdown.')
    model_size = job_input.get('model_size', 'Gundam')
    pages = job_input.get('pages')
    process_all = job_input.get('process_all', True)
    dpi_scale = job_input.get('dpi_scale', 2.0)
    max_image_size = job_input.get('max_image_size', 3072)

    try:
        # Process PDF
        if pdf_url:
            print(f"Processing PDF: {pdf_url}", file=sys.stderr)

            pdf_bytes = download_file(pdf_url)
            print(f"Downloaded PDF: {len(pdf_bytes)} bytes", file=sys.stderr)

            images = pdf_to_images(pdf_bytes, dpi_scale=dpi_scale)
            total_pages = len(images)
            print(f"PDF has {total_pages} pages", file=sys.stderr)

            # Determine pages to process
            if process_all or pages is None or len(pages) == 0:
                pages_to_process = list(range(total_pages))
            else:
                pages_to_process = [p for p in pages if 0 <= p < total_pages]

            print(f"Processing {len(pages_to_process)} pages", file=sys.stderr)

            results = []
            for idx, page_num in enumerate(pages_to_process):
                print(f"Processing page {page_num + 1}/{total_pages}", file=sys.stderr)

                image = images[page_num]

                try:
                    page_result = process_single_image(image, prompt, model_size, max_image_size)
                    results.append({
                        "page_number": page_num,
                        "raw_text": page_result["raw_text"],
                        "parsed_text": page_result["parsed_text"]
                    })
                except Exception as page_error:
                    print(f"Error on page {page_num + 1}: {page_error}", file=sys.stderr)
                    results.append({
                        "page_number": page_num,
                        "raw_text": f"Error: {str(page_error)}",
                        "parsed_text": f"Error: {str(page_error)}"
                    })

            return {
                "success": True,
                "pages": results,
                "total_pages": total_pages,
                "message": f"Processed {len(results)} pages"
            }

        # Process single image
        elif image_url:
            print(f"Processing image: {image_url}", file=sys.stderr)

            image_bytes = download_file(image_url)
            image = Image.open(BytesIO(image_bytes))

            if image.mode != 'RGB':
                image = image.convert('RGB')

            print(f"Image size: {image.size}", file=sys.stderr)

            result = process_single_image(image, prompt, model_size, max_image_size)

            return {
                "success": True,
                "raw_text": result["raw_text"],
                "parsed_text": result["parsed_text"],
                "message": "Processed image"
            }

        else:
            return {"error": "No pdf_url or image_url provided"}

    except requests.exceptions.RequestException as e:
        return {"error": f"Failed to download file: {str(e)}"}
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        return {"error": f"Processing error: {str(e)}"}


# Load model at container startup (not per-request)
print("=" * 60, file=sys.stderr)
print("DeepSeek-OCR RunPod Serverless Worker", file=sys.stderr)
print("=" * 60, file=sys.stderr)
load_model()

# Start the serverless worker
runpod.serverless.start({"handler": handler})
