#!/bin/bash
# Deploy all OCR service files to RunPod
# Usage: Copy this entire script and paste it into your RunPod SSH terminal

set -e
cd /workspace/ocr_service
mkdir -p /workspace/ocr_service

echo "Creating files..."

# config.py
cat > config.py << 'EOF_CONFIG'
"""Configuration for the OCR Service."""
import os
from pydantic_settings import BaseSettings
from functools import lru_cache


class Settings(BaseSettings):
    """OCR Service settings from environment variables.

    RunPod Template Configuration:
    In your RunPod pod template, set these environment variables:
        HF_TOKEN={{ RUNPOD_SECRET_HF_TOKEN }}
        AWS_ACCESS_KEY_ID={{ RUNPOD_SECRET_AWS_SECRET_ACCESS_ID }}
        AWS_SECRET_ACCESS_KEY={{ RUNPOD_SECRET_AWS_SECRET_ACCESS_KEY }}
        DEEPSEEK_API={{ RUNPOD_SECRET_DEEPSEEK_API }}
    """

    # AWS S3 Configuration (standard AWS env var names)
    AWS_ACCESS_KEY_ID: str = ""
    AWS_SECRET_ACCESS_KEY: str = ""
    AWS_DEFAULT_REGION: str = "us-east-1"
    S3_BUCKET_NAME: str = "pharma-labels-ocr"

    # Hugging Face token for model download
    HF_TOKEN: str = ""

    # DeepSeek API (optional)
    DEEPSEEK_API: str = ""

    # Model Configuration - DeepSeek-OCR-2
    MODEL_NAME: str = "deepseek-ai/DeepSeek-OCR-2"
    DEVICE: str = "cuda"
    BASE_SIZE: int = 1024  # Base size for image processing
    IMAGE_SIZE: int = 768  # Image size for inference
    CROP_MODE: bool = True  # Enable crop mode for better OCR

    # Processing
    TEMP_DIR: str = "/tmp/ocr_processing"
    MAX_PAGES: int = 100  # Maximum pages to process per PDF
    DPI: int = 200  # Resolution for PDF to image conversion

    # Server
    HOST: str = "0.0.0.0"
    PORT: int = 8000

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"


@lru_cache()
def get_settings() -> Settings:
    """Get cached settings instance."""
    return Settings()

EOF_CONFIG
echo "âœ“ config.py"

# pdf_processor.py
cat > pdf_processor.py << 'EOF_PDF'
"""PDF processing utilities for OCR service."""
import os
import tempfile
import logging
import httpx
from pathlib import Path
from typing import Optional
from pdf2image import convert_from_path
from PIL import Image

from config import get_settings

logger = logging.getLogger(__name__)
settings = get_settings()


class PDFProcessor:
    """Handles PDF download and conversion to images."""

    def __init__(self):
        self.temp_dir = Path(settings.TEMP_DIR)
        self.temp_dir.mkdir(parents=True, exist_ok=True)

    async def download_pdf(self, url: str) -> Optional[Path]:
        """Download PDF from URL.

        Args:
            url: URL to download PDF from

        Returns:
            Path to downloaded PDF file, or None if failed
        """
        try:
            async with httpx.AsyncClient(timeout=60.0, follow_redirects=True) as client:
                response = await client.get(url)
                response.raise_for_status()

                # Create temp file for PDF
                pdf_path = self.temp_dir / f"temp_{os.urandom(8).hex()}.pdf"

                with open(pdf_path, "wb") as f:
                    f.write(response.content)

                logger.info(f"Downloaded PDF to {pdf_path} ({len(response.content)} bytes)")
                return pdf_path

        except Exception as e:
            logger.error(f"Failed to download PDF from {url}: {e}")
            return None

    def convert_to_images(self, pdf_path: Path) -> list[Path]:
        """Convert PDF pages to images.

        Args:
            pdf_path: Path to PDF file

        Returns:
            List of paths to image files
        """
        try:
            # Convert PDF to images
            images = convert_from_path(
                pdf_path,
                dpi=settings.DPI,
                fmt="png",
                thread_count=4,
            )

            # Limit number of pages
            if len(images) > settings.MAX_PAGES:
                logger.warning(
                    f"PDF has {len(images)} pages, limiting to {settings.MAX_PAGES}"
                )
                images = images[: settings.MAX_PAGES]

            # Save images
            image_paths = []
            for i, image in enumerate(images):
                image_path = self.temp_dir / f"{pdf_path.stem}_page_{i + 1}.png"
                image.save(image_path, "PNG")
                image_paths.append(image_path)
                logger.debug(f"Saved page {i + 1} to {image_path}")

            logger.info(f"Converted {len(image_paths)} pages from PDF")
            return image_paths

        except Exception as e:
            logger.error(f"Failed to convert PDF to images: {e}")
            return []

    def cleanup(self, *paths: Path):
        """Remove temporary files.

        Args:
            paths: Paths to files to remove
        """
        for path in paths:
            try:
                if path and path.exists():
                    path.unlink()
                    logger.debug(f"Removed temporary file: {path}")
            except Exception as e:
                logger.warning(f"Failed to remove {path}: {e}")

    def cleanup_images(self, image_paths: list[Path]):
        """Remove all image files from a list."""
        for path in image_paths:
            self.cleanup(path)

EOF_PDF
echo "âœ“ pdf_processor.py"

# ocr_model.py
cat > ocr_model.py << 'EOF_OCR'
"""DeepSeek-OCR-2 model wrapper for document OCR."""
import os
import logging
import torch
from pathlib import Path
from typing import Optional
from transformers import AutoModel, AutoTokenizer

from config import get_settings

logger = logging.getLogger(__name__)
settings = get_settings()


class DeepSeekOCR:
    """Wrapper for DeepSeek-OCR-2 model.

    DeepSeek-OCR-2 uses a causal flow vision encoder (DeepEncoder V2)
    that processes documents more like human reading - flexibly adjusting
    reading order based on content semantics.

    Reference: https://huggingface.co/deepseek-ai/DeepSeek-OCR-2
    """

    def __init__(self):
        self.model = None
        self.tokenizer = None
        self.device = settings.DEVICE
        self._loaded = False

    def load_model(self):
        """Load the DeepSeek-OCR-2 model and tokenizer."""
        if self._loaded:
            logger.info("Model already loaded")
            return

        logger.info(f"Loading DeepSeek-OCR-2 model: {settings.MODEL_NAME}")

        try:
            # Set HuggingFace token if provided
            if settings.HF_TOKEN:
                os.environ["HF_TOKEN"] = settings.HF_TOKEN

            # Load tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(
                settings.MODEL_NAME,
                trust_remote_code=True,
            )

            # Load model with flash attention for efficiency
            self.model = AutoModel.from_pretrained(
                settings.MODEL_NAME,
                _attn_implementation="flash_attention_2",
                trust_remote_code=True,
                use_safetensors=True,
            )

            # Set to eval mode and move to GPU with bfloat16
            self.model = self.model.eval().cuda().to(torch.bfloat16)

            self._loaded = True
            logger.info("DeepSeek-OCR-2 model loaded successfully")

        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise

    def process_image(
        self,
        image_path: Path,
        output_dir: Path,
        convert_to_markdown: bool = True,
    ) -> str:
        """Process a single image with OCR.

        Args:
            image_path: Path to the image file
            output_dir: Directory to save intermediate results
            convert_to_markdown: If True, convert to markdown format

        Returns:
            Extracted text (markdown formatted if requested)
        """
        if not self._loaded:
            self.load_model()

        # Choose prompt based on output format
        if convert_to_markdown:
            prompt = "<image>\n<|grounding|>Convert the document to markdown."
        else:
            prompt = "<image>\nFree OCR."

        try:
            # Run inference using model's built-in infer method
            result = self.model.infer(
                self.tokenizer,
                prompt=prompt,
                image_file=str(image_path),
                output_path=str(output_dir),
                base_size=settings.BASE_SIZE,
                image_size=settings.IMAGE_SIZE,
                crop_mode=settings.CROP_MODE,
                save_results=False,  # We'll handle saving ourselves
            )

            # Extract text from result
            if isinstance(result, dict):
                text = result.get("text", "") or result.get("content", "") or str(result)
            elif isinstance(result, str):
                text = result
            else:
                text = str(result)

            logger.debug(f"Processed {image_path.name}: {len(text)} chars")
            return text

        except Exception as e:
            logger.error(f"Error processing image {image_path}: {e}")
            return f"[Error processing page: {e}]"

    def process_multiple_images(
        self,
        image_paths: list[Path],
        output_dir: Path,
    ) -> str:
        """Process multiple images and combine into single markdown.

        Args:
            image_paths: List of image file paths (ordered by page number)
            output_dir: Directory for intermediate results

        Returns:
            Combined markdown text from all pages
        """
        if not self._loaded:
            self.load_model()

        markdown_parts = []
        total_pages = len(image_paths)

        for i, image_path in enumerate(image_paths, 1):
            logger.info(f"Processing page {i}/{total_pages}: {image_path.name}")

            # Add page separator
            markdown_parts.append(f"\n\n---\n## Page {i}\n\n")

            # Process the image
            page_text = self.process_image(
                image_path,
                output_dir,
                convert_to_markdown=True,
            )
            markdown_parts.append(page_text)

        # Combine all parts
        combined_markdown = "".join(markdown_parts)

        # Add header
        header = f"# OCR Document\n\n**Total Pages:** {total_pages}\n\n---\n"
        combined_markdown = header + combined_markdown

        return combined_markdown

    def is_loaded(self) -> bool:
        """Check if model is loaded."""
        return self._loaded


# Global model instance (singleton)
_ocr_model: Optional[DeepSeekOCR] = None


def get_ocr_model() -> DeepSeekOCR:
    """Get or create the global OCR model instance."""
    global _ocr_model
    if _ocr_model is None:
        _ocr_model = DeepSeekOCR()
    return _ocr_model

EOF_OCR
echo "âœ“ ocr_model.py"

# s3_service.py
cat > s3_service.py << 'EOF_S3'
"""S3 upload service for OCR results."""
import logging
import boto3
from botocore.exceptions import ClientError
from pathlib import Path
from datetime import datetime
from typing import Optional

from config import get_settings

logger = logging.getLogger(__name__)
settings = get_settings()


class S3Service:
    """Service for uploading OCR results to S3."""

    def __init__(self):
        self.client = None
        self._initialized = False

    def _ensure_client(self):
        """Initialize S3 client if not already done."""
        if self._initialized:
            return

        self.client = boto3.client(
            "s3",
            aws_access_key_id=settings.AWS_ACCESS_KEY_ID,
            aws_secret_access_key=settings.AWS_SECRET_ACCESS_KEY,
            region_name=settings.AWS_DEFAULT_REGION,
        )
        self._initialized = True
        logger.info(f"S3 client initialized for bucket: {settings.S3_BUCKET_NAME}")

    def upload_markdown(
        self,
        content: str,
        filename: str,
        folder: str = "ocr_results",
    ) -> Optional[str]:
        """Upload markdown content to S3.

        Args:
            content: Markdown content to upload
            filename: Name for the file (without extension)
            folder: S3 folder/prefix

        Returns:
            Public URL to the uploaded file, or None if failed
        """
        self._ensure_client()

        # Generate unique filename with timestamp
        timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        safe_filename = "".join(c if c.isalnum() or c in "-_" else "_" for c in filename)
        s3_key = f"{folder}/{safe_filename}_{timestamp}.md"

        try:
            # Upload to S3
            self.client.put_object(
                Bucket=settings.S3_BUCKET_NAME,
                Key=s3_key,
                Body=content.encode("utf-8"),
                ContentType="text/markdown",
                ContentDisposition=f'attachment; filename="{safe_filename}.md"',
            )

            # Generate URL
            url = f"https://{settings.S3_BUCKET_NAME}.s3.{settings.AWS_DEFAULT_REGION}.amazonaws.com/{s3_key}"

            logger.info(f"Uploaded markdown to S3: {s3_key}")
            return url

        except ClientError as e:
            logger.error(f"Failed to upload to S3: {e}")
            return None

    def upload_file(
        self,
        file_path: Path,
        folder: str = "ocr_results",
    ) -> Optional[str]:
        """Upload a file to S3.

        Args:
            file_path: Path to file to upload
            folder: S3 folder/prefix

        Returns:
            Public URL to the uploaded file, or None if failed
        """
        self._ensure_client()

        # Generate S3 key
        timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        s3_key = f"{folder}/{file_path.stem}_{timestamp}{file_path.suffix}"

        try:
            # Determine content type
            content_type = "application/octet-stream"
            if file_path.suffix == ".md":
                content_type = "text/markdown"
            elif file_path.suffix == ".pdf":
                content_type = "application/pdf"
            elif file_path.suffix in [".png", ".jpg", ".jpeg"]:
                content_type = f"image/{file_path.suffix[1:]}"

            # Upload to S3
            self.client.upload_file(
                str(file_path),
                settings.S3_BUCKET_NAME,
                s3_key,
                ExtraArgs={"ContentType": content_type},
            )

            # Generate URL
            url = f"https://{settings.S3_BUCKET_NAME}.s3.{settings.AWS_DEFAULT_REGION}.amazonaws.com/{s3_key}"

            logger.info(f"Uploaded file to S3: {s3_key}")
            return url

        except ClientError as e:
            logger.error(f"Failed to upload file to S3: {e}")
            return None

    def generate_presigned_url(
        self,
        s3_key: str,
        expiration: int = 3600,
    ) -> Optional[str]:
        """Generate a presigned URL for downloading a file.

        Args:
            s3_key: S3 object key
            expiration: URL expiration time in seconds

        Returns:
            Presigned URL or None if failed
        """
        self._ensure_client()

        try:
            url = self.client.generate_presigned_url(
                "get_object",
                Params={
                    "Bucket": settings.S3_BUCKET_NAME,
                    "Key": s3_key,
                },
                ExpiresIn=expiration,
            )
            return url

        except ClientError as e:
            logger.error(f"Failed to generate presigned URL: {e}")
            return None


# Global S3 service instance
_s3_service: Optional[S3Service] = None


def get_s3_service() -> S3Service:
    """Get or create the global S3 service instance."""
    global _s3_service
    if _s3_service is None:
        _s3_service = S3Service()
    return _s3_service

EOF_S3
echo "âœ“ s3_service.py"

# main.py
cat > main.py << 'EOF_MAIN'
"""DeepSeek-OCR-2 FastAPI Service for FDA Document Processing.

This service receives PDF URLs, processes them with DeepSeek-OCR-2,
and uploads the extracted markdown to S3.

Run with: uvicorn main:app --host 0.0.0.0 --port 8000
"""
import os
import logging
import asyncio
from pathlib import Path
from contextlib import asynccontextmanager
from typing import Optional

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, HttpUrl

from config import get_settings
from pdf_processor import PDFProcessor
from ocr_model import get_ocr_model
from s3_service import get_s3_service

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

settings = get_settings()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan - preload model on startup."""
    logger.info("Starting OCR Service...")

    # Preload the model
    try:
        ocr_model = get_ocr_model()
        ocr_model.load_model()
        logger.info("DeepSeek-OCR-2 model loaded and ready")
    except Exception as e:
        logger.error(f"Failed to load model on startup: {e}")
        # Continue anyway - will retry on first request

    yield

    logger.info("Shutting down OCR Service...")


app = FastAPI(
    title="DeepSeek-OCR-2 Service",
    description="OCR service for FDA drug documents using DeepSeek-OCR-2",
    version="1.0.0",
    lifespan=lifespan,
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Request/Response Models
class OCRRequest(BaseModel):
    """Request to process a PDF document."""
    url: HttpUrl = Field(..., description="URL to the PDF document")
    document_id: Optional[str] = Field(None, description="Optional document identifier")
    document_type: Optional[str] = Field(None, description="Document type (Label, Letter, Review)")


class OCRResponse(BaseModel):
    """Response from OCR processing."""
    success: bool
    document_id: Optional[str]
    s3_url: Optional[str] = Field(None, description="URL to download the markdown file")
    page_count: int = 0
    message: str


class BatchOCRRequest(BaseModel):
    """Request to process multiple PDF documents."""
    documents: list[OCRRequest] = Field(..., max_length=10)


class BatchOCRResponse(BaseModel):
    """Response from batch OCR processing."""
    results: list[OCRResponse]
    total: int
    successful: int
    failed: int


# Endpoints
@app.get("/")
async def root():
    """Root endpoint with service info."""
    return {
        "service": "DeepSeek-OCR-2 Service",
        "model": settings.MODEL_NAME,
        "status": "running",
        "endpoints": {
            "process_pdf": "POST /ocr/process",
            "batch_process": "POST /ocr/batch",
            "health": "GET /health",
        },
    }


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    ocr_model = get_ocr_model()
    return {
        "status": "healthy",
        "model_loaded": ocr_model.is_loaded(),
        "model": settings.MODEL_NAME,
    }


@app.post("/ocr/process", response_model=OCRResponse)
async def process_pdf(request: OCRRequest):
    """Process a single PDF document with OCR.

    Downloads the PDF, converts pages to images, runs DeepSeek-OCR-2,
    combines results into markdown, and uploads to S3.
    """
    logger.info(f"Processing PDF: {request.url}")

    pdf_processor = PDFProcessor()
    ocr_model = get_ocr_model()
    s3_service = get_s3_service()

    pdf_path = None
    image_paths = []

    try:
        # Step 1: Download PDF
        pdf_path = await pdf_processor.download_pdf(str(request.url))
        if not pdf_path:
            raise HTTPException(status_code=400, detail="Failed to download PDF")

        # Step 2: Convert PDF to images
        image_paths = pdf_processor.convert_to_images(pdf_path)
        if not image_paths:
            raise HTTPException(status_code=400, detail="Failed to convert PDF to images")

        page_count = len(image_paths)
        logger.info(f"Converted PDF to {page_count} images")

        # Step 3: Process images with OCR
        output_dir = Path(settings.TEMP_DIR) / "output"
        output_dir.mkdir(parents=True, exist_ok=True)

        markdown_content = ocr_model.process_multiple_images(image_paths, output_dir)

        # Step 4: Upload to S3
        doc_id = request.document_id or pdf_path.stem
        doc_type = request.document_type or "document"
        filename = f"{doc_type}_{doc_id}"

        s3_url = s3_service.upload_markdown(
            content=markdown_content,
            filename=filename,
            folder="fda_documents",
        )

        if not s3_url:
            raise HTTPException(status_code=500, detail="Failed to upload to S3")

        logger.info(f"Successfully processed PDF, uploaded to: {s3_url}")

        return OCRResponse(
            success=True,
            document_id=doc_id,
            s3_url=s3_url,
            page_count=page_count,
            message=f"Successfully processed {page_count} pages",
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error processing PDF: {e}")
        raise HTTPException(status_code=500, detail=str(e))

    finally:
        # Cleanup temporary files
        if pdf_path:
            pdf_processor.cleanup(pdf_path)
        if image_paths:
            pdf_processor.cleanup_images(image_paths)


@app.post("/ocr/batch", response_model=BatchOCRResponse)
async def batch_process_pdfs(request: BatchOCRRequest):
    """Process multiple PDF documents.

    Processes documents sequentially to manage GPU memory.
    """
    results = []
    successful = 0
    failed = 0

    for doc_request in request.documents:
        try:
            result = await process_pdf(doc_request)
            results.append(result)
            successful += 1
        except HTTPException as e:
            results.append(
                OCRResponse(
                    success=False,
                    document_id=doc_request.document_id,
                    s3_url=None,
                    page_count=0,
                    message=str(e.detail),
                )
            )
            failed += 1
        except Exception as e:
            results.append(
                OCRResponse(
                    success=False,
                    document_id=doc_request.document_id,
                    s3_url=None,
                    page_count=0,
                    message=str(e),
                )
            )
            failed += 1

    return BatchOCRResponse(
        results=results,
        total=len(request.documents),
        successful=successful,
        failed=failed,
    )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "main:app",
        host=settings.HOST,
        port=settings.PORT,
        reload=False,
    )

EOF_MAIN
echo "âœ“ main.py"

# requirements.txt
cat > requirements.txt << 'EOF_REQ'
# DeepSeek-OCR-2 Service Dependencies
# Python 3.12+ with CUDA 11.8+

# Core ML
torch==2.6.0
transformers==4.46.3
tokenizers==0.20.3
accelerate

# DeepSeek-OCR-2 dependencies
einops
addict
easydict
safetensors

# Flash Attention (install separately with --no-build-isolation)
# flash-attn==2.7.3

# PDF Processing
pdf2image
Pillow

# AWS S3
boto3

# FastAPI
fastapi
uvicorn[standard]
python-multipart
aiofiles
httpx

# Configuration
pydantic-settings
python-dotenv

EOF_REQ
echo "âœ“ requirements.txt"

# start_server.sh
cat > start_server.sh << 'EOF_START'
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

# Install Python dependencies
echo "Installing Python dependencies..."
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

EOF_START
chmod +x start_server.sh
echo "âœ“ start_server.sh"

echo ""
echo "=== All files created successfully ==="
ls -lh