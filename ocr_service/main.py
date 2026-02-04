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
