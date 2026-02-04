"""
Indication Pipeline Service - Orchestrates PDF extraction with intelligent routing.

This service implements the backend-first pipeline:
1. Primary Path: Local PyMuPDF extraction → DeepSeek LLM formatting
2. Fallback Path: RunPod OCR extraction → DeepSeek LLM formatting

Both paths output formatted bullet points for frontend display.
"""

import logging
from typing import Optional
from dataclasses import dataclass
from enum import Enum

from app.services.pdf_extractor import get_pdf_extractor
from app.services.deepseek_llm import get_deepseek_llm_service
from app.services.ocr_client import OCRServiceClient
from app.services.cache_service import get_mongo_cache_service

logger = logging.getLogger(__name__)


class ExtractionMethod(str, Enum):
    """Method used to extract the indications section."""
    LOCAL = "local"
    OCR = "ocr"
    CACHE = "cache"
    FAILED = "failed"


@dataclass
class IndicationResult:
    """Result from the indication extraction pipeline."""
    found: bool
    text: str
    method: ExtractionMethod
    formatted: bool
    raw_text: Optional[str] = None  # Original unformatted text
    error_message: Optional[str] = None
    indication_count: int = 0


class IndicationPipelineService:
    """
    Orchestrates the indication extraction pipeline with intelligent routing.
    
    Flow:
    1. Try local PDF extraction using PyMuPDF
    2. If local fails (image-based PDF or section not found), use OCR fallback
    3. Always format extracted text with DeepSeek LLM
    4. Return formatted bullet points to caller
    """
    
    def __init__(self):
        self.pdf_extractor = get_pdf_extractor()
        self.llm_service = get_deepseek_llm_service()
        self.ocr_client = OCRServiceClient()
        self.cache_service = get_mongo_cache_service()
    
    async def extract_indications(self, pdf_url: str, filename_prefix: str) -> IndicationResult:
        """
        Extract and format INDICATIONS AND USAGE from a PDF.
        
        Args:
            pdf_url: URL of the FDA label PDF
            filename_prefix: Prefix for logging/identification (e.g., drug name)
            
        Returns:
            IndicationResult with formatted bullet points or error message
        """
        logger.info(f"[IndicationPipeline] Starting extraction for: {filename_prefix}")
        logger.info(f"[IndicationPipeline] PDF URL: {pdf_url[:100]}...")
        
        raw_text: Optional[str] = None
        method = ExtractionMethod.FAILED
        
        # ==========================================
        # Step 0: Check MongoDB Cache
        # ==========================================
        cached_doc = await self.cache_service.get_indication(pdf_url)
        if cached_doc:
            logger.info(f"[IndicationPipeline] Returning cached result for {filename_prefix}")
            return IndicationResult(
                found=True,
                text=cached_doc.get("formatted_text"),
                method=ExtractionMethod.CACHE,
                formatted=True,
                raw_text=cached_doc.get("indication_text"),
                indication_count=cached_doc.get("indication_count", 0)
            )

        # ==========================================
        # Step 1: Try local PyMuPDF extraction first
        # ==========================================
        logger.info("[IndicationPipeline] Attempting local PDF extraction...")
        
        section_text, status = await self.pdf_extractor.extract_indications_from_url(pdf_url)
        
        if section_text and status == "success":
            logger.info(f"[IndicationPipeline] Local extraction succeeded: {len(section_text)} chars")
            raw_text = section_text
            method = ExtractionMethod.LOCAL
        else:
            logger.info(f"[IndicationPipeline] Local extraction failed: {status}")
            
            # ==========================================
            # Step 2: Fallback to RunPod OCR
            # ==========================================
            # Fallback for: download_failed (FDA website may block us), no_text (image PDF), section_not_found
            if status in ["download_failed", "no_text", "section_not_found"]:
                logger.info(f"[IndicationPipeline] Falling back to OCR extraction (reason: {status})...")
                
                try:
                    ocr_result = await self.ocr_client.extract_indications(pdf_url, filename_prefix)
                    
                    if ocr_result and len(ocr_result.strip()) > 50:
                        logger.info(f"[IndicationPipeline] OCR extraction succeeded: {len(ocr_result)} chars")
                        raw_text = ocr_result
                        method = ExtractionMethod.OCR
                    else:
                        logger.warning("[IndicationPipeline] OCR extraction returned empty or insufficient text")
                except Exception as e:
                    logger.error(f"[IndicationPipeline] OCR fallback failed: {e}")
        
        # ==========================================
        # Step 3: Both methods failed
        # ==========================================
        if not raw_text:
            logger.error(f"[IndicationPipeline] All extraction methods failed for: {filename_prefix}")
            return IndicationResult(
                found=False,
                text="Unable to extract the INDICATIONS AND USAGE section from this document. "
                     "The PDF may be in an unsupported format or the section may not be present.",
                method=ExtractionMethod.FAILED,
                formatted=False,
                error_message="Both local extraction and OCR fallback failed"
            )
        
        # ==========================================
        # Step 4: Format with DeepSeek LLM
        # ==========================================
        logger.info("[IndicationPipeline] Formatting extracted text with DeepSeek LLM...")
        
        try:
            result = await self.llm_service.process_indications(raw_text)
            formatted_text = result.get("formatted_text")
            count = result.get("indication_count", 0)
            
            if formatted_text:
                logger.info(f"[IndicationPipeline] LLM formatted: {len(formatted_text)} chars, Count: {count}")
                
                # Save to cache
                await self.cache_service.save_indication(
                    url=pdf_url,
                    raw_text=raw_text,
                    formatted_text=formatted_text,
                    count=count
                )
                
                return IndicationResult(
                    found=True,
                    text=formatted_text,
                    method=method,
                    formatted=True,
                    raw_text=raw_text,
                    indication_count=count
                )
            else:
                # LLM formatting failed, return raw text with basic cleanup
                logger.warning("[IndicationPipeline] LLM formatting failed, returning raw text")
                return IndicationResult(
                    found=True,
                    text=raw_text,
                    method=method,
                    formatted=False,
                    raw_text=raw_text,
                    error_message="LLM formatting failed, returning unformatted text"
                )
                
        except Exception as e:
            logger.error(f"[IndicationPipeline] LLM formatting error: {e}")
            return IndicationResult(
                found=True,
                text=raw_text,
                method=method,
                formatted=False,
                raw_text=raw_text,
                error_message=f"LLM formatting error: {str(e)}"
            )
    
    async def close(self):
        """Clean up resources."""
        try:
            await self.llm_service.close()
        except Exception:
            pass
        try:
            await self.ocr_client.close()
        except Exception:
            pass


def get_indication_pipeline() -> IndicationPipelineService:
    """Get a new indication pipeline service instance."""
    return IndicationPipelineService()
