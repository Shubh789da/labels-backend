"""
PDF Extractor Service - Local PDF text extraction using PyMuPDF (fitz).

This service provides the primary path for extracting "INDICATIONS AND USAGE" 
section from FDA pharmaceutical labels without relying on OCR.
"""

import io
import re
import logging
from typing import Optional, Tuple

import httpx
import fitz  # PyMuPDF

logger = logging.getLogger(__name__)


# Compiled regex patterns for INDICATIONS AND USAGE section extraction
# Adapted from deepseek-OCR/dsk-ep/extraction.py
INDICATIONS_HEADER = re.compile(
    r"""
    ^\s*
    (\#{1,6}\s*)?          # Markdown heading
    [-*]*\s*               # decoration before
    (\d+\s*\.?\s*)?        # optional numbering (e.g., "1 " or "1. ")
    INDICATIONS?\s+(?:AND|&)\s+USAGE
    \s*:?\s*               # optional colon
    [-*]*\s*               # decoration after
    $
    """,
    re.IGNORECASE | re.MULTILINE | re.VERBOSE
)

NEXT_SECTION_HEADER = re.compile(
    r"""
    ^\s*
    (\#{1,6}\s*)?          # Markdown heading
    [-*]*\s*
    (\d+\s*\.?\s*)?
    [A-Z][A-Z\s/]{4,}     # FDA section titles (all caps, at least 5 chars)
    \s*:?\s*
    [-*]*\s*
    $
    """,
    re.MULTILINE | re.VERBOSE
)


class PDFExtractor:
    """Service for extracting text from PDFs using PyMuPDF."""
    
    def __init__(self):
        self.timeout = 60.0  # 60 second timeout for PDF download
    
    async def download_pdf_to_memory(self, pdf_url: str) -> Optional[bytes]:
        """
        Download PDF from URL directly into memory.
        
        Args:
            pdf_url: URL of the PDF to download
            
        Returns:
            PDF bytes if successful, None if failed
        """
        try:
            logger.info(f"[PDFExtractor] Downloading PDF from: {pdf_url[:100]}...")
            
            # Use browser-like headers to avoid abuse detection on FDA sites
            headers = {
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
                "Accept": "application/pdf,*/*",
                "Accept-Language": "en-US,en;q=0.9",
                "Referer": "https://www.accessdata.fda.gov/",
            }
            
            async with httpx.AsyncClient(timeout=self.timeout, follow_redirects=True) as client:
                response = await client.get(pdf_url, headers=headers)
                response.raise_for_status()
                
                content_type = response.headers.get("content-type", "")
                if "pdf" not in content_type.lower() and not pdf_url.lower().endswith(".pdf"):
                    logger.warning(f"[PDFExtractor] Content-Type may not be PDF: {content_type}")
                
                pdf_bytes = response.content
                logger.info(f"[PDFExtractor] Downloaded {len(pdf_bytes)} bytes")
                return pdf_bytes
                
        except httpx.HTTPStatusError as e:
            logger.error(f"[PDFExtractor] HTTP error downloading PDF: {e}")
            return None
        except Exception as e:
            logger.error(f"[PDFExtractor] Failed to download PDF: {e}")
            return None
    
    def extract_text_from_bytes(self, pdf_bytes: bytes) -> Tuple[str, int, bool]:
        """
        Extract all text from PDF bytes using PyMuPDF.
        
        Args:
            pdf_bytes: Raw PDF content
            
        Returns:
            Tuple of (extracted_text, page_count, has_text)
            has_text is True if meaningful text was extracted
        """
        try:
            # Open PDF from bytes
            doc = fitz.open(stream=pdf_bytes, filetype="pdf")
            page_count = len(doc)
            
            all_text = []
            total_chars = 0
            
            for page_num in range(page_count):
                page = doc[page_num]
                text = page.get_text("text")
                all_text.append(text)
                total_chars += len(text.strip())
            
            doc.close()
            
            full_text = "\n\n".join(all_text)
            
            # Determine if PDF has meaningful embedded text
            # A PDF is considered image-based if average chars per page < 100
            avg_chars_per_page = total_chars / max(page_count, 1)
            has_text = avg_chars_per_page >= 100
            
            logger.info(f"[PDFExtractor] Extracted {total_chars} chars from {page_count} pages (avg: {avg_chars_per_page:.0f}/page)")
            
            return full_text, page_count, has_text
            
        except Exception as e:
            logger.error(f"[PDFExtractor] Failed to extract text from PDF: {e}")
            return "", 0, False
    
    def find_indications_section(self, text: str) -> Optional[str]:
        """
        Find and extract the INDICATIONS AND USAGE section from PDF text.
        
        Args:
            text: Full text extracted from PDF
            
        Returns:
            Extracted section content or None if not found
        """
        if not text or len(text.strip()) < 50:
            return None
        
        # Find the start of INDICATIONS AND USAGE section
        match = INDICATIONS_HEADER.search(text)
        if not match:
            # Try alternative patterns
            alt_patterns = [
                r"INDICATIONS?\s+AND\s+USAGE",
                r"1\s+INDICATIONS?\s+AND\s+USAGE",
                r"INDICATIONS?\s*&\s*USAGE",
            ]
            for pattern in alt_patterns:
                alt_match = re.search(pattern, text, re.IGNORECASE)
                if alt_match:
                    match = alt_match
                    break
        
        if not match:
            logger.info("[PDFExtractor] INDICATIONS AND USAGE section not found")
            return None
        
        start = match.end()
        
        # Find the next section header to determine end of section
        next_match = NEXT_SECTION_HEADER.search(text, start)
        
        # Also look for numbered sections like "2 DOSAGE"
        numbered_section = re.search(r"\n\s*2\s+[A-Z]", text[start:])
        if numbered_section:
            numbered_pos = start + numbered_section.start()
            if next_match is None or numbered_pos < next_match.start():
                end = numbered_pos
            else:
                end = next_match.start()
        else:
            end = next_match.start() if next_match else len(text)
        
        section_content = text[start:end].strip()
        
        # Validate section is substantial enough
        if len(section_content) < 50:
            logger.warning(f"[PDFExtractor] Section too short ({len(section_content)} chars)")
            return None
        
        # Limit section to reasonable length (first 10000 chars)
        if len(section_content) > 10000:
            section_content = section_content[:10000] + "..."
            logger.info("[PDFExtractor] Truncated section to 10000 chars")
        
        logger.info(f"[PDFExtractor] Found INDICATIONS section: {len(section_content)} chars")
        return section_content
    
    async def extract_indications_from_url(self, pdf_url: str) -> Tuple[Optional[str], str]:
        """
        Full pipeline: Download PDF and extract INDICATIONS AND USAGE section.
        
        Args:
            pdf_url: URL of the FDA label PDF
            
        Returns:
            Tuple of (extracted_section_or_None, status_message)
            Status can be: "success", "no_text", "section_not_found", "download_failed"
        """
        # Step 1: Download PDF
        pdf_bytes = await self.download_pdf_to_memory(pdf_url)
        if not pdf_bytes:
            return None, "download_failed"
        
        # Step 2: Extract text
        text, page_count, has_text = self.extract_text_from_bytes(pdf_bytes)
        
        if not has_text:
            logger.info("[PDFExtractor] PDF appears to be image-based (no embedded text)")
            return None, "no_text"
        
        # Step 3: Find INDICATIONS section
        section = self.find_indications_section(text)
        
        if not section:
            return None, "section_not_found"
        
        return section, "success"


# Singleton instance
_pdf_extractor: Optional[PDFExtractor] = None


def get_pdf_extractor() -> PDFExtractor:
    """Get or create the PDF extractor singleton."""
    global _pdf_extractor
    if _pdf_extractor is None:
        _pdf_extractor = PDFExtractor()
    return _pdf_extractor
