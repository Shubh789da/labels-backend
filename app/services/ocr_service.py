"""Client service for the DeepSeek-OCR-2 RunPod service."""
import logging
import json
from typing import Optional
import httpx

from app.services.base_service import BaseAPIService

logger = logging.getLogger(__name__)


def _mask_auth_header(headers: dict) -> dict:
    """Mask authorization header for safe logging."""
    masked = dict(headers)
    if "authorization" in masked:
        auth = masked["authorization"]
        if auth.startswith("Bearer ") and len(auth) > 15:
            key = auth[7:]
            masked["authorization"] = f"Bearer {key[:4]}...{key[-4:]}"
    return masked


class OCRServiceClient(BaseAPIService):
    """Client for calling the DeepSeek-OCR-2 service on RunPod.

    This service processes FDA document PDFs (labels, letters, reviews)
    and returns URLs to the extracted markdown files in S3.
    """

    @property
    def base_url(self) -> str:
        return self.settings.OCR_SERVICE_URL

    async def process_document(
        self,
        pdf_url: str,
        document_id: Optional[str] = None,
        document_type: Optional[str] = None,
    ) -> Optional[dict]:
        """Process a single PDF document with OCR.

        Args:
            pdf_url: URL to the FDA document PDF
            document_id: Optional identifier for the document
            document_type: Type of document (Label, Letter, Review)

        Returns:
            Dict with s3_url, page_count, etc. or None if failed
        """
        payload = {
            "url": pdf_url,
            "document_id": document_id,
            "document_type": document_type,
        }

        url = f"{self.base_url}/ocr/process"

        try:
            client = await self.get_client()

            # Debug: Log full request details for RunPod API call
            logger.debug("=" * 70)
            logger.debug("[RUNPOD API REQUEST]")
            logger.debug(f"POST {url}")
            logger.debug(f"Headers: {json.dumps(_mask_auth_header(dict(client.headers)), indent=2)}")
            logger.debug(f"Payload: {json.dumps(payload, indent=2)}")
            logger.debug("=" * 70)

            response = await client.post(
                url,
                json=payload,
                timeout=300.0,  # 5 minutes for large PDFs
            )
            response.raise_for_status()
            return response.json()

        except httpx.TimeoutException:
            logger.error(f"Timeout processing document: {pdf_url}")
            return None
        except httpx.HTTPStatusError as e:
            logger.error(f"HTTP error processing document: {e.response.status_code}")
            return None
        except Exception as e:
            logger.error(f"Error processing document: {e}")
            return None

    async def process_batch(
        self,
        documents: list[dict],
    ) -> Optional[dict]:
        """Process multiple PDF documents.

        Args:
            documents: List of dicts with url, document_id, document_type

        Returns:
            Batch response with results for each document
        """
        payload = {"documents": documents}
        url = f"{self.base_url}/ocr/batch"

        try:
            client = await self.get_client()

            # Debug: Log full request details for RunPod API call
            logger.debug("=" * 70)
            logger.debug("[RUNPOD API REQUEST]")
            logger.debug(f"POST {url}")
            logger.debug(f"Headers: {json.dumps(_mask_auth_header(dict(client.headers)), indent=2)}")
            logger.debug(f"Payload: {json.dumps(payload, indent=2)}")
            logger.debug("=" * 70)

            response = await client.post(
                url,
                json=payload,
                timeout=600.0,  # 10 minutes for batch
            )
            response.raise_for_status()
            return response.json()

        except Exception as e:
            logger.error(f"Error processing batch: {e}")
            return None

    async def health_check(self) -> bool:
        """Check if OCR service is healthy.

        Returns:
            True if service is healthy, False otherwise
        """
        url = f"{self.base_url}/health"
        try:
            client = await self.get_client()

            # Debug: Log request details for health check
            logger.debug("=" * 70)
            logger.debug("[RUNPOD API REQUEST]")
            logger.debug(f"GET {url}")
            logger.debug(f"Headers: {json.dumps(_mask_auth_header(dict(client.headers)), indent=2)}")
            logger.debug("=" * 70)

            response = await client.get(url, timeout=10.0)
            return response.status_code == 200
        except Exception:
            return False
