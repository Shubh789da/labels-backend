
import logging
import httpx
import json
from typing import Optional, Dict, Any

from app.config import get_settings

logger = logging.getLogger(__name__)


def _mask_api_key(key: str) -> str:
    """Mask the API key for safe logging."""
    if len(key) > 8:
        return f"{key[:4]}...{key[-4:]}"
    return "***"

class OCRServiceClient:
    """Client for interacting with the DeepSeek-OCR service on RunPod."""

    def __init__(self):
        settings = get_settings()
        self.base_url = settings.OCR_SERVICE_URL
        self.api_key = settings.RUNPOD_API_KEY
        self.headers = {
            "Content-Type": "application/json"
        }
        if self.api_key:
            self.headers["Authorization"] = f"Bearer {self.api_key}"

    async def close(self):
        """Close method for compatibility with dependency injection pattern."""
        # This client creates a new httpx.AsyncClient per request using context manager
        # so there's nothing to explicitly close
        pass

    async def extract_indications(self, pdf_url: str, filename_prefix: str) -> Optional[str]:
        """
        Extract indications from a PDF via the OCR service.
        
        Args:
            pdf_url: URL of the PDF document.
            filename_prefix: Prefix for the S3 filename (usually drug name).
            
        Returns:
            Extracted text for "indications_and_usage" or None if failed.
        """
        payload = {
            "input": {
                "pdf_url": pdf_url,
                "save_to_s3": True,
                "filename_prefix": filename_prefix
            }
        }
        
        try:
            # Debug: Log full request details for RunPod API call
            masked_headers = self.headers.copy()
            if "Authorization" in masked_headers:
                masked_headers["Authorization"] = f"Bearer {_mask_api_key(self.api_key)}"

            logger.debug("=" * 70)
            logger.debug("[RUNPOD API REQUEST]")
            logger.debug(f"POST {self.base_url}")
            logger.debug(f"Headers: {json.dumps(masked_headers, indent=2)}")
            logger.debug(f"Payload: {json.dumps(payload, indent=2)}")
            logger.debug("=" * 70)

            async with httpx.AsyncClient(timeout=300.0) as client: # Generous timeout for OCR
                response = await client.post(self.base_url, json=payload, headers=self.headers)
                response.raise_for_status()
                data = response.json()
                task_id = data.get("id")
                status = data.get("status")

                logger.info(f"[OCR] Initial response - task_id: {task_id}, status: {status}")

                # Check for direct completion
                if status == "COMPLETED" and "output" in data:
                    logger.info(f"[OCR] Task completed immediately")
                    return self._extract_output(data)

                # Check for immediate error
                if status == "FAILED":
                    error_msg = data.get("error", "Unknown error")
                    logger.error(f"[OCR] Task failed immediately: {error_msg}")
                    return None

                # Poll if in queue/progress
                if status in ["IN_QUEUE", "IN_PROGRESS"] and task_id:
                    base_status_url = self.base_url.replace("/run", "")
                    status_url = f"{base_status_url}/status/{task_id}"

                    # Poll for up to 300 seconds
                    import asyncio
                    max_retries = 100
                    for retry in range(max_retries):
                        await asyncio.sleep(3)

                        status_res = await client.get(status_url, headers=self.headers)
                        if status_res.status_code != 200:
                            logger.warning(f"[OCR] Status check failed with code {status_res.status_code}")
                            continue

                        status_data = status_res.json()
                        current_status = status_data.get("status")

                        # Log progress every 10 retries (30 seconds)
                        if retry % 10 == 0:
                            logger.info(f"[OCR] Polling... status: {current_status}, retry: {retry}/{max_retries}")

                        if current_status == "COMPLETED":
                            logger.info(f"[OCR] Task {task_id} completed after {retry * 3} seconds")
                            return self._extract_output(status_data)
                        elif current_status == "FAILED":
                            error_msg = status_data.get("error", "Unknown error")
                            logger.error(f"[OCR] Task {task_id} failed: {error_msg}")
                            return None

                logger.warning(f"[OCR] Request timed out after {max_retries * 3} seconds")
                return None

        except Exception as e:
            logger.error(f"OCR Service call failed: {e}")
            return None

    def _extract_output(self, data: Dict[str, Any]) -> Optional[str]:
        """Helper to extract relevant text from completed output."""
        output = data.get("output", {})

        logger.info(f"[OCR] Extracting output - output type: {type(output).__name__}")

        if isinstance(output, dict):
            indications = output.get("indications_and_usage")
            page_count = output.get("page_count", "unknown")
            logger.info(f"[OCR] page_count: {page_count}, indications length: {len(indications) if indications else 0}")

            if not indications:
                # Log available keys for debugging
                logger.warning(f"[OCR] No indications_and_usage found. Available keys: {list(output.keys())}")

            return indications
        else:
            logger.warning(f"[OCR] Unexpected output format: {str(output)[:200]}")
            return None
