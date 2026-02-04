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
