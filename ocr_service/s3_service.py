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
