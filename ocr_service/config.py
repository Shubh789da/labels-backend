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
