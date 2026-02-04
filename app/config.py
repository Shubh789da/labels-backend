"""Configuration settings for the Drug History API."""
from pydantic_settings import BaseSettings
from functools import lru_cache


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    # API Base URLs
    OPENFDA_BASE_URL: str = "https://api.fda.gov"
    DAILYMED_BASE_URL: str = "https://dailymed.nlm.nih.gov/dailymed/services/v2"
    RXNORM_BASE_URL: str = "https://rxnav.nlm.nih.gov/REST"

    # OCR Service (DeepSeek-OCR-2 on RunPod)
    # OCR Service (DeepSeek-OCR-2 on RunPod)
    OCR_SERVICE_URL: str = "https://api.runpod.ai/v2/qapobo6yo6o9rf/run"

    # API Keys (optional for openFDA, increases rate limit)
    OPENFDA_API_KEY: str = ""
    RUNPOD_API_KEY: str = ""
    
    # DeepSeek LLM API (for formatting extracted text)
    DEEPSEEK_API_KEY: str = ""
    DEEPSEEK_API_URL: str = "https://api.deepseek.com"

    # Rate limiting
    REQUEST_TIMEOUT: int = 30
    MAX_CONCURRENT_REQUESTS: int = 10

    # Cache settings (TTL in seconds)
    CACHE_TTL: int = 3600  # 1 hour

    # App settings
    APP_NAME: str = "Drug Approval History API"
    APP_VERSION: str = "1.0.0"
    DEBUG: bool = False

    # MongoDB Settings
    MONGODB_URI: str = "mongodb+srv://shubhanshu4v_db_user:dZ3zSB6qq1abJOkV@labels.qyeimsy.mongodb.net/?appName=labels"
    MONGODB_DB_NAME: str = "labels"
    MONGODB_COLLECTION_NAME: str = "labels_records"

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"


@lru_cache()
def get_settings() -> Settings:
    """Get cached settings instance."""
    return Settings()
