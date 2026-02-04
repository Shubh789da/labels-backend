"""Base service class with common HTTP functionality."""
import httpx
from typing import Optional, Any
from abc import ABC, abstractmethod
import logging

from app.config import get_settings

logger = logging.getLogger(__name__)


class BaseAPIService(ABC):
    """Base class for API services with common HTTP functionality."""

    def __init__(self):
        self.settings = get_settings()
        self._client: Optional[httpx.AsyncClient] = None

    @property
    @abstractmethod
    def base_url(self) -> str:
        """Return the base URL for this service."""
        pass

    async def get_client(self) -> httpx.AsyncClient:
        """Get or create an async HTTP client."""
        if self._client is None or self._client.is_closed:
            self._client = httpx.AsyncClient(
                timeout=httpx.Timeout(self.settings.REQUEST_TIMEOUT),
                follow_redirects=True,
            )
        return self._client

    async def close(self):
        """Close the HTTP client."""
        if self._client and not self._client.is_closed:
            await self._client.aclose()

    async def _get(
        self,
        endpoint: str,
        params: Optional[dict[str, Any]] = None,
    ) -> Optional[dict]:
        """Make a GET request to the API.

        Args:
            endpoint: API endpoint path
            params: Query parameters

        Returns:
            JSON response as dict, or None if request failed
        """
        url = f"{self.base_url}{endpoint}"
        client = await self.get_client()

        try:
            response = await client.get(url, params=params)
            response.raise_for_status()
            return response.json()
        except httpx.HTTPStatusError as e:
            logger.warning(f"HTTP error from {url}: {e.response.status_code}")
            return None
        except httpx.RequestError as e:
            logger.error(f"Request error for {url}: {e}")
            return None
        except Exception as e:
            logger.error(f"Unexpected error for {url}: {e}")
            return None
