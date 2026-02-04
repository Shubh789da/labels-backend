"""
MongoDB Cache Service for storing extracted indications.
"""
import logging
from datetime import datetime, timezone
from typing import Optional, Dict, Any

from motor.motor_asyncio import AsyncIOMotorClient
from pymongo import IndexModel, ASCENDING

from app.config import get_settings

logger = logging.getLogger(__name__)


class MongoCacheService:
    """Service for caching extracted indications in MongoDB."""

    def __init__(self):
        self.settings = get_settings()
        self.client: Optional[AsyncIOMotorClient] = None
        self.db = None
        self.collection = None
        self._connect()

    def _connect(self):
        """Initialize MongoDB connection."""
        try:
            if not self.settings.MONGODB_URI:
                logger.warning("[MongoCache] No MONGODB_URI configured, cache disabled")
                return

            self.client = AsyncIOMotorClient(self.settings.MONGODB_URI)
            self.db = self.client[self.settings.MONGODB_DB_NAME]
            self.collection = self.db[self.settings.MONGODB_COLLECTION_NAME]
            logger.info(f"[MongoCache] Connected to MongoDB: {self.settings.MONGODB_DB_NAME}.{self.settings.MONGODB_COLLECTION_NAME}")
            
        except Exception as e:
            logger.error(f"[MongoCache] Connection failed: {e}")
            self.client = None

    async def init_indexes(self):
        """Create indexes for the collection."""
        if self.collection is None:
            return

        try:
            # Create unique index on URL
            await self.collection.create_index([("url", ASCENDING)], unique=True)
            logger.info("[MongoCache] Indexes ensured")
        except Exception as e:
            logger.error(f"[MongoCache] Index creation failed: {e}")

    async def get_indication(self, url: str) -> Optional[Dict[str, Any]]:
        """
        Retrieve indication from cache by URL.
        
        Args:
            url: PDF URL
            
        Returns:
            Dict with cached data or None
        """
        if self.collection is None:
            return None

        try:
            doc = await self.collection.find_one({"url": url})
            if doc:
                logger.info(f"[MongoCache] Cache HIT for {url}")
                return doc
            return None
        except Exception as e:
            logger.error(f"[MongoCache] Fetch failed: {e}")
            return None

    async def save_indication(self, url: str, raw_text: str, formatted_text: str, count: int) -> bool:
        """
        Save indication to cache.
        
        Args:
            url: PDF URL
            raw_text: Original extracted text
            formatted_text: LLM formatted bullets
            count: Indication count
            
        Returns:
            True if saved
        """
        if self.collection is None:
            return False

        try:
            now = datetime.now(timezone.utc)
            
            update_data = {
                "url": url,
                "indication_text": raw_text,
                "formatted_text": formatted_text,
                "indication_count": count,
                "updated_at": now
            }
            
            result = await self.collection.update_one(
                {"url": url},
                {
                    "$set": update_data,
                    "$setOnInsert": {"created_at": now}
                },
                upsert=True
            )
            
            logger.info(f"[MongoCache] Saved cache for {url} (Modified: {result.modified_count}, Upserted: {result.upserted_id})")
            return True
        except Exception as e:
            logger.error(f"[MongoCache] Save failed: {e}")
            return False

    def close(self):
        """Close MongoDB connection."""
        if self.client:
            self.client.close()


_cache_service = None

def get_mongo_cache_service() -> MongoCacheService:
    """Singleton getter for cache service."""
    global _cache_service
    if _cache_service is None:
        _cache_service = MongoCacheService()
    return _cache_service
