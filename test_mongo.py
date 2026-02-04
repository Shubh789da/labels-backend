import asyncio
import logging
from app.config import get_settings
from motor.motor_asyncio import AsyncIOMotorClient

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def test_mongo_connection():
    settings = get_settings()
    uri = settings.MONGODB_URI
    db_name = settings.MONGODB_DB_NAME
    coll_name = settings.MONGODB_COLLECTION_NAME
    
    print(f"Connecting to: {uri.split('@')[1]}") # Print host part only for security
    print(f"Database: {db_name}")
    print(f"Collection: {coll_name}")

    try:
        client = AsyncIOMotorClient(uri)
        db = client[db_name]
        collection = db[coll_name]
        
        # Test 1: Ping
        await client.admin.command('ping')
        print("✅ Ping successful")
        
        # Test 2: Write
        test_doc = {
            "url": "https://test.com/sample.pdf",
            "indication_text": "Test Indication",
            "formatted_text": "• Bullet",
            "indication_count": 1,
            "test_flag": True
        }
        
        result = await collection.update_one(
            {"url": test_doc["url"]},
            {"$set": test_doc},
            upsert=True
        )
        
        print(f"✅ Write successful. Modified: {result.modified_count}, Upserted: {result.upserted_id}, Matched: {result.matched_count}")
        
        # Test 3: Read
        doc = await collection.find_one({"url": test_doc["url"]})
        if doc:
            print(f"✅ Read successful: {doc['url']}")
        else:
            print("❌ Read failed")
            
    except Exception as e:
        print(f"❌ Error: {e}")
    finally:
        client.close()

if __name__ == "__main__":
    asyncio.run(test_mongo_connection())
