from motor.motor_asyncio import AsyncIOMotorClient, AsyncIOMotorDatabase

from app.core.config import get_settings


class MongoDatabase:
    client: AsyncIOMotorClient | None = None
    database: AsyncIOMotorDatabase | None = None


db = MongoDatabase()


async def connect_to_mongo() -> None:
    settings = get_settings()
    db.client = AsyncIOMotorClient(settings.mongo_uri)
    db.database = db.client[settings.mongo_db_name]


async def close_mongo_connection() -> None:
    if db.client is not None:
        db.client.close()
        db.client = None
        db.database = None


def get_database() -> AsyncIOMotorDatabase:
    if db.database is None:
        raise RuntimeError("MongoDB connection is not initialized.")
    return db.database
