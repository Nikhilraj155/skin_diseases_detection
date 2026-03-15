from bson import ObjectId
from motor.motor_asyncio import AsyncIOMotorDatabase, AsyncIOMotorGridFSBucket


class ImageService:
    def __init__(self, database: AsyncIOMotorDatabase) -> None:
        self.bucket = AsyncIOMotorGridFSBucket(database)

    async def save_image(self, *, filename: str, content_type: str, data: bytes) -> str:
        file_id = await self.bucket.upload_from_stream(
            filename,
            data,
            metadata={"content_type": content_type},
        )
        return str(file_id)

    async def open_download_stream(self, file_id: str):
        return await self.bucket.open_download_stream(ObjectId(file_id))
