from bson import ObjectId
from bson.errors import InvalidId
from motor.motor_asyncio import AsyncIOMotorDatabase

from app.models.prediction import build_prediction_document


class PredictionRepository:
    def __init__(self, database: AsyncIOMotorDatabase) -> None:
        self.collection = database["predictions"]

    async def create_prediction(
        self,
        *,
        filename: str,
        content_type: str,
        image_file_id: str,
        predicted_label: str,
        confidence: float,
        description: str,
        fallback_used: bool,
        probabilities: list[dict[str, float | str]],
    ) -> dict:
        document = build_prediction_document(
            filename=filename,
            content_type=content_type,
            image_file_id=image_file_id,
            predicted_label=predicted_label,
            confidence=confidence,
            description=description,
            fallback_used=fallback_used,
            probabilities=probabilities,
        )
        result = await self.collection.insert_one(document)
        document["_id"] = result.inserted_id
        return document

    async def get_prediction(self, prediction_id: str) -> dict | None:
        try:
            object_id = ObjectId(prediction_id)
        except InvalidId:
            return None
        return await self.collection.find_one({"_id": object_id})

    async def list_predictions(self, limit: int = 20) -> list[dict]:
        cursor = self.collection.find().sort("created_at", -1).limit(limit)
        return await cursor.to_list(length=limit)
