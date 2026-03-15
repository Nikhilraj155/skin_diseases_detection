from app.core.database import get_database
from app.repositories.prediction_repository import PredictionRepository
from app.services.image_service import ImageService


def get_prediction_repository() -> PredictionRepository:
    database = get_database()
    return PredictionRepository(database)


async def get_image_service() -> ImageService:
    database = get_database()
    return ImageService(database)
