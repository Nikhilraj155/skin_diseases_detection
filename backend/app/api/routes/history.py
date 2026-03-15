from fastapi import APIRouter, Depends, HTTPException

from app.api.dependencies import get_prediction_repository
from app.core.config import get_settings
from app.repositories.prediction_repository import PredictionRepository
from app.schemas.prediction import PredictionHistoryResponse, PredictionResponse

router = APIRouter(prefix="/history", tags=["history"])


def serialize_prediction(document: dict) -> PredictionResponse:
    settings = get_settings()
    return PredictionResponse(
        id=str(document["_id"]),
        filename=document["filename"],
        content_type=document["content_type"],
        predicted_label=document["predicted_label"],
        confidence=document["confidence"],
        description=document["description"],
        fallback_used=document["fallback_used"],
        probabilities=document["probabilities"],
        image_url=f"{settings.api_prefix}/predictions/{document['_id']}/image",
        created_at=document["created_at"],
    )


@router.get("", response_model=PredictionHistoryResponse)
async def list_history(
    repository: PredictionRepository = Depends(get_prediction_repository),
) -> PredictionHistoryResponse:
    items = await repository.list_predictions()
    return PredictionHistoryResponse(items=[serialize_prediction(item) for item in items])


@router.get("/{prediction_id}", response_model=PredictionResponse)
async def get_history_item(
    prediction_id: str,
    repository: PredictionRepository = Depends(get_prediction_repository),
) -> PredictionResponse:
    document = await repository.get_prediction(prediction_id)
    if document is None:
        raise HTTPException(status_code=404, detail="Prediction not found.")
    return serialize_prediction(document)
