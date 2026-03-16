from fastapi import APIRouter, Depends, File, HTTPException, UploadFile
from fastapi.responses import Response

from app.api.dependencies import get_image_service, get_prediction_repository
from app.api.routes.history import serialize_prediction
from app.core.config import get_settings
from app.repositories.prediction_repository import PredictionRepository
from app.schemas.prediction import PredictionResponse
from app.services.grok_service import grok_service
from app.services.image_service import ImageService
from app.services.inference_service import inference_service

router = APIRouter(prefix="/predictions", tags=["predictions"])


@router.post("", response_model=PredictionResponse)
async def create_prediction(
    file: UploadFile = File(...),
    repository: PredictionRepository = Depends(get_prediction_repository),
    image_service: ImageService = Depends(get_image_service),
) -> PredictionResponse:
    settings = get_settings()
    image_bytes = await file.read()

    if not image_bytes:
        raise HTTPException(status_code=400, detail="Uploaded file is empty.")

    max_size = settings.max_upload_size_mb * 1024 * 1024
    if len(image_bytes) > max_size:
        raise HTTPException(status_code=413, detail="Uploaded file is too large.")

    if not (file.content_type or "").startswith("image/"):
        raise HTTPException(status_code=400, detail="Only image uploads are supported.")

    try:
        prediction = inference_service.predict(image_bytes)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {exc}") from exc

    description, fallback_used = await grok_service.get_disease_description(prediction["predicted_label"])

    image_file_id = await image_service.save_image(
        filename=file.filename or "upload.jpg",
        content_type=file.content_type or "image/jpeg",
        data=image_bytes,
    )
    document = await repository.create_prediction(
        filename=file.filename or "upload.jpg",
        content_type=file.content_type or "image/jpeg",
        image_file_id=image_file_id,
        predicted_label=prediction["predicted_label"],
        confidence=prediction["confidence"],
        description=description,
        fallback_used=fallback_used,
        probabilities=prediction["probabilities"],
    )
    return serialize_prediction(document)


@router.get("/{prediction_id}/image")
async def get_prediction_image(
    prediction_id: str,
    repository: PredictionRepository = Depends(get_prediction_repository),
    image_service: ImageService = Depends(get_image_service),
):
    document = await repository.get_prediction(prediction_id)
    if document is None:
        raise HTTPException(status_code=404, detail="Prediction not found.")

    grid_out = await image_service.open_download_stream(document["image_file_id"])
    image_bytes = await grid_out.read()
    return Response(content=image_bytes, media_type=document["content_type"])


@router.get("/{prediction_id}/report")
async def get_prediction_report(
    prediction_id: str,
    repository: PredictionRepository = Depends(get_prediction_repository),
):
    """Get detailed disease report with symptoms and recommendations"""
    document = await repository.get_prediction(prediction_id)
    if document is None:
        raise HTTPException(status_code=404, detail="Prediction not found.")

    # Get detailed report from Grok service
    detailed_report = await grok_service.get_detailed_report(document["predicted_label"])
    
    # Add prediction metadata
    detailed_report["prediction_id"] = str(document["_id"])
    detailed_report["filename"] = document["filename"]
    detailed_report["confidence"] = document["confidence"]
    detailed_report["confidence_percentage"] = round(document["confidence"] * 100, 2)
    detailed_report["image_url"] = f"/api/v1/predictions/{prediction_id}/image"
    detailed_report["created_at"] = document["created_at"].isoformat() if hasattr(document["created_at"], "isoformat") else str(document["created_at"])
    
    return detailed_report
