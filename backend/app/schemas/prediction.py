from datetime import datetime

from pydantic import BaseModel, Field


class PredictionProbability(BaseModel):
    label: str
    confidence: float


class PredictionResponse(BaseModel):
    id: str
    filename: str
    content_type: str
    predicted_label: str
    confidence: float
    description: str
    fallback_used: bool
    probabilities: list[PredictionProbability]
    image_url: str
    created_at: datetime


class PredictionHistoryResponse(BaseModel):
    items: list[PredictionResponse] = Field(default_factory=list)
