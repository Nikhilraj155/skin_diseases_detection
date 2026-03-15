from datetime import datetime
from typing import Any


def build_prediction_document(
    *,
    filename: str,
    content_type: str,
    image_file_id: str,
    predicted_label: str,
    confidence: float,
    description: str,
    fallback_used: bool,
    probabilities: list[dict[str, Any]],
) -> dict[str, Any]:
    return {
        "filename": filename,
        "content_type": content_type,
        "image_file_id": image_file_id,
        "predicted_label": predicted_label,
        "confidence": confidence,
        "description": description,
        "fallback_used": fallback_used,
        "probabilities": probabilities,
        "created_at": datetime.utcnow(),
    }
