import json
from io import BytesIO
from pathlib import Path
import shutil
import tempfile

import h5py
import numpy as np
from PIL import Image
import tensorflow as tf
from tensorflow import keras

from app.core.config import get_settings


class InferenceService:
    def __init__(self) -> None:
        self.settings = get_settings()
        self._model = None
        self._idx_to_label: dict[str, str] | None = None
        self._sanitized_model_path: Path | None = None

    def _resolve_path(self, path: Path) -> Path:
        if path.is_absolute():
            return path
        return (Path(__file__).resolve().parents[3] / path).resolve()

    def load(self) -> None:
        if self._model is not None and self._idx_to_label is not None:
            return

        model_path = self._resolve_path(self.settings.model_path)
        labels_path = self._resolve_path(self.settings.labels_path)

        try:
            self._model = keras.models.load_model(str(model_path), compile=False)
        except TypeError as exc:
            if "quantization_config" not in str(exc):
                raise
            sanitized_path = self._create_sanitized_h5_copy(model_path)
            self._sanitized_model_path = sanitized_path
            self._model = keras.models.load_model(str(sanitized_path), compile=False)

        with labels_path.open("r", encoding="utf-8") as file:
            labels_data = json.load(file)
        self._idx_to_label = labels_data["idx_to_label"]

    def _create_sanitized_h5_copy(self, model_path: Path) -> Path:
        temp_dir = Path(tempfile.mkdtemp(prefix="skin_model_", dir=Path.cwd()))
        sanitized_path = temp_dir / model_path.name
        shutil.copy2(model_path, sanitized_path)

        with h5py.File(sanitized_path, "r+") as h5_file:
            model_config = h5_file.attrs.get("model_config")
            if model_config is None:
                raise RuntimeError("Saved model does not contain model_config metadata.")

            if isinstance(model_config, bytes):
                decoded = model_config.decode("utf-8")
            else:
                decoded = model_config

            config_data = json.loads(decoded)
            cleaned_config = self._remove_quantization_config(config_data)
            h5_file.attrs["model_config"] = json.dumps(cleaned_config).encode("utf-8")

        return sanitized_path

    def _remove_quantization_config(self, value):
        if isinstance(value, dict):
            return {
                key: self._remove_quantization_config(item)
                for key, item in value.items()
                if key != "quantization_config"
            }
        if isinstance(value, list):
            return [self._remove_quantization_config(item) for item in value]
        return value

    def _load_image(self, image_bytes: bytes) -> Image.Image:
        image = Image.open(BytesIO(image_bytes))
        if image.mode != "RGB":
            image = image.convert("RGB")
        image = image.resize((self.settings.image_size, self.settings.image_size))
        return image

    def predict(self, image_bytes: bytes) -> dict:
        self.load()
        if self._model is None or self._idx_to_label is None:
            raise RuntimeError("Inference service failed to initialize.")

        image = self._load_image(image_bytes)
        image_array = np.array(image, dtype=np.float32)
        image_array = tf.keras.applications.mobilenet_v2.preprocess_input(image_array)
        image_array = np.expand_dims(image_array, axis=0)

        predictions = self._model.predict(image_array, verbose=0)[0]
        top_idx = int(np.argmax(predictions))
        ranked_indices = np.argsort(predictions)[::-1][:3]

        probabilities = [
            {
                "label": self._idx_to_label[str(int(index))],
                "confidence": float(predictions[int(index)]),
            }
            for index in ranked_indices
        ]

        return {
            "predicted_label": self._idx_to_label[str(top_idx)],
            "confidence": float(predictions[top_idx]),
            "probabilities": probabilities,
        }


inference_service = InferenceService()
