import json
from io import BytesIO
from pathlib import Path
import shutil

import h5py
import numpy as np
from PIL import Image
import tensorflow as tf
from tensorflow import keras

from app.core.config import get_settings


class InferenceService:
    INVALID_IMAGE_MESSAGE = "Invalid Image: This does not appear to be a human skin image."

    def __init__(self) -> None:
        self.settings = get_settings()
        self._model = None
        self._idx_to_label: dict[str, str] | None = None
        self._sanitized_model_path: Path | None = None

    def _resolve_path(self, path: Path) -> Path:
        if path.is_absolute():
            return path
        return (Path(__file__).resolve().parents[3] / path).resolve()

    def _get_sanitized_model_path(self, model_path: Path) -> Path:
        cache_dir = Path(__file__).resolve().parents[2] / ".model_cache"
        cache_dir.mkdir(parents=True, exist_ok=True)
        return cache_dir / f"{model_path.stem}.sanitized{model_path.suffix}"

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
        sanitized_path = self._get_sanitized_model_path(model_path)

        if sanitized_path.exists():
            source_mtime = model_path.stat().st_mtime
            cached_mtime = sanitized_path.stat().st_mtime
            if cached_mtime >= source_mtime:
                return sanitized_path

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
        return image

    def _rgb_to_hsv(self, image_array: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        normalized = image_array.astype(np.float32) / 255.0
        red = normalized[..., 0]
        green = normalized[..., 1]
        blue = normalized[..., 2]

        channel_max = np.max(normalized, axis=-1)
        channel_min = np.min(normalized, axis=-1)
        delta = channel_max - channel_min

        hue = np.zeros_like(channel_max)
        saturation = np.zeros_like(channel_max)
        value = channel_max

        non_zero = channel_max > 0
        saturation[non_zero] = delta[non_zero] / channel_max[non_zero]

        delta_non_zero = delta > 1e-6
        red_is_max = (channel_max == red) & delta_non_zero
        green_is_max = (channel_max == green) & delta_non_zero
        blue_is_max = (channel_max == blue) & delta_non_zero

        hue[red_is_max] = ((green[red_is_max] - blue[red_is_max]) / delta[red_is_max]) % 6
        hue[green_is_max] = ((blue[green_is_max] - red[green_is_max]) / delta[green_is_max]) + 2
        hue[blue_is_max] = ((red[blue_is_max] - green[blue_is_max]) / delta[blue_is_max]) + 4
        hue = hue * 60.0

        return hue, saturation, value

    def _contains_human_skin(self, image: Image.Image) -> bool:
        validation_image = image.copy()
        validation_image.thumbnail((256, 256))
        image_array = np.array(validation_image, dtype=np.uint8)

        if image_array.size == 0:
            return False

        red = image_array[..., 0].astype(np.int16)
        green = image_array[..., 1].astype(np.int16)
        blue = image_array[..., 2].astype(np.int16)

        rgb_mask = (
            (red > 95)
            & (green > 40)
            & (blue > 20)
            & ((np.maximum.reduce([red, green, blue]) - np.minimum.reduce([red, green, blue])) > 15)
            & (np.abs(red - green) > 15)
            & (red > green)
            & (red > blue)
        )

        # Dermoscopy and close-up lesion images often have pale, low-saturation skin.
        relaxed_rgb_mask = (
            (red > 110)
            & (green > 85)
            & (blue > 70)
            & (red >= green - 10)
            & (green >= blue - 15)
            & ((red - blue) > 10)
        )

        y = (0.299 * red) + (0.587 * green) + (0.114 * blue)
        cb = 128 - (0.168736 * red) - (0.331264 * green) + (0.5 * blue)
        cr = 128 + (0.5 * red) - (0.418688 * green) - (0.081312 * blue)
        ycbcr_mask = (y > 40) & (cb >= 77) & (cb <= 140) & (cr >= 133) & (cr <= 180)

        hue, saturation, value = self._rgb_to_hsv(image_array)
        hsv_mask = (hue <= 65) & (saturation >= 0.02) & (saturation <= 0.75) & (value >= 0.2)

        skin_mask = (rgb_mask | relaxed_rgb_mask) & ycbcr_mask & hsv_mask

        skin_ratio = float(np.mean(skin_mask))
        height, width = skin_mask.shape
        center_mask = skin_mask[height // 4:(3 * height) // 4, width // 4:(3 * width) // 4]
        center_ratio = float(np.mean(center_mask)) if center_mask.size else 0.0

        return skin_ratio >= 0.03 and center_ratio >= 0.03

    def predict(self, image_bytes: bytes) -> dict:
        self.load()
        if self._model is None or self._idx_to_label is None:
            raise RuntimeError("Inference service failed to initialize.")

        image = self._load_image(image_bytes)
        if not self._contains_human_skin(image):
            raise ValueError(self.INVALID_IMAGE_MESSAGE)

        model_image = image.resize((self.settings.image_size, self.settings.image_size))
        image_array = np.array(model_image, dtype=np.float32)
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
