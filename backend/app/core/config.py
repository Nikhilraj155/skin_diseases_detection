from functools import lru_cache
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    app_name: str = "Skin Disease Detection API"
    app_version: str = "1.0.0"
    api_prefix: str = "/api/v1"
    debug: bool = True

    mongo_uri: str = Field(default="mongodb://localhost:27017")
    mongo_db_name: str = Field(default="skin_disease_app")

    grok_api_key: str = Field(default="")
    grok_model: str = Field(default="grok-2-latest")
    grok_base_url: str = Field(default="https://api.x.ai/v1")

    frontend_origin: str = Field(default="http://localhost:5173")

    model_path: Path = Field(default=Path("trained_model/best_model.h5"))
    labels_path: Path = Field(default=Path("trained_model/labels.json"))
    image_size: int = Field(default=224)
    max_upload_size_mb: int = Field(default=10)

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )


@lru_cache
def get_settings() -> Settings:
    return Settings()
