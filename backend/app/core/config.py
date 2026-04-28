# backend/app/core/config.py
import os
from pathlib import Path
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

# Compute the project root (where .env file lives)
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent

class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=PROJECT_ROOT / ".env",   # absolute path to .env
        env_file_encoding="utf-8",
        extra="ignore"
    )

    # Database
    DATABASE_URL: str = Field(
        "postgresql+asyncpg://travel_user:travel_pass@localhost:5432/travel_db"
    )

    # LLM
    GROQ_API_KEY: str = Field(..., min_length=1)

    # Weather API (optional)
    OPENWEATHER_API_KEY: str | None = None

    # Paths – absolute paths from project root
    MODEL_PATH: str = Field(
        default_factory=lambda: str(PROJECT_ROOT / "backend" / "models" / "best_model.pkl")
    )
    DATA_PATH: str = Field(
        default_factory=lambda: str(PROJECT_ROOT / "data" / "destinations.csv")
    )

    # Cache TTL (seconds)
    WEATHER_CACHE_TTL: int = Field(600, ge=0)

settings = Settings()