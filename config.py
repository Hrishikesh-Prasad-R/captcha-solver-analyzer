"""
Configuration module using pydantic-settings for robust, validated environment management.
All configuration is centralized here for industrial-grade reliability.
"""

from pydantic_settings import BaseSettings
from pydantic import Field
from typing import Optional
import os


class AppSettings(BaseSettings):
    """
    Application settings loaded from environment variables.
    Validates all required values at startup - fail fast principle.
    """

    # API Keys
    gemini_api_key: Optional[str] = Field(default=None, alias="API_KEY")

    # Model Paths - Use relative paths for portability
    yolo_model_name: str = Field(
        default="yolov8n.pt",
        description="YOLO model filename. Will be auto-downloaded by ultralytics if not found."
    )

    # Thresholds
    ocr_confidence_threshold: float = Field(default=0.5, ge=0.0, le=1.0)
    object_detection_threshold: float = Field(default=0.5, ge=0.0, le=1.0)

    # Determinism
    random_seed: int = Field(default=42, description="Global seed for reproducibility")

    # Feature Flags
    use_gemini_fallback: bool = Field(default=True)

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        extra = "ignore"  # Ignore unknown env vars


# Singleton instance - load once at startup
settings = AppSettings()


def get_model_path() -> str:
    """
    Returns the resolved path for the YOLO model.
    Tries local 'models/' directory first, then 'solvers/', then lets ultralytics handle it.
    """
    base_dirs = [
        os.path.join(os.getcwd(), "models"),
        os.path.join(os.getcwd(), "solvers"),
        os.getcwd(),
    ]
    for base in base_dirs:
        path = os.path.join(base, settings.yolo_model_name)
        if os.path.exists(path):
            return path
    # Fallback: let ultralytics download it
    return settings.yolo_model_name
