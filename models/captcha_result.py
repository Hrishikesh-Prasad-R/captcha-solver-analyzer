"""
Pydantic models for CAPTCHA solver results.
These provide strict type safety and validation - no more untyped dictionaries.
"""

from pydantic import BaseModel, Field
from typing import Optional, List
from enum import Enum


class SolverSource(str, Enum):
    """Enum for tracking which solver produced the result."""
    EASYOCR = "easyocr"
    YOLOV8 = "yolov8"
    GEMINI = "gemini"
    LOCAL = "local"  # For math eval


class BaseCaptchaResult(BaseModel):
    """Base result model with common fields."""
    source: SolverSource
    difficulty: Optional[float] = Field(default=None, ge=1.0, le=10.0)
    error: Optional[str] = None
    elapsed_time: Optional[float] = None


class TextCaptchaResult(BaseCaptchaResult):
    """Result for text-based CAPTCHAs."""
    text: Optional[str] = None
    refined: Optional[str] = None  # Gemini-refined version
    confidence: Optional[float] = Field(default=None, ge=0.0, le=1.0)


class MathCaptchaResult(BaseCaptchaResult):
    """Result for math expression CAPTCHAs."""
    expression: Optional[str] = None
    answer: Optional[str] = None
    confidence: Optional[float] = Field(default=None, ge=0.0, le=1.0)


class BoundingBox(BaseModel):
    """Bounding box for detected objects."""
    x_min: int
    y_min: int
    x_max: int
    y_max: int


class DetectedObject(BaseModel):
    """Single detected object in an image."""
    name: str
    confidence: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    bbox: Optional[BoundingBox] = None


class ObjectCaptchaResult(BaseCaptchaResult):
    """Result for object detection CAPTCHAs."""
    objects: List[DetectedObject] = Field(default_factory=list)
