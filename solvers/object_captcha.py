"""
Object Detection solver for CAPTCHA using YOLOv8.
Uses pretrained YOLO model (auto-downloaded by ultralytics if not found).
Returns results with confidence state classification.
"""

import os
import logging
from typing import Optional

from ultralytics import YOLO

from app_config import settings, get_model_path
from models.captcha_result import (
    ObjectCaptchaResult,
    DetectedObject,
    BoundingBox,
    SolverSource,
)
from models.confidence_state import ConfidenceState, classify_state
from .gemini import refine_captcha_guess

# Configure module logger
logger = logging.getLogger(__name__)

# --- Model Loading (Lazy Singleton) ---
_model: Optional[YOLO] = None


def _get_model() -> Optional[YOLO]:
    """Lazy-load the YOLO model."""
    global _model
    if _model is None:
        model_path = get_model_path()
        try:
            _model = YOLO(model_path)
            logger.info(f"YOLOv8 model loaded from: {model_path}")
        except Exception as e:
            logger.error(f"Failed to load YOLOv8 model: {e}")
            _model = None
    return _model


def detect_objects(
    image_path: str,
    conf_threshold: Optional[float] = None,
    use_gemini_fallback: Optional[bool] = None,
) -> dict:
    """
    Detect objects in an image using YOLOv8 with state classification.

    Returns:
        Dict with 'objects', 'source', 'difficulty', 'state', 'state_reason', optionally 'error'.
    """
    if conf_threshold is None:
        conf_threshold = settings.object_detection_threshold
    if use_gemini_fallback is None:
        use_gemini_fallback = settings.use_gemini_fallback

    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image not found: {image_path}")

    model = _get_model()
    if model is None:
        state, reason = classify_state(None, has_output=False, error="Model not loaded")
        return {
            "objects": [],
            "source": "yolov8",
            "error": "YOLO model could not be loaded",
            "difficulty": 10.0,
            "state": state.value,
            "state_reason": reason,
        }

    # Run inference
    try:
        results = model(image_path, verbose=False)
    except Exception as e:
        logger.exception("YOLO inference failed")
        state, reason = classify_state(None, has_output=False, error=str(e))
        return {
            "objects": [],
            "source": "yolov8",
            "error": f"YOLO inference failed: {e}",
            "difficulty": 10.0,
            "state": state.value,
            "state_reason": reason,
        }

    # Validate results
    if not results or not hasattr(results[0], "boxes") or results[0].boxes is None:
        state, reason = classify_state(None, has_output=False, error="No detections")
        return {
            "objects": [],
            "source": "yolov8",
            "error": "No detection results returned from YOLO",
            "difficulty": 10.0,
            "state": state.value,
            "state_reason": reason,
        }

    # Process detections
    detected_objects = []
    confidences = []

    for box in results[0].boxes:
        cls_id = int(box.cls[0])
        label = model.names[cls_id]
        conf = float(box.conf[0])
        coords = box.xyxy[0].tolist()

        if conf >= conf_threshold:
            detected_objects.append({
                "name": label,
                "confidence": round(conf, 3),
                "bbox": [int(c) for c in coords],
            })
            confidences.append(conf)

    # Return successful detection
    if detected_objects:
        avg_conf = sum(confidences) / len(confidences)
        difficulty = 1.0 + max(0, 6 - len(detected_objects)) + 3 * (1 - avg_conf)
        difficulty = max(1.0, min(round(difficulty, 1), 10.0))

        state, reason = classify_state(avg_conf, has_output=True)
        return {
            "objects": detected_objects,
            "source": "yolov8",
            "difficulty": difficulty,
            "confidence": avg_conf,
            "state": state.value,
            "state_reason": reason,
        }

    # No detections - try Gemini text fallback (won't work well without image)
    if use_gemini_fallback:
        logger.info("No objects detected, Gemini fallback not useful for object detection")
        # Gemini can't help here since we're text-only
    
    state, reason = classify_state(None, has_output=False, error="No objects detected")
    return {
        "objects": [],
        "source": "yolov8",
        "error": "No objects detected",
        "difficulty": 10.0,
        "state": state.value,
        "state_reason": reason,
    }
