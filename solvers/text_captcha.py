"""
Text CAPTCHA solver using EasyOCR with OpenCV preprocessing.
Falls back to Gemini TEXT-ONLY for difficult cases.
Returns results with confidence state classification.
"""

import os
import logging
from typing import Optional, Tuple, Dict, Any

import cv2

from app_config import settings
from models.confidence_state import (
    ConfidenceState,
    StateClassification,
    classify_state,
)
from .gemini import refine_captcha_guess
from .ocr_reader import reader

# Configure module logger
logger = logging.getLogger(__name__)

# Use the shared EasyOCR reader
READER = reader


def preprocess_image(image_path: str):
    """
    Robust preprocessing to maximize OCR accuracy on CAPTCHAs.
    Uses deterministic parameters for reproducibility.
    """
    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"Image not found or unreadable: {image_path}")

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    gray = clahe.apply(gray)
    gray = cv2.GaussianBlur(gray, (3, 3), 0)
    thresh = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 15, 9
    )
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    cleaned = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
    return cleaned


def normalize_text(text: str) -> Tuple[str, int]:
    """Normalize OCR text by mapping confusable characters."""
    mapping = {
        "O": "0", "o": "0", "I": "1", "l": "1", "|": "1",
        "S": "5", "s": "5", "B": "8", "Z": "2", "z": "2",
        "!": "1", "'": "1", '"': "1",
    }
    text = text.strip()
    normalized = []
    replacements = 0
    for ch in text:
        if ch in mapping:
            normalized.append(mapping[ch])
            replacements += 1
        else:
            normalized.append(ch)
    return "".join(normalized).lower(), replacements


def solve_text_captcha(
    image_path: str,
    confidence_threshold: Optional[float] = None,
    use_gemini_fallback: Optional[bool] = None,
) -> Dict[str, Any]:
    """
    OCR text CAPTCHA solver with state classification and complete explainability.

    Returns:
        Dict with 'text', 'confidence', 'source', 'difficulty', 'state', 'state_reason',
        and 'phases' (list of processing steps).
    """
    if confidence_threshold is None:
        confidence_threshold = settings.ocr_confidence_threshold
    if use_gemini_fallback is None:
        use_gemini_fallback = settings.use_gemini_fallback

    # Initialize phases list for explainability
    phases = []
    
    phases.append({
        "step": 1,
        "phase": "Input",
        "action": "Load image",
        "input": image_path,
        "output": f"Image loaded from: {os.path.basename(image_path)}",
        "status": "success"
    })

    if not os.path.exists(image_path):
        phases.append({
            "step": 2,
            "phase": "Validation",
            "action": "Check file exists",
            "output": "File not found",
            "status": "failed"
        })
        raise FileNotFoundError(f"Image not found: {image_path}")

    # Preprocess
    try:
        processed_img = preprocess_image(image_path)
        phases.append({
            "step": 2,
            "phase": "Preprocessing",
            "action": "Apply image transformations",
            "details": [
                "Convert to grayscale (BGR → GRAY)",
                "Apply CLAHE (clipLimit=3.0, tileGridSize=8x8)",
                "Gaussian blur (3x3 kernel)",
                "Adaptive thresholding (GAUSSIAN_C, block=15, C=9)",
                "Morphological opening (2x2 kernel)"
            ],
            "output": "Preprocessed binary image ready for OCR",
            "status": "success"
        })
    except Exception as e:
        logger.exception("Image preprocessing failed")
        phases.append({
            "step": 2,
            "phase": "Preprocessing",
            "action": "Apply image transformations",
            "output": f"FAILED: {e}",
            "status": "failed"
        })
        state, reason = classify_state(None, has_output=False, error=str(e))
        return {
            "text": None,
            "source": "easyocr",
            "error": f"Preprocessing failed: {e}",
            "difficulty": 10,
            "state": state.value,
            "state_reason": reason,
            "phases": phases,
        }

    # Run OCR
    try:
        ocr_results = READER.readtext(
            processed_img,
            detail=1,
            allowlist="ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789",
        )
        raw_texts = [text for _, text, conf in ocr_results]
        phases.append({
            "step": 3,
            "phase": "OCR Recognition",
            "action": "EasyOCR text extraction",
            "model": "EasyOCR (CPU mode)",
            "allowlist": "A-Z, a-z, 0-9 (alphanumeric only)",
            "raw_detections": [
                {"text": text, "confidence": f"{conf:.2%}"} 
                for _, text, conf in ocr_results
            ],
            "output": f"Detected {len(ocr_results)} text region(s): {raw_texts}",
            "status": "success" if ocr_results else "no_detection"
        })
    except Exception as e:
        logger.exception("EasyOCR failed")
        phases.append({
            "step": 3,
            "phase": "OCR Recognition",
            "action": "EasyOCR text extraction",
            "output": f"FAILED: {e}",
            "status": "failed"
        })
        ocr_results = []

    # Handle empty results
    if not ocr_results:
        logger.info("OCR returned no results")
        phases.append({
            "step": 4,
            "phase": "Result",
            "action": "No text detected",
            "output": "OCR could not extract any text from the image",
            "status": "failed"
        })
        if use_gemini_fallback:
            # Can't use Gemini without text input
            state, reason = classify_state(None, has_output=False, error="OCR failed, no text for Gemini")
            return {
                "text": None,
                "confidence": None,
                "source": "easyocr",
                "difficulty": 10,
                "state": state.value,
                "state_reason": reason,
                "phases": phases,
            }
        state, reason = classify_state(None, has_output=False)
        return {
            "text": None,
            "confidence": None,
            "source": "easyocr",
            "difficulty": 10,
            "state": state.value,
            "state_reason": reason,
            "phases": phases,
        }

    # Extract text and confidence
    texts = [text for _, text, conf in ocr_results if text.strip()]
    final_text = "".join(texts).replace(" ", "").strip()
    
    avg_conf = None
    if texts:
        avg_conf = sum(conf for _, text, conf in ocr_results if text.strip()) / len(texts)

    normalized_text, replacements = normalize_text(final_text)
    
    # Build normalization explanation
    char_mappings_applied = []
    for char in final_text:
        mapping = {
            "O": "0", "o": "0", "I": "1", "l": "1", "|": "1",
            "S": "5", "s": "5", "B": "8", "Z": "2", "z": "2",
            "!": "1", "'": "1", '"': "1",
        }
        if char in mapping:
            char_mappings_applied.append(f"'{char}' → '{mapping[char]}'")
    
    phases.append({
        "step": 4,
        "phase": "Text Normalization",
        "action": "Clean and normalize OCR output",
        "input": final_text,
        "transformations": [
            "Join all detected text parts",
            "Remove spaces",
            "Apply character mappings for confusable chars",
            "Convert to lowercase"
        ],
        "char_replacements": char_mappings_applied if char_mappings_applied else ["None needed"],
        "replacement_count": replacements,
        "output": normalized_text,
        "status": "success"
    })

    # Calculate difficulty
    base_difficulty = 10 * (1 - (avg_conf if avg_conf is not None else 0.0))
    length_factor = min(len(normalized_text) / 6, 1) * 3
    replacement_factor = min(replacements, 3)
    difficulty = base_difficulty + length_factor + replacement_factor
    difficulty = max(1, min(round(difficulty, 1), 10))
    
    phases.append({
        "step": 5,
        "phase": "Difficulty Calculation",
        "action": "Compute CAPTCHA difficulty score",
        "factors": {
            "base_difficulty": f"{base_difficulty:.2f} (from confidence)",
            "length_factor": f"{length_factor:.2f} (text length penalty)",
            "replacement_factor": f"{replacement_factor} (char replacements penalty)"
        },
        "formula": "base_difficulty + length_factor + replacement_factor",
        "output": difficulty,
        "status": "success"
    })

    # Classify state
    has_output = bool(normalized_text)
    state, reason = classify_state(avg_conf, jitter_score=0.0, has_output=has_output)

    # Try Gemini fallback for low confidence
    if use_gemini_fallback and normalized_text and avg_conf and avg_conf < confidence_threshold:
        logger.info("Attempting Gemini text refinement")
        phases.append({
            "step": 6,
            "phase": "Gemini Fallback",
            "action": "Attempt AI text refinement",
            "reason": f"Confidence ({avg_conf:.2%}) below threshold ({confidence_threshold:.2%})",
            "input": normalized_text,
            "status": "attempting"
        })
        fallback_result = refine_captcha_guess(normalized_text)
        if fallback_result.get("refined"):
            phases.append({
                "step": 7,
                "phase": "Gemini Result",
                "action": "AI refinement successful",
                "input": normalized_text,
                "output": fallback_result["refined"],
                "status": "success"
            })
            # Gemini refinement doesn't give confidence, so mark as REVIEW
            state = ConfidenceState.REVIEW_REQUIRED
            reason = "Gemini refinement applied, needs verification"
            return {
                "text": fallback_result["refined"],
                "ocr_text": normalized_text,
                "confidence": avg_conf,
                "source": "gemini",
                "difficulty": difficulty,
                "state": state.value,
                "state_reason": reason,
                "phases": phases,
            }
        else:
            phases.append({
                "step": 7,
                "phase": "Gemini Result",
                "action": "AI refinement failed or unavailable",
                "output": "Using original OCR result",
                "status": "fallback_failed"
            })

    phases.append({
        "step": 6 if not (use_gemini_fallback and normalized_text and avg_conf and avg_conf < confidence_threshold) else 8,
        "phase": "Classification",
        "action": "Determine confidence state",
        "confidence": f"{avg_conf:.2%}" if avg_conf else "None",
        "thresholds": {
            "SAFE_OUTPUT": "≥85%",
            "REVIEW_REQUIRED": "50-85%",
            "NO_ACTION": "<50%"
        },
        "output": f"State: {state.value} - {reason}",
        "status": "complete"
    })

    return {
        "text": normalized_text,
        "confidence": avg_conf,
        "source": "easyocr",
        "difficulty": difficulty,
        "state": state.value,
        "state_reason": reason,
        "phases": phases,
    }
