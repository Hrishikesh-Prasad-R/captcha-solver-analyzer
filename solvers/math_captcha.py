"""
Math CAPTCHA solver using EasyOCR with OpenCV preprocessing.
Falls back to Gemini TEXT-ONLY for difficult cases.
Returns results with confidence state classification.
"""

import os
import re
import logging
from typing import Optional, Dict, Any

import cv2

from app_config import settings
from models.confidence_state import ConfidenceState, classify_state
from .gemini import gemini_solve_math
from .ocr_reader import reader

# Configure module logger
logger = logging.getLogger(__name__)

READER = reader


def preprocess_image(image_path: str):
    """Preprocess image for better math expression OCR."""
    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"Image not found or unreadable: {image_path}")

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.convertScaleAbs(gray, alpha=1.5, beta=0)
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    denoised = cv2.medianBlur(thresh, 3)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    opened = cv2.morphologyEx(denoised, cv2.MORPH_OPEN, kernel)
    return opened


def clean_expression(expr: str) -> str:
    """Clean OCR output to extract a valid math expression."""
    replacements = {
        "O": "0", "o": "0", "l": "1", "I": "1",
        "?": "", "=": "", ".": "",
    }
    for k, v in replacements.items():
        expr = expr.replace(k, v)
    expr = re.sub(r"[^0-9+\-*/().]", "", expr)
    return expr.strip()


def safe_eval(expression: str) -> Optional[float]:
    """Safely evaluate a math expression."""
    if not expression:
        return None
    if not re.match(r"^[0-9+\-*/(). ]+$", expression):
        return None
    try:
        return eval(expression)
    except Exception:
        return None


def calculate_difficulty(expression: Optional[str], avg_conf: float, fallback_used: bool) -> float:
    """Calculate difficulty score based on expression complexity."""
    if not expression:
        return 10.0
    length_score = min(len(expression) / 10, 5)
    op_count = sum(expression.count(op) for op in "+-*/")
    op_score = min(op_count, 3)
    conf_score = (1 - avg_conf) * 2
    difficulty = length_score + op_score + conf_score
    if fallback_used:
        difficulty = max(difficulty, 8)
    return max(1.0, min(round(difficulty, 1), 10.0))


def solve_math_captcha(
    image_path: str,
    use_gemini_fallback: Optional[bool] = None,
) -> Dict[str, Any]:
    """
    Solve a math CAPTCHA with state classification and complete explainability.

    Returns:
        Dict with 'expression', 'answer', 'confidence', 'source', 'difficulty', 
        'state', 'state_reason', and 'phases' (list of processing steps).
    """
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

    # Preprocess
    try:
        processed_image = preprocess_image(image_path)
        phases.append({
            "step": 2,
            "phase": "Preprocessing",
            "action": "Apply image transformations",
            "details": [
                "Convert to grayscale (BGR → GRAY)",
                "Increase contrast (alpha=1.5)",
                "Apply Otsu's thresholding",
                "Denoise with median blur (kernel=3)",
                "Morphological opening (2x2 kernel)"
            ],
            "output": "Preprocessed binary image ready for OCR",
            "status": "success"
        })
    except FileNotFoundError:
        raise
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
            "expression": None,
            "answer": None,
            "source": "easyocr",
            "error": f"Preprocessing failed: {e}",
            "difficulty": 10,
            "state": state.value,
            "state_reason": reason,
            "phases": phases,
        }

    # Run OCR
    try:
        ocr_results = READER.readtext(processed_image, detail=1)
        raw_texts = [text for _, text, conf in ocr_results]
        confidences = [conf for _, text, conf in ocr_results if text.strip()]
        phases.append({
            "step": 3,
            "phase": "OCR Recognition",
            "action": "EasyOCR text extraction",
            "model": "EasyOCR (CPU mode)",
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
        logger.info("OCR returned no results for math CAPTCHA")
        phases.append({
            "step": 4,
            "phase": "Result",
            "action": "No text detected",
            "output": "OCR could not extract any text from the image",
            "status": "failed"
        })
        state, reason = classify_state(None, has_output=False, error="OCR failed")
        return {
            "expression": None,
            "answer": None,
            "confidence": None,
            "source": "easyocr",
            "difficulty": 10,
            "state": state.value,
            "state_reason": reason,
            "phases": phases,
        }

    # Extract expression
    expression_parts = [text for _, text, conf in ocr_results if text.strip()]
    raw_expression = "".join(expression_parts).replace(" ", "")
    expression = clean_expression(raw_expression)
    
    phases.append({
        "step": 4,
        "phase": "Expression Cleaning",
        "action": "Normalize OCR output to valid math expression",
        "input": raw_expression,
        "transformations": [
            "Join all detected text parts",
            "Remove spaces",
            "Replace confusable chars (O→0, l→1, I→1)",
            "Remove invalid characters (keep 0-9, +, -, *, /, (, ))"
        ],
        "output": expression,
        "status": "success"
    })
    
    avg_conf = (
        sum(conf for _, text, conf in ocr_results if text.strip()) / len(expression_parts)
        if expression_parts else 0.0
    )

    fallback_used = False

    # Check for valid operators
    if not any(op in expression for op in "+-*/"):
        logger.info("No operators found in expression")
        phases.append({
            "step": 5,
            "phase": "Validation",
            "action": "Check for math operators",
            "input": expression,
            "output": "No valid operators (+, -, *, /) found",
            "status": "warning"
        })
        if use_gemini_fallback and expression:
            fallback_used = True
            result = gemini_solve_math(expression)
            phases.append({
                "step": 6,
                "phase": "Gemini Fallback",
                "action": "Use Gemini AI to parse expression",
                "input": expression,
                "output": f"Gemini result: {result}",
                "status": "fallback"
            })
            difficulty = calculate_difficulty(expression, avg_conf, fallback_used)
            state = ConfidenceState.REVIEW_REQUIRED
            reason = "Gemini solved, needs verification"
            return {
                "expression": result.get("expression", expression),
                "answer": result.get("answer"),
                "confidence": avg_conf,
                "source": "gemini",
                "difficulty": difficulty,
                "state": state.value,
                "state_reason": reason,
                "phases": phases,
            }
        state, reason = classify_state(avg_conf, has_output=bool(expression))
        return {
            "expression": expression,
            "answer": None,
            "confidence": avg_conf,
            "source": "easyocr",
            "difficulty": calculate_difficulty(expression, avg_conf, fallback_used),
            "state": state.value,
            "state_reason": reason,
            "phases": phases,
        }

    phases.append({
        "step": 5,
        "phase": "Validation",
        "action": "Check for math operators",
        "input": expression,
        "operators_found": [op for op in "+-*/" if op in expression],
        "output": "Valid math expression detected",
        "status": "success"
    })

    # Try local evaluation
    answer = safe_eval(expression)
    if answer is not None:
        # Build the calculation explanation
        phases.append({
            "step": 6,
            "phase": "Calculation",
            "action": "Evaluate math expression locally",
            "input": expression,
            "calculation": f"{expression} = {answer}",
            "method": "Python safe_eval (restricted to math operators)",
            "output": answer,
            "status": "success"
        })
        
        difficulty = calculate_difficulty(expression, avg_conf, fallback_used)
        state, reason = classify_state(avg_conf, has_output=True)
        
        phases.append({
            "step": 7,
            "phase": "Classification",
            "action": "Determine confidence state",
            "confidence": f"{avg_conf:.2%}",
            "thresholds": {
                "SAFE_OUTPUT": "≥85%",
                "REVIEW_REQUIRED": "50-85%",
                "NO_ACTION": "<50%"
            },
            "output": f"State: {state.value} - {reason}",
            "status": "complete"
        })
        
        return {
            "expression": expression,
            "answer": answer,
            "confidence": avg_conf,
            "source": "local",
            "difficulty": difficulty,
            "state": state.value,
            "state_reason": reason,
            "phases": phases,
        }

    # Local eval failed, try Gemini
    phases.append({
        "step": 6,
        "phase": "Calculation",
        "action": "Local evaluation failed",
        "input": expression,
        "output": "Expression could not be safely evaluated",
        "status": "failed"
    })
    
    if use_gemini_fallback:
        logger.info("Local eval failed, attempting Gemini")
        fallback_used = True
        result = gemini_solve_math(expression)
        phases.append({
            "step": 7,
            "phase": "Gemini Fallback",
            "action": "Use Gemini AI to solve expression",
            "input": expression,
            "output": f"Gemini answer: {result.get('answer')}",
            "status": "fallback"
        })
        difficulty = calculate_difficulty(expression, avg_conf, fallback_used)
        state = ConfidenceState.REVIEW_REQUIRED
        reason = "Gemini solved after local eval failed"
        return {
            "expression": expression,
            "answer": result.get("answer"),
            "confidence": avg_conf,
            "source": "gemini",
            "difficulty": difficulty,
            "state": state.value,
            "state_reason": reason,
            "phases": phases,
        }

    state, reason = classify_state(avg_conf, has_output=False, error="Local eval failed")
    phases.append({
        "step": 7,
        "phase": "Classification",
        "action": "Determine confidence state",
        "output": f"State: {state.value} - {reason}",
        "status": "complete"
    })
    return {
        "expression": expression,
        "answer": None,
        "confidence": avg_conf,
        "source": "easyocr",
        "difficulty": calculate_difficulty(expression, avg_conf, fallback_used),
        "state": state.value,
        "state_reason": reason,
        "phases": phases,
    }
