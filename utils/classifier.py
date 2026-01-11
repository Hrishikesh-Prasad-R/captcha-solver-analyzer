"""
CAPTCHA Type Classifier - Smart Auto-Detection System

Uses multiple signals to accurately classify CAPTCHA types:
1. Image analysis (dimensions, aspect ratio, color distribution)
2. Quick OCR scan (detects math operators, text patterns)
3. Edge detection (complexity analysis for object CAPTCHAs)
4. Pattern matching (common CAPTCHA characteristics)
"""

import os
import re
import logging
from typing import Tuple, Dict, Any, Optional
from enum import Enum

import cv2
import numpy as np

logger = logging.getLogger(__name__)


class CaptchaType(str, Enum):
    """Detected CAPTCHA types."""
    TEXT = "Text CAPTCHA"
    MATH = "Math CAPTCHA"
    OBJECT = "Object CAPTCHA"
    UNKNOWN = "Unknown"


class ClassificationResult:
    """Result of CAPTCHA type classification."""
    
    def __init__(self, detected_type: CaptchaType, confidence: float, signals: Dict[str, Any]):
        self.detected_type = detected_type
        self.confidence = confidence
        self.signals = signals
    
    def __repr__(self):
        return f"ClassificationResult(type={self.detected_type.value}, confidence={self.confidence:.0%})"


def analyze_image_properties(image_path: str) -> Dict[str, Any]:
    """
    Analyze basic image properties.
    
    Returns dict with:
    - dimensions, aspect_ratio
    - is_grayscale, dominant_colors
    - edge_density (complexity measure)
    """
    image = cv2.imread(image_path)
    if image is None:
        return {"error": "Could not load image"}
    
    height, width = image.shape[:2]
    aspect_ratio = width / height if height > 0 else 1.0
    
    # Check if grayscale
    if len(image.shape) == 2:
        is_grayscale = True
    else:
        b, g, r = cv2.split(image)
        is_grayscale = np.allclose(b, g, atol=10) and np.allclose(g, r, atol=10)
    
    # Edge detection for complexity analysis
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
    edges = cv2.Canny(gray, 50, 150)
    edge_density = np.sum(edges > 0) / (width * height)
    
    # Color variance (high variance = likely object CAPTCHA)
    if len(image.shape) == 3:
        color_std = np.std(image)
    else:
        color_std = np.std(gray)
    
    # Contour analysis
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    num_contours = len(contours)
    
    return {
        "width": width,
        "height": height,
        "aspect_ratio": aspect_ratio,
        "is_grayscale": is_grayscale,
        "edge_density": edge_density,
        "color_std": color_std,
        "num_contours": num_contours,
        "total_pixels": width * height,
    }


def quick_ocr_scan(image_path: str) -> Dict[str, Any]:
    """
    Perform quick OCR to detect text patterns without full processing.
    Uses a lightweight approach for speed.
    """
    try:
        # Import here to avoid circular imports
        from solvers.ocr_reader import reader
        
        image = cv2.imread(image_path)
        if image is None:
            return {"error": "Could not load image"}
        
        # Quick preprocessing
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        # Fast OCR scan
        results = reader.readtext(thresh, detail=1)
        
        if not results:
            return {
                "has_text": False,
                "text": "",
                "confidence": 0.0,
                "has_math_operators": False,
                "has_equals": False,
                "char_count": 0,
            }
        
        # Combine all detected text
        full_text = " ".join([text for _, text, _ in results])
        avg_confidence = sum([conf for _, _, conf in results]) / len(results)
        
        # Detect math patterns
        math_operators = ['+', '-', '×', 'x', '*', '÷', '/', '=']
        has_math_operators = any(op in full_text for op in ['+', '-', '*', '/', 'x', '×', '÷'])
        has_equals = '=' in full_text or '?' in full_text
        
        # Check for digit patterns typical of math CAPTCHAs
        digit_pattern = re.findall(r'\d+\s*[\+\-\*\/x×÷]\s*\d+', full_text)
        has_math_expression = len(digit_pattern) > 0
        
        # Check character types
        alpha_count = sum(1 for c in full_text if c.isalpha())
        digit_count = sum(1 for c in full_text if c.isdigit())
        
        return {
            "has_text": True,
            "text": full_text,
            "confidence": avg_confidence,
            "has_math_operators": has_math_operators,
            "has_equals": has_equals,
            "has_math_expression": has_math_expression,
            "char_count": len(full_text.replace(" ", "")),
            "alpha_count": alpha_count,
            "digit_count": digit_count,
            "alpha_ratio": alpha_count / max(len(full_text), 1),
            "digit_ratio": digit_count / max(len(full_text), 1),
        }
    except Exception as e:
        logger.warning(f"Quick OCR scan failed: {e}")
        return {
            "has_text": False,
            "text": "",
            "error": str(e),
        }


def detect_object_captcha_signals(image_path: str) -> Dict[str, Any]:
    """
    Detect signals that indicate an object-based CAPTCHA.
    
    Object CAPTCHAs typically have:
    - Larger dimensions (photos/images)
    - High color variance
    - Multiple distinct regions
    - Complex edges
    """
    image = cv2.imread(image_path)
    if image is None:
        return {"error": "Could not load image"}
    
    height, width = image.shape[:2]
    
    # Object CAPTCHAs are usually larger
    is_large = width >= 200 and height >= 200
    
    # High color variance indicates photos
    if len(image.shape) == 3:
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        hue_std = np.std(hsv[:, :, 0])
        saturation_mean = np.mean(hsv[:, :, 1])
        is_colorful = hue_std > 30 and saturation_mean > 50
    else:
        is_colorful = False
        hue_std = 0
        saturation_mean = 0
    
    # Edge complexity - objects have more complex edges
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
    edges = cv2.Canny(gray, 50, 150)
    
    # Find contours and analyze their properties
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Object CAPTCHAs have varied contour sizes
    if contours:
        areas = [cv2.contourArea(c) for c in contours]
        area_std = np.std(areas) if len(areas) > 1 else 0
        has_varied_objects = area_std > 1000
    else:
        has_varied_objects = False
        area_std = 0
    
    return {
        "is_large": is_large,
        "is_colorful": is_colorful,
        "hue_std": hue_std,
        "saturation_mean": saturation_mean,
        "area_std": area_std,
        "has_varied_objects": has_varied_objects,
        "total_pixels": width * height,
    }


def classify_captcha(image_path: str) -> ClassificationResult:
    """
    Main classification function - determines CAPTCHA type using multiple signals.
    
    Classification Logic:
    1. If has math operators/expressions → MATH
    2. If large, colorful, complex → OBJECT
    3. If has text without math → TEXT
    4. Fallback to TEXT (most common)
    
    Returns ClassificationResult with type, confidence, and all signals.
    """
    signals = {
        "image_properties": {},
        "ocr_scan": {},
        "object_signals": {},
    }
    
    # Gather all signals
    signals["image_properties"] = analyze_image_properties(image_path)
    signals["ocr_scan"] = quick_ocr_scan(image_path)
    signals["object_signals"] = detect_object_captcha_signals(image_path)
    
    # Score each type
    math_score = 0.0
    text_score = 0.0
    object_score = 0.0
    
    ocr = signals["ocr_scan"]
    img = signals["image_properties"]
    obj = signals["object_signals"]
    
    # === MATH CAPTCHA SCORING ===
    if ocr.get("has_math_operators", False):
        math_score += 0.4
    if ocr.get("has_equals", False):
        math_score += 0.2
    if ocr.get("has_math_expression", False):
        math_score += 0.3
    if ocr.get("digit_ratio", 0) > 0.5:
        math_score += 0.1
    # Math CAPTCHAs are usually small and simple
    if img.get("edge_density", 0) < 0.15:
        math_score += 0.1
    
    # === OBJECT CAPTCHA SCORING ===
    if obj.get("is_large", False):
        object_score += 0.2
    if obj.get("is_colorful", False):
        object_score += 0.3
    if obj.get("has_varied_objects", False):
        object_score += 0.2
    if img.get("edge_density", 0) > 0.2:
        object_score += 0.15
    if img.get("color_std", 0) > 60:
        object_score += 0.15
    # Large pixel count often means photos
    if img.get("total_pixels", 0) > 100000:
        object_score += 0.1
    # No clear text found
    if not ocr.get("has_text", False):
        object_score += 0.2
    
    # === TEXT CAPTCHA SCORING ===
    if ocr.get("has_text", False) and not ocr.get("has_math_operators", False):
        text_score += 0.4
    if ocr.get("alpha_ratio", 0) > 0.5:
        text_score += 0.2
    if ocr.get("char_count", 0) >= 4 and ocr.get("char_count", 0) <= 10:
        text_score += 0.15
    # Text CAPTCHAs are usually smaller
    if img.get("total_pixels", 0) < 50000:
        text_score += 0.1
    # Usually grayscale or limited colors
    if img.get("is_grayscale", False):
        text_score += 0.1
    
    # Normalize scores
    total = math_score + text_score + object_score
    if total == 0:
        total = 1.0
    
    math_score /= total
    text_score /= total
    object_score /= total
    
    # Determine winner
    scores = {
        CaptchaType.MATH: math_score,
        CaptchaType.TEXT: text_score,
        CaptchaType.OBJECT: object_score,
    }
    
    detected_type = max(scores, key=scores.get)
    confidence = scores[detected_type]
    
    # Add scores to signals for explainability
    signals["scores"] = {
        "math": round(math_score, 3),
        "text": round(text_score, 3),
        "object": round(object_score, 3),
    }
    signals["detected_type"] = detected_type.value
    signals["confidence"] = round(confidence, 3)
    
    logger.info(f"Classified as {detected_type.value} with {confidence:.0%} confidence")
    
    return ClassificationResult(
        detected_type=detected_type,
        confidence=confidence,
        signals=signals
    )


def get_captcha_type_name(result: ClassificationResult) -> str:
    """Get the display name for the detected CAPTCHA type."""
    return result.detected_type.value
