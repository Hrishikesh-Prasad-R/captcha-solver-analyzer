import os
import re
import cv2
import easyocr
from .gemini import gemini_solve_math
from .ocr_reader import reader

READER = reader

def preprocess_image(image_path):
    """Preprocess image for better OCR accuracy."""
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

def clean_expression(expr):
    """Remove noise and fix common OCR mistakes."""
    replacements = {
        "O": "0", "o": "0",
        "l": "1", "I": "1",
        "?": "", "=": "", ".": ""
    }
    for k, v in replacements.items():
        expr = expr.replace(k, v)
    # Keep only numbers, operators, and parentheses
    expr = re.sub(r"[^0-9+\-*/().]", "", expr)
    return expr.strip()

def safe_eval(expression):
    """Safely evaluate a math expression with basic validation."""
    if not re.match(r"^[0-9+\-*/(). ]+$", expression):
        return None
    try:
        return eval(expression)
    except Exception:
        return None

def calculate_difficulty(expression, avg_conf, fallback_used):
    """Heuristic difficulty score from 1 (easy) to 10 (hard)."""
    if not expression:
        return 10  # no expression = very hard
    
    length_score = min(len(expression) / 10, 5)  # max 5 pts for length
    op_count = sum(expression.count(op) for op in "+-*/")
    op_score = min(op_count, 3)  # max 3 pts for operators
    conf_score = (1 - avg_conf) * 2  # max 2 pts for low confidence

    difficulty = length_score + op_score + conf_score
    if fallback_used:
        difficulty = max(difficulty, 8)  # fallback means pretty hard

    difficulty = max(1, min(round(difficulty, 1), 10))
    return difficulty

def solve_math_captcha(image_path, use_gemini_fallback=True):
    """Solve a math CAPTCHA, prioritizing local solving before Gemini."""
    processed_image = preprocess_image(image_path)
    ocr_results = reader.readtext(processed_image, detail=1)

    if not ocr_results:
        if use_gemini_fallback:
            result = gemini_solve_math(None)
            result["difficulty"] = 10
            return result
        else:
            return {
                "expression": None,
                "answer": None,
                "confidence": None,
                "source": "easyocr",
                "difficulty": 10
            }

    # Extract and clean text
    expression_parts = [text for _, text, conf in ocr_results if text.strip()]
    expression = "".join(expression_parts).replace(" ", "")
    expression = clean_expression(expression)

    avg_conf = sum(conf for _, text, conf in ocr_results if text.strip()) / len(expression_parts)

    fallback_used = False

    # If no operators after cleaning
    if not any(op in expression for op in "+-*/"):
        if use_gemini_fallback:
            fallback_used = True
            result = gemini_solve_math(expression)
            result["difficulty"] = calculate_difficulty(expression, avg_conf, fallback_used)
            return result
        else:
            difficulty = calculate_difficulty(expression, avg_conf, fallback_used)
            return {
                "expression": expression,
                "answer": None,
                "confidence": avg_conf,
                "source": "easyocr",
                "difficulty": difficulty
            }

    # Try solving locally
    answer = safe_eval(expression)
    if answer is not None:
        difficulty = calculate_difficulty(expression, avg_conf, fallback_used)
        return {
            "expression": expression,
            "answer": answer,
            "confidence": avg_conf,
            "source": "easyocr",
            "difficulty": difficulty
        }

    # Local solving failed, fallback
    if use_gemini_fallback:
        fallback_used = True
        result = gemini_solve_math(expression)
        result["difficulty"] = calculate_difficulty(expression, avg_conf, fallback_used)
        return result

    difficulty = calculate_difficulty(expression, avg_conf, fallback_used)
    return {
        "expression": expression,
        "answer": None,
        "confidence": avg_conf,
        "source": "easyocr",
        "difficulty": difficulty
    }
