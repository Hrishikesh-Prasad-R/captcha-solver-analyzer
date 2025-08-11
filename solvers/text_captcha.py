import os
import cv2
from .gemini import refine_captcha_guess
from .ocr_reader import reader

READER = reader

def preprocess_image(image_path):
    """Robust preprocessing to maximize OCR accuracy on captchas."""
    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"Image not found or unreadable: {image_path}")

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # CLAHE for local contrast enhancement
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
    gray = clahe.apply(gray)

    # Gaussian blur to reduce noise but preserve edges
    gray = cv2.GaussianBlur(gray, (3, 3), 0)

    # Adaptive threshold (binary inverse) to highlight text
    thresh = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV, 15, 9
    )

    # Morphological opening to remove noise specks
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2,2))
    cleaned = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)

    return cleaned

def normalize_text(text: str) -> (str, int):
    """Normalize OCR text by mapping confusable chars and lowercasing.
    Returns normalized text and count of replaced characters."""
    mapping = {
        'O': '0', 'o': '0',
        'I': '1', 'l': '1', '|': '1',
        'S': '5', 's': '5',
        'B': '8',
        'Z': '2', 'z': '2',
        '!': '1', '‘': '1', '”': '1'
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
    return ''.join(normalized).lower(), replacements

def solve_text_captcha(image_path, confidence_threshold=0.5, use_gemini_fallback=True):
    """
    OCR text captcha solver with strong preprocessing, normalized output,
    fallback to Gemini on low confidence or empty results,
    and difficulty scoring.
    """
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image not found: {image_path}")

    processed_img = preprocess_image(image_path)

    ocr_results = READER.readtext(processed_img, detail=1, allowlist='ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789')

    if not ocr_results:
        if use_gemini_fallback:
            return refine_captcha_guess(None)
        return {"text": None, "confidence": None, "source": "easyocr", "difficulty": 10}  # max difficulty if nothing detected

    texts = [text for _, text, conf in ocr_results if text.strip()]
    final_text = "".join(texts).replace(" ", "").strip()

    avg_conf = None
    if texts:
        avg_conf = sum(conf for _, text, conf in ocr_results if text.strip()) / len(texts)

    normalized_text, replacements = normalize_text(final_text)

    # Difficulty heuristics:
    # - Base difficulty starts with inverse confidence (lower conf = harder)
    # - Longer text = harder (scaled)
    # - More replaced chars = harder
    # Clamp difficulty between 1 and 10

    base_difficulty = 10 * (1 - (avg_conf if avg_conf is not None else 0.0))  # e.g. conf 0.8 → difficulty 2
    length_factor = min(len(normalized_text) / 6, 1) * 3  # max 3 points for length (assuming 6+ chars is hard)
    replacement_factor = min(replacements, 3)  # max 3 points for replacements

    difficulty = base_difficulty + length_factor + replacement_factor
    difficulty = max(1, min(round(difficulty, 1), 10))

    if normalized_text and (avg_conf is None or avg_conf >= confidence_threshold):
        return {
            "text": normalized_text,
            "confidence": avg_conf,
            "source": "easyocr",
            "difficulty": difficulty
        }

    if use_gemini_fallback:
        fallback_result = refine_captcha_guess(normalized_text if normalized_text else None)
        # Include difficulty score in fallback result if not present
        if "difficulty" not in fallback_result:
            fallback_result["difficulty"] = difficulty
        return fallback_result

    return {
        "text": normalized_text,
        "confidence": avg_conf,
        "source": "easyocr",
        "difficulty": difficulty
    }
