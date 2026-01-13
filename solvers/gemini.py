"""
Gemini AI integration for CAPTCHA solving fallback.
TEXT-ONLY mode - does not send images to Gemini.
"""

import logging
from typing import Optional, Dict, Any

import google.generativeai as genai

from app_config import settings

# Configure module logger
logger = logging.getLogger(__name__)

# --- Lazy Gemini Configuration ---
_configured = False


def _ensure_configured() -> bool:
    """
    Configure Gemini API lazily. Returns True if configured, False otherwise.
    """
    global _configured
    if _configured:
        return True

    if not settings.gemini_api_key:
        logger.warning("Gemini API key not found. AI features will be unavailable.")
        return False

    try:
        genai.configure(api_key=settings.gemini_api_key)
        _configured = True
        logger.info("Gemini API configured successfully.")
        return True
    except Exception as e:
        logger.error(f"Failed to configure Gemini API: {e}")
        return False


def gemini_solve_math(expression: Optional[str] = None) -> Dict[str, Any]:
    """
    Ask Gemini to solve a math expression (TEXT ONLY).

    Args:
        expression: Math expression string to solve.

    Returns:
        Dict with 'expression', 'answer', 'source', and optionally 'error'.
    """
    if not _ensure_configured():
        return {"source": "gemini", "error": "Gemini API not configured"}

    if not expression:
        return {
            "expression": None,
            "answer": None,
            "source": "gemini",
            "error": "No expression provided",
        }

    prompt = f"Solve this math expression exactly and return only the numeric result: {expression}"

    try:
        model = genai.GenerativeModel("gemini-1.5-flash")
        response = model.generate_content(prompt)
        answer = response.text.strip()
        return {
            "expression": expression,
            "answer": answer,
            "source": "gemini",
        }
    except Exception as e:
        logger.exception("Gemini API call failed for math solving")
        return {
            "expression": expression,
            "answer": None,
            "source": "gemini",
            "error": str(e),
        }


def refine_captcha_guess(ocr_output: Optional[str] = None) -> Dict[str, Any]:
    """
    Use Gemini to refine OCR output (TEXT ONLY).
    Does NOT send images - only sends the OCR text for correction.

    Args:
        ocr_output: Raw OCR output to refine.

    Returns:
        Dict with 'ocr_output', 'refined', 'source', and optionally 'error'.
    """
    if not _ensure_configured():
        return {"source": "gemini", "error": "Gemini API not configured"}

    if not ocr_output:
        return {
            "ocr_output": None,
            "refined": None,
            "source": "gemini",
            "error": "No OCR output provided",
        }

    prompt = (
        f'The OCR output for a CAPTCHA was: "{ocr_output}". '
        "It may contain errors due to distortion or noise. "
        "Return your best guess for the correct text, "
        "containing only alphanumeric characters, without spaces or punctuation."
    )

    try:
        model = genai.GenerativeModel("gemini-1.5-flash")
        response = model.generate_content(prompt)
        refined = response.text.strip().replace(" ", "")
        return {
            "ocr_output": ocr_output,
            "refined": refined,
            "source": "gemini",
        }
    except Exception as e:
        logger.exception("Gemini API call failed for CAPTCHA refinement")
        return {
            "ocr_output": ocr_output,
            "refined": None,
            "source": "gemini",
            "error": str(e),
        }
