import google.generativeai as genai
from dotenv import load_dotenv
import os

load_dotenv() 
api_key = os.getenv("API_KEY")

# Store your Gemini API key here

genai.configure(api_key)

def gemini_solve_math(expression):
    """
    Ask Gemini 1.5 Flash to solve a math expression and return only the result.
    """
    if not expression:
        return {
            "expression": None,
            "answer": None,
            "source": "gemini",
            "error": "No OCR text to send"
        }

    try:
        model = genai.GenerativeModel("gemini-1.5-flash")
        prompt = f"Solve this math expression exactly and return only the numeric result: {expression}"
        response = model.generate_content(prompt)
        answer = response.text.strip()
        return {
            "expression": expression,
            "answer": answer,
            "source": "gemini"
        }
    except Exception as e:
        return {
            "expression": expression,
            "answer": None,
            "source": "gemini",
            "error": str(e)
        }


def refine_captcha_guess(ocr_output):
    """
    Use free Gemini text-only API to guess the correct CAPTCHA text from OCR output.
    Removes spaces and punctuation, keeping only alphanumeric characters.
    """
    if not ocr_output:
        return {
            "ocr_output": None,
            "refined": None,
            "source": "gemini",
            "error": "No OCR output provided"
        }

    try:
        model = genai.GenerativeModel("gemini-1.5-flash")
        prompt = (
            f'The OCR output for a CAPTCHA was: "{ocr_output}". '
            "It may contain errors due to distortion or noise. "
            "Return your best guess for the correct text, "
            "containing only alphanumeric characters, without spaces or punctuation."
        )
        response = model.generate_content(prompt)
        refined = response.text.strip()
        return {
            "ocr_output": ocr_output,
            "refined": refined,
            "source": "gemini"
        }
    except Exception as e:
        return {
            "ocr_output": ocr_output,
            "refined": None,
            "source": "gemini",
            "error": str(e)
        }
