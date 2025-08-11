import os
from ultralytics import YOLO
from .gemini import refine_captcha_guess

MODEL_PATH = r"C:/Users/DELL/Downloads/yolov8s.pt"

try:
    model = YOLO(MODEL_PATH)
    print("✅ YOLOv8 model loaded successfully.")
except Exception as e:
    raise RuntimeError(f"❌ Failed to load YOLOv8 model: {e}")

def detect_objects(image_path, conf_threshold=0.5):
    """
    Detect objects in an image using YOLOv8.
    Falls back to Gemini if detection fails or returns empty.
    Returns a dict with:
        - 'objects': list of dicts { 'name': str, 'confidence': float }
        - 'source': 'yolov8' or 'gemini'
        - 'difficulty': float (1 to 10)
        - optionally 'error' if fallback triggered
    """
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"❌ Image not found: {image_path}")

    try:
        results = model(image_path)
    except Exception as e:
        return {
            "objects": [],
            "source": "yolov8",
            "error": f"YOLO inference failed: {str(e)}",
            "difficulty": 10  # Max difficulty on error
        }

    detected_objects = []  # Will store dicts with name and confidence
    confidences = []

    for box in results[0].boxes:
        cls_id = int(box.cls[0])
        label = model.names[cls_id]
        conf = float(box.conf[0])

        if conf >= conf_threshold:
            detected_objects.append({
                "name": label,
                "confidence": round(conf, 3)  # Rounded for nicer display
            })
            confidences.append(conf)

    if detected_objects:
        # Difficulty heuristic:
        # Fewer objects detected → harder (max 6 points)
        # Lower average confidence → harder (max 3 points)
        # Base difficulty at 1 (easy)
        count = len(detected_objects)
        avg_conf = sum(confidences) / len(confidences) if confidences else 0

        difficulty = 1
        difficulty += max(0, 6 - count)  # fewer objects → +difficulty
        difficulty += 3 * (1 - avg_conf)  # low confidence → +difficulty
        difficulty = max(1, min(round(difficulty, 1), 10))

        return {
            "objects": detected_objects,
            "source": "yolov8",
            "difficulty": difficulty
        }

    # No detected objects, fallback triggered: max difficulty 9-10
    fallback_guess = refine_captcha_guess(None)
    difficulty = 10
    if fallback_guess.get("refined"):
        difficulty = 9

    # Format fallback guesses into same dict structure
    fallback_objects = []
    for obj in fallback_guess.get("refined") or []:
        fallback_objects.append({
            "name": obj,
            "confidence": None  # No confidence from fallback
        })

    return {
        "objects": fallback_objects,
        "source": "gemini",
        "error": fallback_guess.get("error"),
        "difficulty": difficulty
    }
