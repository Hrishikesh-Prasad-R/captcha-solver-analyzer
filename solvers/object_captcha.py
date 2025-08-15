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
    Now also returns bounding box coordinates (x_min, y_min, x_max, y_max) as integers.
    Falls back to Gemini if detection fails or returns empty.
    Returns a dict with:
        - 'objects': list of dicts { 'name': str, 'confidence': float, 'bbox': list or None }
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
            "difficulty": 10
        }

    # Safety: Ensure detection results are valid
    if not results or not hasattr(results[0], "boxes") or results[0].boxes is None:
        return {
            "objects": [],
            "source": "yolov8",
            "error": "No detection results returned from YOLO",
            "difficulty": 10
        }

    detected_objects = []
    confidences = []

    for box in results[0].boxes:
        cls_id = int(box.cls[0])
        label = model.names[cls_id]
        conf = float(box.conf[0])
        coords = box.xyxy[0].tolist()  # [x_min, y_min, x_max, y_max]

        if conf >= conf_threshold:
            detected_objects.append({
                "name": label,
                "confidence": round(conf, 3),
                "bbox": [int(c) for c in coords]  # Integer pixel coords
            })
            confidences.append(conf)

    if detected_objects:
        count = len(detected_objects)
        avg_conf = sum(confidences) / len(confidences) if confidences else 0

        difficulty = 1
        difficulty += max(0, 6 - count)
        difficulty += 3 * (1 - avg_conf)
        difficulty = max(1, min(round(difficulty, 1), 10))

        return {
            "objects": detected_objects,
            "source": "yolov8",
            "difficulty": difficulty
        }

    # No detections → fallback
    fallback_guess = refine_captcha_guess(None)
    difficulty = 9 if fallback_guess.get("refined") else 10

    fallback_objects = [{
        "name": obj,
        "confidence": None,
        "bbox": None
    } for obj in fallback_guess.get("refined") or []]

    return {
        "objects": fallback_objects,
        "source": "gemini",
        "error": fallback_guess.get("error"),
        "difficulty": difficulty
    }
