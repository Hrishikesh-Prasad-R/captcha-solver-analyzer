"""Test auto-detection classifier."""
from utils.classifier import classify_captcha
import os

# Test on all sample images
test_images = [
    ("samples/math_captcha/math1.png", "Math"),
    ("samples/math_captcha/math5.png", "Math"),
    ("samples/text_captcha/text1.png", "Text"),
    ("samples/text_captcha/text3.png", "Text"),
]

print("=" * 60)
print("TESTING CAPTCHA AUTO-DETECTION")
print("=" * 60)

for image_path, expected in test_images:
    if os.path.exists(image_path):
        result = classify_captcha(image_path)
        match = "✅" if expected in result.detected_type.value else "❌"
        print(f"\n{match} {image_path}")
        print(f"   Detected: {result.detected_type.value}")
        print(f"   Confidence: {result.confidence:.0%}")
        print(f"   Scores: {result.signals.get('scores', {})}")
    else:
        print(f"❌ File not found: {image_path}")

print("\n" + "=" * 60)
print("TEST COMPLETE")
print("=" * 60)
