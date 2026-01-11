"""
Test script to verify CAPTCHA solvers on sample images.
Tests math and text captcha solvers with full explainability.
"""
import os
import json

# Add project root to path
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from solvers.math_captcha import solve_math_captcha
from solvers.text_captcha import solve_text_captcha


def print_phases(phases: list):
    """Pretty print the phases with step-by-step explanation."""
    print("\n  📋 PROCESSING PHASES:")
    print("  " + "-" * 56)
    for phase in phases:
        step = phase.get("step", "?")
        phase_name = phase.get("phase", "Unknown")
        action = phase.get("action", "")
        status = phase.get("status", "")
        
        # Status emoji
        status_emoji = {
            "success": "✅",
            "failed": "❌",
            "warning": "⚠️",
            "complete": "🏁",
            "fallback": "🔄",
            "attempting": "⏳",
            "no_detection": "🔍",
            "fallback_failed": "⚠️"
        }.get(status, "•")
        
        print(f"  Step {step}: [{phase_name}] {status_emoji}")
        print(f"         Action: {action}")
        
        # Show input/output
        if "input" in phase and phase["input"]:
            inp = phase["input"]
            if isinstance(inp, str) and len(inp) > 50:
                inp = inp[:50] + "..."
            print(f"         Input: {inp}")
        
        if "details" in phase:
            print("         Details:")
            for detail in phase["details"]:
                print(f"           • {detail}")
        
        if "transformations" in phase:
            print("         Transformations:")
            for t in phase["transformations"]:
                print(f"           • {t}")
        
        if "char_replacements" in phase:
            print(f"         Char replacements: {', '.join(phase['char_replacements'])}")
        
        if "raw_detections" in phase:
            print("         Raw OCR detections:")
            for det in phase["raw_detections"]:
                print(f"           • '{det['text']}' (conf: {det['confidence']})")
        
        if "calculation" in phase:
            print(f"         🔢 Calculation: {phase['calculation']}")
        
        if "factors" in phase:
            print("         Factors:")
            for k, v in phase["factors"].items():
                print(f"           • {k}: {v}")
        
        if "thresholds" in phase:
            print("         Thresholds:")
            for k, v in phase["thresholds"].items():
                print(f"           • {k}: {v}")
        
        if "output" in phase:
            out = phase["output"]
            if isinstance(out, str) and len(out) > 60:
                out = out[:60] + "..."
            print(f"         Output: {out}")
        
        print()


def test_math_captchas():
    """Test all math captcha samples."""
    print("\n" + "=" * 60)
    print("🔢 TESTING MATH CAPTCHA SOLVER")
    print("=" * 60)
    
    samples_dir = os.path.join(os.path.dirname(__file__), "samples", "math_captcha")
    
    if not os.path.exists(samples_dir):
        print(f"Error: Samples directory not found: {samples_dir}")
        return
    
    image_files = sorted([f for f in os.listdir(samples_dir) if f.endswith(('.png', '.jpg', '.jpeg'))])
    
    if not image_files:
        print("No image files found in math_captcha samples")
        return
    
    print(f"\nFound {len(image_files)} math captcha images\n")
    
    for img_file in image_files:
        img_path = os.path.join(samples_dir, img_file)
        print(f"\n{'=' * 60}")
        print(f"📷 Processing: {img_file}")
        print("=" * 60)
        try:
            result = solve_math_captcha(img_path)
            
            # Summary
            print(f"\n  📊 RESULT SUMMARY:")
            print(f"  Expression: {result.get('expression', 'N/A')}")
            print(f"  Answer: {result.get('answer', 'N/A')}")
            print(f"  Confidence: {result.get('confidence', 'N/A'):.2%}" if result.get('confidence') else "  Confidence: N/A")
            print(f"  Source: {result.get('source', 'N/A')}")
            print(f"  State: {result.get('state', 'N/A')}")
            print(f"  State Reason: {result.get('state_reason', 'N/A')}")
            print(f"  Difficulty: {result.get('difficulty', 'N/A')}")
            
            # Phases
            if "phases" in result:
                print_phases(result["phases"])
            
        except Exception as e:
            print(f"  Error: {e}")


def test_text_captchas():
    """Test all text captcha samples."""
    print("\n" + "=" * 60)
    print("📝 TESTING TEXT CAPTCHA SOLVER")
    print("=" * 60)
    
    samples_dir = os.path.join(os.path.dirname(__file__), "samples", "text_captcha")
    
    if not os.path.exists(samples_dir):
        print(f"Error: Samples directory not found: {samples_dir}")
        return
    
    image_files = sorted([f for f in os.listdir(samples_dir) if f.endswith(('.png', '.jpg', '.jpeg'))])
    
    if not image_files:
        print("No image files found in text_captcha samples")
        return
    
    print(f"\nFound {len(image_files)} text captcha images\n")
    
    for img_file in image_files:
        img_path = os.path.join(samples_dir, img_file)
        print(f"\n{'=' * 60}")
        print(f"📷 Processing: {img_file}")
        print("=" * 60)
        try:
            result = solve_text_captcha(img_path)
            
            # Summary
            print(f"\n  📊 RESULT SUMMARY:")
            print(f"  Text: {result.get('text', 'N/A')}")
            print(f"  OCR Text: {result.get('ocr_text', 'N/A')}")
            print(f"  Confidence: {result.get('confidence', 'N/A'):.2%}" if result.get('confidence') else "  Confidence: N/A")
            print(f"  Source: {result.get('source', 'N/A')}")
            print(f"  State: {result.get('state', 'N/A')}")
            print(f"  State Reason: {result.get('state_reason', 'N/A')}")
            print(f"  Difficulty: {result.get('difficulty', 'N/A')}")
            
            # Phases
            if "phases" in result:
                print_phases(result["phases"])
            
        except Exception as e:
            print(f"  Error: {e}")


if __name__ == "__main__":
    print("🚀 Starting CAPTCHA Solver Verification with Full Explainability...")
    test_math_captchas()
    test_text_captchas()
    print("\n" + "=" * 60)
    print("✅ VERIFICATION COMPLETE")
    print("=" * 60)
