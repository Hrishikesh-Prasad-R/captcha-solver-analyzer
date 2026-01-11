"""
Unit tests for CAPTCHA solvers.
Run with: pytest tests/
"""

import os
import pytest
from unittest.mock import patch, MagicMock

# Import config first to set up environment
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import settings, AppSettings
from utils.determinism import set_seed
from models.captcha_result import (
    TextCaptchaResult,
    MathCaptchaResult,
    ObjectCaptchaResult,
    DetectedObject,
    BoundingBox,
    SolverSource,
)


class TestConfig:
    """Test configuration module."""

    def test_settings_loads_defaults(self):
        """Verify default settings are loaded."""
        assert settings.random_seed == 42
        assert settings.ocr_confidence_threshold == 0.5
        assert settings.object_detection_threshold == 0.5

    def test_model_path_function(self):
        """Test get_model_path returns a string."""
        from config import get_model_path
        path = get_model_path()
        assert isinstance(path, str)
        assert "yolov8" in path.lower()


class TestDeterminism:
    """Test determinism utilities."""

    def test_set_seed_is_deterministic(self):
        """Verify setting seed produces deterministic random values."""
        import random
        
        set_seed(42)
        value1 = random.random()
        
        set_seed(42)
        value2 = random.random()
        
        assert value1 == value2, "Random values should be identical with same seed"

    def test_different_seeds_produce_different_values(self):
        """Verify different seeds produce different values."""
        import random
        
        set_seed(42)
        value1 = random.random()
        
        set_seed(123)
        value2 = random.random()
        
        assert value1 != value2, "Different seeds should produce different values"


class TestPydanticModels:
    """Test Pydantic model validation."""

    def test_text_captcha_result_valid(self):
        """Test valid TextCaptchaResult."""
        result = TextCaptchaResult(
            source=SolverSource.EASYOCR,
            text="ABC123",
            confidence=0.95,
            difficulty=3.5,
        )
        assert result.text == "ABC123"
        assert result.confidence == 0.95

    def test_math_captcha_result_valid(self):
        """Test valid MathCaptchaResult."""
        result = MathCaptchaResult(
            source=SolverSource.LOCAL,
            expression="5+3",
            answer="8",
            difficulty=2.0,
        )
        assert result.expression == "5+3"
        assert result.answer == "8"

    def test_object_captcha_result_valid(self):
        """Test valid ObjectCaptchaResult."""
        bbox = BoundingBox(x_min=10, y_min=20, x_max=100, y_max=200)
        obj = DetectedObject(name="car", confidence=0.89, bbox=bbox)
        result = ObjectCaptchaResult(
            source=SolverSource.YOLOV8,
            objects=[obj],
            difficulty=4.0,
        )
        assert len(result.objects) == 1
        assert result.objects[0].name == "car"

    def test_confidence_validation(self):
        """Test confidence value validation (0-1 range)."""
        # This should not raise
        result = TextCaptchaResult(
            source=SolverSource.EASYOCR,
            confidence=0.0,
        )
        assert result.confidence == 0.0
        
        result = TextCaptchaResult(
            source=SolverSource.EASYOCR,
            confidence=1.0,
        )
        assert result.confidence == 1.0

    def test_difficulty_validation(self):
        """Test difficulty value validation (1-10 range)."""
        result = TextCaptchaResult(
            source=SolverSource.EASYOCR,
            difficulty=1.0,
        )
        assert result.difficulty == 1.0
        
        result = TextCaptchaResult(
            source=SolverSource.EASYOCR,
            difficulty=10.0,
        )
        assert result.difficulty == 10.0


class TestSolverImports:
    """Test that solvers can be imported without errors."""

    def test_import_gemini(self):
        """Test gemini module imports."""
        from solvers.gemini import gemini_solve_math, refine_captcha_guess
        assert callable(gemini_solve_math)
        assert callable(refine_captcha_guess)


# Golden test placeholder - requires sample images
class TestGoldenImages:
    """
    Golden image tests for determinism verification.
    These tests use known input images with expected outputs.
    
    To add golden tests:
    1. Add sample images to tests/samples/
    2. Record expected outputs
    3. Verify outputs match on every run
    """

    @pytest.mark.skip(reason="Requires sample images in tests/samples/")
    def test_text_captcha_golden(self):
        """Test text CAPTCHA with known image."""
        pass

    @pytest.mark.skip(reason="Requires sample images in tests/samples/")
    def test_math_captcha_golden(self):
        """Test math CAPTCHA with known image."""
        pass
