"""
Confidence State System for CAPTCHA Solver.
Classifies outputs into three states based on confidence, jitter, and rules.
"""

from enum import Enum
from pydantic import BaseModel, Field
from typing import Optional, Any


class ConfidenceState(str, Enum):
    """
    Three-state confidence classification.
    
    NO_ACTION: Model doesn't know / can't solve / high jitter
    REVIEW_REQUIRED: Model gives output but is underconfident or shows jitter
    SAFE_OUTPUT: Model is confident and consistent
    """
    NO_ACTION = "NO_ACTION"
    REVIEW_REQUIRED = "REVIEW_REQUIRED"
    SAFE_OUTPUT = "SAFE_OUTPUT"


class StateClassification(BaseModel):
    """Result with confidence state classification."""
    state: ConfidenceState
    output: Optional[Any] = None
    confidence: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    jitter_score: Optional[float] = Field(
        default=None, 
        ge=0.0, 
        le=1.0,
        description="0 = no jitter (consistent), 1 = max jitter (inconsistent)"
    )
    reason: str = ""  # Why this state was assigned
    raw_result: Optional[dict] = None  # Original solver output


# Thresholds for state classification
class StateThresholds:
    """
    Configurable thresholds for state classification.
    These can be tuned based on your requirements.
    """
    # Confidence thresholds
    SAFE_CONFIDENCE_MIN: float = 0.85  # >= 85% confidence for SAFE
    REVIEW_CONFIDENCE_MIN: float = 0.50  # >= 50% but < 85% for REVIEW
    # Below 50% = NO_ACTION
    
    # Jitter thresholds (lower is better)
    SAFE_JITTER_MAX: float = 0.0  # Must be perfectly consistent for SAFE
    REVIEW_JITTER_MAX: float = 0.3  # Up to 30% inconsistency for REVIEW
    # Above 30% jitter = NO_ACTION
    
    # Number of runs for jitter detection
    JITTER_RUNS: int = 3


def classify_state(
    confidence: Optional[float],
    jitter_score: float = 0.0,
    has_output: bool = True,
    error: Optional[str] = None,
) -> tuple[ConfidenceState, str]:
    """
    Classify the output into one of three states based on rules.
    
    Rules (in priority order):
    1. If error or no output → NO_ACTION
    2. If high jitter (>30%) → NO_ACTION
    3. If low confidence (<50%) → NO_ACTION
    4. If moderate jitter (>0% but <=30%) → REVIEW_REQUIRED
    5. If moderate confidence (50-85%) → REVIEW_REQUIRED
    6. If high confidence (>=85%) AND no jitter → SAFE_OUTPUT
    
    Args:
        confidence: Model confidence (0-1)
        jitter_score: Jitter score (0-1, lower is better)
        has_output: Whether the model produced any output
        error: Error message if any
        
    Returns:
        Tuple of (state, reason)
    """
    thresholds = StateThresholds()
    
    # Rule 1: Error or no output
    if error:
        return ConfidenceState.NO_ACTION, f"Error: {error}"
    if not has_output:
        return ConfidenceState.NO_ACTION, "No output produced"
    
    # Rule 2: High jitter
    if jitter_score > thresholds.REVIEW_JITTER_MAX:
        return ConfidenceState.NO_ACTION, f"High jitter ({jitter_score:.0%}): model is inconsistent"
    
    # Rule 3: Low confidence
    if confidence is None or confidence < thresholds.REVIEW_CONFIDENCE_MIN:
        conf_str = f"{confidence:.0%}" if confidence else "None"
        return ConfidenceState.NO_ACTION, f"Low confidence ({conf_str})"
    
    # Rule 4: Moderate jitter
    if jitter_score > thresholds.SAFE_JITTER_MAX:
        return ConfidenceState.REVIEW_REQUIRED, f"Some jitter ({jitter_score:.0%}): needs review"
    
    # Rule 5: Moderate confidence
    if confidence < thresholds.SAFE_CONFIDENCE_MIN:
        return ConfidenceState.REVIEW_REQUIRED, f"Moderate confidence ({confidence:.0%})"
    
    # Rule 6: High confidence and consistent
    return ConfidenceState.SAFE_OUTPUT, f"High confidence ({confidence:.0%}), consistent output"
