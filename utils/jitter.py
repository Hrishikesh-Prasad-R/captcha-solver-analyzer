"""
Jitter detection module for measuring model consistency.
Runs the same input multiple times and measures output variance.
"""

import logging
from typing import Callable, Any, List, Optional

logger = logging.getLogger(__name__)


def calculate_jitter(
    solver_func: Callable,
    image_path: str,
    num_runs: int = 3,
    extract_output: Optional[Callable[[dict], Any]] = None,
    **solver_kwargs,
) -> tuple[float, List[Any], Any]:
    """
    Run a solver multiple times and calculate jitter (output inconsistency).
    
    Jitter Score:
    - 0.0: All outputs identical (perfectly consistent)
    - 1.0: All outputs different (maximally inconsistent)
    
    Args:
        solver_func: The solver function to call
        image_path: Path to the image
        num_runs: Number of times to run the solver
        extract_output: Function to extract the relevant output from result dict
                       Defaults to extracting 'text' or 'answer' or 'objects'
        **solver_kwargs: Additional arguments for the solver
        
    Returns:
        Tuple of (jitter_score, list_of_outputs, most_common_output)
    """
    if num_runs < 2:
        # Can't measure jitter with less than 2 runs
        result = solver_func(image_path, **solver_kwargs)
        output = _extract_default_output(result) if extract_output is None else extract_output(result)
        return 0.0, [output], output
    
    outputs = []
    
    for i in range(num_runs):
        try:
            result = solver_func(image_path, **solver_kwargs)
            if extract_output:
                output = extract_output(result)
            else:
                output = _extract_default_output(result)
            outputs.append(output)
            logger.debug(f"Jitter run {i+1}/{num_runs}: {output}")
        except Exception as e:
            logger.warning(f"Jitter run {i+1} failed: {e}")
            outputs.append(None)
    
    # Calculate jitter score
    jitter_score = _calculate_jitter_score(outputs)
    
    # Find most common output
    most_common = _get_most_common(outputs)
    
    return jitter_score, outputs, most_common


def _extract_default_output(result: dict) -> Any:
    """Extract the primary output from a solver result dict."""
    if isinstance(result, dict):
        # Priority order for output extraction
        for key in ['text', 'refined', 'answer', 'expression', 'objects']:
            if key in result and result[key]:
                return result[key]
    return result


def _calculate_jitter_score(outputs: List[Any]) -> float:
    """
    Calculate jitter score from a list of outputs.
    
    Uses simple matching: count how many outputs differ from the mode.
    """
    if not outputs or len(outputs) < 2:
        return 0.0
    
    # Filter out None values
    valid_outputs = [o for o in outputs if o is not None]
    if not valid_outputs:
        return 1.0  # All failed = max jitter
    
    # Find most common
    mode = _get_most_common(valid_outputs)
    
    # Count matches
    matches = sum(1 for o in valid_outputs if _outputs_equal(o, mode))
    
    # Jitter = (total - matches) / total
    jitter = (len(valid_outputs) - matches) / len(valid_outputs)
    
    return jitter


def _outputs_equal(a: Any, b: Any) -> bool:
    """Compare two outputs for equality, handling different types."""
    if type(a) != type(b):
        return False
    if isinstance(a, str):
        return a.lower().strip() == b.lower().strip()
    if isinstance(a, (list, tuple)):
        if len(a) != len(b):
            return False
        return all(_outputs_equal(x, y) for x, y in zip(a, b))
    return a == b


def _get_most_common(items: List[Any]) -> Any:
    """Get the most common item from a list."""
    if not items:
        return None
    
    # For simple types, use counting
    from collections import Counter
    
    # Convert to hashable for counting
    hashable_items = []
    for item in items:
        if isinstance(item, (list, dict)):
            hashable_items.append(str(item))
        else:
            hashable_items.append(item)
    
    counter = Counter(hashable_items)
    most_common_str = counter.most_common(1)[0][0]
    
    # Return the original item (not the string version)
    for i, h in enumerate(hashable_items):
        if h == most_common_str:
            return items[i]
    
    return items[0]
