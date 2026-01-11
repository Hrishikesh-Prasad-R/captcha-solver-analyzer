"""
Utility functions for determinism and reproducibility.
Ensures identical outputs for identical inputs across runs.
"""

import random
import os

# Attempt to import numpy - it's optional but recommended
try:
    import numpy as np
    HAS_NUMPY = True
except ImportError:
    HAS_NUMPY = False

# Attempt to import torch - it's optional
try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

# Attempt to import cv2 - for OpenCV determinism
try:
    import cv2
    HAS_CV2 = True
except ImportError:
    HAS_CV2 = False


def set_seed(seed: int = 42) -> None:
    """
    Set global random seeds for reproducibility across all libraries.
    This ensures deterministic behavior for:
    - Python's random module
    - NumPy (if installed)
    - PyTorch (if installed)
    - CUDA (if available)
    - OpenCV (if installed)
    
    Args:
        seed: The seed value for all random number generators.
    """
    # Python's built-in random
    random.seed(seed)
    
    # Environment variable for hash randomization
    os.environ["PYTHONHASHSEED"] = str(seed)
    
    # NumPy
    if HAS_NUMPY:
        np.random.seed(seed)
    
    # PyTorch
    if HAS_TORCH:
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
            # For full determinism (may impact performance)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
    
    # OpenCV - set number of threads for reproducibility
    if HAS_CV2:
        cv2.setNumThreads(1)  # Single thread = deterministic
        cv2.setRNGSeed(seed)


def get_deterministic_hash(value: str) -> int:
    """
    Get a deterministic hash for a string.
    Python's built-in hash() is randomized per session.
    This provides consistent hashing across runs.
    
    Args:
        value: String to hash
        
    Returns:
        Deterministic integer hash
    """
    import hashlib
    return int(hashlib.md5(value.encode()).hexdigest(), 16)
