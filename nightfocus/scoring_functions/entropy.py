from typing import Tuple, Union

import cv2
import numpy as np
from scipy.stats import entropy


def _calculate_local_entropy(image: np.ndarray, window_size: int = 32) -> np.ndarray:
    """
    Calculate local entropy for each window in the image.

    Parameters:
    - image: Input image as numpy array
    - window_size: Size of the sliding window

    Returns:
    - Array of local entropy values
    """
    height, width = image.shape
    entropy_map = np.zeros((height, width))

    # Slide window over image
    for i in range(height - window_size):
        for j in range(width - window_size):
            window = image[i : i + window_size, j : j + window_size]

            # Calculate histogram and normalize
            hist, _ = np.histogram(window.ravel(), bins=256, range=(0, 256))
            hist = hist / hist.sum()

            # Calculate entropy for this window
            ent = entropy(hist, base=2)
            entropy_map[i + window_size // 2, j + window_size // 2] = ent

    return entropy_map


def score(star_image: np.ndarray) -> float:
    """
    Evaluate focus quality of a star image using entropy.

    Parameters:
    - star_image: Input star image as numpy array

    Returns:
    - Focus score (higher is better)
    - Entropy map for visualization
    """
    # Convert to grayscale if necessary
    if len(star_image.shape) == 3:
        gray = cv2.cvtColor(star_image, cv2.COLOR_BGR2GRAY)
    else:
        gray = star_image.copy()

    # Apply local entropy calculation
    entropy_map = _calculate_local_entropy(gray)

    # Find peak entropy values (likely star centers)
    threshold = np.percentile(entropy_map, 95)
    star_points = entropy_map > threshold

    # Calculate average entropy of bright points
    focus_score = np.mean(entropy_map[star_points])

    return focus_score
