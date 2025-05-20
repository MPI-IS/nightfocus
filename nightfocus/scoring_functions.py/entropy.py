import numpy as np
from scipy.ndimage import gaussian_filter


def _calculate_entropy(image: np.ndarray) -> float:
    """
    Calculate the entropy of an image
    """
    hist, _ = np.histogram(image.ravel(), bins=256)
    hist = hist / hist.sum()  # Normalize to probabilities
    hist = hist[hist != 0]  # Avoid log(0)
    return -np.sum(hist * np.log2(hist))


def score(image: np.ndarray, sigma: float = 2.0) -> float:
    """
    Calculate a focus score for a night sky image based on local entropy

    Args:
        image: Input image as numpy array
        sigma: Standard deviation for local entropy calculation

    Returns:
        Focus score (higher is better)
    """
    if not isinstance(image, np.ndarray):
        raise TypeError("Input must be a numpy array")

    # Convert to float and normalize to [0, 1] range
    image = image.astype(np.float32)
    image = (image - image.min()) / (image.max() - image.min())

    # Calculate local entropy using Gaussian filter
    blurred = gaussian_filter(image, sigma=sigma)
    local_entropy = _calculate_entropy(blurred)

    # Normalize score to [0, 1] range
    max_entropy = np.log2(256)  # Maximum possible entropy for 8-bit image
    score = local_entropy / max_entropy

    return score
