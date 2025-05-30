"""
Collection of fast focus scoring functions for star images.

Each function takes a numpy array (grayscale or RGB) and returns a float score.
Higher scores typically indicate better focus, though this may vary by function.
"""

from typing import Union

import cv2
import numpy as np


def normalize_image(image: np.ndarray) -> np.ndarray:
    """Convert image to grayscale and normalize to [0, 1] range."""
    if len(image.shape) == 3:
        image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    return image.astype(np.float32) / 255.0


def variance_of_laplacian(image: np.ndarray) -> float:
    """
    Compute the variance of the Laplacian of the image.
    Higher values indicate sharper images.
    """
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    else:
        gray = image

    # Using a small kernel (3x3) for faster computation
    return cv2.Laplacian(gray, cv2.CV_64F).var()


def modified_laplacian(image: np.ndarray) -> float:
    """
    Compute the modified Laplacian focus measure.
    Faster than variance of Laplacian and works well for star fields.
    """
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    else:
        gray = image

    kernel = np.array([-1, 2, -1], dtype=np.float32)
    dx = cv2.filter2D(gray, -1, kernel.reshape(1, -1))
    dy = cv2.filter2D(gray, -1, kernel.reshape(-1, 1))
    return float(np.mean(dx**2 + dy**2))


def tenengrad(image: np.ndarray, ksize: int = 3) -> float:
    """
    Tenengrad focus measure based on gradient magnitude.
    Works well for star images and is computationally efficient.
    """
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    else:
        gray = image

    gx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=ksize)
    gy = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=ksize)
    return float(np.mean(gx**2 + gy**2))


def normalized_gray_level_variance(image: np.ndarray) -> float:
    """
    Normalized gray-level variance focus measure.
    Fast and works well for both stars and general astrophotography.
    """
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY).astype(np.float32)
    else:
        gray = image.astype(np.float32)

    mean = np.mean(gray)
    if mean < 1e-10:  # Avoid division by zero
        return 0.0
    return float(np.var(gray) / mean)


def spectral_energy(image: np.ndarray) -> float:
    """
    Compute the spectral energy in the high frequencies.
    Good for detecting sharp transitions in star fields.
    """
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    else:
        gray = image

    # Use FFT to get frequency domain representation
    f_transform = np.fft.fft2(gray.astype(float))
    f_shift = np.fft.fftshift(f_transform)
    magnitude_spectrum = np.abs(f_shift)

    # Focus on high frequencies (edges of the spectrum)
    rows, cols = gray.shape
    crow, ccol = rows // 2, cols // 2
    mask = np.ones((rows, cols), np.uint8)
    mask[crow - 10 : crow + 11, ccol - 10 : ccol + 11] = 0  # Remove low frequencies

    # Calculate energy in high frequencies
    high_freq = magnitude_spectrum * mask
    return np.sum(high_freq**2)


def brenner_gradient(image: np.ndarray) -> float:
    """
    Brenner's gradient measure.
    Simple and fast, works well for star fields.
    """
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    else:
        gray = image

    # Calculate horizontal and vertical gradients
    dx = gray[1:, :] - gray[:-1, :]
    dy = gray[:, 1:] - gray[:, :-1]

    # Sum of squared gradients
    return np.sum(dx**2) + np.sum(dy**2)


def threshold_pixel_count(image: np.ndarray, threshold: float = 0.9) -> float:
    """
    Count of pixels above a certain brightness threshold.
    Works well for star fields where in-focus stars are brighter.
    """
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    else:
        gray = image

    # Normalize to [0, 1]
    normalized = gray.astype(np.float32) / 255.0
    # Count pixels above threshold
    return float(np.sum(normalized > threshold))


def fast_entropy(image: np.ndarray, bins: int = 16) -> float:
    """
    A faster version of entropy calculation using histogram binning.
    """
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    else:
        gray = image

    # Calculate histogram
    hist, _ = np.histogram(gray, bins=bins, range=(0, 256))
    hist = hist.astype(np.float32) / (gray.shape[0] * gray.shape[1])

    # Calculate entropy (avoid log(0))
    hist = hist[hist > 0]
    return -np.sum(hist * np.log2(hist))


def wavelet_based_measure(image: np.ndarray) -> float:
    """
    A wavelet-based focus measure that's fast and effective for star fields.
    Uses the sum of absolute values of the wavelet coefficients.
    """
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    else:
        gray = image

    # Simple wavelet transform using Haar wavelet (approximated with box filters)
    # This is much faster than a full wavelet transform
    h, w = gray.shape
    cA = cv2.boxFilter(gray, -1, (2, 2))[::2, ::2]  # Approximation
    cH = cv2.boxFilter(gray, -1, (2, 2))[::2, 1::2] - cA  # Horizontal detail
    cV = cv2.boxFilter(gray, -1, (2, 2))[1::2, ::2] - cA  # Vertical detail
    cD = cv2.boxFilter(gray, -1, (2, 2))[1::2, 1::2] - cA  # Diagonal detail

    # Sum of absolute values of detail coefficients
    return np.sum(np.abs(cH)) + np.sum(np.abs(cV)) + np.sum(np.abs(cD))


def best_measure(image: np.ndarray) -> float:
    return tenengrad(image)


def second_best_measure(image: np.ndarray) -> float:
    return spectral_energy(image)


# Dictionary of all available focus measures
FOCUS_MEASURES = {
    "variance_laplacian": variance_of_laplacian,
    "modified_laplacian": modified_laplacian,
    "tenengrad": tenengrad,
    "normalized_variance": normalized_gray_level_variance,
    "spectral_energy": spectral_energy,
    "brenner_gradient": brenner_gradient,
    "threshold_count": threshold_pixel_count,
    "fast_entropy": fast_entropy,
    "wavelet_measure": wavelet_based_measure,
    "best_measure": best_measure,
    "second_best_measure": second_best_measure,
}


def get_focus_measure(name: str):
    """
    Get a focus measure function by name.

    Args:
        name: Name of the focus measure function

    Returns:
        Callable focus measure function

    Raises:
        ValueError: If the specified focus measure is not found
    """
    if name not in FOCUS_MEASURES:
        available = ", ".join(FOCUS_MEASURES.keys())
        raise ValueError(f"Unknown focus measure: {name}. Available: {available}")
    return FOCUS_MEASURES[name]
