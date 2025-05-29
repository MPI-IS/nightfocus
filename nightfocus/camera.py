"""Camera abstraction and focus optimization."""

from __future__ import annotations

import os
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple, TypeVar, Union

import cv2
import numpy as np
import numpy.typing as npt
from skopt import gp_minimize
from skopt.space import Integer
from skopt.utils import use_named_args

from .dataset import Dataset
from .focus_metrics import best_measure

# Type variables for better type hints
T = TypeVar("T", int, float)

# Define types
Focus = int
# Type for numpy arrays (2D or 3D with color channels)
Image = Union[
    npt.NDArray[np.uint8],  # Grayscale
    npt.NDArray[np.float32],  # Normalized grayscale
    npt.NDArray[np.uint8],  # RGB/BGR
    npt.NDArray[np.float32],  # Normalized color
]
FocusMeasure = Callable[[Image], float]
FocusHistory = List[Tuple[float, float]]


class Camera(ABC):
    """Abstract base class for cameras with focus control."""

    @abstractmethod
    def take_picture(self, focus: Focus) -> Image:
        """
        Take a picture at the specified focus value.

        Args:
            focus: The focus value to use

        Returns:
            The captured image as a numpy array
        """
        pass


def optimize_focus(
    camera: "Camera",
    focus_measure: FocusMeasure,
    bounds: Tuple[Focus, Focus] = (0, 100),
    initial_points: int = 5,
    max_iter: int = 20,
    noise_level: float = 1e-6,
    random_state: Optional[int] = None,
) -> Tuple[Focus, FocusHistory]:
    """
    Find the optimal focus using Bayesian optimization with Gaussian Processes.

    Args:
        camera: The camera instance to use
        focus_measure: The focus measure function to optimize (default: best_measure)
        bounds: Tuple of (min_focus, max_focus)
        initial_points: Number of initial random points to sample
        max_iter: Maximum number of iterations
        noise_level: Expected noise level in the observations
        random_state: Random seed for reproducibility

    Returns:
        Tuple of (best_focus, history) where history is a list of (focus, score) pairs
    """
    if focus_measure is None:
        focus_measure = best_measure

    min_focus, max_focus = bounds
    history: List[Tuple[float, float]] = []

    # Define the search space
    space = [Integer(min_focus, max_focus, name="focus")]

    # Cache for already evaluated points
    eval_cache: dict[int, float] = {}

    # Objective function
    @use_named_args(space)
    def objective(focus: int) -> float:
        focus_int = int(round(focus))

        # Check cache first
        if focus_int in eval_cache:
            return -eval_cache[focus_int]  # Negative because we want to maximize

        # Take picture and evaluate focus
        img = camera.take_picture(focus_int)
        score = float(focus_measure(img))

        # Store in cache and history
        eval_cache[focus_int] = score
        history.append((float(focus_int), score))

        return -score  # Negative because scikit-optimize minimizes

    # Run the optimization
    result = gp_minimize(
        func=objective,
        dimensions=space,
        n_calls=initial_points + max_iter,
        n_initial_points=initial_points,
        noise=noise_level,
        random_state=random_state,
        n_jobs=1,
        verbose=False,
    )

    # Get the best focus and convert history to the correct format
    best_focus = Focus(int(round(result.x[0])))

    # Ensure history is sorted by focus value for better visualization
    history_sorted = sorted(history, key=lambda x: x[0])

    return best_focus, history_sorted


class SimulatedCamera(Camera):
    """
    A simulated camera for testing, which generates synthetic images with varying focus.
    The best focus is at focus=50.
    """

    def __init__(self, image_shape=(100, 100), noise_level=0.1):
        """
        Initialize the simulated camera.

        Args:
            image_shape: Shape of the output images (height, width)
            noise_level: Amount of noise to add to images
        """
        self.image_shape = image_shape
        self.noise_level = noise_level
        self.best_focus = 50  # Known best focus for testing

    def take_picture(self, focus: Focus) -> Image:
        """
        Generate a synthetic image with the given focus value.

        Args:
            focus: The focus value to simulate (0-100)

        Returns:
            A synthetic image with the specified focus
        """
        # Create a grid of coordinates
        h, w = self.image_shape
        y, x = np.ogrid[:h, :w]

        # Distance from center
        center_y, center_x = h // 2, w // 2
        y = y - center_y
        x = x - center_x
        r = np.sqrt(x * x + y * y)

        # Create a test pattern (concentric circles)
        pattern = (np.sin(r * 0.5) + 1) * 0.5

        # Simulate defocus blur - more blur as we get further from best focus
        blur_amount = 0.5 + 5 * abs(focus - self.best_focus) / 100.0

        # Apply blur
        if blur_amount > 0.5:
            ksize = max(3, int(blur_amount * 2) // 2 * 2 + 1)
            pattern = cv2.GaussianBlur(pattern, (ksize, ksize), blur_amount)

        # Add noise
        noise = self.noise_level * np.random.randn(*pattern.shape)
        pattern = np.clip(pattern + noise, 0, 1)

        return (pattern * 255).astype(np.uint8)


class DatasetCamera(Camera):
    """
    A camera that loads images from a pre-generated dataset file.
    The dataset should be a pickle file containing a Dataset object.
    """

    def __init__(self, dataset_path: Union[str, os.PathLike]):
        """
        Initialize the dataset camera.

        Args:
            dataset_path: Path to the dataset pickle file
        """
        self.dataset_path = Path(dataset_path)
        if not self.dataset_path.exists():
            raise FileNotFoundError(f"Dataset file not found: {self.dataset_path}")

        # Load the dataset
        self.dataset = Dataset.load(str(self.dataset_path))

        # Create a mapping of focus values to images
        self.images: Dict[Focus, np.ndarray] = {}
        for focus, image in self.dataset.dataset.items():
            # Convert to grayscale if needed
            if len(image.shape) == 3 and image.shape[2] == 3:
                self.images[focus] = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
            else:
                self.images[focus] = image

        # Get available focus values
        self.available_focus = sorted(self.images.keys())
        self.min_focus = min(self.available_focus)
        self.max_focus = max(self.available_focus)

        if self.dataset.correct_focus is not None:
            self.best_focus = self.dataset.correct_focus
        else:
            # If correct_focus is not specified, use the middle of the range
            self.best_focus = (self.min_focus + self.max_focus) // 2

    def take_picture(self, focus: Focus) -> Image:
        """
        Get an image from the dataset with the specified focus.
        If the exact focus is not found, returns the closest available focus.

        Args:
            focus: The focus value to retrieve

        Returns:
            The image at the specified focus or the closest available focus
        """
        # Return exact match if available
        if focus in self.images:
            return self.images[focus].copy()

        # Find the closest focus value
        closest_focus = min(self.available_focus, key=lambda x: abs(x - focus))

        return self.images[closest_focus].copy()

    @property
    def focus_range(self) -> Tuple[Focus, Focus]:
        """Return the minimum and maximum available focus values."""
        return (self.min_focus, self.max_focus)

    @property
    def image_shape(self) -> tuple[int, int]:
        """Return the shape of the images in the dataset."""
        return next(iter(self.images.values())).shape[:2]

    def __str__(self) -> str:
        """Return a string representation of the dataset camera."""
        return (
            f"DatasetCamera(dataset='{self.dataset_path.name}', "
            f"focus_range=({self.min_focus}, {self.max_focus}), "
            f"best_focus={self.best_focus}, "
            f"image_shape={self.image_shape})"
        )
