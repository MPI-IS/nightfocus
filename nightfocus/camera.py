"""Camera abstraction and focus optimization."""

from __future__ import annotations

import os
from typing import List, Tuple, Sequence, Optional, Union
from io import StringIO
import numpy as np
import plotext as plt
from rich.console import Console
from rich.table import Table
from loguru import logger
from abc import ABC, abstractmethod
from io import StringIO
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Tuple, TypeVar, Union

import cv2
import numpy as np
import numpy.typing as npt
from loguru import logger
from rich.console import Console
from rich.table import Table
from skopt import gp_minimize
from skopt.space import Integer
from skopt.utils import use_named_args
from tqdm import tqdm

# Configure loguru to use a more compact format
logger.remove()  # Remove default handler
logger.add(
    lambda msg: tqdm.write(msg, end=""),  # type: ignore
    colorize=True,
    format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <level>{message}</level>",
)

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

        Raises:
            KeyError: If the requested focus value is not in the dataset
        """
        logger.debug(f"Getting image from dataset with focus={focus}")
        try:
            image = self.dataset.get_image(focus)
            logger.trace(
                f"Retrieved image with shape={image.shape}, dtype={image.dtype}"
            )
            return image
        except KeyError:
            logger.error(f"Focus value {focus} not found in dataset")
            raise
        except Exception as e:
            logger.error(f"Error getting image with focus={focus}: {str(e)}")
            raise


def _log_optimization_history(history: Sequence[Tuple[float, float]]) -> None:
    """
    Log the optimization history as a formatted table and plot using Rich and Plotext.

    Args:
        history: List of (focus, score) tuples to be displayed in the table and plot
    """
    if not history:
        logger.warning("No history data to display")
        return

    # Create a string buffer to capture the table
    table_output = StringIO()
    console = Console(file=table_output, force_terminal=True, force_jupyter=False)

    # Create and format the table
    table = Table(
        title="Focus Optimization History",
        show_header=True,
        header_style="bold magenta",
        show_lines=True,
    )

    # Add columns with formatting
    table.add_column("Focus", justify="right", style="cyan", no_wrap=True)
    table.add_column("Score", justify="right", style="green")

    # Add rows to the table
    for focus, score in history:
        table.add_row(f"{focus:.1f}", f"{score:.4f}")

    # Render the table to the string buffer
    console.print(table)

    # Log the table using loguru
    logger.info("\n" + table_output.getvalue())

    
    # Create a terminal plot
    try:
        # Extract focus and score values
        focus_vals = [h[0] for h in history]
        scores = [h[1] for h in history]
        
        # Sort by focus for better visualization
        sorted_pairs = sorted(zip(focus_vals, scores), key=lambda x: x[0])
        focus_vals, scores = zip(*sorted_pairs)
        
        # Create the plot
        plt.clear_figure()
        plt.plot(focus_vals, scores, marker="dot", label="Score")
        plt.title("Focus Optimization History")
        plt.xlabel("Focus Value")
        plt.ylabel("Score")
        plt.grid(True)
        
        # Add markers for all points
        plt.scatter(focus_vals, scores, marker="dot")
        
        # Find and highlight the best focus
        best_idx = np.argmax(scores)
        plt.scatter([focus_vals[best_idx]], [scores[best_idx]], marker="*", color="red")
        
        # Show the plot
        logger.info("\nFocus Optimization Plot:")
        plt.show()
        
    except Exception as e:
        logger.warning(f"Failed to create terminal plot: {str(e)}")


def optimize_focus(
    camera: "Camera",
    focus_measure: FocusMeasure,
    bounds: Tuple[Focus, Focus] = (0, 100),
    initial_points: int = 5,
    max_iter: int = 20,
    noise_level: float = 1e-6,
    random_state: Optional[int] = None,
    output_folder: Optional[Union[str, os.PathLike]] = None,
) -> Tuple[Focus, FocusHistory]:
    """
    Find the optimal focus using Bayesian optimization with Gaussian Processes.

    Args:
        camera: The camera instance to use
        focus_measure: The focus measure function to use
        bounds: Tuple of (min_focus, max_focus)
        initial_points: Number of initial random points to sample
        max_iter: Maximum number of iterations
        noise_level: Expected noise level in the observations
        random_state: Random seed for reproducibility
        output_folder: If provided, save captured images to this folder with format:
                     focus_<focus>_<score>.tiff

    Returns:
        Tuple of (best_focus, history) where history is a list of (focus, score) pairs
    """
    logger.info(
        f"Starting focus optimization with bounds={bounds}, initial_points={initial_points}, max_iter={max_iter}"
    )

    min_focus, max_focus = bounds
    history: List[Tuple[float, float]] = []

    # Define the search space
    space = [Integer(min_focus, max_focus, name="focus")]
    logger.debug(f"Search space defined: {space}")

    # Create output folder if specified
    if output_folder is not None:
        output_folder = Path(output_folder)
        output_folder.mkdir(parents=True, exist_ok=True)
        logger.info(f"Saving captured images to: {output_folder.absolute()}")

    def save_image(image: np.ndarray, focus: int, score: float) -> None:
        """Save image to the output folder if one was specified."""
        if output_folder is None:
            return

        filename = output_folder / f"focus_{focus}_{score:.4f}.tiff"
        try:
            cv2.imwrite(str(filename), image)
            logger.debug(f"Saved image to: {filename}")
        except Exception as e:
            logger.error(f"Failed to save image {filename}: {str(e)}")
            raise

    # Objective function
    @use_named_args(space)
    def objective(focus: int) -> float:
        focus_int = int(round(focus))

        # Take picture and evaluate focus
        try:
            img = camera.take_picture(focus_int)
            score = float(focus_measure(img))
            logger.debug(f"Focus={focus_int}, score={score:.4f}")

            # Save the image if output folder is specified
            save_image(img, focus_int, score)

        except Exception as e:
            logger.error(f"Error evaluating focus={focus_int}: {str(e)}")
            raise

        # Store in history
        history.append((float(focus_int), score))

        return -score  # Negative because scikit-optimize minimizes

    # Run the optimization
    logger.info("Starting Bayesian optimization...")
    try:
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
    except Exception as e:
        logger.error(f"Optimization failed: {str(e)}")
        raise

    # Get the best focus and convert history to the correct format
    best_focus = Focus(int(round(result.x[0])))
    best_score = -result.fun  # Convert back to positive score

    logger.info(
        f"Optimization complete. Best focus: {best_focus} with score: {best_score:.4f}"
    )
    logger.debug(f"Evaluated {len(history)} focus positions")

    # Ensure history is sorted by focus value for better visualization
    history_sorted = sorted(history, key=lambda x: x[0])

    # Log the optimization history as a table
    _log_optimization_history(history_sorted)

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
        logger.info(
            f"Initialized SimulatedCamera with shape={image_shape}, noise_level={noise_level}, best_focus={self.best_focus}"
        )

    def take_picture(self, focus: Focus) -> Image:
        """
        Generate a synthetic image with the given focus value.

        Args:
            focus: The focus value to simulate (0-100)

        Returns:
            A synthetic image with the specified focus
        """
        logger.debug(f"Simulating image capture at focus={focus}")

        try:
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
            focus_diff = abs(focus - self.best_focus)
            blur_amount = 0.5 + 5 * focus_diff / 100.0
            logger.trace(
                f"Focus diff: {focus_diff:.2f}, blur_amount: {blur_amount:.2f}"
            )

            # Apply blur
            if blur_amount > 0.5:
                ksize = max(3, int(blur_amount * 2) // 2 * 2 + 1)
                pattern = cv2.GaussianBlur(pattern, (ksize, ksize), blur_amount)
                logger.trace(f"Applied Gaussian blur with kernel size {ksize}")

            # Add noise
            noise = self.noise_level * np.random.randn(*pattern.shape)
            pattern = np.clip(pattern + noise, 0, 1)

            logger.trace(
                f"Generated image with shape={pattern.shape}, range=[{pattern.min():.2f}, {pattern.max():.2f}]"
            )

            return (pattern * 255).astype(np.uint8)

        except Exception as e:
            logger.error(f"Error in take_picture(focus={focus}): {str(e)}")
            raise


class DatasetCamera(Camera):
    """
    A camera that loads images from a pre-generated dataset file.
    The dataset should be a pickle file containing a Dataset object.
    """

    def __init__(self, dataset_path: Union[str, os.PathLike]):
        """
        Initialize the dataset camera.

        Args:
            dataset_path: Path to the dataset file (pickle file containing a Dataset object)
        """
        self.dataset_path = Path(dataset_path)
        if not self.dataset_path.exists():
            error_msg = f"Dataset file not found: {self.dataset_path}"
            logger.error(error_msg)
            raise FileNotFoundError(error_msg)

        logger.info(f"Loading dataset from {self.dataset_path}")
        try:
            self.dataset = Dataset.load(str(self.dataset_path))
            focus_keys = list(self.dataset.dataset.keys())
            logger.info(
                f"Loaded dataset with {len(self.dataset.dataset)} images, "
                f"focus range: {min(focus_keys)}-{max(focus_keys)}"
            )
        except Exception as e:
            logger.error(f"Failed to load dataset from {self.dataset_path}: {str(e)}")
            raise

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
