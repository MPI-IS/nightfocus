import pickle
from dataclasses import dataclass
from multiprocessing import Pool
from pathlib import Path
from typing import Dict, NewType, Optional, Tuple

import cv2
import numpy as np
import scipy.ndimage as ndi
import tqdm
from multipledispatch import dispatch
from PIL import Image

from .workers import get_num_workers

# Type alias for focus values (integers representing focus positions)
Focus = NewType("Focus", int)


@dataclass
class Dataset:
    dataset: Dict[Focus, np.ndarray]
    correct_focus: Focus

    def dump(self, output_file: str) -> None:
        with open(output_file, "wb") as f:
            pickle.dump(self, f)

    @staticmethod
    def load(output_file: str) -> "Dataset":
        with open(output_file, "rb") as f:
            return pickle.load(f)

    def display(self) -> None:
        """Display images from the dataset with their focus values.

        Controls:
        - 'n': Show next image
        - 'p': Show previous image
        - 'q': Quit the viewer
        """
        # Get all focus values and sort them
        focus_values = sorted(self.dataset.keys())

        print(f"\nFocus range: {min(focus_values)} to {max(focus_values)}")
        print(f"Correct focus: {self.correct_focus}")
        print("Press 'n' for next image, 'p' for previous, 'q' to quit")

        current_idx = 0

        while True:
            focus = focus_values[current_idx]
            image = self.dataset[focus]

            # Convert to BGR for OpenCV if needed
            if len(image.shape) == 3 and image.shape[2] == 3:
                display_img = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            else:
                # Convert grayscale to BGR for consistent display
                display_img = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

            # Add focus text to the image
            focus_text = f"Focus: {focus}"
            if focus == self.correct_focus:
                focus_text += " (CORRECT)"

            cv2.putText(
                display_img,
                focus_text,
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2,
            )

            # Show the image
            cv2.imshow("Dataset Viewer", display_img)

            # Wait for key press
            key = cv2.waitKey(0) & 0xFF

            # Handle key presses
            if key == ord("q"):  # Quit
                break
            elif key == ord("n"):  # Next image
                current_idx = (current_idx + 1) % len(focus_values)
            elif key == ord("p"):  # Previous image
                current_idx = (current_idx - 1) % len(focus_values)

        cv2.destroyAllWindows()


def display_dataset(dataset_path: str) -> None:
    """Load and display a dataset from file.

    Args:
        dataset_path: Path to the dataset file (.pkl)
    """
    try:
        dataset = Dataset.load(dataset_path)
        print(f"\nDataset: {Path(dataset_path).name}")
        dataset.display()
    except Exception as e:
        print(f"Error loading dataset: {str(e)}")
        if "cv2" in locals():
            cv2.destroyAllWindows()


@dataclass
class BlurConfig:
    """Configuration for blur generation"""

    f_min: int
    f_max: int
    correct_focus: Focus
    bell_curve_std: float = 1.0


def _compute_single_blurr(
    args: Tuple[np.ndarray, Focus, Focus, float, int, int],
) -> Tuple[Focus, np.ndarray]:
    """Worker function for computing a single blurred image

    Args:
        args: Tuple containing (base_image, focus_value, correct_focus, std, f_min, f_max)

    Returns:
        Tuple of (focus_value, blurred_image)
    """
    base_image, focus_value, correct_focus, std, f_min, f_max = args

    distance_from_correct = abs(focus_value - correct_focus)
    max_distance = max(
        abs(f_max - correct_focus), abs(correct_focus - f_min)
    )  # Maximum possible distance

    # Avoid division by zero and ensure we have a valid range
    if max_distance == 0:
        normalized_distance = 0.0
    else:
        normalized_distance = distance_from_correct / max_distance

    # Apply non-linear scaling to make the blur more pronounced at the extremes
    # Using a cubic function to make the blur increase more rapidly
    blur_factor = normalized_distance**3

    # Scale the sigma to be more aggressive with the blur
    # The max sigma is set to 10.0 which provides significant blur
    sigma = std + (blur_factor * 10.0)  # Scale up the effect

    # Ensure we don't get excessive blur for very large values
    sigma = min(sigma, 15.0)

    # Apply Gaussian blur with the calculated sigma
    blurred_image = ndi.gaussian_filter(base_image, sigma=sigma)
    return focus_value, blurred_image


@dispatch(np.ndarray, BlurConfig, int)
def generate_dataset(
    focused_image: np.ndarray, config: BlurConfig, num_workers: int
) -> Dataset:
    """
    Generate dataset of images with varying focus levels using multiprocessing.

    Args:
        focused_image: Numpy array containing the input image
        config: Blur configuration
        num_workers: Number of worker processes to use

    Returns:
        Dataset object containing blurred images
    """
    if num_workers is None:
        num_workers = get_num_workers()

    focus_values = list(range(config.f_min, config.f_max + 1))

    # Prepare arguments for parallel processing
    args_list = [
        (
            focused_image,
            f,
            config.correct_focus,
            config.bell_curve_std,
            config.f_min,
            config.f_max,
        )
        for f in focus_values
    ]

    # Process in chunks to manage memory
    chunk_size = max(1, len(focus_values) // num_workers)

    with Pool(processes=num_workers) as pool:
        results = list(
            tqdm.tqdm(
                pool.imap_unordered(
                    _compute_single_blurr, args_list, chunksize=chunk_size
                ),
                total=len(focus_values),
                desc="Generating blurred images",
            )
        )

    # Collect results into dataset
    dataset = {focus: image for focus, image in results}
    return Dataset(dataset=dataset, correct_focus=config.correct_focus)


@dispatch(str, BlurConfig, int)
def generate_dataset(image_path: str, config: BlurConfig, num_workers: int) -> Dataset:
    """
    Generate dataset of images with varying focus levels using multiprocessing.

    Args:
        image_path: Path to the input image file
        config: Blur configuration
        num_workers: Number of worker processes to use

    Returns:
        Dataset object containing blurred images
    """
    focused_image = np.array(Image.open(image_path))
    return generate_dataset(focused_image, config, num_workers)
