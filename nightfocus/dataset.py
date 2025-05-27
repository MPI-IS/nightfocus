import pickle
from dataclasses import dataclass
from multiprocessing import Pool
from typing import Dict, Optional, Tuple

import numpy as np
import scipy.ndimage as ndi
import tqdm
from multipledispatch import dispatch
from PIL import Image

from .workers import get_num_workers


@dataclass
class Dataset:
    dataset: Dict[int, np.ndarray]
    correct_focus: int

    def dump(self, output_file: str) -> None:
        with open(output_file, "wb") as f:
            pickle.dump(self, f)

    @staticmethod
    def load(output_file: str) -> "Dataset":
        with open(output_file, "rb") as f:
            return pickle.load(f)


@dataclass
class BlurConfig:
    """Configuration for blur generation"""

    f_min: int
    f_max: int
    correct_focus: int
    bell_curve_std: float = 1.0


def _compute_single_blurr(
    args: Tuple[np.ndarray, int, int, float, int, int],
) -> Tuple[int, np.ndarray]:
    """Worker function for computing a single blurred image

    Args:
        args: Tuple containing (base_image, focus_value, correct_focus, std, f_min, f_max)

    Returns:
        Tuple of (focus_value, blurred_image)
    """
    base_image, focus_value, correct_focus, std, f_min, f_max = args
    
    distance_from_correct = abs(focus_value - correct_focus)
    max_distance = max(abs(f_max - correct_focus), abs(correct_focus - f_min))  # Maximum possible distance
    normalized_distance = distance_from_correct / max(max_distance, 1)  # Avoid division by zero
    sigma = std * normalized_distance**2

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
        (focused_image, f, config.correct_focus, config.bell_curve_std, config.f_min, config.f_max)
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
