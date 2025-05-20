import os
from dataclasses import dataclass
from multiprocessing import Pool
from typing import Dict, Optional, Tuple

import numpy as np
import scipy.ndimage as ndi
import tqdm
from PIL import Image

from .workers import get_num_workers


@dataclass
class BlurConfig:
    """Configuration for blur generation"""

    f_min: int
    f_max: int
    correct_focus: int
    bell_curve_std: float = 1.0


def _compute_single_blurr(
    args: Tuple[np.ndarray, int, int, float],
) -> Tuple[int, np.ndarray]:

    def _csb(
        base_image: np.ndarray, focus_value: int, correct_focus: int, std: float
    ) -> Tuple[int, np.ndarray]:
        """Worker function for computing a single blurred image

        Args:
            base_image
            focus_value
            correct_focus
            std (float)

        Returns:
            Tuple[int, np.ndarray]:
        """

        distance_from_correct = abs(focus_value - correct_focus)
        max_distance = max(
            abs(correct_focus - focus_value), abs(args[2] - correct_focus)
        )
        normalized_distance = distance_from_correct / max_distance
        sigma = std * normalized_distance**2

        blurred_image = ndi.gaussian_filter(base_image, sigma=sigma)
        return focus_value, blurred_image

    return _csb(*args)


class FocusDatasetGenerator:
    def __init__(self, focused_image_path: str, config: BlurConfig) -> None:
        """
        Initialize the dataset generator with a focused image

        Args:
            focused_image_path: Path to the focused TIFF image
        """
        self._focused_image = np.array(Image.open(focused_image_path))
        self._focus_range: Tuple[int, int] = (config.f_min, config.f_max)
        self._correct_focus: int = config.correct_focus
        self._std: float = config.bell_curve_std

    def generate_dataset(
        self, num_workers: Optional[int] = None
    ) -> Dict[int, np.ndarray]:
        """
        Generate dataset of images with varying focus levels using multiprocessing

        Args:
            config: Blur configuration
            num_workers: Number of worker processes to use

        Returns:
            Dictionary mapping focus values to blurred images
        """

        if num_workers is None:
            num_workers = get_num_workers()

        dataset: Dict[int, np.ndarray] = {}
        focus_values = list(range(*self._focus_range))

        # Prepare arguments for parallel processing
        args_list = [
            (self._focused_image, f, self._correct_focus, self._std)
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
        self._dataset = {focus: image for focus, image in results}
        return self._dataset
