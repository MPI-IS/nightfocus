"""Test utilities for camera tests."""

import random
from pathlib import Path
from typing import Optional

from nightfocus import SimulatedCamera


def create_simulated_camera(image_shape=(100, 100), noise_level=0.1):
    """Create a simulated camera with known best focus at 50.

    Args:
        image_shape: Shape of the output images (height, width)
        noise_level: Amount of noise to add to images
    """
    return SimulatedCamera(image_shape=image_shape, noise_level=noise_level)


def create_sample_dataset(dataset_name: Optional[str] = None) -> str:
    """Get the path to an existing dataset file from the images directory.

    Args:
        dataset_name: Optional name of the dataset file (e.g., 'image_001_dataset.pkl').
                     If None, a random dataset will be selected.

    Returns:
        Absolute path to the dataset file
    """
    images_dir = Path(__file__).parent.parent / "images"

    if dataset_name:
        dataset_path = images_dir / dataset_name
        if not dataset_path.exists():
            raise FileNotFoundError(f"Dataset file not found: {dataset_path}")
        return str(dataset_path)

    # Find all dataset files
    dataset_files = list(images_dir.glob("*_dataset.pkl"))
    if not dataset_files:
        raise FileNotFoundError(f"No dataset files found in {images_dir}")

    # Return a random dataset file
    return str(random.choice(dataset_files))
