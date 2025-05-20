import logging
import os
import random

import numpy as np
from PIL import Image
from tifffile import imread

logger = logging.getLogger(__name__)


def _bits_reduction(data: np.ndarray, target: np.dtype) -> np.ndarray:
    original_max = np.iinfo(data.dtype).max
    target_max = np.iinfo(target).max
    ratio = target_max / original_max
    return (data * ratio).astype(target)


def _to_8bits(image: np.ndarray) -> np.ndarray:
    """
    Convert image to 8 bits (i.e. returns an array
    of dtype numpy uint8)
    """
    return _bits_reduction(image, np.dtype(np.uint8))


def create_random_crops(
    input_path: str,
    output_dir: str,
    num_crops: int = 10,
    crop_size: int = 200,
    center_radius: int = 500,
) -> None:
    """
    Create random crops from a large TIFF image around its center
    Args:
        input_path: Path to input TIFF image
        output_dir: Directory to save cropped images
        num_crops: Number of crops to generate
        crop_size: Size of each crop (square)
        center_radius: Maximum distance from center for crop centers
    """
    logger.info(f"Loading image from {input_path}")

    # Load image and cast it to 8 bits
    img_array = imread(input_path)
    img_array = _to_8bits(img_array)

    height, width = img_array.shape[:2]
    center_x = width // 2
    center_y = height // 2

    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    logger.info(f"Output directory is set to {output_dir}")

    crops_created = 0
    attempts = 0

    # Generate crops
    for i in range(num_crops):
        # Generate random center point within radius of image center
        angle = random.uniform(0, 2 * np.pi)
        radius = random.uniform(0, center_radius)
        crop_center_x = int(center_x + radius * np.cos(angle))
        crop_center_y = int(center_y + radius * np.sin(angle))

        # Calculate crop bounds
        left = crop_center_x - crop_size // 2
        top = crop_center_y - crop_size // 2

        # Ensure crop stays within image bounds
        if left < 0 or top < 0 or left + crop_size > width or top + crop_size > height:
            logger.warning(
                f"Crop {i}: Crop at ({crop_center_x}, {crop_center_y}) with size {crop_size}x{crop_size} is out of bounds. Skipping."
            )
            continue

        # Extract crop and convert to 8-bit
        crop = img_array[top : top + crop_size, left : left + crop_size]

        # Save as TIFF
        crop_img = Image.fromarray(crop)
        output_path = os.path.join(output_dir, f"crop_{i:03d}.tiff")
        crop_img.save(output_path, compression=None)
        logger.info(f"Saved crop {i} to {output_path}")

    logger.info("Random crop generation complete.")
