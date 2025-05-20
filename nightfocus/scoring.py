import os
from multiprocessing import Pool
from typing import Callable, Dict, Optional

import numpy as np
from tqdm import tqdm

from .workers import get_num_workers


def _compute_single_score(args: tuple) -> tuple:
    """
    Helper function to compute score for a single focus value

    Args:
        args: Tuple containing (image, focus_value, score_function)

    Returns:
        Tuple of (focus_value, score)
    """
    image, focus_value, score_function = args
    return focus_value, score_function(image)


def compute_focus_scores(
    dataset: Dict[int, np.ndarray],
    score_function: Callable[[np.ndarray], float],
    num_workers: Optional[int] = None,
) -> Dict[int, float]:
    """
    Compute focus scores for all images in dataset using multiprocessing

    Args:
        dataset: Dictionary mapping focus values to images
        score_function: Function that takes an image and returns a score
        num_workers: Number of worker processes to use

    Returns:
        Dictionary mapping focus values to their scores
    """

    num_workers = num_workers if num_workers is not None else get_num_workers()

    # Prepare arguments for parallel processing
    args_list = [(image, focus, score_function) for focus, image in dataset.items()]

    # Process in chunks to manage memory
    chunk_size = max(1, len(dataset) // num_workers)

    with Pool(processes=num_workers) as pool:
        if tqdm:
            results = list(
                tqdm(
                    pool.imap_unordered(
                        _compute_single_score, args_list, chunksize=chunk_size
                    ),
                    total=len(dataset),
                    desc="Computing focus scores",
                )
            )
        else:
            results = list(
                pool.imap_unordered(
                    _compute_single_score, args_list, chunksize=chunk_size
                )
            )

    # Collect results into dictionary
    return dict(results)
