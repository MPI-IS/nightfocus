import os
from multiprocessing import Pool
from typing import Any, Callable, Dict, List, Optional, TypeVar

import numpy as np
from scipy.stats import kendalltau
from tqdm import tqdm

from .workers import get_num_workers

# Type variable for the focus value type (typically int or float)
T = TypeVar("T", int, float)


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


def score_focus_quality(focus_scores: Dict[T, float], correct_focus: T) -> float:
    """
    Evaluate how close the obtained order of focus scores is to the ideal order
    where values increase with distance from the correct focus.

    A perfect score of 0.0 means the focus values are perfectly ordered with
    the minimum at the correct focus and increasing as we move away.

    Args:
        focus_scores: Dictionary mapping focus values to their scores
        correct_focus: The known correct focus value

    Returns:
        A score between 0.0 (best) and 1.0 (worst)
    """
    if not focus_scores:
        return 1.0  # No data is the worst case

    # Get all focus values and their distances from correct focus
    focus_values = list(focus_scores.keys())
    distances = {f: abs(f - correct_focus) for f in focus_values}

    # Group focus values by distance (values with same distance are equivalent in ideal order)
    distance_groups: Dict[float, List[T]] = {}
    for f in focus_values:
        d = distances[f]
        if d not in distance_groups:
            distance_groups[d] = []
        distance_groups[d].append(f)

    # Sort groups by distance (ascending)
    sorted_groups = [distance_groups[d] for d in sorted(distance_groups.keys())]

    # Sort values within each group by their scores (ascending)
    # This represents the ideal order where values with same distance are ordered by their scores
    ideal_order = []
    for group in sorted_groups:
        ideal_order.extend(sorted(group, key=lambda x: focus_scores[x]))

    # Sort focus values by their scores (obtained order)
    obtained_order = sorted(focus_scores.keys(), key=lambda x: focus_scores[x])

    # Calculate Kendall's tau rank correlation between the two orderings

    # Create rank dictionaries
    ideal_ranks = {val: i for i, val in enumerate(ideal_order)}
    obtained_ranks = {val: i for i, val in enumerate(obtained_order)}

    # Get ranks in the same order for both sortings
    rank1 = [ideal_ranks[val] for val in focus_scores]
    rank2 = [obtained_ranks[val] for val in focus_scores]

    try:
        # Calculate Kendall's tau (ranges from -1 to 1)
        # 1.0 means perfect agreement, -1.0 means perfect inversion
        tau, _ = kendalltau(rank1, rank2)

        # Convert to [0, 1] range where 0 is perfect agreement
        # (tau + 1) / 2 converts [-1, 1] to [0, 1]
        # 1 - that converts to 1 for perfect agreement, 0 for perfect inversion
        return float(max(0.0, min(1.0, 1.0 - (tau + 1) / 2)))
    except:
        return 1.0  # Error in calculation
