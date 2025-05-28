import os
from dataclasses import dataclass
from multiprocessing import Pool
from pathlib import Path
from typing import Any, Callable, Dict, List, NewType, Optional, Tuple

import numpy as np
from loguru import logger
from rich.console import Console
from rich.table import Table
from scipy.stats import kendalltau
from tqdm import tqdm

from .dataset import Dataset, Focus
from .workers import get_num_workers

# Configure loguru to use a more compact format
logger.remove()  # Remove default handler
logger.add(
    lambda msg: tqdm.write(msg, end=""),  # type: ignore
    colorize=True,
    format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <level>{message}</level>",
)

# Initialize console for rich output
console = Console()


@dataclass
class DatasetResult:
    """Container for dataset evaluation results."""

    file_name: str
    scores: Dict[Focus, float]
    quality: float
    correct_focus: Focus


def _compute_single_score(args: tuple) -> tuple[Focus, float]:
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
    dataset: Dict[Focus, np.ndarray],
    score_function: Callable[[np.ndarray], float],
    num_workers: Optional[int] = None,
) -> Dict[Focus, float]:
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
        results = list(
            tqdm(
                pool.imap_unordered(
                    _compute_single_score, args_list, chunksize=chunk_size
                ),
                total=len(dataset),
                desc="Computing focus scores",
            )
        )

    # Collect results into dictionary
    return dict(results)


def _load_dataset_file(dataset_path: Path) -> List[Path]:
    """Load and validate dataset files from directory.

    Args:
        dataset_path: Path to the directory containing dataset files

    Returns:
        List of Path objects to dataset files

    Raises:
        FileNotFoundError: If directory doesn't exist or no dataset files found
    """
    if not dataset_path.exists():
        logger.error(f"Dataset directory not found: {dataset_path}")
        raise FileNotFoundError(f"Dataset directory not found: {dataset_path}")

    dataset_files = sorted(dataset_path.glob("*_dataset.pkl"))
    if not dataset_files:
        error_msg = f"No dataset files (*_dataset.pkl) found in {dataset_path}"
        logger.error(error_msg)
        raise FileNotFoundError(error_msg)

    return dataset_files


def _calculate_scores(
    dataset: Dataset, score_func: Callable[[np.ndarray], float]
) -> Dict[Focus, float]:
    """Calculate scores for all images in a dataset using the provided scoring function.

    Args:
        dataset: Dataset object containing focus levels and images
        score_func: Function that takes an image and returns a score

    Returns:
        Dictionary mapping focus values to their scores
    """
    focus_scores = {}
    for focus, image in dataset.dataset.items():
        print(focus, image.shape)
        # Convert to grayscale if needed (assuming single channel for scoring)
        if len(image.shape) == 3 and image.shape[2] == 3:
            # Simple average of RGB channels for grayscale conversion
            image = image.mean(axis=2)
        focus_scores[focus] = score_func(image)
    return focus_scores


def _print_scores_summary(results: List[Dict[str, Any]]) -> None:
    """Print summary statistics for focus quality scores using rich tables.

    Args:
        results: List of dataset evaluation results
    """
    if not results:
        return

    # Create table
    table = Table(
        title="Evaluation Results", show_header=True, header_style="bold magenta"
    )
    table.add_column("Dataset", style="cyan")
    table.add_column("Quality Score", justify="right")
    table.add_column("Best Focus", justify="center")
    table.add_column("Correct Focus", justify="center")

    for result in results:
        table.add_row(
            result["file_name"],
            f"{result['quality']:.4f}",
            str(min(result["scores"].items(), key=lambda x: x[1])[0]),
            str(result["correct_focus"]),
        )

    console.print()
    console.print(table)


def evaluate_scoring(
    dataset_dir: str,
    score_func: Callable[[np.ndarray], float],
) -> None:
    """
    Evaluate a focus scoring function on a dataset of images.

    This function loads a dataset of images with different focus levels, computes
    scores for each image using the provided scoring function, and evaluates how
    well the scores correlate with the known focus distances.

    Args:
        dataset_dir: Path to the directory containing dataset files (*_dataset.pkl)
        score_func: Function that takes an image and returns a score
    """
    dataset_path = Path(dataset_dir)
    dataset_files = _load_dataset_file(dataset_path)
    logger.info(f"Found {len(dataset_files)} dataset files in {dataset_path}")

    results = []

    for dataset_file in dataset_files:
        # for dataset_file in tqdm(dataset_files, desc="Processing datasets"):
        # Load dataset and calculate scores
        print(dataset_file)
        dataset = Dataset.load(str(dataset_file))
        focus_scores = _calculate_scores(dataset, score_func)

        # Log individual scores
        for focus, score in focus_scores.items():
            logger.debug(f"Focus {focus}: {score_func.__name__} = {score:.4f}")

        # Evaluate and store results
        quality = score_focus_quality(focus_scores, dataset.correct_focus)
        logger.info(
            f"Dataset: {dataset_file.stem} | Quality: {quality:.4f} | "
            f"Best focus: {min(focus_scores.items(), key=lambda x: x[1])[0]}"
        )

        results.append(
            {
                "file_name": dataset_file.stem,
                "scores": focus_scores,
                "quality": quality,
                "correct_focus": dataset.correct_focus,
            }
        )

    _print_scores_summary(results)


def score_focus_quality(
    focus_scores: Dict[Focus, float], correct_focus: Focus
) -> float:
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

    if len(focus_scores) == 1:
        # If there's only one value, it's trivially perfectly ordered
        return 0.0

    # Get all focus values and their distances from correct focus
    focus_values = list(focus_scores.keys())
    distances = {f: abs(f - correct_focus) for f in focus_values}

    # Group focus values by distance (values with same distance are equivalent in ideal order)
    distance_groups: Dict[float, List[Focus]] = {}
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

        # If all ranks are the (which can happen with duplicate scores),
        # kendalltau returns NaN. In this case, we consider it a perfect match.
        if np.isnan(tau):
            return 0.0

        # Convert to [0, 1] range where 0 is perfect agreement
        # (tau + 1) / 2 converts [-1, 1] to [0, 1]
        # 1 - that converts to 1 for perfect agreement, 0 for perfect inversion
        return float(max(0.0, min(1.0, 1.0 - (tau + 1) / 2)))
    except Exception as e:
        # In case of any error, return worst score
        return 1.0
