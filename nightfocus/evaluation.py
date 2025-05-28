"""Module for evaluating focus metrics on datasets."""
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Union

import cv2
import numpy as np
from rich.console import Console
from rich.table import Table

from .dataset import Dataset
from .focus_metrics import FOCUS_MEASURES, get_focus_measure
from .workers import get_num_workers

def _process_single_metric_dataset(
    dataset_path: str, metric_name: str, metric_func: Callable[[np.ndarray], float]
) -> Tuple[str, float, float, float]:
    """
    Process a single dataset file with a single metric.
    
    Args:
        dataset_path: Path to the dataset file
        metric_name: Name of the metric
        metric_func: The metric function to evaluate
        
    Returns:
        Tuple of (dataset_name, best_score, best_focus, correct_focus) where:
        - dataset_name: Base name of the dataset file
        - best_score: Best score found
        - best_focus: Focus value that gave the best score
        - correct_focus: The correct focus value from the dataset
    """
    try:
        # Load the dataset
        dataset = Dataset.load(dataset_path)
        
        # Get the correct focus from the dataset
        correct_focus = float(dataset.correct_focus) if hasattr(dataset, 'correct_focus') else float('nan')
        
        # Evaluate the metric on this dataset
        _, scores = _evaluate_single_metric(metric_name, metric_func, dataset)
        
        if not scores:
            return (Path(dataset_path).name, float('nan'), float('nan'), correct_focus)
            
        # Find the best focus and score
        best_focus, best_score = max(scores.items(), key=lambda x: x[1] if not np.isnan(x[1]) else float('-inf'))
        return (Path(dataset_path).name, best_score, best_focus, correct_focus)
        
    except Exception as e:
        print(f"Error processing {dataset_path}: {e}")
        return (Path(dataset_path).name, float('nan'), float('nan'), float('nan'))

def _evaluate_single_metric(
    metric_name: str, metric_func: Callable[[np.ndarray], float], dataset: Dataset
) -> Tuple[str, Dict[float, float]]:
    """
    Evaluate a single focus metric on a dataset.

    Args:
        metric_name: Name of the metric
        metric_func: The metric function to evaluate
        dataset: Dataset containing images with different focus values

    Returns:
        Tuple of (metric_name, scores_dict) where scores_dict maps focus values to scores
    """
    scores = {}
    for focus, image in dataset.dataset.items():
        try:
            # Convert to grayscale if needed
            if len(image.shape) == 3 and image.shape[2] == 3:
                gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
            else:
                gray = image

            # Calculate score and store with focus value as key
            score = metric_func(gray)
            scores[float(focus)] = float(score)
        except Exception as e:
            print(f"Error evaluating {metric_name} at focus {focus}: {e}")
            scores[float(focus)] = float("nan")

    return (metric_name, scores)

def evaluate_metric_on_directory(
    directory: str, 
    metric: str, 
    workers: Optional[int] = None
) -> None:
    """
    Evaluate a single focus metric on all dataset files in a directory.
    
    Args:
        directory: Path to directory containing dataset files
        metric: Name of the metric to evaluate
        workers: Number of worker processes to use (default: number of CPU cores)
    """
    # Validate metric
    if metric not in FOCUS_MEASURES:
        print(f"Error: Unknown metric '{metric}'. Available metrics: {', '.join(FOCUS_MEASURES.keys())}")
        return
    
    # Get dataset files
    dataset_dir = Path(directory)
    dataset_files = list(dataset_dir.glob("*_dataset.pkl"))
    
    if not dataset_files:
        print(f"No dataset files found in {directory}")
        return
    
    print(f"Found {len(dataset_files)} dataset files in {directory}")
    
    # Get metric function
    metric_func = get_focus_measure(metric)
    
    # Process datasets in parallel
    workers = workers or get_num_workers()
    results = []
    
    with ProcessPoolExecutor(max_workers=workers) as executor:
        # Submit all tasks
        future_to_file = {
            executor.submit(_process_single_metric_dataset, str(f), metric, metric_func): f 
            for f in dataset_files
        }
        
        # Process results as they complete
        for future in as_completed(future_to_file):
            result = future.result()
            if result:
                results.append(result)
    
    # Sort results by best score (descending)
    results.sort(key=lambda x: x[1] if not np.isnan(x[1]) else float('-inf'), reverse=True)
    
    # Display results in a table
    _display_results_table(results, metric)

def _get_heatmap_color(value: float, reverse: bool = False) -> str:
    """Get a color from red to green based on the value's position between min and max."""
    if np.isnan(value):
        return "white"
        
    # Normalize value between 0 and 1
    normalized = max(0.0, min(1.0, value / 50.0))  # Assuming max error of 50 is bad
    
    if reverse:
        normalized = 1.0 - normalized
    
    # Convert to 0-255 for RGB (red to green gradient)
    r = int(255 * (1 - normalized))
    g = int(255 * normalized)
    b = 0
    return f"#{r:02x}{g:02x}{b:02x}"

def _display_results_table(
    results: List[Tuple[str, float, float, float]], 
    metric: str
) -> None:
    """Display evaluation results in a formatted table."""
    console = Console()
    
    # Create table
    table = Table(title=f"Metric: {metric} - Focus Analysis")
    table.add_column("Dataset", style="cyan", no_wrap=True)
    table.add_column("Best Focus", justify="right")
    table.add_column("Correct Focus", justify="right")
    table.add_column("Focus Error", justify="right")
    
    # Add rows with colored output
    for dataset_name, _, best_focus, correct_focus in results:  # _ is best_score which we're not using
        # Format best focus
        best_focus_str = f"{best_focus:.1f}"
        
        # Calculate focus error
        focus_error = abs(best_focus - correct_focus) if not np.isnan(best_focus) and not np.isnan(correct_focus) else float('nan')
        
        # Format correct focus
        correct_focus_str = f"{correct_focus:.1f}"
        
        # Format focus error with heatmap color (lower is better)
        error_color = _get_heatmap_color(focus_error, reverse=True)
        error_str = f"[{error_color}]{focus_error:.1f}" if not np.isnan(focus_error) else "N/A"
        
        table.add_row(
            dataset_name,
            best_focus_str,
            correct_focus_str,
            error_str
        )
    
    console.print()
    console.print(table)
