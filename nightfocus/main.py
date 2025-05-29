import glob
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Callable, Dict, List, Optional, Set, Tuple

import click
import cv2
import numpy as np
from rich.console import Console
from rich.table import Table

from .dataset import Dataset, display_dataset, generate_dataset
from .evaluation import evaluate_metric_on_directory
from .focus_metrics import FOCUS_MEASURES, get_focus_measure
from .processing import create_random_crops
from .scoring import evaluate_scoring
from .workers import get_num_workers

CONTEXT_SETTINGS = dict(help_option_names=["-h", "--help"])


@click.group(context_settings=CONTEXT_SETTINGS)
@click.version_option()
def cli():
    """NightFocus - Tools for focus evaluation and dataset generation"""
    pass


@cli.command()
@click.argument("input_image", type=click.Path(exists=True, dir_okay=False))
@click.option(
    "--num-crops",
    type=int,
    default=10,
    show_default=True,
    help="Number of crops to create",
)
@click.option(
    "--crop-size", type=int, default=200, show_default=True, help="Size of each crop"
)
@click.option(
    "--center-radius",
    type=int,
    default=500,
    show_default=True,
    help="Radius around center for crop selection",
)
def crops(input_image, num_crops, crop_size, center_radius):
    """Create random crops from an image."""
    output_dir = os.getcwd()
    create_random_crops(
        input_path=input_image,
        output_dir=output_dir,
        num_crops=num_crops,
        crop_size=crop_size,
        center_radius=center_radius,
    )


@cli.command()
@click.argument("input_folder", type=click.Path(exists=True, file_okay=False))
@click.option(
    "--f-min", type=int, default=0, show_default=True, help="Minimum focus value"
)
@click.option(
    "--f-max", type=int, default=100, show_default=True, help="Maximum focus value"
)
@click.option(
    "--correct-focus",
    type=int,
    default=50,
    show_default=True,
    help="Focus value where the image is perfectly in focus",
)
@click.option(
    "--blur-scale",
    type=float,
    default=2.0,
    show_default=True,
    help="Scaling factor for blur intensity. Higher values = more blur away from focus.",
)
@click.option(
    "--output-suffix",
    "output_suffix",
    default="_dataset.pkl",
    show_default=True,
    help="Suffix for output dataset files",
)
def dataset(input_folder, f_min, f_max, correct_focus, blur_scale, output_suffix):
    """Generate blurred dataset from TIFF files with adjustable blur intensity."""
    # Calculate bell_curve_std based on the range and desired blur scale
    range_size = max(f_max - correct_focus, correct_focus - f_min)
    bell_curve_std = range_size / (
        10.0 * blur_scale
    )  # Adjust divisor to get reasonable default blur

    config = BlurConfig(
        f_min=f_min,
        f_max=f_max,
        correct_focus=correct_focus,
        bell_curve_std=bell_curve_std,
    )

    # Get all TIFF files in input folder
    tiff_files = glob.glob(os.path.join(input_folder, "*.tiff"))
    if not tiff_files:
        click.echo("No TIFF files found in the input folder.")
        return

    # Get number of workers
    num_workers = get_num_workers()

    # Process each TIFF file
    for input_file in tiff_files:
        # Generate output filename
        file_stem = os.path.splitext(os.path.basename(input_file))[0]
        output_file = os.path.join(input_folder, f"{file_stem}{output_suffix}")

        try:
            click.echo(f"Processing {input_file}...")
            # Generate and save dataset
            dataset = generate_dataset(input_file, config, num_workers)
            dataset.dump(output_file)
            click.echo(f"Saved dataset to {output_file}")
        except Exception as e:
            click.echo(f"Error processing {input_file}: {str(e)}", err=True)


@cli.command()
@click.option(
    "--dataset-dir",
    type=click.Path(exists=True, file_okay=False),
    default="images",
    show_default=True,
    help="Directory containing dataset files",
)
def evaluate(dataset_dir):
    """Evaluate focus scoring on a dataset."""
    evaluate_scoring(dataset_dir, entropy_score)


@cli.command()
@click.argument("dataset_file", type=click.Path(exists=True, dir_okay=False))
def view(dataset_file: str) -> None:
    """View images from a dataset file with their focus values.

    Controls:
    - 'n': Show next image
    - 'p': Show previous image
    - 'q': Quit the viewer
    """
    display_dataset(dataset_file)


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
            # Convert to grayscale if needed (assuming single channel for scoring)
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


@cli.command()
@click.argument("dataset_file", type=click.Path(exists=True, dir_okay=False))
@click.option(
    "--workers",
    type=int,
    default=None,
    help="Number of worker processes to use. Defaults to number of CPU cores.",
)
@click.option(
    "--metrics",
    type=str,
    default=None,
    help="Comma-separated list of metrics to evaluate. If not provided, all metrics are evaluated.",
)
def evaluate_metrics(
    dataset_file: str, workers: Optional[int] = None, metrics: Optional[str] = None
) -> None:
    """
    Evaluate multiple focus metrics on a dataset and display results in a table.

    Available metrics:
    - variance_laplacian: Fast Laplacian variance measure
    - modified_laplacian: Modified Laplacian focus measure
    - tenengrad: Tenengrad focus measure based on gradient magnitude
    - normalized_variance: Normalized gray-level variance
    - spectral_energy: Spectral energy in high frequencies
    - brenner_gradient: Brenner's gradient measure
    - threshold_count: Count of bright pixels above threshold
    - fast_entropy: Faster entropy calculation with binning
    - wavelet_measure: Wavelet-based focus measure
    """
    # Load the dataset
    try:
        dataset = Dataset.load(dataset_file)
    except Exception as e:
        click.echo(f"Error loading dataset: {e}", err=True)
        return

    # Determine which metrics to evaluate
    if metrics:
        metric_names = [m.strip() for m in metrics.split(",")]
        invalid_metrics = [m for m in metric_names if m not in FOCUS_MEASURES]
        if invalid_metrics:
            click.echo(
                f"Warning: Unknown metrics: {', '.join(invalid_metrics)}", err=True
            )
            click.echo(f"Available metrics: {', '.join(FOCUS_MEASURES.keys())}")
            metric_names = [m for m in metric_names if m in FOCUS_MEASURES]
    else:
        metric_names = list(FOCUS_MEASURES.keys())

    if not metric_names:
        click.echo("No valid metrics to evaluate.", err=True)
        return

    # Set up parallel processing
    workers = workers or get_num_workers()
    console = Console()

    # Process each metric
    results = {}
    tasks = []

    with ProcessPoolExecutor(max_workers=workers) as executor:
        # Submit all tasks
        for name in metric_names:
            metric_func = get_focus_measure(name)
            tasks.append(
                executor.submit(_evaluate_single_metric, name, metric_func, dataset)
            )

        # Process results as they complete
        with console.status("[bold green]Evaluating metrics...") as status:
            for future in as_completed(tasks):
                try:
                    name, scores = future.result()
                    results[name] = scores
                    console.log(f"[green]✓[/green] Completed: {name}")
                except Exception as e:
                    console.log(f"[red]✗ Error: {e}")

    # Display results in a table
    if not results:
        console.print("[red]No results to display.[/red]")
        return

    # Get all unique focus values from all results
    all_focus_values: Set[float] = set()
    for scores in results.values():
        all_focus_values.update(scores.keys())

    if not all_focus_values:
        console.print("[red]No focus values found in results.[/red]")
        return

    # Sort focus values
    sorted_focus = sorted(all_focus_values)

    # Prepare data for finding max values and normalizing
    metrics_data: Dict[str, List[float]] = {}
    max_values: Dict[str, float] = {}
    min_values: Dict[str, float] = {}

    # Find min and max for each metric
    for name, scores_dict in results.items():
        valid_scores = [s for s in scores_dict.values() if not np.isnan(s)]
        if valid_scores:  # Only process if we have valid scores
            metrics_data[name] = valid_scores
            max_values[name] = max(valid_scores)
            min_values[name] = min(valid_scores)

    # Create and display table
    table = Table(title=f"Focus Metrics Evaluation - {os.path.basename(dataset_file)}")
    table.add_column("Focus", justify="right")

    # Add metric columns
    for name in results.keys():
        table.add_column(name, justify="right")

    # Function to get color based on value (heatmap style)
    def get_heatmap_color(value: float, min_val: float, max_val: float) -> str:
        if np.isnan(value) or min_val == max_val:
            return "white"
        # Normalize value between 0 and 1
        normalized = (value - min_val) / (max_val - min_val)
        # Convert to 0-255 for RGB
        r = int(255 * normalized)
        g = int(255 * (1 - normalized))
        b = 0
        return f"#{r:02x}{g:02x}{b:02x}"

    # Add rows for each focus value
    for focus in sorted_focus:
        row = [f"[white]{focus:.1f}"]

        for name in results.keys():
            score = results[name].get(focus, float("nan"))

            if np.isnan(score):
                row.append("N/A")
                continue

            # Check if this is the max value for this metric
            is_max = abs(score - max_values.get(name, -float("inf"))) < 1e-10

            # Get heatmap color
            color = get_heatmap_color(
                score, min_values.get(name, 0), max_values.get(name, 1)
            )

            # Format the value with color and bold if it's the max
            value_str = f"{score:.4f}"
            if is_max:
                value_str = f"[bold white on {color}]{value_str}"
            else:
                value_str = f"[{color}]{value_str}"

            row.append(value_str)

        table.add_row(*row)

    # Add a legend
    console.print(
        "[bold]Legend:[/bold] [red]Higher values[/red] are better. [bold]Bold values[/bold] indicate the best focus for each metric."
    )
    console.print()
    console.print(table)

    # Save results to CSV
    output_file = f"{os.path.splitext(dataset_file)[0]}_metrics.csv"
    with open(output_file, "w") as f:
        # Write header
        f.write("Focus," + ",".join(results.keys()) + "\n")

        # Write data
        for focus in sorted_focus:
            row = [f"{focus:.1f}"]
            for name in results.keys():
                score = results[name].get(focus, float("nan"))
                row.append(f"{score:.6f}" if not np.isnan(score) else "")


@cli.command()
@click.argument(
    "directory", type=click.Path(exists=True, file_okay=False, dir_okay=True)
)
@click.argument("metric", type=click.Choice(list(FOCUS_MEASURES.keys())))
@click.option(
    "--workers",
    type=int,
    default=None,
    help="Number of worker processes to use. Defaults to number of CPU cores.",
)
def evaluate_directory(
    directory: str, metric: str, workers: Optional[int] = None
) -> None:
    """
    Evaluate a single focus metric on all dataset files in a directory.
    
    Available metrics:
    """ + "\n    ".join(
        [
            f"- {n}: {f.__doc__.split('\n')[0] if f.__doc__ else 'No description'}"
            for n, f in FOCUS_MEASURES.items()
        ]
    )

    # Delegate to the evaluation module
    evaluate_metric_on_directory(directory, metric, workers)


def main():
    cli()


if __name__ == "__main__":
    main()
