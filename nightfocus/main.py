import glob
import os
from pathlib import Path

import click

from .dataset import BlurConfig, Dataset, display_dataset, generate_dataset
from .processing import create_random_crops
from .scoring import entropy_score, evaluate_scoring
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


def main():
    cli()


if __name__ == "__main__":
    main()
