import argparse
import os

from .dataset import generate_dataset
from .processing import create_random_crops
from .workers import get_num_workers


def create_crops():
    parser = argparse.ArgumentParser(description="Create random crops from an image")
    parser.add_argument("input_image", help="Path to the input image")
    parser.add_argument(
        "--num-crops",
        type=int,
        default=10,
        help="Number of crops to create (default: 10)",
    )
    parser.add_argument(
        "--crop-size", type=int, default=200, help="Size of each crop (default: 200)"
    )
    parser.add_argument(
        "--center-radius",
        type=int,
        default=500,
        help="Radius around center for crop selection (default: 500)",
    )

    args = parser.parse_args()

    # Create crops in current directory
    output_dir = os.getcwd()
    create_random_crops(
        input_path=args.input_image,
        output_dir=output_dir,
        num_crops=args.num_crops,
        crop_size=args.crop_size,
        center_radius=args.center_radius,
    )


def generate_blurred_dataset():
    import glob
    import json
    import os
    from pathlib import Path

    from .dataset import BlurConfig, generate_dataset

    parser = argparse.ArgumentParser(
        description="Generate blurred dataset from TIFF files"
    )
    parser.add_argument(
        "input_folder", help="Path to the folder containing TIFF files to process"
    )
    parser.add_argument(
        "--f_min", type=int, default=0, help="Minimum focus value (default: 0)"
    )
    parser.add_argument(
        "--f_max", type=int, default=100, help="Maximum focus value (default: 100)"
    )
    parser.add_argument(
        "--correct_focus",
        type=int,
        default=50,
        help="The correct focus value (default: 50)",
    )
    parser.add_argument(
        "--bell_curve_std",
        type=float,
        default=1.0,
        help="Standard deviation for the blur bell curve (default: 1.0)",
    )
    parser.add_argument(
        "--output_suffix",
        type=str,
        default="_dataset.pkl",
        help="Suffix for output dataset files (default: _dataset.pkl)",
    )

    args = parser.parse_args()

    # Create blur config
    config = BlurConfig(
        f_min=args.f_min,
        f_max=args.f_max,
        correct_focus=args.correct_focus,
        bell_curve_std=args.bell_curve_std,
    )

    # Get all TIFF files in input folder
    tiff_files = glob.glob(os.path.join(args.input_folder, "*.tiff"))

    if not tiff_files:
        print(f"No TIFF files found in {args.input_folder}")
        return

    # Process each TIFF file
    for input_file in tiff_files:
        # Generate output filename
        file_stem = Path(input_file).stem
        output_file = os.path.join(
            args.input_folder, f"{file_stem}{args.output_suffix}"
        )

        print(f"Processing {input_file}...")

        try:
            # Generate and save dataset
            dataset = generate_dataset(input_file, config, get_num_workers())
            dataset.dump(output_file)
            print(f"Saved dataset to {output_file}")

        except Exception as e:
            print(f"Error processing {input_file}: {str(e)}")
