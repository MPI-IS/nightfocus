import argparse
import os

from .processing import create_random_crops


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
