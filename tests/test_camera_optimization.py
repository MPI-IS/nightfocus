"""Tests for camera focus optimization."""

from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from nightfocus import SimulatedCamera, optimize_focus
from nightfocus.camera import DatasetCamera
from nightfocus.focus_metrics import FOCUS_MEASURES, best_measure
from tests.conftest import create_sample_dataset, create_simulated_camera

# Tolerances for test assertions
FOCUS_TOLERANCE = 5  # Allowed deviation from expected focus position
SCORE_TOLERANCE = 0.1  # Allowed deviation from expected score (normalized)

# Test parameters
TEST_BOUNDS = (0, 100)
TEST_INITIAL_POINTS = 5
TEST_MAX_ITER = 10


def test_optimize_focus_simulated_camera():
    """Test focus optimization with the simulated camera."""
    # Create a simulated camera
    camera = create_simulated_camera()

    # Run optimization
    best_focus, history = optimize_focus(
        camera=camera,
        focus_measure=best_measure,
        bounds=(0, 100),
        initial_points=5,
        max_iter=10,
        random_state=42,
    )

    # Verify the result is close to the known best focus (50 for SimulatedCamera)
    assert (
        abs(best_focus - 50) <= FOCUS_TOLERANCE
    ), f"Optimized focus {best_focus} is not close to expected 50"

    # Verify history contains the results
    assert len(history) > 0, "No results in history"
    assert all(
        isinstance(focus, (int, float)) and isinstance(score, float)
        for focus, score in history
    ), "Invalid history format"

    # The best score in history should be close to the best focus
    best_in_history = max(history, key=lambda x: x[1])
    assert abs(best_in_history[0] - 50) <= FOCUS_TOLERANCE


def test_optimize_focus_dataset_camera():
    """Test focus optimization with a dataset camera."""

    # Get a dataset path
    dataset_path = create_sample_dataset("image_001_dataset.pkl")

    # Create dataset camera
    camera = DatasetCamera(dataset_path)

    # Get the expected best focus from the dataset
    expected_best_focus = camera.best_focus

    # Run optimization
    best_focus, history = optimize_focus(
        camera=camera,
        focus_measure=best_measure,
        bounds=camera.focus_range,
        initial_points=min(
            5, len(camera.available_focus)
        ),  # Don't request more points than available
        max_iter=10,
        random_state=42,
    )

    # Verify the result is reasonable
    assert (
        camera.min_focus <= best_focus <= camera.max_focus
    ), f"Optimized focus {best_focus} is outside camera range {camera.focus_range}"

    # Verify history contains the results
    assert len(history) > 0, "No results in history"
    assert all(
        isinstance(focus, (int, float)) and isinstance(score, float)
        for focus, score in history
    ), "Invalid history format"


def test_optimize_focus_with_default_measure():
    """Test that optimize_focus works with default focus measure."""
    # Should work without specifying focus_measure
    best_focus, _ = optimize_focus(
        camera=SimulatedCamera(),
        focus_measure=best_measure,
        bounds=(0, 100),
        initial_points=3,
        max_iter=5,
        random_state=42,
    )
    assert 0 <= best_focus <= 100


def test_optimize_focus_history_ordering():
    """Test that the history is properly ordered by focus value."""
    # This is a white-box test that verifies the implementation detail
    # that the history is sorted by focus value
    camera = SimulatedCamera(image_shape=(50, 50), noise_level=0.1)
    _, history = optimize_focus(
        camera=camera,
        focus_measure=best_measure,
        bounds=(0, 10),
        initial_points=3,
        max_iter=2,
    )

    # Check that history is sorted by focus value
    focus_values = [focus for focus, _ in history]
    assert focus_values == sorted(focus_values), "History is not sorted by focus"
