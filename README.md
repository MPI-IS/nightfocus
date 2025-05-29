# NightFocus

[![Tests](https://github.com/MPI-IS/nightfocus/actions/workflows/tests.yml/badge.svg)](https://github.com/MPI-IS/nightfocus/actions/workflows/tests.yml)

## What it is

Simple python package for automated focus optimization of astrophotography image patches.

## How to use it

1. create a sublcass of Camera for your camera. It must implement the take_picture method:


```python
from nightfocus.camera import Camera

class MyCamera(Camera):
    def take_picture(self, focus: int) -> np.ndarray:
        raise NotImplementedError
```

2. perform focus optimization:

```python
from nightfocus import optimize_focus

best_focus, _ = optimize_focus(camera, bounds=(0, 100))
```

## Under the hood

NightFocus uses Bayesian optimization to find the best focus position. The default metric used is the tenengrad focus measure:

```python
def tenengrad(image: np.ndarray, ksize: int = 3) -> float:
    """
    Tenengrad focus measure based on gradient magnitude.
    Works well for star images and is computationally efficient.
    """
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    else:
        gray = image

    gx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=ksize)
    gy = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=ksize)
    return float(np.mean(gx**2 + gy**2))
```

Other metrics are available in the `nightfocus.focus_metrics` module:

```python
from nightfocus.focus_metrics import FOCUS_MEASURES

FOCUS_MEASURES = {
    "tenengrad": tenengrad,
    "modified_laplacian": modified_laplacian,
    "normalized_variance": normalized_variance,
    "spectral_energy": spectral_energy,
    "brenner_gradient": brenner_gradient,
    "threshold_count": threshold_count,
    "fast_entropy": fast_entropy,
    "wavelet_measure": wavelet_measure,
}
```

### How we know it works

It could find the correct focus when running on the datasets located in the `images` folder. A dataset file corresponds to a corresponding tiff image on which increasing values of blur where applied.

## Command Line Interface

NightFocus includes a CLI for common tasks:

```bash
nightfocus --help
```

It provides this commands:

- crops: Create random crops from an image.
- dataset: Generate blurred dataset from TIFF files with increasing blur.
- evaluate: Evaluate focus scoring on a dataset.
- evaluate-directory: Evaluate focus scoring on a directory of images.
- evaluate-metrics: Evaluate multiple focus metrics on a dataset and print results.
- view: View images from a dataset file with their focus values.

## Installation

```bash
pip install nightfocus
```

## Author

Vincent Berenz, Max Planck Institute for Intelligent Systems, Tuebingen, Germany

