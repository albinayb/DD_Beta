"""Loading raw images/arrays for sliding-FFT analysis."""

from pathlib import Path

import matplotlib.image as mpimg
import numpy as np
from skimage import color

NPY_EXTENSIONS = {".npy"}


def load_input(path: str | Path) -> np.ndarray:
    """Load `path` as a 2D grayscale array.

    `.npy` files are loaded directly (assumed to already be a 2D array).
    Anything else is loaded as an image and converted to grayscale.
    """
    path = Path(path)
    if path.suffix.lower() in NPY_EXTENSIONS:
        arr = np.load(path)
    else:
        arr = color.rgb2gray(mpimg.imread(path))

    arr = np.asarray(arr)
    if arr.ndim != 2:
        raise ValueError(
            f"Expected a 2D grayscale array, got shape {arr.shape} from {path.name}"
        )
    return arr
