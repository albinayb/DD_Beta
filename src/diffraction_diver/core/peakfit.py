"""2D Gaussian fitting for a single peak in a diffraction pattern.

Ported from the notebook's `gaussian` / `moments` / `fitgaussian`, generalized
to fit any region (not just a symmetric peak pair) and to clamp cleanly at
image edges instead of the ad hoc if-clamps duplicated in `structuremaps` /
`structuremaps_set`.
"""

from collections.abc import Callable

import numpy as np
from scipy import optimize


def gaussian_2d(height: float, center_row: float, center_col: float, width: float, offset: float) -> Callable:
    """Returns a callable 2D Gaussian-plus-offset: fn(rows, cols) -> values."""
    width = float(width)

    def fn(rows, cols):
        return offset + height * np.exp(
            -(((center_row - rows) / width) ** 2 + ((center_col - cols) / width) ** 2) / 2
        )

    return fn


def _initial_guess(region: np.ndarray) -> tuple[float, float, float, float, float]:
    """Parameter guess from the region's moments: (height, row, col, width, offset)."""
    total = region.sum()
    row_idx, col_idx = np.indices(region.shape)
    row = (row_idx * region).sum() / total
    col = (col_idx * region).sum() / total

    row_i = int(np.clip(row, 0, region.shape[0] - 1))
    col_i = int(np.clip(col, 0, region.shape[1] - 1))
    col_slice = region[:, col_i]
    width_row = np.sqrt(np.abs((np.arange(col_slice.size) - row) ** 2 * col_slice).sum() / col_slice.sum())
    row_slice = region[row_i, :]
    width_col = np.sqrt(np.abs((np.arange(row_slice.size) - col) ** 2 * row_slice).sum() / row_slice.sum())

    return region.max(), row, col, min(width_row, width_col), 0.0


def extract_region(image: np.ndarray, row: int, col: int, fit_rad: int) -> tuple[np.ndarray, int, int]:
    """Clamp (row, col) so a `2*fit_rad` square around it stays inside `image`.

    Returns (region, row0, col0), where (row0, col0) is the region's top-left
    corner in `image` coordinates, so callers can convert a fit result in the
    region's local coordinates back to absolute image coordinates.
    """
    height, width = image.shape
    row = int(np.clip(row, fit_rad, height - fit_rad))
    col = int(np.clip(col, fit_rad, width - fit_rad))
    region = image[row - fit_rad : row + fit_rad, col - fit_rad : col + fit_rad]
    return region, row - fit_rad, col - fit_rad


def fit_peak(region: np.ndarray) -> tuple[float, float, float, float, float]:
    """Fit a 2D Gaussian-plus-offset to `region`.

    Returns (height, center_row, center_col, width, offset), all in `region`'s
    own local pixel coordinates.
    """
    guess = _initial_guess(region)
    rows, cols = np.indices(region.shape)

    def residuals(params):
        return np.ravel(gaussian_2d(*params)(rows, cols) - region)

    fit, _ = optimize.leastsq(residuals, guess)
    return tuple(fit)
