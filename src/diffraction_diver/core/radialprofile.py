"""Radial (azimuthally-averaged) FFT intensity profiles.

A patch's radial profile is computed once per sliding-window position across
the whole image, into a 3D dataset (size_x, size_y, n_radius_bins). Scanning
"which spacing shows where" afterward is then just summing a slice of this
already-computed cube - no new FFTs or profiles need to be recomputed as the
user moves a slider.

The profile itself replaces the notebook's `fftint`/`profpeak` radial-binning
(argsort + cumsum over sorted radii) with an equivalent but simpler
`np.bincount`-based azimuthal average over integer-pixel radius bins.
"""

from collections.abc import Callable

import numpy as np

from diffraction_diver.core import fft as core_fft


def radial_profile(spectrum_magnitude: np.ndarray) -> np.ndarray:
    """Azimuthally-averaged intensity vs. integer-pixel radius from center.

    Returns an array indexed by radius bin (0, 1, 2, ...), each entry the
    mean magnitude of all pixels at that integer radius from the
    zero-frequency center. Truncated at the largest radius that's still a
    *complete* circle within the (square) array - beyond that, bins only
    include the few pixels near the corners, so their average is built from
    too little data and tends to swing noisily rather than reflect anything
    real. Length depends only on `spectrum_magnitude.shape` (specifically
    `min(shape) // 2 + 1`), so it's identical for every patch sharing the
    same crop size.
    """
    size_row, size_col = spectrum_magnitude.shape
    rows, cols = np.indices(spectrum_magnitude.shape)
    r = np.sqrt((rows - size_row / 2) ** 2 + (cols - size_col / 2) ** 2)
    r_bin = r.astype(np.int64).ravel()

    sums = np.bincount(r_bin, weights=spectrum_magnitude.ravel())
    counts = np.bincount(r_bin)
    with np.errstate(invalid="ignore", divide="ignore"):
        profile = sums / counts
    profile = np.nan_to_num(profile)

    max_radius = min(size_row, size_col) // 2
    return profile[: max_radius + 1]


def compute_radial_profile_stack(
    raw_image: np.ndarray,
    window_size: int,
    window_step: int,
    hamming: bool,
    crop_size: int,
    progress_cb: Callable[[int, int], None] | None = None,
) -> tuple[np.ndarray, int, int]:
    """Compute the radial profile of every sliding-window patch in `raw_image`.

    Returns (profile_stack[size_x, size_y, n_bins], size_x, size_y).
    """
    pos_mat, size_x, size_y = core_fft.generate_xy_positions(
        window_size, window_step, raw_image.shape[0], raw_image.shape[1]
    )
    total = size_x * size_y
    if total == 0:
        raise ValueError(
            "No window positions fit in this image with the given window_size/window_step."
        )

    # Number of radius bins depends only on crop_size - compute it once from
    # the first patch, then fill the rest into a preallocated array.
    xpos0, ypos0 = pos_mat[0]
    first_profile = radial_profile(np.abs(core_fft.window_fft(raw_image, xpos0, ypos0, window_size, hamming, crop_size)))
    stack = np.zeros((total, first_profile.shape[0]))
    stack[0] = first_profile

    for i in range(1, total):
        xpos, ypos = pos_mat[i]
        spectrum = core_fft.window_fft(raw_image, xpos, ypos, window_size, hamming, crop_size)
        stack[i] = radial_profile(np.abs(spectrum))
        if progress_cb is not None and (i % 100 == 0 or i == total - 1):
            progress_cb(i + 1, total)

    return stack.reshape(size_x, size_y, first_profile.shape[0]), size_x, size_y


def subtract_background(profile: np.ndarray, outer_fraction: float = 0.2) -> np.ndarray:
    """Subtract each profile's own outer-radius average as a flat background.

    Operates along the last axis, so it works the same way on a single
    profile (1D) or a whole stack (any leading shape, e.g.
    `(size_x, size_y, n_bins)`) - each profile gets its own background
    estimate from its own outer radius range, not one value shared across
    every patch. Cheap enough to apply on the fly at display time rather
    than baking it into the stored (raw) profile/stack.
    """
    n_bins = profile.shape[-1]
    outer_bins = max(1, round(n_bins * outer_fraction))
    background = profile[..., n_bins - outer_bins :].mean(axis=-1, keepdims=True)
    return profile - background


def integrate_window(profile_stack: np.ndarray, center_bin: int, width: int) -> np.ndarray:
    """Sum `profile_stack` over radius bins centered on `center_bin`.

    `width` is the total number of bins in the integration window. Returns a
    2D (size_x, size_y) map of integrated intensity for that radial band.
    """
    n_bins = profile_stack.shape[2]
    half = max(1, width) // 2
    lo = max(0, center_bin - half)
    hi = min(n_bins, center_bin + half + 1)
    return profile_stack[:, :, lo:hi].sum(axis=2)
