"""Whole-image structure maps: fit a chosen reflection - and its
centrosymmetric ("Friedel") twin - across every sliding-window patch,
producing spacing/angle/intensity maps.

Ported from the notebook's `structuremaps_set`: the reflection's nominal
position is fixed from a peak already picked interactively (not re-searched
per patch, unlike the notebook's auto-max-finding `structuremaps`), and only
its local sub-pixel fit is refined patch by patch.
"""

import math
from collections.abc import Callable

import numpy as np

from diffraction_diver.core import fft as core_fft
from diffraction_diver.core import peakfit


def mirror_position(row: int, col: int, size: int) -> tuple[int, int]:
    """The centrosymmetric twin of (row, col) in a `size`x`size` fftshift'd spectrum.

    Real-valued images have Friedel-symmetric FFTs: every genuine reflection
    has a mirror image through the zero-frequency center at (size/2, size/2).
    """
    return size - row, size - col


def mask_center(spectrum_magnitude: np.ndarray, radius: float) -> np.ndarray:
    """Zero out a disk of `radius` around the spectrum's zero-frequency center.

    The central/DC peak is always present regardless of whether the tracked
    reflection is - masking it before any presence test or fit keeps it from
    ever being mistaken for (or contaminating the background estimate
    around) the reflection actually being tracked. Matches the notebook's
    `rhighpass` highpass step.
    """
    if radius <= 0:
        return spectrum_magnitude
    size_row, size_col = spectrum_magnitude.shape
    rows, cols = np.indices(spectrum_magnitude.shape)
    r = np.sqrt((rows - size_row / 2) ** 2 + (cols - size_col / 2) ** 2)
    masked = spectrum_magnitude.copy()
    masked[r < radius] = 0
    return masked


def has_reflection(region: np.ndarray, min_peak_ratio: float = 3.0) -> bool:
    """Cheap pre-check for whether `region` plausibly contains a real peak.

    Compares the region's max against its median: a genuine Bragg reflection
    stands out sharply above the local background, while a patch with no
    detectable reflection (different phase, amorphous region, grain
    boundary, etc.) is comparatively flat. Runs before committing to the
    more expensive Gaussian fit, so batch processing over many patches with
    no signal doesn't waste time fitting noise - or worse, silently report
    a bogus fit as if it were real.
    """
    if np.count_nonzero(region == 0) > region.size * 0.25:
        # Too much of this region is exactly zero - most likely it overlaps
        # a masked-out central peak (see `mask_center`) rather than genuinely
        # flat background, so there isn't enough real data here to call: a
        # stray unmasked noise pixel shouldn't count as "found a peak".
        return False
    background = np.median(region)
    if background <= 0:
        return bool(region.max() > 0)
    return bool(region.max() / background >= min_peak_ratio)


def fit_reflection_pair(
    spectrum_magnitude: np.ndarray,
    row: int,
    col: int,
    fit_rad: int,
    min_peak_ratio: float = 3.0,
    highpass_radius: float = 0.0,
) -> dict | None:
    """Fit `(row, col)` and its centrosymmetric mirror twin.

    Returns `None` if either region has no detectable peak (see
    `has_reflection`) or its fit produces a non-finite result - callers doing
    batch processing should skip/zero-fill that patch rather than treating
    it as real. Otherwise returns a dict with both peaks' full fit
    parameters plus the derived distance (reciprocal pixels), angle
    (degrees, from the vector between the two peak centers), and mean
    height.

    `highpass_radius` masks out the always-present central/DC peak (see
    `mask_center`) before the presence test or fit runs, so it can't be
    mistaken for the tracked reflection.
    """
    spectrum_magnitude = mask_center(spectrum_magnitude, highpass_radius)
    size = spectrum_magnitude.shape[0]
    mirror_row, mirror_col = mirror_position(row, col, size)

    region1, row0_1, col0_1 = peakfit.extract_region(spectrum_magnitude, row, col, fit_rad)
    region2, row0_2, col0_2 = peakfit.extract_region(spectrum_magnitude, mirror_row, mirror_col, fit_rad)

    if not (has_reflection(region1, min_peak_ratio) and has_reflection(region2, min_peak_ratio)):
        return None

    try:
        fit1 = peakfit.fit_peak(region1)
        fit2 = peakfit.fit_peak(region2)
    except Exception:
        return None

    height1, r1, c1, width1, offset1 = fit1
    height2, r2, c2, width2, offset2 = fit2
    center1 = (row0_1 + r1, col0_1 + c1)
    center2 = (row0_2 + r2, col0_2 + c2)

    if not all(np.isfinite(v) for v in (*center1, *center2, height1, height2)):
        return None

    # d-spacing = 1/|g|, where g is the reciprocal lattice vector from the
    # zero-frequency center to ONE reflection - not the peak-to-peak
    # distance. The two fitted peaks are Friedel-symmetric twins of the same
    # vector, so halving their separation (averaging both peaks' distance
    # from center) gives that single-vector magnitude.
    distance_px = math.hypot(center2[0] - center1[0], center2[1] - center1[1]) / 2
    angle_deg = math.degrees(math.atan2(center2[0] - center1[0], center2[1] - center1[1])) % 180
    mean_height = (height1 + height2) / 2

    return {
        "region1": region1,
        "region2": region2,
        "fit1": fit1,
        "fit2": fit2,
        "center1": center1,
        "center2": center2,
        "distance_px": distance_px,
        "angle_deg": angle_deg,
        "mean_height": mean_height,
    }


def pixel_distance_to_dspacing(distance_px: float, window_size: int, pixel_size_nm: float) -> float:
    """Convert a reciprocal-space pixel distance to a real-space d-spacing (nm).

    `window_size * pixel_size_nm` is the real-space width of the FFT'd patch;
    the FFT's reciprocal-pixel spacing is its inverse, so a peak
    `distance_px` bins from its mirror twin sits at
    `distance_px / (window_size * pixel_size_nm)` cycles/nm - i.e. a spacing
    of `(window_size * pixel_size_nm) / distance_px`.
    """
    if distance_px <= 0:
        return float("inf")
    return (window_size * pixel_size_nm) / distance_px


def dspacing_to_pixel_size(distance_px: float, window_size: int, known_dspacing_nm: float) -> float:
    """Inverse of `pixel_distance_to_dspacing`: back-calculate the real-space
    pixel size from a measured reciprocal-pixel distance and a known
    d-spacing for that same reflection (calibration from a known
    standard/reflection instead of an image field-of-view).
    """
    if distance_px <= 0:
        raise ValueError("distance_px must be positive")
    return (distance_px * known_dspacing_nm) / window_size


def compute_structure_maps(
    raw_image: np.ndarray,
    window_size: int,
    window_step: int,
    hamming: bool,
    crop_size: int,
    peak_row: int,
    peak_col: int,
    fit_rad: int,
    pixel_size_nm: float | None,
    min_peak_ratio: float = 3.0,
    highpass_radius: float = 0.0,
    progress_cb: Callable[[int, int], None] | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Fit the reflection at `(peak_row, peak_col)` (in the crop-size spectrum's
    coordinates) across every sliding-window patch in `raw_image`.

    Returns (spacing_map, angle_map, intensity_map), each of shape
    (size_x, size_y). `spacing_map` is in nm if `pixel_size_nm` is given,
    otherwise it's left as a raw reciprocal-pixel distance. A patch with no
    detectable reflection gets 0 in all three maps (matches the notebook's
    zero-fill for "no signal" patches).
    """
    pos_mat, size_x, size_y = core_fft.generate_xy_positions(
        window_size, window_step, raw_image.shape[0], raw_image.shape[1]
    )
    total = size_x * size_y

    spacing_map = np.zeros(total)
    angle_map = np.zeros(total)
    intensity_map = np.zeros(total)

    for i in range(total):
        xpos, ypos = pos_mat[i]
        spectrum = core_fft.window_fft(raw_image, xpos, ypos, window_size, hamming, crop_size)
        result = fit_reflection_pair(
            np.abs(spectrum), peak_row, peak_col, fit_rad, min_peak_ratio, highpass_radius
        )
        if result is not None:
            spacing_map[i] = (
                pixel_distance_to_dspacing(result["distance_px"], window_size, pixel_size_nm)
                if pixel_size_nm is not None
                else result["distance_px"]
            )
            angle_map[i] = result["angle_deg"]
            intensity_map[i] = result["mean_height"]
        if progress_cb is not None and (i % 20 == 0 or i == total - 1):
            progress_cb(i + 1, total)

    shape = (size_x, size_y)
    return spacing_map.reshape(shape), angle_map.reshape(shape), intensity_map.reshape(shape)
