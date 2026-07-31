"""Carbon Tools: workflow for graphitic/carbonaceous materials, where one
dominant reflection (graphite (002), d ~ 3.35-3.4 A) is present everywhere
but its orientation varies domain to domain.

Builds on `core/structuremaps.py`'s peak-fit-and-map machinery (this module
just adds a way to *locate* that one reflection automatically, since its
approximate spacing is already known) and ports the downstream half of
`CarbonTools.ipynb`: color-wheel orientation coloring, SLIC segmentation of
the colorized map, and lognormal/bi-lognormal domain size-distribution
fitting.
"""

import math
from collections.abc import Callable

import numpy as np
from scipy.optimize import curve_fit
from skimage.color import label2rgb
from skimage.segmentation import slic

from diffraction_diver.core import fft as core_fft
from diffraction_diver.core import structuremaps as core_structuremaps


def find_reflection_in_annulus(
    spectrum_magnitude: np.ndarray,
    r_min: float,
    r_max: float,
    highpass_radius: float = 0.0,
    min_peak_ratio: float = 3.0,
) -> tuple[int, int] | None:
    """Locate the brightest pixel within radius range `[r_min, r_max]` of center.

    A coarse locate step for a reflection whose approximate spacing is
    already known (e.g. graphite's (002) at ~3.35-3.4 A) - hand its result to
    `core.structuremaps.fit_reflection_pair` for the precise sub-pixel fit.

    Requires the annulus's own max to be at least `min_peak_ratio` times its
    own median before returning anything - this is a genuine "is there a
    peak anywhere in this whole ring" test, unlike just taking the argmax and
    later checking a small window immediately around it (`has_reflection`
    in `core.structuremaps`): picking the max of a wide region and then
    testing whether *that specific point* is locally elevated is subject to
    selection bias and tends to pass even on pure noise, since the argmax of
    any region is, almost by construction, higher than its own immediate
    surroundings. Returns `None` if no pixel falls in the annulus, or if
    nothing there rises above the noise floor.
    """
    size_row, size_col = spectrum_magnitude.shape
    rows, cols = np.indices(spectrum_magnitude.shape)
    r = np.sqrt((rows - size_row / 2) ** 2 + (cols - size_col / 2) ** 2)
    annulus = (r >= max(r_min, highpass_radius)) & (r <= r_max)
    if not annulus.any():
        return None

    values = spectrum_magnitude[annulus]
    peak_value = values.max()
    background = np.median(values)
    if background <= 0:
        if peak_value <= 0:
            return None
    elif peak_value / background < min_peak_ratio:
        return None

    masked = np.where(annulus, spectrum_magnitude, -np.inf)
    row, col = np.unravel_index(np.argmax(masked), masked.shape)
    return int(row), int(col)


def compute_orientation_maps(
    raw_image: np.ndarray,
    window_size: int,
    window_step: int,
    hamming: bool,
    crop_size: int,
    r_min: float,
    r_max: float,
    fit_rad: int,
    pixel_size_nm: float | None,
    min_peak_ratio: float = 3.0,
    highpass_radius: float = 0.0,
    mask: np.ndarray | None = None,
    progress_cb: Callable[[int, int], None] | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Fit the graphitic reflection independently for every patch.

    Unlike `core.structuremaps.compute_structure_maps` (which tracks one
    reflection from a fixed position, refining it only locally patch to
    patch - appropriate for gradual orientation drift), this re-searches the
    whole `[r_min, r_max]` annulus for every patch before fitting. Needed
    when orientation can vary across the *full* range between neighboring
    patches (tight rings, random-textured regions) rather than drifting
    gradually - a reflection that jumps far around the ring would fall
    outside a small fixed-position fit window.

    `mask`, if given, is a boolean array the same shape as `raw_image`
    (typically thresholded from a companion dark-field image - see
    `threshold_mask`) marking where material is actually present; a patch
    centered on a `False` pixel skips the FFT and detection entirely and is
    zero-filled, rather than relying solely on the (noisier) FFT-domain
    signal-presence test to reject empty/vacuum regions.

    Returns (spacing_map, angle_map, intensity_map), each of shape
    (size_x, size_y). A patch with no reflection found in the annulus, or
    whose fit fails, gets 0 in all three maps.
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
        if mask is not None:
            center_row = min(xpos + window_size // 2, mask.shape[0] - 1)
            center_col = min(ypos + window_size // 2, mask.shape[1] - 1)
            if not mask[center_row, center_col]:
                if progress_cb is not None and (i % 20 == 0 or i == total - 1):
                    progress_cb(i + 1, total)
                continue

        magnitude = np.abs(core_fft.window_fft(raw_image, xpos, ypos, window_size, hamming, crop_size))
        found = find_reflection_in_annulus(magnitude, r_min, r_max, highpass_radius, min_peak_ratio)
        if found is not None:
            row, col = found
            result = core_structuremaps.fit_reflection_pair(
                magnitude, row, col, fit_rad, min_peak_ratio, highpass_radius
            )
            if result is not None:
                spacing_map[i] = (
                    core_structuremaps.pixel_distance_to_dspacing(result["distance_px"], window_size, pixel_size_nm)
                    if pixel_size_nm is not None
                    else result["distance_px"]
                )
                angle_map[i] = result["angle_deg"]
                intensity_map[i] = result["mean_intensity"]
        if progress_cb is not None and (i % 20 == 0 or i == total - 1):
            progress_cb(i + 1, total)

    shape = (size_x, size_y)
    return spacing_map.reshape(shape), angle_map.reshape(shape), intensity_map.reshape(shape)


def threshold_mask(companion_image: np.ndarray, threshold: float) -> np.ndarray:
    """Binary material mask from a companion dark-field/HAADF-style image.

    Dark-field imaging is bright where material scatters and dark where
    there's none (vacuum reads black) - a single, fixed polarity, unlike
    bright-field (where material vs. vacuum contrast can go either way
    depending on thickness/defocus, so it isn't used for masking here).
    """
    return companion_image > threshold


def generate_color_wheel() -> np.ndarray:
    """Cyclic red -> green -> blue RGB lookup table (361 entries, uint8).

    Ported from the notebook's `generateColorGradient` (two concatenated
    181-step linear gradients). Index with `colorize_orientation`, which
    handles the angle-to-index mapping for our `[0, 180)` angle convention
    (see `core.structuremaps.fit_reflection_pair`) - the notebook's own
    indexing additionally shifted by +90 first, to convert its `[-90, 90)`
    convention into `[0, 180)`.
    """

    def _segment(rgb1, rgb2, steps):
        return [[a + k * (b - a) / (steps - 1) for a, b in zip(rgb1, rgb2)] for k in range(steps)]

    red_to_green = _segment((255, 0, 0), (0, 255, 0), 180)  # k=0..179, excludes green
    green_to_blue = _segment((0, 255, 0), (0, 0, 255), 181)  # k=0..180, includes blue
    wheel = np.array(red_to_green + green_to_blue)
    return np.clip(wheel, 0, 255).astype(np.uint8)


def colorize_orientation(
    angle_deg_map: np.ndarray, color_wheel: np.ndarray, intensity_map: np.ndarray | None = None
) -> np.ndarray:
    """Map an angle map (degrees, `[0, 180)` convention) to an RGB image via `color_wheel`.

    If `intensity_map` is given, it's normalized against its own raw max
    (ignoring the zero "no reflection" sentinel) and multiplies the RGB -
    the "scaled by intensity" pre-segmentation view. `intensity_map` is now
    the raw regional mean around each fitted peak (see
    `structuremaps.fit_reflection_pair`), not the fitted Gaussian height, so
    it's naturally far less noisy patch to patch - no percentile clipping
    needed to keep outliers from washing out the rest of the map.
    """
    n = len(color_wheel)
    index = np.clip((angle_deg_map * (n - 1) / 180.0).astype(int), 0, n - 1)
    rgb = color_wheel[index].astype(np.float64)
    if intensity_map is not None:
        valid = intensity_map[intensity_map != 0]
        if valid.size == 0:
            scale = np.zeros_like(intensity_map, dtype=np.float64)
        else:
            max_intensity = valid.max()
            scale = intensity_map / max_intensity if max_intensity > 0 else np.zeros_like(intensity_map, dtype=np.float64)
        rgb = rgb * scale[..., None]
    return np.clip(rgb, 0, 255).astype(np.uint8)


def segment_orientation_map(color_image: np.ndarray, n_segments: int = 600, compactness: float = 20.0) -> np.ndarray:
    """SLIC superpixel segmentation of a colorized orientation image."""
    return slic(color_image, n_segments=n_segments, compactness=compactness)


def label_to_average_rgb(labels: np.ndarray, color_image: np.ndarray) -> np.ndarray:
    """Each segment's pixels replaced by their average color, for visualization."""
    return label2rgb(labels, color_image, kind="avg")


def segment_areas(labels: np.ndarray, map_pixel_area: float) -> np.ndarray:
    """Physical area (in whatever unit `map_pixel_area` is, e.g. nm^2) of each segment.

    `map_pixel_area` is the area one map pixel represents - for a structure
    map that's `(window_step * pixel_size)**2`, since adjacent map positions
    are `window_step` raw-image pixels apart (see
    `core.fft.generate_xy_positions`), not `window_size` apart.
    """
    _segment_ids, counts = np.unique(labels, return_counts=True)
    return counts * map_pixel_area


def lognormal(x: np.ndarray, ampl: float, width: float, xc: float, offset: float) -> np.ndarray:
    """Lognormal-shaped curve (not a normalized PDF - `ampl`/`offset` let it match raw histogram counts)."""
    return offset + (ampl / (np.sqrt(2 * math.pi) * width * x)) * np.exp(-(np.log(x / xc) ** 2) / (2 * width**2))


def _lognormal_initial_guess(xdata: np.ndarray, ydata: np.ndarray) -> tuple[float, float, float, float]:
    """Moment-based initial guess - the notebook calls `curve_fit` with no
    initial guess at all, which is fragile for this shape."""
    weights = np.clip(ydata, 0, None)
    total = weights.sum()
    if total <= 0:
        xc = float(np.median(xdata))
        width = 0.5
    else:
        xc = float(np.average(xdata, weights=weights))
        variance = np.average((xdata - xc) ** 2, weights=weights)
        width = max(0.1, math.sqrt(variance) / max(xc, 1e-9))
    ampl = float(ydata.max() * width * math.sqrt(2 * math.pi) * max(xc, 1e-9))
    offset = float(max(0.0, ydata.min()))
    return ampl, width, xc, offset


def fit_lognormal(xdata: np.ndarray, ydata: np.ndarray) -> tuple[float, float, float, float]:
    """Fit `lognormal` to a histogram (`xdata`=bin centers/upper edges, `ydata`=counts)."""
    xdata = np.asarray(xdata, dtype=float)
    ydata = np.asarray(ydata, dtype=float)
    guess = _lognormal_initial_guess(xdata, ydata)
    popt, _ = curve_fit(lognormal, xdata, ydata, p0=guess, maxfev=10000)
    return tuple(popt)


def bi_lognormal(
    x: np.ndarray,
    ampl1: float,
    width1: float,
    xc1: float,
    ampl2: float,
    width2: float,
    xc2: float,
    offset: float,
) -> np.ndarray:
    """Sum of two `lognormal`-shaped curves sharing one `offset`."""
    return (
        offset
        + (ampl1 / (np.sqrt(2 * math.pi) * width1 * x)) * np.exp(-(np.log(x / xc1) ** 2) / (2 * width1**2))
        + (ampl2 / (np.sqrt(2 * math.pi) * width2 * x)) * np.exp(-(np.log(x / xc2) ** 2) / (2 * width2**2))
    )


def fit_bi_lognormal(
    xdata: np.ndarray, ydata: np.ndarray
) -> tuple[float, float, float, float, float, float, float]:
    """Fit `bi_lognormal`, splitting the data at its weighted median for the
    two components' initial guesses (the notebook calls `curve_fit` with no
    initial guess at all, which is fragile for a 7-parameter fit)."""
    xdata = np.asarray(xdata, dtype=float)
    ydata = np.asarray(ydata, dtype=float)
    weights = np.clip(ydata, 0, None)
    cum = np.cumsum(weights)
    if cum[-1] <= 0:
        split = len(xdata) // 2
    else:
        split = int(np.searchsorted(cum, cum[-1] / 2))
    split = min(max(split, 1), len(xdata) - 1)

    ampl1, width1, xc1, offset1 = _lognormal_initial_guess(xdata[:split], ydata[:split])
    ampl2, width2, xc2, offset2 = _lognormal_initial_guess(xdata[split:], ydata[split:])
    guess = (ampl1, width1, xc1, ampl2, width2, xc2, (offset1 + offset2) / 2)
    popt, _ = curve_fit(bi_lognormal, xdata, ydata, p0=guess, maxfev=10000)
    return tuple(popt)
