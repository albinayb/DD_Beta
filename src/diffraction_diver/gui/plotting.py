"""Matplotlib figure builders for the raw-image/patch preview and result grids."""

import math

import numpy as np
from matplotlib.figure import Figure
from matplotlib.patches import Rectangle
from mpl_toolkits.axes_grid1 import make_axes_locatable


def draw_raw_image(fig: Figure, image: np.ndarray, rect: tuple[int, int, int] | None = None) -> None:
    """(Re)draw the raw image into `fig`, with an optional window-position overlay.

    `rect` is (xpos, ypos, window_size) using the same (row, col) convention as
    `core.fft` — drawn as a rectangle in (col, row) = (x, y) image coordinates.
    """
    fig.clear()
    ax = fig.add_subplot(111)
    ax.imshow(image, cmap="gray", interpolation="none")
    ax.set_xticks([])
    ax.set_yticks([])
    if rect is not None:
        xpos, ypos, size = rect
        ax.add_patch(
            Rectangle((ypos, xpos), size, size, edgecolor="red", facecolor="none", linewidth=1.5)
        )
    fig.tight_layout()


def draw_patch_preview(fig: Figure, patch: np.ndarray, spectrum: np.ndarray) -> None:
    """(Re)draw a two-panel patch + FFT-magnitude preview into `fig`."""
    fig.clear()
    ax1 = fig.add_subplot(1, 2, 1)
    ax1.imshow(patch, cmap="gray", interpolation="none")
    ax1.set_title("Patch", fontsize=9)
    ax1.set_xticks([])
    ax1.set_yticks([])

    ax2 = fig.add_subplot(1, 2, 2)
    ax2.imshow(np.log(np.abs(spectrum) + 1e-8), interpolation="none")
    ax2.set_title("FFT (log magnitude)", fontsize=9)
    ax2.set_xticks([])
    ax2.set_yticks([])
    fig.tight_layout()


def draw_diffractogram(
    fig: Figure, spectrum: np.ndarray, selections: list[tuple[int, int, int, str]] | None = None
) -> None:
    """(Re)draw a log-magnitude diffractogram, with optional peak-selection overlays.

    `selections` is a list of (row, col, fit_rad, color) in `spectrum` array
    coordinates - e.g. one for the picked peak, one for its symmetric twin.
    """
    fig.clear()
    ax = fig.add_subplot(111)
    ax.imshow(np.log(np.abs(spectrum) + 1e-8), interpolation="none")
    ax.set_xticks([])
    ax.set_yticks([])
    for row, col, fit_rad, color in selections or []:
        ax.plot(col, row, "+", color=color, markersize=14, markeredgewidth=2)
        ax.add_patch(
            Rectangle(
                (col - fit_rad, row - fit_rad),
                2 * fit_rad,
                2 * fit_rad,
                edgecolor=color,
                facecolor="none",
                linewidth=1.2,
            )
        )
    fig.tight_layout()


def draw_peak_pair_fit(
    fig: Figure, region1: np.ndarray, model1: np.ndarray, region2: np.ndarray, model2: np.ndarray
) -> None:
    """(Re)draw a 2x2 comparison: primary/mirror peak data vs. fitted model.

    A dedicated 2x2 layout (rather than the general `draw_grid`) since this
    panel lives in a narrow fixed-width column - a 1x4 row wouldn't fit.
    """
    fig.clear()
    panels = [
        (region1, "Peak 1 data"),
        (model1, "Peak 1 fit"),
        (region2, "Peak 2 (mirror) data"),
        (model2, "Peak 2 (mirror) fit"),
    ]
    for index, (image, title) in enumerate(panels):
        ax = fig.add_subplot(2, 2, index + 1)
        ax.imshow(image, interpolation="none")
        ax.set_title(title, fontsize=8)
        ax.set_xticks([])
        ax.set_yticks([])
    fig.tight_layout()


def draw_radial_profile(
    fig: Figure, profile: np.ndarray, band: tuple[int, int] | None = None, skip_bins: int = 3
) -> None:
    """(Re)draw a radial (azimuthally-averaged) intensity profile line plot.

    `band` is an optional (lo, hi) radius-bin range to highlight - e.g. the
    integration window currently selected by a scan slider elsewhere.

    The always-present central/DC peak is typically orders of magnitude
    brighter than the Bragg reflections that actually matter, so the y-axis
    is autoscaled from radius bin `skip_bins` onward instead of from 0 -
    otherwise every other peak would be flattened to an invisible sliver.
    """
    fig.clear()
    ax = fig.add_subplot(111)
    ax.plot(np.arange(len(profile)), profile, "b-")
    if band is not None:
        lo, hi = band
        ax.axvspan(lo, hi - 1, color="red", alpha=0.25)
    visible = profile[skip_bins:]
    if visible.size > 0 and visible.max() > 0:
        ax.set_ylim(0, visible.max() * 1.1)
    ax.set_xlabel("Radius (px)")
    ax.set_ylabel("Mean intensity")
    ax.set_title("Radial Profile")
    fig.tight_layout()


def _grid_shape(n: int) -> tuple[int, int]:
    ncols = min(4, n)
    nrows = math.ceil(n / ncols)
    return nrows, ncols


def percentile_clip(values: np.ndarray, low: float = 5.0, high: float = 95.0) -> tuple[float, float] | None:
    """(vmin, vmax) excluding the bottom/top `low`/`high` percentile tails.

    Ignores exact zeros - the "no reflection detected" sentinel used in
    structure maps - so a large undetected area doesn't skew the range.
    Returns None if there's no valid data to compute a range from (falls
    back to imshow's own auto-scaling).
    """
    valid = values[values != 0]
    if valid.size == 0:
        return None
    return float(np.percentile(valid, low)), float(np.percentile(valid, high))


def draw_grid(
    fig: Figure,
    images: list[np.ndarray],
    titles: list[str],
    suptitle: str | None = None,
    cmap: str = "viridis",
    ncols: int | None = None,
    vlimits: list[tuple[float, float] | None] | None = None,
) -> None:
    """(Re)draw a grid of images (with colorbars) into `fig`.

    `ncols` overrides the automatic layout (e.g. `ncols=1` to stack images
    vertically instead). `vlimits`, if given, is one (vmin, vmax) or None
    per image, overriding imshow's default full-range auto-scaling for that
    panel (e.g. to clip outlier-dominated color scales).
    """
    fig.clear()
    n = len(images)
    nrows, ncols = _grid_shape(n) if ncols is None else (math.ceil(n / ncols), ncols)
    fig.set_size_inches(3.2 * ncols, 3.0 * nrows)
    if suptitle:
        fig.suptitle(suptitle)

    for index, (image, title) in enumerate(zip(images, titles)):
        vmin, vmax = (vlimits[index] or (None, None)) if vlimits else (None, None)
        ax = fig.add_subplot(nrows, ncols, index + 1)
        im = ax.imshow(image, cmap=cmap, interpolation="none", vmin=vmin, vmax=vmax)
        ax.set_title(title, fontsize=9)
        ax.set_xticks([])
        ax.set_yticks([])
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.08)
        fig.colorbar(im, cax=cax)

    fig.tight_layout()


def make_grid_figure(
    images: list[np.ndarray],
    titles: list[str],
    suptitle: str | None = None,
    cmap: str = "viridis",
    ncols: int | None = None,
    vlimits: list[tuple[float, float] | None] | None = None,
) -> Figure:
    """Build a new Figure with one imshow + colorbar per image, arranged in a grid."""
    n = len(images)
    grid_ncols = ncols if ncols is not None else _grid_shape(n)[1]
    grid_nrows = math.ceil(n / grid_ncols)
    fig = Figure(figsize=(3.2 * grid_ncols, 3.0 * grid_nrows))
    draw_grid(fig, images, titles, suptitle=suptitle, cmap=cmap, ncols=ncols, vlimits=vlimits)
    return fig


def scree_figure(singular_values: np.ndarray) -> Figure:
    fig = Figure(figsize=(5, 3.5))
    ax = fig.add_subplot(111)
    ax.semilogy(np.arange(1, len(singular_values) + 1), singular_values, "b-o")
    ax.set_xlabel("Principal Component")
    ax.set_ylabel("Singular value")
    ax.set_title("Scree Plot")
    fig.tight_layout()
    return fig


def map_images(maps: np.ndarray, n: int) -> list[np.ndarray]:
    """`maps` has shape (size_x, size_y, n) — abundance/mixing/loading coefficients.

    size_x/size_y follow the same (row, col) convention as the raw image (see
    `core.fft.generate_xy_positions`), so no rotation/flip is applied here -
    that would otherwise misalign the maps against the raw-image preview.
    """
    return [maps[:, :, i] for i in range(n)]
