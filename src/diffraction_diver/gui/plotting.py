"""Matplotlib figure builders for the raw-image/patch preview and result grids."""

import math

import numpy as np
from matplotlib import patheffects as pe
from matplotlib.figure import Figure
from matplotlib.patches import Rectangle
from mpl_toolkits.axes_grid1 import make_axes_locatable


def choose_nice_scale_length(field_of_view: float) -> float:
    """Pick a round 1/2/5 x 10^n length near ~20% of `field_of_view`."""
    target = max(field_of_view * 0.2, 1e-9)
    exponent = math.floor(math.log10(target))
    for candidate in (1, 2, 5, 10):
        value = candidate * 10**exponent
        if value >= target:
            return value
    return 10 * 10**exponent


def draw_scale_bar(
    ax,
    pixel_size: float,
    units: str = "nm",
    length: float | None = None,
    color: str = "white",
    fontsize: float = 10,
    text: bool = True,
) -> None:
    """Draw a scale bar in the bottom-left corner of an already-`imshow`'d `ax`.

    `pixel_size` is in `units` per pixel. `length`, if not given, is picked
    automatically via `choose_nice_scale_length` from the axes' displayed
    width. Ported from the notebook's `AddScaleBar`, reading the axes'
    current limits instead of introspecting an image's extent directly.
    """
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    width_px = abs(xlim[1] - xlim[0])
    height_px = abs(ylim[1] - ylim[0])
    if length is None:
        length = choose_nice_scale_length(width_px * pixel_size)

    bar_px = length / pixel_size
    bar_thickness = height_px / 50.0
    x0 = min(xlim) + bar_thickness
    y0 = max(ylim) - 2 * bar_thickness  # near the bottom edge (row-major, origin='upper')

    ax.add_patch(Rectangle((x0, y0), bar_px, bar_thickness, fc=color, ec="black", lw=0.5))
    if text:
        label = ax.text(
            x0 + bar_px / 2,
            y0,
            f"{length:g} {units}",
            fontweight="bold",
            color=color,
            fontsize=fontsize,
            ha="center",
            va="bottom",
        )
        label.set_path_effects([pe.withStroke(linewidth=1, foreground="black")])


def draw_raw_image(
    fig: Figure,
    image: np.ndarray,
    rect: tuple[int, int, int] | None = None,
    pixel_size: float | None = None,
    pixel_size_units: str = "nm",
) -> None:
    """(Re)draw the raw image into `fig`, with an optional window-position overlay.

    `rect` is (xpos, ypos, window_size) using the same (row, col) convention as
    `core.fft` — drawn as a rectangle in (col, row) = (x, y) image coordinates.
    A scale bar is drawn when `pixel_size` (units per raw-image pixel) is given.
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
    if pixel_size is not None:
        draw_scale_bar(ax, pixel_size, units=pixel_size_units)
    fig.tight_layout()


def draw_rgb_image(fig: Figure, rgb_image: np.ndarray, title: str, scale_bar: dict | None = None) -> None:
    """(Re)draw an already-RGB image (e.g. a colorized orientation map) with a title, no colorbar."""
    fig.clear()
    ax = fig.add_subplot(111)
    ax.imshow(rgb_image, interpolation="none")
    ax.set_title(title, fontsize=10)
    ax.set_xticks([])
    ax.set_yticks([])
    if scale_bar is not None:
        draw_scale_bar(ax, **scale_bar)
    fig.tight_layout()


def _color_wheel_disc(color_wheel: np.ndarray, size: int = 220) -> np.ndarray:
    """Render `color_wheel` as a circular disc (RGBA).

    The disc's full 360-degree visual rotation maps directly onto the
    lookup table's full range, matching `colorize_orientation`'s
    `angle * (n-1) / 180` indexing: the *data* angle is an undirected axis
    spanning only `[0, 180)`, doubled before indexing, so one full turn of
    this disc shows that entire range with no color repeated - 0 and ~180
    (data) land at the same visual spot, which is correct (they're the same
    axis orientation).
    """
    y, x = np.indices((size, size))
    center = (size - 1) / 2
    dx = x - center
    dy = center - y  # flip so +y (screen up) is standard mathematical "up"
    radius = np.sqrt(dx**2 + dy**2)
    theta = np.degrees(np.arctan2(dy, dx)) % 360.0

    n = len(color_wheel)
    index = np.clip((theta * (n - 1) / 360.0).astype(int), 0, n - 1)
    rgb = color_wheel[index]

    alpha = np.where(radius <= center, 255, 0).astype(np.uint8)
    return np.dstack([rgb, alpha]).astype(np.uint8)


def draw_orientation_with_legend(
    fig: Figure,
    rgb_image: np.ndarray,
    color_wheel: np.ndarray,
    title: str,
    scale_bar: dict | None = None,
) -> None:
    """(Re)draw a color-wheel-colorized orientation image with a color-wheel legend.

    The legend disc uses the same `color_wheel` LUT as
    `core.carbontools.colorize_orientation`, so the colors in the main image
    can actually be read quantitatively.
    """
    fig.clear()
    ax_img = fig.add_axes((0.03, 0.05, 0.72, 0.9))
    ax_img.imshow(rgb_image, interpolation="none")
    ax_img.set_title(title, fontsize=10)
    ax_img.set_xticks([])
    ax_img.set_yticks([])
    if scale_bar is not None:
        draw_scale_bar(ax_img, **scale_bar)

    ax_legend = fig.add_axes((0.78, 0.32, 0.2, 0.36))
    disc = _color_wheel_disc(color_wheel)
    size = disc.shape[0]
    ax_legend.imshow(disc, interpolation="none")
    ax_legend.set_xticks([])
    ax_legend.set_yticks([])
    ax_legend.set_title("Orientation", fontsize=8)
    ax_legend.axis("off")

    center = (size - 1) / 2
    label_radius = center * 1.18
    for data_angle in (0, 45, 90, 135):
        visual_theta = math.radians(2 * data_angle)
        tx = center + label_radius * math.cos(visual_theta)
        ty = center - label_radius * math.sin(visual_theta)
        ax_legend.text(tx, ty, f"{data_angle}°", ha="center", va="center", fontsize=7)
    ax_legend.set_xlim(-0.3 * size, 1.3 * size)
    ax_legend.set_ylim(1.3 * size, -0.3 * size)
    # No fig.tight_layout() here - both axes are manually positioned via
    # add_axes rects (needed for the fixed-size legend disc), and
    # tight_layout only warns/does nothing useful with manual placement.


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


def draw_labeled_image(
    fig: Figure,
    image: np.ndarray,
    title: str,
    cmap: str = "viridis",
    vlimits: tuple[float, float] | None = None,
    scale_bar: dict | None = None,
) -> None:
    """(Re)draw a single image with a colorbar and title into `fig`.

    Unlike `draw_grid`/`make_grid_figure`, this never changes the figure's
    size - it's for a canvas that's redrawn repeatedly in place (e.g. a live
    scan display), where a figure that resizes itself on every redraw (as
    `draw_grid` does, sized for an N-image grid) causes visible instability
    in the embedding Qt widget. `scale_bar`, if given, is a dict of kwargs
    forwarded to `draw_scale_bar`.
    """
    fig.clear()
    ax = fig.add_subplot(111)
    vmin, vmax = vlimits or (None, None)
    im = ax.imshow(image, cmap=cmap, interpolation="none", vmin=vmin, vmax=vmax)
    ax.set_title(title, fontsize=10)
    ax.set_xticks([])
    ax.set_yticks([])
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.08)
    fig.colorbar(im, cax=cax)
    if scale_bar is not None:
        draw_scale_bar(ax, **scale_bar)
    fig.tight_layout()


def draw_radial_profile(
    fig: Figure,
    profile: np.ndarray,
    band: tuple[int, int] | None = None,
    skip_bins: int = 3,
    title: str = "Radial Profile",
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
    ax.set_title(title)
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
    scale_bar: dict | None = None,
) -> None:
    """(Re)draw a grid of images (with colorbars) into `fig`.

    `ncols` overrides the automatic layout (e.g. `ncols=1` to stack images
    vertically instead). `vlimits`, if given, is one (vmin, vmax) or `None`
    per image, overriding imshow's default full-range auto-scaling for that
    panel - e.g. pass `percentile_clip(image)` per panel to clip
    outlier-dominated color scales. There is no automatic default: PCA/ICA/NMF
    grids want full range (signed/non-negative data without a "no data"
    convention), while structure/radial-profile maps want percentile
    clipping - each caller decides explicitly.
    `scale_bar`, if given, is a dict of kwargs forwarded to `draw_scale_bar`
    for the *first* panel only (real-space grids only - the "top one" is
    enough, and it's meaningless on FFT-domain grids).
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
        if index == 0 and scale_bar is not None:
            draw_scale_bar(ax, **scale_bar)

    fig.tight_layout()


def make_grid_figure(
    images: list[np.ndarray],
    titles: list[str],
    suptitle: str | None = None,
    cmap: str = "viridis",
    ncols: int | None = None,
    vlimits: list[tuple[float, float] | None] | None = None,
    scale_bar: dict | None = None,
) -> Figure:
    """Build a new Figure with one imshow + colorbar per image, arranged in a grid."""
    n = len(images)
    grid_ncols = ncols if ncols is not None else _grid_shape(n)[1]
    grid_nrows = math.ceil(n / grid_ncols)
    fig = Figure(figsize=(3.2 * grid_ncols, 3.0 * grid_nrows))
    draw_grid(fig, images, titles, suptitle=suptitle, cmap=cmap, ncols=ncols, vlimits=vlimits, scale_bar=scale_bar)
    return fig


def draw_structure_maps(
    fig: Figure,
    spacing_map: np.ndarray,
    angle_map: np.ndarray,
    intensity_map: np.ndarray,
    spacing_label: str,
    map_pixel_size: float | None = None,
    map_pixel_size_units: str = "nm",
) -> None:
    """(Re)draw the standard spacing/angle/intensity map stack: vertical,
    each panel percentile-clipped, scale bar on the top (spacing) panel when
    `map_pixel_size` is given. Shared by `core.structuremaps`- and
    `core.carbontools`-based batch results, which only differ in how the
    maps were computed, not how they're displayed.
    """
    scale_bar = {"pixel_size": map_pixel_size, "units": map_pixel_size_units} if map_pixel_size else None
    draw_grid(
        fig,
        [spacing_map, angle_map, intensity_map],
        [spacing_label, "Angle (deg)", "Intensity"],
        ncols=1,
        vlimits=[percentile_clip(spacing_map), percentile_clip(angle_map), percentile_clip(intensity_map)],
        scale_bar=scale_bar,
    )


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
