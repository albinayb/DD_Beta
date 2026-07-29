"""Small Qt/matplotlib glue shared across module windows."""

from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure


def size_canvas_to_figure(canvas: FigureCanvas, fixed_width: bool = False) -> None:
    """Set `canvas`'s minimum size to its figure's actual pixel dimensions.

    Without this, a scroll area with `setWidgetResizable(True)` will happily
    squash a canvas down to fit the viewport instead of scrolling.
    `fixed_width` additionally caps the canvas at its natural width, so a
    narrow plot doesn't get stretched to match a wider figure stacked
    above/beside it.
    """
    fig = canvas.figure
    dpi = fig.get_dpi()
    width = int(fig.get_figwidth() * dpi)
    height = int(fig.get_figheight() * dpi)
    canvas.setMinimumSize(width, height)
    if fixed_width:
        canvas.setMaximumWidth(width)


def embed_figure(fig: Figure, fixed_width: bool = False) -> FigureCanvas:
    """Wrap a newly-built `fig` in a canvas sized to its actual pixel dimensions."""
    canvas = FigureCanvas(fig)
    size_canvas_to_figure(canvas, fixed_width=fixed_width)
    return canvas
