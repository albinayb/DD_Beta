"""Small Qt/matplotlib glue shared across module windows."""

from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from PySide6.QtWidgets import QScrollArea, QWidget


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


def harden_scroll_repaint(scroll: QScrollArea, content: QWidget) -> None:
    """Guard `scroll` against a stale-pixel repaint artifact beside a
    narrower/shorter-than-viewport canvas.

    `content.setAutoFillBackground(True)` alone doesn't reliably clear that
    margin on every scroll tick, and a scheduled `update()` can still get
    coalesced away without ever actually repainting. Forcing an immediate,
    synchronous `repaint()` of both the viewport and the content widget on
    every scrollbar move is the version of this fix that actually holds up.
    """
    content.setAutoFillBackground(True)

    def _force_repaint(_value: int) -> None:
        content.repaint()
        scroll.viewport().repaint()

    scroll.verticalScrollBar().valueChanged.connect(_force_repaint)
    scroll.horizontalScrollBar().valueChanged.connect(_force_repaint)


def embed_figure(fig: Figure, fixed_width: bool = False) -> FigureCanvas:
    """Wrap a newly-built `fig` in a canvas sized to its actual pixel dimensions."""
    canvas = FigureCanvas(fig)
    size_canvas_to_figure(canvas, fixed_width=fixed_width)
    return canvas
