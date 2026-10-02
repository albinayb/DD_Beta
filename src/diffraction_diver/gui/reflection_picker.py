"""Click-a-peak-to-calibrate helper for windows that show a diffractogram but
don't have `ReflectionFitWidget`'s full fit UI (`SlidingFFTWindow`,
`RadialProfileWindow`).

Clicking a peak fits it and its Friedel twin (same `fit_reflection_pair` the
Peak Fit module uses) and feeds the measured distance to the window's
`CalibrationWidget`, enabling "Set calibration from selected reflection".
"""

from collections.abc import Callable

import numpy as np
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from PySide6.QtCore import QObject
from PySide6.QtWidgets import QLabel

from diffraction_diver.core import structuremaps as core_structuremaps
from diffraction_diver.gui.calibration_widget import CalibrationWidget
from diffraction_diver.gui.patch_selector import PatchSelectorWidget
from diffraction_diver.gui.reflection_fit_widget import MIRROR_COLOR, PRIMARY_COLOR

DEFAULT_FIT_RADIUS = 5
DEFAULT_CENTER_MASK_RADIUS = 3
DEFAULT_DETECTION_THRESHOLD = 3.0

HINT_TEXT = "Click a peak in the FFT to pick a reflection for calibration."


class ReflectionPicker(QObject):
    """Wires a canvas showing the current patch's spectrum to a `CalibrationWidget`.

    `axes_index` selects which axes of the canvas's figure is the spectrum
    (0 for a lone diffractogram, 1 for the patch preview's FFT panel).
    `on_selections_changed` is called with the overlay list (or `None` when
    cleared) so the host can redraw its own preview with the markers.
    """

    def __init__(
        self,
        canvas: FigureCanvas,
        axes_index: int,
        patch_selector: PatchSelectorWidget,
        calibration: CalibrationWidget,
        on_selections_changed: Callable[[list[tuple[int, int, int, str]] | None], None],
        parent: QObject | None = None,
    ) -> None:
        super().__init__(parent)
        self._canvas = canvas
        self._axes_index = axes_index
        self._patch_selector = patch_selector
        self._calibration = calibration
        self._on_selections_changed = on_selections_changed
        self._distance_px: float | None = None
        self._has_selection = False

        self.status_label = QLabel(HINT_TEXT)
        self.status_label.setWordWrap(True)

        canvas.mpl_connect("button_press_event", self._on_click)
        calibration.changed.connect(self._update_label)

    def clear(self) -> None:
        """Forget the picked reflection - call whenever the shown spectrum changes."""
        self._distance_px = None
        self._calibration.set_reflection_reference(None, None)
        self.status_label.setText(HINT_TEXT)
        if self._has_selection:
            self._has_selection = False
            self._on_selections_changed(None)

    def _on_click(self, event) -> None:
        axes = self._canvas.figure.axes
        if self._axes_index >= len(axes) or event.inaxes is not axes[self._axes_index]:
            return
        if event.xdata is None or event.ydata is None:
            return
        spectrum = self._patch_selector.current_spectrum()
        if spectrum is None:
            return

        magnitude = np.abs(spectrum)
        size = magnitude.shape[0]
        fit_rad = min(DEFAULT_FIT_RADIUS, max(2, self._patch_selector.crop_size // 2))
        row, col = int(round(event.ydata)), int(round(event.xdata))

        result = core_structuremaps.fit_reflection_pair(
            magnitude,
            row,
            col,
            fit_rad,
            min_peak_ratio=DEFAULT_DETECTION_THRESHOLD,
            highpass_radius=DEFAULT_CENTER_MASK_RADIUS,
        )
        if result is None:
            self.status_label.setText(
                "No reflection detected there (neither it nor its mirror stands out). "
                "Click closer to the center of a peak."
            )
            return

        self._distance_px = result["distance_px"]
        self._calibration.set_reflection_reference(self._distance_px, self._patch_selector.window_size)

        row1, col1 = (round(v) for v in result["center1"])
        row2, col2 = (round(v) for v in result["center2"])
        self._has_selection = True
        self._on_selections_changed([(row1, col1, fit_rad, PRIMARY_COLOR), (row2, col2, fit_rad, MIRROR_COLOR)])
        self._update_label()

    def _update_label(self) -> None:
        if self._distance_px is None:
            return
        spacing = self._calibration.format_spacing(self._distance_px, self._patch_selector.window_size)
        self.status_label.setText(
            f"Selected reflection: {self._distance_px:.2f} reciprocal px ({spacing}). "
            "Enter its known spacing under Calibration, then set calibration from it."
        )
