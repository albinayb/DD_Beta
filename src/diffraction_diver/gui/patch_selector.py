"""Reusable 'load an image, pick a square FFT patch within it' widget.

Shared by every module that needs this same load-and-patch-selection UI
(`SlidingFFTWindow`, `PeakFitWindow`). Each embeds its own independent
instance - no state is shared between modules.
"""

from pathlib import Path

import numpy as np
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from PySide6.QtCore import Signal
from PySide6.QtWidgets import (
    QCheckBox,
    QFileDialog,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QPushButton,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from diffraction_diver.core import fft as core_fft
from diffraction_diver.core import io as core_io
from diffraction_diver.gui import plotting


class PatchSelectorWidget(QWidget):
    """Load a raw image and pick a square FFT patch within it.

    Emits `changed` whenever the loaded image, patch position, or any FFT
    parameter (window size, crop size, hamming) changes, so callers can
    recompute derived state (e.g. re-fit a peak) off this signal instead of
    polling. Emits `image_loaded` specifically when a *new* image is loaded
    (a strict subset of `changed`), for state that's tied to the image
    itself rather than the current view of it - e.g. a calibration value
    should survive a window/crop/position tweak but not a new image load.
    """

    changed = Signal()
    image_loaded = Signal()

    def __init__(
        self,
        show_patch_preview: bool = True,
        parent: QWidget | None = None,
        load_button_label: str = "Load Image / .npy…",
    ) -> None:
        super().__init__(parent)
        self._show_patch_preview = show_patch_preview
        self._load_button_label = load_button_label
        self.raw_image: np.ndarray | None = None
        self.patch_canvas: FigureCanvas | None = None
        self._pixel_size: float | None = None
        self._spectrum_selections: list[tuple[int, int, int, str]] | None = None
        self._build_ui()

    # ------------------------------------------------------------------ UI

    def _build_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self._build_input_group())
        layout.addWidget(self._build_patch_group())

    def _build_input_group(self) -> QGroupBox:
        group = QGroupBox("Input")
        v = QVBoxLayout(group)

        load_btn = QPushButton(self._load_button_label)
        load_btn.clicked.connect(self.on_load_clicked)
        v.addWidget(load_btn)

        self.file_label = QLabel("No file loaded")
        self.file_label.setWordWrap(True)
        v.addWidget(self.file_label)

        self.preview_canvas = FigureCanvas(Figure(figsize=(4, 4)))
        self.preview_canvas.setMinimumHeight(320)
        self.preview_canvas.mpl_connect("button_press_event", self.on_preview_click)
        v.addWidget(self.preview_canvas)

        return group

    def _build_patch_group(self) -> QGroupBox:
        group = QGroupBox("Patch parameters")
        v = QVBoxLayout(group)

        # FFT params and patch position side by side (instead of stacked)
        # so this box takes roughly half the vertical space.
        row = QHBoxLayout()

        form = QFormLayout()
        self.window_size_spin = QSpinBox()
        self.window_size_spin.setRange(4, 4096)
        self.window_size_spin.setValue(128)

        self.crop_size_spin = QSpinBox()
        self.crop_size_spin.setRange(2, 4096)
        self.crop_size_spin.setValue(64)

        self.hamming_check = QCheckBox("Apply Hamming window")
        self.hamming_check.setChecked(True)

        form.addRow("Window size", self.window_size_spin)
        form.addRow("Crop size (zoom)", self.crop_size_spin)
        form.addRow(self.hamming_check)
        row.addLayout(form)

        pos_form = QFormLayout()
        self.patch_x_spin = QSpinBox()
        self.patch_y_spin = QSpinBox()
        pos_form.addRow("Patch x", self.patch_x_spin)
        pos_form.addRow("Patch y", self.patch_y_spin)
        row.addLayout(pos_form)
        v.addLayout(row)
        v.addWidget(QLabel("Tip: click a point on the image above to move the patch."))

        if self._show_patch_preview:
            self.patch_canvas = FigureCanvas(Figure(figsize=(5, 2.5)))
            # Tall enough to click a peak in the FFT panel for calibration.
            self.patch_canvas.setMinimumHeight(260)
            v.addWidget(self.patch_canvas)

        for widget, signal_name in [
            (self.window_size_spin, "valueChanged"),
            (self.crop_size_spin, "valueChanged"),
            (self.patch_x_spin, "valueChanged"),
            (self.patch_y_spin, "valueChanged"),
        ]:
            getattr(widget, signal_name).connect(self._on_params_changed)
        self.hamming_check.stateChanged.connect(self._on_params_changed)

        return group

    # --------------------------------------------------------------- loading

    def on_load_clicked(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Load image or array",
            "",
            "Images/Arrays (*.npy *.png *.tif *.tiff *.jpg *.jpeg);;All files (*)",
        )
        if not path:
            return

        try:
            image = core_io.load_input(path)
        except Exception as exc:
            QMessageBox.critical(self, "Failed to load file", str(exc))
            return

        self.raw_image = image
        self.file_label.setText(f"{Path(path).name}  ({image.shape[0]}×{image.shape[1]})")

        self.patch_x_spin.blockSignals(True)
        self.patch_y_spin.blockSignals(True)
        self.patch_x_spin.setValue(0)
        self.patch_y_spin.setValue(0)
        self.patch_x_spin.blockSignals(False)
        self.patch_y_spin.blockSignals(False)

        self._update_spin_ranges()
        self._refresh_preview()
        self._refresh_patch_preview()
        # image_loaded first: hosts reset image-tied state (calibration,
        # computed stacks) before the generic `changed` refresh runs.
        self.image_loaded.emit()
        self.changed.emit()

    def _update_spin_ranges(self) -> None:
        if self.raw_image is None:
            return
        height, width = self.raw_image.shape
        max_window = max(4, min(height, width))

        self.window_size_spin.setRange(4, max_window)
        window_size = self.window_size_spin.value()

        self.crop_size_spin.setRange(2, window_size)
        if self.crop_size_spin.value() > window_size:
            self.crop_size_spin.setValue(window_size)

        max_x, max_y = core_fft.max_window_position(window_size, height, width)
        self.patch_x_spin.setRange(0, max_x)
        self.patch_y_spin.setRange(0, max_y)

    # --------------------------------------------------------------- preview

    def on_preview_click(self, event) -> None:
        if self.raw_image is None or event.xdata is None or event.ydata is None:
            return
        ypos = int(round(event.xdata))
        xpos = int(round(event.ydata))
        self.patch_x_spin.setValue(max(self.patch_x_spin.minimum(), min(xpos, self.patch_x_spin.maximum())))
        self.patch_y_spin.setValue(max(self.patch_y_spin.minimum(), min(ypos, self.patch_y_spin.maximum())))

    def _on_params_changed(self) -> None:
        if self.raw_image is None:
            return
        self._update_spin_ranges()
        self._refresh_preview()
        self._refresh_patch_preview()
        self.changed.emit()

    def _refresh_preview(self) -> None:
        if self.raw_image is None:
            return
        rect = (self.patch_x_spin.value(), self.patch_y_spin.value(), self.window_size_spin.value())
        plotting.draw_raw_image(self.preview_canvas.figure, self.raw_image, rect=rect, pixel_size=self._pixel_size)
        self.preview_canvas.draw_idle()

    def set_pixel_size(self, pixel_size: float | None) -> None:
        """Set the raw image's calibration (units per pixel) for the preview's scale bar.

        Decoupled from `CalibrationWidget` on purpose - the host window owns
        both and wires `calibration.changed` to call this with
        `calibration.pixel_size_nm`, so this widget stays reusable without
        depending on calibration's UI.
        """
        self._pixel_size = pixel_size
        self._refresh_preview()

    def _refresh_patch_preview(self) -> None:
        if self.raw_image is None or self.patch_canvas is None:
            return
        plotting.draw_patch_preview(
            self.patch_canvas.figure,
            self.current_patch(),
            self.current_spectrum(),
            selections=self._spectrum_selections,
        )
        self.patch_canvas.draw_idle()

    def set_spectrum_selections(self, selections: list[tuple[int, int, int, str]] | None) -> None:
        """Overlay peak markers on the patch preview's FFT panel (see `draw_diffractogram`)."""
        self._spectrum_selections = selections
        self._refresh_patch_preview()

    # ---------------------------------------------------------------- state

    @property
    def window_size(self) -> int:
        return self.window_size_spin.value()

    @property
    def crop_size(self) -> int:
        return min(self.crop_size_spin.value(), self.window_size_spin.value())

    @property
    def hamming(self) -> bool:
        return self.hamming_check.isChecked()

    @property
    def patch_position(self) -> tuple[int, int]:
        return self.patch_x_spin.value(), self.patch_y_spin.value()

    def current_patch(self) -> np.ndarray | None:
        if self.raw_image is None:
            return None
        xpos, ypos = self.patch_position
        window_size = self.window_size
        return self.raw_image[xpos : xpos + window_size, ypos : ypos + window_size]

    def current_spectrum(self) -> np.ndarray | None:
        if self.raw_image is None:
            return None
        xpos, ypos = self.patch_position
        return core_fft.window_fft(self.raw_image, xpos, ypos, self.window_size, self.hamming, self.crop_size)
