"""Reusable pixel-size calibration widget.

Shared by every module that needs to convert a reciprocal-pixel distance
into a real-space spacing - either from the image's field of view, or from a
known spacing of a reflection already measured elsewhere in that module.
"""

from PySide6.QtCore import Signal
from PySide6.QtWidgets import (
    QDoubleSpinBox,
    QFormLayout,
    QGroupBox,
    QLabel,
    QMessageBox,
    QPushButton,
    QVBoxLayout,
)

from diffraction_diver.core import structuremaps as core_structuremaps

NM_PER_ANGSTROM = 0.1


class CalibrationWidget(QGroupBox):
    """Holds a `pixel_size_nm` calibration value and the UI to set it.

    The host window feeds it context it doesn't own itself:
    - `set_image_reference(image_height_px)` whenever the loaded image
      changes (enables "from field of view").
    - `set_reflection_reference(distance_px, window_size)` whenever the
      host's current best measured reflection changes, e.g. after a peak fit
      (enables "from known reflection spacing"); `None` to disable it again.

    Emits `changed` whenever the stored calibration changes (including being
    cleared via `reset()`), so callers can refresh any calibration-dependent
    display.
    """

    changed = Signal()

    def __init__(self, parent=None) -> None:
        super().__init__("Calibration", parent)
        self.pixel_size_nm: float | None = None
        self._image_height_px: int | None = None
        self._distance_px: float | None = None
        self._window_size: int | None = None
        self._build_ui()

    def _build_ui(self) -> None:
        v = QVBoxLayout(self)

        self.status_label = QLabel("Uncalibrated (spacing shown in reciprocal pixels).")
        self.status_label.setWordWrap(True)
        v.addWidget(self.status_label)

        fov_form = QFormLayout()
        self.fov_spin = QDoubleSpinBox()
        self.fov_spin.setRange(0.001, 1e7)
        self.fov_spin.setDecimals(4)
        self.fov_spin.setValue(100.0)
        self.fov_spin.setSuffix(" nm")
        fov_form.addRow("Image field of view", self.fov_spin)
        v.addLayout(fov_form)
        self.fov_btn = QPushButton("Set calibration from field of view")
        self.fov_btn.setEnabled(False)
        self.fov_btn.clicked.connect(self.on_set_from_fov)
        v.addWidget(self.fov_btn)

        known_form = QFormLayout()
        self.known_dspacing_spin = QDoubleSpinBox()
        self.known_dspacing_spin.setRange(0.001, 1e5)
        self.known_dspacing_spin.setDecimals(4)
        self.known_dspacing_spin.setValue(2.0)
        self.known_dspacing_spin.setSuffix(" Å")
        known_form.addRow("Known spacing of selected reflection", self.known_dspacing_spin)
        v.addLayout(known_form)
        self.known_btn = QPushButton("Set calibration from selected reflection")
        self.known_btn.setEnabled(False)
        self.known_btn.clicked.connect(self.on_set_from_reflection)
        v.addWidget(self.known_btn)

    # -------------------------------------------------------------- context

    def set_image_reference(self, image_height_px: int | None) -> None:
        self._image_height_px = image_height_px
        self.fov_btn.setEnabled(image_height_px is not None)

    def set_reflection_reference(self, distance_px: float | None, window_size: int | None) -> None:
        self._distance_px = distance_px
        self._window_size = window_size
        self.known_btn.setEnabled(distance_px is not None)

    def reset(self) -> None:
        """Clear the stored calibration - call when a new image loads."""
        self.pixel_size_nm = None
        self._update_label()
        self.changed.emit()

    # ----------------------------------------------------------------- set

    def on_set_from_fov(self) -> None:
        if self._image_height_px is None:
            QMessageBox.warning(self, "No image loaded", "Load an image first.")
            return
        self.pixel_size_nm = self.fov_spin.value() / self._image_height_px
        self._update_label()
        self.changed.emit()

    def on_set_from_reflection(self) -> None:
        if self._distance_px is None or self._window_size is None:
            return
        known_dspacing_nm = self.known_dspacing_spin.value() * NM_PER_ANGSTROM
        self.pixel_size_nm = core_structuremaps.dspacing_to_pixel_size(
            self._distance_px, self._window_size, known_dspacing_nm
        )
        self._update_label()
        self.changed.emit()

    def _update_label(self) -> None:
        if self.pixel_size_nm is None:
            self.status_label.setText("Uncalibrated (spacing shown in reciprocal pixels).")
        else:
            self.status_label.setText(f"Calibrated: {self.pixel_size_nm:.5g} nm/pixel.")

    # ------------------------------------------------------------- display

    def format_spacing(self, distance_px: float, window_size: int) -> str:
        """A reciprocal-pixel distance -> display string (Å if calibrated, else raw px)."""
        if self.pixel_size_nm is None:
            return f"{distance_px:.3f} px (reciprocal, uncalibrated)"
        dspacing_nm = core_structuremaps.pixel_distance_to_dspacing(distance_px, window_size, self.pixel_size_nm)
        return f"{dspacing_nm / NM_PER_ANGSTROM:.4g} Å"
