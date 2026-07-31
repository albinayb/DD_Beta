"""Diffractogram Peak Fit window: inspect one diffraction pattern at a time,
pick a Bragg peak by eye, auto-identify and fit its centrosymmetric twin,
calibrate pixel size, and (optionally) batch that same reflection across the
whole image into spacing/angle/intensity maps."""

from PySide6.QtWidgets import QHBoxLayout, QMainWindow, QWidget

from diffraction_diver.gui.calibration_widget import CalibrationWidget
from diffraction_diver.gui.patch_selector import PatchSelectorWidget
from diffraction_diver.gui.reflection_fit_widget import ReflectionFitWidget


class PeakFitWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("Diffractogram Peak Fit")
        self.resize(1500, 850)

        central = QWidget()
        self.setCentralWidget(central)
        root = QHBoxLayout(central)

        self.patch_selector = PatchSelectorWidget(show_patch_preview=False)
        self.patch_selector.setFixedWidth(440)
        root.addWidget(self.patch_selector)

        self.calibration = CalibrationWidget()
        self.calibration.changed.connect(self._on_calibration_changed)
        self.reflection_fit = ReflectionFitWidget(self.patch_selector, self.calibration)
        root.addWidget(self.reflection_fit, stretch=1)

    def _on_calibration_changed(self) -> None:
        self.patch_selector.set_pixel_size(self.calibration.pixel_size_nm)
