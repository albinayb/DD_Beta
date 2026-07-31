"""Radial Profile Analysis window: compute the full azimuthally-averaged
radial FFT profile for every patch across the whole image (a 3D dataset),
then interactively scan through it - integrating a width-window of radius
bins into a live 2D intensity map - without recomputing any FFTs."""

from pathlib import Path

import numpy as np
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from PySide6.QtCore import Qt, QThread
from PySide6.QtWidgets import (
    QCheckBox,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QInputDialog,
    QLabel,
    QMainWindow,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QScrollArea,
    QSlider,
    QSpinBox,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from diffraction_diver.core import radialprofile as core_radialprofile
from diffraction_diver.gui import plotting
from diffraction_diver.gui.calibration_widget import CalibrationWidget
from diffraction_diver.gui.canvas_utils import size_canvas_to_figure
from diffraction_diver.gui.patch_selector import PatchSelectorWidget
from diffraction_diver.gui.workers import RadialProfileWorker


class RadialProfileWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("Radial Profile Analysis")
        self.resize(1500, 850)

        self._profile_stack: np.ndarray | None = None
        self._mean_profile: np.ndarray | None = None
        self._stack_window_size: int | None = None
        self._last_map: np.ndarray | None = None

        self._profile_thread: QThread | None = None
        self._profile_worker: RadialProfileWorker | None = None

        self._build_ui()

    # ------------------------------------------------------------------ UI

    def _build_ui(self) -> None:
        central = QWidget()
        self.setCentralWidget(central)
        root = QHBoxLayout(central)

        self.patch_selector = PatchSelectorWidget(show_patch_preview=False)
        self.patch_selector.changed.connect(self._on_patch_changed)
        self.patch_selector.image_loaded.connect(self._on_image_loaded)
        self.patch_selector.setFixedWidth(440)
        root.addWidget(self.patch_selector)

        root.addWidget(self._build_center_tabs(), stretch=1)

        right_column = QWidget()
        right_layout = QVBoxLayout(right_column)
        right_layout.addWidget(self._build_batch_group())
        right_layout.addWidget(self._build_scan_group())
        self.calibration = CalibrationWidget()
        self.calibration.changed.connect(self._on_calibration_changed)
        right_layout.addWidget(self.calibration)
        right_layout.addStretch(1)

        right_scroll = QScrollArea()
        right_scroll.setWidgetResizable(True)
        right_scroll.setWidget(right_column)
        right_scroll.setFixedWidth(400)
        root.addWidget(right_scroll)

    def _build_center_tabs(self) -> QTabWidget:
        tabs = QTabWidget()

        preview_tab = QWidget()
        preview_layout = QVBoxLayout(preview_tab)
        preview_layout.addWidget(QLabel("Diffractogram and radial profile for the current patch."))

        self.diffractogram_canvas = FigureCanvas(Figure(figsize=(5, 5)))
        self.diffractogram_canvas.setMinimumSize(400, 400)
        preview_layout.addWidget(self.diffractogram_canvas)

        self.profile_canvas = FigureCanvas(Figure(figsize=(5, 3)))
        self.profile_canvas.setMinimumHeight(260)
        preview_layout.addWidget(self.profile_canvas)
        tabs.addTab(preview_tab, "Diffractogram + Profile")

        map_tab = QWidget()
        map_tab_layout = QVBoxLayout(map_tab)

        map_scroll = QScrollArea()
        map_scroll.setWidgetResizable(True)
        map_content = QWidget()
        # Without this, Qt can leave stale pixels un-repainted in the empty
        # margin beside these (narrower-than-the-viewport) canvases when
        # scrolling - a partial-repaint artifact, not a data/plotting bug.
        map_content.setAutoFillBackground(True)
        map_layout = QVBoxLayout(map_content)
        # Both figures keep a fixed size for the window's whole life (see
        # draw_labeled_image/draw_radial_profile) - neither is ever resized
        # by draw_grid's N-image grid sizing, which was causing the canvas
        # to shrink/regrow on every scan update and left stale paint behind.
        self.map_canvas = FigureCanvas(Figure(figsize=(6, 6)))
        size_canvas_to_figure(self.map_canvas)
        map_layout.addWidget(self.map_canvas)
        # The mean profile (across every patch) persists here too, with the
        # current scan band highlighted, so the "which spacing am I looking
        # at" context stays visible after leaving the live single-patch tab.
        self.map_profile_canvas = FigureCanvas(Figure(figsize=(6, 3)))
        size_canvas_to_figure(self.map_profile_canvas)
        map_layout.addWidget(self.map_profile_canvas)
        map_layout.addStretch(1)
        map_scroll.setWidget(map_content)
        map_tab_layout.addWidget(map_scroll)

        self.map_save_btn = QPushButton("Save current map…")
        self.map_save_btn.setEnabled(False)
        self.map_save_btn.clicked.connect(self.on_save_map_clicked)
        map_tab_layout.addWidget(self.map_save_btn)

        self.stack_save_btn = QPushButton("Save full radial profile stack…")
        self.stack_save_btn.setEnabled(False)
        self.stack_save_btn.clicked.connect(self.on_save_stack_clicked)
        map_tab_layout.addWidget(self.stack_save_btn)

        tabs.addTab(map_tab, "Map")

        self._center_tabs = tabs
        self._map_tab_index = 1
        return tabs

    def _build_batch_group(self) -> QGroupBox:
        group = QGroupBox("Radial Profile Batch")
        v = QVBoxLayout(group)
        v.addWidget(QLabel("Computes the radial profile of every patch in the image."))

        form = QFormLayout()
        self.batch_window_step_spin = QSpinBox()
        self.batch_window_step_spin.setRange(1, 4096)
        self.batch_window_step_spin.setValue(4)
        form.addRow("Window step", self.batch_window_step_spin)
        v.addLayout(form)

        self.compute_btn = QPushButton("Compute Radial Profiles")
        self.compute_btn.setEnabled(False)
        self.compute_btn.clicked.connect(self.on_compute_clicked)
        v.addWidget(self.compute_btn)

        self.compute_progress = QProgressBar()
        v.addWidget(self.compute_progress)

        return group

    def _build_scan_group(self) -> QGroupBox:
        group = QGroupBox("Spacing Window")
        group.setEnabled(False)
        v = QVBoxLayout(group)

        form = QFormLayout()
        self.profile_width_spin = QSpinBox()
        self.profile_width_spin.setRange(1, 32)
        self.profile_width_spin.setValue(3)
        self.profile_width_spin.valueChanged.connect(self._on_scan_changed)
        form.addRow("Profile width (px)", self.profile_width_spin)
        v.addLayout(form)

        self.subtract_background_check = QCheckBox("Subtract background (outer radius average)")
        self.subtract_background_check.stateChanged.connect(self._on_scan_changed)
        v.addWidget(self.subtract_background_check)

        bg_form = QFormLayout()
        self.background_fraction_spin = QDoubleSpinBox()
        self.background_fraction_spin.setRange(5.0, 50.0)
        self.background_fraction_spin.setValue(20.0)
        self.background_fraction_spin.setSuffix(" %")
        self.background_fraction_spin.valueChanged.connect(self._on_scan_changed)
        bg_form.addRow("Outer radius used", self.background_fraction_spin)
        v.addLayout(bg_form)

        self.radius_slider = QSlider(Qt.Horizontal)
        self.radius_slider.valueChanged.connect(self._on_scan_changed)
        v.addWidget(self.radius_slider)

        self.radius_label = QLabel("-")
        self.radius_label.setWordWrap(True)
        v.addWidget(self.radius_label)

        self._scan_group = group
        return group

    # --------------------------------------------------------- patch state

    def _on_image_loaded(self) -> None:
        # Both calibration and any already-computed stack are tied to the
        # image that was loaded when they were made - a new image
        # invalidates both.
        self.calibration.reset()
        self._profile_stack = None
        self._mean_profile = None
        self._scan_group.setEnabled(False)
        self.map_save_btn.setEnabled(False)
        self.stack_save_btn.setEnabled(False)
        self.map_canvas.figure.clear()
        self.map_canvas.draw_idle()
        self.map_profile_canvas.figure.clear()
        self.map_profile_canvas.draw_idle()

    def _on_patch_changed(self) -> None:
        raw_image = self.patch_selector.raw_image
        self.compute_btn.setEnabled(raw_image is not None)
        self.calibration.set_image_reference(raw_image.shape[0] if raw_image is not None else None)
        self._refresh_preview()

    def _maybe_subtract_background(self, profile: np.ndarray) -> np.ndarray:
        if not self.subtract_background_check.isChecked():
            return profile
        return core_radialprofile.subtract_background(profile, self.background_fraction_spin.value() / 100.0)

    def _refresh_preview(self) -> None:
        spectrum = self.patch_selector.current_spectrum()
        if spectrum is None:
            return
        plotting.draw_diffractogram(self.diffractogram_canvas.figure, spectrum)
        self.diffractogram_canvas.draw_idle()

        profile = self._maybe_subtract_background(core_radialprofile.radial_profile(np.abs(spectrum)))
        band = self._current_band(len(profile)) if self._scan_group.isEnabled() else None
        plotting.draw_radial_profile(self.profile_canvas.figure, profile, band=band, title="Local Radial Profile")
        self.profile_canvas.draw_idle()

    def _on_calibration_changed(self) -> None:
        self.patch_selector.set_pixel_size(self.calibration.pixel_size_nm)
        self._update_scan_display()

    # ------------------------------------------------------------- compute

    def on_compute_clicked(self) -> None:
        raw_image = self.patch_selector.raw_image
        if raw_image is None:
            return

        self.compute_btn.setEnabled(False)
        self.compute_progress.setRange(0, 1)
        self.compute_progress.setValue(0)

        self._profile_thread = QThread()
        self._profile_worker = RadialProfileWorker(
            raw_image,
            self.patch_selector.window_size,
            self.batch_window_step_spin.value(),
            self.patch_selector.hamming,
            self.patch_selector.crop_size,
        )
        self._profile_worker.moveToThread(self._profile_thread)
        self._profile_thread.started.connect(self._profile_worker.run)
        self._profile_worker.progress.connect(self._on_compute_progress)
        self._profile_worker.finished.connect(self._on_compute_finished)
        self._profile_worker.error.connect(self._on_compute_error)
        self._profile_worker.finished.connect(self._profile_thread.quit)
        self._profile_worker.error.connect(self._profile_thread.quit)
        self._profile_thread.finished.connect(self._profile_thread.deleteLater)
        self._profile_thread.start()

    def _on_compute_progress(self, done: int, total: int) -> None:
        self.compute_progress.setRange(0, total)
        self.compute_progress.setValue(done)

    def _on_compute_finished(self, profile_stack: np.ndarray, size_x: int, size_y: int) -> None:
        self.compute_btn.setEnabled(True)
        self._profile_stack = profile_stack
        self._mean_profile = profile_stack.mean(axis=(0, 1))
        # The stack's radius-to-spacing conversion depends on the window
        # size it was computed with, not whatever the patch selector's
        # window size happens to be later.
        self._stack_window_size = self.patch_selector.window_size
        self.stack_save_btn.setEnabled(True)

        n_bins = profile_stack.shape[2]
        self.radius_slider.blockSignals(True)
        self.radius_slider.setRange(0, n_bins - 1)
        self.radius_slider.setValue(min(n_bins - 1, max(1, n_bins // 4)))
        self.radius_slider.blockSignals(False)
        self.profile_width_spin.setRange(1, n_bins)

        self._scan_group.setEnabled(True)
        self._update_scan_display()
        self._refresh_preview()
        self._center_tabs.setCurrentIndex(self._map_tab_index)

    def _on_compute_error(self, message: str) -> None:
        self.compute_btn.setEnabled(True)
        QMessageBox.critical(self, "Compute radial profiles failed", message)

    # ---------------------------------------------------------------- scan

    def _current_band(self, n_bins: int) -> tuple[int, int]:
        center = self.radius_slider.value()
        width = self.profile_width_spin.value()
        half = max(1, width) // 2
        return max(0, center - half), min(n_bins, center + half + 1)

    def _on_scan_changed(self) -> None:
        self._update_scan_display()
        self._refresh_preview()

    def _radius_readout(self, radius_bin: int) -> str:
        window_size = self._stack_window_size or self.patch_selector.window_size
        if radius_bin <= 0:
            return "radius 0 (DC, no periodicity)"
        return f"radius {radius_bin} px → {self.calibration.format_spacing(radius_bin, window_size)}"

    def _update_scan_display(self) -> None:
        center = self.radius_slider.value()
        width = self.profile_width_spin.value()
        self.radius_label.setText(
            f"Scanning {self._radius_readout(center)}  (window ±{width // 2} px)"
        )
        if self._profile_stack is None:
            return

        mean_profile = self._maybe_subtract_background(self._mean_profile)
        band = self._current_band(len(mean_profile))
        plotting.draw_radial_profile(
            self.map_profile_canvas.figure, mean_profile, band=band, title="Average Radial Profile"
        )
        self.map_profile_canvas.draw_idle()

        stack = self._maybe_subtract_background(self._profile_stack)
        intensity_map = core_radialprofile.integrate_window(stack, center, width)
        self._last_map = intensity_map
        scale_bar = None
        if self.calibration.pixel_size_nm is not None:
            map_pixel_size = self.batch_window_step_spin.value() * self.calibration.pixel_size_nm
            scale_bar = {"pixel_size": map_pixel_size, "units": "nm"}
        plotting.draw_labeled_image(
            self.map_canvas.figure,
            intensity_map,
            self._radius_readout(center),
            vlimits=plotting.percentile_clip(intensity_map),
            scale_bar=scale_bar,
        )
        self.map_canvas.draw_idle()
        self.map_save_btn.setEnabled(True)

    # ----------------------------------------------------------------- save

    def on_save_map_clicked(self) -> None:
        if self._last_map is None:
            return
        directory = QFileDialog.getExistingDirectory(self, "Choose output folder")
        if not directory:
            return
        prefix, ok = QInputDialog.getText(self, "Filename prefix", "Prefix for saved file:", text="radial_intensity")
        if not ok:
            return
        prefix = prefix.strip() or "radial_intensity"
        out = Path(directory) / f"{prefix}.npy"
        np.save(out, self._last_map)
        QMessageBox.information(self, "Saved", f"Saved current map to {out}")

    def on_save_stack_clicked(self) -> None:
        if self._profile_stack is None:
            return
        directory = QFileDialog.getExistingDirectory(self, "Choose output folder")
        if not directory:
            return
        prefix, ok = QInputDialog.getText(
            self, "Filename prefix", "Prefix for saved file:", text="radial_profile_stack"
        )
        if not ok:
            return
        prefix = prefix.strip() or "radial_profile_stack"
        out = Path(directory) / f"{prefix}.npy"
        np.save(out, self._profile_stack)
        QMessageBox.information(self, "Saved", f"Saved full radial profile stack to {out}")
