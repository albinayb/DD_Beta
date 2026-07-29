"""Diffractogram Peak Fit window: inspect one diffraction pattern at a time,
pick a Bragg peak by eye, auto-identify and fit its centrosymmetric twin,
calibrate pixel size, and (optionally) batch that same reflection across the
whole image into spacing/angle/intensity maps."""

from pathlib import Path

import numpy as np
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from PySide6.QtCore import QThread
from PySide6.QtWidgets import (
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
    QSpinBox,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from diffraction_diver.core import peakfit
from diffraction_diver.core import structuremaps as core_structuremaps
from diffraction_diver.gui import plotting
from diffraction_diver.gui.calibration_widget import NM_PER_ANGSTROM, CalibrationWidget
from diffraction_diver.gui.canvas_utils import size_canvas_to_figure
from diffraction_diver.gui.patch_selector import PatchSelectorWidget
from diffraction_diver.gui.workers import StructureMapWorker

PRIMARY_COLOR = "red"
MIRROR_COLOR = "orange"


class PeakFitWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("Diffractogram Peak Fit")
        self.resize(1500, 850)

        self._selection: tuple[int, int, int] | None = None  # (row, col, fit_rad)
        self._last_fit: dict | None = None
        self._last_maps: tuple[np.ndarray, np.ndarray, np.ndarray] | None = None

        self._maps_thread: QThread | None = None
        self._maps_worker: StructureMapWorker | None = None

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
        right_layout.addWidget(self._build_fit_group())
        self.calibration = CalibrationWidget()
        self.calibration.changed.connect(self._on_calibration_changed)
        right_layout.addWidget(self.calibration)
        right_layout.addWidget(self._build_batch_group())
        right_layout.addStretch(1)

        right_scroll = QScrollArea()
        right_scroll.setWidgetResizable(True)
        right_scroll.setWidget(right_column)
        right_scroll.setFixedWidth(400)
        root.addWidget(right_scroll)

    def _build_center_tabs(self) -> QTabWidget:
        tabs = QTabWidget()

        diffractogram_tab = QWidget()
        diffractogram_layout = QVBoxLayout(diffractogram_tab)
        diffractogram_layout.addWidget(QLabel("Click a peak below - its opposite reflection is found automatically."))
        self.diffractogram_canvas = FigureCanvas(Figure(figsize=(6, 6)))
        self.diffractogram_canvas.setMinimumSize(480, 480)
        self.diffractogram_canvas.mpl_connect("button_press_event", self.on_diffractogram_click)
        diffractogram_layout.addWidget(self.diffractogram_canvas)
        tabs.addTab(diffractogram_tab, "Diffractogram")

        maps_tab = QWidget()
        maps_tab_layout = QVBoxLayout(maps_tab)

        maps_scroll = QScrollArea()
        maps_scroll.setWidgetResizable(True)
        maps_content = QWidget()
        maps_layout = QVBoxLayout(maps_content)
        # Maps are stacked vertically (spacing/angle/intensity), so this can
        # get tall - the canvas's minimum size (set after each redraw) is
        # what makes this scroll instead of squashing the maps down.
        self.maps_canvas = FigureCanvas(Figure(figsize=(6, 9)))
        maps_layout.addWidget(self.maps_canvas)
        maps_scroll.setWidget(maps_content)
        maps_tab_layout.addWidget(maps_scroll)

        self.maps_save_btn = QPushButton("Save maps…")
        self.maps_save_btn.setEnabled(False)
        self.maps_save_btn.clicked.connect(self.on_save_maps_clicked)
        maps_tab_layout.addWidget(self.maps_save_btn)
        tabs.addTab(maps_tab, "Maps")

        self._center_tabs = tabs
        self._maps_tab_index = 1
        return tabs

    def _build_fit_group(self) -> QGroupBox:
        group = QGroupBox("2D Peak Fit")
        v = QVBoxLayout(group)

        form = QFormLayout()
        self.fit_radius_spin = QSpinBox()
        self.fit_radius_spin.setRange(2, 32)
        self.fit_radius_spin.setValue(5)
        self.fit_radius_spin.valueChanged.connect(self._on_fit_radius_changed)
        form.addRow("Fit radius (px)", self.fit_radius_spin)

        self.highpass_radius_spin = QSpinBox()
        self.highpass_radius_spin.setRange(0, 64)
        self.highpass_radius_spin.setValue(3)
        form.addRow("Center mask radius (px)", self.highpass_radius_spin)

        self.detection_threshold_spin = QDoubleSpinBox()
        self.detection_threshold_spin.setRange(1.0, 50.0)
        self.detection_threshold_spin.setValue(3.0)
        self.detection_threshold_spin.setSingleStep(0.5)
        form.addRow("Detection threshold", self.detection_threshold_spin)
        v.addLayout(form)

        self.fit_btn = QPushButton("Fit 2D Gaussian")
        self.fit_btn.setEnabled(False)
        self.fit_btn.clicked.connect(self.on_fit_clicked)
        v.addWidget(self.fit_btn)

        self.result_label = QLabel("Load an image, then click a peak in the diffractogram.")
        self.result_label.setWordWrap(True)
        v.addWidget(self.result_label)

        self.fit_canvas = FigureCanvas(Figure(figsize=(4, 4)))
        self.fit_canvas.setMinimumHeight(320)
        v.addWidget(self.fit_canvas)

        return group

    def _build_batch_group(self) -> QGroupBox:
        group = QGroupBox("Generate Maps")
        v = QVBoxLayout(group)
        v.addWidget(QLabel("Fits the currently selected reflection across every patch in the image."))

        form = QFormLayout()
        self.batch_window_step_spin = QSpinBox()
        self.batch_window_step_spin.setRange(1, 4096)
        self.batch_window_step_spin.setValue(4)
        form.addRow("Window step", self.batch_window_step_spin)
        v.addLayout(form)

        self.generate_maps_btn = QPushButton("Generate Maps")
        self.generate_maps_btn.setEnabled(False)
        self.generate_maps_btn.clicked.connect(self.on_generate_maps_clicked)
        v.addWidget(self.generate_maps_btn)

        self.maps_progress = QProgressBar()
        v.addWidget(self.maps_progress)

        return group

    # --------------------------------------------------------- patch state

    def _on_image_loaded(self) -> None:
        # Pixel size is a property of the raw image itself, not the current
        # view of it - it must not survive to a different image.
        self.calibration.reset()

    def _on_patch_changed(self) -> None:
        # The diffractogram itself just changed (new image, or a different
        # window/crop/hamming/position) - any previous peak selection or fit
        # was about the old pattern and no longer applies.
        self._clear_selection()
        raw_image = self.patch_selector.raw_image
        self.calibration.set_image_reference(raw_image.shape[0] if raw_image is not None else None)
        if raw_image is not None:
            # A fit window can't exceed the diffractogram itself.
            self.fit_radius_spin.setRange(2, max(2, self.patch_selector.crop_size // 2))
        spectrum = self.patch_selector.current_spectrum()
        if spectrum is not None:
            plotting.draw_diffractogram(self.diffractogram_canvas.figure, spectrum)
            self.diffractogram_canvas.draw_idle()

    def _clear_selection(self) -> None:
        self._selection = None
        self._last_fit = None
        self.fit_btn.setEnabled(False)
        self.calibration.set_reflection_reference(None, None)
        self.generate_maps_btn.setEnabled(False)
        self.result_label.setText("Click a peak in the diffractogram to begin.")
        self.fit_canvas.figure.clear()
        self.fit_canvas.draw_idle()

    def _on_calibration_changed(self) -> None:
        if self._last_fit is not None:
            self.result_label.setText(self._compose_result_text())

    def _on_fit_radius_changed(self) -> None:
        if self._selection is None:
            return
        row, col, _ = self._selection
        self._set_selection(row, col)

    def _set_selection(self, row: int, col: int) -> None:
        fit_rad = self.fit_radius_spin.value()
        self._selection = (row, col, fit_rad)
        self.fit_btn.setEnabled(True)
        self.result_label.setText(
            f"Selected peak at (row={row}, col={col}) - opposite reflection identified automatically.\n"
            "Click 'Fit 2D Gaussian' to fit both."
        )
        self._redraw_diffractogram_overlay()

    def _redraw_diffractogram_overlay(self) -> None:
        spectrum = self.patch_selector.current_spectrum()
        if spectrum is None:
            return
        selections = None
        if self._selection is not None:
            row, col, fit_rad = self._selection
            size = np.abs(spectrum).shape[0]
            mirror_row, mirror_col = core_structuremaps.mirror_position(row, col, size)
            selections = [(row, col, fit_rad, PRIMARY_COLOR), (mirror_row, mirror_col, fit_rad, MIRROR_COLOR)]
        plotting.draw_diffractogram(self.diffractogram_canvas.figure, spectrum, selections=selections)
        self.diffractogram_canvas.draw_idle()

    # ----------------------------------------------------------------- fit

    def on_diffractogram_click(self, event) -> None:
        if self.patch_selector.current_spectrum() is None or event.xdata is None or event.ydata is None:
            return
        col = int(round(event.xdata))
        row = int(round(event.ydata))
        self._set_selection(row, col)

    def on_fit_clicked(self) -> None:
        spectrum = self.patch_selector.current_spectrum()
        if spectrum is None or self._selection is None:
            return
        row, col, fit_rad = self._selection

        result = core_structuremaps.fit_reflection_pair(
            np.abs(spectrum),
            row,
            col,
            fit_rad,
            min_peak_ratio=self.detection_threshold_spin.value(),
            highpass_radius=self.highpass_radius_spin.value(),
        )
        if result is None:
            QMessageBox.warning(
                self,
                "No reflection detected",
                "Neither this peak nor its mirror stands out enough above the local "
                "background here. Try a different location, a larger fit radius, or "
                "lowering the detection threshold.",
            )
            return

        self._last_fit = result
        self.calibration.set_reflection_reference(result["distance_px"], self.patch_selector.window_size)
        self.generate_maps_btn.setEnabled(True)
        self.result_label.setText(self._compose_result_text())

        region1, region2 = result["region1"], result["region2"]
        model1 = peakfit.gaussian_2d(*result["fit1"])(*np.indices(region1.shape))
        model2 = peakfit.gaussian_2d(*result["fit2"])(*np.indices(region2.shape))
        plotting.draw_peak_pair_fit(self.fit_canvas.figure, region1, model1, region2, model2)
        self.fit_canvas.draw_idle()

        # Refine the overlay/target to the fitted (sub-pixel) primary center.
        abs_row, abs_col = result["center1"]
        self._selection = (round(abs_row), round(abs_col), fit_rad)
        self._redraw_diffractogram_overlay()

    def _compose_result_text(self) -> str:
        f = self._last_fit
        height1, _, _, width1, _ = f["fit1"]
        height2, _, _, width2, _ = f["fit2"]
        spacing_text = self.calibration.format_spacing(f["distance_px"], self.patch_selector.window_size)
        return (
            f"Peak 1 center: (row={f['center1'][0]:.2f}, col={f['center1'][1]:.2f})  height={height1:.3g}\n"
            f"Peak 2 (mirror) center: (row={f['center2'][0]:.2f}, col={f['center2'][1]:.2f})  height={height2:.3g}\n"
            f"Spacing: {spacing_text}\n"
            f"Angle: {f['angle_deg']:.2f}°\n"
            f"Mean intensity: {f['mean_height']:.3g}   Mean width: {(width1 + width2) / 2:.3f} px"
        )

    # ----------------------------------------------------------------- maps

    def on_generate_maps_clicked(self) -> None:
        raw_image = self.patch_selector.raw_image
        if raw_image is None or self._selection is None:
            return
        row, col, fit_rad = self._selection

        self.generate_maps_btn.setEnabled(False)
        self.maps_progress.setRange(0, 1)
        self.maps_progress.setValue(0)

        self._maps_thread = QThread()
        self._maps_worker = StructureMapWorker(
            raw_image,
            self.patch_selector.window_size,
            self.batch_window_step_spin.value(),
            self.patch_selector.hamming,
            self.patch_selector.crop_size,
            row,
            col,
            fit_rad,
            self.calibration.pixel_size_nm,
            self.detection_threshold_spin.value(),
            self.highpass_radius_spin.value(),
        )
        self._maps_worker.moveToThread(self._maps_thread)
        self._maps_thread.started.connect(self._maps_worker.run)
        self._maps_worker.progress.connect(self._on_maps_progress)
        self._maps_worker.finished.connect(self._on_maps_finished)
        self._maps_worker.error.connect(self._on_maps_error)
        self._maps_worker.finished.connect(self._maps_thread.quit)
        self._maps_worker.error.connect(self._maps_thread.quit)
        self._maps_thread.finished.connect(self._maps_thread.deleteLater)
        self._maps_thread.start()

    def _on_maps_progress(self, done: int, total: int) -> None:
        self.maps_progress.setRange(0, total)
        self.maps_progress.setValue(done)

    def _on_maps_finished(self, spacing_map: np.ndarray, angle_map: np.ndarray, intensity_map: np.ndarray) -> None:
        self.generate_maps_btn.setEnabled(True)
        self.maps_save_btn.setEnabled(True)

        if self.calibration.pixel_size_nm is not None:
            # compute_structure_maps returns nm when calibrated - convert to
            # the displayed/saved unit (Angstroms, the natural scale for
            # lattice spacings) once, here, so what's shown and what's saved
            # always agree.
            spacing_map = spacing_map / NM_PER_ANGSTROM
            spacing_label = "Spacing (Å)"
        else:
            spacing_label = "Spacing (reciprocal px)"
        self._last_maps = (spacing_map, angle_map, intensity_map)

        # A few edge/artifact patches can otherwise dominate the angle and
        # intensity colorbars and wash out the rest of the map.
        angle_vlimits = plotting.percentile_clip(angle_map)
        intensity_vlimits = plotting.percentile_clip(intensity_map)
        plotting.draw_grid(
            self.maps_canvas.figure,
            [spacing_map, angle_map, intensity_map],
            [spacing_label, "Angle (deg)", "Intensity"],
            ncols=1,
            vlimits=[None, angle_vlimits, intensity_vlimits],
        )
        size_canvas_to_figure(self.maps_canvas)
        self.maps_canvas.draw_idle()
        self._center_tabs.setCurrentIndex(self._maps_tab_index)

    def _on_maps_error(self, message: str) -> None:
        self.generate_maps_btn.setEnabled(True)
        QMessageBox.critical(self, "Generate maps failed", message)

    def on_save_maps_clicked(self) -> None:
        if self._last_maps is None:
            return
        directory = QFileDialog.getExistingDirectory(self, "Choose output folder")
        if not directory:
            return
        prefix, ok = QInputDialog.getText(self, "Filename prefix", "Prefix for saved files:", text="structure")
        if not ok:
            return
        prefix = prefix.strip() or "structure"
        out = Path(directory)
        spacing_map, angle_map, intensity_map = self._last_maps
        np.save(out / f"{prefix}_spacing.npy", spacing_map)
        np.save(out / f"{prefix}_angle.npy", angle_map)
        np.save(out / f"{prefix}_intensity.npy", intensity_map)
        QMessageBox.information(self, "Saved", f"Saved structure maps to {out}")
