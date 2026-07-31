"""Carbon Tools: workflow for graphitic/carbonaceous materials, where one
dominant reflection (graphite (002), d ~ 3.35-3.4 A) is present everywhere
but its orientation varies domain to domain - potentially across the full
range patch to patch (tight rings, random-textured regions), not just
gradual drift.

Reuses `ReflectionFitWidget` (same single-patch fit UI as `PeakFitWindow`,
with an added auto-find step since the spacing is already roughly known) for
previewing the reflection and driving calibration. The actual whole-image
orientation maps, though, come from `core.carbontools.compute_orientation_maps`
- which re-searches the full spacing annulus for *every* patch independently,
unlike `ReflectionFitWidget`'s own "Generate Maps" (which tracks one
reflection from a fixed position, appropriate for gradual drift but not for
the full-range orientation variation this module is meant to handle).

Downstream of that: color-wheel orientation coloring, SLIC segmentation of
the colorized map, and lognormal/bi-lognormal domain size-distribution
fitting.
"""

from pathlib import Path

import numpy as np
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from PySide6.QtCore import QThread
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
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

from diffraction_diver.core import carbontools as core_carbontools
from diffraction_diver.core import io as core_io
from diffraction_diver.gui import plotting
from diffraction_diver.gui.calibration_widget import NM_PER_ANGSTROM, CalibrationWidget
from diffraction_diver.gui.canvas_utils import harden_scroll_repaint, size_canvas_to_figure
from diffraction_diver.gui.patch_selector import PatchSelectorWidget
from diffraction_diver.gui.reflection_fit_widget import ReflectionFitWidget
from diffraction_diver.gui.workers import OrientationMapWorker


class CarbonToolsWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("Carbon Tools")
        self.resize(1650, 900)

        self._angle_map: np.ndarray | None = None
        self._intensity_map: np.ndarray | None = None
        self._map_window_step: int | None = None
        self._color_wheel = core_carbontools.generate_color_wheel()
        self._orientation_image: np.ndarray | None = None
        self._segment_labels: np.ndarray | None = None
        self._segment_areas: np.ndarray | None = None
        self._last_orientation_maps: tuple[np.ndarray, np.ndarray, np.ndarray] | None = None
        self._mask_source_image: np.ndarray | None = None
        self._mask: np.ndarray | None = None

        self._orientation_thread: QThread | None = None
        self._orientation_worker: OrientationMapWorker | None = None

        self._build_ui()

    # ------------------------------------------------------------------ UI

    def _build_ui(self) -> None:
        central = QWidget()
        self.setCentralWidget(central)
        root = QHBoxLayout(central)

        # For carbon specifically (a very light element), a bright-field
        # image has more diffraction/lattice information and needs no
        # thresholding, while a companion dark-field image (higher
        # material-vs-vacuum contrast, vacuum reads black) is better suited
        # to building a mask - so BF is the primary analysis image here and
        # DF is loaded separately, mask-only, below. This BF/DF split is
        # carbon-specific; for heavier materials either image works fine for
        # diffraction, so only Carbon Tools uses this labeling.
        left_column = QWidget()
        left_column.setFixedWidth(440)
        left_layout = QVBoxLayout(left_column)
        left_layout.setContentsMargins(0, 0, 0, 0)

        self.patch_selector = PatchSelectorWidget(
            show_patch_preview=False, load_button_label="Load BF Image…"
        )
        self.patch_selector.changed.connect(self._update_autofind_enabled)
        left_layout.addWidget(self.patch_selector)

        self.calibration = CalibrationWidget()
        self.calibration.changed.connect(self._on_calibration_changed)
        left_layout.addWidget(self.calibration)
        left_layout.addStretch(1)

        root.addWidget(left_column)

        # Smaller than the default 480 - Carbon Tools drives selection almost
        # entirely via auto-find rather than manual zoom-in clicking, so the
        # diffractogram doesn't need as much room; freed space goes to the
        # auto-find and mask groups sharing this tab. Calibration is shown in
        # the persistent left column instead (so it's visible on every tab,
        # not just this one).
        self.reflection_fit = ReflectionFitWidget(
            self.patch_selector,
            self.calibration,
            show_batch=False,
            diffractogram_min_size=320,
            show_calibration=False,
        )

        tabs = QTabWidget()
        tabs.addTab(self._build_reflection_tab(), "Reflection")
        tabs.addTab(self._build_orientation_tab(), "Orientation")
        tabs.addTab(self._build_segmentation_tab(), "Segmentation")
        tabs.addTab(self._build_size_distribution_tab(), "Size Distribution")
        root.addWidget(tabs, stretch=1)

    def _build_reflection_tab(self) -> QWidget:
        tab = QWidget()
        layout = QVBoxLayout(tab)
        top_row = QHBoxLayout()
        top_row.addWidget(self._build_autofind_group())
        top_row.addWidget(self._build_mask_group())
        layout.addLayout(top_row)
        layout.addWidget(self.reflection_fit, stretch=1)
        return tab

    def _build_autofind_group(self) -> QGroupBox:
        group = QGroupBox("Auto-Find Reflection")
        # Full rationale in the tooltip rather than an inline label - this
        # group sits beside the mask group and above the diffractogram, so
        # keeping it compact matters more than an always-visible paragraph.
        group.setToolTip(
            "Graphitic materials show one dominant reflection at a roughly known "
            "spacing - once calibrated, this searches for the strongest peak in "
            "that range instead of requiring a manual click."
        )
        v = QVBoxLayout(group)

        spacing_row = QHBoxLayout()
        spacing_row.addWidget(QLabel("Spacing"))
        self.spacing_min_spin = QDoubleSpinBox()
        self.spacing_min_spin.setRange(0.5, 20.0)
        self.spacing_min_spin.setValue(3.0)
        self.spacing_min_spin.setSuffix(" Å")
        spacing_row.addWidget(self.spacing_min_spin)

        spacing_row.addWidget(QLabel("to"))
        self.spacing_max_spin = QDoubleSpinBox()
        self.spacing_max_spin.setRange(0.5, 20.0)
        self.spacing_max_spin.setValue(4.5)
        self.spacing_max_spin.setSuffix(" Å")
        spacing_row.addWidget(self.spacing_max_spin)
        v.addLayout(spacing_row)

        self.autofind_btn = QPushButton("Auto-Find && Fit")
        self.autofind_btn.setEnabled(False)
        self.autofind_btn.clicked.connect(self.on_autofind_clicked)
        v.addWidget(self.autofind_btn)

        return group

    def _build_orientation_tab(self) -> QWidget:
        # One scroll area for the whole tab (rather than the previous mix of
        # an internally-capped 420px scroll for the maps group plus an
        # unscrolled canvas below it) - that fragmented setup was both
        # artificially squashing the 3-stacked maps to a sliver AND was the
        # actual source of the repaint ghosting. A single outer scroll lets
        # every panel show at its natural size and only scrolls the page as
        # a whole when it doesn't fit.
        tab = QWidget()
        tab_layout = QVBoxLayout(tab)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        content = QWidget()
        layout = QVBoxLayout(content)

        layout.addWidget(self._build_generate_maps_group())

        btn_row = QHBoxLayout()
        self.color_wheel_btn = QPushButton("Color Wheel")
        self.color_wheel_btn.setEnabled(False)
        self.color_wheel_btn.clicked.connect(lambda: self.on_colorize_clicked(False))
        btn_row.addWidget(self.color_wheel_btn)

        self.color_wheel_intensity_btn = QPushButton("Color Wheel (scaled by intensity)")
        self.color_wheel_intensity_btn.setEnabled(False)
        self.color_wheel_intensity_btn.clicked.connect(lambda: self.on_colorize_clicked(True))
        btn_row.addWidget(self.color_wheel_intensity_btn)
        layout.addLayout(btn_row)

        self.orientation_canvas = FigureCanvas(Figure(figsize=(6, 6)))
        size_canvas_to_figure(self.orientation_canvas)
        layout.addWidget(self.orientation_canvas)

        scroll.setWidget(content)
        harden_scroll_repaint(scroll, content)
        tab_layout.addWidget(scroll)

        return tab

    def _build_mask_group(self) -> QGroupBox:
        group = QGroupBox("Material Mask (optional)")
        v = QVBoxLayout(group)
        v.addWidget(
            QLabel(
                "Load a dark-field/HAADF companion image (same field of view as the "
                "BF image) to skip empty/vacuum regions entirely when generating "
                "orientation maps - vacuum reads black in DF, so thresholding it "
                "directly is more reliable than relying on FFT-domain detection alone."
            )
        )

        load_df_btn = QPushButton("Load DF Image…")
        load_df_btn.clicked.connect(self.on_load_df_image_clicked)
        v.addWidget(load_df_btn)

        self.mask_file_label = QLabel("No DF image loaded.")
        self.mask_file_label.setWordWrap(True)
        v.addWidget(self.mask_file_label)

        form = QFormLayout()
        self.mask_threshold_spin = QDoubleSpinBox()
        self.mask_threshold_spin.setDecimals(4)
        self.mask_threshold_spin.setRange(-1e12, 1e12)
        self.mask_threshold_spin.valueChanged.connect(self._on_mask_threshold_changed)
        form.addRow("Threshold", self.mask_threshold_spin)
        v.addLayout(form)

        self.use_mask_check = QCheckBox("Use mask when generating orientation maps")
        self.use_mask_check.setEnabled(False)
        v.addWidget(self.use_mask_check)

        self.mask_preview_canvas = FigureCanvas(Figure(figsize=(4, 4)))
        size_canvas_to_figure(self.mask_preview_canvas)
        v.addWidget(self.mask_preview_canvas)

        return group

    def _build_generate_maps_group(self) -> QGroupBox:
        group = QGroupBox("Generate Orientation Maps")
        v = QVBoxLayout(group)
        v.addWidget(
            QLabel(
                "Searches the same spacing range independently for every patch "
                "(not a fixed position refined locally like the Reflection tab's "
                "own 'Generate Maps') - needed since orientation can vary across "
                "the full range patch to patch, not just drift gradually."
            )
        )

        form = QFormLayout()
        self.orientation_window_step_spin = QSpinBox()
        self.orientation_window_step_spin.setRange(1, 4096)
        self.orientation_window_step_spin.setValue(4)
        form.addRow("Window step", self.orientation_window_step_spin)
        v.addLayout(form)

        self.generate_orientation_maps_btn = QPushButton("Generate Orientation Maps")
        self.generate_orientation_maps_btn.setEnabled(False)
        self.generate_orientation_maps_btn.clicked.connect(self.on_generate_orientation_maps_clicked)
        v.addWidget(self.generate_orientation_maps_btn)

        self.orientation_maps_progress = QProgressBar()
        v.addWidget(self.orientation_maps_progress)

        # No nested scroll area here - the whole tab now scrolls as one unit
        # (see _build_orientation_tab), so this just shows at its natural
        # (tall, 3-stacked) size.
        self.orientation_maps_canvas = FigureCanvas(Figure(figsize=(6, 9)))
        size_canvas_to_figure(self.orientation_maps_canvas)
        v.addWidget(self.orientation_maps_canvas)

        self.save_orientation_maps_btn = QPushButton("Save maps…")
        self.save_orientation_maps_btn.setEnabled(False)
        self.save_orientation_maps_btn.clicked.connect(self.on_save_orientation_maps_clicked)
        v.addWidget(self.save_orientation_maps_btn)

        return group

    def _build_segmentation_tab(self) -> QWidget:
        tab = QWidget()
        layout = QVBoxLayout(tab)
        layout.addWidget(QLabel("Generate an orientation coloring in the Orientation tab first."))

        form = QFormLayout()
        self.n_segments_spin = QSpinBox()
        self.n_segments_spin.setRange(2, 5000)
        self.n_segments_spin.setValue(600)
        form.addRow("Target segments", self.n_segments_spin)

        self.compactness_spin = QDoubleSpinBox()
        self.compactness_spin.setRange(0.1, 100.0)
        self.compactness_spin.setValue(20.0)
        form.addRow("Compactness", self.compactness_spin)
        layout.addLayout(form)

        self.segment_btn = QPushButton("Segment")
        self.segment_btn.setEnabled(False)
        self.segment_btn.clicked.connect(self.on_segment_clicked)
        layout.addWidget(self.segment_btn)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        content = QWidget()
        content.setAutoFillBackground(True)
        content_layout = QVBoxLayout(content)
        self.segmentation_canvas = FigureCanvas(Figure(figsize=(6, 6)))
        size_canvas_to_figure(self.segmentation_canvas)
        content_layout.addWidget(self.segmentation_canvas)
        scroll.setWidget(content)
        layout.addWidget(scroll)

        return tab

    def _build_size_distribution_tab(self) -> QWidget:
        tab = QWidget()
        layout = QVBoxLayout(tab)

        self.compute_sizes_btn = QPushButton("Compute Segment Sizes")
        self.compute_sizes_btn.setEnabled(False)
        self.compute_sizes_btn.clicked.connect(self.on_compute_sizes_clicked)
        layout.addWidget(self.compute_sizes_btn)

        form = QFormLayout()
        self.fit_type_combo = QComboBox()
        self.fit_type_combo.addItems(["Lognormal", "Bi-lognormal"])
        form.addRow("Fit type", self.fit_type_combo)

        self.hist_bins_spin = QSpinBox()
        self.hist_bins_spin.setRange(5, 200)
        self.hist_bins_spin.setValue(28)
        form.addRow("Histogram bins", self.hist_bins_spin)
        layout.addLayout(form)

        self.fit_btn = QPushButton("Fit Distribution")
        self.fit_btn.setEnabled(False)
        self.fit_btn.clicked.connect(self.on_fit_distribution_clicked)
        layout.addWidget(self.fit_btn)

        self.save_areas_btn = QPushButton("Save segment areas…")
        self.save_areas_btn.setEnabled(False)
        self.save_areas_btn.clicked.connect(self.on_save_areas_clicked)
        layout.addWidget(self.save_areas_btn)

        self.fit_result_label = QLabel("Segment the orientation map first.")
        self.fit_result_label.setWordWrap(True)
        layout.addWidget(self.fit_result_label)

        self.histogram_canvas = FigureCanvas(Figure(figsize=(6, 4.5)))
        size_canvas_to_figure(self.histogram_canvas)
        layout.addWidget(self.histogram_canvas)

        return tab

    # ------------------------------------------------------- shared state

    def _on_calibration_changed(self) -> None:
        self.patch_selector.set_pixel_size(self.calibration.pixel_size_nm)
        self._update_autofind_enabled()

    def _update_autofind_enabled(self) -> None:
        enabled = self.calibration.pixel_size_nm is not None and self.patch_selector.raw_image is not None
        self.autofind_btn.setEnabled(enabled)
        self.generate_orientation_maps_btn.setEnabled(enabled)

    def _spacing_annulus_px(self, window_size: int) -> tuple[float, float]:
        """(r_min, r_max) in reciprocal pixels for the current spacing spinboxes."""
        pixel_size_nm = self.calibration.pixel_size_nm
        d_min_nm = self.spacing_min_spin.value() * NM_PER_ANGSTROM
        d_max_nm = self.spacing_max_spin.value() * NM_PER_ANGSTROM
        # d = window_size*pixel_size_nm / r  =>  r = window_size*pixel_size_nm / d
        # (larger spacing -> smaller radius, so d_max maps to r_min)
        r_min = (window_size * pixel_size_nm) / d_max_nm
        r_max = (window_size * pixel_size_nm) / d_min_nm
        return r_min, r_max

    # ------------------------------------------------------------ auto-find

    def on_autofind_clicked(self) -> None:
        spectrum = self.patch_selector.current_spectrum()
        if spectrum is None or self.calibration.pixel_size_nm is None:
            return

        r_min, r_max = self._spacing_annulus_px(self.patch_selector.window_size)
        highpass_radius = self.reflection_fit.highpass_radius_spin.value()
        min_peak_ratio = self.reflection_fit.detection_threshold_spin.value()
        found = core_carbontools.find_reflection_in_annulus(
            np.abs(spectrum), r_min, r_max, highpass_radius, min_peak_ratio
        )
        if found is None:
            QMessageBox.warning(
                self,
                "No reflection found",
                f"No pixel found in the {self.spacing_min_spin.value():g}-"
                f"{self.spacing_max_spin.value():g} Å spacing range.",
            )
            return

        row, col = found
        self.reflection_fit.set_selection(row, col)
        self.reflection_fit.fit()

    # --------------------------------------------------------------- mask

    def on_load_df_image_clicked(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Load dark-field image",
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

        raw_image = self.patch_selector.raw_image
        if raw_image is not None and image.shape != raw_image.shape:
            QMessageBox.warning(
                self,
                "Shape mismatch",
                f"DF image shape {image.shape} doesn't match the loaded BF image "
                f"shape {raw_image.shape}. Load a companion image with the same field of view.",
            )
            return

        self._mask_source_image = image
        self.mask_file_label.setText(f"DF image: {Path(path).name}  ({image.shape[0]}×{image.shape[1]})")

        lo, hi = float(image.min()), float(image.max())
        self.mask_threshold_spin.blockSignals(True)
        self.mask_threshold_spin.setRange(lo, hi)
        self.mask_threshold_spin.setValue(lo + (hi - lo) / 2)
        self.mask_threshold_spin.blockSignals(False)

        self.use_mask_check.setEnabled(True)
        self.use_mask_check.setChecked(True)
        self._update_mask_preview()

    def _on_mask_threshold_changed(self) -> None:
        self._update_mask_preview()

    def _update_mask_preview(self) -> None:
        if self._mask_source_image is None:
            return
        self._mask = core_carbontools.threshold_mask(self._mask_source_image, self.mask_threshold_spin.value())
        plotting.draw_labeled_image(
            self.mask_preview_canvas.figure,
            self._mask.astype(float),
            "Material Mask (white = material)",
            cmap="gray",
            vlimits=(0.0, 1.0),
        )
        self.mask_preview_canvas.draw_idle()

    # ---------------------------------------------------------- orientation

    def on_generate_orientation_maps_clicked(self) -> None:
        raw_image = self.patch_selector.raw_image
        if raw_image is None or self.calibration.pixel_size_nm is None:
            return

        window_size = self.patch_selector.window_size
        r_min, r_max = self._spacing_annulus_px(window_size)
        mask = self._mask if (self.use_mask_check.isChecked() and self._mask is not None) else None
        if mask is not None and mask.shape != raw_image.shape:
            QMessageBox.warning(
                self,
                "Shape mismatch",
                "The loaded DF mask no longer matches the current BF image shape - "
                "reload a matching DF image, or uncheck 'Use mask'.",
            )
            return

        self.generate_orientation_maps_btn.setEnabled(False)
        self.orientation_maps_progress.setRange(0, 1)
        self.orientation_maps_progress.setValue(0)

        self._orientation_thread = QThread()
        self._orientation_worker = OrientationMapWorker(
            raw_image,
            window_size,
            self.orientation_window_step_spin.value(),
            self.patch_selector.hamming,
            self.patch_selector.crop_size,
            r_min,
            r_max,
            self.reflection_fit.fit_radius_spin.value(),
            self.calibration.pixel_size_nm,
            self.reflection_fit.detection_threshold_spin.value(),
            self.reflection_fit.highpass_radius_spin.value(),
            mask=mask,
        )
        self._orientation_worker.moveToThread(self._orientation_thread)
        self._orientation_thread.started.connect(self._orientation_worker.run)
        self._orientation_worker.progress.connect(self._on_orientation_maps_progress)
        self._orientation_worker.finished.connect(self._on_orientation_maps_finished)
        self._orientation_worker.error.connect(self._on_orientation_maps_error)
        self._orientation_worker.finished.connect(self._orientation_thread.quit)
        self._orientation_worker.error.connect(self._orientation_thread.quit)
        self._orientation_thread.finished.connect(self._orientation_thread.deleteLater)
        self._orientation_thread.start()

    def _on_orientation_maps_progress(self, done: int, total: int) -> None:
        self.orientation_maps_progress.setRange(0, total)
        self.orientation_maps_progress.setValue(done)

    def _on_orientation_maps_finished(
        self, spacing_map: np.ndarray, angle_map: np.ndarray, intensity_map: np.ndarray
    ) -> None:
        self.generate_orientation_maps_btn.setEnabled(True)
        self.save_orientation_maps_btn.setEnabled(True)
        self._angle_map = angle_map
        self._intensity_map = intensity_map
        self._map_window_step = self.orientation_window_step_spin.value()

        if self.calibration.pixel_size_nm is not None:
            spacing_map = spacing_map / NM_PER_ANGSTROM
            spacing_label = "Spacing (Å)"
        else:
            spacing_label = "Spacing (reciprocal px)"
        self._last_orientation_maps = (spacing_map, angle_map, intensity_map)

        map_pixel_size = self._map_window_step * (self.calibration.pixel_size_nm or 0)
        plotting.draw_structure_maps(
            self.orientation_maps_canvas.figure,
            spacing_map,
            angle_map,
            intensity_map,
            spacing_label,
            map_pixel_size=map_pixel_size or None,
        )
        size_canvas_to_figure(self.orientation_maps_canvas)
        self.orientation_maps_canvas.draw_idle()

        self.color_wheel_btn.setEnabled(True)
        self.color_wheel_intensity_btn.setEnabled(True)

    def _on_orientation_maps_error(self, message: str) -> None:
        self.generate_orientation_maps_btn.setEnabled(True)
        QMessageBox.critical(self, "Generate orientation maps failed", message)

    def on_save_orientation_maps_clicked(self) -> None:
        if self._last_orientation_maps is None:
            return
        directory = QFileDialog.getExistingDirectory(self, "Choose output folder")
        if not directory:
            return
        prefix, ok = QInputDialog.getText(self, "Filename prefix", "Prefix for saved files:", text="orientation")
        if not ok:
            return
        prefix = prefix.strip() or "orientation"
        out = Path(directory)
        spacing_map, angle_map, intensity_map = self._last_orientation_maps
        np.save(out / f"{prefix}_spacing.npy", spacing_map)
        np.save(out / f"{prefix}_angle.npy", angle_map)
        np.save(out / f"{prefix}_intensity.npy", intensity_map)
        QMessageBox.information(self, "Saved", f"Saved orientation maps to {out}")

    def on_colorize_clicked(self, scaled: bool) -> None:
        if self._angle_map is None:
            return
        intensity = self._intensity_map if scaled else None
        rgb = core_carbontools.colorize_orientation(self._angle_map, self._color_wheel, intensity_map=intensity)
        self._orientation_image = rgb

        title = "Orientation (color wheel, intensity-scaled)" if scaled else "Orientation (color wheel)"
        scale_bar = None
        if self.calibration.pixel_size_nm is not None and self._map_window_step is not None:
            map_pixel_size = self._map_window_step * self.calibration.pixel_size_nm
            scale_bar = {"pixel_size": map_pixel_size, "units": "nm"}
        plotting.draw_orientation_with_legend(
            self.orientation_canvas.figure, rgb, self._color_wheel, title, scale_bar=scale_bar
        )
        size_canvas_to_figure(self.orientation_canvas)
        self.orientation_canvas.draw_idle()

        self.segment_btn.setEnabled(True)

    # --------------------------------------------------------- segmentation

    def on_segment_clicked(self) -> None:
        if self._orientation_image is None:
            return
        labels = core_carbontools.segment_orientation_map(
            self._orientation_image,
            n_segments=self.n_segments_spin.value(),
            compactness=self.compactness_spin.value(),
        )
        self._segment_labels = labels
        avg_rgb = core_carbontools.label_to_average_rgb(labels, self._orientation_image)
        n_found = len(np.unique(labels))
        plotting.draw_rgb_image(self.segmentation_canvas.figure, avg_rgb, f"Segmented ({n_found} segments)")
        size_canvas_to_figure(self.segmentation_canvas)
        self.segmentation_canvas.draw_idle()

        self.compute_sizes_btn.setEnabled(True)

    # ---------------------------------------------------- size distribution

    def on_compute_sizes_clicked(self) -> None:
        if self._segment_labels is None:
            return
        if self.calibration.pixel_size_nm is None or self._map_window_step is None:
            QMessageBox.warning(self, "Not calibrated", "Calibration is required to compute physical segment areas.")
            return
        map_pixel_area = (self._map_window_step * self.calibration.pixel_size_nm) ** 2
        self._segment_areas = core_carbontools.segment_areas(self._segment_labels, map_pixel_area)
        self.fit_btn.setEnabled(True)
        self.save_areas_btn.setEnabled(True)
        self.fit_result_label.setText(
            f"{len(self._segment_areas)} segments. Choose a fit type and click 'Fit Distribution'."
        )
        self._draw_histogram()

    def _draw_histogram(self, fit_curve: np.ndarray | None = None, fit_label: str | None = None) -> None:
        fig = self.histogram_canvas.figure
        fig.clear()
        ax = fig.add_subplot(111)
        bins = self.hist_bins_spin.value()
        max_area = float(self._segment_areas.max()) if self._segment_areas.size else 1.0
        _yh, xh = np.histogram(self._segment_areas, bins=bins, range=(0, max_area))
        centers = xh[1:]
        ax.hist(self._segment_areas, bins=bins, range=(0, max_area), facecolor="b", alpha=0.7)
        if fit_curve is not None:
            ax.plot(centers, fit_curve, "g", linewidth=2, label=fit_label)
            ax.legend()
        ax.set_xlabel("Area (nm²)")
        ax.set_ylabel("Segment count")
        ax.set_title("Segment Size Distribution")
        fig.tight_layout()
        self.histogram_canvas.draw_idle()

    def on_fit_distribution_clicked(self) -> None:
        if self._segment_areas is None:
            return
        bins = self.hist_bins_spin.value()
        max_area = float(self._segment_areas.max())
        yh, xh = np.histogram(self._segment_areas, bins=bins, range=(0, max_area))
        xh = xh[1:].astype(float)
        yh = yh.astype(float)

        try:
            if self.fit_type_combo.currentText() == "Lognormal":
                popt = core_carbontools.fit_lognormal(xh, yh)
                curve = core_carbontools.lognormal(xh, *popt)
                _ampl, width, xc, offset = popt
                text = f"Lognormal fit: center={xc:.3g} nm², width={width:.3g}, offset={offset:.3g}"
            else:
                popt = core_carbontools.fit_bi_lognormal(xh, yh)
                curve = core_carbontools.bi_lognormal(xh, *popt)
                _ampl1, width1, xc1, _ampl2, width2, xc2, offset = popt
                text = (
                    f"Component 1: center={xc1:.3g} nm², width={width1:.3g}\n"
                    f"Component 2: center={xc2:.3g} nm², width={width2:.3g}\n"
                    f"Offset: {offset:.3g}"
                )
        except Exception as exc:
            QMessageBox.critical(self, "Fit failed", str(exc))
            return

        self.fit_result_label.setText(text)
        self._draw_histogram(fit_curve=curve, fit_label=self.fit_type_combo.currentText())

    def on_save_areas_clicked(self) -> None:
        if self._segment_areas is None:
            return
        directory = QFileDialog.getExistingDirectory(self, "Choose output folder")
        if not directory:
            return
        prefix, ok = QInputDialog.getText(self, "Filename prefix", "Prefix for saved file:", text="segment_areas")
        if not ok:
            return
        prefix = prefix.strip() or "segment_areas"
        out = Path(directory) / f"{prefix}.npy"
        np.save(out, self._segment_areas)
        QMessageBox.information(self, "Saved", f"Saved segment areas to {out}")
